struct ExactFourierAccelerator{T<:AbstractFloat, PFFT, PIFFT}

    η::T
    ω::Matrix{T}
    m::Matrix{T}
    x̃::Matrix{Complex{T}}
    p̃::Matrix{Complex{T}}
    u::Matrix{Complex{T}}
    pfft::PFFT
    pifft::PIFFT
end

function ExactFourierAccelerator(
    Ω::Vector{T},
    M::Vector{T},
    β::T,
    Δτ::T,
    η::T
) where {T<:AbstractFloat}

    # Number of phonon modes
    Nph = length(Ω)

    # Length of imaginary-time axis
    Lτ = round(Int, β/Δτ)

    # allocate matrices
    ω = zeros(T, Nph, Lτ)
    m = zeros(T, Nph, Lτ)
    x̃ = zeros(Complex{T}, Nph, Lτ)
    p̃ = zeros(Complex{T}, Nph, Lτ)
    u = zeros(Complex{T}, Nph, Lτ)

    # iterate over fourier modes
    for n in axes(ω, 2)
        # iterate over phonon modes
        for i in axes(ω, 1)
            # if phonon mass is infinite
            if isinf(M[i])
                # set dynamical mass to infinity
                m[i,n] = Inf
                # set dynamical frequency to zero
                ω[i,n] = 0.0
            # if finite phonon mass
            else
                # calculate the spring constant
                k = Δτ*M[i]*Ω[i]^2 + 4*M[i]*sin(π*(n-1)/Lτ)^2/Δτ
                # calculate fourier mode mass
                Ω′ = iszero(Ω[i]) ? η : sqrt((1+η^2) * Ω[i]^2)
                m[i,n] = isinf(η) ? Δτ*M[i] : Δτ*M[i] * (Ω′^2 + 4/Δτ^2*sin(π*(n-1)/Lτ)^2) / Ω′^2
                # calculate fourier mode frequency
                ω[i,n] = sqrt(k/m[i,n])
            end
        end
    end

    # initialize fft and inverse fft plans
    pfft = plan_fft(x̃, (2,), flags=FFTW.PATIENT)
    pifft = plan_ifft(x̃, (2,), flags=FFTW.PATIENT)

    return ExactFourierAccelerator(η, ω, m, x̃, p̃, u, pfft, pifft)
end

# function to exactly evolve the equation of motion of the harmonic bosonic action
function evolve_eom!(
    x::AbstractMatrix{T},
    p::AbstractMatrix{T},
    Δt::T,
    efa::ExactFourierAccelerator{T, PFFT, PIFFT}
) where {T, PFFT, PIFFT}

    (; m, ω, x̃, p̃, u, pfft, pifft) = efa

    # length of imaginary time axis
    Lτ = size(x,2)

    # fourier transform phonon fields from imaginary time to fourier space
    @. u = x / sqrt(Lτ)
    mul!(x̃, pfft, u)

    # fourier transform momentum from imaginary time to fourier space
    @. u = p / sqrt(Lτ)
    mul!(p̃, pfft, u)

    # iterate over fourier modes
    @simd for n in axes(ω,2)
        # iterate over phonon modes
        for i in axes(ω,1)
            # get relevant frequency
            ωₙ = ω[i,n]
            # get the relevant mass
            mᵢ = m[i,n]
            # make sure mass if finite
            if isfinite(mᵢ)
                # get initial position and momentum
                x̃′ = x̃[i,n]
                p̃′ = p̃[i,n]
                # if finite frequency
                if ωₙ > 1e-10
                    # update position analytically
                    x̃[i,n] = x̃′*cos(ωₙ*Δt) + p̃′/(ωₙ*mᵢ) * sin(ωₙ*Δt)
                # if frequency is very near zero
                elseif abs(ωₙ) ≤ 1e-10
                    # update position numerically using taylor expansion of analytic expression
                    x̃[i,n] = x̃′ + (Δt - Δt^3*ωₙ^2/6 + Δt^5*ωₙ^4/120) * p̃[i,n]/mᵢ
                end
                # update momentum
                p̃[i,n] = p̃′*cos(ωₙ*Δt) - x̃′*(ωₙ*mᵢ) * sin(ωₙ*Δt)
            end
        end
    end

    # fourier transform phonon fields from fourier space to imaginary time
    mul!(u, pifft, x̃)
    @. x = real(u) * sqrt(Lτ)

    # fourier transform momentum from fourier space to imaginary time
    mul!(u, pifft, p̃)
    @. p = real(u) * sqrt(Lτ)

    return nothing
end

# calculate and return kinetic energy
function kinetic_energy(
    p::Matrix{T},
    efa::ExactFourierAccelerator{T, PFFT, PIFFT},
)::T where {T, PFFT, PIFFT}

    (; p̃, m, η, u, pfft) = efa
    Lτ = size(p, 2)

    # if finite regularization, then transform to frequency space to calculate kinetic energy
    if isfinite(η)
        @. u = p / sqrt(Lτ)
        mul!(p̃, pfft, u)
        p′ = p̃
    else
        p′ = p
    end

    # calculate kinetic energy, setting terms with infinite mass to zero
    K = zero(T)
    for i in eachindex(p′)
        K += isinf(m[i]) ? 0.0 : abs2(p′[i])/(2*m[i])
    end
    
    return K
end

# initialize momentum and calculate and return corresponding kinetic energy
function initialize_momentum!(
    p::Matrix{T},
    efa::ExactFourierAccelerator{T, PFFT, PIFFT},
    rng::AbstractRNG
)::T where {T, PFFT, PIFFT}

    (; p̃, m, η, u, pfft, pifft) = efa
    Lτ = size(p, 2)

    # sample vector of random normal numbers
    randn!(rng, p)

    # if finite regularization, then transform to frequency space to calculate kinetic energy
    if isfinite(η)
        @. u = p / sqrt(Lτ)
        mul!(p̃, pfft, u)
        p′ = p̃
    else
        p′ = p
    end

    # initialize momentum such that p = √(M)⋅R
    for i in eachindex(p′)
        p′[i] = isinf(m[i]) ? 0.0 : sqrt(m[i]) * p′[i]
    end

    # calculate kinetic energy, setting terms with infinite mass to zero
    K = zero(T)
    for i in eachindex(p′)
        K += isinf(m[i]) ? 0.0 : abs2(p′[i])/(2*m[i])
    end

    # if finite regularization, then transform back to imaginary time
    if isfinite(η)
        mul!(u, pifft, p′)
        @. p = real(u) * sqrt(Lτ)
    end

    return K
end