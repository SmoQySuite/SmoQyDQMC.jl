@doc raw"""
    FourierMassMatrix{E<:AbstractFloat, PFFT, PIFFT}

Defines the mass matrix that implements fourier acceleration when performing either
hybrid/hamiltonian monte carlo or langevin monte carlo updates to the phonon fields.

# Fields

- `M̃::Matrix{E}`: Matrix elements of fourier mass matrix in frequency space.
- `v′::Matrix{Complex{E}}`: Temporary storage array to contain velocities as complex numbers to avoid dynamic memory allocations.
- `ṽ::Matrix{Complex{E}}`: Temporary storage to avoid some dynamic memory allocations when performing fourier transforms.
- `pfft::PFFT`: Forward FFT plan to perform transformation from imaginary time to frequency space without allocations.
- `pifft::PIFFT`: Inverse FFT plan to perform transformation from frequency to imaginary time space without allocations.
- `is_scalar::Bool`: If the mass matrix is equivalent to a simple scalar.
"""
struct FourierMassMatrix{E<:AbstractFloat, PFFT, PIFFT}

    # matrix elements of mass matrix in frequency space where the matrix is diagonal
    M̃::Matrix{E}

    # array to contain velocities as complex numbers
    v′::Matrix{Complex{E}}

    # frequency space matrix
    ṽ::Matrix{Complex{E}}

    # forward FFT plan
    pfft::PFFT

    # reverse FFT plan
    pifft::PIFFT

    # if mass matrix is simply a scalar
    is_scalar::Bool
end

@doc raw"""
    FourierMassMatrix(electron_phonon_parameters::ElectronPhononParameters{T,E}, reg::E=1.0) where {T,E}

Initialize and return an instance of `FouerierMassMatrix`.
Given a regularization value of `reg`, represented by the symbol ``m_{\rm reg}``, the matrix elements of
the fouerier mass matrix in frequency space, where it is diagonal, are given by
```math
\tilde{M}_{\omega,\omega} = 
    M \Delta\tau \frac{(1+m_{{\rm reg}}^2)M\Omega^{2}+\frac{4M}{\Delta\tau^{2}}\sin^{2}\big(\frac{\pi\omega}{L_{\tau}}\big)}{(1+m_{{\rm reg}}^2)M\Omega^{2}},
```
where ``\omega \in [0, L_\tau)`` corresponds to the frequency after fourier transforming from imaginary time to frequency space,
and ``L_\tau`` is the length of the imaginary time axis.
Also, ``\Omega`` and ``M`` are the phonon frequency and mass respectively.
"""
function FourierMassMatrix(electron_phonon_parameters::ElectronPhononParameters{T,E}, reg::E=1.0) where {T,E}

    # get phonon fields
    x = electron_phonon_parameters.x::Matrix{E}

    # get length of imaginary time axis
    Lτ = size(x, 2)

    # get discretization in imaginary time
    Δτ = electron_phonon_parameters.Δτ::E

    # get phonon parameters
    phonon_parameters = electron_phonon_parameters.phonon_parameters::PhononParameters{E}
    M = phonon_parameters.M::Vector{E} # phonon mass
    Ω = phonon_parameters.Ω::Vector{E} # phonon frequency

    # allocate velocity arrays
    v′ = zeros(Complex{E}, size(x))
    ṽ  = zeros(Complex{E}, size(x))

    # plan forward and reverse fft
    pfft = plan_fft(v′, (2,), flags=FFTW.PATIENT)
    pifft = plan_ifft(v′, (2,), flags=FFTW.PATIENT)

    # allocate fourier mass matrix elements
    M̃ = zeros(E, size(x))

    # iterate over frequency
    for ω in axes(M̃, 2)
        # iterate over phonon fields
        for n in axes(M̃, 1)
            # if inifinite phonon mass
            if isinf(M[n])
                # set corresponding mass matrix to infinity
                M̃[n, ω] = Inf
            # if regularization is infinite
            elseif isinf(reg)
                # set corresponding fourier mass matrix to identity
                M̃[n, ω] = M[n] * Δτ
            # if finite phonon mass and regularization
            else
                # CHANGE IN FUTURE: if bare on-site phonon frequency is zero, default to 1.0
                Ω′ = iszero(Ω[n]) ? reg : sqrt((1+reg^2) * Ω[n]^2)
                # set the fourier mass matrix element
                M̃[n, ω] = M[n] * Δτ * ((1+reg^2)*Ω′^2 + 4/Δτ^2*sin(π*(ω-1)/Lτ)^2) / ((1+reg^2)*Ω′^2)
            end
        end
    end

    # determine if mass matrix is equal to a simple scalar
    is_scalar = all(m -> (isinf(m)||(m≈Δτ)), M̃)

    return FourierMassMatrix(M̃, v′, ṽ, pfft, pifft, is_scalar)
end

# Given the [`FourierMassMatrix`](@ref) `M` and velocity `v`, return the total kinetic energy `K = (v⋅M⋅v)/2`.
function velocity_to_kinetic_energy(M::FourierMassMatrix{E}, v::Matrix{E}) where {E<:AbstractFloat}

    (; M̃, pfft, pifft, is_scalar) = M
    Mv = M.v′
    Mṽ = M.ṽ

    # copy velocity
    copyto!(Mv, v)

    # calculate M⋅v
    if is_scalar
        _apply_fourier_mass_matrix!(M̃, Mv, 1.0)
    else
        mul!(Mṽ, pfft, Mv)
        _apply_fourier_mass_matrix!(M̃, Mṽ, 1.0)
        mul!(Mv, pifft, Mṽ)
    end

    # calculate the kinetic energy K = (v⋅M⋅v)/2
    v′  = vec(v)
    Mv′ = vec(Mv)
    K = dot(v′, Mv′)/2

    return real(K)
end

# momentum_to_kinetic_energy(M::FourierMassMatrix{E}, p::Matrix{E}) where {E<:AbstractFloat}
function momentum_to_kinetic_energy(M::FourierMassMatrix{E}, p::Matrix{E}) where {E<:AbstractFloat}

    (; M̃, pfft, pifft, is_scalar) = M
    M⁻¹p = M.v′
    M⁻¹p̃ = M.ṽ

    # copy velocity
    copyto!(M⁻¹p, p)

    # calculate M⁻¹⋅p

    if is_scalar
        _apply_fourier_mass_matrix!(M̃, M⁻¹p, -1.0)
    else
        mul!(M⁻¹p̃, pfft, M⁻¹p)
        _apply_fourier_mass_matrix!(M̃, M⁻¹p̃, -1.0)
        mul!(M⁻¹p, pifft, M⁻¹p̃)
    end

    # calculate the kinetic energy K = (p⋅M⁻¹⋅p)/2
    p′    = vec(p)
    M⁻¹p′ = vec(M⁻¹p)
    K = dot(p′, M⁻¹p′)/2

    return real(K)
end

# Left multiply `v` by the [`FourierMassMatrix`](@ref) `M` raised to the power `α`, modifying `v` in-place.
# The rows of `v` correspond to phonon modes, and the columns correspond to imaginary time slices.
function lmul!(M::FourierMassMatrix{E}, v::Matrix{E}, α::E=1.0) where {E<:AbstractFloat}

    (; v′) = M

    copyto!(v′, v)
    lmul!(M, v′, α)
    @. v = real(v′)

    return nothing
end

function lmul!(M::FourierMassMatrix{E}, v::Matrix{Complex{E}}, α::E=1.0) where {E<:AbstractFloat}

    (; M̃, ṽ, pfft, pifft, is_scalar) = M

    # apply fourier mass matrix
    if is_scalar
        _apply_fourier_mass_matrix!(M̃, v, α)
    else
        mul!(ṽ, pfft, v)
        _apply_fourier_mass_matrix!(M̃, ṽ, α)
        mul!(v, pifft, ṽ)
    end

    return nothing
end


# Multiply `v` by the [`FourierMassMatrix`](@ref) `M` raised to the power `α`, writing the result to `Mv`.
# The rows of `Mv` and `v` correspond to phonon modes, and the columns correspond to imaginary time slices.
function mul!(Mv::Matrix, M::FourierMassMatrix{E}, v::Matrix, α::E=1.0) where {E<:AbstractFloat}

    copyto!(Mv, v)
    lmul!(M, Mv, α)

    return nothing
end


# apply fourier mass matrix in frequency space
function _apply_fourier_mass_matrix!(M̃::Matrix{E}, ṽ::Matrix{Complex{E}}, α::E) where {E<:AbstractFloat}

    # iterate over frequencies
    @inbounds for ω in axes(ṽ,2)
        # iterate over phonon modes
        for n in axes(ṽ,1)
            # if mass matrix element is finite
            if isfinite(M̃[n,ω])
                # apply fourier mass matrix
                ṽ[n,ω] *= M̃[n,ω]^α
            # if mass matrix is infite set velocityt zero
            else
                ṽ[n,ω] = 0.0
            end
        end
    end

    return nothing
end