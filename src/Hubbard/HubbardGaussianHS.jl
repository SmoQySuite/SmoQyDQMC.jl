struct HubbardGaussianHSParameters{T<:AbstractFloat} <: AbstractHubbardHS{T}

    # inverse temperature
    β::T

    # discretization in imaginary time
    Δτ::T

    # length of imaginary time axis
    Lτ::Int

    # number of orbitals with finite Hubbard U
    N::Int

    # each finite hubbard interaction
    U::Vector{T}

    # site index associated with each Hubbard U
    sites::Vector{Int}

    # hubbard-stratonvich fields
    s::Matrix{T}

    # array to store initial HS field in HMC update
    s0::Matrix{T}

    # conjugate velocity for performing hmc update
    v::Matrix{T}

    # array for evaluating derivative of action for hmc updates
    dSds::Matrix{T}

    # array for storing original derivative of action
    dSds0::Matrix{T}
end

function HubbardGaussianHSParameters(; β::T, Δτ::T, hubbard_parameters::HubbardParameters{T},
                                     rng::AbstractRNG) where {T<:AbstractFloat}

    (; U, sites) = hubbard_parameters

    # calcualte length of imaginary time axis
    Lτ = eval_length_imaginary_axis(β, Δτ)

    # get the number of HS transformations per imaginary time-slice
    N = length(U)

    # initial random gaussian HS fields
    s = randn(rng, N, Lτ)

    # array to store initial HS field in HMC update
    s0 = copy(s)

    # conjugate velocities for hmc update
    v = zeros(T, N, Lτ)

    # allocate array for derivative of action with respect to HS fields for HMC updates
    dSds = zeros(T, N, Lτ)

    # allocate array for derivative of action with respect to HS fields for HMC updates
    dSds0 = zeros(T, N, Lτ)
    
    return HubbardGaussianHSParameters(β, Δτ, Lτ, N, U, sites, s, s0, v, dSds, dSds0)
end

# evaluate the function a(s) = sqrt(Δτ|U|)s
function eval_a(i::Int, l::Int, hubbard_hs_parameters::HubbardGaussianHSParameters{T}) where {T}

    (; U, Δτ, s) = hubbard_hs_parameters
    a = sqrt(Δτ*abs(U[i])) * s[i,l]

    return a
end

# evaluate derivative da/ds = sqrt(Δτ|U|)
function eval_dads(i::Int, l::Int, hubbard_hs_parameters::HubbardGaussianHSParameters{T}) where {T}

    (; U, Δτ, s) = hubbard_hs_parameters
    dads = sqrt(Δτ*abs(U[i]))

    return dads
end

# calculate the bosonic action Sb = s^2/2
function _bosonic_action(hubbard_hs_parameters::HubbardGaussianHSParameters{T}) where {T}

    (; Δτ, U, s) = hubbard_hs_parameters

    Sb = zero(T)
    @inbounds for l in axes(s,2)
        for i in axes(s,1)
            Sb += s[i,l]^2/2
        end
    end

    return Sb
end

# calculate the derivative of the bosonic action dSb/ds = s
function _bosonic_action_derivative!(dSds::Matrix{E}, hubbard_hs_parameters::HubbardGaussianHSParameters{E}) where {E}

    (; U, s, Δτ) = hubbard_hs_parameters

    @inbounds for l in axes(s, 2)
        for i in axes(s, 1)
            dSds[i,l] += s[i,l]
        end
    end

    return nothing
end