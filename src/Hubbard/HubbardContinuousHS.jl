struct HubbardContinuousHSParameters{T<:AbstractFloat} <: AbstractHubbardHS{T}

    # interpolating parameter in HS transformation
    p::T

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
    
    # constant of HS transformation
    c::Vector{T}

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

function HubbardContinuousHSParameters(; β::T, Δτ::T, p::T,
                                       hubbard_parameters::HubbardParameters{T},
                                       rng::AbstractRNG) where {T<:AbstractFloat}

    (; U, sites) = hubbard_parameters

    # calcualte length of imaginary time axis
    Lτ = eval_length_imaginary_axis(β, Δτ)

    # get the number of HS transformations per imaginary time-slice
    N = length(U)

    # calculate relevant constant c for HS transformation
    c = similar(U)
    for i in eachindex(U)
        c[i] = solve_eqn35(p, U[i], Δτ)
    end

    # initialize random hs field configuration
    s = zeros(T, N, Lτ)
    @fastmath @inbounds for i in eachindex(s)
        # sample HS field uniformly in the interval (-π,π)
        s[i] = 2 * π * (rand(rng) - 0.5)
    end

    # array to store initial HS field in HMC update
    s0::Matrix{T}

    # conjugate velocities for hmc update
    v = zeros(T, N, Lτ)

    # allocate array for derivative of action with respect to HS fields for HMC updates
    dSds = zeros(T, N, Lτ)

    # allocate array for derivative of action with respect to HS fields for HMC updates
    dSds0 = zeros(T, N, Lτ)

    return HubbardContinuousHS(p, β, Δτ, Lτ, N, U, c, sites, s, s0, v, dSds, dSds0)
end


function initialize!(fermion_path_integral_up::FermionPathIntegral{T,E},
                     fermion_path_integral_dn::FermionPathIntegral{T,E},
                     hubbard_continuous_parameters::HubbardContinuousHSParameters{E}) where {T,E}

    (; p, c, U, Δτ, s, sites) = hubbard_continuous_parameters
    Vup = fermion_path_integral_up.V
    Vdn = fermion_path_integral_dn.V

    # add continuous HS field contribution to diagonal on-site energy matrices
    for l in axes(Vup,2)
        for i in eachindex(sites)
            site = sites[i]
            Vup[site,l] = Vup[site,l] - sign(U[i])/Δτ * eval_a(i, l, hubbard_continuous_parameters)
            Vdn[site,l] = Vdn[site,l] + sign(U[i])/Δτ * eval_a(i, l, hubbard_continuous_parameters)
        end
    end

    return nothing
end

function initialize!(fermion_path_integral::FermionPathIntegral{T,E},
                     hubbard_continuous_parameters::HubbardContinuousHSParameters{E}) where {T,E}

    (; p, c, U, Δτ, s, sites) = hubbard_continuous_parameters
    V = fermion_path_integral.V

    # make sure its a strictly attractive hubbard interaction
    @assert all(u -> u < 0.0, U)

    # add continuous HS field contribution to diagonal on-site energy matrices
    for l in axes(Vup,2)
        for i in eachindex(sites)
            site = sites[i]
            V[site,l] = V[site,l] - sign(U[i])/Δτ * eval_a(i, l, hubbard_continuous_parameters)
        end
    end

    return nothing
end


# solve eqn (35) from the paper "A flexible class of exact Hubbard-Stratonovich transformations"
# note that if p=0 eqn (24) is solved instead
function solve_eqn35(p, U, Δτ)

    x = Δτ*abs(U)/2
    f(c) = quadgk(s -> cosh(sqrt(c) * (iszero(p) ? sin(s) : (atan(p * sin(s)) / atan(p))) ), -π, π; rtol=1e-12)[1]/(2π) - exp(x)
    return find_zero(f, x)
end

# evaluate the function a(s) defined in equation (30) from
# the paper "A flexible class of exact Hubbard-Stratonovich transformations"
function eval_a(i::Int, l::Int, hubbard_parameters::HubbardContinuousHSParameters{E})

    p = hubbard_parameters.p::E
    c = hubbard_parameters.c::Vector{E}
    s = hubbard_parameters.s::Matrix{E}

    return iszero(p) ? sqrt(c[i]) * sin(s[i,l]) : sqrt(c[i]) * atan(p * sin(s[i,l]))/atan(p)
end

# evaluate the derivative function da(s)/ds where a(s) us defined in equation (30) from
# the paper "A flexible class of exact Hubbard-Stratonovich transformations"
function eval_dads(i::Int, l::Int, hubbard_parameters::HubbardContinuousHSParameters{E})

    p = hubbard_parameters.p::E
    c = hubbard_parameters.c::Vector{E}
    s = hubbard_parameters.s::Matrix{E}

    return iszero(p) ? sqrt(c[i])*cos(s[i,l]) : sqrt(c[i])*p*cos(s[i,l])/(atan(p)*(1 + p^2*sin(s[i,l])^2))
end

# bosonic action derivative accounting for the -1 factor appearing in
# exp{-Δτ⋅U⋅(nup-1/2)⋅(ndn-1/2)} = ∫ds exp{a(s)(nup+ndn-1) + ln(b(s))}
#                                = ∫ds exp{a(s)(nup+ndn) + ln(b(s)) - a(s)}, 
# where b(s) is a constant whose derivative goes to zero.
function _bosonic_action_derivative(dSds::Matrix{E}, hubbard_hs_parameters::HubbardContinuousHSParameters{E}) where {E}

    (; U, s) = hubbard_hs_parameters

    # iterate over imaginary time slice
    @fastmath @inbounds for l in axes(s, 2)
        # iterate of finite hubbard U terms
        for i in eachindex(U)
            # if negative U attractive hubbard interaction
            if U[i] < 0
                # ∂S/∂s(i,l) += ∂a/∂s(i,l)
                dSds[i,l] += eval_dads(i, l, hubbard_hs_parameters)
            end
        end
    end

    return nothing
end