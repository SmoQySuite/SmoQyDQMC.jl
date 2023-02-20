struct HubbardContinuousHS{T<:AbstractFloat} <: AbstractHubbardHS

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
end

function HubbardContinuousHS(; β::T, Δτ::T, p::T,
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



    return HubbardContinuousHS(p, β, Δτ, Lτ, N, U, c, sites, s)
end

# solve eqn (35) from the paper "A flexible class of exact Hubbard-Stratonovich transformations"
function solve_eqn35(p, U, Δτ)

    x = Δτ*abs(U)/2
    f(c) = quadgk(s -> cosh(sqrt(c) * atan(p * sin(s)) / atan(p)), -π, π; rtol=1e-12)[1]/(2π) - exp(x)
    return find_zero(f, x)
end