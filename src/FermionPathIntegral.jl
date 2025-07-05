@doc raw"""
    FermionPathIntegral{H<:Number, T<:Number, U<:Number, R<:AbstractFloat}

A type (mutable struct) to represent a fermion path integral. In particular, this type contains the information required to reconstruct
the diagonal on-site energy matrices ``V_l`` and hopping matrices ``K_l`` for each imaginary time
slice ``l \in [1, L_\tau],`` where ``\tau = \Delta\tau \cdot l`` and ``\beta = \Delta\tau \cdot L_\tau.``

# Types

- `H<:Number`: ``H_l = (K_l + V_l)`` Hamiltonian matrix element type.
- `T<:Number`: ``K_l`` kintetic energy matrix element type.
- `U<:Number`: ``V_l`` potential energy matrix element type.
- `R<:AbstractFloat`: Real number type.

# Fields

- `β::R`: Inverse temperature.
- `Δτ::R`: Discretization in imaginary time.
- `Lτ::Int`: Length of the imaginary time axis.
- `N::Int`: Number of orbitals in the lattice.
- `neighbor_table::Matrix{Int}`: Neighbor table for each pair of orbitals in the lattice connected by a hopping.
- `t::Matrix{T}`: Hopping amplitudes for imaginary-time slice ``l`` are stored in `t[:,l]`.
- `V::Matrix{U}`: The diagonal on-site energy matrices ``V_l`` for imaginary-time slice ``l`` are stored in `V[:,l]`.
- `K::Matrix{T}`: Used to construct hopping matrix to cacluate exponentiated hopping matrix if checkerboard approximation is not being used.
- `Sb::H`: Keeps track of total bosonic action associated with fermionic path integral.
- `eigen_ws::HermitianEigenWs{T,Matrix{T},R}`: For calculating eigenvalues and eigenvectors of `K` while avoiding dynamic memory allocations.
- `u::Matrix{H}`: Temporary matrix to avoid dynamic allocation when performing local updates.
- `v::Matrix{H}`: Temporary matrix to avoid dynamic allocation when performing local updates.
"""
mutable struct FermionPathIntegral{H<:Number, T<:Number, U<:Number, R<:AbstractFloat}

    β::R

    Δτ::R

    Lτ::Int

    N::Int

    neighbor_table::Matrix{Int}

    t::Matrix{T}

    V::Matrix{U}

    K::Matrix{T}

    Sb::H

    eigen_ws::HermitianEigenWs{T,Matrix{T},R}

    u::Matrix{H}

    v::Matrix{H}
end


@doc raw"""
    FermionPathIntegral(;
        # KEYWORD ARGUMENTS
        tight_binding_parameters::TightBindingParameters{T,R},
        β::R, Δτ::R,
        forced_complex_kinetic::Bool = false,
        forced_complex_potential::Bool = false
    ) where {T<:Number, R<:AbstractFloat}

Initialize an instance of [`FermionPathIntegral`](@ref) an instance of [`TightBindingParameters`](@ref).

If `forced_complex_kinetic = true`, then the off-diagonal kinetic energy matrices ``K_l`` are assumed to be complex,
otherwise the matrix element type is inferred.

If `forced_complex_potential = true`, then the diagonal potential energy matrices ``V_l`` are assumed to be complex,
otherwise the matrix element type is inferred.
"""
function FermionPathIntegral(;
    # KEYWORD ARGUMENTS
    tight_binding_parameters::TightBindingParameters{T,R},
    β::R, Δτ::R,
    forced_complex_kinetic::Bool = false,
    forced_complex_potential::Bool = false
) where {T<:Number, R<:AbstractFloat}

    # rename for convenience
    tbp = tight_binding_parameters
    
    (; ϵ, neighbor_table) = tbp

    # type of kinetic energy matrix
    Tk = forced_complex_kinetic ? Complex{R} : T

    # type of potential energy matrix
    Tv = forced_complex_potential ? Complex{R} : R

    # type of eventual green's function matrix elements
    H = (Tk <: Complex || Tv <: Complex) ? Complex{R} : R

    # initialize bosonic phase factor to unit
    Sb = zero(H)

    # evaluate length of imaginary time axis
    Lτ = eval_length_imaginary_axis(β, Δτ)

    # get number of orbitals in lattice
    Nsites = length(ϵ)

    # number of pairs of neighboring orbitals connected by a bond in the lattice
    Nneighbors = size(neighbor_table, 2)

    # initialize hoppings to zero
    t = zeros(Tk, Nneighbors, Lτ)

    # set the value of the hopping for each imaginary time slice based on the
    # non-interacting tight binding model parameters
    for l in axes(t, 2)
        @views @. t[:,l] = tbp.t
    end

    # initialize diagonal on-site energy matrix
    V = zeros(Tv, Nsites, Lτ)

    # set the value on-site energy for each imaginary time slice based on the
    # non-interacting tight-binding model parameters
    for l in axes(V, 2)
        @views @. V[:,l] = tbp.ϵ - tbp.μ
    end

    # initialize hopping matrix
    K = zeros(Tk, Nsites, Nsites)

    # initialize workspace for calculating eigenvalue and eigenvector of K while avoid allocations
    eigen_ws = HermitianEigenWs(K, vecs=true)

    # allocate temporary vectors
    u = zeros(H, Nsites, 1)
    v = zeros(H, Nsites, 1)

    return FermionPathIntegral{H,Tk,Tv,R}(β, Δτ, Lτ, Nsites, neighbor_table, t, V, K, Sb, eigen_ws, u, v)
end


@doc raw"""
    initialize_propagators(
        # ARGUMENTS
        fermion_path_integral::FermionPathIntegral;
        # KEYWORD ARGUMENTS
        symmetric::Bool,
        checkerboard::Bool
    )

Initialize a propagator for each imaginary time slice, returning a vector of type `Vector{<:AbstractPropagators{T,E}}`.
"""
function initialize_propagators(
    # ARGUMENTS
    fermion_path_integral::FermionPathIntegral;
    # KEYWORD ARGUMENTS
    symmetric::Bool,
    checkerboard::Bool
)

    # allocate propagators
    B = allocate_propagators(fermion_path_integral, symmetric=symmetric, checkerboard=checkerboard)

    # calculate propagator matrices
    calculate_propagators!(B, fermion_path_integral, calculate_exp_V=true, calculate_exp_K=true)

    return B
end


# allocate propagator matrices
function allocate_propagators(fpi::FermionPathIntegral; symmetric::Bool, checkerboard::Bool)

    # initialize symmetric propagators using the checkerboard approximation
    if symmetric && checkerboard

        B = _allocate_symmetric_checkerboard_propagators(fpi)

    # initialize symmetric propagators using the exactly exponentiated hopping matrix
    elseif symmetric && !checkerboard

        B = _allocate_symmetric_exact_propagators(fpi)

    # initialize asymmetric propagators using the checkerboard approximation
    elseif !symmetric && checkerboard

        B = _allocate_asymmetric_checkerboard_propagators(fpi)

    # initialize asymmetric propagators using the exactly exponentiated hopping matrix
    else

        B = _allocate_asymmetric_exact_propagators(fpi)
    end

    return B
end

# allocate symmetric propagators with exactly exponentiated hopping matrix
function _allocate_symmetric_exact_propagators(fpi::FermionPathIntegral{H,T,U,R}) where {H,T,U,R}

    (; Lτ, N) = fpi

    # initialize vector of propagators
    B = SymExactPropagator{T,U}[]

    # iterate over imaginary time slice
    for l in 1:Lτ

        # allocate propagator for the current imaginary time slice
        expmΔτKo2_l = Matrix{T}(I, N, N)
        exppΔτKo2_l = Matrix{T}(I, N, N)
        expmΔτV_l = ones(U, N)
        B_l = SymExactPropagator{T,U}(expmΔτV_l, expmΔτKo2_l, exppΔτKo2_l)
        push!(B, B_l)
    end

    return B
end

# allocate asymmetric propagators with exactly exponentiated hopping matrix
function _allocate_asymmetric_exact_propagators(fpi::FermionPathIntegral{H,T,U,R}) where {H,T,U,R}

    (; Lτ, N) = fpi

    # initialize vector of propagators
    B = AsymExactPropagator{T,U}[]

    # iterate over imaginary time slice
    for l in 1:Lτ

        # allocate propagator for the current imaginary time slice
        expmΔτK_l = Matrix{T}(I, N, N)
        exppΔτK_l = Matrix{T}(I, N, N)
        expmΔτV_l = ones(U, N)
        B_l = AsymExactPropagator{T,U}(expmΔτV_l, expmΔτK_l, exppΔτK_l)
        push!(B, B_l)
    end

    return B
end

# allocate symmetric checkerboard propagators
function _allocate_symmetric_checkerboard_propagators(fpi::FermionPathIntegral{H,T,U,R}) where {H,T,U,R}

    (; Δτ, Lτ, N, neighbor_table) = fpi

    # initialize vector of propagators
    B = SymChkbrdPropagator{T,U}[]

    # get the number of neighbors
    Nneighbors = size(neighbor_table, 2)

    # initialize vector of zero hoppings
    t = zeros(T, Nneighbors)

    # initialize identity checkberoard matrix
    expnΔτKo2 = CheckerboardMatrix(neighbor_table, t, Δτ/2)

    # iterate of imaginary time slices
    for l in 1:Lτ

        # allocate propagator for current imaginary time slice
        expmΔτKo2_l = CheckerboardMatrix(expnΔτKo2, new_matrix=true)
        expmΔτV_l = ones(U, N)
        B_l = SymChkbrdPropagator{T,U}(expmΔτV_l, expmΔτKo2_l)
        push!(B, B_l)
    end

    return B
end

# allocate asymmetric checkerboard propagators
function _allocate_asymmetric_checkerboard_propagators(fpi::FermionPathIntegral{H,T,U,R}) where {H,T,U,R}

    (; Δτ, Lτ, N, neighbor_table) = fpi

    # initialize vector of propagators
    B = AsymChkbrdPropagator{T,U}[]

    # get the number of neighbors
    Nneighbors = size(neighbor_table, 2)

    # initialize vector of zero hoppings
    t = zeros(T, Nneighbors)

    # initialize identity checkberoard matrix
    expnΔτK = CheckerboardMatrix(neighbor_table, t, Δτ)

    # iterate of imaginary time slices
    for l in 1:Lτ

        # allocate propagator for current imaginary time slice
        expmΔτK_l = CheckerboardMatrix(expnΔτK, new_matrix=true)
        expmΔτV_l = ones(U, N)
        B_l = AsymChkbrdPropagator{T,U}(expmΔτV_l, expmΔτK_l)
        push!(B, B_l)
    end

    return B
end


@doc raw"""
    calculate_propagators!(
        # ARGUMENTS
        B::Vector{P},
        fpi::FermionPathIntegral;
        # KEYWORD ARGUMENTS
        calculate_exp_V::Bool,
        calculate_exp_K::Bool
    ) where {P<:AbstractPropagator}

Calculate the propagator matrices ``B_l``, given by `B[l]`, for all imaginary time slice ``\tau = \Delta\tau \cdot l.``
If `calculate_exp_V = true`, then calculate the diagonal exponentiated on-site energy matrices.
If `calculate_exp_K = true`, then calculate the exponentiated hopping matrices.
"""
function calculate_propagators!(
    # ARGUMENTS
    B::Vector{P},
    fpi::FermionPathIntegral;
    # KEYWORD ARGUMENTS
    calculate_exp_V::Bool,
    calculate_exp_K::Bool
) where {P<:AbstractPropagator}

    # iterate over imaginary time slices
    for l in eachindex(B)

        # calculate propagator for current imaginary time slice τ=Δτ⋅l
        calculate_propagator!(B[l], fpi, l, calculate_exp_V=calculate_exp_V, calculate_exp_K=calculate_exp_K)
    end

    return nothing
end

@doc raw"""
    calculate_propagator!(
        # ARGUMENTS
        B::P,
        fpi::FermionPathIntegral{H,T,U},
        l::Int;
        # KEYWORD ARGUMENTS
        calculate_exp_V::Bool,
        calculate_exp_K::Bool
    ) where {H<:Number, T<:Number, U<:Number, P<:AbstractPropagator{T,U}}

Calculate the propagator matrix ``B_l`` for imaginary time slice ``\tau = \Delta\tau \cdot l.``
If `calculate_exp_V = true`, then calculate the diagonal exponentiated on-site energy matrix.
If `calculate_exp_K = true`, then calculate the exponentiated hopping matrix.
"""
function calculate_propagator!(
    # ARGUMENTS
    B::P,
    fpi::FermionPathIntegral{H,T,U},
    l::Int;
    # KEYWORD ARGUMENTS
    calculate_exp_V::Bool,
    calculate_exp_K::Bool
) where {H<:Number, T<:Number, U<:Number, P<:AbstractPropagator{T,U}}

    # calculate exponentiated on-site energy matrix exp(-Δτ⋅V[l])
    if calculate_exp_V
        calculate_exp_V!(B, fpi, l)
    end

    # calculate exponentiated hopping matrix exp(-Δτ⋅K[l]) or exp(-Δτ/2⋅K[l])
    if calculate_exp_K
        calculate_exp_K!(B, fpi, l)
    end

    return nothing
end


# calculate exponentiated potential energy matrices for vector of propagators
function calculate_exp_V!(
    B::Vector{P},
    fpi::FermionPathIntegral{H,T,U}
) where {H, T, U, P<:AbstractPropagator{T,U}}

    # iterate over imaginary time slices
    @inbounds for l in eachindex(B)

        # calculate exponentiated diagonal on-site energy matrix for imaginary time slice τ=Δτ⋅l
        calculate_exp_V!(B[l], fpi, l)
    end

    return nothing
end

# calculate exponentiated potential energy matrix for propagator for l'th imaginary time slice
function calculate_exp_V!(
    B::AbstractPropagator{T,U},
    fpi::FermionPathIntegral{H,T,U},
    l::Int
) where {H, T, U}

    @views @. B.expmΔτV = exp(-fpi.Δτ * fpi.V[:,l])

    return nothing
end


# calculate exponentiated hopping matrices for vector of propagators
function calculate_exp_K!(
    B::Vector{P},
    fpi::FermionPathIntegral{H,T,U}
) where {H, T, U, P<:AbstractPropagator{T,U}}

    # iterate over imaginary time slices
    for l in eachindex(B)

        # calculate exponentiated hopping matrix for imaginary time slice τ=Δτ⋅l
        calculate_exp_K!(B[l], fpi, l)
    end

    return nothing
end

# calculate exponentiated hopping matrix for propagator for l'th imaginary time slice
function calculate_exp_K!(
    B::AbstractPropagator{T,U},
    fpi::FermionPathIntegral{H,T,U},
    l::Int
) where {H, T, U}

    _calculate_exp_K!(B, fpi, l)

    return nothing
end

# calculate exponentiated hopping matrix for imaginary time slice l
function _calculate_exp_K!(
    B::SymExactPropagator{T,U},
    fpi::FermionPathIntegral{H,T,U},
    l::Int
) where {H, T, U}

    (; expmΔτKo2, exppΔτKo2) = B
    (; Δτ, K, neighbor_table, eigen_ws) = fpi

    # get hopping energeis for imaginary time slice τ=Δτ⋅l
    t = @view fpi.t[:,l]

    # build the hopping matrix
    build_hopping_matrix!(K, neighbor_table, t)

    # calculate exponentiated hopping matrices
    exp!(expmΔτKo2, exppΔτKo2, K, -Δτ/2, workspace = eigen_ws, tol = 1e-9)

    return nothing
end

# calculate exponentiated hopping matrix for imaginary time slice l
function _calculate_exp_K!(
    B::AsymExactPropagator{T,U},
    fpi::FermionPathIntegral{H,T,U},
    l::Int
) where {H, T, U}

    (; expmΔτK, exppΔτK) = B
    (; Δτ, K, neighbor_table, eigen_ws) = fpi

    # get hopping energeis for imaginary time slice τ=Δτ⋅l
    t = @view fpi.t[:,l]

    # build the hopping matrix
    build_hopping_matrix!(K, neighbor_table, t)

    # calculate exponentiated hopping matrices
    exp!(expmΔτK, exppΔτK, K, -Δτ, workspace = eigen_ws, tol = 1e-9)

    return nothing
end

# calculate checkerboard approximation for imaginary time slice l
function _calculate_exp_K!(
    B::SymChkbrdPropagator{T,U},
    fpi::FermionPathIntegral{H,T,U},
    l::Int
) where {H, T, U}

    (; expmΔτKo2) = B
    (; Δτ) = fpi

    # get hopping energeis for imaginary time slice τ=Δτ⋅l
    t = @view fpi.t[:,l]

    # update checkerboard factorization
    Checkerboard.update!(expmΔτKo2, t, Δτ/2)

    return nothing
end

# calculate checkerboard approximation for imaginary time slice l
function _calculate_exp_K!(
    B::AsymChkbrdPropagator{T,U},
    fpi::FermionPathIntegral{H,T,U},
    l::Int
) where {H, T, U}

    (; expmΔτK) = B
    (; Δτ) = fpi

    # get hopping energeis for imaginary time slice τ=Δτ⋅l
    t = @view fpi.t[:,l]

    # update checkerboard factorization
    Checkerboard.update!(expmΔτK, t, Δτ)

    return nothing
end