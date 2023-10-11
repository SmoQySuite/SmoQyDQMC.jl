@doc raw"""
    FermionPathIntegral{T<:Number, E<:AbstractFloat}

Represents a fermion path integral. In particular, contains the information to represent each
diagonal on-site energy matrix ``V_l`` and hopping matrix ``K_l`` for each imaginary time
slice``l \in [1, L_\tau],`` such that ``\tau = \Delta\tau \cdot l`` and ``\beta = \Delta\tau \cdot L_\tau.``

# Fields

- `β::E`: Inverse temperature.
- `Δτ::E`: Discretization in imaginary time.
- `Lτ::Int`: Length of the imaginary time axis.
- `N::Int`: Number of orbitals in the lattice.
- `neighbor_table::Matrix{Int}`: Neighbor table for each pair of orbitals in the lattice connected by a hopping.
- `t::Matrix{T}`: Hopping amplitudes for imaginary time slice ``l`` is stored in `t[:,l]`.
- `V::Matrix{T}`: Diagonal on-site energy matrix ``V_l`` for imaginary time slice ``l`` is stored in `V[:,l]`.
- `K::Matrix{T}`: Used to construct hopping matrix to cacluate exponentiated hopping matrix if checkerboard approximation is not being used.
- `eigen_ws::HermitianEigenWs{T,Matrix{T},E}`: For calculating eigenvalues and eigenvectors of `K` while avoiding dynamic memory allocations.
- `u::Vector{T}`: Temporary vector to avoid dynamic allocation when performing local updates.
- `v::Vector{T}`: Temporary vector to avoid dynamic allocation when performing local updates.
"""
struct FermionPathIntegral{T<:Number, E<:AbstractFloat}

    β::E

    Δτ::E

    Lτ::Int

    N::Int

    neighbor_table::Matrix{Int}

    t::Matrix{T}

    V::Matrix{T}

    K::Matrix{T}

    eigen_ws::HermitianEigenWs{T,Matrix{T},E}

    u::Vector{T}

    v::Vector{T}
end


@doc raw"""
    FermionPathIntegral(; tight_binding_parameters::TightBindingParameters{T,E},
                        β::E, Δτ::E) where {T,E}

Initialize an instance of [`FermionPathIntegral`](@ref) an instance of [`TightBindingParameters`](@ref).
"""
function FermionPathIntegral(; tight_binding_parameters::TightBindingParameters{T,E},
                             β::E, Δτ::E) where {T,E}

    # rename for convenience
    tbp = tight_binding_parameters
    
    (; ϵ, neighbor_table) = tbp

    # evaluate length of imaginary time axis
    Lτ = eval_length_imaginary_axis(β, Δτ)

    # get number of orbitals in lattice
    Norbitals = length(ϵ)

    # number of pairs of neighboring orbitals connected by a bond in the lattice
    Nneighbors = size(neighbor_table, 2)

    # initialize hoppings to zero
    t = zeros(T, Nneighbors, Lτ)

    # set the value of the hopping for each imaginary time slice based on the
    # non-interacting tight binding model parameters
    for l in axes(t, 2)
        @views @. t[:,l] = tbp.t
    end

    # initialize diagonal on-site energy matrix
    V = zeros(T, Norbitals, Lτ)

    # set the value on-site energy for each imaginary time slice based on the
    # non-interacting tight-binding model parameters
    for l in axes(V, 2)
        @views @. V[:,l] = tbp.ϵ - tbp.μ
    end

    # initialize hopping matrix
    K = zeros(T, Norbitals, Norbitals)

    # initialize workspace for calculating eigenvalue and eigenvector of K while avoid allocations
    eigen_ws = HermitianEigenWs(K, vecs=true)

    # allocate temporary vectors
    u = zeros(T, Norbitals)
    v = zeros(T, Norbitals)

    return FermionPathIntegral(β, Δτ, Lτ, Norbitals, neighbor_table, t, V, K, eigen_ws, u, v)
end


@doc raw"""
    initialize_propagators(fpi::FermionPathIntegral{T,E}; symmetric::Bool, checkerboard::Bool) where {T,E}

Initialize a propagator for each imaginary time slice, returning a vector of type `Vector{<:AbstractPropagators{T,E}}`.
"""
function initialize_propagators(fpi::FermionPathIntegral{T,E}; symmetric::Bool, checkerboard::Bool) where {T,E}

    # allocate propagators
    B = allocate_propagators(fpi, symmetric=symmetric, checkerboard=checkerboard)

    # calculate propagator matrices
    calculate_propagators!(B, fpi, calculate_exp_V=true, calculate_exp_K=true)

    return B
end


@doc raw"""
    allocate_propagators(fpi::FermionPathIntegral{T,E}; symmetric::Bool, checkerboard::Bool) where {T,E}

Allocate and return a vector of propagators of type `Vector{<:AbstractPropagator{T,E}}`, such that each propagator
in the returned vector is initialized to equal the identity matrix.
If `symmetric = true`, then each propagator matrix is symmetric/Hermitian.
If `checkerboard = true`, then the exponentiated hopping matrix is represented using the checkerboard approximation.
"""
function allocate_propagators(fpi::FermionPathIntegral{T,E}; symmetric::Bool, checkerboard::Bool) where {T,E}

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
function _allocate_symmetric_exact_propagators(fpi::FermionPathIntegral{T,E}) where {T,E}

    (; Lτ, N) = fpi

    # initialize vector of propagators
    B = SymExactPropagator{T,E}[]

    # iterate over imaginary time slice
    for l in 1:Lτ

        # allocate propagator for the current imaginary time slice
        expmΔτKo2_l = Matrix{T}(I, N, N)
        exppΔτKo2_l = Matrix{T}(I, N, N)
        expmΔτV_l = ones(E, N)
        B_l = SymExactPropagator(expmΔτV_l, expmΔτKo2_l, exppΔτKo2_l)
        push!(B, B_l)
    end

    return B
end

# allocate asymmetric propagators with exactly exponentiated hopping matrix
function _allocate_asymmetric_exact_propagators(fpi::FermionPathIntegral{T,E}) where {T,E}

    (; Lτ, N) = fpi

    # initialize vector of propagators
    B = AsymExactPropagator{T,E}[]

    # iterate over imaginary time slice
    for l in 1:Lτ

        # allocate propagator for the current imaginary time slice
        expmΔτK_l = Matrix{T}(I, N, N)
        exppΔτK_l = Matrix{T}(I, N, N)
        expmΔτV_l = ones(E, N)
        B_l = AsymExactPropagator(expmΔτV_l, expmΔτK_l, exppΔτK_l)
        push!(B, B_l)
    end

    return B
end

# allocate symmetric checkerboard propagators
function _allocate_symmetric_checkerboard_propagators(fpi::FermionPathIntegral{T,E}) where {T,E}

    (; Δτ, Lτ, N, neighbor_table) = fpi

    # initialize vector of propagators
    B = SymChkbrdPropagator{T,E}[]

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
        expmΔτV_l = ones(E, N)
        B_l = SymChkbrdPropagator(expmΔτV_l, expmΔτKo2_l)
        push!(B, B_l)
    end

    return B
end

# allocate asymmetric checkerboard propagators
function _allocate_asymmetric_checkerboard_propagators(fpi::FermionPathIntegral{T,E}) where {T,E}

    (; Δτ, Lτ, N, neighbor_table) = fpi

    # initialize vector of propagators
    B = AsymChkbrdPropagator{T,E}[]

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
        expmΔτV_l = ones(E, N)
        B_l = AsymChkbrdPropagator(expmΔτV_l, expmΔτK_l)
        push!(B, B_l)
    end

    return B
end


@doc raw"""
    calculate_propagators!(B::Vector{P}, fpi::FermionPathIntegral{T,E};
                           calculate_exp_V::Bool, calculate_exp_K::Bool) where {T, E, P<:AbstractPropagator{T,E}}

Calculate the propagator matrices ``B_l``, given by `B[l]`, for all imaginary time slice ``\tau = \Delta\tau \cdot l.``
If `calculate_exp_V = true`, then calculate the diagonal exponentiated on-site energy matrices.
If `calculate_exp_K = true`, then calculate the exponentiated hopping matrices.
"""
function calculate_propagators!(B::Vector{P}, fpi::FermionPathIntegral{T,E};
                                calculate_exp_V::Bool, calculate_exp_K::Bool) where {T, E, P<:AbstractPropagator{T,E}}

    # iterate over imaginary time slices
    for l in eachindex(B)

        # calculate propagator for current imaginary time slice τ=Δτ⋅l
        calculate_propagator!(B[l], fpi, l, calculate_exp_V=calculate_exp_V, calculate_exp_K=calculate_exp_K)
    end

    return nothing
end

@doc raw"""
    calculate_propagator!(B::AbstractPropagator{T,E}, fpi::FermionPathIntegral{T,E}, l::Int;
                          calculate_exp_V::Bool, calculate_exp_K::Bool) where {T,E}

Calculate the propagator matrix ``B_l`` for imaginary time slice ``\tau = \Delta\tau \cdot l.``
If `calculate_exp_V = true`, then calculate the diagonal exponentiated on-site energy matrix.
If `calculate_exp_K = true`, then calculate the exponentiated hopping matrix.
"""
function calculate_propagator!(B::AbstractPropagator{T,E}, fpi::FermionPathIntegral{T,E}, l::Int;
                               calculate_exp_V::Bool, calculate_exp_K::Bool) where {T,E}

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


@doc raw"""
    calculate_exp_V!(B::Vector{P}, fpi::FermionPathIntegral{T,E}) where {T, E, P<:AbstractPropagator{T,E}}

Calculate the diagonal exponentiated on-site energy matrices ``exp(-\Delta\tau V_l)`` appearing in each propagator ``B_l``
for all imaginary time slices ``\tau = \Delta\tau \cdot l.``
"""
function calculate_exp_V!(B::Vector{P}, fpi::FermionPathIntegral{T,E}) where {T, E, P<:AbstractPropagator{T,E}}

    # iterate over imaginary time slices
    @fastmath @inbounds for l in eachindex(B)

        # calculate exponentiated diagonal on-site energy matrix for imaginary time slice τ=Δτ⋅l
        calculate_exp_V!(B[l], fpi, l)
    end

    return nothing
end

@doc raw"""
    calculate_exp_V!(B::AbstractPropagator{T,E}, fpi::FermionPathIntegral{T,E}, l::Int) where {T, E}

Calculate the diagonal exponentiated on-site energy matrix ``exp(-\Delta\tau V_l)`` appearing in the propagator `B`
for imaginary time slice `l`.
"""
function calculate_exp_V!(B::AbstractPropagator{T,E}, fpi::FermionPathIntegral{T,E}, l::Int) where {T, E}

    @views @. B.expmΔτV = exp(-fpi.Δτ * fpi.V[:,l])

    return nothing
end


@doc raw"""
    calculate_exp_K!(B::Vector{P}, fpi::FermionPathIntegral{T,E}) where {T, E, P<:AbstractPropagator{T,E}}

Calculate the exponentiated hopping matrix for each propagator matrix in the vector `B`.
"""
function calculate_exp_K!(B::Vector{P}, fpi::FermionPathIntegral{T,E}) where {T, E, P<:AbstractPropagator{T,E}}

    # iterate over imaginary time slices
    for l in eachindex(B)

        # calculate exponentiated hopping matrix for imaginary time slice τ=Δτ⋅l
        calculate_exp_K!(B[l], fpi, l)
    end

    return nothing
end

@doc raw"""
    calculate_exp_K!(B::AbstractPropagator{T,E}, fpi::FermionPathIntegral{T,E}, l::Int) where {T, E}

Calculate the exponentiated hopping matrix appearing in `B` for imaginary time slice `l`.
"""
function calculate_exp_K!(B::AbstractPropagator{T,E}, fpi::FermionPathIntegral{T,E}, l::Int) where {T, E}

    _calculate_exp_K!(B, fpi, l)

    return nothing
end

# calculate exponentiated hopping matrix for imaginary time slice l
function _calculate_exp_K!(B::SymExactPropagator{T,E}, fpi::FermionPathIntegral{T,E}, l::Int) where {T,E}

    (; expmΔτKo2, exppΔτKo2) = B
    (; Δτ, K, neighbor_table, eigen_ws) = fpi

    # get hopping energeis for imaginary time slice τ=Δτ⋅l
    t = @view fpi.t[:,l]

    # build the hopping matrix
    build_hopping_matrix!(K, neighbor_table, t)

    # calculate exponentiated hopping matrices
    exp!(expmΔτKo2, exppΔτKo2, K, -Δτ/2, workspace = eigen_ws)

    return nothing
end

# calculate exponentiated hopping matrix for imaginary time slice l
function _calculate_exp_K!(B::AsymExactPropagator{T,E}, fpi::FermionPathIntegral{T,E}, l::Int) where {T,E}

    (; expmΔτK, exppΔτK) = B
    (; Δτ, K, neighbor_table, eigen_ws) = fpi

    # get hopping energeis for imaginary time slice τ=Δτ⋅l
    t = @view fpi.t[:,l]

    # build the hopping matrix
    build_hopping_matrix!(K, neighbor_table, t)

    # calculate exponentiated hopping matrices
    exp!(expmΔτK, exppΔτK, K, -Δτ, workspace = eigen_ws)

    return nothing
end

# calculate checkerboard approximation for imaginary time slice l
function _calculate_exp_K!(B::SymChkbrdPropagator{T,E}, fpi::FermionPathIntegral{T,E}, l::Int) where {T,E}

    (; expmΔτKo2) = B
    (; Δτ) = fpi

    # get hopping energeis for imaginary time slice τ=Δτ⋅l
    t = @view fpi.t[:,l]

    # update checkerboard factorization
    Checkerboard.update!(expmΔτKo2, t, Δτ/2)

    return nothing
end

# calculate checkerboard approximation for imaginary time slice l
function _calculate_exp_K!(B::AsymChkbrdPropagator{T,E}, fpi::FermionPathIntegral{T,E}, l::Int) where {T,E}

    (; expmΔτK) = B
    (; Δτ) = fpi

    # get hopping energeis for imaginary time slice τ=Δτ⋅l
    t = @view fpi.t[:,l]

    # update checkerboard factorization
    Checkerboard.update!(expmΔτK, t, Δτ)

    return nothing
end