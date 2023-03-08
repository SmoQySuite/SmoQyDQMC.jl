@doc raw"""
    TightBindingModel{T<:Number, E<:AbstractFloat, D}

Defines a tight binding model in `D` dimensions.

# Fields

- `μ::E`: Chemical potential.
- `ϵ_mean::Vector{E}`: Mean on-site energy for each orbital in the unit cell. 
- `ϵ_std::Vector{E}`: Standard deviation of on-site energy for each orbital in the unit cell.
- `t_bond_ids::Vector{Int}`: The bond ID for each bond/hopping definition.
- `t_bonds::Vector{Bond{D}}`: Bond definition for each type of hopping in the tight binding model.
- `t_mean::Vector{T}`: Mean hopping energy for each type of hopping.
- `t_std::Vector{E}`: Standard deviation of hopping energy for each type of hopping.
"""
struct TightBindingModel{T<:Number, E<:AbstractFloat, D}
    
    # chemical potential
    μ::E

    # mean on-site energy for each orbital in unit cell
    ϵ_mean::Vector{E}

    # standard deviation of on-site energy for each orbital in unit cell
    ϵ_std::Vector{E}

    # bond ID associated with each hopping in tight-binding model
    t_bond_ids::Vector{Int}

    # bond definition associated with each hopping
    t_bonds::Vector{Bond{D}}

    # mean hopping energy for each type of hopping in tight-binding model
    t_mean::Vector{T}

    # standard deviation of hopping energy for each type of hopping in tight-binding model
    t_std::Vector{E}
end

@doc raw"""
    TightBindingModel(; model_geometry::ModelGeometry{D,E,N},
                      μ::E,
                      ϵ_mean::Vector{E},
                      ϵ_std::Vector{E} = zeros(eltype(ϵ_mean), length(ϵ_mean)),
                      t_bonds::Vector{Bond{D}} = Bond{ndims(model_geometry.unit_cell)}[],
                      t_mean::Vector{T} = eltype(ϵ_mean)[],
                      t_std::Vector{E} = zeros(eltype(ϵ_mean), length(t_mean))) where {T<:Number, E<:AbstractFloat, D, N}

Initialize and return an instance of [`TightBindingModel`](@ref), also adding/recording the bond defintions `t_bonds` to the
[`ModelGeometry`](@ref) instance `model_geometry`.
"""
function TightBindingModel(; model_geometry::ModelGeometry{D,E,N},
                           μ::E,
                           ϵ_mean::Vector{E},
                           ϵ_std::Vector{E} = zeros(eltype(ϵ_mean), length(ϵ_mean)),
                           t_bonds::Vector{Bond{D}} = Bond{ndims(model_geometry.unit_cell)}[],
                           t_mean::Vector{T} = eltype(ϵ_mean)[],
                           t_std::Vector{E} = zeros(eltype(ϵ_mean), length(t_mean))) where {T<:Number, E<:AbstractFloat, D, N}

    # get the number of orbitals per unit cell
    unit_cell = model_geometry.unit_cell::UnitCell{D,E,N}
    n = unit_cell.n

    # check that on-site energy is defined for each orbitals
    @assert length(ϵ_mean) == n
    @assert length(ϵ_std) == n

    # record bonds associated with hoppings
    t_bond_ids = zeros(Int, length(t_bonds))
    for i in eachindex(t_bonds)
        t_bond_ids[i] = add_bond!(model_geometry, t_bonds[i])
    end

    return TightBindingModel(μ, ϵ_mean, ϵ_std, t_bond_ids, t_bonds, t_mean, t_std)
end


# show struct info as TOML formatted string for real hopping energies
function Base.show(io::IO, ::MIME"text/plain", tbm::TightBindingModel{T,E,D}) where {T<:AbstractFloat,E,D}

    @printf io "[tight_binding_model]\n\n"
    @printf io "chemical_potential = %.8f\n\n" tbm.μ
    @printf io "[tight_binding_model.onsite_energy]\n\n"
    @printf io "e_mean = %s\n" string(tbm.ϵ_mean)
    @printf io "e_std  = %s\n\n" string(tbm.ϵ_std)
    for i in eachindex(tbm.t_bonds)
        @printf io "[[tight_binding_model.hopping]]\n\n"
        @printf io "ID           = %d\n" i
        @printf io "bond_id      = %d\n" tbm.t_bond_ids[i]
        @printf io "orbitals     = [%d, %d]\n" tbm.t_bonds[i].orbitals[1] tbm.t_bonds[i].orbitals[2]
        @printf io "displacement = %s\n" string(tbm.t_bonds[i].displacement)
        @printf io "t_mean       = %.8f\n" tbm.t_mean[i]
        @printf io "t_std        = %.8f\n\n" real(tbm.t_std[i])
    end

    return nothing
end

# show struct info as TOML formatted string for complex hopping energies
function Base.show(io::IO, ::MIME"text/plain", tbm::TightBindingModel{T,E,D}) where {T<:Complex,E,D}

    @printf io "[tight_binding_model]\n\n"
    @printf io "chemical_potential = %.8f\n\n" tbm.μ
    @printf io "[tight_binding_model.onsite_energy]\n\n"
    @printf io "e_mean = %s\n" string(tbm.ϵ_mean)
    @printf io "e_std  = %s\n\n" string(tbm.ϵ_mean)
    for i in eachindex(tbm.t_bonds)
        @printf io "[[tight_binding_model.hopping]]\n\n"
        @printf io "ID           = %d\n" i
        @printf io "bond_id      = %d\n" tbm.t_bond_ids[i]
        @printf io "orbitals     = [%d, %d]\n" tbm.t_bonds[i].orbitals[1] tbm.t_bonds[i].orbitals[2]
        @printf io "displacement = %s\n" string(tbm.t_bonds[i].displacement)
        @printf io "t_mean_real  = %.8f\n" real(tbm.t_mean[i])
        @printf io "t_mean_imag  = %.8f\n" imag(tbm.t_mean[i])
        @printf io "t_std        = %.8f\n\n" real(tbm.t_std[i])
    end

    return nothing
end

@doc raw"""
    TightBindingParameters{T<:Number, E<:AbstractFloat}

A mutable struct containing all the parameters needed to characterize a finite tight-binding Hamiltonian
for a single spin species ``\sigma`` on a finite lattice with periodic boundary conditions of the form
```math
\hat{H}_{0,\sigma}=-\sum_{\langle i,j\rangle}(t_{ij} \hat{c}_{\sigma,i}^{\dagger}\hat{c}_{\sigma,j}+\textrm{h.c.})+\sum_{i}(\epsilon_{i}-\mu)\hat{n}_{\sigma,i},
```
where ``\hat{c}_{\sigma,i}^\dagger`` is the fermion creation operator for an electron with spin ``\sigma`` on orbital ``i,``
``t_{i,j}`` are the hopping energies, ``\epsilon_i`` are the on-site energies for each orbital in the lattice,
and ``\mu`` is the chemical potential.

# Fields

- `μ::E`: The chemical potential ``\mu.``
- `const ϵ::Vector{E}`: A vector containing the on-site energy ``\epsilon_i`` for each orbital in the lattice.
- `const t::Vector{T}`: The hopping energy ``t_{i,j}`` associated with each pair of neighboring orbitals connected by a bond in the lattice.
- `const neighbor_table::Matrix{Int}`: Neighbor table containing all pairs of orbitals in the lattices connected by a bond, with a non-zero hopping energy between them.
- `const bond_ids::Vector{Int}`: The bond ID definitions that define the types of hopping in the lattice.
- `const bond_slices::Vector{UnitRange{Int}}`: Slices of `neighbor_table` corresponding to given bond ID i.e. the neighbors `neighbor_table[:,bond_slices[1]]` corresponds the `bond_ids[1]` bond defintion.
- `const norbital::Int`: Number of orbitals per unit cell.
"""
mutable struct TightBindingParameters{T<:Number, E<:AbstractFloat}

    # chemical potential
    μ::E

    # on-site energy for each orbital
    const ϵ::Vector{E}

    # hopping energies for all pairs of orbitals connected by a bond in the lattice
    const t::Vector{T}

    # neighbor table for all pairs of orbitals connected by a bond in the lattice
    const neighbor_table::Matrix{Int}

    # bond IDs that define hoppings
    const bond_ids::Vector{Int}

    # view into neighbor table for each bond ID
    const bond_slices::Vector{UnitRange{Int}}

    # number of orbitals per unit cell
    const norbital::Int
end


@doc raw"""
    TightBindingParameters(; tight_binding_model::TightBindingModel{T,E,D},
                           model_geometry::ModelGeometry{D,E},
                           rng::AbstractRNG) where {T,E,D}

Initialize and return an instance of [`TightBindingParameters`](@ref).
"""
function TightBindingParameters(; tight_binding_model::TightBindingModel{T,E,D},
                                model_geometry::ModelGeometry{D,E},
                                rng::AbstractRNG) where {T,E,D}

    (; unit_cell, lattice) = model_geometry
    N = lattice.N # number of unit cells in lattice
    n = unit_cell.n # number of orbital per unit cell

    # set chemical potential
    μ = tight_binding_model.μ

    # get the number of orbitals in the lattice
    N_sites = nsites(unit_cell, lattice)

    # set on-site energy for each orbital in lattice
    ϵ = zeros(E, N_sites)
    ϵ′ = reshape(ϵ, n, N)
    for i in 1:n
        ϵ_i = @view ϵ′[i,:]
        randn!(rng, ϵ_i)
        @. ϵ_i = tight_binding_model.ϵ_mean[i] + tight_binding_model.ϵ_std[i] * ϵ_i
    end

    # get number of bond definitions in model
    nbonds = length(tight_binding_model.t_bonds)

    # get the id for for each bond
    bond_ids = copy(tight_binding_model.t_bond_ids)
    bond_slices = UnitRange{Int}[]
    for i in eachindex(bond_ids)
        push!(bond_slices, (i-1)*N+1:i*N)
    end

    # construct neighbor table for hoppings
    t_bonds = tight_binding_model.t_bonds::Vector{Bond{D}}
    if length(t_bonds) > 0
        neighbor_table = build_neighbor_table(t_bonds, unit_cell, lattice)
    else
        neighbor_table = Matrix{Int}(undef,2,0)
    end

    # get total number of bonds in lattice
    Nbonds = size(neighbor_table, 2)

    # set hopping energy for each bond in lattice
    t = zeros(T, Nbonds)
    if Nbonds > 0
        t′ = reshape(t, (N, nbonds))
        for b in axes(t′,2)
            t_b = @view t′[:,b]
            randn!(rng, t_b)
            @. t_b = tight_binding_model.t_mean[b] + tight_binding_model.t_std[b] * t_b
        end
    end

    return TightBindingParameters(μ, ϵ, t, neighbor_table, bond_ids, bond_slices, n)
end