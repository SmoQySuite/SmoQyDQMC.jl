@doc raw"""
    HubbardModel{T<:AbstractFloat}

A type to represent a, in general, multi-orbital Hubbard model.

If the type field `ph_sym_form = false`, then the particle-hole asymmetric form for the Hubbard interaction
```math
\hat{H}_{U}=\sum_{\mathbf{i},\nu}U_{\nu,\mathbf{i}}\hat{n}_{\uparrow,\nu,\mathbf{i}}\hat{n}_{\downarrow,\nu,\mathbf{i}}
```
is used, where ``\mathbf{i}`` specifies the unit cell, and ``\nu`` denotes the orbital in the unit cell.
In the case of a bipartite lattice with only nearest neighbor hopping, this convention results in
an on-site energy corresponding to half-filling and particle-hole symmetry
given by ``\epsilon_{\nu,\mathbf{i}} = -U_{\nu,\mathbf{i}}/2.``

If `ph_sym_form = true`, then the particle-hole symmetric form for the Hubbard interaction
```math
\hat{H}_{U}=\sum_{\mathbf{i},\nu}U_{\nu,\mathbf{i}}(\hat{n}_{\uparrow,\nu,\mathbf{i}}-\tfrac{1}{2})(\hat{n}_{\downarrow,\nu,\mathbf{i}}-\tfrac{1}{2})
```
is used instead. In this case, for a bipartite lattice with only nearest neighbor hopping, the on-site energy corresponding to half-filling
and particle-hole symmetry is ``\epsilon_{\nu,\mathbf{i}} = 0.``

# Fields

- `ph_sym_form::Bool`: Determines whether the particle-hole symmetric form of the Hubbard interaction is used.
- `U_orbital_ids::Vector{Int}`: Orbital species/IDs in unit cell with finite Hubbard interaction.
- `U_mean::Vector{T}`: Average Hubbard interaction strength ``U_\nu`` for a given orbital species in the lattice.
- `U_std::Vector{T}`: Standard deviation of Hubbard interaction strength ``U_\nu`` for a given orbital species in the lattice.
"""
struct HubbardModel{T<:AbstractFloat}

    # whether zero on-site energy corresponds to half-filling in atomic limit
    ph_sym_form::Bool

    # orbital species
    U_orbital_ids::Vector{Int}

    # average Hubbard U
    U_mean::Vector{T}

    # standard deviation of Hubbard U
    U_std::Vector{T}
end

@doc raw"""
    HubbardModel(;
        # KEYWORD ARGUMENTS
        ph_sym_form::Bool,
        U_orbital::AbstractVector{Int},
        U_mean::AbstractVector{T},
        U_std::AbstractVector{T} = zero(U_mean)
    ) where {T<:AbstractFloat}

Initialize and return an instance of the type [`HubbardModel`](@ref).

# Keyword Arguments

- `ph_sym_form::Bool`: Determines whether the particle-hole symmetric form of the Hubbard interaction is used.
- `U_orbital::Vector{Int}`: Orbital species/IDs in unit cell with finite Hubbard interaction.
- `U_mean::Vector{T}`: Average Hubbard interaction strength ``U_\nu`` for a given orbital species in the lattice.
- `U_std::Vector{T}`: Standard deviation of Hubbard interaction strength ``U_\nu`` for a given orbital species in the lattice.
"""
function HubbardModel(;
    # KEYWORD ARGUMENTS
    ph_sym_form::Bool,
    U_orbital::AbstractVector{Int},
    U_mean::AbstractVector{T},
    U_std::AbstractVector{T} = zero(U_mean)
) where {T<:AbstractFloat}
    
    return HubbardModel(ph_sym_form, U_orbital, U_mean, U_std)
end


# show struct info as TOML formatted string
function Base.show(io::IO, ::MIME"text/plain", hm::HubbardModel)

    (; U_orbital_ids, U_mean, U_std, ph_sym_form) = hm

    @printf io "[HubbardModel]\n\n"
    @printf io "HUBBARD_IDS = %s\n" string(collect(1:length(U_orbital_ids)))
    @printf io "ORBITAL_IDS = %s\n" string(U_orbital_ids)
    @printf io "U_mean      = %s\n" string(round.(U_mean, digits=6))
    @printf io "U_std       = %s\n" string(round.(U_std, digits=6))
    @printf io "ph_sym_form = %s\n\n" string(ph_sym_form)

    return nothing
end


@doc raw"""
    HubbardParameters{T<:AbstractFloat}

Hubbard parameters for finite lattice.

# Fields

- `U::Vector{T}`: On-site Hubbard interaction for each site with finite Hubbard interaction.
- `sites::Vector{Int}`: Site index associated with each finite Hubbard `U` interaction.
- `orbital_ids::Vector{Int}`: Orbital ID/species in unit cell with finite Hubbard interaction.
- `ph_sym_form::Bool`: Convention used for Hubbard interaction, refer to [`HubbardModel`](@ref) for more information.
"""
struct HubbardParameters{T<:AbstractFloat}

    # Hubbard U for each orbital in the lattice
    U::Vector{T}

    # site index associated with each Hubbard U
    sites::Vector{Int}

    # orbital species in unit cell with finite hubbard interaction
    orbital_ids::Vector{Int}

    # whether zero on-site energy corresponds to half-filling in atomic limit
    ph_sym_form::Bool
end

@doc raw"""
    HubbardParameters(;
        hubbard_model::HubbardModel{T},
        model_geometry::ModelGeometry{D,T},
        rng::AbstractRNG
    ) where {D, T<:AbstractFloat}

Initialize an instance of [`HubbardParameters`](@ref).
"""
function HubbardParameters(;
    hubbard_model::HubbardModel{T},
    model_geometry::ModelGeometry{D,T},
    rng::AbstractRNG
) where {D, T<:AbstractFloat}

    (; U_orbital_ids, U_mean, U_std, ph_sym_form) = hubbard_model
    (; lattice, unit_cell) = model_geometry

    # number of orbitals with finite hubbard interaction in unit cell
    n_hubbard = length(hubbard_model.U_orbital_ids)

    # number of unit cell in lattice
    N_unitcells = lattice.N

    # get the number of HS transformations per time-slice τ that is applied
    N_hubbard = N_unitcells * n_hubbard

    # allocate arrays
    U     = zeros(T, N_hubbard)
    sites = zeros(Int, N_hubbard)

    # reshape the allocated arrays
    U′     = reshape(U, (N_unitcells, n_hubbard))
    sites′ = reshape(sites, (N_unitcells, n_hubbard))

    # total number of orbitals in lattice
    N_orbitals = nsites(unit_cell, lattice)

    # iterate over orbitals in the unit cell with finite hubbard interaction
    for (n,o) in enumerate(U_orbital_ids)
        # iterate over unit cells in the lattice
        for u in 1:N_unitcells
            # calculate the site associated with the hubbard interaction
            sites′[u,n] = loc_to_site(u, o, unit_cell)
            # get the Hubbard U interaction on the site
            U′[u,n] = U_mean[n] + U_std[n] * randn(rng)
        end
    end

    return HubbardParameters(U, sites, U_orbital_ids, ph_sym_form)
end


@doc raw"""
    initialize!(
        fermion_path_integral_up::FermionPathIntegral,
        fermion_path_integral_dn::FermionPathIntegral,
        hubbard_parameters::HubbardParameters
    )

    initialize!(
        fermion_path_integral::FermionPathIntegral,
        hubbard_parameters::HubbardParameters
    )

Initialize the contribution from the Hubbard interaction to a [`FermionPathIntegral`](@ref) instance.
"""
function initialize!(
    fermion_path_integral_up::FermionPathIntegral,
    fermion_path_integral_dn::FermionPathIntegral,
    hubbard_parameters::HubbardParameters
)

    initialize!(fermion_path_integral_up, hubbard_parameters)
    initialize!(fermion_path_integral_dn, hubbard_parameters)

    return nothing
end

function initialize!(
    fermion_path_integral::FermionPathIntegral,
    hubbard_parameters::HubbardParameters
)

    (; ph_sym_form, U, sites) = hubbard_parameters
    (; V) = fermion_path_integral

    # shift on-site energies if necessary
    if !ph_sym_form
        for l in axes(V,2)
            for i in eachindex(U)
                # shift on-site energies by +U/2
                site = sites[i]
                V[site,l] = V[site,l] + U[i]/2
            end
        end
    end

    return nothing
end