@doc raw"""
    ExtendedHubbardParameters{T<:AbstractFloat}

Extended Hubbard interaction parameters for finite lattice.

# Fields

- `V::Vector{T}`: Extended Hubbard interaction strength for each pair neighbors in the lattice.
- `neighbor_table::Matrix{Int}`: Neighbor table for extended Hubbard interactions on lattice.
- `bond_ids::Vector{Int}`: Bond IDs used to define extended Hubbard interactions.
- `ph_sym_form::Bool`: Whether particle-hole symmetric form of extended Hubbard interaction was used.
"""
struct ExtendedHubbardParameters{T<:AbstractFloat}

    # extended hubbard interaction strength between pair of orbitals
    V::Vector{T}

    # neighbor table for extended hubbard interactions
    neighbor_table::Matrix{Int}

    # bond IDs used to define extended hubbard interactions
    bond_ids::Vector{Int}

    # whether particle-hole symmetric form for interaction was used
    ph_sym_form::Bool
end

@doc raw"""
    ExtendedHubbardParameters(;
        # KEYWORD ARGUMENTS
        extended_hubbard_model::ExtendedHubbardModel{T},
        model_geometry::ModelGeometry{D,T},
        rng::AbstractRNG
    ) where {D, T<:AbstractFloat}

Initialize an instance of the [`ExtendedHubbardParameters`](@ref) type.
"""
function ExtendedHubbardParameters(;
    # KEYWORD ARGUMENTS
    extended_hubbard_model::ExtendedHubbardModel{T},
    model_geometry::ModelGeometry{D,T},
    rng::AbstractRNG
) where {D, T<:AbstractFloat}

    (; V_bond_ids, V_mean, V_std, ph_sym_form) = extended_hubbard_model
    (; bonds, unit_cell, lattice) = model_geometry

    # number of unit cells in lattice
    N = lattice.N

    # construct neighbor table to extended Hubbard interactions
    V_bonds = @view bonds[V_bond_ids]
    neighbor_table = build_neighbor_table(V_bonds, unit_cell, lattice)

    # number of types of extended hubbard interactions
    n = length(V_bond_ids)

    # calculate the interaction stength between each pair of sites
    V  = zeros(T, N*n)
    V′ = reshape(V, (N, n))

    # iterate of types of extended Hubbard interactions
    for i in 1:n
        # iterate over unit cells
        for u in 1:N
            V′[u,i] = V_mean[i] + randn(rng) * V_std[i]
        end
    end

    return ExtendedHubbardParameters{T}(V, neighbor_table, V_bond_ids, ph_sym_form)
end


@doc raw"""
    init_renormalized_hubbard_parameters(;
        # KEYWORD ARGUMENTS
        hubbard_parameters::HubbardParameters{T},
        extended_hubbard_parameters::ExtendedHubbardParameters{T},
        model_geometry::ModelGeometry{D,T}
    ) where {D, T<:AbstractFloat}

Returns a new instance of the [`HubbardParameters`](@ref) type with the Hubbard interactions renormalized
based on the [`ExtendedHubbardParameters`](@ref) definition. Refer to the definition of the
[`ExtendedHubbardModel`](@ref) to see where this renormalization of the local Hubbard interaction comes from.

Note that either both the local and extended Hubbard interactions need to be initialized using the particle-hole
symmetric or asymmetric form for the interaction (as specified by `ph_sym_form` keyword argument), and cannot use opposite conventions.
Additionally, the [`HubbardModel`](@ref) defintion used to create the `hubbard_parameters` instance of the [`HubbardParameters`](@ref)
passed to this function must initialize a Hubbard interaction on each type of orbital species/ID appearing
in an extended Hubbard interaction, even if this means initializing the local Hubbard interaction to ``U = 0``.
"""
function init_renormalized_hubbard_parameters(;
    # KEYWORD ARGUMENTS
    hubbard_parameters::HubbardParameters{T},
    extended_hubbard_parameters::ExtendedHubbardParameters{T},
    model_geometry::ModelGeometry{D,T}
) where {D, T<:AbstractFloat}

    @assert(
        hubbard_parameters.ph_sym_form == extended_hubbard_parameters.ph_sym_form,
        "Either both the local and extended Hubbard interactions need to use the particle-hole symmetric or asymmetric form, they cannot use opposite conventions."
    )

    (; U, sites, orbital_ids, ph_sym_form) = hubbard_parameters
    (; V, neighbor_table, bond_ids) = extended_hubbard_parameters
    (; bonds, unit_cell, lattice) = model_geometry

    
    # iterate over bonds that define extended hubbard interactions
    for bond_id in bond_ids

        # get orbital species associated with bond
        a, b = bonds[bond_id].orbitals

        # check to make sure Hubbard interaction are defined for each type of bond
        @assert (a ∈ orbital_ids) "Hubbard interaction for ORBITAL_ID = $a needs to initialized in HubbardModel definition. Note that initialization to U = 0 is allowed."
        @assert (b ∈ orbital_ids) "Hubbard interaction for ORBITAL_ID = $b needs to initialized in HubbardModel definition. Note that initialization to U = 0 is allowed."
    end

    # number of unit cells in lattice
    N = lattice.N

    # number of types of extended Hubbard interactions
    n_V = length(bond_ids)

    # number of types of Hubbard interactions
    n_U = length(orbital_ids)

    # copy bare hubbard interaction
    Ũ = copy(U)

    # iterate over extended Hubbard interaction neighbors
    for n in axes(neighbor_table, 2)

        # get the pair of orbitals with extended Hubbard interactions
        i = neighbor_table[1,n]
        j = neighbor_table[2,n]

        # get the Hubbard U index associated with each site
        m_i = findfirst(e -> e == i, sites)
        m_j = findfirst(e -> e == j, sites)

        # calculate renormalized hubbard interaction
        Ũ[m_i] -= V[n]
        Ũ[m_j] -= V[n]
    end

    return HubbardParameters(Ũ, sites, orbital_ids, ph_sym_form)
end


@doc raw"""
    initialize!(
        fermion_path_integral_up::FermionPathIntegral,
        fermion_path_integral_dn::FermionPathIntegral,
        extended_hubbard_parameters::ExtendedHubbardParameters
    )

    initialize!(
        fermion_path_integral::FermionPathIntegral,
        extended_hubbard_parameters::ExtendedHubbardParameters
    )

Initialize the contribution from the Hubbard interaction to a [`FermionPathIntegral`](@ref) instance.
"""
function initialize!(
    fermion_path_integral_up::FermionPathIntegral,
    fermion_path_integral_dn::FermionPathIntegral,
    extended_hubbard_parameters::ExtendedHubbardParameters
)

    initialize!(fermion_path_integral_up, extended_hubbard_parameters)
    initialize!(fermion_path_integral_dn, extended_hubbard_parameters)

    return nothing
end

function initialize!(
    fermion_path_integral::FermionPathIntegral,
    extended_hubbard_parameters::ExtendedHubbardParameters
)

    (; ph_sym_form, neighbor_table) = extended_hubbard_parameters
    V′ = extended_hubbard_parameters.V
    (; V) = fermion_path_integral

    # shift on-site energies if necessary
    if !ph_sym_form
        # iterate over imaginary-time slices
        for l in axes(V,2)
            # iterate of extended Hubbard interaction coupling
            for n in eachindex(V′)
                # get the pair of sites connected by extended hubbard interaction
                i = neighbor_table[1,n]
                j = neighbor_table[2,n]
                # shift on-site energies by +3V/2
                V[i,l] = V[i,l] + 1.5*V′[n]
                V[j,l] = V[j,l] + 1.5*V′[n]
            end
        end
    end

    return nothing
end