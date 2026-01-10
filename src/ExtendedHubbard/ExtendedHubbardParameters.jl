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
                # shift on-site energies by +V
                V[i,l] = V[i,l] + 1.0*V′[n]
                V[j,l] = V[j,l] + 1.0*V′[n]
            end
        end
    end

    return nothing
end