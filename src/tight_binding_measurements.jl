@doc raw"""
    measure_onsite_energy(tight_binding_parameters::TightBindingParameters{T,E},
                          Gup::Matrix{T}, Gdn::Matrix{T},
                          orbital_id::Int) where {T<:Number, E<:AbstractFloat, D, N}

Measure and return the on-site energy ``\epsilon_\textrm{on-site} = (\epsilon - \mu)\langle \hat{n}_\uparrow + \hat{n}_\downarrow \rangle``
for the `orbital_id` in the unit cell.
"""
function measure_onsite_energy(tight_binding_parameters::TightBindingParameters{T,E},
                               Gup::Matrix{T}, Gdn::Matrix{T},
                               orbital_id::Int) where {T<:Number, E<:AbstractFloat}

    (; ϵ, μ, norbital) = tight_binding_parameters

    # initialize energy to zero
    e = zero(E)

    # number of orbitals in lattice
    Norbital = size(Gup, 1)

    # number of unit cells in lattice
    Nunitcells = Norbital ÷ norbital

    # iterate over unit cells
    for u in 1:Nunitcells
        # get the site in the unit cell associated with specified orbital species
        i = (u-1)*norbital + orbital_id
        # calculate on-site energy
        e += (ϵ[i] - μ) * (2 - Gup[i,i] - Gdn[i,i])
    end
    # normalize measurement
    e /= Nunitcells

    return e
end


@doc raw"""
    measure_hopping_energy(tight_binding_parameters::TightBindingParameters{T,E},
                           Gup::Matrix{T}, Gdn::Matrix{T},
                           bond_id::Int) where {T<:Number, E<:AbstractFloat}

Calculate the average hopping energy
``\epsilon_{\rm hopping} = -\sum_{\sigma} \langle t_{ij} \hat{c}^\dagger_{\sigma,i} \hat{c}_{\sigma,j} + {\rm h.c.} \rangle``
for the hopping defined by the bond corresponding to `bond_id`.
"""
function measure_hopping_energy(tight_binding_parameters::TightBindingParameters{T,E},
                                Gup::Matrix{T}, Gdn::Matrix{T},
                                bond_id::Int) where {T<:Number, E<:AbstractFloat}

    (; t, neighbor_table, bond_slices, bond_ids) = tight_binding_parameters

    # initialize hopping energy to zero
    h = zero(T)

    # check if hopping associated with bond ID
    if bond_id in bond_ids

        # get the bond slice index associated with bond id
        bond_id_index = findfirst(i -> i==bond_id, bond_ids)

        # get the neighbor table associated with the bond/hopping in question
        nt = @view neighbor_table[:, bond_slices[bond_id_index]]

        # get the hopping associated with the bond/hopping in question
        t′ = @view t[bond_slices[bond_id_index]]

        # iterate over each bond/hopping
        @fastmath @inbounds for n in axes(nt, 2)
            # get the pair of hoppings
            i = nt[1,n]
            j = nt[2,n]
            # calculate the hopping energy
            hup = -Gup[i,j]
            hdn = -Gdn[i,j]
            h -= t′[n] * (hup + hdn)
            hup = -Gup[j,i]
            hdn = -Gdn[j,i]
            h -= conj(t′[n]) * (hup + hdn)
        end

        # noramalize the measurement
        h /= length(t′)
    end

    return h
end