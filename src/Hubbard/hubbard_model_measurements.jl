@doc raw"""
    measure_hubbard_energy(hubbard_parameters::HubbardParameters{E},
                        Gup::Matrix{T}, Gdn::Matrix{T},
                        orbital_id::Int) where {T<:Number, E<:AbstractFloat, D, N}

Calculate the average Hubbard energy ``U \langle \hat{n}_\uparrow \hat{n}_\downarrow \rangle``
for the orbital corresponding `orbital_id` in the unit cell.
"""
function measure_hubbard_energy(hubbard_parameters::HubbardParameters{E},
                             Gup::Matrix{T}, Gdn::Matrix{T},
                             orbital_id::Int) where {T<:Number, E<:AbstractFloat, D, N}

    (; U, orbitals, sites) = hubbard_parameters

    # initialize hubbard energy
    e = zero(E)

    # if orbital has finite hubbard interaction
    if orbital_id in orbitals

        # number of orbitals in the unit cell with finite hubbard interaction
        n_hubbard = length(orbitals)

        # number hubbard interactions in lattice
        N_hubbard = length(U)

        # get number of unit cells in lattice
        N_unitcells = N_hubbard ÷ n_hubbard

        # reshape so each column corresponds to a given orbital with finite hubbard interaction
        U′ = reshape(U, (N_unitcells, n_hubbard))
        sites′ = reshape(sites, (N_unitcells, n_hubbard))

        # get the index associated with the orbital species
        index = findfirst(i -> i==orbital_id, orbitals)

        # calculate the average hubbard interaction for specified orbital species
        @fastmath @inbounds for i in axes(U′,1)
            site = sites′[i,index]
            e += real( U′[i,index] * (1 - Gup[site,site]) * (1 - Gdn[site,site]) )
        end
        e /= N_unitcells
    end

    return e
end