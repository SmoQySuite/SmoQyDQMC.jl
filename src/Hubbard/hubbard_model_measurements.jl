@doc raw"""
    measure_hubbard_energy(
        hubbard_parameters::HubbardParameters{E},
        Gup::Matrix{T}, Gdn::Matrix{T},
        hubbard_id::Int
    ) where {T<:Number, E<:AbstractFloat}

Calculate the average Hubbard energy ``U \langle \hat{n}_\uparrow \hat{n}_\downarrow \rangle``
if `ph_sym_form = false` and ``U \langle (\hat{n}_\uparrow - \tfrac{1}{2})(\hat{n}_\downarrow - \tfrac{1}{2})\rangle``
if `ph_sym_form = true` for the orbital corresponding `orbital_id` in the unit cell.
"""
function measure_hubbard_energy(
    hubbard_parameters::HubbardParameters{E},
    Gup::Matrix{T}, Gdn::Matrix{T},
    hubbard_id::Int
) where {T<:Number, E<:AbstractFloat}

    (; U, orbital_ids, sites, ph_sym_form) = hubbard_parameters

    # initialize hubbard energy
    e = zero(T)

    # number of orbitals in the unit cell with finite hubbard interaction
    n_hubbard = length(orbital_ids)

    # number hubbard interactions in lattice
    N_hubbard = length(U)

    # get number of unit cells in lattice
    N_unitcells = N_hubbard ÷ n_hubbard

    # reshape so each column corresponds to a given orbital with finite hubbard interaction
    U′ = reshape(U, (N_unitcells, n_hubbard))
    sites′ = reshape(sites, (N_unitcells, n_hubbard))

    # calculate the average hubbard interaction for specified orbital species
    @fastmath @inbounds for i in axes(U′,1)
        site = sites′[i,hubbard_id]
        nup = 1 - Gup[site,site]
        ndn = 1 - Gdn[site,site]
        if ph_sym_form
            e += U′[i,hubbard_id] * (nup-0.5) * (ndn-0.5)
        else
            e += U′[i,hubbard_id] * nup * ndn
        end
    end
    e /= N_unitcells

    return e
end