@doc raw"""
    measure_ext_hub_energy(
        ext_hub_params::ExtendedHubbardParameters{E},
        Gup::Matrix{T}, Gdn::Matrix{T},
        ext_hub_id::Int
    ) where {T<:Number, E<:AbstractFloat}

Measure the extended Hubbard interaction energy
```math
V (\hat{n}_i-1)(\hat{n}_j-1)
```
if `ph_sym_form = true` and
```math
V \hat{n}_i \hat{n}_j
``` 
if `ph_sym_form = false` for the specified `EXT_HUB_ID`.
"""
function measure_ext_hub_energy(
    ext_hub_params::ExtendedHubbardParameters{E},
    Gup::Matrix{T}, Gdn::Matrix{T},
    ext_hub_id::Int
) where {T<:Number, E<:AbstractFloat}

    (; V, neighbor_table, bond_ids, ph_sym_form) = ext_hub_params

    # number of types of extended hubbard interactions
    n = length(bond_ids)

    # number of unit cells
    N = size(neighbor_table, 2) ÷ n

    # get appropriate view in interaction
    V′ = reshape(V, (N, n))
    V′ = @view V′[:,ext_hub_id]

    # get appropriate view into neighbor table
    nt = reshape(neighbor_table, (2, N, n))
    nt = @view nt[:,:,ext_hub_id]

    # initialize measurement to zero
    ϵ = zero(T)

    # iterate over coupled sites
    for n in 1:N
        # get the pair of sites that are coupled
        i = nt[1,n]
        j = nt[2,n]
        # get the relevant densities
        n_i_up = 1 - Gup[i,i]
        n_i_dn = 1 - Gdn[i,i]
        n_j_up = 1 - Gup[j,j]
        n_j_dn = 1 - Gdn[j,j]
        # get interaction strength
        Vij = V′[n]
        # calculate interaction energy
        if ph_sym_form
            ϵ += Vij * (n_i_up + n_i_dn - 1) * (n_j_up + n_j_dn - 1)
        else
            ϵ += Vij * (n_i_up + n_i_dn) * (n_j_up + n_j_dn)
        end
    end

    # normalize measurement
    ϵ = ϵ / N

    return ϵ
end