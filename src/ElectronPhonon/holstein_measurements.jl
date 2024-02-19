#############################
## MEASURE HOLSTEIN ENERGY ##
#############################

@doc raw"""
    measure_holstein_energy(
        electron_phonon_parameters::ElectronPhononParameters{T,E},
        Gup::Matrix{T}, Gdn::Matrix{T},
        holstein_id::Int
    ) where {T<:Number, E<:AbstractFloat}

Calculate and return both the spin-resolved Holstein interaction energy
```math
\epsilon_{{\rm hol},\sigma} = 
                            \left\langle
                                [
                                    \alpha   \hat{X}   + \alpha_2 \hat{X}^2
                                  + \alpha_3 \hat{X}^3 + \alpha_4 \hat{X}^4
                                ]
                                \left(
                                    \hat{n}_\sigma - \frac{1}{2}
                                \right)
                            \right\rangle,
```
and the total Holstein interaction energy coupling definition corresponding to `holstein_id`.
The method returns `(ϵ_hol, ϵ_hol_up, ϵ_hol_dn)` where `ϵ_hol = ϵ_hol_up + ϵ_hol_dn`.
"""
function measure_holstein_energy(
    electron_phonon_parameters::ElectronPhononParameters{T,E},
    Gup::Matrix{T}, Gdn::Matrix{T},
    holstein_id::Int
) where {T<:Number, E<:AbstractFloat}

    x = electron_phonon_parameters.x::Matrix{E}
    holstein_parameters_up = electron_phonon_parameters.holstein_parameters_up::HolsteinParameters{E}
    ϵ_hol_up = measure_holstein_energy(holstein_parameters_up, Gup, Gdn, x, holstein_id)
    holstein_parameters_dn = electron_phonon_parameters.holstein_parameters_dn::HolsteinParameters{E}
    ϵ_hol_dn = measure_holstein_energy(holstein_parameters_dn, Gup, Gdn, x, holstein_id)
    ϵ_hol = ϵ_hol_up + ϵ_hol_dn

    return ϵ_hol, ϵ_hol_up, ϵ_hol_dn
end

function measure_holstein_energy(
    holstein_parameters::HolsteinParameters{E},
    G::Matrix{T}, x::Matrix{E}, holstein_id::Int
) where {T<:Number, E<:AbstractFloat}

    (; nholstein, Nholstein, α, α2, α3, α4, neighbor_table, coupling_to_phonon, shifted) = holstein_parameters

    # initialize holstein electron-phonon coupling energy to zero
    ϵ_hol = zero(E)

    # length of imaginary time axis
    Lτ = size(x,2)

    # number of unit cells in the lattice
    Nunitcell = Nholstein ÷ nholstein

    # get relevant views into arrays to holstein coupling id 
    slice = (holstein_id-1)*Nunitcell+1 : holstein_id*Nunitcell
    α′  = @view  α[slice]
    α2′ = @view α2[slice]
    α3′ = @view α3[slice]
    α4′ = @view α4[slice]
    nt  = @view neighbor_table[:,slice]
    ctp = @view coupling_to_phonon[slice]

    # if using shifted defintion
    shift = shifted[holstein_id]

    # iterate over unit cells
    for u in eachindex(ctp)
        x_ul = x[ctp[u],Lτ]
        i_ul = nt[2,1]
        n_ul = 1 - real(G[i_ul, i_ul])
        ϵ_hol += (α2′[u]*x_ul^2 + α4′[u]*x_ul^4) * n_ul
        if shift
            ϵ_hol += (α′[u]*x_ul + α3′[u]*x_ul^3) * (n_ul - 0.5)
        else
            ϵ_hol += (α′[u]*x_ul + α3′[u]*x_ul^3) * n_ul
        end
    end

    # normalize measurement
    ϵ_hol /= (Nunitcell)

    return ϵ_hol
end