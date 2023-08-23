#############################
## MEASURE HOLSTEIN ENERGY ##
#############################

@doc raw"""
    measure_holstein_energy(electron_phonon_parameters::ElectronPhononParameters{T,E},
                         Gup::Matrix{T}, Gdn::Matrix{T},
                         holstein_id::Int) where {T<:Number, E<:AbstractFloat}

Calculate and return the Holstein interaction energy
```math
\epsilon_{\rm hol} = \left\langle [ \alpha \hat{X}     + \alpha_2 \hat{X}^2
                                  + \alpha_3 \hat{X}^3 + \alpha_4 \hat{X}^4]
                                (\hat{n}_\uparrow + \hat{n}_\downarrow - 1) \right\rangle,
```
for the Holstein coupling definition corresponding to `holstein_id`.
"""
function measure_holstein_energy(electron_phonon_parameters::ElectronPhononParameters{T,E},
                              Gup::Matrix{T}, Gdn::Matrix{T},
                              holstein_id::Int) where {T<:Number, E<:AbstractFloat}

    x = electron_phonon_parameters.x::Matrix{E}
    holstein_parameters = electron_phonon_parameters.holstein_parameters::HolsteinParameters{E}
    ϵ_hol = measure_holstein_energy(holstein_parameters, Gup, Gdn, x, holstein_id)

    return ϵ_hol
end

function measure_holstein_energy(holstein_parameters::HolsteinParameters{E},
                                 Gup::Matrix{T}, Gdn::Matrix{T},
                                 x::Matrix{E}, holstein_id::Int) where {T<:Number, E<:AbstractFloat}

    (; nholstein, Nholstein, α, α2, α3, α4, neighbor_table, coupling_to_phonon) = holstein_parameters

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

    # iterate over unit cells
    for u in eachindex(ctp)
        x_ul = x[ctp[u],Lτ]
        i_ul = nt[2,1]
        nup_ul = 1 - real(Gup[i_ul, i_ul])
        ndn_ul = 1 - real(Gdn[i_ul, i_ul])
        # [α⋅x + α₂⋅x² + α₃⋅x³ + α₄⋅x⁴]⋅(n₊ + n₋ - 1)
        ϵ_hol += (α′[u]*x_ul + α2′[u]*x_ul^2 + α3′[u]*x_ul^3 + α4′[u]*x_ul^4) * (nup_ul + ndn_ul - 1)
    end

    # normalize measurement
    ϵ_hol /= (Nunitcell)

    return ϵ_hol
end