@doc raw"""
    measure_dispersion_energy(electron_phonon_parameters::ElectronPhononParameters{T,E},
                           dispersion_id::Int) where {T<:Number, E<:AbstractFloat}

Evaluate the average dispersion energy
```math
\epsilon_{\rm disp} = \frac{1}{2} M_{\rm red} \Omega^2 \langle(\hat{X}_i - \hat{X}_j)^2\rangle
                    + \frac{1}{24} M{\rm red} \Omega_4^2 \langle(\hat{X}_i - \hat{X}_j)^4\rangle,
```
where ``M_{\rm red} = \frac{M_i M_j}{M_i + M_j}`` is the reduced mass, for the dispersive coupling
definition specified by `dispersion_id`.
"""
function measure_dispersion_energy(electron_phonon_parameters::ElectronPhononParameters{T,E},
                                dispersion_id::Int) where {T<:Number, E<:AbstractFloat}

    dispersion_parameters = electron_phonon_parameters.dispersion_parameters::DispersionParameters{E}
    phonon_parameters = electron_phonon_parameters.phonon_parameters::PhononParameters{E}
    x = electron_phonon_parameters.x::Matrix{E}
    mass = phonon_parameters.M::Vector{E}

    ϵ_disp = measure_dispersion_energy(dispersion_parameters, mass, x, dispersion_id)

    return ϵ_disp
end

function measure_dispersion_energy(dispersion_parameters::DispersionParameters{T},
                                M::Vector{T}, x::Matrix{T},
                                dispersion_id::Int) where {T<:AbstractFloat}

    (; ndispersion, Ndispersion, Ω, Ω4, dispersion_to_phonon) = dispersion_parameters

    # length of imaginary time axis
    Lτ = size(x,2)

    # number of unit cells in lattice
    Nunitcell = Ndispersion ÷ ndispersion

    # initialize dispersion energy to zero
    ϵ_disp = zero(T)

    # construct views into relevant arrays corresponding to specified dispersive coupling
    slice = (dispersion_id-1)*Nunitcell+1 : dispersion_id*Nunitcell
    Ω′  = @view  Ω[slice]
    Ω4′ = @view Ω4[slice]
    dtp = @view dispersion_to_phonon[:, slice]

    # iterate over imaginary time slice
    @inbounds for l in axes(x,2)
        # iterate over unit cells
        for u in axes(dtp,2)
            p  = dtp[1,u]
            p′ = dtp[2,u]
            Δx = x[p′,l] - x[p,l]
            # calculate the reduced mass M″ = (M⋅M′)/(M + M′)
            M″ = reduced_mass(M[p′], M[p])
            # calculate dispersive potential energy
            ϵ_disp += M″*(Ω′[u]^2*Δx^2/2 + Ω4′[u]^2*Δx^4/12)
        end
    end

    # normalize measurement
    ϵ_disp /= (Nunitcell * Lτ)

    return ϵ_disp
end