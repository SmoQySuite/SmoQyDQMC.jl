####################################
## MEASURE SSH INTERACTION ENERGY ##
####################################

@doc raw"""
    measure_ssh_energy(
        electron_phonon_parameters::ElectronPhononParameters{T,E},
        Gup::Matrix{T}, Gdn::Matrix{T},
        ssh_id::Int
    ) where {T<:Number, E<:AbstractFloat}

Calculate the return the SSH interaction energy
```math
\epsilon_{\rm ssh} = \sum_\sigma \left\langle [\alpha \hat{X}     + \alpha_2 \hat{X}^2
                                               \alpha_3 \hat{X}^3 + \alpha_4 \hat{X}^4]
                                              (\hat{c}^\dagger_{\sigma,i} \hat{c}_{\sigma,j} + {\rm h.c.}) \right\rangle
```
for coupling definition specified by `ssh_id`.
"""
function measure_ssh_energy(electron_phonon_parameters::ElectronPhononParameters{T,E},
                            Gup::Matrix{T}, Gdn::Matrix{T},
                            ssh_id::Int) where {T<:Number, E<:AbstractFloat}

    x = electron_phonon_parameters.x::Matrix{E}
    ssh_parameters_up = electron_phonon_parameters.ssh_parameters_up::SSHParameters{T}
    ϵ_ssh_up = measure_ssh_energy(ssh_parameters_up, Gup, x, ssh_id)
    ssh_parameters_up = electron_phonon_parameters.ssh_parameters_dn::SSHParameters{T}
    ϵ_ssh_dn = measure_ssh_energy(ssh_parameters_dn, Gdn, x, ssh_id)
    ϵ_ssh = ϵ_ssh_up + ϵ_ssh_dn

    return ϵ_ssh, ϵ_ssh_up, ϵ_ssh_dn
end

@doc raw"""
    measure_ssh_energy(
        ssh_parameters::SSHParameters{T},
        G::Matrix{T}, x::Matrix{E}, ssh_id::Int
    ) where {T<:Number, E<:AbstractFloat}

Calculate the return the SSH interaction energy
```math
\epsilon_{\rm ssh} = \left\langle [\alpha \hat{X}     + \alpha_2 \hat{X}^2
                                   \alpha_3 \hat{X}^3 + \alpha_4 \hat{X}^4]
                        (\hat{c}^\dagger_{\sigma,i} \hat{c}_{\sigma,j} + {\rm h.c.}) \right\rangle
```
for coupling definition specified by `ssh_id`.
"""
function measure_ssh_energy(
    ssh_parameters::SSHParameters{T},
    G::Matrix{T}, x::Matrix{E}, ssh_id::Int
) where {T<:Number, E<:AbstractFloat}

    (; nssh, Nssh, α, α2, α3, α4, neighbor_table, coupling_to_phonon) = ssh_parameters

    # length of imaginary time axis
    Lτ = size(x,2)

    # initialize ssh energy to zero
    ϵ_ssh = zero(T)

    # number of unit cells in lattice
    Nunitcell = Nssh ÷ nssh

    # get relevant views into arrays corresponding to ssh coupling id
    slice = (ssh_id-1)*Nunitcell+1:ssh_id*Nunitcell
    α′  = @view  α[slice]
    α2′ = @view α2[slice]
    α3′ = @view α3[slice]
    α4′ = @view α4[slice]
    nt  = @view neighbor_table[:,slice]
    ctp = @view coupling_to_phonon[:,slice]

    # iterate over unit cells
    for u in axes(ctp, 2)
        p  = ctp[1,u]
        p′ = ctp[2,u]
        Δx = x[p′,Lτ] - x[p,Lτ]
        j  = nt[1,u]
        i  = nt[2,u]
        hij = -G[j,i]
        ϵij = α′[u]*Δx + α2′[u]*Δx^2 + α3′[u]*Δx^3 + α4′[u]*Δx^4
        ϵ_ssh += ϵij*hij + conj(ϵij*hij)
    end

    # normalize measurement
    ϵ_ssh /= (Nunitcell)

    return ϵ_ssh
end


###############################################################
## MEASURE THE FREQUENCY WITH WHICH THE HOPPING CHANGES SIGN ##
###############################################################

@doc raw"""
    measure_ssh_sgn_switch(
        electron_phonon_parameters::ElectronPhononParameters{T,E},
        tight_binding_parameters::TightBindingParameters{T,E},
        ssh_id::Int;
        spin::Int = +1
    ) where {T<:Number, E<:AbstractFloat}

Calculate the fraction of the time the sign of the hopping is changed as a result of the
SSH coupling associated with `ssh_id`.
"""
function measure_ssh_sgn_switch(
    electron_phonon_parameters::ElectronPhononParameters{T,E},
    tight_binding_parameters::TightBindingParameters{T,E},
    ssh_id::Int;
    spin::Int = +1
) where {T<:Number, E<:AbstractFloat}

    x = electron_phonon_parameters.x::Matrix{E}
    if isone(spin)
        ssh_parameters = electron_phonon_parameters.ssh_parameters_up::SSHParameters{T}
    else
        ssh_parameters = electron_phonon_parameters.ssh_parameters_dn::SSHParameters{T}
    end
    sgn = measure_ssh_sgn_switch(ssh_parameters, tight_binding_parameters, x, ssh_id)

    return sgn
end

function measure_ssh_sgn_switch(
    ssh_parameters::SSHParameters{T},
    tight_binding_parameters::TightBindingParameters{T,E},
    x::Matrix{E}, ssh_id::Int
) where {T<:Number, E<:AbstractFloat}

    (; t) = tight_binding_parameters
    (; nssh, Nssh, α, α2, α3, α4, coupling_to_phonon, coupling_to_hopping) = ssh_parameters

    # length of imaginary time axis
    Lτ = size(x,2)

    # initialize frequency of hopping sign switch to zero
    sgn = zero(E)

    # number of unit cells in lattice
    Nunitcell = Nssh ÷ nssh

    # get relevant views into arrays corresponding to ssh coupling id
    slice = (ssh_id-1)*Nunitcell+1:ssh_id*Nunitcell
    α′  = @view  α[slice]
    α2′ = @view α2[slice]
    α3′ = @view α3[slice]
    α4′ = @view α4[slice]
    ctp = @view coupling_to_phonon[:,slice]
    cth = @view coupling_to_hopping[slice]

    # iterate over imaginary time slice
    for l in axes(x, 2)
        # iterate over unit cells
        for u in axes(ctp, 2)
            p  = ctp[1,u]
            p′ = ctp[2,u]
            Δx = x[p′,l] - x[p,l]
            # get the ssh coupling associated with the current unit cell
            c = slice[u]
            # get hopping associated with ssh coupling
            h = cth[u]
            # calculate effective hopping
            t′ = t[h] - (α′[u]*Δx + α2′[u]*Δx^2 + α3′[u]*Δx^3 + α4′[u]*Δx^4)
            # check if sign of effective hopping is different from bare hopping
            sgn += !(sign(t′) ≈ sign(t[h]))
        end
    end

    # normalize measurement
    sgn /= (Nunitcell * Lτ)

    return sgn
end