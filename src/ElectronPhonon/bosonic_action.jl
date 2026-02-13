# evaluate the total bosonic action.
function bosonic_action(
    electron_phonon_parameters::ElectronPhononParameters{T,E};
    holstein_correction::Bool = true
) where {T,E}

    (; x, Δτ, phonon_parameters, dispersion_parameters, holstein_parameters_up, holstein_parameters_dn) = electron_phonon_parameters

    # evaluate the contribution to the bosonic action from the bare phonon modes
    Sb = eval_local_phonon_action(x, Δτ, phonon_parameters)

    # evaluate the contribution to the bosonic action from phonon dispersion
    Sb += eval_dispersive_action(x, Δτ, dispersion_parameters, phonon_parameters)

    # evaluate the contribution to the bosonic action from the holstein couplings
    if holstein_correction
        Sb += eval_holstein_action(x, Δτ, holstein_parameters_up, phonon_parameters)
        Sb += eval_holstein_action(x, Δτ, holstein_parameters_dn, phonon_parameters)
    end

    return Sb
end

# evaluate the bosonic action of local phonon model
function eval_local_phonon_action(
    x::Matrix{E}, Δτ::E,
    phonon_parameters::PhononParameters{E}
) where {E<:AbstractFloat}

    Sb = zero(E)

    # evaluate quantum harmonic oscillator action
    Sb += eval_qho_action(x, Δτ, phonon_parameters)

    # evaluate action associated with on-site anharmonic potential term
    Sb += eval_anharmonic_action(x, Δτ, phonon_parameters)

    return Sb
end

# evaluate quantum harmonic oscillator action
function eval_qho_action(
    x::Matrix{E}, Δτ::E,
    phonon_parameters::PhononParameters{E}
) where {E<:AbstractFloat}

    M = phonon_parameters.M::Vector{E}
    Ω = phonon_parameters.Ω::Vector{E}
    Lτ = size(x, 2)

    # initialize bosonic action
    Sb = zero(E)

    # iterate over imaginary time slice
    @inbounds for l in axes(x,2)
        # iterate over phonon modes
        for n in axes(x,1)
            # make sure phonon mass is finite
            if isfinite(M[n])
                # potential energy Δτ⋅M⋅Ω²⋅x²/2
                Sb += Δτ * M[n] * Ω[n]^2/2 * x[n,l]^2
                # kinetic energy Δτ⋅M/2⋅[(x[l]-x[l-1])²/Δτ²]
                Sb += M[n] * (x[n,l]-x[n,mod1(l-1,Lτ)])^2 / (2*Δτ)
            end
        end
    end
    return Sb
end

# evaluate anharmonic potential
function eval_anharmonic_action(
    x::Matrix{E},
    Δτ::E,
    phonon_parameters::PhononParameters{E}
) where {E<:AbstractFloat}

    M = phonon_parameters.M::Vector{E}
    Ω4 = phonon_parameters.Ω4::Vector{E}

    # initialize bosonic action
    Sb = zero(E)

    # iterate over imaginary time slice
    @inbounds for l in axes(x,2)
        # iterate over phonon modes
        for n in axes(x,1)
            # make sure phonon mass is finite
            if isfinite(M[n]) && !iszero(Ω4[n])
                # potential energy Δτ⋅M⋅Ω₄²⋅x⁴/24
                Sb += Δτ * M[n] * Ω4[n]^2/24 * x[n,l]^4
            end
        end
    end

    return Sb
end

# contribution to bosonic action from phonon dispersion potential energy
function eval_dispersive_action(
    x::Matrix{E}, Δτ::E,
    dispersion_parameters::DispersionParameters{E},
    phonon_parameters::PhononParameters{E}
) where {E<:AbstractFloat}

    (; M) = phonon_parameters
    (; Ndispersion, Ω, Ω4, ζ, dispersion_to_phonon) = dispersion_parameters
    Lτ = size(x, 2)

    # initialize bosonic action
    Sb = zero(E)

    if Ndispersion > 0
        # iterate over imaginary time slice
        @fastmath @inbounds for l in 1:Lτ
            # iterate over dispersive phonon coupling
            for n in 1:Ndispersion
                # get the pair of coupled phonon modes
                p  = dispersion_to_phonon[1,n]
                p′ = dispersion_to_phonon[2,n]
                # calculate the reduced mass M″ = (M⋅M′)/(M + M′)
                M″ = reduced_mass(M[p′], M[p])
                # calculate the difference in phonon position
                Δx = x[p′,l] - ζ[n]*x[p,l]
                # calculate the potential energy M″⋅Ω²⋅(xᵢ-xⱼ)² + M″⋅Ω₄²⋅(xᵢ-xⱼ)⁴/12
                Sb += Δτ*M″*(Ω[n]^2*Δx^2 + Ω4[n]^2*Δx^4/12)
            end
        end
    end

    return Sb
end

# contribution to bosonic action of holstein coupling b/c used coupling of form X⋅(n_s-1/2) instead of X⋅n_s
function eval_holstein_action(
    x::Matrix{E},
    Δτ::E,
    holstein_parameters::HolsteinParameters{E},
    phonon_parameters::PhononParameters{E}
) where {E<:AbstractFloat}

    (; M) = phonon_parameters
    (; Nholstein, nholstein, α, α3, ph_sym_form, coupling_to_phonon) = holstein_parameters
    Lτ = size(x,2)

    # initialize bosonic action
    Sb = zero(E)

    if Nholstein > 0
        # number of unit cells
        Nunitcells = Nholstein ÷ nholstein
        # iterate over imaginary time slices
        @fastmath @inbounds for l in 1:Lτ
            # iterate over types of holstein couplings
            for h in 1:nholstein
                # if particle-hole symmetric holstein term
                if ph_sym_form[h]
                    # iterate over unit cells
                    for i in 1:Nunitcells
                        # get the holstein coupling index
                        n = (h-1) * Nunitcells + i
                        # if finite phonon mass
                        if isfinite(M[n])
                            # get the phonon mode associated with the holstein coupling
                            p = coupling_to_phonon[n]
                            # calculate the contribution to the potential energy
                            Sb -= Δτ * (α[n] * x[p,l] + α3[n] * x[p,l]^3)/2
                        end
                    end
                end
            end
        end
    end

    return Sb
end


# evaluate the derivative of the total bosonic action.
# if (holstein_correction = 1) then the correction arising from
# X⋅(n-1) parameterization of the coupling is included.
function bosonic_action_derivative!(
    dSdx::Matrix{E},
    electron_phonon_parameters::ElectronPhononParameters{T,E};
    holstein_correction::Bool = true,
) where {T,E}

    (; x, Δτ, phonon_parameters, dispersion_parameters, holstein_parameters_up, holstein_parameters_dn) = electron_phonon_parameters

    # evaluate the contribution to the bosonic action a single local disperionless phonon mode
    eval_derivative_local_phonon_action!(dSdx, x, Δτ, phonon_parameters)

    # evaluate the contribution to the bosonic action from phonon dispersion
    eval_derivative_dispersive_action!(dSdx, x, Δτ, dispersion_parameters, phonon_parameters)

    # evaluate the contribution to the bosonic action from the holstein couplings
    if holstein_correction
        eval_derivative_holstein_action!(dSdx, x, Δτ, holstein_parameters_up, phonon_parameters)
        eval_derivative_holstein_action!(dSdx, x, Δτ, holstein_parameters_dn, phonon_parameters)
    end

    return nothing
end

# evaluate derivative of local phonon mode
function eval_derivative_local_phonon_action!(
    dSdx::Matrix{E}, x::Matrix{E}, Δτ::E,
    phonon_parameters::PhononParameters{E}
) where {E<:AbstractFloat}

    # evaluate derivative of QHO action
    eval_derivative_qho_action!(dSdx, x, Δτ, phonon_parameters)

    # evaluate derivative of anharmonic contribution to action
    eval_derivative_anharmonic_action!(dSdx, x, Δτ, phonon_parameters)

    return nothing
end

# evaluate derivative of quantum harmonic oscillator action
function eval_derivative_qho_action!(
    dSdx::Matrix{E}, x::Matrix{E}, Δτ::E,
    phonon_parameters::PhononParameters{E}
) where {E<:AbstractFloat}

    (; M, Ω) = phonon_parameters

    # get length of imaginary time axis
    Lτ = size(x, 2)

    # iterate over imaginary time slice
    @inbounds for l in axes(x,2)
        # iterate over phonon modes
        for n in axes(x,1)
            # make sure phonon mass is finite
            if isfinite(M[n])
                # evaluate derivative potential energy
                dSdx[n,l] += Δτ*M[n]*Ω[n]^2*x[n,l]
                # evaluate derivative of kinetic energy
                dSdx[n,l] += M[n]*(2*x[n,l] - x[n,mod1(l+1,Lτ)] - x[n,mod1(l-1,Lτ)])/Δτ
            end
        end
    end

    return nothing
end

# evaluate derivative of anharmonic potential contribution to action
function eval_derivative_anharmonic_action!(
    dSdx::Matrix{E}, x::Matrix{E}, Δτ::E,
    phonon_parameters::PhononParameters{E}
) where {E<:AbstractFloat}

    (; M, Ω4) = phonon_parameters

    # iterate over imaginary time slice
    @inbounds for l in axes(x,2)
        # iterate over phonon modes
        for n in axes(x,1)
            # make sure phonon mass is finite
            if !iszero(Ω4[n]) && isfinite(M[n])
                # evaluate derivative potential energy
                dSdx[n,l] += Δτ*M[n]*Ω4[n]^2/6*x[n,l]^3
            end
        end
    end

    return nothing
end

# evaluate derivative of phonon dispersion potential energy contribution to bosonic action
function eval_derivative_dispersive_action!(
    dSdx::Matrix{E}, x::Matrix{E}, Δτ::E,
    dispersion_parameters::DispersionParameters{E},
    phonon_parameters::PhononParameters{E}
) where {E<:AbstractFloat}

    (; M) = phonon_parameters
    (; Ndispersion, Ω, Ω4, ζ, dispersion_to_phonon) = dispersion_parameters
    Lτ = size(x, 2)

    # iterate over imaginary time slice
    @inbounds for l in 1:Lτ
        # iterate over dispersive couplings
        for n in 1:Ndispersion
            # get the pair of coupled phonon modes
            p  = dispersion_to_phonon[1,n]
            p′ = dispersion_to_phonon[2,n]
            # calculate the reduced mass M″ = (M⋅M′)/(M + M′)
            M″ = reduced_mass(M[p′], M[p])
            # calculate the difference in phonon position
            Δx = x[p′,l] - ζ[n]*x[p,l]
            # evaluate derivative with respect to first phonon field
            if isfinite(M[p])
                dSdx[p,l] -= ζ[n]*Δτ*M″*(2*Ω[n]^2*Δx + Ω4[n]^2*Δx^3/3)
            end
            # evaluate derivative with respect to second phonon field
            if isfinite(M[p′])
                dSdx[p′,l] += Δτ*M″*(2*Ω[n]^2*Δx + Ω4[n]^2*Δx^3/3)
            end
        end
    end

    return nothing
end

# contribution to bosonic action of holstein coupling b/c used coupling of form X⋅(n-1) instead of X⋅n
function eval_derivative_holstein_action!(
    dSdx::Matrix{E}, x::Matrix{E}, Δτ::E,
    holstein_parameters::HolsteinParameters{E},
    phonon_parameters::PhononParameters{E}
) where {E<:AbstractFloat}

    (; M) = phonon_parameters
    (; Nholstein, nholstein, α, α3, coupling_to_phonon, ph_sym_form) = holstein_parameters
    Lτ = size(x,2)

    # check if there are holstein couplings
    if nholstein > 0
        # iterate over imaginary time slices
        @fastmath @inbounds for l in 1:Lτ
            # number of unit cells
            Nunitcells = Nholstein ÷ nholstein
            # iterate over types of holstein couplings
            for h in 1:nholstein
                # if particle-hole symmetric holstein term
                if ph_sym_form[h]
                    # iterate over unit cells
                    for i in 1:Nunitcells
                        # get the holstein coupling index
                        n = (h-1) * Nunitcells + i
                        # if phonon mass is finite
                        if isfinite(M[n])
                            # get the phonon mode associated with the holstein coupling
                            p = coupling_to_phonon[n]
                            # calculate the contribution to the potential energy
                            dSdx[p,l] -= Δτ * (α[n] + 3 * α3[n] * x[p,l]^2)/2
                        end
                    end
                end
            end
        end
    end

    return nothing
end