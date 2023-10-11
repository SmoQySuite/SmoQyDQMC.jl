@doc raw"""
    bosonic_action(electron_phonon_parameters::ElectronPhononParameters{T,E}) where {T,E}

Evaluate and return bosonic action ``S_{\rm b}(x)`` associated with the phonons.
"""
function bosonic_action(electron_phonon_parameters::ElectronPhononParameters{T,E}) where {T,E}

    (; x, Δτ, phonon_parameters, dispersion_parameters, holstein_parameters) = electron_phonon_parameters

    # evaluate the contribution to the bosonic action from the bare phonon modes
    Sb = _eval_bosonic_action(x, Δτ, phonon_parameters)

    # evaluate the contribution to the bosonic action from phonon dispersion
    Sb += _eval_bosonic_action(x, Δτ, dispersion_parameters, phonon_parameters)

    # evaluate the contribution to the bosonic action from the holstein couplings
    Sb += _eval_bosonic_action(x, Δτ, holstein_parameters)

    return Sb
end

# evaluate bare phonon mode contribuation to bosonic action (potential and kinetic energy contribution)
function _eval_bosonic_action(x::Matrix{E}, Δτ::E, phonon_parameters::PhononParameters{E}) where {E<:AbstractFloat}

    M = phonon_parameters.M::Vector{E}
    Ω = phonon_parameters.Ω::Vector{E}
    Ω4 = phonon_parameters.Ω4::Vector{E}
    Lτ = size(x, 2)

    # initialize bosonic action
    Sb = zero(E)

    # iterate over imaginary time slice
    @inbounds for l in axes(x,2)
        # iterate over phonon modes
        for n in axes(x,1)
            # make sure phonon mass is finite
            if isfinite(M[n])
                # potential energy Δτ⋅M⋅Ω²⋅x²/2 + Δτ⋅M⋅Ω₄²⋅x⁴/24
                Sb += Δτ*M[n] * (Ω[n]^2/2*x[n,l]^2 + Ω4[n]^2/24*x[n,l]^4)
                # kintetic energy Δτ⋅M/2⋅[(x[l]-x[l-1])²/Δτ²]
                Sb += M[n] * (x[n,l]-x[n,mod1(l-1,Lτ)])^2 / (2*Δτ)
            end
        end
    end
    return Sb
end

# contribution to bosonic action from phonon dispersion potential energy
function _eval_bosonic_action(x::Matrix{E}, Δτ::E, dispersion_parameters::DispersionParameters{E},
                             phonon_parameters::PhononParameters{E}) where {E<:AbstractFloat}

    (; M) = phonon_parameters
    (; Ndispersion, Ω, Ω4, dispersion_to_phonon) = dispersion_parameters
    Lτ = size(x, 2)

    # initialize bosonic action
    Sb = zero(E)

    if Ndispersion > 0
        # iterate over imaginary time slice
        @fastmath @inbounds for l in 1:Lτ
            # iterarte over dispersive phonon coupling
            for n in 1:Ndispersion
                # get the pair of coupled phonon modes
                p  = dispersion_to_phonon[1,n]
                p′ = dispersion_to_phonon[2,n]
                # calculate the reduced mass M″ = (M⋅M′)/(M + M′)
                M″ = reduced_mass(M[p′], M[p])
                # calculate the difference in phonon position
                Δx = x[p′,l] - x[p,l]
                # calculate the potential energy M″⋅Ω²⋅(xᵢ-xⱼ)² + M″⋅Ω₄²⋅(xᵢ-xⱼ)⁴/12
                Sb += Δτ*M″*(Ω[n]^2*Δx^2 + Ω4[n]^2*Δx^4/12)
            end
        end
    end

    return Sb
end

# contribution to bosonic action of holstein coupling b/c used coupling of form X⋅(n-1) instead of X⋅n
function _eval_bosonic_action(x::Matrix{E}, Δτ::E, holstein_parameters::HolsteinParameters{E}) where {E<:AbstractFloat}

    (; Nholstein, α, α2, α3, α4, coupling_to_phonon) = holstein_parameters
    Lτ = size(x,2)

    # initialize bosonic action
    Sb = zero(E)

    if Nholstein > 0
        # iterate over imaginary time slices
        @fastmath @inbounds for l in 1:Lτ
            # iterate over holstein couplings
            for n in 1:Nholstein
                # get the phonon mode associated with the holstein coupling
                p = coupling_to_phonon[n]
                # calculate the contribution to the potential energy
                Sb -= Δτ * (α[n]*x[p,l] + α2[n]*x[p,l]^2 + α3[n]*x[p,l]^3 + α4[n]*x[p,l]^4)
            end
        end
    end

    return Sb
end


@doc raw"""
    bosonic_action_derivative!(dSdx::Matrix{T}, electron_phonon_parameters::ElectronPhononParameters{T,E}) where {T,E}

Evaluate the derivative of the bosonic action ``\frac{\partial S_{\rm b}}{\parital x_{i,l}}`` with respect to each
phonon field ``x_{i,l},`` adding the result to the array `dSdx`.
"""
function bosonic_action_derivative!(dSdx::Matrix{E}, electron_phonon_parameters::ElectronPhononParameters{T,E}) where {T,E}

    (; x, Δτ, phonon_parameters, dispersion_parameters, holstein_parameters) = electron_phonon_parameters

    # evaluate the contribution to the bosonic action from the bare phonon modes
    _eval_derivative_bosonic_action(dSdx, x, Δτ, phonon_parameters)

    # evaluate the contribution to the bosonic action from phonon dispersion
    _eval_derivative_bosonic_action(dSdx, x, Δτ, dispersion_parameters, phonon_parameters)

    # evaluate the contribution to the bosonic action from the holstein couplings
    _eval_derivative_bosonic_action(dSdx, x, Δτ, holstein_parameters)

    return nothing
end

# evaluate derivative of bare phonon contribution to bosonic action
function _eval_derivative_bosonic_action(dSdx::Matrix{E}, x::Matrix{E}, Δτ::E,
                                         phonon_parameters::PhononParameters{E}) where {E<:AbstractFloat}

    (; M, Ω, Ω4) = phonon_parameters

    # get length of imaginary time axis
    Lτ = size(x, 2)

    # iterate over imaginary time slice
    @inbounds for l in axes(x,2)
        # iterate over phonon modes
        for n in axes(x,1)
            # make sure phonon mass is finite
            if isfinite(M[n])
                # evaluate derivative potential energy
                dSdx[n,l] += Δτ*M[n]*(Ω[n]^2*x[n,l] + Ω4[n]^2/6*x[n,l]^3)
                # evaluate derivative of kinetic energy
                dSdx[n,l] += M[n]*(2*x[n,l] - x[n,mod1(l+1,Lτ)] - x[n,mod1(l-1,Lτ)])/Δτ
            end
        end
    end

    return nothing
end

# evaluate derivative of phonon dispersion potential energy contribution to bosonic action
function _eval_derivative_bosonic_action(dSdx::Matrix{E}, x::Matrix{E}, Δτ::E, dispersion_parameters::DispersionParameters{E},
                                         phonon_parameters::PhononParameters{E}) where {E<:AbstractFloat}

    (; M) = phonon_parameters
    (; Ndispersion, Ω, Ω4, dispersion_to_phonon) = dispersion_parameters
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
            Δx = x[p′,l] - x[p,l]
            # evaluate derivative with respect to first phonon field
            if isfinite(M[p])
                dSdx[p,l] -= Δτ*M″*(2*Ω[n]^2*Δx + Ω4[n]^2*Δx^3/3)
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
function _eval_derivative_bosonic_action(dSdx::Matrix{E}, x::Matrix{E}, Δτ::E,
                                         holstein_parameters::HolsteinParameters{E}) where {E<:AbstractFloat}

    (; Nholstein, α, α2, α3, α4, coupling_to_phonon) = holstein_parameters
    Lτ = size(x,2)

    # iterate over imaginary time slices
    @fastmath @inbounds for l in 1:Lτ
        # iterate over holstein couplings
        for n in 1:Nholstein
            # get the phonon mode associated with the holstein coupling
            p = coupling_to_phonon[n]
            # calculate the contribution to the potential energy
            dSdx[p,l] -= Δτ * (α[n] + 2 * α2[n] * x[p,l] + 3 * α3[n] * x[p,l]^2 + 4 * α4[n] * x[p,l]^3)
        end
    end

    return nothing
end