function radial_update!(
    # ARGUMENTS
    Gup::Matrix{T}, logdetGup::E, sgndetGup::T,
    Gdn::Matrix{T}, logdetGdn::E, sgndetGdn::T,
    electron_phonon_parameters::ElectronPhononParameters{T,E};
    # KEYWORD ARGUMENTS
    fermion_path_integral_up::FermionPathIntegral{T,E},
    fermion_path_integral_dn::FermionPathIntegral{T,E},
    fermion_greens_calculator_up::FermionGreensCalculator{T,E},
    fermion_greens_calculator_dn::FermionGreensCalculator{T,E},
    fermion_greens_calculator_up_alt::FermionGreensCalculator{T,E},
    fermion_greens_calculator_dn_alt::FermionGreensCalculator{T,E},
    Bup::Vector{P}, Bdn::Vector{P}, rng::AbstractRNG,
    phonon_types = nothing
) where {T<:Number, E<:AbstractFloat, P<:AbstractPropagator{T,E}}

    Gup′ = fermion_greens_calculator_up_alt.G′
    Gdn′ = fermion_greens_calculator_dn_alt.G′
    phonon_parameters = electron_phonon_parameters.phonon_parameters::PhononParameters{E}
    holstein_parameters_up = electron_phonon_parameters.holstein_parameters_up::HolsteinParameters{E}
    holstein_parameters_dn = electron_phonon_parameters.holstein_parameters_dn::HolsteinParameters{E}
    ssh_parameters_up = electron_phonon_parameters.ssh_parameters_up::SSHParameters{T}
    ssh_parameters_dn = electron_phonon_parameters.ssh_parameters_dn::SSHParameters{T}
    x = electron_phonon_parameters.x
    M = phonon_parameters.M

    return (accepted, logdetGup, sgndetGup, logdetGdn, sgndetGdn)
end


function radial_update!(
    # ARGUMENTS
    G::Matrix{T}, logdetG::E, sgndetG::T,
    electron_phonon_parameters::ElectronPhononParameters{T,E};
    # KEYWORD ARGUMENTS
    fermion_path_integral::FermionPathIntegral{T,E},
    fermion_greens_calculator::FermionGreensCalculator{T,E},
    fermion_greens_calculator_alt::FermionGreensCalculator{T,E},
    B::Vector{P}, rng::AbstractRNG,
    σ::E = 1.0,
    phonon_id::Union{Int, Nothing}
) where {T<:Number, E<:AbstractFloat, P<:AbstractPropagator{T,E}}

    G′ = fermion_greens_calculator_alt.G′
    phonon_parameters = electron_phonon_parameters.phonon_parameters::PhononParameters{E}
    holstein_parameters = electron_phonon_parameters.holstein_parameters_up::HolsteinParameters{E}
    ssh_parameters = electron_phonon_parameters.ssh_parameters_up::SSHParameters{T}
    x = electron_phonon_parameters.x
    M = phonon_parameters.M

    # make sure stabilization frequencies match
    if fermion_greens_calculator.n_stab != fermion_greens_calculator_alt.n_stab
        resize!(fermion_greens_calculator_alt, fermion_greens_calculator.n_stab)
    end

    # length of imaginary time axis
    Lτ = electron_phonon_parameters.Lτ

    # get the number of phonon modes per unit cell
    nphonon = phonon_parameters.nphonon

    # total number of phonon modes
    Nphonon = phonon_parameters.Nphonon

    # number of unit cells
    Nunitcells = Nphonon ÷ nphonon

    # sample random phonon mode
    phonon_mode = _sample_phonon_mode(rng, nphonon, Nunitcells, M, phonon_types)

    # whether the exponentiated on-site energy matrix needs to be updated with the phonon field,
    # true if phonon mode appears in holstein coupling
    calculate_exp_V = (phonon_mode in holstein_parameters.coupling_to_phonon)

    # whether the exponentiated hopping matrix needs to be updated with the phonon field,
    # true if phonon mode appears in SSH coupling
    calculate_exp_K = (phonon_mode in ssh_parameters.coupling_to_phonon)

    # get phonon fields and mass for specified phonon mode if necessary
    if !isnothing(phonon_id)
        M′ = @view M[(phonon_id-1)*Nunitcells+1:phonon_id*Nunitcells]
        x′ = @view x[(phonon_id-1)*Nunitcells+1:phonon_id*Nunitcells, :]
    else
        M′ = M
        x′ = x
    end

    # number of fields to update, excluding phonon fields that correspond
    # to phonon modes with infinite mass
    d = count(m -> isfinite(m), M′)

    # calculate standard deviation for normal distribution
    σ = σ / sqrt(d)

    # randomly sample expansion/contraction coefficient
    γ = randn(rng) * σ
    expγ = exp(γ)

    # calculate initial bosonic action
    Sb = bosonic_action(electron_phonon_parameters)

    # calculate initial fermionic action
    Sf = 2*logdetG

    # calculate initial total action
    S = Sb + Sf

    # substract off the effect of the current phonon configuration on the fermion path integrals
    if calculate_exp_V
        update!(fermion_path_integral, holstein_parameters, x, -1)
    end
    if calculate_exp_K
        update!(fermion_path_integral, ssh_parameters, x, -1)
    end

    # apply expansion/contraction to phonon fields
    @. x′ = expγ * x′

    # update the fermion path integrals to reflect new phonon field configuration
    if calculate_exp_V
        update!(fermion_path_integral, holstein_parameters, x, +1)
    end
    if calculate_exp_K
        update!(fermion_path_integral, ssh_parameters, x, +1)
    end

    # update the spin up and spin down propagators to reflect current phonon configuration
    calculate_propagators!(B, fermion_path_integral, calculate_exp_K = calculate_exp_K, calculate_exp_V = calculate_exp_V)

    # update the Green's function to reflect the new phonon configuration
    logdetG′, sgndetG′ = calculate_equaltime_greens!(G′, fermion_greens_calculator_alt, B)

    # calculate the final bosonic action
    Sb′ = bosonic_action(electron_phonon_parameters)

    # calculate final fermionci action
    Sf′ = 2*logdetG′

    # calculate final action
    S′ = Sb′ + Sf′

    # calculate the change in action
    ΔS = S′ - S

    # calculate acceptance probability
    if isfinite(logdetG′)
        P_γ = min(1.0, exp(-ΔS + d*γ))
    else
        P_γ = 0.0
    end

    # accept or reject the update
    if rand(rng) < P_γ
        logdetG = logdetG′
        sgndetG = sgndetG′
        copyto!(G, G′)
        copyto!(fermion_greens_calculator, fermion_greens_calculator_alt)
        accepted = true
    else
        # substract off the effect of the current phonon configuration on the fermion path integrals
        if calculate_exp_V
            update!(fermion_path_integral, holstein_parameters, x, -1)
        end
        if calculate_exp_K
            update!(fermion_path_integral, ssh_parameters, x, -1)
        end
        # revert to the original phonon configuration
        @. x′ = x′ / expγ
        # update the fermion path integrals to reflect new phonon field configuration
        if calculate_exp_V
            update!(fermion_path_integral, holstein_parameters, x, +1)
        end
        if calculate_exp_K
            update!(fermion_path_integral, ssh_parameters, x, +1)
        end
        # update the fermion path integrals to reflect the original phonon configuration
        calculate_propagators!(B, fermion_path_integral, calculate_exp_K = calculate_exp_K, calculate_exp_V = calculate_exp_V)
        accepted = false
    end

    return (accepted, logdetG, sgndetG)
end