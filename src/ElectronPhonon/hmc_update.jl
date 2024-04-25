# performs a hybrid/hamiltonian monte carlo update to the phonon configuration
function _hmc_update!(Gup::Matrix{T}, logdetGup::E, sgndetGup::T, Gup′::Matrix{T},
                      Gdn::Matrix{T}, logdetGdn::E, sgndetGdn::T, Gdn′::Matrix{T},
                      electron_phonon_parameters::ElectronPhononParameters{T,E},
                      fermion_path_integral_up::FermionPathIntegral{T,E},
                      fermion_path_integral_dn::FermionPathIntegral{T,E},
                      fermion_greens_calculator_up::FermionGreensCalculator{T,E},
                      fermion_greens_calculator_dn::FermionGreensCalculator{T,E},
                      fermion_greens_calculator_up_alt::FermionGreensCalculator{T,E},
                      fermion_greens_calculator_dn_alt::FermionGreensCalculator{T,E},
                      Bup::Vector{P}, Bdn::Vector{P},
                      dSdx::Matrix{E}, dSdx′::Matrix{E},
                      v::Matrix{E}, x′::Matrix{E}, x0::Matrix{E},
                      fourier_mass_matrix::FourierMassMatrix{E},
                      Nt::Int, nt::Int, Δt::E, initialize_force::Bool, first_update::Bool,
                      δG_max::E, δG::E, δθ::E, rng::AbstractRNG,
                      δG_reject::E = sqrt(δG_max),
                      recenter!::Function = identity,
                      update_stabilization_frequency::Bool = true) where {T, E, P<:AbstractPropagator{T,E}}

    (; β, Lτ, Δτ) = electron_phonon_parameters
    holstein_parameters_up = electron_phonon_parameters.holstein_parameters_up::HolsteinParameters{E}
    holstein_parameters_dn = electron_phonon_parameters.holstein_parameters_dn::HolsteinParameters{E}
    ssh_parameters_up = electron_phonon_parameters.ssh_parameters_up::SSHParameters{T}
    ssh_parameters_dn = electron_phonon_parameters.ssh_parameters_dn::SSHParameters{T}
    phonon_parameters = electron_phonon_parameters.phonon_parameters::PhononParameters{E}

    # whether the exponentiated on-site energy matrix needs to be updated with the phonon field,
    # true if there is a non-zero number of holstein couplings Nholstein
    calculate_exp_V = (holstein_parameters_up.Nholstein > 0)

    # whether the exponentiated hopping matrix needs to be updated with the phonon field,
    # true if there is a non-zero number of ssh couplings Nssh
    calculate_exp_K = (ssh_parameters_up.Nssh > 0)

    # get the phonon fields
    x = electron_phonon_parameters.x

    # calculate effective bosonic time-step
    Δt′ = Δt/nt

    # record initial phonon configuration
    copyto!(x0, x)
    copyto!(x′, x)

    # calculate initial derivative of fermionic action ∂S̃f/∂x = M⁻¹⋅∂Sf/∂x
    if initialize_force || first_update

        # set derivative of action to zero initially
        fill!(dSdx, 0.0)

        # calculate derivative of fermionic action for spin up
        (logdetGup, sgndetGup, δG, δθ) = fermionic_action_derivative!(dSdx, Gup, logdetGup, sgndetGup, δG, δθ,
                                                                      electron_phonon_parameters,
                                                                      fermion_greens_calculator_up, Bup, spin = +1)

        # calculate derivative of fermionic action for spin down
        (logdetGdn, sgndetGdn, δG, δθ) = fermionic_action_derivative!(dSdx, Gdn, logdetGdn, sgndetGdn, δG, δθ,
                                                                      electron_phonon_parameters,
                                                                      fermion_greens_calculator_dn, Bdn, spin = -1)

        # calculate derivative associated with Holstein correction
        eval_derivative_holstein_action!(dSdx, x, Δτ, holstein_parameters_up, phonon_parameters)
        eval_derivative_holstein_action!(dSdx, x, Δτ, holstein_parameters_dn, phonon_parameters)

        # calculate ∂S̃f/∂x = M⁻¹⋅∂Sf/∂x
        lmul!(fourier_mass_matrix, dSdx, -1.0)
    end

    # initialize fermion green's function matrices and their determinants determinants
    copyto!(Gup′, Gup)
    copyto!(Gdn′, Gdn)
    logdetGup′ = logdetGup
    sgndetGup′ = sgndetGup
    logdetGdn′ = logdetGdn
    sgndetGdn′ = sgndetGdn
    
    # initialize the alternate fermion greens calculators
    copyto!(fermion_greens_calculator_up_alt, fermion_greens_calculator_up)
    copyto!(fermion_greens_calculator_dn_alt, fermion_greens_calculator_dn)

    # initialize the momentum as v = R/√(M)
    randn!(rng, v)
    lmul!(fourier_mass_matrix, v, -0.5)

    # calculate the initial kinetic energy
    K = velocity_to_kinetic_energy(fourier_mass_matrix, v)

    # calculate initial bosonic action
    Sb = bosonic_action(electron_phonon_parameters)

    # calculate initial fermionic action
    Sf = logdetGup + logdetGdn

    # record initial total action
    S = Sb + Sf

    # calculate the initial total energy
    H = S + K

    # record the initial derivative of the fermionic action
    copyto!(dSdx′, dSdx)

    # flag to indicate whether or not hmc trajectory remains numerically stable
    numerically_stable = true

    # initialize numerical error for hmc trajectory
    δG′ = δG

    # iterate of fermionic time-steps
    for t in 1:Nt

        # calculate v(t+Δt/2) = v(t) - Δt/2⋅∂S̃f/∂x(t) = v(t) - Δt/2⋅M⁻¹⋅dSf/dx(t)
        @. v = v - Δt/2 * dSdx

        # initialize derivative of bosonic action ∂Sb/∂x
        fill!(dSdx, 0.0)
        bosonic_action_derivative!(dSdx, electron_phonon_parameters, holstein_correction=false)

        # calculate ∂S̃b/∂x = M⁻¹⋅∂Sb/∂x
        lmul!(fourier_mass_matrix, dSdx, -1.0)

        # iterate over bosonic time-steps
        for t′ in 1:nt

            # calculate v(t+Δt′/2) = v(t) - Δt′/2⋅∂S̃b/∂x(t) = v(t) - Δt′/2⋅M⁻¹⋅dSb/dx(t)
            @. v = v - Δt′/2 * dSdx

            # calculate x(t+Δt′) = x(t) + Δt′⋅v(t+Δt′/2)
            @. x = x + Δt′*v

            # re-center phonon fields assuming translationally invariant hamiltonian.
            # this typically does nothing, and is just the identity operator.
            recenter!(x)

            # calculate derivative of bosonic action ∂Sb/∂x
            fill!(dSdx, 0.0)
            bosonic_action_derivative!(dSdx, electron_phonon_parameters, holstein_correction=false)

            # calculate ∂S̃b/∂x = M⁻¹⋅∂Sb/∂x
            lmul!(fourier_mass_matrix, dSdx, -1.0)

            # calculate v(t+Δt′) = v(t+Δt′/2) - Δt′/2⋅∂S̃b/∂x(t+Δt′) = v(t+Δt′) - Δt′/2⋅M⁻¹⋅dSb/dx(t+Δt′)
            @. v = v - Δt′/2 * dSdx
        end

        # update the fermion path integrals for spin up and down sectors to reflect current phonon configuration
        update!(fermion_path_integral_up, fermion_path_integral_dn, electron_phonon_parameters, x, x′)
        
        # record the current phonon configuration
        copyto!(x′, x)

        # update the spin up and spin down propagators to reflect current phonon configuration
        calculate_propagators!(Bup, fermion_path_integral_up,
                               calculate_exp_K = calculate_exp_K, calculate_exp_V = calculate_exp_V)
        calculate_propagators!(Bdn, fermion_path_integral_dn,
                               calculate_exp_K = calculate_exp_K, calculate_exp_V = calculate_exp_V)

        # update the Green's function to reflect the new phonon configuration
        try
            logdetGup′, sgndetGup′ = calculate_equaltime_greens!(Gup′, fermion_greens_calculator_up_alt, Bup)
            logdetGdn′, sgndetGdn′ = calculate_equaltime_greens!(Gdn′, fermion_greens_calculator_dn_alt, Bdn)
        catch
            numerically_stable = false
            break
        end

        # if nan occurs terminate update
        if !isfinite(logdetGup′) || !isfinite(sgndetGdn′)

            # record that numerically instability was encountered
            numerically_stable = false

            # terminate the HMC trajectory
            break
        end

        # initialize derivative of action to zero
        fill!(dSdx, 0.0)

        try
            # calculate derivative of fermionic action for spin up
            (logdetGup′, sgndetGup′, δGup′, δθ) = fermionic_action_derivative!(dSdx, Gup′, logdetGup′, sgndetGup′, δG′, δθ,
                                                                            electron_phonon_parameters,
                                                                            fermion_greens_calculator_up_alt, Bup, spin = +1)

            # calculate derivative of fermionic action for spin down
            (logdetGdn′, sgndetGdn′, δGdn′, δθ) = fermionic_action_derivative!(dSdx, Gdn′, logdetGdn′, sgndetGdn′, δG′, δθ,
                                                                            electron_phonon_parameters,
                                                                            fermion_greens_calculator_dn_alt, Bdn, spin = -1)

            # record max numerical error
            δG′ = max(δG′, δGup′, δGdn′)
        catch
            numerically_stable = false
            break
        end

        # calculate derivative associated with Holstein correction
        eval_derivative_holstein_action!(dSdx, x, Δτ, holstein_parameters_up, phonon_parameters)
        eval_derivative_holstein_action!(dSdx, x, Δτ, holstein_parameters_dn, phonon_parameters)

        # if numerical error too large or nan occurs
        if !isfinite(δG′) || !isfinite(logdetGup) || !isfinite(logdetGdn) || δG′ > δG_reject

            # record that numerically instability was encountered
            numerically_stable = false

            # terminate the HMC trajectory
            break
        end

        # calculate ∂S̃f/∂x = M⁻¹⋅∂Sf/∂x
        lmul!(fourier_mass_matrix, dSdx, -1.0)

        # calculate v(t+Δt) = v(t+Δt/2) - Δt/2⋅∂S̃f/∂x(t+Δt) = v(t+Δt/2) - Δt/2⋅M⁻¹⋅dSf/dx(t+Δt)
        @. v = v - Δt/2 * dSdx
    end

    # if numerically stable
    if numerically_stable

        # calculate the final kinetic energy
        K′ = velocity_to_kinetic_energy(fourier_mass_matrix, v)

        # calculate final bosonic action
        Sb′ = bosonic_action(electron_phonon_parameters)

        # calculate final fermionic action
        Sf′ = logdetGup′ + logdetGdn′

        # record final total action
        S′ = Sb′ + Sf′

        # calculate the final total energy
        H′ = S′ + K′

        # calculate the change in energy
        ΔH = H′ - H

        # calculate the probability of accepting the final configuration
        p = min(1.0, exp(-ΔH))

    # if numerically unstable
    else

        # set acceptance probability to zero
        p = zero(E)
    end

    # determine if update accepted
    accepted = rand(rng) < p

    # if proposed phonon configuration is accepted and hmc trajecotry remained numerically stable
    if accepted
        # record final green function matrices
        copyto!(Gup, Gup′)
        copyto!(Gdn, Gdn′)
        # record final green function determinant
        logdetGup = logdetGup′
        sgndetGup = sgndetGup′
        logdetGdn = logdetGdn′
        sgndetGdn = sgndetGdn′
        # record the final state of the fermion greens calculator
        copyto!(fermion_greens_calculator_up, fermion_greens_calculator_up_alt)
        copyto!(fermion_greens_calculator_dn, fermion_greens_calculator_dn_alt)
        # record numerical error associated up hmc update
        δG = δG′
        # update stabilization frequency if required
        if update_stabilization_frequency
            (updated, logdetGup, sgndetGup, logdetGdn, sgndetGdn, δG, δθ) = update_stabalization_frequency!(
                Gup, logdetGup, sgndetGup,
                Gdn, logdetGdn, sgndetGdn,
                fermion_greens_calculator_up = fermion_greens_calculator_up,
                fermion_greens_calculator_dn = fermion_greens_calculator_dn,
                Bup = Bup, Bdn = Bdn, δG = δG, δθ = δθ, δG_max = δG_max
            )
        end
    # otherwise reject proposed update and revert to oringial phonon configuration
    else
        # update fermion path integrals to reflect initial phonon configuration
        update!(fermion_path_integral_up, fermion_path_integral_dn, electron_phonon_parameters, x0, x)
        # revert to initial phonon configuration
        copyto!(x, x0)
        # update propagators to reflect initial phonon configuration
        calculate_propagators!(Bup, fermion_path_integral_up,
                               calculate_exp_K = calculate_exp_K, calculate_exp_V = calculate_exp_V)
        calculate_propagators!(Bdn, fermion_path_integral_dn,
                               calculate_exp_K = calculate_exp_K, calculate_exp_V = calculate_exp_V)
        # revert to oringial action derivative
        copyto!(dSdx, dSdx′)
    end

    return (accepted, logdetGup, sgndetGup, logdetGdn, sgndetGdn, δG, δθ)
end



# performs a hybrid/hamiltonian monte carlo update to the phonon configuration assuming
# the spin up and spin down sectors are equivalent
function  _hmc_update!(G::Matrix{T}, logdetG::E, sgndetG::T, G′::Matrix{T},
                       electron_phonon_parameters::ElectronPhononParameters{T,E},
                       fermion_path_integral::FermionPathIntegral{T,E},
                       fermion_greens_calculator::FermionGreensCalculator{T,E},
                       fermion_greens_calculator_alt::FermionGreensCalculator{T,E},
                       B::Vector{P}, dSdx::Matrix{E}, dSdx′::Matrix{E},
                       v::Matrix{E}, x′::Matrix{E}, x0::Matrix{E},
                       fourier_mass_matrix::FourierMassMatrix{E},
                       Nt::Int, nt::Int, Δt::E, initialize_force::Bool, first_update::Bool,
                       δG_max::E, δG::E, δθ::E, rng::AbstractRNG,
                       δG_reject::E = sqrt(δG_max),
                       recenter!::Function = identity,
                       update_stabilization_frequency::Bool = true) where {T, E, P<:AbstractPropagator{T,E}}

    (; β, Lτ, Δτ) = electron_phonon_parameters
    holstein_parameters = electron_phonon_parameters.holstein_parameters_up::HolsteinParameters{E}
    ssh_parameters = electron_phonon_parameters.ssh_parameters_up::SSHParameters{T}
    phonon_parameters = electron_phonon_parameters.phonon_parameters::PhononParameters{E}

    # whether the exponentiated on-site energy matrix needs to be updated with the phonon field,
    # true if there is a finite number of holstein couplings Nholstein
    calculate_exp_V = (holstein_parameters.Nholstein > 0)

    # whether the exponentiated hopping matrix needs to be updated with the phonon field,
    # true if there is a finite number of ssh couplings Nssh
    calculate_exp_K = (ssh_parameters.Nssh > 0)

    # get the phonon fields
    x = electron_phonon_parameters.x

    # calculate effective bosonic time-step
    Δt′ = Δt/nt

    # record initial phonon configuration
    copyto!(x0, x)
    copyto!(x′, x)

    # calculate initial derivative of fermionic action ∂S̃f/∂x = M⁻¹⋅∂Sf/∂x
    if initialize_force || first_update

        # set derivative of action to zero initially
        fill!(dSdx, 0.0)

        # calculate derivative of fermionic action
        (logdetG, sgndetG, δG, δθ) = fermionic_action_derivative!(dSdx, G, logdetG, sgndetG, δG, δθ,
                                                                  electron_phonon_parameters,
                                                                  fermion_greens_calculator, B)

        # multiply fermionic action derivative by two to acount for spin degeneracy
        @. dSdx = 2 * dSdx

        # calculate derivative associated with Holstein correction
        eval_derivative_holstein_action!(dSdx, x, Δτ, holstein_parameters, phonon_parameters) # spin-up
        eval_derivative_holstein_action!(dSdx, x, Δτ, holstein_parameters, phonon_parameters) # spin-down

        # calculate ∂S̃f/∂x = M⁻¹⋅∂Sf/∂x
        lmul!(fourier_mass_matrix, dSdx, -1.0)
    end

    # initialize fermion green's function matrix and determinants
    copyto!(G′, G)
    logdetG′ = logdetG
    sgndetG′ = sgndetG

    # intialize the alternate fermion greens calculators
    copyto!(fermion_greens_calculator_alt, fermion_greens_calculator)

    # record the initial fermionic action
    copyto!(dSdx′, dSdx)

    # initialize the momentum as v = R/√(M)
    randn!(rng, v)
    lmul!(fourier_mass_matrix, v, -0.5)

    # calculate the initial kinetic energy
    K = velocity_to_kinetic_energy(fourier_mass_matrix, v)

    # calculate initial bosonic action
    Sb = bosonic_action(electron_phonon_parameters)

    # calculate initial fermionic action
    Sf = 2*logdetG

    # record initial total action
    S = Sb + Sf

    # calculate the initial total energy
    H = S + K

    # flag to indicate whether or not hmc trajectory remains numerically stable
    numerically_stable = true

    # initialize numerical error for hmc trajectory
    δG′ = δG

    # iterate of fermionic time-steps
    for t in 1:Nt

        # calculate v(t+Δt/2) = v(t) - Δt/2⋅∂S̃f/∂x(t) = v(t) - Δt/2⋅M⁻¹⋅dSf/dx(t)
        @. v = v - Δt/2 * dSdx

        # initialize derivative of bosonic action ∂Sb/∂x
        fill!(dSdx, 0.0)
        bosonic_action_derivative!(dSdx, electron_phonon_parameters, holstein_correction=false)

        # calculate ∂S̃b/∂x = M⁻¹⋅∂Sb/∂x
        lmul!(fourier_mass_matrix, dSdx, -1.0)

        # iterate over bosonic time-steps
        for t′ in 1:nt

            # calculate v(t+Δt′/2) = v(t) - Δt′/2⋅∂S̃b/∂x(t) = v(t) - Δt′/2⋅M⁻¹⋅dSb/dx(t)
            @. v = v - Δt′/2 * dSdx

            # calculate x(t+Δt′) = x(t) + Δt′⋅v(t+Δt′/2)
            @. x = x + Δt′*v

            # re-center phonon fields assuming translationally invariant hamiltonian.
            # this typically does nothing, and is just the identity operator.
            recenter!(x)

            # calculate derivative of bosonic action ∂Sb/∂x
            fill!(dSdx, 0.0)
            bosonic_action_derivative!(dSdx, electron_phonon_parameters, holstein_correction=false)

            # calculate ∂S̃b/∂x = M⁻¹⋅∂Sb/∂x
            lmul!(fourier_mass_matrix, dSdx, -1.0)

            # calculate v(t+Δt′) = v(t+Δt′/2) - Δt′/2⋅∂S̃b/∂x(t+Δt′) = v(t+Δt′) - Δt′/2⋅M⁻¹⋅dSb/dx(t+Δt′)
            @. v = v - Δt′/2 * dSdx
        end

        # update the fermion path integrals for spin up and down sectors to reflect current phonon configuration
        update!(fermion_path_integral, electron_phonon_parameters, x, x′)
        
        # record the current phonon configuration
        copyto!(x′, x)

        # update the propagators to reflect current phonon configuration
        calculate_propagators!(B, fermion_path_integral, calculate_exp_K = calculate_exp_K, calculate_exp_V = calculate_exp_V)

        # update the Green's function to reflect the new phonon configuration
        try
            logdetG′, sgndetG′ = calculate_equaltime_greens!(G′, fermion_greens_calculator_alt, B)
        catch
            numerically_stable = false
            break
        end

        # if nan occurs terminate update
        if !isfinite(logdetG′)

            # record that numerically instability was encountered
            numerically_stable = false

            # terminate the HMC trajectory
            break
        end

        # initialize derivative of action to zero
        fill!(dSdx, 0.0)

        # calculate derivative of fermionic action
        try
            (logdetG′, sgndetG′, δG′, δθ) = fermionic_action_derivative!(dSdx, G′, logdetG′, sgndetG′, δG′, δθ,
                                                                        electron_phonon_parameters,
                                                                        fermion_greens_calculator_alt, B)
        catch
            numerically_stable = false
            break
        end

        # if numerical error too large or nan occurs
        if !isfinite(δG′) || !isfinite(logdetG′) || δG′ > δG_reject

            # record that numerically instability was encountered
            numerically_stable = false

            # terminate the HMC trajectory
            break
        end

        # multiply fermionic action derivative by two to acount for spin degeneracy
        @. dSdx = 2 * dSdx

        # calculate derivative associated with Holstein correction
        eval_derivative_holstein_action!(dSdx, x, Δτ, holstein_parameters, phonon_parameters) # spin-up
        eval_derivative_holstein_action!(dSdx, x, Δτ, holstein_parameters, phonon_parameters) # spin-down

        # calculate ∂S̃f/∂x = M⁻¹⋅∂Sf/∂x
        lmul!(fourier_mass_matrix, dSdx, -1.0)

        # calculate v(t+Δt) = v(t+Δt/2) - Δt/2⋅∂S̃f/∂x(t+Δt) = v(t+Δt/2) - Δt/2⋅M⁻¹⋅dSf/dx(t+Δt)
        @. v = v - Δt/2 * dSdx
    end

    # if numerically stable
    if numerically_stable

        # calculate the final kinetic energy given the velocity
        K′ = velocity_to_kinetic_energy(fourier_mass_matrix, v)

        # calculate final bosonic action
        Sb′ = bosonic_action(electron_phonon_parameters)

        # calculate final fermionic action
        Sf′ = 2*logdetG′

        # record final total action
        S′ = Sb′ + Sf′

        # calculate the final total energy
        H′ = S′ + K′

        # calculate the change in energy
        ΔH = H′ - H

        # calculate the probability of accepting the final configuration
        p = min(1.0, exp(-ΔH))

    # if numerically unstable
    else

        # set acceptance probability to zero
        p = zero(E)
    end

    # determine if update accepted
    accepted = rand(rng) < p
    
    # if proposed phonon configuration is accepted and hmc trajecotry remained numerically stable
    if accepted
        # record final greens function matrix
        copyto!(G, G′)
        # record final greens function determinant
        logdetG = logdetG′
        sgndetG = sgndetG′
        # record the final state of the fermion greens calculator
        copyto!(fermion_greens_calculator, fermion_greens_calculator_alt)
        # record numerical error associated up hmc update
        δG = δG′
        # update stabilization frequency if required
        if update_stabilization_frequency
            (updated, logdetG, sgndetG, δG, δθ) = update_stabalization_frequency!(
                G, logdetG, sgndetG,
                fermion_greens_calculator = fermion_greens_calculator,
                B = B, δG = δG, δθ = δθ, δG_max = δG_max
            )
        end
    # otherwise reject proposed update and revert to oringial phonon configuration
    else
        # update fermion path integrals to reflect initial phonon configuration
        update!(fermion_path_integral, electron_phonon_parameters, x0, x)
        # revert to initial phonon configuration
        copyto!(x, x0)
        # update propagators to reflect initial phonon configuration
        calculate_propagators!(B, fermion_path_integral, calculate_exp_K = calculate_exp_K, calculate_exp_V = calculate_exp_V)
        # revert to oringial action derivative
        copyto!(dSdx, dSdx′)
    end

    return (accepted, logdetG, sgndetG, δG, δθ)
end