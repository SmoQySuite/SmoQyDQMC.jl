@doc raw"""
    EFAHMCUpdater{T<:Number, E<:AbstractFloat, PFFT, PIFFT}

Defines an Exact Fourier Acceleration Hamiltonian/Hybrid Monte Carlo (EFA-HMC) update
for the phonon degrees of freedom.

# Fields

- `Nt::Int`: Number of time-steps in HMC trajectory.
- `Δt::E`: Average time-step size used in HMC update.
- `δ::E`: Time-step used in EFA-HMC update is jittered by an amount `Δt = Δt * (1 + δ*(2*rand(rng)-1))`.
- `x::Matrix{E}`: Records initial phonon configuration in position space.
- `p::Matrix{E}`: Conjugate momentum in HMC dynamics.
- `dSdx::Matrix{E}`: Stores the derivative of the action.
- `Gup′::Matrix{T}`: Intermediate spin-up Green's function matrix during HMC trajectory.
- `Gdn′::Matrix{T}`: Intermediate spin-down Green's function matrix during HMC trajectory.
- `efa::ExactFourierAccelerator{E,PFFT,TFFT}`: Type to perform exact integration of equations of motion of quantum harmonic oscillator. 
"""
struct EFAHMCUpdater{T<:Number, E<:AbstractFloat, PFFT, PIFFT}

    # Number of time-step
    Nt::Int

    # Time-step
    Δt::E

    # Amount of disorder in HMC time-step.
    δ::E

    # position space phonon field configuration
    x::Matrix{E}

    # momentum
    p::Matrix{E}

    # fourier space phonon config
    x̃::Matrix{Complex{E}}

    # fourier space momentum
    p̃::Matrix{Complex{E}}

    # action derivatives
    dSdx::Matrix{E}

    # matrix to contain intermediate spin up Green's function matrices
    Gup′::Matrix{T}

    # matrix to contain intermediate spin down Green's function matrices
    Gdn′::Matrix{T}

    # exact fourier accelerator
    efa::ExactFourierAccelerator{E,PFFT,TFFT}
end

@doc raw"""
    EFAHMCUpdater(;
        # Keyword Arguments Start Here
        electron_phonon_parameters::ElectronPhononParameters{T,E},
        G::Matrix{T},
        Nt::Int,
        Δt::E,
        reg::E = 0.0,
        δ::E = 0.05
    ) where {T<:Number, E<:AbstractFloat}

# Arguments

- `electron_phonon_parameters::ElectronPhononParameters{T,E}`: Defines electron-phonon model.
- `G::Matrix{T}`: Sample Green's function matrix.
- `Nt::Int`: Number of time-steps used in EFA-HMC update.
- `Δt::E`: Average step size used for HMC update.
- `reg::E = Inf`: Regularization used for mass in equations of motion.
- `δ::E = 0.05`: Amount of jitter added to time-step used in EFA-HMC update.
"""
function EFAHMCUpdater(;
    # Keyword Arguments Start Here
    electron_phonon_parameters::ElectronPhononParameters{T,E},
    G::Matrix{T},
    Nt::Int,
    Δt::E,
    reg::E = 0.0,
    δ::E = 0.05
) where {T<:Number, E<:AbstractFloat}

    (; Δτ, phonon_parameters, x) = electron_phonon_parameters
    (; Ω, M) = phonon_parameters
    x0 = zero(x)
    p = zero(x)
    dSdx = zero(x)
    Gup′ = zero(G)
    Gdn′ = zero(G)

    # allocate and initialize ExactFourierAccelerator
    efa = ExactFourierAccelerator(Ω, M, β, Δτ, reg)

    return EFAHMCUpdater(Nt, Δt, δ, x0, p, u, dSdx, Gup′, Gdn′, efa)
end

@doc raw"""
    hmc_update!(
        Gup::Matrix{T}, logdetGup::E, sgndetGup::T,
        Gdn::Matrix{T}, logdetGdn::E, sgndetGdn::T,
        electron_phonon_parameters::ElectronPhononParameters{T,E},
        hmc_updater::EFAHMCUpdater{T,E};
        # Keyword Arguments Start Here
        fermion_path_integral_up::FermionPathIntegral{T,E},
        fermion_path_integral_dn::FermionPathIntegral{T,E},
        fermion_greens_calculator_up::FermionGreensCalculator{T,E},
        fermion_greens_calculator_dn::FermionGreensCalculator{T,E},
        fermion_greens_calculator_up_alt::FermionGreensCalculator{T,E},
        fermion_greens_calculator_dn_alt::FermionGreensCalculator{T,E},
        Bup::Vector{P}, Bdn::Vector{P},
        δG_max::E, δG::E, δθ::E, rng::AbstractRNG,
        update_stabilization_frequency::Bool = true,
        δG_reject::E = 1e-2,
        recenter!::Function = identity,
        Nt::Int = hmc_updater.Nt,
        Δt::E = hmc_updater.Δt,
        δ::E = hmc_updater.δ
    ) where {T, E, P<:AbstractPropagator{T,E}}

Perform EFA-HMC update to the phonon degrees of freedom.
This method returns `(accepted, logdetGup, sgndetGup, logdetGdn, sgndetGdn, δG, δθ)`, where `accepted`
is a boolean field indicating whether the proposed HMC update was accepted or rejected.
"""
function hmc_update!(
    Gup::Matrix{T}, logdetGup::E, sgndetGup::T,
    Gdn::Matrix{T}, logdetGdn::E, sgndetGdn::T,
    electron_phonon_parameters::ElectronPhononParameters{T,E},
    hmc_updater::EFAHMCUpdater{T,E};
    # Keyword Arguments Start Here
    fermion_path_integral_up::FermionPathIntegral{T,E},
    fermion_path_integral_dn::FermionPathIntegral{T,E},
    fermion_greens_calculator_up::FermionGreensCalculator{T,E},
    fermion_greens_calculator_dn::FermionGreensCalculator{T,E},
    fermion_greens_calculator_up_alt::FermionGreensCalculator{T,E},
    fermion_greens_calculator_dn_alt::FermionGreensCalculator{T,E},
    Bup::Vector{P}, Bdn::Vector{P},
    δG_max::E, δG::E, δθ::E, rng::AbstractRNG,
    update_stabilization_frequency::Bool = true,
    δG_reject::E = 1e-2,
    recenter!::Function = identity,
    Nt::Int = hmc_updater.Nt,
    Δt::E = hmc_updater.Δt,
    δ::E = hmc_updater.δ
) where {T, E, P<:AbstractPropagator{T,E}}

    (; reg, pfft, pifft, m, p, dSdx, Gup′, Gdn′) = hmc_updater

    Δτ = electron_phonon_parameters.Δτ::E
    holstein_parameters_up = electron_phonon_parameters.holstein_parameters_up::HolsteinParameters{E}
    holstein_parameters_dn = electron_phonon_parameters.holstein_parameters_dn::HolsteinParameters{E}
    ssh_parameters_up = electron_phonon_parameters.ssh_parameters_up::SSHParameters{T}
    ssh_parameters_up = electron_phonon_parameters.ssh_parameters_dn::SSHParameters{T}
    phonon_parameters = electron_phonon_parameters.phonon_parameters::PhononParameters{E}
    dispersion_parameters = electron_phonon_parameters.dispersion_parameters::DispersionParameters{E}

    # add a bit of noise to the time-step
    Δt = Δt * (1.0 + (2*rand(rng)-1)*δ)
    
    # whether the exponentiated on-site energy matrix needs to be updated with the phonon field,
    # true if there is a non-zero number of holstein couplings Nholstein
    calculate_exp_V = (holstein_parameters_up.Nholstein > 0)

    # whether the exponentiated hopping matrix needs to be updated with the phonon field,
    # true if there is a non-zero number of ssh couplings Nssh
    calculate_exp_K = (ssh_parameters_up.Nssh > 0)

    # flag to indicate numerical stability
    numerically_stable = true

    # record initial phonon configuration
    x = electron_phonon_parameters.x
    x_init = hmc_updater.x
    copyto!(x_init, x)

    # initialize fermion green's function matrices and their determinants determinants
    copyto!(Gup′, Gup)
    copyto!(Gdn′, Gdn)
    logdetGup′ = logdetGup
    sgndetGup′ = sgndetGup
    logdetGdn′ = logdetGdn
    sgndetGdn′ = sgndetGdn

    # initialize momentum and calculate initial kinetic energy
    K = kinetic_energy(p, hmc_updater, rng, initialize_momentum = true)

    # calculate initial bosonic action
    Sb = bosonic_action(electron_phonon_parameters)

    # calculate initial fermionic action
    Sf = logdetGup + logdetGdn

    # record initial total action
    S = Sb + Sf

    # calculate the initial total energy
    H = S + K

    # initialize numerical error
    δG′ = δG
    δθ′ = δθ

    # evolve momentum and phonon fields according to bosonic action and update the
    # fermion path integrals to reflect the change in the phonon fields
    update!(fermion_path_integral_up, electron_phonon_parameters, x, -1, spin = +1)
    update!(fermion_path_integral_dn, electron_phonon_parameters, x, -1, spin = -1)
    evolve_qho_action!(x, p, Δt/2, hmc_updater)
    recenter!(x)
    update!(fermion_path_integral_up, electron_phonon_parameters, x, +1, spin = +1)
    update!(fermion_path_integral_dn, electron_phonon_parameters, x, +1, spin = -1)

    # iterate over HMC time-steps
    for t in 1:Nt

        # intialize derivative of action to zero
        fill!(dSdx, 0)

        # update the spin up and spin down propagators to reflect current phonon configuration
        calculate_propagators!(
            Bup, fermion_path_integral_up,
            calculate_exp_K = calculate_exp_K,
            calculate_exp_V = calculate_exp_V
        )
        calculate_propagators!(
            Bdn, fermion_path_integral_dn,
            calculate_exp_K = calculate_exp_K,
            calculate_exp_V = calculate_exp_V
        )

        # attempt to calculate derivative of fermionic action
        try

            # update the Green's function to reflect the new phonon configuration
            logdetGup′, sgndetGup′ = calculate_equaltime_greens!(Gup′, fermion_greens_calculator_up_alt, Bup)
            logdetGdn′, sgndetGdn′ = calculate_equaltime_greens!(Gdn′, fermion_greens_calculator_dn_alt, Bdn)

            # calculate derivative of fermionic action for spin up
            (logdetGup′, sgndetGup′, δGup′, δθup′) = fermionic_action_derivative!(
                dSdx, Gup′, logdetGup′, sgndetGup′, δG′, δθ,
                electron_phonon_parameters,
                fermion_greens_calculator_up_alt,
                Bup,
                spin = +1
            )

            # calculate derivative of fermionic action for spin down
            (logdetGdn′, sgndetGdn′, δGdn′, δθdn′) = fermionic_action_derivative!(
                dSdx, Gdn′, logdetGdn′, sgndetGdn′, δG′, δθ,
                electron_phonon_parameters,
                fermion_greens_calculator_dn_alt,
                Bdn,
                spin = -1
            )

            # record max numerical error
            δG′ = max(δG, δGup′, δGdn′)
            δθ′ = max(δθ, δθup′, δθdn′)

        # if failed to calculate derivative of fermionic action
        catch

            # record that numerically instability was encountered
            numerically_stable = false

            # terminate the HMC trajectory
            break
        end

        # detect numerical instability if occurred
        if !isfinite(δG′) || !isfinite(logdetGup) || !isfinite(logdetGdn) || δG′ > δG_reject

            # record that numerically instability was encountered
            numerically_stable = false

            # terminate the HMC trajectory
            break
        end

        # calculate the anharmonic contribution to the action derivative
        eval_derivative_anharmonic_action!(dSdx, x, Δτ, phonon_parameters)

        # calculate the dispersive contribution to the action derivative
        eval_derivative_dispersive_action!(dSdx, x, Δτ, dispersion_parameters, phonon_parameters)

        # calculate the holstein contribution to the derivative of the bosonic action
        eval_derivative_holstein_action!(dSdx, x, Δτ, holstein_parameters_up, phonon_parameters)
        eval_derivative_holstein_action!(dSdx, x, Δτ, holstein_parameters_dn, phonon_parameters)

        # update momentum
        @. p = p - Δt * dSdx

        # evolve momentum and phonon fields according to bosonic action and update the
        # fermion path integrals to reflect the change in the phonon fields
        update!(fermion_path_integral_up, electron_phonon_parameters, x, -1, spin = +1)
        update!(fermion_path_integral_dn, electron_phonon_parameters, x, -1, spin = -1)
        Δt′ = (t==Nt) ? Δt/2 : Δt
        evolve_qho_action!(x, p, Δt′, hmc_updater)
        recenter!(x)
        update!(fermion_path_integral_up, electron_phonon_parameters, x, +1, spin = +1)
        update!(fermion_path_integral_dn, electron_phonon_parameters, x, +1, spin = -1)
    end

    # update the spin up and spin down propagators to reflect current phonon configuration
    calculate_propagators!(
        Bup, fermion_path_integral_up,
        calculate_exp_K = calculate_exp_K,
        calculate_exp_V = calculate_exp_V
    )
    calculate_propagators!(
        Bdn, fermion_path_integral_dn,
        calculate_exp_K = calculate_exp_K,
        calculate_exp_V = calculate_exp_V
    )

    try
        # attempt to update the Green's function to reflect the new phonon configuration
        logdetGup′, sgndetGup′ = calculate_equaltime_greens!(Gup′, fermion_greens_calculator_up_alt, Bup)
        logdetGdn′, sgndetGdn′ = calculate_equaltime_greens!(Gdn′, fermion_greens_calculator_dn_alt, Bdn)
    catch
        # record if a numerical instability is encountered
        numerically_stable = false
    end

    # if the simulation remained numerical stable
    if numerically_stable

        # calculate final kinetic energy
        K′ = kinetic_energy(p, hmc_updater, rng, initialize_momentum = false)

        # calculate final bosonic action
        Sb′ = bosonic_action(electron_phonon_parameters)

        # calculate final fermionic action
        Sf′ = logdetGup′ + logdetGdn′

        # record final total action
        S′ = Sb′ + Sf′

        # calculate the initial total energy
        H′ = S′ + K′

        # calculate the change in energy
        ΔH = H′ - H

        # calculate the acceptance probability
        P_accept = min(1.0, exp(-ΔH))

    # if update went numerically unstable
    else

        # reject the proposed update
        P_accept = zero(E)
    end

    # determine if update accepted
    accepted = rand(rng) < P_accept

    # if update was accepted
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
        δG = max(δG, δG′)
        δθ = max(δθ, δθ′)

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

    # if update is rejected
    else

        # update fermion path integrals to reflect initial phonon configuration
        update!(fermion_path_integral_up, fermion_path_integral_dn, electron_phonon_parameters, x_init, x)

        # revert to initial phonon configuration
        copyto!(electron_phonon_parameters.x, x_init)

        # update the spin up and spin down propagators to reflect initial phonon configuration
        calculate_propagators!(
            Bup, fermion_path_integral_up,
            calculate_exp_K = calculate_exp_K,
            calculate_exp_V = calculate_exp_V
        )
        calculate_propagators!(
            Bdn, fermion_path_integral_dn,
            calculate_exp_K = calculate_exp_K,
            calculate_exp_V = calculate_exp_V
        )
    end

    return (accepted, logdetGup, sgndetGup, logdetGdn, sgndetGdn, δG, δθ)
end

@doc raw"""
    hmc_update!(
        G::Matrix{T}, logdetG::E, sgndetG::T,
        electron_phonon_parameters::ElectronPhononParameters{T,E},
        hmc_updater::EFAHMCUpdater{T,E};
        fermion_path_integral::FermionPathIntegral{T,E},
        fermion_greens_calculator::FermionGreensCalculator{T,E},
        fermion_greens_calculator_alt::FermionGreensCalculator{T,E},
        B::Vector{P},
        δG_max::E, δG::E, δθ::E, rng::AbstractRNG,
        update_stabilization_frequency::Bool = true,
        δG_reject::E = 1e-2,
        recenter!::Function = identity,
        Nt::Int = hmc_updater.Nt,
        Δt::E = hmc_updater.Δt,
        δ::E = hmc_updater.δ
    ) where {T, E, P<:AbstractPropagator{T,E}}

Perform EFA-HMC update to the phonon degrees of freedom.
This method returns `(accepted, logdetG, sgndetG, δG, δθ)`, where `accepted`
is a boolean field indicating whether the proposed HMC update was accepted or rejected.
"""
function hmc_update!(
    G::Matrix{T}, logdetG::E, sgndetG::T,
    electron_phonon_parameters::ElectronPhononParameters{T,E},
    hmc_updater::EFAHMCUpdater{T,E};
    fermion_path_integral::FermionPathIntegral{T,E},
    fermion_greens_calculator::FermionGreensCalculator{T,E},
    fermion_greens_calculator_alt::FermionGreensCalculator{T,E},
    B::Vector{P},
    δG_max::E, δG::E, δθ::E, rng::AbstractRNG,
    update_stabilization_frequency::Bool = true,
    δG_reject::E = 1e-2,
    recenter!::Function = identity,
    Nt::Int = hmc_updater.Nt,
    Δt::E = hmc_updater.Δt,
    δ::E = hmc_updater.δ
) where {T, E, P<:AbstractPropagator{T,E}}

    (; m, p, dSdx) = hmc_updater
    G′ = hmc_updater.Gup′

    Δτ = electron_phonon_parameters.Δτ::E
    holstein_parameters = electron_phonon_parameters.holstein_parameters_up::HolsteinParameters{E}
    ssh_parameters = electron_phonon_parameters.ssh_parameters_up::SSHParameters{T}
    phonon_parameters = electron_phonon_parameters.phonon_parameters::PhononParameters{E}
    dispersion_parameters = electron_phonon_parameters.dispersion_parameters::DispersionParameters{E}

    # add a bit of noise to the time-step
    Δt = Δt * (1.0 + (2*rand(rng)-1)*δ)
    
    # whether the exponentiated on-site energy matrix needs to be updated with the phonon field,
    # true if there is a non-zero number of holstein couplings Nholstein
    calculate_exp_V = (holstein_parameters.Nholstein > 0)

    # whether the exponentiated hopping matrix needs to be updated with the phonon field,
    # true if there is a non-zero number of ssh couplings Nssh
    calculate_exp_K = (ssh_parameters.Nssh > 0)

    # flag to indicate numerical stability
    numerically_stable = true

    # record initial phonon configuration
    x = electron_phonon_parameters.x
    x_init = hmc_updater.x
    copyto!(x_init, x)

    # intialize the alternate fermion greens calculators
    copyto!(fermion_greens_calculator_alt, fermion_greens_calculator)

    # initialize fermion green's function matrices and their determinants determinants
    copyto!(G′, G)
    logdetG′ = logdetG
    sgndetG′ = sgndetG

    # initialize momentum and calculate initial kinetic energy
    K = kinetic_energy(p, hmc_updater, rng, initialize_momentum = true)

    # calculate initial bosonic action
    Sb = bosonic_action(electron_phonon_parameters)

    # calculate initial fermionic action
    Sf = 2*logdetG

    # record initial total action
    S = Sb + Sf

    # calculate the initial total energy
    H = S + K

    # initialize numerical error
    δG′ = δG
    δθ′ = δθ

    # evolve momentum and phonon fields according to bosonic action and update the
    # fermion path integrals to reflect the change in the phonon fields
    update!(fermion_path_integral, electron_phonon_parameters, x, -1)
    evolve_qho_action!(x, p, Δt/2, hmc_updater)
    recenter!(x)
    update!(fermion_path_integral, electron_phonon_parameters, x, +1)

    # iterate over HMC time-steps
    for t in 1:Nt

        # intialize derivative of action to zero
        fill!(dSdx, 0)

        # update the propagators to reflect current phonon configuration
        calculate_propagators!(
            B, fermion_path_integral,
            calculate_exp_K = calculate_exp_K,
            calculate_exp_V = calculate_exp_V
        )

        # attempt to calculate the derivative of the fermionic action
        try

            # update the Green's function to reflect the new phonon configuration
            logdetG′, sgndetG′ = calculate_equaltime_greens!(G′, fermion_greens_calculator_alt, B)

            # calculate derivative of fermionic action for spin up
            (logdetG′, sgndetG′, δG″, δθ″) = fermionic_action_derivative!(
                dSdx, G′, logdetG′, sgndetG′, δG′, δθ,
                electron_phonon_parameters,
                fermion_greens_calculator_alt, B
            )

            # record max numerical error
            δG′ = max(δG, δG″)
            δθ′ = max(δθ, δθ″)
            
        # if failed to calculate fermionic deterimant
        catch

            # record that numerically instability was encountered
            numerically_stable = false

            # terminate the HMC trajectory
            break
        end

        # detect numerical instability if occurred
        if !isfinite(δG′) || !isfinite(logdetG) || δG′ > δG_reject

            # record that numerically instability was encountered
            numerically_stable = false

            # terminate the HMC trajectory
            break
        end

        # calculate the holstein contribution to the derivative of the bosonic action
        eval_derivative_holstein_action!(dSdx, x, Δτ, holstein_parameters, phonon_parameters)

        # account for both spin species with the derivative
        @. dSdx = 2 * dSdx

        # calculate the anharmonic contribution to the action derivative
        eval_derivative_anharmonic_action!(dSdx, x, Δτ, phonon_parameters)

        # calculate the dispersive contribution to the action derivative
        eval_derivative_dispersive_action!(dSdx, x, Δτ, dispersion_parameters, phonon_parameters)

        # update momentum
        @. p = p - Δt * dSdx

        # evolve momentum and phonon fields according to bosonic action and update the
        # fermion path integrals to reflect the change in the phonon fields
        update!(fermion_path_integral, electron_phonon_parameters, x, -1)
        Δt′ = (t==Nt) ? Δt/2 : Δt
        evolve_qho_action!(x, p, Δt′, hmc_updater)
        recenter!(x)
        update!(fermion_path_integral, electron_phonon_parameters, x, +1)
    end

    # update the spin up and spin down propagators to reflect current phonon configuration
    calculate_propagators!(
        B, fermion_path_integral,
        calculate_exp_K = calculate_exp_K,
        calculate_exp_V = calculate_exp_V
    )

    try
        # attempt to update the Green's function to reflect the new phonon configuration
        logdetG′, sgndetG′ = calculate_equaltime_greens!(G′, fermion_greens_calculator_alt, B)
    catch
        # record if a numerical instability is encountered
        numerically_stable = false
    end

    # if the simulation remained numerical stable
    if numerically_stable

        # calculate final kinetic energy
        K′ = kinetic_energy(p, hmc_updater, rng, initialize_momentum = false)

        # calculate final bosonic action
        Sb′ = bosonic_action(electron_phonon_parameters)

        # calculate final fermionic action
        Sf′ = 2*logdetG′

        # record final total action
        S′ = Sb′ + Sf′

        # calculate the initial total energy
        H′ = S′ + K′

        # calculate the change in energy
        ΔH = H′ - H

        # calculate the acceptance probability
        P_accept = min(1.0, exp(-ΔH))

    # if update went numerically unstable
    else

        # reject the proposed update
        P_accept = zero(E)
    end

    # determine if update accepted
    accepted = rand(rng) < P_accept

    # if update was accepted
    if accepted

        # record final green function matrices
        copyto!(G, G′)

        # record final green function determinant
        logdetG = logdetG′
        sgndetG = sgndetG′

        # record the final state of the fermion greens calculator
        copyto!(fermion_greens_calculator, fermion_greens_calculator_alt)

        # record numerical error associated up hmc update
        δG = max(δG, δG′)
        δθ = max(δθ, δθ′)

        # update stabilization frequency if required
        if update_stabilization_frequency
            (updated, logdetG, sgndetG, δG, δθ) = update_stabalization_frequency!(
                G, logdetG, sgndetG,
                fermion_greens_calculator = fermion_greens_calculator,
                B = B, δG = δG, δθ = δθ, δG_max = δG_max
            )
        end

    # if update is rejected
    else

        # update fermion path integrals to reflect initial phonon configuration
        update!(fermion_path_integral, electron_phonon_parameters, x_init, x)

        # revert to initial phonon configuration
        copyto!(x, x_init)

        # update the spin up and spin down propagators to reflect initial phonon configuration
        calculate_propagators!(
            B, fermion_path_integral,
            calculate_exp_K = calculate_exp_K,
            calculate_exp_V = calculate_exp_V
        )
    end

    return (accepted, logdetG, sgndetG, δG, δθ)
end


# analytically evolve quantum harmonic oscillator action
function evolve_qho_action!(
    x::Matrix{E}, p::Matrix{E}, Δt::E,
    hmc_updater::EFAHMCUpdater{T, E, PFFT, PIFFT}
) where {T<:Number, E<:AbstractFloat, PFFT, PIFFT}

    (; m, ω, x̃, p̃, u, pfft, pifft) = hmc_updater

    # length of imaginary time axis
    Lτ = size(x,2)

    # fourier transform phonon fields from imaginary time to fourier space
    @. u = x / sqrt(Lτ)
    mul!(x̃, pfft, u)

    # fourier transform momentum from imaginary time to fourier space
    @. u = p / sqrt(Lτ)
    mul!(p̃, pfft, u)

    # iterate over fourier modes
    @simd for n in axes(ω,2)
        # iterate over phonon modes
        for i in axes(ω,1)
            # get relevant frequency
            ωₙ = ω[i,n]
            # get the relevant mass
            mᵢ = m[i,n]
            # make sure mass if finite
            if isfinite(mᵢ)
                # get initial position and momentum
                x̃′ = x̃[i,n]
                p̃′ = p̃[i,n]
                # if finite frequency
                if ωₙ > 1e-10
                    # update position analytically
                    x̃[i,n] = x̃′*cos(ωₙ*Δt) + p̃′/(ωₙ*mᵢ) * sin(ωₙ*Δt)
                # if frequency is very near zero
                elseif abs(ωₙ) ≤ 1e-10
                    # update position numerically using taylor expansion of analytic expression
                    x̃[i,n] = x̃′ + (Δt - Δt^3*ωₙ^2/6 + Δt^5*ωₙ^4/120) * p̃[i,n]/mᵢ
                end
                # update momentum
                p̃[i,n] = p̃′*cos(ωₙ*Δt) - x̃′*(ωₙ*mᵢ) * sin(ωₙ*Δt)
            end
        end
    end

    # fourier transform phonon fields from fourier space to imaginary time
    mul!(u, pifft, x̃)
    @. x = real(u) * sqrt(Lτ)

    # fourier transform momentum from fourier space to imaginary time
    mul!(u, pifft, p̃)
    @. p = real(u) * sqrt(Lτ)

    return nothing
end

# calculate kinetic energy
function kinetic_energy(
    p::Matrix{E},
    hmc_updater::EFAHMCUpdater{T, E, PFFT, PIFFT},
    rng::AbstractRNG;
    initialize_momentum::Bool = false
) where {T<:Number, E<:AbstractFloat, PFFT, PIFFT}

    (; m, p̃, reg, pfft, pifft, u) = hmc_updater

    # length of imaginary time axis
    Lτ = size(p, 2)

    # if initializing momentum
    if initialize_momentum

        # sample random normal number
        randn!(rng, p)
    end

    # initialize kinetic energy to zero
    K = zero(E)

    # if regularization is infinte
    if isinf(reg)

        # if initializing momentum
        if initialize_momentum
            # iterate over momentum
            for i in eachindex(p)
                # initialize momentum
                p[i] = isinf(m[i]) ? 0.0 : sqrt(m[i]) * p[i]
            end
        end

        # iterate over each momentum
        for i in eachindex(p)
            # calculate contribution to kinetic energy
            K += isinf(m[i]) ? 0.0 : abs2(p[i])/(2*m[i])
        end

    # if finite regularization
    else

        # fourier transform momentum from imaginary time to fourier space
        @. u = p / sqrt(Lτ)
        mul!(p̃, pfft, u)

        # if initializing momentum
        if initialize_momentum
            # iterate over momentum
            for i in eachindex(p)
                # initialize momentum
                p̃[i] = isinf(m[i]) ? 0.0 : sqrt(m[i]) * p̃[i]
            end
        end

        # iterate over each momentum
        for i in eachindex(p)
            # calculate contribution to kinetic energy
            K += isinf(m[i]) ? 0.0 : abs2(p̃[i])/(2*m[i])
        end

        # fourier transform momentum from fourier space to imaginary time
        if initialize_momentum
            mul!(u, pifft, p̃)
            @. p = real(u) * sqrt(Lτ)
        end
    end

    return K
end