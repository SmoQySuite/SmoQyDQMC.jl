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
- `efa::ExactFourierAccelerator{E,PFFT,PIFFT}`: Type to perform exact integration of equations of motion of quantum harmonic oscillator. 
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

    # action derivatives
    dSdx::Matrix{E}

    # matrix to contain intermediate spin up Green's function matrices
    Gup′::Matrix{T}

    # matrix to contain intermediate spin down Green's function matrices
    Gdn′::Matrix{T}

    # exact fourier accelerator
    efa::ExactFourierAccelerator{E,PFFT,PIFFT}
end

@doc raw"""
    EFAHMCUpdater(;
        # KEYWORD ARGUMENTS
        electron_phonon_parameters::ElectronPhononParameters{T,E},
        G::Matrix{T},
        Nt::Int,
        Δt::E,
        reg::E = 0.0,
        δ::E = 0.05
    ) where {T<:Number, E<:AbstractFloat}

# Keyword Arguments

- `electron_phonon_parameters::ElectronPhononParameters{T,E}`: Defines electron-phonon model.
- `G::Matrix{T}`: Sample Green's function matrix.
- `Nt::Int`: Number of time-steps used in EFA-HMC update.
- `Δt::E`: Average step size used for HMC update.
- `reg::E = Inf`: Regularization used for mass in equations of motion.
- `δ::E = 0.05`: Amount of jitter added to time-step used in EFA-HMC update.
"""
function EFAHMCUpdater(;
    # KEYWORD ARGUMENTS
    electron_phonon_parameters::ElectronPhononParameters{T,E},
    G::Matrix{T},
    Nt::Int,
    Δt::E,
    reg::E = 0.0,
    δ::E = 0.05
) where {T<:Number, E<:AbstractFloat}

    (; β, Δτ, phonon_parameters, x) = electron_phonon_parameters
    (; Ω, M) = phonon_parameters
    x0 = zero(x)
    p = zero(x)
    dSdx = zero(x)
    Gup′ = zero(G)
    Gdn′ = zero(G)

    # allocate and initialize ExactFourierAccelerator
    efa = ExactFourierAccelerator(Ω, M, β, Δτ, reg)

    return EFAHMCUpdater(Nt, Δt, δ, x0, p, dSdx, Gup′, Gdn′, efa)
end

@doc raw"""
    hmc_update!(
        # ARGUMENTS
        Gup::Matrix{H}, logdetGup::R, sgndetGup::H,
        Gdn::Matrix{H}, logdetGdn::R, sgndetGdn::H,
        electron_phonon_parameters::ElectronPhononParameters{T,R},
        hmc_updater::EFAHMCUpdater{T,R};
        # KEYWORD ARGUMENTS
        fermion_path_integral_up::FermionPathIntegral{H,T},
        fermion_path_integral_dn::FermionPathIntegral{H,T},
        fermion_greens_calculator_up::FermionGreensCalculator{H,R},
        fermion_greens_calculator_dn::FermionGreensCalculator{H,R},
        fermion_greens_calculator_up_alt::FermionGreensCalculator{H,R},
        fermion_greens_calculator_dn_alt::FermionGreensCalculator{H,R},
        Bup::Vector{P}, Bdn::Vector{P},
        δG::R, δθ::R, rng::AbstractRNG, 
        update_stabilization_frequency::Bool = false,
        δG_max::R = 1e-5,
        δG_reject::R = 1e-2,
        recenter!::Function = identity,
        Nt::Int = hmc_updater.Nt,
        Δt::R = hmc_updater.Δt,
        δ::R = hmc_updater.δ
    ) where {H<:Number, T<:Number, R<:Real, P<:AbstractPropagator{T}}

Perform EFA-HMC update to the phonon degrees of freedom.
This method returns `(accepted, logdetGup, sgndetGup, logdetGdn, sgndetGdn, δG, δθ)`, where `accepted`
is a boolean field indicating whether the proposed HMC update was accepted or rejected.

# Arguments

- `Gup::Matrix{H}`: Green's function matrix for spin up.
- `logdetGup::R`: Log determinant of Green's function matrix for spin up.
- `sgndetGup::H`: Sign of determinant of Green's function matrix for spin up.
- `Gdn::Matrix{H}`: Green's function matrix for spin down.
- `logdetGdn::R`: Log determinant of Green's function matrix for spin down.
- `electron_phonon_parameters::ElectronPhononParameters{T,R}`: Electron-phonon model parameters.
- `hmc_updater::EFAHMCUpdater{T,R}`: EFA-HMC updater.

# Keyword Arguments

- `fermion_path_integral_up::FermionPathIntegral{H}`: Fermion path integral for spin up.
- `fermion_path_integral_dn::FermionPathIntegral{H}`: Fermion path integral for spin down.
- `fermion_greens_calculator_up::FermionGreensCalculator{H,R}`: Fermion greens calculator for spin up.
- `fermion_greens_calculator_dn::FermionGreensCalculator{H,R}`: Fermion greens calculator for spin down.
- `fermion_greens_calculator_up_alt::FermionGreensCalculator{H,R}`: Alternative fermion greens calculator for spin up.
- `fermion_greens_calculator_dn_alt::FermionGreensCalculator{H,R}`: Alternative fermion greens calculator for spin down.
- `Bup::Vector{P}`: Spin up propagators.
- `Bdn::Vector{P}`: Spin down propagators.
- `δG::R`: Numerical error in Green's function corrected by numerical stabilization.
- `δθ::R`: Numerical error in the phase of the determinant of the Green's function matrix corrected by numerical stabilization.
- `rng::AbstractRNG`: Random number generator.
- `update_stabilization_frequency::Bool = false`: Whether to update the stabilization frequency.
- `δG_max::R = 1e-5`: Maximum numerical error in Green's function corrected by numerical stabilization.
- `δG_reject::R = 1e-2`: Reject the update if the numerical error in Green's function corrected by numerical stabilization is greater than this value.
- `Nt::Int = hmc_updater.Nt`: Number of time-steps used in EFA-HMC update.
- `Δt::R = hmc_updater.Δt`: Average step size used for HMC update.
- `δ::R = hmc_updater.δ`: Amount of jitter added to time-step used in EFA-HMC update.
"""
function hmc_update!(
    # ARGUMENTS
    Gup::Matrix{H}, logdetGup::R, sgndetGup::H,
    Gdn::Matrix{H}, logdetGdn::R, sgndetGdn::H,
    electron_phonon_parameters::ElectronPhononParameters{T,R},
    hmc_updater::EFAHMCUpdater{T,R};
    # KEYWORD ARGUMENTS
    fermion_path_integral_up::FermionPathIntegral{H,T},
    fermion_path_integral_dn::FermionPathIntegral{H,T},
    fermion_greens_calculator_up::FermionGreensCalculator{H,R},
    fermion_greens_calculator_dn::FermionGreensCalculator{H,R},
    fermion_greens_calculator_up_alt::FermionGreensCalculator{H,R},
    fermion_greens_calculator_dn_alt::FermionGreensCalculator{H,R},
    Bup::Vector{P}, Bdn::Vector{P},
    δG::R, δθ::R, rng::AbstractRNG, 
    update_stabilization_frequency::Bool = false,
    δG_max::R = 1e-5,
    δG_reject::R = 1e-2,
    recenter!::Function = identity,
    Nt::Int = hmc_updater.Nt,
    Δt::R = hmc_updater.Δt,
    δ::R = hmc_updater.δ
) where {H<:Number, T<:Number, R<:Real, P<:AbstractPropagator{T}}

    @assert fermion_path_integral_up.Sb == fermion_path_integral_dn.Sb "$(fermion_path_integral_up.Sb) ≠ $(fermion_path_integral_dn.Sb)"

    (; p, dSdx, Gup′, Gdn′, efa) = hmc_updater
    Δτ = electron_phonon_parameters.Δτ::E
    holstein_parameters_up = electron_phonon_parameters.holstein_parameters_up
    holstein_parameters_dn = electron_phonon_parameters.holstein_parameters_dn
    ssh_parameters_up = electron_phonon_parameters.ssh_parameters_up
    ssh_parameters_dn = electron_phonon_parameters.ssh_parameters_dn
    phonon_parameters = electron_phonon_parameters.phonon_parameters
    dispersion_parameters = electron_phonon_parameters.dispersion_parameters

    # add a bit of noise to the time-step
    Δt = Δt * (1.0 + (2*rand(rng)-1)*δ)
    
    # whether the exponentiated on-site energy matrix needs to be updated with the phonon field,
    # true if there is a non-zero number of holstein couplings Nholstein
    calculate_exp_V = (holstein_parameters_up.Nholstein > 0) || (holstein_parameters_dn.Nholstein > 0)

    # whether the exponentiated hopping matrix needs to be updated with the phonon field,
    # true if there is a non-zero number of ssh couplings Nssh
    calculate_exp_K = (ssh_parameters_up.Nssh > 0) || (ssh_parameters_dn.Nssh > 0)

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
    K = initialize_momentum!(p, efa, rng)

    # calculate initial bosonic action
    Sb = bosonic_action(electron_phonon_parameters)

    # calculate initial fermionic action
    Sf = logdetGup + logdetGdn

    # record initial total action
    S = Sb + Sf

    # calculate the initial total energy
    E = S + K

    # initialize numerical error
    δG′ = δG
    δθ′ = δθ

    # evolve momentum and phonon fields according to bosonic action and update the
    # fermion path integrals to reflect the change in the phonon fields
    update!(fermion_path_integral_up, electron_phonon_parameters, x, -1, spin = +1)
    update!(fermion_path_integral_dn, electron_phonon_parameters, x, -1, spin = -1)
    evolve_eom!(x, p, Δt/2, efa)
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
        evolve_eom!(x, p, Δt′, efa)
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
        K′ = kinetic_energy(p, efa)

        # calculate final bosonic action
        Sb′ = bosonic_action(electron_phonon_parameters)

        # calculate final fermionic action
        Sf′ = logdetGup′ + logdetGdn′

        # record final total action
        S′ = Sb′ + Sf′

        # calculate the initial total energy
        E′ = S′ + K′

        # calculate the change in energy
        ΔE = E′ - E

        # calculate the acceptance probability
        P_accept = min(1.0, exp(-ΔE))

    # if update went numerically unstable
    else

        # reject the proposed update
        P_accept = zero(R)
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

        # update bosonic action
        fermion_path_integral_up.Sb += (Sb′ - Sb)

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
        # ARGUMENTS
        G::Matrix{H}, logdetG::R, sgndetG::H,
        electron_phonon_parameters::ElectronPhononParameters{T,R},
        hmc_updater::EFAHMCUpdater{T,R};
        # KEYWORD ARGUMENTS
        fermion_path_integral::FermionPathIntegral{H,T},
        fermion_greens_calculator::FermionGreensCalculator{H,R},
        fermion_greens_calculator_alt::FermionGreensCalculator{H,R},
        B::Vector{P},
        δG::R, δθ::R, rng::AbstractRNG,
        update_stabilization_frequency::Bool = false,
        δG_max::R = 1e-5,
        δG_reject::R = 1e-2,
        recenter!::Function = identity,
        Nt::Int = hmc_updater.Nt,
        Δt::R = hmc_updater.Δt,
        δ::R = hmc_updater.δ
    ) where {H<:Number, T<:Number, R<:Real, P<:AbstractPropagator{T}}

Perform EFA-HMC update to the phonon degrees of freedom.
This method returns `(accepted, logdetG, sgndetG, δG, δθ)`, where `accepted`
is a boolean field indicating whether the proposed HMC update was accepted or rejected.

# Arguments

- `G::Matrix{H}`: Green's function matrix for spin up.
- `logdetG::R`: Log determinant of Green's function matrix for spin up.
- `sgndetG::H`: Sign of determinant of Green's function matrix for spin up.
- `electron_phonon_parameters::ElectronPhononParameters{T,R}`: Electron-phonon model parameters.
- `hmc_updater::EFAHMCUpdater{T,R}`: EFA-HMC updater.

# Keyword Arguments

- `fermion_path_integral::FermionPathIntegral{H}`: Fermion path integral.
- `fermion_greens_calculator::FermionGreensCalculator{H,R}`: Fermion greens calculator.
- `fermion_greens_calculator_alt::FermionGreensCalculator{H,R}`: Alternative fermion greens calculator.
- `B::Vector{P}`: Spin up propagators.
- `δG::R`: Numerical error in Green's function corrected by numerical stabilization.
- `δθ::R`: Numerical error in the phase of the determinant of the Green's function matrix corrected by numerical stabilization.
- `rng::AbstractRNG`: Random number generator.
- `update_stabilization_frequency::Bool = false`: Whether to update the stabilization frequency.
- `δG_max::R = 1e-5`: Maximum numerical error in Green's function corrected by numerical stabilization.
- `δG_reject::R = 1e-2`: Reject the update if the numerical error in Green's function corrected by numerical stabilization is greater than this value.
- `Nt::Int = hmc_updater.Nt`: Number of time-steps used in EFA-HMC update.
- `Δt::R = hmc_updater.Δt`: Average step size used for HMC update.
- `δ::R = hmc_updater.δ`: Amount of jitter added to time-step used in EFA-HMC update.
"""
function hmc_update!(
    # ARGUMENTS
    G::Matrix{H}, logdetG::R, sgndetG::H,
    electron_phonon_parameters::ElectronPhononParameters{T,R},
    hmc_updater::EFAHMCUpdater{T,R};
    # KEYWORD ARGUMENTS
    fermion_path_integral::FermionPathIntegral{H,T},
    fermion_greens_calculator::FermionGreensCalculator{H,R},
    fermion_greens_calculator_alt::FermionGreensCalculator{H,R},
    B::Vector{P},
    δG::R, δθ::R, rng::AbstractRNG,
    update_stabilization_frequency::Bool = false,
    δG_max::R = 1e-5,
    δG_reject::R = 1e-2,
    recenter!::Function = identity,
    Nt::Int = hmc_updater.Nt,
    Δt::R = hmc_updater.Δt,
    δ::R = hmc_updater.δ
) where {H<:Number, T<:Number, R<:Real, P<:AbstractPropagator{T}}

    (; p, dSdx, efa) = hmc_updater
    G′ = hmc_updater.Gup′

    Δτ = electron_phonon_parameters.Δτ::R
    holstein_parameters = electron_phonon_parameters.holstein_parameters_up
    ssh_parameters = electron_phonon_parameters.ssh_parameters_up
    phonon_parameters = electron_phonon_parameters.phonon_parameters
    dispersion_parameters = electron_phonon_parameters.dispersion_parameters

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
    K = initialize_momentum!(p, efa, rng)

    # calculate initial bosonic action
    Sb = bosonic_action(electron_phonon_parameters)

    # calculate initial fermionic action
    Sf = 2*logdetG

    # record initial total action
    S = Sb + Sf

    # calculate the initial total energy
    E = S + K

    # initialize numerical error
    δG′ = δG
    δθ′ = δθ

    # evolve momentum and phonon fields according to bosonic action and update the
    # fermion path integrals to reflect the change in the phonon fields
    update!(fermion_path_integral, electron_phonon_parameters, x, -1)
    evolve_eom!(x, p, Δt/2, efa)
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
        evolve_eom!(x, p, Δt′, efa)
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
        K′ = kinetic_energy(p, efa)

        # calculate final bosonic action
        Sb′ = bosonic_action(electron_phonon_parameters)

        # calculate final fermionic action
        Sf′ = 2*logdetG′

        # record final total action
        S′ = Sb′ + Sf′

        # calculate the initial total energy
        E′ = S′ + K′

        # calculate the change in energy
        ΔE = E′ - E

        # calculate the acceptance probability
        P_accept = min(1.0, exp(-ΔE))

    # if update went numerically unstable
    else

        # reject the proposed update
        P_accept = zero(R)
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

        # update bosonic action
        fermion_path_integral.Sb += (Sb - Sb′)

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