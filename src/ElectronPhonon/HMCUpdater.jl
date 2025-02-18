@doc raw"""
    HMCUpdater{T<:Number, E<:AbstractFloat, PFFT, PIFFT}

Defines a hybrid/hamiltonian monte carlo (HMC) update for the phonon degrees of freedom.

# Fields

- `Nt::Int`: Mean trajectory length in terms of the number of fermionic time-steps.
- `Δt::E`: Average fermionic time-step size used in HMC update.
- `nt::Int`: Number of bosonic time-steps per fermionic time-step using a bosonic time-step `Δt′=Δt/nt`.
- `δ::E`: Time-step used in HMC update is jittered by an amount `Δt = Δt * (1 + δ*(2*rand(rng)-1))`.
- `M::FourierMassMatrix{E,PFFT,PIFFT}`: Defines the [`FourierMassMatrix`](@ref).
- `dSdx::Matrix{E}`: Array to contain derivative of fermionic and bosonic action during HMC trajectory.
- `dSfdx0::Matrix{E}`: Initial derivative of fermionic action associated with the initial phonon configuration.
- `Gup′::Matrix{T}`: Matrix to contain itermediate spin-up Green's function matrices.
- `Gdn′::Matrix{T}`: Matrix to contain itermediate spin-down Green's function matrices.
- `x′::Matrix{E}`: Array to record intermediate phonon configurations.
- `x0::Matrix{E}`: Array to record initial phonon configuration.
- `v::Matrix{E}`: Conjugate momentum to phonon fields in HMC trajectory.
- `first_update::Bool`: A flag indicating whether the next update will be the first update
"""
mutable struct HMCUpdater{T<:Number, E<:AbstractFloat, PFFT, PIFFT}

    # average number of fermionic time-steps
    const Nt::Int

    # fermionic time-step
    const Δt::E

    # fraction to jitter the time-step size by
    const δ::E

    # number of bosinic time-steps per fermionic time-step
    const nt::Int

    # fourier mass matrix for fourier acceleration
    const M::FourierMassMatrix{E, PFFT, PIFFT}

    # matrix to contain derivitive of action during trajectory
    const dSdx::Matrix{E}

    # matrix to contain initial fermionic derivative of action
    const dSfdx0::Matrix{E}

    # matrix to contain intermediate spin up Green's function matrices
    const Gup′::Matrix{T}

    # matrix to contain intermediate spin down Green's function matrices
    const Gdn′::Matrix{T}

    # contains intermediate phonon configuration
    const x′::Matrix{E}

    # record initial phonon configuration at the start of the trajectory
    const x0::Matrix{E}

    # conjugate velocities on hamiltonian dynamices
    const v::Matrix{E}

    # flag indicating whether the next update will be the first update
    first_update::Bool
end

@doc raw"""
    HMCUpdater(;
        electron_phonon_parameters::ElectronPhononParameters{T,E},
        G::Matrix{T},
        Nt::Int,
        Δt::E,
        nt::Int,
        reg::E
    ) where {T,E}

Initialize and return an instance of [`HMCUpdater`](@ref).

# Arguments

- `electron_phonon_parameters::ElectronPhononParameters{T,E}`: Defines electron phonon model.
- `G::Matrix{T}`: Template Green's function matrix.
- `Nt::Int`: Number of fermionic timesteps in HMC trajectory.
- `Δt::E`: Fermionic time-step.
- `nt::Int`: Number of bosonic time-steps per fermionic time-step.
- `reg::E`: Regularization parameter for defining an instance of [`FourierMassMatrix`](@ref).
- `δ::E = 0.05`: Proportion by which the HMC time-step is jittered befored each update.
"""
function HMCUpdater(;
    electron_phonon_parameters::ElectronPhononParameters{T,E},
    G::Matrix{T},
    Nt::Int,
    Δt::E,
    nt::Int,
    reg::E,
    δ::E = 0.05
) where {T,E}

    M = FourierMassMatrix(electron_phonon_parameters, reg)
    x = electron_phonon_parameters.x
    x′ = similar(x)
    x0 = similar(x)
    v = similar(x)
    dSdx = similar(x)
    dSfdx0 = similar(x)
    Gup′ = similar(G)
    Gdn′ = similar(G)
    first_update = true

    # initialize derivative of action to zero
    fill!(dSdx, 0)
    fill!(dSfdx0, 0)

    return HMCUpdater(Nt, Δt, δ, nt, M, dSdx, dSfdx0, Gup′, Gdn′, x′, x0, v, first_update)
end


@doc raw"""
    hmc_update!(
        Gup::Matrix{T}, logdetGup::E, sgndetGup::T,
        Gdn::Matrix{T}, logdetGdn::E, sgndetGdn::T,
        electron_phonon_parameters::ElectronPhononParameters{T,E},
        hmc_updater::HMCUpdater{T,E};
        # Keyword Arguments Start Here
        fermion_path_integral_up::FermionPathIntegral{T,E},
        fermion_path_integral_dn::FermionPathIntegral{T,E},
        fermion_greens_calculator_up::FermionGreensCalculator{T,E},
        fermion_greens_calculator_dn::FermionGreensCalculator{T,E},
        fermion_greens_calculator_up_alt::FermionGreensCalculator{T,E},
        fermion_greens_calculator_dn_alt::FermionGreensCalculator{T,E},
        Bup::Vector{P}, Bdn::Vector{P},
        δG::E, δθ::E, rng::AbstractRNG,
        update_stabilization_frequency::Bool = false,
        δG_max::E = 1e-5,
        δG_reject::E = 1e-2,
        initialize_force::Bool = true,
        recenter!::Function = identity,
        Nt::Int = hmc_updater.Nt,
        nt::Int = hmc_updater.nt,
        Δt::E = hmc_updater.Δt,
        δ::E = hmc_updater.δ
    ) where {T<:Number, E<:AbstractFloat, P<:AbstractPropagator{T,E}}

Perform HMC update to the phonon degrees of freedom.
This method returns `(accepted, logdetGup, sgndetGup, logdetGdn, sgndetGdn, δG, δθ)`, where `accepted`
is a boolean field indicating whether the proposed HMC update was accepted or rejected.
"""
function hmc_update!(
    Gup::Matrix{T}, logdetGup::E, sgndetGup::T,
    Gdn::Matrix{T}, logdetGdn::E, sgndetGdn::T,
    electron_phonon_parameters::ElectronPhononParameters{T,E},
    hmc_updater::HMCUpdater{T,E};
    # Keyword Arguments Start Here
    fermion_path_integral_up::FermionPathIntegral{T,E},
    fermion_path_integral_dn::FermionPathIntegral{T,E},
    fermion_greens_calculator_up::FermionGreensCalculator{T,E},
    fermion_greens_calculator_dn::FermionGreensCalculator{T,E},
    fermion_greens_calculator_up_alt::FermionGreensCalculator{T,E},
    fermion_greens_calculator_dn_alt::FermionGreensCalculator{T,E},
    Bup::Vector{P}, Bdn::Vector{P},
    δG::E, δθ::E, rng::AbstractRNG,
    update_stabilization_frequency::Bool = false,
    δG_max::E = 1e-5,
    δG_reject::E = 1e-2,
    initialize_force::Bool = true,
    recenter!::Function = identity,
    Nt::Int = hmc_updater.Nt,
    nt::Int = hmc_updater.nt,
    Δt::E = hmc_updater.Δt,
    δ::E = hmc_updater.δ
) where {T<:Number, E<:AbstractFloat, P<:AbstractPropagator{T,E}}

    (; M, dSdx, dSfdx0, x′, x0, v, Gup′, Gdn′, first_update) = hmc_updater

    # jitter time-step
    Δt′ = Δt * (1 + δ*(2*rand(rng)-1))

    # perform HMC update
    (accepted, logdetGup, sgndetGup, logdetGdn, sgndetGdn, δG, δθ) = _hmc_update!(
        Gup, logdetGup, sgndetGup, Gup′,
        Gdn, logdetGdn, sgndetGdn, Gdn′,
        electron_phonon_parameters,
        fermion_path_integral_up, fermion_path_integral_dn,
        fermion_greens_calculator_up, fermion_greens_calculator_dn,
        fermion_greens_calculator_up_alt, fermion_greens_calculator_dn_alt,
        Bup, Bdn, dSdx, dSfdx0, v, x′, x0, M, Nt, nt, Δt′, initialize_force, first_update, δG_max, δG, δθ, rng,
        δG_reject, recenter!, update_stabilization_frequency
    )

    # set first update to false as an update was just performed
    hmc_updater.first_update = false

    return (accepted, logdetGup, sgndetGup, logdetGdn, sgndetGdn, δG, δθ)
end

@doc raw"""
    hmc_update!(
        G::Matrix{T}, logdetG::E, sgndetG::T,
        electron_phonon_parameters::ElectronPhononParameters{T,E},
        hmc_updater::HMCUpdater{T,E};
        # Keyword Arguments Start Here
        fermion_path_integral::FermionPathIntegral{T,E},
        fermion_greens_calculator::FermionGreensCalculator{T,E},
        fermion_greens_calculator_alt::FermionGreensCalculator{T,E},
        B::Vector{P}, δG::E, δθ::E, rng::AbstractRNG,
        update_stabilization_frequency::Bool = false,
        δG_max::E = 1e-5,
        δG_reject::E = 1e-2,
        initialize_force::Bool = true,
        recenter!::Function = identity,
        Nt::Int = hmc_updater.Nt,
        nt::Int = hmc_updater.nt,
        Δt::E = hmc_updater.Δt,
        δ::E = hmc_updater.δ
    ) where {T<:Number, E<:AbstractFloat, P<:AbstractPropagator{T,E}}

Perform HMC update to the phonon degrees of freedom assuming the spin-up and spin-down sectors are equivalent.
This method returns `(accepted, logdetG, sgndetG, δG, δθ)`, where `accepted`
is a boolean field indicating whether the proposed HMC update was accepted or rejected.
"""
function hmc_update!(
    G::Matrix{T}, logdetG::E, sgndetG::T,
    electron_phonon_parameters::ElectronPhononParameters{T,E},
    hmc_updater::HMCUpdater{T,E};
    # Keyword Arguments Start Here
    fermion_path_integral::FermionPathIntegral{T,E},
    fermion_greens_calculator::FermionGreensCalculator{T,E},
    fermion_greens_calculator_alt::FermionGreensCalculator{T,E},
    B::Vector{P}, δG::E, δθ::E, rng::AbstractRNG,
    update_stabilization_frequency::Bool = false,
    δG_max::E = 1e-5,
    δG_reject::E = 1e-2,
    initialize_force::Bool = true,
    recenter!::Function = identity,
    Nt::Int = hmc_updater.Nt,
    nt::Int = hmc_updater.nt,
    Δt::E = hmc_updater.Δt,
    δ::E = hmc_updater.δ
) where {T<:Number, E<:AbstractFloat, P<:AbstractPropagator{T,E}}

    (; M, dSdx, dSfdx0, x′, x0, v, Gup′, first_update) = hmc_updater

    # jitter time-step
    Δt′ = Δt * (1 + δ*(2*rand(rng)-1))

    # perform HMC update
    (accepted, logdetG, sgndetG, δG, δθ) = _hmc_update!(
        G, logdetG, sgndetG, Gup′, electron_phonon_parameters,
        fermion_path_integral, fermion_greens_calculator, fermion_greens_calculator_alt,
        B, dSdx, dSfdx0, v, x′, x0, M, Nt, nt, Δt′, initialize_force, first_update, δG_max, δG, δθ, rng,
        δG_reject, recenter!, update_stabilization_frequency
    )

    # set first update to false as an update was just performed
    hmc_updater.first_update = false

    return (accepted, logdetG, sgndetG, δG, δθ)
end