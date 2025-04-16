@doc raw"""
    radial_update!(
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
        phonon_id::Union{Int, Nothing} = nothing,
        σ::E = 1.0
    ) where {T<:Number, E<:AbstractFloat, P<:AbstractPropagator{T,E}}

Perform a radial update to the phonon fields, as described by Algorithm 1 in the paper
[arXiv:2411.18218](https://arxiv.org/abs/2411.18218).
Specifically, the proposed update to the phonon fields ``x`` is a rescaling such that
``x \rightarrow e^{\gamma} x`` where ``\gamma \sim N(0, \sigma/\sqrt{d})`` and ``d`` is
the number of phonon fields being updated.

# Arguments

- `Gup::Matrix{T}`: Spin-up eqaul-time Greens function matrix.
- `logdetGup::E`: Log of the determinant of the spin-up eqaul-time Greens function matrix.
- `sgndetGup::T`: Sign/phase of the determinant of the spin-up eqaul-time Greens function matrix.
- `Gdn::Matrix{T}`: Spin-down eqaul-time Greens function matrix.
- `logdetGdn::E`: Log of the determinant of the spin-down eqaul-time Greens function matrix.
- `sgndetGdn::T`: Sign/phase of the determinant of the spin-down eqaul-time Greens function matrix.
- `electron_phonon_parameters::ElectronPhononParameters{T,E}`: Electron-phonon parameters, including the current phonon configuration.

# Keyword Arguments

- `fermion_path_integral_up::FermionPathIntegral{T,E}`: An instance of [`FermionPathIntegral`](@ref) type for spin-up electrons.
- `fermion_path_integral_dn::FermionPathIntegral{T,E}`: An instance of [`FermionPathIntegral`](@ref) type for spin-down electrons.
- `fermion_greens_calculator_up::FermionGreensCalculator{T,E}`: Contains matrix factorization information for current spin-up sector state.
- `fermion_greens_calculator_dn::FermionGreensCalculator{T,E}`: Contains matrix factorization information for current spin-down sector state.
- `fermion_greens_calculator_up_alt::FermionGreensCalculator{T,E}`: Used to calculate matrix factorizations for proposed spin-up sector state.
- `fermion_greens_calculator_dn_alt::FermionGreensCalculator{T,E}`: Used to calculate matrix factorizations for proposed spin-up sector state.
- `Bup::Vector{P}`: Spin-up propagators for each imaginary time slice.
- `Bdn::Vector{P}`: Spin-down propagators for each imaginary time slice.
- `rng::AbstractRNG`: Random number generator used in method instead of global random number generator, important for reproducibility.
- `phonon_id::Union{Int, Nothing} = nothing`: Apply radial update to phonon fields corresponding tp specified `PHONON_ID`. If `phonon_id = nothing`, then radial update is applied to all phonon fields.
- `σ::E = 1.0`: Relative size of the radial update.
"""
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
    phonon_id::Union{Int, Nothing} = nothing,
    σ::E = 1.0
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

    # make sure stabilization frequencies match
    if fermion_greens_calculator_up.n_stab != fermion_greens_calculator_up_alt.n_stab
        resize!(fermion_greens_calculator_up_alt, fermion_greens_calculator_up.n_stab)
    end

    # make sure stabilization frequencies match
    if fermion_greens_calculator_dn.n_stab != fermion_greens_calculator_dn_alt.n_stab
        resize!(fermion_greens_calculator_dn_alt, fermion_greens_calculator_dn.n_stab)
    end

    # get the number of phonon modes per unit cell
    nphonon = phonon_parameters.nphonon

    # total number of phonon modes
    Nphonon = phonon_parameters.Nphonon

    # number of unit cells
    Nunitcells = Nphonon ÷ nphonon

    # whether the exponentiated on-site energy matrix needs to be updated with the phonon field,
    # true if phonon mode appears in holstein coupling
    calculate_exp_V = ((holstein_parameters_up.nholstein > 0) ||
                       (holstein_parameters_dn.nholstein > 0))

    # whether the exponentiated hopping matrix needs to be updated with the phonon field,
    # true if phonon mode appears in SSH coupling
    calculate_exp_K = ((ssh_parameters_up.nssh > 0) ||
                       (ssh_parameters_dn.nssh > 0))

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
    σR = σ / sqrt(d)

    # randomly sample expansion/contraction coefficient
    γ = randn(rng) * σR
    expγ = exp(γ)

    # calculate initial bosonic action
    Sb = bosonic_action(electron_phonon_parameters)

    # calculate initial fermionic action
    Sf = logdetGup + logdetGdn

    # calculate initial total action
    S = Sb + Sf

    # substract off the effect of the current phonon configuration on the fermion path integrals
    if calculate_exp_V
        update!(fermion_path_integral_up, holstein_parameters_up, x, -1)
        update!(fermion_path_integral_dn, holstein_parameters_dn, x, -1)
    end
    if calculate_exp_K
        update!(fermion_path_integral_up, ssh_parameters_up, x, -1)
        update!(fermion_path_integral_dn, ssh_parameters_dn, x, -1)
    end

    # apply expansion/contraction to phonon fields
    @. x′ = expγ * x′

    # update the fermion path integrals to reflect new phonon field configuration
    if calculate_exp_V
        update!(fermion_path_integral_up, holstein_parameters_up, x, +1)
        update!(fermion_path_integral_dn, holstein_parameters_dn, x, +1)
    end
    if calculate_exp_K
        update!(fermion_path_integral_up, ssh_parameters_up, x, +1)
        update!(fermion_path_integral_dn, ssh_parameters_dn, x, +1)
    end

    # update the spin up and spin down propagators to reflect current phonon configuration
    calculate_propagators!(Bup, fermion_path_integral_up, calculate_exp_K = calculate_exp_K, calculate_exp_V = calculate_exp_V)
    calculate_propagators!(Bdn, fermion_path_integral_dn, calculate_exp_K = calculate_exp_K, calculate_exp_V = calculate_exp_V)

    # update the Green's function to reflect the new phonon configuration
    logdetGup′, sgndetGup′ = logdetGup, sgndetGup
    logdetGdn′, sgndetGdn′ = logdetGdn, sgndetGdn
    try
        logdetGup′, sgndetGup′ = calculate_equaltime_greens!(Gup′, fermion_greens_calculator_up_alt, Bup)
        logdetGdn′, sgndetGdn′ = calculate_equaltime_greens!(Gdn′, fermion_greens_calculator_dn_alt, Bdn)
    catch
        logdetGup′, sgndetGup′ = NaN, NaN
        logdetGdn′, sgndetGdn′ = NaN, NaN
    end

    # if finite fermionic determiantn
    if isfinite(logdetGup′) && isfinite(logdetGdn′)

        # calculate the final bosonic action
        Sb′ = bosonic_action(electron_phonon_parameters)

        # calculate final fermionci action
        Sf′ = logdetGup′ + logdetGdn′

        # calculate final action
        S′ = Sb′ + Sf′

        # calculate the change in action
        ΔS = S′ - S

        # calculate final acceptance rate
        P_γ = min(1.0, exp(-ΔS + d*γ))
    else
        P_γ = 0.0
    end

    # accept or reject the update
    if rand(rng) < P_γ
        logdetGup = logdetGup′
        logdetGdn = logdetGdn′
        sgndetGup = sgndetGup′
        sgndetGdn = sgndetGdn′
        copyto!(Gup, Gup′)
        copyto!(Gdn, Gdn′)
        copyto!(fermion_greens_calculator_up, fermion_greens_calculator_up_alt)
        copyto!(fermion_greens_calculator_dn, fermion_greens_calculator_dn_alt)
        accepted = true
    else
        # substract off the effect of the current phonon configuration on the fermion path integrals
        if calculate_exp_V
            update!(fermion_path_integral_up, holstein_parameters_up, x, -1)
            update!(fermion_path_integral_dn, holstein_parameters_dn, x, -1)
        end
        if calculate_exp_K
            update!(fermion_path_integral_up, ssh_parameters_up, x, -1)
            update!(fermion_path_integral_dn, ssh_parameters_dn, x, -1)
        end
        # revert to the original phonon configuration
        @. x′ = x′ / expγ
        # update the fermion path integrals to reflect new phonon field configuration
        if calculate_exp_V
            update!(fermion_path_integral_up, holstein_parameters_up, x, +1)
            update!(fermion_path_integral_dn, holstein_parameters_dn, x, +1)
        end
        if calculate_exp_K
            update!(fermion_path_integral_up, ssh_parameters_up, x, +1)
            update!(fermion_path_integral_dn, ssh_parameters_dn, x, +1)
        end
        # update the fermion path integrals to reflect the original phonon configuration
        calculate_propagators!(Bup, fermion_path_integral_up, calculate_exp_K = calculate_exp_K, calculate_exp_V = calculate_exp_V)
        calculate_propagators!(Bdn, fermion_path_integral_dn, calculate_exp_K = calculate_exp_K, calculate_exp_V = calculate_exp_V)
        accepted = false
    end

    return (accepted, logdetGup, sgndetGup, logdetGdn, sgndetGdn)
end


@doc raw"""
    radial_update!(
        # ARGUMENTS
        G::Matrix{T}, logdetG::E, sgndetG::T,
        electron_phonon_parameters::ElectronPhononParameters{T,E};
        # KEYWORD ARGUMENTS
        fermion_path_integral::FermionPathIntegral{T,E},
        fermion_greens_calculator::FermionGreensCalculator{T,E},
        fermion_greens_calculator_alt::FermionGreensCalculator{T,E},
        B::Vector{P}, rng::AbstractRNG,
        phonon_id::Union{Int, Nothing} = nothing,
        σ::E = 1.0
    ) where {T<:Number, E<:AbstractFloat, P<:AbstractPropagator{T,E}}

Perform a radial update to the phonon fields, as described by Algorithm 1 in the paper
[arXiv:2411.18218](https://arxiv.org/abs/2411.18218).
Specifically, the proposed update to the phonon fields ``x`` is a rescaling such that
``x \rightarrow e^{\gamma} x`` where ``\gamma \sim N(0, \sigma/\sqrt{d})`` and ``d`` is
the number of phonon fields being updated.

# Arguments

- `G::Matrix{T}`: Eqaul-time Greens function matrix.
- `logdetG::E`: Log of the determinant of the eqaul-time Greens function matrix.
- `sgndetG::T`: Sign/phase of the determinant of the eqaul-time Greens function matrix.
- `electron_phonon_parameters::ElectronPhononParameters{T,E}`: Electron-phonon parameters, including the current phonon configuration.

# Keyword Arguments

- `fermion_path_integral::FermionPathIntegral{T,E}`: An instance of [`FermionPathIntegral`](@ref) type.
- `fermion_greens_calculator::FermionGreensCalculator{T,E}`: Contains matrix factorization information for current state.
- `fermion_greens_calculator_alt::FermionGreensCalculator{T,E}`: Used to calculate matrix factorizations for proposed state.
- `B::Vector{P}`: Propagators for each imaginary time slice.
- `rng::AbstractRNG`: Random number generator used in method instead of global random number generator, important for reproducibility.
- `phonon_id::Union{Int, Nothing} = nothing`: Apply radial update to phonon fields corresponding tp specified `PHONON_ID`. If `phonon_id = nothing`, then radial update is applied to all phonon fields.
- `σ::E = 1.0`: Relative size of the radial update.
"""
function radial_update!(
    # ARGUMENTS
    G::Matrix{T}, logdetG::E, sgndetG::T,
    electron_phonon_parameters::ElectronPhononParameters{T,E};
    # KEYWORD ARGUMENTS
    fermion_path_integral::FermionPathIntegral{T,E},
    fermion_greens_calculator::FermionGreensCalculator{T,E},
    fermion_greens_calculator_alt::FermionGreensCalculator{T,E},
    B::Vector{P}, rng::AbstractRNG,
    phonon_id::Union{Int, Nothing} = nothing,
    σ::E = 1.0
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

    # get the number of phonon modes per unit cell
    nphonon = phonon_parameters.nphonon

    # total number of phonon modes
    Nphonon = phonon_parameters.Nphonon

    # number of unit cells
    Nunitcells = Nphonon ÷ nphonon

    # whether the exponentiated on-site energy matrix needs to be updated with the phonon field,
    # true if phonon mode appears in holstein coupling
    calculate_exp_V = (holstein_parameters.nholstein > 0)

    # whether the exponentiated hopping matrix needs to be updated with the phonon field,
    # true if phonon mode appears in SSH coupling
    calculate_exp_K = (ssh_parameters.nssh > 0)

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
    σR = σ / sqrt(d)

    # randomly sample expansion/contraction coefficient
    γ = randn(rng) * σR
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
    logdetG′, sgndetG′ = logdetG, sgndetG
    try
        logdetG′, sgndetG′ = calculate_equaltime_greens!(G′, fermion_greens_calculator_alt, B)
    catch
        logdetG′, sgndetG′ = NaN, NaN
    end

    # if finite fermionic determiantn
    if isfinite(logdetG′)

        # calculate the final bosonic action
        Sb′ = bosonic_action(electron_phonon_parameters)

        # calculate final fermionci action
        Sf′ = 2*logdetG′

        # calculate final action
        S′ = Sb′ + Sf′

        # calculate the change in action
        ΔS = S′ - S

        # calculate final acceptance rate
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