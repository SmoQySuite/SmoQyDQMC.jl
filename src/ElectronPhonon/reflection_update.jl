@doc raw"""
    reflection_update!(
        # ARGUMENTS
        Gup::Matrix{H}, logdetGup::R, sgndetGup::H,
        Gdn::Matrix{H}, logdetGdn::R, sgndetGdn::H,
        electron_phonon_parameters::ElectronPhononParameters{T,R};
        # KEYWORD ARGUMENTS
        fermion_path_integral_up::FermionPathIntegral{T,U},
        fermion_path_integral_dn::FermionPathIntegral{T,U},
        fermion_greens_calculator_up::FermionGreensCalculator{H,R},
        fermion_greens_calculator_dn::FermionGreensCalculator{H,R},
        fermion_greens_calculator_up_alt::FermionGreensCalculator{H,R},
        fermion_greens_calculator_dn_alt::FermionGreensCalculator{H,R},
        Bup::Vector{P}, Bdn::Vector{P}, rng::AbstractRNG,
        phonon_types = nothing
    ) where {H<:Number, T<:Number, U<:Number, R<:Real, P<:AbstractPropagator{T,U}}

Randomly sample a phonon mode in the lattice, and propose an update that reflects all the phonon fields associated with that phonon mode ``x \rightarrow -x.``
This function returns `(accepted, logdetGup, sgndetGup, logdetGdn, sgndetGdn)`.

# Arguments

- `Gup::Matrix{H}`: Spin-up equal-time Greens function matrix.
- `logdetGup::R`: Log of the determinant of the spin-up equal-time Greens function matrix.
- `sgndetGup::H`: Sign/phase of the determinant of the spin-up equal-time Greens function matrix.
- `Gdn::Matrix{H}`: Spin-down equal-time Greens function matrix.
- `logdetGdn::R`: Log of the determinant of the spin-down equal-time Greens function matrix.
- `sgndetGdn::H`: Sign/phase of the determinant of the spin-down equal-time Greens function matrix.
- `electron_phonon_parameters::ElectronPhononParameters{T,R}`: Electron-phonon parameters, including the current phonon configuration.

# Keyword Arguments

- `fermion_path_integral_up::FermionPathIntegral{T,U}`: An instance of [`FermionPathIntegral`](@ref) type for spin-up electrons.
- `fermion_path_integral_dn::FermionPathIntegral{T,U}`: An instance of [`FermionPathIntegral`](@ref) type for spin-down electrons.
- `fermion_greens_calculator_up::FermionGreensCalculator{H,R}`: Contains matrix factorization information for current spin-up sector state.
- `fermion_greens_calculator_dn::FermionGreensCalculator{H,R}`: Contains matrix factorization information for current spin-down sector state.
- `fermion_greens_calculator_up_alt::FermionGreensCalculator{H,R}`: Used to calculate matrix factorizations for proposed spin-up sector state.
- `fermion_greens_calculator_dn_alt::FermionGreensCalculator{H,R}`: Used to calculate matrix factorizations for proposed spin-up sector state.
- `Bup::Vector{P}`: Spin-up propagators for each imaginary time slice.
- `Bdn::Vector{P}`: Spin-down propagators for each imaginary time slice.
- `rng::AbstractRNG`: Random number generator used in method instead of global random number generator, important for reproducibility.
- `phonon_types = nothing`: Collection of phonon types in the unit cell to randomly sample a phonon mode from. If `nothing` then all phonon modes in the unit cell are considered.
"""
function reflection_update!(
    # ARGUMENTS
    Gup::Matrix{H}, logdetGup::R, sgndetGup::H,
    Gdn::Matrix{H}, logdetGdn::R, sgndetGdn::H,
    electron_phonon_parameters::ElectronPhononParameters{T,R};
    # KEYWORD ARGUMENTS
    fermion_path_integral_up::FermionPathIntegral{H,T},
    fermion_path_integral_dn::FermionPathIntegral{H,T},
    fermion_greens_calculator_up::FermionGreensCalculator{H,R},
    fermion_greens_calculator_dn::FermionGreensCalculator{H,R},
    fermion_greens_calculator_up_alt::FermionGreensCalculator{H,R},
    fermion_greens_calculator_dn_alt::FermionGreensCalculator{H,R},
    Bup::Vector{P}, Bdn::Vector{P}, rng::AbstractRNG,
    phonon_types = nothing
) where {H<:Number, T<:Number, R<:Real, P<:AbstractPropagator{T}}

    @assert fermion_path_integral_up.Sb == fermion_path_integral_dn.Sb "$(fermion_path_integral_up.Sb) ≠ $(fermion_path_integral_dn.Sb)"

    Gup′ = fermion_greens_calculator_up_alt.G′
    Gdn′ = fermion_greens_calculator_dn_alt.G′
    phonon_parameters = electron_phonon_parameters.phonon_parameters::PhononParameters{R}
    holstein_parameters_up = electron_phonon_parameters.holstein_parameters_up::HolsteinParameters{R}
    holstein_parameters_dn = electron_phonon_parameters.holstein_parameters_dn::HolsteinParameters{R}
    ssh_parameters_up = electron_phonon_parameters.ssh_parameters_up::SSHParameters{T}
    ssh_parameters_dn = electron_phonon_parameters.ssh_parameters_dn::SSHParameters{T}
    x = electron_phonon_parameters.x

    # make sure stabilization frequencies match
    if fermion_greens_calculator_up.n_stab != fermion_greens_calculator_up_alt.n_stab
        resize!(fermion_greens_calculator_up_alt, fermion_greens_calculator_up.n_stab)
    end

    # make sure stabilization frequencies match
    if fermion_greens_calculator_dn.n_stab != fermion_greens_calculator_dn_alt.n_stab
        resize!(fermion_greens_calculator_dn_alt, fermion_greens_calculator_dn.n_stab)
    end

    # get the mass associated with each phonon
    M = phonon_parameters.M

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
    calculate_exp_V = ((phonon_mode in holstein_parameters_up.coupling_to_phonon) ||
                       (phonon_mode in holstein_parameters_dn.coupling_to_phonon))

    # whether the exponentiated hopping matrix needs to be updated with the phonon field,
    # true if phonon mode appears in SSH coupling
    calculate_exp_K = ((phonon_mode in ssh_parameters_up.coupling_to_phonon) ||
                       (phonon_mode in ssh_parameters_dn.coupling_to_phonon))

    # get the corresponding phonon fields
    x_i = @view x[phonon_mode, :]

    # calculate the initial bosonic action
    Sb = bosonic_action(electron_phonon_parameters)

    # calculate initial ferimonic action
    Sf = logdetGup + logdetGdn

    # calculate the total initial action
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

    # reflection phonon fields for chosen mode
    @. x_i = -x_i

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

    # check of fermion determinants are finite
    if isfinite(logdetGup′) && isfinite(logdetGdn′)

        # calculate the final bosonic action
        Sb′ = bosonic_action(electron_phonon_parameters)

        # calculate final fermionic action
        Sf′ = logdetGup′ + logdetGdn′

        # calculate total final action
        S′ = Sb′ + Sf′

        # calculate final total action
        ΔS = S′ - S

        # calculate acceptance probability
        P_i = min(1.0, exp(-ΔS))
    else
        P_i = zero(R)
    end

    # accept or reject the update
    if rand(rng) < P_i
        logdetGup = logdetGup′
        logdetGdn = logdetGdn′
        sgndetGup = sgndetGup′
        sgndetGdn = sgndetGdn′
        copyto!(Gup, Gup′)
        copyto!(Gdn, Gdn′)
        copyto!(fermion_greens_calculator_up, fermion_greens_calculator_up_alt)
        copyto!(fermion_greens_calculator_dn, fermion_greens_calculator_dn_alt)
        ΔSb = Sb′ - Sb
        fermion_path_integral_up.Sb += ΔSb
        fermion_path_integral_dn.Sb += ΔSb
        fermion_path_integral_up.Sb
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
        @. x_i = -x_i
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
    reflection_update!(
        # ARGUMENTS
        G::Matrix{H}, logdetG::R, sgndetG::H,
        electron_phonon_parameters::ElectronPhononParameters{T,R};
        # KEYWORD ARGUMENTS
        fermion_path_integral::FermionPathIntegral{H,T},
        fermion_greens_calculator::FermionGreensCalculator{H,R},
        fermion_greens_calculator_alt::FermionGreensCalculator{H,R},
        B::Vector{P}, rng::AbstractRNG,
        phonon_types = nothing
    ) where {H<:Number, T<:Number, R<:Real, P<:AbstractPropagator{T}}

Randomly sample a phonon mode in the lattice, and propose an update that reflects all the phonon fields associated with that phonon mode ``x \rightarrow -x.``
This function returns `(accepted, logdetG, sgndetG)`.

# Arguments

- `G::Matrix{H}`: equal-time Greens function matrix.
- `logdetG::R`: Log of the determinant of the equal-time Greens function matrix.
- `sgndetG::H`: Sign/phase of the determinant of the equal-time Greens function matrix.
- `electron_phonon_parameters::ElectronPhonParameters{T,R}`: Electron-phonon parameters, including the current phonon configuration.

# Keyword Arguments

- `fermion_path_integral::FermionPathIntegral{H,T}`: An instance of [`FermionPathIntegral`](@ref) type.
- `fermion_greens_calculator::FermionGreensCalculator{H,R}`: Contains matrix factorization information for current state.
- `fermion_greens_calculator_alt::FermionGreensCalculator{H,R}`: Used to calculate matrix factorizations for proposed state.
- `B::Vector{P}`: Propagators for each imaginary time slice.
- `rng::AbstractRNG`: Random number generator used in method instead of global random number generator, important for reproducibility.
- `phonon_types = nothing`: Collection of phonon types (specified my `PHONON_ID`) in the unit cell to randomly sample a phonon mode from. If `nothing` then all phonon modes in the unit cell are considered.
"""
function reflection_update!(
    # ARGUMENTS
    G::Matrix{H}, logdetG::R, sgndetG::H,
    electron_phonon_parameters::ElectronPhononParameters{T,R};
    # KEYWORD ARGUMENTS
    fermion_path_integral::FermionPathIntegral{H,T},
    fermion_greens_calculator::FermionGreensCalculator{H,R},
    fermion_greens_calculator_alt::FermionGreensCalculator{H,R},
    B::Vector{P}, rng::AbstractRNG,
    phonon_types = nothing
) where {H<:Number, T<:Number, R<:Real, P<:AbstractPropagator{T}}

    G′ = fermion_greens_calculator_alt.G′
    phonon_parameters = electron_phonon_parameters.phonon_parameters::PhononParameters{R}
    holstein_parameters = electron_phonon_parameters.holstein_parameters_up::HolsteinParameters{R}
    ssh_parameters = electron_phonon_parameters.ssh_parameters_up::SSHParameters{T}
    x = electron_phonon_parameters.x

    # make sure stabilization frequencies match
    if fermion_greens_calculator.n_stab != fermion_greens_calculator_alt.n_stab
        resize!(fermion_greens_calculator_alt, fermion_greens_calculator.n_stab)
    end

    # get the mass associated with each phonon
    M = phonon_parameters.M

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

    # get the corresponding phonon fields
    x_i = @view x[phonon_mode, :]

    # calculate the initial bosonic action
    Sb = bosonic_action(electron_phonon_parameters)

    # calculate the initial fermionic action
    Sf = 2*logdetG

    # calculate the total initial action
    S = Sb + Sf

    # substract off the effect of the current phonon configuration on the fermion path integrals
    if calculate_exp_V
        update!(fermion_path_integral, holstein_parameters, x, -1)
    end
    if calculate_exp_K
        update!(fermion_path_integral, ssh_parameters, x, -1)
    end

    # reflection phonon fields for chosen mode
    @. x_i = -x_i

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
    
    # check if fermion determinant is finite
    if isfinite(logdetG′)

        # calculate the final bosonic action
        Sb′ = bosonic_action(electron_phonon_parameters)

        # calculate final fermionic action
        Sf′ = 2*logdetG′

        # calculate total final action
        S′ = Sb′ + Sf′

        # calculate the change in action
        ΔS = S′ - S

        # calculate acceptance probability
        P_i = min(1.0, exp(-ΔS))
    else

        P_i = zero(R)
    end

    # accept or reject the update
    if rand(rng) < P_i
        logdetG = logdetG′
        sgndetG = sgndetG′
        copyto!(G, G′)
        copyto!(fermion_greens_calculator, fermion_greens_calculator_alt)
        fermion_path_integral.Sb += (Sb′ - Sb)
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
        @. x_i = -x_i
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


# sample a single random phonon mode
function _sample_phonon_mode(rng::AbstractRNG, nphonon::Int, Nunitcells::Int, masses::Vector{T}, phonon_types = nothing) where {T<:AbstractFloat}

    # initialize phonon mode to zero
    phonon_mode = 0

    # initialize phonon mass
    mass = one(T)

    # if set of phonon types unit cell is not specified set to all phonon modes in unit cell
    if isnothing(phonon_types)
        phonon_types = 1:nphonon
    end

    # sample phonon mode with finite mass
    while iszero(phonon_mode) || isinf(mass)

        # randomly sample phonon type
        phonon_type = rand(rng, phonon_types)

        # randomly sample unit cell
        unit_cell = rand(rng, 1:Nunitcells)

        # get the phonon mode
        phonon_mode = (phonon_type - 1) * Nunitcells + unit_cell

        # get phonon mass
        mass = masses[phonon_mode]
    end

    return phonon_mode
end