@doc raw"""
    swap_update!(Gup::Matrix{T}, logdetGup::E, sgndetGup::T,
                 Gdn::Matrix{T}, logdetGdn::E, sgndetGdn::T,
                 electron_phonon_parameters::ElectronPhononParameters{T,E};
                 fermion_path_integral_up::FermionPathIntegral{T,E},
                 fermion_path_integral_dn::FermionPathIntegral{T,E},
                 fermion_greens_calculator_up::FermionGreensCalculator{T,E},
                 fermion_greens_calculator_dn::FermionGreensCalculator{T,E},
                 fermion_greens_calculator_up_alt::FermionGreensCalculator{T,E},
                 fermion_greens_calculator_dn_alt::FermionGreensCalculator{T,E},
                 Bup::Vector{P}, Bdn::Vector{P}, rng::AbstractRNG,
                 phonon_type_pairs = nothing) where {T<:Number, E<:AbstractFloat, P<:AbstractPropagator{T,E}}

Randomly sample a pairs of phonon modes and exchange the phonon fields associated with the pair of phonon modes.
This function returns `(accepted, logdetGup, sgndetGup, logdetGdn, sgndetGdn)`.

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
- `phonon_type_pairs = nothing`: Collection of phonon type pairs in the unit cell to randomly sample a phonon modes from. If `nothing` then all phonon mode pairs in the unit cell are considered.
"""
function swap_update!(Gup::Matrix{T}, logdetGup::E, sgndetGup::T,
                      Gdn::Matrix{T}, logdetGdn::E, sgndetGdn::T,
                      electron_phonon_parameters::ElectronPhononParameters{T,E};
                      fermion_path_integral_up::FermionPathIntegral{T,E},
                      fermion_path_integral_dn::FermionPathIntegral{T,E},
                      fermion_greens_calculator_up::FermionGreensCalculator{T,E},
                      fermion_greens_calculator_dn::FermionGreensCalculator{T,E},
                      fermion_greens_calculator_up_alt::FermionGreensCalculator{T,E},
                      fermion_greens_calculator_dn_alt::FermionGreensCalculator{T,E},
                      Bup::Vector{P}, Bdn::Vector{P}, rng::AbstractRNG,
                      phonon_type_pairs = nothing) where {T<:Number, E<:AbstractFloat, P<:AbstractPropagator{T,E}}

    Gup′ = fermion_greens_calculator_up_alt.G′
    Gdn′ = fermion_greens_calculator_dn_alt.G′
    phonon_parameters = electron_phonon_parameters.phonon_parameters::PhononParameters{E}
    holstein_parameters_up = electron_phonon_parameters.holstein_parameters_up::HolsteinParameters{E}
    holstein_parameters_dn = electron_phonon_parameters.holstein_parameters_dn::HolsteinParameters{E}
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
    phonon_mode_i, phonon_mode_j = _sample_phonon_mode_pair(rng, nphonon, Nunitcells, M, phonon_type_pairs)

    # whether the exponentiated on-site energy matrix needs to be updated with the phonon field,
    # true if phonon mode appears in holstein coupling
    calculate_exp_V = (phonon_mode_i in holstein_parameters_up.coupling_to_phonon) || (phonon_mode_j in holstein_parameters_up.coupling_to_phonon)

    # whether the exponentiated hopping matrix needs to be updated with the phonon field,
    # true if phonon mode appears in SSH coupling
    calculate_exp_K = (phonon_mode_i in ssh_parameters_dn.coupling_to_phonon) || (phonon_mode_j in ssh_parameters_dn.coupling_to_phonon)

    # get the corresponding phonon fields
    x_i = @view x[phonon_mode_i, :]
    x_j = @view x[phonon_mode_j, :]

    # calculate the initial bosonic action
    Sb = bosonic_action(electron_phonon_parameters)

    # substract off the effect of the current phonon configuration on the fermion path integrals
    if calculate_exp_V
        update!(fermion_path_integral_up, holstein_parameters_up, x, -1)
        update!(fermion_path_integral_dn, holstein_parameters_dn, x, -1)
    end
    if calculate_exp_K
        update!(fermion_path_integral_up, ssh_parameters_up, x, -1)
        update!(fermion_path_integral_dn, ssh_parameters_dn, x, -1)
    end

    # swap phonon fields
    swap!(x_i, x_j)

    # calculate the final bosonic action
    Sb′ = bosonic_action(electron_phonon_parameters)

    # caclulate the change in the bosonic action
    ΔSb = Sb′ - Sb

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
    logdetGup′, sgndetGup′ = calculate_equaltime_greens!(Gup′, fermion_greens_calculator_up_alt, Bup)
    logdetGdn′, sgndetGdn′ = calculate_equaltime_greens!(Gdn′, fermion_greens_calculator_dn_alt, Bdn)

    # calculate acceptance probability P = exp(-ΔS_b)⋅|det(Gup)/det(Gup′)|⋅|det(Gdn)/det(Gdn′)|
    #                                    = exp(-ΔS_b)⋅|det(Mup′)/det(Mup)|⋅|det(Mdn′)/det(Mdn)|
    if isfinite(logdetGup′) && isfinite(logdetGdn′)
        P_i = exp(-ΔSb + logdetGup + logdetGdn - logdetGup′ - logdetGdn′)
    else
        P_i = 0.0
    end

    # accept/reject outcome
    accepted = rand(rng) < P_i

    # accept or reject the update
    if accepted
        logdetGup = logdetGup′
        logdetGdn = logdetGdn′
        sgndetGup = sgndetGup′
        sgndetGdn = sgndetGdn′
        copyto!(Gup, Gup′)
        copyto!(Gdn, Gdn′)
        copyto!(fermion_greens_calculator_up, fermion_greens_calculator_up_alt)
        copyto!(fermion_greens_calculator_dn, fermion_greens_calculator_dn_alt)
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
        swap!(x_i, x_j)
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
    end

    return (accepted, logdetGup, sgndetGup, logdetGdn, sgndetGdn)
end

@doc raw"""
    swap_update!(G::Matrix{T}, logdetG::E, sgndetG::T,
                 electron_phonon_parameters::ElectronPhononParameters{T,E};
                 fermion_path_integral::FermionPathIntegral{T,E},
                 fermion_greens_calculator::FermionGreensCalculator{T,E},
                 fermion_greens_calculator_alt::FermionGreensCalculator{T,E},
                 B::Vector{P}, rng::AbstractRNG,
                 phonon_type_pairs = nothing) where {T<:Number, E<:AbstractFloat, P<:AbstractPropagator{T,E}}

Randomly sample a pairs of phonon modes and exchange the phonon fields associated with the pair of phonon modes.
This function returns `(accepted, logdetG, sgndetG)`.

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
- `phonon_type_pairs = nothing`: Collection of phonon type pairs in the unit cell to randomly sample a phonon modes from. If `nothing` then all phonon mode pairs in the unit cell are considered.
"""
function swap_update!(G::Matrix{T}, logdetG::E, sgndetG::T,
                      electron_phonon_parameters::ElectronPhononParameters{T,E};
                      fermion_path_integral::FermionPathIntegral{T,E},
                      fermion_greens_calculator::FermionGreensCalculator{T,E},
                      fermion_greens_calculator_alt::FermionGreensCalculator{T,E},
                      B::Vector{P}, rng::AbstractRNG,
                      phonon_type_pairs = nothing) where {T<:Number, E<:AbstractFloat, P<:AbstractPropagator{T,E}}

    G′ = fermion_greens_calculator_alt.G′
    phonon_parameters = electron_phonon_parameters.phonon_parameters::PhononParameters{E}
    holstein_parameters = electron_phonon_parameters.holstein_parameters_up::HolsteinParameters{E}
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
    phonon_mode_i, phonon_mode_j = _sample_phonon_mode_pair(rng, nphonon, Nunitcells, M, phonon_type_pairs)

    # whether the exponentiated on-site energy matrix needs to be updated with the phonon field,
    # true if phonon mode appears in holstein coupling
    calculate_exp_V = (phonon_mode_i in holstein_parameters.coupling_to_phonon) || (phonon_mode_j in holstein_parameters.coupling_to_phonon)

    # whether the exponentiated hopping matrix needs to be updated with the phonon field,
    # true if phonon mode appears in SSH coupling
    calculate_exp_K = (phonon_mode_i in ssh_parameters.coupling_to_phonon) || (phonon_mode_j in ssh_parameters.coupling_to_phonon)

    # get the corresponding phonon fields
    x_i = @view x[phonon_mode_i, :]
    x_j = @view x[phonon_mode_j, :]

    # calculate the initial bosonic action
    Sb = bosonic_action(electron_phonon_parameters)

    # substract off the effect of the current phonon configuration on the fermion path integrals
    if calculate_exp_V
        update!(fermion_path_integral, holstein_parameters, x, -1)
    end
    if calculate_exp_K
        update!(fermion_path_integral, ssh_parameters, x, -1)
    end

    # reflection phonon fields for chosen mode
    swap!(x_i, x_j)

    # calculate the final bosonic action
    Sb′ = bosonic_action(electron_phonon_parameters)

    # caclulate the change in the bosonic action
    ΔSb = Sb′ - Sb

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

    # calculate acceptance probability P = exp(-ΔS_b)⋅|det(G)/det(G′)|²
    #                                    = exp(-ΔS_b)⋅|det(M′)/det(M)|²
    if isfinite(logdetG′)
        P_i = exp(-ΔSb + 2*logdetG - 2*logdetG′)
    else
        P_i = 0.0
    end

    # accept/reject outcome
    accepted = rand(rng) < P_i

    # accept or reject the update
    if accepted
        logdetG = logdetG′
        sgndetG = sgndetG′
        copyto!(G, G′)
        copyto!(fermion_greens_calculator, fermion_greens_calculator_alt)
    else
        # substract off the effect of the current phonon configuration on the fermion path integrals
        if calculate_exp_V
            update!(fermion_path_integral, holstein_parameters, x, -1)
        end
        if calculate_exp_K
            update!(fermion_path_integral, ssh_parameters, x, -1)
        end
        # revert to the original phonon configuration
        swap!(x_i, x_j)
        # update the fermion path integrals to reflect new phonon field configuration
        if calculate_exp_V
            update!(fermion_path_integral, holstein_parameters, x, +1)
        end
        if calculate_exp_K
            update!(fermion_path_integral, ssh_parameters, x, +1)
        end
        # update the fermion path integrals to reflect the original phonon configuration
        calculate_propagators!(B, fermion_path_integral, calculate_exp_K = calculate_exp_K, calculate_exp_V = calculate_exp_V)
    end

    return (accepted, logdetG, sgndetG)
end


# sample a pair of random phonon modes
function _sample_phonon_mode_pair(rng::AbstractRNG, nphonon::Int, Nunitcells::Int, masses::Vector{T}, phonon_type_pairs = nothing) where {T<:AbstractFloat}

    # sample a pair of phonon types
    if isnothing(phonon_type_pairs)
        phonon_type_pair = ( rand(rng, 1:nphonon) , rand(rng, 1:nphonon) )
    else
        n = rand(rng, 1:length(phonon_type_pairs))
        phonon_type_pair = phonon_type_pairs[n]
    end
    
    return _sample_phonon_mode_pair(rng, nphonon, Nunitcells, masses, phonon_type_pair)
end

# sample a pair of random phonon modes
function _sample_phonon_mode_pair(rng::AbstractRNG, nphonon::Int, Nunitcells::Int, masses::Vector{T}, phonon_type_pair::NTuple{2,Int}) where {T<:AbstractFloat}

    # initialize phonon mode 1 to zero
    phonon_mode_1 = 0
    
    # initialize phonon mode 1 mass to zero
    mass = zero(T)

    # sample phonon mode 1
    while iszero(phonon_mode_1) || isinf(mass)

        # sample unit cell 1
        unit_cell_1 = rand(rng, 1:Nunitcells)

        # sample phonon mode 1
        phonon_mode_1 = (phonon_type_pair[1] - 1) * Nunitcells + unit_cell_1

        # get mass of phonon mode 1
        mass = masses[phonon_mode_1]
    end

    # initialize phonon mode 2 to phonon mode 1
    phonon_mode_2 = phonon_mode_1

    # sample phonon mode 2
    while phonon_mode_1 == phonon_mode_2 || isinf(mass)

        # sample unit cell 2
        unit_cell_2 = rand(rng, 1:Nunitcells)

        # sample phonon mode 2
        phonon_mode_2 = (phonon_type_pair[2] - 1) * Nunitcells + unit_cell_2

        # get mass of phonon mode 2
        mass = masses[phonon_mode_2]
    end

    return (phonon_mode_1, phonon_mode_2)
end