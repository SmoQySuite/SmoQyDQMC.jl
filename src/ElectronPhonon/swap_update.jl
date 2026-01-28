@doc raw"""
    swap_update!(
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
        phonon_id_pairs = nothing
    ) where {H<:Number, T<:Number, R<:Real, P<:AbstractPropagator{T}}

Randomly sample a pairs of phonon modes and exchange the phonon fields associated with the pair of phonon modes.
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

- `fermion_path_integral_up::FermionPathIntegral{H,T}`: An instance of [`FermionPathIntegral`](@ref) type for spin-up electrons.
- `fermion_path_integral_dn::FermionPathIntegral{H,T}`: An instance of [`FermionPathIntegral`](@ref) type for spin-down electrons.
- `fermion_greens_calculator_up::FermionGreensCalculator{H,R}`: Contains matrix factorization information for current spin-up sector state.
- `fermion_greens_calculator_dn::FermionGreensCalculator{H,R}`: Contains matrix factorization information for current spin-down sector state.
- `fermion_greens_calculator_up_alt::FermionGreensCalculator{H,R}`: Used to calculate matrix factorizations for proposed spin-up sector state.
- `fermion_greens_calculator_dn_alt::FermionGreensCalculator{H,R}`: Used to calculate matrix factorizations for proposed spin-up sector state.
- `Bup::Vector{P}`: Spin-up propagators for each imaginary time slice.
- `Bdn::Vector{P}`: Spin-down propagators for each imaginary time slice.
- `rng::AbstractRNG`: Random number generator used in method instead of global random number generator, important for reproducibility.
- `phonon_id_pairs = nothing`: Collection of phonon type pairs (specified by pairs of `PHONON_ID` values) in the unit cell to randomly sample a phonon modes from. If `nothing` then all phonon mode pairs in the unit cell are considered.
"""
function swap_update!(
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
    phonon_id_pairs = nothing
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
    @assert Nphonon > 1 "There is only one phonon mode in the lattice, therefore a swap update cannot be performed."

    # number of unit cells
    Nunitcells = Nphonon ÷ nphonon

    # sample random phonon mode
    phonon_mode_i, phonon_mode_j = _sample_phonon_mode_pair(rng, nphonon, Nunitcells, M, phonon_id_pairs)

    # whether the exponentiated on-site energy matrix needs to be updated with the phonon field,
    # true if phonon mode appears in holstein coupling
    calculate_exp_V = ((phonon_mode_i in holstein_parameters_up.coupling_to_phonon) ||
                       (phonon_mode_j in holstein_parameters_up.coupling_to_phonon) ||
                       (phonon_mode_i in holstein_parameters_dn.coupling_to_phonon) ||
                       (phonon_mode_j in holstein_parameters_dn.coupling_to_phonon))

    # whether the exponentiated hopping matrix needs to be updated with the phonon field,
    # true if phonon mode appears in SSH coupling
    calculate_exp_K = ((phonon_mode_i in ssh_parameters_up.coupling_to_phonon) ||
                       (phonon_mode_j in ssh_parameters_up.coupling_to_phonon) ||
                       (phonon_mode_i in ssh_parameters_dn.coupling_to_phonon) ||
                       (phonon_mode_j in ssh_parameters_dn.coupling_to_phonon))

    # get the corresponding phonon fields
    x_i = @view x[phonon_mode_i, :]
    x_j = @view x[phonon_mode_j, :]

    # calculate the initial bosonic action
    Sb = bosonic_action(electron_phonon_parameters)

    # calculate the initial fermionic action
    Sf = logdetGup + logdetGdn

    # calculate initial total action
    S = Sb + Sf

    # subtract off the effect of the current phonon configuration on the fermion path integrals
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

    # check the fermionic determinants are finite
    if isfinite(logdetGup′) && isfinite(logdetGdn′)

        # calculate the final bosonic action
        Sb′ = bosonic_action(electron_phonon_parameters)

        # calculate final fermionic action
        Sf′ = logdetGup′ + logdetGdn′

        # calculate final total action
        S′ = Sb′ + Sf′

        # calculate the change in the action
        ΔS = S′ - S

        # calculate acceptance probability
        P_i = min(1.0, exp(-ΔS))
    else
        P_i = zero(R)
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
        ΔSb = Sb′ - Sb
        fermion_path_integral_up.Sb += ΔSb
        fermion_path_integral_dn.Sb += ΔSb
    else
        # subtract off the effect of the current phonon configuration on the fermion path integrals
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
    swap_update!(
        # ARGUMENTS
        G::Matrix{H}, logdetG::R, sgndetG::H,
        electron_phonon_parameters::ElectronPhononParameters{T,R};
        # KEYWORD ARGUMENTS
        fermion_path_integral::FermionPathIntegral{H,T},
        fermion_greens_calculator::FermionGreensCalculator{H,R},
        fermion_greens_calculator_alt::FermionGreensCalculator{H,R},
        B::Vector{P}, rng::AbstractRNG,
        phonon_id_pairs = nothing
    ) where {H<:Number, T<:Number, R<:Real, P<:AbstractPropagator{T}}

Randomly sample a pairs of phonon modes and exchange the phonon fields associated with the pair of phonon modes.
This function returns `(accepted, logdetG, sgndetG)`.

# Arguments

- `G::Matrix{H}`: equal-time Greens function matrix.
- `logdetG::R`: Log of the determinant of the equal-time Greens function matrix.
- `sgndetG::H`: Sign/phase of the determinant of the equal-time Greens function matrix.
- `electron_phonon_parameters::ElectronPhononParameters{T,R}`: Electron-phonon parameters, including the current phonon configuration.

# Keyword Arguments

- `fermion_path_integral::FermionPathIntegral{H,T}`: An instance of [`FermionPathIntegral`](@ref) type.
- `fermion_greens_calculator::FermionGreensCalculator{H,R}`: Contains matrix factorization information for current state.
- `fermion_greens_calculator_alt::FermionGreensCalculator{H,R}`: Used to calculate matrix factorizations for proposed state.
- `B::Vector{P}`: Propagators for each imaginary time slice.
- `rng::AbstractRNG`: Random number generator used in method instead of global random number generator, important for reproducibility.
- `phonon_id_pairs = nothing`: Collection of phonon type pairs in the unit cell to randomly sample a phonon modes from. If `nothing` then all phonon mode pairs in the unit cell are considered.
"""
function swap_update!(
    # ARGUMENTS
    G::Matrix{H}, logdetG::R, sgndetG::H,
    electron_phonon_parameters::ElectronPhononParameters{T,R};
    # KEYWORD ARGUMENTS
    fermion_path_integral::FermionPathIntegral{H,T},
    fermion_greens_calculator::FermionGreensCalculator{H,R},
    fermion_greens_calculator_alt::FermionGreensCalculator{H,R},
    B::Vector{P}, rng::AbstractRNG,
    phonon_id_pairs = nothing
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
    @assert Nphonon > 1 "There is only one phonon mode in the lattice, therefore a swap update cannot be performed."

    # number of unit cells
    Nunitcells = Nphonon ÷ nphonon

    # sample random phonon mode
    phonon_mode_i, phonon_mode_j = _sample_phonon_mode_pair(rng, nphonon, Nunitcells, M, phonon_id_pairs)

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

    # calculate initial fermionic action
    Sf = 2*logdetG

    # calculate initial total action
    S = Sb + Sf

    # subtract off the effect of the current phonon configuration on the fermion path integrals
    if calculate_exp_V
        update!(fermion_path_integral, holstein_parameters, x, -1)
    end
    if calculate_exp_K
        update!(fermion_path_integral, ssh_parameters, x, -1)
    end

    # reflection phonon fields for chosen mode
    swap!(x_i, x_j)

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

    # check the fermionic determinant is finite
    if isfinite(logdetG′)

        # calculate the final bosonic action
        Sb′ = bosonic_action(electron_phonon_parameters)

        # calculate final fermionic action
        Sf′ = 2*logdetG′

        # calculate final total action
        S′ = Sb′ + Sf′

        # calculate the change in the action
        ΔS = S′ - S

        # calculate acceptance rate
        P_i = min(1.0, exp(-ΔS))
    else
        P_i = zero(R)
    end

    # accept/reject outcome
    accepted = rand(rng) < P_i

    # accept or reject the update
    if accepted
        logdetG = logdetG′
        sgndetG = sgndetG′
        copyto!(G, G′)
        copyto!(fermion_greens_calculator, fermion_greens_calculator_alt)
        fermion_path_integral.Sb += Sb′ - Sb
    else
        # subtract off the effect of the current phonon configuration on the fermion path integrals
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
function _sample_phonon_mode_pair(rng::AbstractRNG, nphonon::Int, Nunitcells::Int, masses::Vector{T}, phonon_id_pairs = nothing) where {T<:AbstractFloat}

    # if phonon id pairs not passed
    if isnothing(phonon_id_pairs)

        # determine which phonon IDs have finite and not infinite mass
        m = reshape(masses, Nunitcells, nphonon)
        phonon_ids = filter(n -> !isinf(m[1,n]), 1:nphonon)
        # sample a pair of phonon IDs
        phonon_id_pair = (rand(rng, phonon_ids), rand(rng, phonon_ids))
    else

        # sample one of the phonon ID pairs
        phonon_id_pair = rand(rng, phonon_id_pairs)
        phonon_id_pair = isa(phonon_id_pair, NTuple{2,Int}) ? phonon_id_pair : tuple(phonon_id_pair...)
    end

    # check to make sure phonon_id_pair is correct type
    @assert isa(phonon_id_pair, NTuple{2,Int}) "Each element of `phonon_id_pairs` must be convertible to a `Tuple{Int,Int}` type."
    
    return _sample_phonon_mode_pair(rng, nphonon, Nunitcells, masses, phonon_id_pair)
end

# sample a pair of random phonon modes
function _sample_phonon_mode_pair(rng::AbstractRNG, nphonon::Int, Nunitcells::Int, masses::Vector{T}, phonon_id_pair::NTuple{2,Int}) where {T<:AbstractFloat}

    # get the id pair
    id1, id2 = phonon_id_pair

    # if two phonon IDs are the same
    if id1 == id2

        # sample two phonon modes, making sure they are different
        index1, index2 = draw2(rng, Nunitcells)
        phonon1 = index1 + (id1-1) * Nunitcells
        phonon2 = index2 + (id2-1) * Nunitcells

    # if two phonon IDs are different
    else

        # sample two phonon modes
        phonon1 = rand(rng, 1:Nunitcells) + (id1-1) * Nunitcells
        phonon2 = rand(rng, 1:Nunitcells) + (id2-1) * Nunitcells
    end

    return (phonon1, phonon2)
end