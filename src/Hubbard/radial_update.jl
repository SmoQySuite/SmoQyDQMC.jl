function radial_update!(
    Gup::Matrix{T}, logdetGup::E, sgndetGup::T,
    Gdn::Matrix{T}, logdetGdn::E, sgndetGdn::T,
    hubbard_hs_parameters::AbstractHubbardHS{F};
    fermion_path_integral_up::FermionPathIntegral{T,E},
    fermion_path_integral_dn::FermionPathIntegral{T,E},
    fermion_greens_calculator_up::FermionGreensCalculator{T,E},
    fermion_greens_calculator_dn::FermionGreensCalculator{T,E},
    fermion_greens_calculator_up_alt::FermionGreensCalculator{T,E},
    fermion_greens_calculator_dn_alt::FermionGreensCalculator{T,E},
    Bup::Vector{P}, Bdn::Vector{P},
    rng::AbstractRNG, σ::E = 1.0
) where {F<:Number, T<:Number, E<:AbstractFloat, P<:AbstractPropagator{T,E}}

    (; s) = hubbard_hs_parameters

    # initialize the alternate fermion greens calculators
    copyto!(fermion_greens_calculator_up_alt, fermion_greens_calculator_up)
    copyto!(fermion_greens_calculator_dn_alt, fermion_greens_calculator_dn)

    # initialize fermion green's function matrices and their determinants determinants
    Gup′ = fermion_path_integral_up.K
    Gdn′ = fermion_path_integral_dn.K
    copyto!(Gup′, Gup)
    copyto!(Gdn′, Gdn)
    logdetGup′, sgndetGup′ = logdetGup, sgndetGup
    logdetGdn′, sgndetGdn′ = logdetGdn, sgndetGdn

    # get the number of HS fields
    d = length(s)

    # calculate standard deviation for normal distribution
    σR = σ / sqrt(d)

    # randomly sample expansion/contraction coefficient
    γ = randn(rng) * σR
    expγ = exp(γ)

    # calculate initial fermionic action
    Sf = logdetGup′ + logdetGdn′

    # calculate initial bosonic action
    Sb = _bosonic_action(hubbard_hs_parameters)

    # calculate initial total action
    S = Sf + Sb

    # subtract off the effect of the current HS configuration from the fermion path integrals
    update!(fermion_path_integral_up, hubbard_hs_parameters, +1, -1)
    update!(fermion_path_integral_dn, hubbard_hs_parameters, -1, -1)

    # apply expansion/contraction to phonon fields
    @. s = expγ * s

    # update the fermion path integrals to reflect new HS configuration
    update!(fermion_path_integral_up, hubbard_hs_parameters, +1, +1)
    update!(fermion_path_integral_dn, hubbard_hs_parameters, -1, +1)

    # update the propagator matrices to reflect new HS configuration
    calculate_propagators!(Bup, fermion_path_integral_up, calculate_exp_K = false, calculate_exp_V = true)
    calculate_propagators!(Bdn, fermion_path_integral_dn, calculate_exp_K = false, calculate_exp_V = true)

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
        Sb′ = _bosonic_action(hubbard_hs_parameters)

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
        # subtract off the effect of the current HS configuration from the fermion path integrals
        update!(fermion_path_integral_up, hubbard_hs_parameters, +1, -1)
        update!(fermion_path_integral_dn, hubbard_hs_parameters, -1, -1)
        # revert to the original phonon configuration
        @. s = s / expγ
        # update the fermion path integrals to reflect new HS configuration
        update!(fermion_path_integral_up, hubbard_hs_parameters, +1, +1)
        update!(fermion_path_integral_dn, hubbard_hs_parameters, -1, +1)
        # update the fermion path integrals to reflect the original phonon configuration
        calculate_propagators!(Bup, fermion_path_integral_up, calculate_exp_K = false, calculate_exp_V = true)
        calculate_propagators!(Bdn, fermion_path_integral_dn, calculate_exp_K = false, calculate_exp_V = true)
        accepted = false
    end

    return (accepted, logdetGup, sgndetGup, logdetGdn, sgndetGdn)
end