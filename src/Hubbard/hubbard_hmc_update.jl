function lmc_update!(
    Gup::Matrix{T}, logdetGup::E, sgndetGup::E,
    Gdn::Matrix{T}, logdetGdn::E, sgndetGdn::E,
    hubbard_hs_parameters::AbstractHubbardHS{E};
    Δt::E,
    fermion_path_integral_up::FermionPathIntegral{T,E},
    fermion_path_integral_dn::FermionPathIntegral{T,E},
    fermion_greens_calculator_up::FermionGreensCalculator{T,E},
    fermion_greens_calculator_dn::FermionGreensCalculator{T,E},
    fermion_greens_calculator_up_alt::FermionGreensCalculator{T,E},
    fermion_greens_calculator_dn_alt::FermionGreensCalculator{T,E},
    Bup::Vector{P}, Bdn::Vector{P},
    δG_max::E, δG::E, δθ::E, rng::AbstractRNG,
    initialize_force::Bool = true,
    δG_reject::E = 1e-2, hs_filter = I) where {T<:Number, E<:AbstractFloat, P<:AbstractPropagator{T,E}}

    # randomly sample step size
    Δt′ = Δt * randexp(rng)

    (accepted, logdetGup, sgndetGup, logdetGdn, sgndetGdn, δG, δθ) = _hmc_update!(
        Gup, logdetGup, sgndetGup, Gdn, logdetGdn, sgndetGdn,
        hubbard_hs_parameters, 1, Δt′,
        fermion_path_integral_up, fermion_path_integral_dn,
        fermion_greens_calculator_up, fermion_greens_calculator_dn,
        fermion_greens_calculator_up_alt, fermion_greens_calculator_dn_alt,
        Bup, Bdn, δG_max, δG, δθ, rng, initialize_force, δG_reject, hs_filter
    )

    return (accepted, logdetGup, sgndetGup, logdetGdn, sgndetGdn, δG, δθ)
end

function lmc_update!(
    G::Matrix{T}, logdetG::E, sgndetG::E,
    hubbard_hs_parameters::AbstractHubbardHS{E};
    Δt::E,
    fermion_path_integral::FermionPathIntegral{T,E},
    fermion_greens_calculator::FermionGreensCalculator{T,E},
    fermion_greens_calculator_alt::FermionGreensCalculator{T,E},
    B::Vector{P}, δG_max::E, δG::E, δθ::E, rng::AbstractRNG, initialize_force::Bool = true,
    δG_reject::E = 1e-2, hs_filter = I) where {T<:Number, E<:AbstractFloat, P<:AbstractPropagator{T,E}}

    # randomly sample step size
    Δt′ = Δt * randexp(rng)

    (accepted, logdetG, sgndetG, δG, δθ) = _hmc_update!(
        G, logdetG, sgndetG, hubbard_hs_parameters, 1, Δt′,
        fermion_path_integral, fermion_greens_calculator, fermion_greens_calculator_alt,
        B, δG_max, δG, δθ, rng, initialize_force, δG_reject, hs_filter
    )

    return (accepted, logdetG, sgndetG, δG, δθ)
end


function hmc_update!(
    Gup::Matrix{T}, logdetGup::E, sgndetGup::E,
    Gdn::Matrix{T}, logdetGdn::E, sgndetGdn::E,
    hubbard_hs_parameters::AbstractHubbardHS{E};
    Nt::Int, Δt::E,
    fermion_path_integral_up::FermionPathIntegral{T,E},
    fermion_path_integral_dn::FermionPathIntegral{T,E},
    fermion_greens_calculator_up::FermionGreensCalculator{T,E},
    fermion_greens_calculator_dn::FermionGreensCalculator{T,E},
    fermion_greens_calculator_up_alt::FermionGreensCalculator{T,E},
    fermion_greens_calculator_dn_alt::FermionGreensCalculator{T,E},
    Bup::Vector{P}, Bdn::Vector{P},
    δG_max::E, δG::E, δθ::E, rng::AbstractRNG,
    initialize_force::Bool = true,
    δG_reject::E = 1e-2, hs_filter = I) where {T<:Number, E<:AbstractFloat, P<:AbstractPropagator{T,E}}

    # sample the trajectory length from geometric distribution with mean given by Nt
    Nt′ = Nt > 1 ? floor(Int, log(rand())/log(1-1/Nt)) + 1 : 1

    (accepted, logdetGup, sgndetGup, logdetGdn, sgndetGdn, δG, δθ) = _hmc_update!(
        Gup, logdetGup, sgndetGup, Gdn, logdetGdn, sgndetGdn,
        hubbard_hs_parameters, Nt′, Δt,
        fermion_path_integral_up, fermion_path_integral_dn,
        fermion_greens_calculator_up, fermion_greens_calculator_dn,
        fermion_greens_calculator_up_alt, fermion_greens_calculator_dn_alt,
        Bup, Bdn, δG_max, δG, δθ, rng, initialize_force, δG_reject, hs_filter
    )

    return (accepted, logdetGup, sgndetGup, logdetGdn, sgndetGdn, δG, δθ)
end

function hmc_update!(
    G::Matrix{T}, logdetG::E, sgndetG::E,
    hubbard_hs_parameters::AbstractHubbardHS{E};
    Nt::Int, Δt::E,
    fermion_path_integral::FermionPathIntegral{T,E},
    fermion_greens_calculator::FermionGreensCalculator{T,E},
    fermion_greens_calculator_alt::FermionGreensCalculator{T,E},
    B::Vector{P}, δG_max::E, δG::E, δθ::E, rng::AbstractRNG, initialize_force::Bool = true,
    δG_reject::E = 1e-2, hs_filter = I) where {T<:Number, E<:AbstractFloat, P<:AbstractPropagator{T,E}}

    # sample the trajectory length from geometric distribution with mean given by Nt
    Nt′ = Nt > 1 ? floor(Int, log(rand())/log(1-1/Nt)) + 1 : 1

    (accepted, logdetG, sgndetG, δG, δθ) = _hmc_update!(
        G, logdetG, sgndetG, hubbard_hs_parameters, Nt′, Δt,
        fermion_path_integral, fermion_greens_calculator, fermion_greens_calculator_alt,
        B, δG_max, δG, δθ, rng, initialize_force, δG_reject, hs_filter
    )

    return (accepted, logdetG, sgndetG, δG, δθ)
end


# perform HMC update assuming two spin species
function _hmc_update!(
    Gup::Matrix{T}, logdetGup::E, sgndetGup::E,
    Gdn::Matrix{T}, logdetGdn::E, sgndetGdn::E,
    hubbard_hs_parameters::AbstractHubbardHS{E},
    Nt::Int, Δt::E,
    fermion_path_integral_up::FermionPathIntegral{T,E},
    fermion_path_integral_dn::FermionPathIntegral{T,E},
    fermion_greens_calculator_up::FermionGreensCalculator{T,E},
    fermion_greens_calculator_dn::FermionGreensCalculator{T,E},
    fermion_greens_calculator_up_alt::FermionGreensCalculator{T,E},
    fermion_greens_calculator_dn_alt::FermionGreensCalculator{T,E},
    Bup::Vector{P}, Bdn::Vector{P},
    δG_max::E, δG::E, δθ::E, rng::AbstractRNG,
    initialize_force::Bool = true,
    δG_reject::E = 1e-2, hs_filter = I) where {T<:Number, E<:AbstractFloat, P<:AbstractPropagator{T,E}}

    (; s, s0, v, dSds, dSds0) = hubbard_hs_parameters

    # calculate the initial force
    if initialize_force
        fill!(dSds, zero(E))
        (logdetGup, sgndetGup, logdetGdn, sgndetGdn, δG, δθ) = hubbard_action_derivative!(
            dSds, hubbard_hs_parameters, Gup, logdetGup, sgndetGup, Gdn, logdetGdn, sgndetGdn,
            δG, δθ, fermion_greens_calculator_up, fermion_greens_calculator_dn, Bup, Bdn
        )
    end

    # initialize fermion green's function matrices and their determinants determinants
    Gup′ = fermion_path_integral_up.K
    Gdn′ = fermion_path_integral_dn.K
    copyto!(Gup′, Gup)
    copyto!(Gdn′, Gdn)
    logdetGup′ = logdetGup
    sgndetGup′ = sgndetGup
    logdetGdn′ = logdetGdn
    sgndetGdn′ = sgndetGdn

    # initialize numerical error for hmc trajectory
    δG′ = δG

    # initialize the alternate fermion greens calculators
    copyto!(fermion_greens_calculator_up_alt, fermion_greens_calculator_up)
    copyto!(fermion_greens_calculator_dn_alt, fermion_greens_calculator_dn)

    # record initial phonon configuration
    copyto!(s0, s)

    # record initial force
    copyto!(dSds0, dSds)

    # initialize velocities v ~ N(0,1)
    randn!(v)
    apply_filter!(v, hs_filter)

    # calculate initial kinetic energy K = |v|²/2
    K = dot(v,v)/2

    # calculate initial fermionic action
    Sf = logdetGup′ + logdetGdn′

    # calculate initial bosonic action
    Sb = _bosonic_action(hubbard_hs_parameters)

    # calculate initial total action
    S = Sf + Sb

    # calculate initial total energy
    H = S + K

    # flag to indicate whether or not hmc trajectory remains numerically stable
    numerically_stable = true

    # iterate over timesteps
    for t in 1:Nt

        # calculate v(t+Δt/2) = v(t) - Δt/2⋅∂S/∂s(t)
        @. v = v - Δt/2 * dSds
        apply_filter!(v, hs_filter)

        # subtract off the effect of the current HS configuration from the fermion path integrals
        update!(fermion_path_integral_up, hubbard_hs_parameters, +1, -1)
        update!(fermion_path_integral_dn, hubbard_hs_parameters, -1, -1)

        # update the HS fields
        @. s = s + Δt * v
        apply_filter!(s, hs_filter)

        # update the fermion path integrals to reflect new HS configuration
        update!(fermion_path_integral_up, hubbard_hs_parameters, +1, +1)
        update!(fermion_path_integral_dn, hubbard_hs_parameters, -1, +1)

        # update the propagator matrices to reflect new HS configuration
        calculate_propagators!(Bup, fermion_path_integral_up, calculate_exp_K = false, calculate_exp_V = true)
        calculate_propagators!(Bdn, fermion_path_integral_dn, calculate_exp_K = false, calculate_exp_V = true)

        # update the Green's function to reflect the new phonon configuration
        logdetGup′, sgndetGup′ = calculate_equaltime_greens!(Gup′, fermion_greens_calculator_up_alt, Bup)
        logdetGdn′, sgndetGdn′ = calculate_equaltime_greens!(Gdn′, fermion_greens_calculator_dn_alt, Bdn)

        # if nan occurs terminate update
        if !isfinite(logdetGup′) || !isfinite(logdetGdn′)

            # record that numerically instability was encountered
            numerically_stable = false

            # terminate the HMC trajectory
            break
        end
        
        # re-calculate the forces according to the new HS configuration
        fill!(dSds, zero(E))
        (logdetGup′, sgndetGup′, logdetGdn′, sgndetGdn′, δG′, δθ) = hubbard_action_derivative!(
            dSds, hubbard_hs_parameters, Gup′, logdetGup′, sgndetGup′, Gdn′, logdetGdn′, sgndetGdn′,
            δG′, δθ, fermion_greens_calculator_up_alt, fermion_greens_calculator_dn_alt, Bup, Bdn
        )

        # if numerical error too large or nan occurs
        if (!isfinite(δG′)) || (!isfinite(logdetGup′)) || (!isfinite(logdetGdn′)) || (δG′ > δG_reject)

            # record that numerically instability was encountered
            numerically_stable = false

            # terminate the HMC trajectory
            break
        end

        # calculate v(t+Δt) = v(t+Δt/2) - Δt/2⋅∂S/∂s(t+Δt)
        @. v = v - Δt/2 * dSds
        apply_filter!(v, hs_filter)
    end

    # if numerically stable
    if numerically_stable

        # calculate the final kinetic energy
        K′ = dot(v,v)/2

        # calculate final fermionic action
        Sf′ = logdetGup′ + logdetGdn′

        # calculate initial bosonic action
        Sb′ = _bosonic_action(hubbard_hs_parameters)

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
        (updated, logdetGup, sgndetGup, logdetGdn, sgndetGdn, δG, δθ) = update_stabalization_frequency!(
            Gup, logdetGup, sgndetGup,
            Gdn, logdetGdn, sgndetGdn,
            fermion_greens_calculator_up = fermion_greens_calculator_up,
            fermion_greens_calculator_dn = fermion_greens_calculator_dn,
            Bup = Bup, Bdn = Bdn, δG = δG, δθ = δθ, δG_max = δG_max
        )
    # otherwise reject proposed update and revert to oringial phonon configuration
    else
        # subtract off effect of current HS fields from fermion path integrals
        update!(fermion_path_integral_up, hubbard_hs_parameters, +1, -1)
        update!(fermion_path_integral_dn, hubbard_hs_parameters, -1, -1)
        # revert to initial phonon configuration
        copyto!(s, s0)
        # revert fermion path integrals to reflect initial HS configuration
        update!(fermion_path_integral_up, hubbard_hs_parameters, +1, +1)
        update!(fermion_path_integral_dn, hubbard_hs_parameters, -1, +1)
        # update propagators to reflect initial phonon configuration
        calculate_propagators!(Bup, fermion_path_integral_up, calculate_exp_K = false, calculate_exp_V = true)
        calculate_propagators!(Bdn, fermion_path_integral_dn, calculate_exp_K = false, calculate_exp_V = true)
        # revert to oringial action derivative
        copyto!(dSds, dSds0)
    end

    return (accepted, logdetGup, sgndetGup, logdetGdn, sgndetGdn, δG, δθ)
end

# perform HMC update assuming single spin species
function _hmc_update!(
    G::Matrix{T}, logdetG::E, sgndetG::E,
    hubbard_hs_parameters::AbstractHubbardHS{E},
    Nt::Int, Δt::E,
    fermion_path_integral::FermionPathIntegral{T,E},
    fermion_greens_calculator::FermionGreensCalculator{T,E},
    fermion_greens_calculator_alt::FermionGreensCalculator{T,E},
    B::Vector{P}, δG_max::E, δG::E, δθ::E, rng::AbstractRNG, initialize_force::Bool = true,
    δG_reject::E = 1e-2, hs_filter = I) where {T<:Number, E<:AbstractFloat, P<:AbstractPropagator{T,E}}

    (; s, s0, v, dSds, dSds0) = hubbard_hs_parameters

    # calculate the initial force
    if initialize_force
        fill!(dSds, zero(E))
        (logdetG, sgndetG, δG, δθ) = hubbard_action_derivative!(
            dSds, hubbard_hs_parameters, G, logdetG, sgndetG,
            δG, δθ, fermion_greens_calculator, B
        )
    end

    # initialize fermion green's function matrices and their determinants determinants
    G′ = fermion_path_integral.K
    copyto!(G′, G)
    logdetG′ = logdetG
    sgndetG′ = sgndetG
    
    # initialize the alternate fermion greens calculators
    copyto!(fermion_greens_calculator_alt, fermion_greens_calculator)

    # record initial phonon configuration
    copyto!(s0, s)

    # record initial force
    copyto!(dSds0, dSds)

    # initialize velocities v ~ N(0,1)
    randn!(v)
    apply_filter!(v, hs_filter)

    # calculate initial kinetic energy K = |v|²/2
    K = dot(v,v)/2

    # calculate initial fermionic action
    Sf = 2*logdetG

    # calculate initial bosonic action
    Sb = _bosonic_action(hubbard_hs_parameters)

    # calculate initial total action
    S = Sf + Sb

    # calculate initial total energy
    H = S + K

    # flag to indicate whether or not hmc trajectory remains numerically stable
    numerically_stable = true

    # initialize numerical error for hmc trajectory
    δG′ = δG

    # iterate over timesteps
    for t in 1:Nt

        # calculate v(t+Δt/2) = v(t) - Δt/2⋅∂S/∂s(t)
        @. v = v - Δt/2 * dSds
        apply_filter!(v, hs_filter)

        # subtract off the effect of the current HS configuration from the fermion path integrals
        update!(fermion_path_integral, hubbard_hs_parameters, +1, -1)

        # update the HS fields
        @. s = s + Δt * v
        apply_filter!(s, hs_filter)

        # update the fermion path integrals to reflect new HS configuration
        update!(fermion_path_integral, hubbard_hs_parameters, +1, +1)

        # update the propagator matrices to reflect new HS configuration
        calculate_propagators!(B, fermion_path_integral, calculate_exp_K = false, calculate_exp_V = true)

        # update the Green's function to reflect the new phonon configuration
        logdetG′, sgndetG′ = calculate_equaltime_greens!(G′, fermion_greens_calculator_alt, B)

        # if nan occurs terminate update
        if !isfinite(logdetG′)

            # record that numerically instability was encountered
            numerically_stable = false

            # terminate the HMC trajectory
            break
        end
        
        # re-calculate the forces according to the new HS configuration
        fill!(dSds, zero(E))
        (logdetG′, sgndetG′, δG′, δθ) = hubbard_action_derivative!(
            dSds, hubbard_hs_parameters, G′, logdetG′, sgndetG′,
            δG′, δθ, fermion_greens_calculator_alt, Bup
        )

        # if numerical error too large or nan occurs
        if !isfinite(δG) || !isfinite(logdetG′)|| δG′ > δG_reject

            # record that numerically instability was encountered
            numerically_stable = false

            # terminate the HMC trajectory
            break
        end

        # calculate v(t+Δt) = v(t+Δt/2) - Δt/2⋅∂S/∂s(t+Δt)
        @. v = v - Δt/2 * dSds
        apply_filter!(v, hs_filter)
    end

    # if numerically stable
    if numerically_stable

        # calculate the final kinetic energy
        K′ = dot(v,v)/2

        # calculate final fermionic action
        Sf′ = 2*logdetG′

        # calculate initial bosonic action
        Sb′ = _bosonic_action(hubbard_hs_parameters)

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
        copyto!(G, G′)
        # record final green function determinant
        logdetG = logdetG′
        sgndetG = sgndetG′
        # record the final state of the fermion greens calculator
        copyto!(fermion_greens_calculator, fermion_greens_calculator_alt)
        # record numerical error associated up hmc update
        δG = δG′
        # update stabilization frequency if required
        (updated, logdetG, sgndetG, δG, δθ) = update_stabalization_frequency!(
            G, logdetG, sgndetG,
            fermion_greens_calculator = fermion_greens_calculator,
            B = B, δG = δG, δθ = δθ, δG_max = δG_max
        )
    # otherwise reject proposed update and revert to oringial phonon configuration
    else
        # subtract off effect of current HS fields from fermion path integrals
        update!(fermion_path_integral, hubbard_hs_parameters, +1, -1)
        # revert to initial phonon configuration
        copyto!(s, s0)
        # revert fermion path integrals to reflect initial HS configuration
        update!(fermion_path_integral_, hubbard_hs_parameters, +1, +1)
        # update propagators to reflect initial phonon configuration
        calculate_propagators!(B, fermion_path_integral, calculate_exp_K = false, calculate_exp_V = true)
        # revert to oringial action derivative
        copyto!(dSds, dSds0)
    end

    return (accepted, logdetG, sgndetG, δG, δθ)
end