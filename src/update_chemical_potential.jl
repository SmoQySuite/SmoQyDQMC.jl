@doc raw"""
    update_chemical_potential!(Gup::Matrix{T}, logdetGup::E, sgndetGup::T,
                               Gdn::Matrix{T}, logdetGdn::E, sgndetGdn::T;
                               chemical_potential_tuner::MuTunerLogger{E,T},
                               tight_binding_parameters::TightBindingParameters{T,E},
                               fermion_path_integral_up::FermionPathIntegral{T,E},
                               fermion_path_integral_dn::FermionPathIntegral{T,E},
                               fermion_greens_calculator_up::FermionGreensCalculator{T,E},
                               fermion_greens_calculator_dn::FermionGreensCalculator{T,E},
                               Bup::Vector{P}, Bdn::Vector{P}) where {T<:Number, E<:AbstractFloat, P<:AbstractPropagator{T,E}}

Update the chemical potential ``\mu`` in the simulation to approach the target density/filling.
This method returns the new values for `(logdetGup, sgndetGup, logdetGup, sgndetGup)`.
"""
function update_chemical_potential!(Gup::Matrix{T}, logdetGup::E, sgndetGup::T,
                                    Gdn::Matrix{T}, logdetGdn::E, sgndetGdn::T;
                                    chemical_potential_tuner::MuTunerLogger{E,T},
                                    tight_binding_parameters::TightBindingParameters{T,E},
                                    fermion_path_integral_up::FermionPathIntegral{T,E},
                                    fermion_path_integral_dn::FermionPathIntegral{T,E},
                                    fermion_greens_calculator_up::FermionGreensCalculator{T,E},
                                    fermion_greens_calculator_dn::FermionGreensCalculator{T,E},
                                    Bup::Vector{P}, Bdn::Vector{P}) where {T<:Number, E<:AbstractFloat, P<:AbstractPropagator{T,E}}

    # record the initial chemical potential
    μ′ = tight_binding_parameters.μ

    # calculate sign
    sgn = sgndetGup * sgndetGdn

    # calculate average density
    nup = measure_n(Gup)
    ndn = measure_n(Gdn)
    n = nup + ndn

    # calculate ⟨N²⟩
    N² = measure_Nsqrd(Gup, Gdn)

    # update the chemical potential
    μ = MuTuner.update!(μtuner=chemical_potential_tuner, n=n, N²=N², s=sgn)

    # update tight binding parameter chemical potential
    tight_binding_parameters.μ = μ

    # update fermion path integrals
    Vup = fermion_path_integral_up.V
    Vdn = fermion_path_integral_dn.V
    @. Vup += -μ + μ′
    @. Vdn += -μ + μ′

    # update/calculate propagator matrices
    calculate_propagators!(Bup, fermion_path_integral_up, calculate_exp_K = false, calculate_exp_V = true)
    calculate_propagators!(Bdn, fermion_path_integral_dn, calculate_exp_K = false, calculate_exp_V = true)

    # update the Green's function
    logdetGup, sgndetGup = calculate_equaltime_greens!(Gup, fermion_greens_calculator_up, Bup)
    logdetGdn, sgndetGdn = calculate_equaltime_greens!(Gdn, fermion_greens_calculator_dn, Bdn)

    return (logdetGup, sgndetGup, logdetGdn, sgndetGdn)
end

@doc raw"""
    update_chemical_potential!(G::Matrix{T}, logdetG::E, sgndetG::T;
                               chemical_potential_tuner::MuTunerLogger{E,T},
                               tight_binding_parameters::TightBindingParameters{T,E},
                               fermion_path_integral::FermionPathIntegral{T,E},
                               fermion_greens_calculator::FermionGreensCalculator{T,E},
                               B::Vector{P}) where {T<:Number, E<:AbstractFloat, P<:AbstractPropagator{T,E}}

Update the chemical potential ``\mu`` in the simulation to approach the target density/filling.
This method returns the new values for `(logdetG, sgndetG)`.
"""
function update_chemical_potential!(G::Matrix{T}, logdetG::E, sgndetG::T;
                                    chemical_potential_tuner::MuTunerLogger{E,T},
                                    tight_binding_parameters::TightBindingParameters{T,E},
                                    fermion_path_integral::FermionPathIntegral{T,E},
                                    fermion_greens_calculator::FermionGreensCalculator{T,E},
                                    B::Vector{P}) where {T<:Number, E<:AbstractFloat, P<:AbstractPropagator{T,E}}

    # record the initial chemical potential
    μ′ = tight_binding_parameters.μ

    # calculate sign
    sgn = one(E)

    # calculate average density
    n = 2 * measure_n(G)

    # calculate ⟨N²⟩
    Nsqrd = measure_Nsqrd(G, G)

    # update the chemical potential
    μ = MuTuner.update!(μtuner=chemical_potential_tuner, n=n, N²=Nsqrd, s=sgn)

    # update tight binding parameter chemical potential
    tight_binding_parameters.μ = μ

    # update fermion path integrals
    V = fermion_path_integral.V
    @. V += -μ + μ′

    # update/calculate propagator matrices
    calculate_propagators!(B, fermion_path_integral, calculate_exp_K = false, calculate_exp_V = true)

    # update the Green's function
    logdetG, sgndetG = calculate_equaltime_greens!(G, fermion_greens_calculator, B)

    return (logdetG, sgndetG)
end


@doc raw"""
    save_density_tuning_profile(simulation_info::SimulationInfo,
                                chemical_potential_tuner::MuTunerLogger{E,T}) where {E,T}

Write the full density tuning history to a CSV file, typically done at the end of a simulation.
"""
function save_density_tuning_profile(simulation_info::SimulationInfo,
                                     chemical_potential_tuner::MuTunerLogger{E,T}) where {E,T}

    # write the density tuning to file
    (; datafolder, pID) = simulation_info
    filename = @sprintf("density_tuning_profile_pID-%d.csv", pID)
    MuTuner.save(chemical_potential_tuner, filename, datafolder)

    return nothing
end