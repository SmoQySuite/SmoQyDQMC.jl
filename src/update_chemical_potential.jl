@doc raw"""
    update_chemical_potential!(
        # ARGUMENTS
        Gup::Matrix{H}, logdetGup::R, sgndetGup::H,
        Gdn::Matrix{H}, logdetGdn::R, sgndetGdn::H;
        # KEYWORD ARGUMENTS
        chemical_potential_tuner::MuTunerLogger{R,H},
        tight_binding_parameters::Union{TightBindingParameters, Nothing} = nothing,
        tight_binding_parameters_up::Union{TightBindingParameters, Nothing} = nothing,
        tight_binding_parameters_dn::Union{TightBindingParameters, Nothing} = nothing,
        fermion_path_integral_up::FermionPathIntegral{H},
        fermion_path_integral_dn::FermionPathIntegral{H},
        fermion_greens_calculator_up::FermionGreensCalculator{H},
        fermion_greens_calculator_dn::FermionGreensCalculator{H},
        Bup::Vector{P}, Bdn::Vector{P}
    ) where {H<:Number, R<:AbstractFloat, P<:AbstractPropagator}

Update the chemical potential ``\mu`` in the simulation to approach the target density/filling.
This method returns the new values for `(logdetGup, sgndetGup, logdetGup, sgndetGup)`.
Note that either the keywork `tight_binding_parameters` needs to be specified, or
`tight_binding_parameters_up` and `tight_binding_parameters_dn` both need to be specified.
"""
function update_chemical_potential!(
    # ARGUMENTS
    Gup::Matrix{H}, logdetGup::R, sgndetGup::H,
    Gdn::Matrix{H}, logdetGdn::R, sgndetGdn::H;
    # KEYWORD ARGUMENTS
    chemical_potential_tuner::MuTunerLogger{R,H},
    tight_binding_parameters::Union{TightBindingParameters, Nothing} = nothing,
    tight_binding_parameters_up::Union{TightBindingParameters, Nothing} = nothing,
    tight_binding_parameters_dn::Union{TightBindingParameters, Nothing} = nothing,
    fermion_path_integral_up::FermionPathIntegral{H},
    fermion_path_integral_dn::FermionPathIntegral{H},
    fermion_greens_calculator_up::FermionGreensCalculator{H},
    fermion_greens_calculator_dn::FermionGreensCalculator{H},
    Bup::Vector{P}, Bdn::Vector{P}
) where {H<:Number, R<:AbstractFloat, P<:AbstractPropagator}

    # set up and down tight binding parameters if symmetric
    if !isnothing(tight_binding_parameters)
        tight_binding_parameters_up = tight_binding_parameters
        tight_binding_parameters_dn = tight_binding_parameters
    end

    # record the initial chemical potential
    @assert tight_binding_parameters_up.μ == tight_binding_parameters_dn.μ
    μ′ = tight_binding_parameters_up.μ

    # calculate sign
    @assert fermion_path_integral_up.Sb == fermion_path_integral_dn.Sb "$(fermion_path_integral_up.Sb) ≠ $(fermion_path_integral_dn.Sb)"
    Sb = fermion_path_integral_up
    sgn = isreal(Sb) ? sign(inv(sgndetGup) * inv(sgndetGdn)) : sign(exp(-1im*imag(Sb)) * inv(sgndetGup) * inv(sgndetGdn))

    # calculate average density
    nup = measure_n(Gup)
    ndn = measure_n(Gdn)
    n = nup + ndn

    # calculate ⟨N²⟩
    N² = measure_Nsqrd(Gup, Gdn)

    # update the chemical potential
    μ = MuTuner.update!(μtuner=chemical_potential_tuner, n=n, N²=N², s=sgn)

    # update tight binding parameter chemical potential
    tight_binding_parameters_up.μ = μ
    tight_binding_parameters_dn.μ = μ

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
    update_chemical_potential!(
        # ARGUMENTS
        G::Matrix{H}, logdetG::R, sgndetG::H;
        # KEYWORD ARGUMENTS
        chemical_potential_tuner::MuTunerLogger{R,H},
        tight_binding_parameters::TightBindingParameters,
        fermion_path_integral::FermionPathIntegral{H},
        fermion_greens_calculator::FermionGreensCalculator{H},
        B::Vector{P}
    ) where {H<:Number, R<:AbstractFloat, P<:AbstractPropagator}

Update the chemical potential ``\mu`` in the simulation to approach the target density/filling.
This method returns the new values for `(logdetG, sgndetG)`.
"""
function update_chemical_potential!(
    # ARGUMENTS
    G::Matrix{H}, logdetG::R, sgndetG::H;
    # KEYWORD ARGUMENTS
    chemical_potential_tuner::MuTunerLogger{R,H},
    tight_binding_parameters::TightBindingParameters,
    fermion_path_integral::FermionPathIntegral{H},
    fermion_greens_calculator::FermionGreensCalculator{H},
    B::Vector{P}
) where {H<:Number, R<:AbstractFloat, P<:AbstractPropagator}

    # record the initial chemical potential
    μ′ = tight_binding_parameters.μ

    # calculate sign
    Sb = fermion_path_integral.Sb
    sgn = isreal(Sb) ? sign(inv(sgndetG)^2) : sign(exp(-1im*imag(Sb)) * inv(sgndetG)^2)

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
    save_density_tuning_profile(
        # ARGUMENTS
        simulation_info::SimulationInfo,
        chemical_potential_tuner::MuTunerLogger{R, H};
        # KEYWORD ARGUMENTS
        export_to_h5::Bool = true,
        export_to_csv::Bool = false,
        scientific_notation::Bool = false,
        decimals::Int = 9,
        delimiter::String = " ",
    ) where {R<:AbstractFloat, H<:Number}

Record the history of chemical potential and density tuning that occured during the simulation,
writing the information to an HDF5 and/or CSV file.
"""
function save_density_tuning_profile(
    # ARGUMENTS
    simulation_info::SimulationInfo,
    chemical_potential_tuner::MuTunerLogger{R, H};
    # KEYWORD ARGUMENTS
    export_to_h5::Bool = true,
    export_to_csv::Bool = false,
    scientific_notation::Bool = false,
    decimals::Int = 9,
    delimiter::String = " ",
) where {R<:AbstractFloat, H<:Number}

    # save density tuning profile to HDF5 file
    if export_to_h5
        _save_density_tuning_profile_to_h5(
            simulation_info,
            chemical_potential_tuner
        )
    end

    # save the density tuning profile to CVS file
    if export_to_csv
        _save_density_tuning_profile_to_csv(
            simulation_info,
            chemical_potential_tuner,
            scientific_notation,
            decimals,
            delimiter
        )
    end

    return nothing
end

# save density tuning profile to HDF5 file
function _save_density_tuning_profile_to_h5(
    simulation_info::SimulationInfo,
    μtuner::MuTunerLogger{T, S}
) where {T<:AbstractFloat, S<:Number}

    (; datafolder, pID) = simulation_info

    # construct filename
    filename = joinpath(
        datafolder,
        @sprintf("density_tuning_profile_pID-%d.h5", pID)
    )

    # open HDF5 file
    h5open(filename, "w") do H5File

        # recording algorithm parameters
        (; β, n₀, V, u₀) = μtuner
        H5File["beta"] = β
        H5File["target_density"] = n₀
        H5File["system_size"] = V
        H5File["intensive_energy_scale"] = u₀
        
        # trajectory length
        Nt = length(μtuner.μ_traj)

        # create group to contain tuning history
        History = create_group(H5File, "HISTORY")

        # allocate HDF5 file
        mu_traj         = create_dataset(History, "chemical_potential",     T, (Nt,))
        n_traj          = create_dataset(History, "density",                T, (Nt,))
        Nsqrd_traj      = create_dataset(History, "N_sqrd",                 S, (Nt,))
        sign_traj       = create_dataset(History, "sign",                   S, (Nt,))
        mu_bar_traj     = create_dataset(History, "chemical_potential_avg", T, (Nt,))
        mu_var_traj     = create_dataset(History, "chemical_potential_var", T, (Nt,))
        n_bar_traj      = create_dataset(History, "density_avg",            S, (Nt,))
        N_var_traj      = create_dataset(History, "N_var",                  T, (Nt,))
        Nsqrd_avg_traj  = create_dataset(History, "N_sqrd_avg",             S, (Nt,))
        kappa_bar_traj  = create_dataset(History, "compressibility_avg",    T, (Nt,))
        kappa_fluc_traj = create_dataset(History, "compressibility_fluc",   T, (Nt,))
        kappa_min_traj  = create_dataset(History, "compressibility_min",    T, (Nt,))
        kappa_max_traj  = create_dataset(History, "compressibility_max",    T, (Nt,))
        sign_avg_traj   = create_dataset(History, "sign_avg",               S, (Nt,))
        sign_var_traj   = create_dataset(History, "sign_var",               S, (Nt,))

        # iterate over history
        for t in 0:(Nt-1)

            # update the chemical potential based on the latest measurements
            (μ_tp1, μtuner.μ_bar, μtuner.μ_var, μtuner.N_bar, μtuner.N_var, μtuner.s_bar, μtuner.s_var,
            μtuner.N²_bar, μtuner.κ_bar, κ_fluc, κ_min, κ_max) = MuTuner._update!(μtuner, t)

            # record values
            mu_traj[t+1]         = μtuner.μ_traj[t+1]
            n_traj[t+1]          = μtuner.N_traj[t+1]/V
            Nsqrd_traj[t+1]      = μtuner.N²_traj[t+1]
            sign_traj[t+1]       = μtuner.s_traj[t+1]
            mu_bar_traj[t+1]     = μtuner.μ_bar
            mu_var_traj[t+1]     = μtuner.μ_var
            n_bar_traj[t+1]      = μtuner.N_bar/V
            N_var_traj[t+1]      = μtuner.N_var
            Nsqrd_avg_traj[t+1]  = μtuner.N²_bar
            kappa_bar_traj[t+1]  = μtuner.κ_bar
            kappa_fluc_traj[t+1] = κ_fluc
            kappa_min_traj[t+1]  = κ_min
            kappa_max_traj[t+1]  = κ_max
            sign_avg_traj[t+1]   = μtuner.s_bar
            sign_var_traj[t+1]   = μtuner.s_var
        end
    end

    return nothing
end

function _save_density_tuning_profile_to_csv(
    simulation_info::SimulationInfo,
    μtuner::MuTunerLogger{T, S},
    scientific_notation::Bool = false,
    decimals::Int = 9,
    delimiter::String = " ",
) where {T<:AbstractFloat, S<:Number}

    (; datafolder, pID) = simulation_info

    # construct filename
    filename = joinpath(
        datafolder,
        @sprintf("density_tuning_profile_pID-%d.csv", pID)
    )

    # initialize formatter for string to number
    formatter = num_to_string_formatter(decimals, scientific_notation)

    # open CSV file
    open(filename, "w") do CSVFile

        # write header to file
        join(
            CSVFile,
            [
                "chemical_potential", "density", "N_sqrd_real", "N_sqrd_imag", "sign_real", "sign_imag",
                "chemical_potential_avg", "chemical_potential_var",
                "density_avg_real", "density_avg_imag", "N_var",
                "N_sqrd_avg_real", "N_sqrd_avg_imag",
                "compressibility_avg", "compressibility_fluc", "compressibility_min", "compressibility_max",
                "sign_avg_real", "sign_avg_imag", "sign_var_real", "sign_var_imag"
            ],
            delimiter
        )
        write(CSVFile, "\n")

        # trajectory length
        Nt = length(μtuner.μ_traj)

        # get system size
        V = μtuner.V

        # iterate over history
        for t in 0:(Nt-1)

            # update the chemical potential based on the latest measurements
            (μ_tp1, μtuner.μ_bar, μtuner.μ_var, μtuner.N_bar, μtuner.N_var, μtuner.s_bar, μtuner.s_var,
            μtuner.N²_bar, μtuner.κ_bar, κ_fluc, κ_min, κ_max) = MuTuner._update!(μtuner, t)

            # write data to CSV file
            join(
                CSVFile,
                formatter.((
                    μtuner.μ_traj[t+1], μtuner.N_traj[t+1]/V, real(μtuner.N²_traj[t+1]), imag(μtuner.N²_traj[t+1]),
                    real(μtuner.s_traj[t+1]), imag(μtuner.s_traj[t+1]),
                    μtuner.μ_bar, μtuner.μ_var, real(μtuner.N_bar/V), imag(μtuner.N_bar/V), μtuner.N_var,
                    real(μtuner.N²_bar), imag(μtuner.N²_bar),
                    μtuner.κ_bar, κ_fluc, κ_min, κ_max,
                    real(μtuner.s_bar), imag(μtuner.s_bar), real(μtuner.s_var), imag(μtuner.s_var)
                )),
                delimiter
            )
            write(CSVFile, "\n")
        end
    end

    return nothing
end