# In this example we will work through an example for simulating the standard repulsive Hubbard model
# on a square lattice at half-filling.
#
# First let us start by importing the relevant packages, including [SmoQyDQMC](@ref)
# and it's relevant submodules.

using LinearAlgebra
using Random
using Printf

using SmoQyDQMC
import SmoQyDQMC.LatticeUtilities  as lu
import SmoQyDQMC.JDQMCFramework    as dqmcf
import SmoQyDQMC.JDQMCMeasurements as dqmcm
import SmoQyDQMC.MuTuner           as mt


# Next let us define a top-level function to that acutally runs our simulations.
# Let us include as function arguments standard parameters in the model we may want to change.

function run_hubbard_square_simulation(U, β, L)

    t = 1.0
    μ = 0.0

    datafolder_prefix = @sprintf "hubbard_square_U%.2f_mu%.2f_L%d_b%.2f" U μ L β

    simulation_info = SimulationInfo(
        filepath = ".",                     
        datafolder_prefix = datafolder_prefix,
        sID = 1
    )

    initialize_datafolder(simulation_info)

    seed = abs(rand(Int))
    rng = Xoshiro(seed)
    Δτ = 0.10
    Lτ = dqmcf.eval_length_imaginary_axis(β, Δτ)
    checkerboard = false
    symmetric = false
    n_stab = 10
    δG_max = 1e-6
    N_burnin = 2_000
    N_updates = 5_000
    N_bins = 100
    bin_size = div(N_updates, N_bins)
    additional_info = Dict(
        "dG_max" => δG_max,
        "N_burnin" => N_burnin,
        "N_updates" => N_updates,
        "N_bins" => N_bins,
        "bin_size" => bin_size,
        "local_acceptance_rate" => 0.0,
        "reflection_acceptance_rate" => 0.0,
        "n_stab_init" => n_stab,
        "symmetric" => symmetric,
        "checkerboard" => checkerboard,
        "seed" => seed,
    )

    unit_cell = lu.UnitCell(lattice_vecs = [[1.0, 0.0], [0.0, 1.0]],
                            basis_vecs   = [[0.0, 0.0]])

    lattice = lu.Lattice(
        L = [L, L],
        periodic = [true, true] # must be true for now
    )

    model_geometry = ModelGeometry(unit_cell, lattice)

    N = lu.nsites(unit_cell, lattice)

    bond_x = lu.Bond(orbitals = (1,1), displacement = [1,0])
    bond_x_id = add_bond!(model_geometry, bond_x)

    bond_y = lu.Bond(orbitals = (1,1), displacement = [0,1])
    bond_y_id = add_bond!(model_geometry, bond_y)

    tight_binding_model = TightBindingModel(
        model_geometry = model_geometry,
        t_bonds = [bond_x, bond_y],
        t_mean = [t, t],
        μ = μ,
        ϵ_mean = [0.]
    )

    hubbard_model = HubbardModel(
        shifted = false,
        U_orbital = [1],
        U_mean = [U],
    )

    model_summary(
        simulation_info = simulation_info,
        β = β, Δτ = Δτ,
        model_geometry = model_geometry,
        tight_binding_model = tight_binding_model,
        interactions = (hubbard_model,)
    )

    tight_binding_parameters = TightBindingParameters(
        tight_binding_model = tight_binding_model,
        model_geometry = model_geometry,
        rng = rng
    )

    hubbard_parameters = HubbardParameters(
        model_geometry = model_geometry,
        hubbard_model = hubbard_model,
        rng = rng
    )

    hubbard_ising_parameters = HubbardIsingHSParameters(
        β = β, Δτ = Δτ,
        hubbard_parameters = hubbard_parameters,
        rng = rng
    )

    measurement_container = initialize_measurement_container(model_geometry, β, Δτ)

    initialize_measurements!(measurement_container, tight_binding_model)

    initialize_measurements!(measurement_container, hubbard_model)

    initialize_correlation_measurements!(
        measurement_container = measurement_container,
        model_geometry = model_geometry,
        correlation = "greens_up",
        time_displaced = true,
        pairs = [(1, 1)]
    )

    initialize_correlation_measurements!(
        measurement_container = measurement_container,
        model_geometry = model_geometry,
        correlation = "greens_dn",
        time_displaced = true,
        pairs = [(1, 1)]
    )

    initialize_correlation_measurements!(
        measurement_container = measurement_container,
        model_geometry = model_geometry,
        correlation = "density",
        time_displaced = true,
        pairs = [(1, 1)]
    )

    initialize_correlation_measurements!(
        measurement_container = measurement_container,
        model_geometry = model_geometry,
        correlation = "pair",
        time_displaced = true,
        pairs = [(1, 1)]
    )

    initialize_correlation_measurements!(
        measurement_container = measurement_container,
        model_geometry = model_geometry,
        correlation = "spin_x",
        time_displaced = true,
        pairs = [(1, 1)]
    )

    initialize_correlation_measurements!(
        measurement_container = measurement_container,
        model_geometry = model_geometry,
        correlation = "spin_z",
        time_displaced = true,
        pairs = [(1, 1)]
    )

    initialize_correlation_measurements!(
        measurement_container = measurement_container,
        model_geometry = model_geometry,
        correlation = "spin_x",
        time_displaced = false,
        pairs = [(1, 1)]
    )

    initialize_correlation_measurements!(
        measurement_container = measurement_container,
        model_geometry = model_geometry,
        correlation = "spin_z",
        time_displaced = false,
        pairs = [(1, 1)]
    )

    initialize_measurement_directories(
        simulation_info = simulation_info,
        measurement_container = measurement_container
    )

    fermion_path_integral_up = FermionPathIntegral(tight_binding_parameters = tight_binding_parameters, β = β, Δτ = Δτ)
    fermion_path_integral_dn = FermionPathIntegral(tight_binding_parameters = tight_binding_parameters, β = β, Δτ = Δτ)

    initialize!(fermion_path_integral_up, fermion_path_integral_dn, hubbard_parameters)
    initialize!(fermion_path_integral_up, fermion_path_integral_dn, hubbard_ising_parameters)

    Bup = initialize_propagators(fermion_path_integral_up, symmetric=symmetric, checkerboard=checkerboard)
    Bdn = initialize_propagators(fermion_path_integral_dn, symmetric=symmetric, checkerboard=checkerboard)

    fermion_greens_calculator_up = dqmcf.FermionGreensCalculator(Bup, β, Δτ, n_stab)
    fermion_greens_calculator_dn = dqmcf.FermionGreensCalculator(Bdn, β, Δτ, n_stab)

    fermion_greens_calculator_up_alt = dqmcf.FermionGreensCalculator(fermion_greens_calculator_up)
    fermion_greens_calculator_dn_alt = dqmcf.FermionGreensCalculator(fermion_greens_calculator_dn)

    Gup = zeros(eltype(Bup[1]), size(Bup[1]))
    Gdn = zeros(eltype(Bdn[1]), size(Bdn[1]))
    logdetGup, sgndetGup = dqmcf.calculate_equaltime_greens!(Gup, fermion_greens_calculator_up)
    logdetGdn, sgndetGdn = dqmcf.calculate_equaltime_greens!(Gdn, fermion_greens_calculator_dn)

    Gup_ττ = similar(Gup)
    Gup_τ0 = similar(Gup)
    Gup_0τ = similar(Gup)
    Gdn_ττ = similar(Gdn)
    Gdn_τ0 = similar(Gdn)
    Gdn_0τ = similar(Gdn)

    δG = zero(typeof(logdetGup))
    δθ = zero(typeof(sgndetGup))

    for n in 1:N_burnin

        (acceptance_rate, logdetGup, sgndetGup, logdetGdn, sgndetGdn, δG, δθ) = local_updates!(
            Gup, logdetGup, sgndetGup, Gdn, logdetGdn, sgndetGdn,
            hubbard_ising_parameters,
            fermion_path_integral_up = fermion_path_integral_up,
            fermion_path_integral_dn = fermion_path_integral_dn,
            fermion_greens_calculator_up = fermion_greens_calculator_up,
            fermion_greens_calculator_dn = fermion_greens_calculator_dn,
            Bup = Bup, Bdn = Bdn, δG_max = δG_max, δG = δG, δθ = δθ, rng = rng
        )

        additional_info["local_acceptance_rate"] += acceptance_rate

        (accepted, logdetGup, sgndetGup, logdetGdn, sgndetGdn) = reflection_update!(
            Gup, logdetGup, sgndetGup, Gdn, logdetGdn, sgndetGdn,
            hubbard_ising_parameters,
            fermion_path_integral_up = fermion_path_integral_up,
            fermion_path_integral_dn = fermion_path_integral_dn,
            fermion_greens_calculator_up = fermion_greens_calculator_up,
            fermion_greens_calculator_dn = fermion_greens_calculator_dn,
            fermion_greens_calculator_up_alt = fermion_greens_calculator_up_alt,
            fermion_greens_calculator_dn_alt = fermion_greens_calculator_dn_alt,
            Bup = Bup, Bdn = Bdn, rng = rng
        )

        additional_info["reflection_acceptance_rate"] += accepted
    end

    δG = zero(typeof(logdetGup))
    δθ = zero(typeof(sgndetGup))

    for bin in 1:N_bins

        for n in 1:bin_size

            (acceptance_rate, logdetGup, sgndetGup, logdetGdn, sgndetGdn, δG, δθ) = local_updates!(
                Gup, logdetGup, sgndetGup, Gdn, logdetGdn, sgndetGdn,
                hubbard_ising_parameters,
                fermion_path_integral_up = fermion_path_integral_up,
                fermion_path_integral_dn = fermion_path_integral_dn,
                fermion_greens_calculator_up = fermion_greens_calculator_up,
                fermion_greens_calculator_dn = fermion_greens_calculator_dn,
                Bup = Bup, Bdn = Bdn, δG_max = δG_max, δG = δG, δθ = δθ, rng = rng
            )

            additional_info["local_acceptance_rate"] += acceptance_rate

            (accepted, logdetGup, sgndetGup, logdetGdn, sgndetGdn) = reflection_update!(
                Gup, logdetGup, sgndetGup, Gdn, logdetGdn, sgndetGdn,
                hubbard_ising_parameters,
                fermion_path_integral_up = fermion_path_integral_up,
                fermion_path_integral_dn = fermion_path_integral_dn,
                fermion_greens_calculator_up = fermion_greens_calculator_up,
                fermion_greens_calculator_dn = fermion_greens_calculator_dn,
                fermion_greens_calculator_up_alt = fermion_greens_calculator_up_alt,
                fermion_greens_calculator_dn_alt = fermion_greens_calculator_dn_alt,
                Bup = Bup, Bdn = Bdn, rng = rng
            )

            additional_info["reflection_acceptance_rate"] += accepted

            (logdetGup, sgndetGup, logdetGdn, sgndetGdn, δG, δθ) = make_measurements!(
                measurement_container,
                logdetGup, sgndetGup, Gup, Gup_ττ, Gup_τ0, Gup_0τ,
                logdetGdn, sgndetGdn, Gdn, Gdn_ττ, Gdn_τ0, Gdn_0τ,
                fermion_path_integral_up = fermion_path_integral_up,
                fermion_path_integral_dn = fermion_path_integral_dn,
                fermion_greens_calculator_up = fermion_greens_calculator_up,
                fermion_greens_calculator_dn = fermion_greens_calculator_dn,
                Bup = Bup, Bdn = Bdn, δG_max = δG_max, δG = δG, δθ = δθ,
                model_geometry = model_geometry, tight_binding_parameters = tight_binding_parameters,
                coupling_parameters = (hubbard_parameters,)
            )
        end

        write_measurements!(
            measurement_container = measurement_container,
            simulation_info = simulation_info,
            model_geometry = model_geometry,
            bin = bin,
            bin_size = bin_size,
            Δτ = Δτ
        )
    end

    additional_info["local_acceptance_rate"] /= (N_updates + N_burnin)
    additional_info["reflection_acceptance_rate"] /= (N_updates + N_burnin)
    additional_info["n_stab_final"] = fermion_greens_calculator_up.n_stab
    additional_info["dG"] = δG
    save_simulation_info(simulation_info, additional_info)
    process_measurements(simulation_info.datafolder, 20)

    return nothing
end

run_hubbard_square_simulation()