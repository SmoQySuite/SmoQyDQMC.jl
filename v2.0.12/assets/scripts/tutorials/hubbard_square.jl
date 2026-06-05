using SmoQyDQMC
import SmoQyDQMC.LatticeUtilities as lu
import SmoQyDQMC.JDQMCFramework as dqmcf

using Random
using Printf

# Top-level function to run simulation.
function run_simulation(;
    # KEYWORD ARGUMENTS
    sID, # Simulation ID.
    U, # Hubbard interaction.
    t′, # Next-nearest-neighbor hopping amplitude.
    μ, # Chemical potential.
    L, # System size.
    β, # Inverse temperature.
    N_therm, # Number of thermalization updates.
    N_measurements, # Total number of measurements.
    N_bins, # Number of measurement bins.
    N_updates, # Number of updates per measurement.
    Δτ = 0.05, # Discretization in imaginary time.
    n_stab = 10, # Numerical stabilization period in imaginary-time slices.
    δG_max = 1e-6, # Threshold for numerical error corrected by stabilization.
    symmetric = false, # Whether symmetric propagator definition is used.
    checkerboard = false, # Whether checkerboard approximation is used.
    seed = abs(rand(Int)), # Seed for random number generator.
    filepath = "." # Filepath to where data folder will be created.
)

    # Construct the foldername the data will be written to.
    datafolder_prefix = @sprintf "hubbard_square_U%.2f_tp%.2f_mu%.2f_L%d_b%.2f" U t′ μ L β

    # Initialize simulation info.
    simulation_info = SimulationInfo(
        filepath = filepath,
        datafolder_prefix = datafolder_prefix,
        write_bins_concurrent = (L > 10),
        sID = sID
    )

    # Initialize the directory the data will be written to.
    initialize_datafolder(simulation_info)

    # Initialize random number generator
    rng = Xoshiro(seed)

    # Initialize metadata dictionary
    metadata = Dict()

    # Record simulation parameters.
    metadata["N_therm"] = N_therm
    metadata["N_measurements"] = N_measurements
    metadata["N_updates"] = N_updates
    metadata["N_bins"] = N_bins
    metadata["n_stab_init"] = n_stab
    metadata["dG_max"] = δG_max
    metadata["symmetric"] = symmetric
    metadata["checkerboard"] = checkerboard
    metadata["seed"] = seed
    metadata["local_acceptance_rate"] = 0.0
    metadata["reflection_acceptance_rate"] = 0.0

    # Define unit cell.
    unit_cell = lu.UnitCell(
        lattice_vecs = [[1.0, 0.0],
                        [0.0, 1.0]],
        basis_vecs = [[0.0, 0.0]]
    )

    # Define finite lattice with periodic boundary conditions.
    lattice = lu.Lattice(
        L = [L, L],
        periodic = [true, true]
    )

    # Initialize model geometry.
    model_geometry = ModelGeometry(
        unit_cell, lattice
    )

    # Define the nearest-neighbor bond in +x direction.
    bond_px = lu.Bond(
        orbitals = (1,1),
        displacement = [1, 0]
    )

    # Add this bond definition to the model, by adding it the model_geometry.
    bond_px_id = add_bond!(model_geometry, bond_px)

    # Define the nearest-neighbor bond in +y direction.
    bond_py = lu.Bond(
        orbitals = (1,1),
        displacement = [0, 1]
    )

    # Add this bond definition to the model, by adding it the model_geometry.
    bond_py_id = add_bond!(model_geometry, bond_py)

    # Define the nearest-neighbor bond in -x direction.
    # Will be used to make measurements later in this tutorial.
    bond_nx = lu.Bond(
        orbitals = (1,1),
        displacement = [-1, 0]
    )

    # Add this bond definition to the model, by adding it the model_geometry.
    bond_nx_id = add_bond!(model_geometry, bond_nx)

    # Define the nearest-neighbor bond in -y direction.
    # Will be used to make measurements later in this tutorial.
    bond_ny = lu.Bond(
        orbitals = (1,1),
        displacement = [0, -1]
    )

    # Add this bond definition to the model, by adding it the model_geometry.
    bond_ny_id = add_bond!(model_geometry, bond_ny)

    # Define the next-nearest-neighbor bond in +x+y direction.
    bond_pxpy = lu.Bond(
        orbitals = (1,1),
        displacement = [1, 1]
    )

    # Add this bond definition to the model, by adding it the model_geometry.
    bond_pxpy_id = add_bond!(model_geometry, bond_pxpy)

    # Define the next-nearest-neighbor bond in +x-y direction.
    bond_pxny = lu.Bond(
        orbitals = (1,1),
        displacement = [1, -1]
    )

    # Add this bond definition to the model, by adding it the model_geometry.
    bond_pxny_id = add_bond!(model_geometry, bond_pxny)

    # Set nearest-neighbor hopping amplitude to unity,
    # setting the energy scale in the model.
    t = 1.0

    # Define the non-interacting tight-binding model.
    tight_binding_model = TightBindingModel(
        model_geometry = model_geometry,
        t_bonds = [bond_px, bond_py, bond_pxpy, bond_pxny], # defines hopping
        t_mean = [t, t, t′, t′], # defines corresponding mean hopping amplitude
        t_std = [0., 0., 0., 0.], # defines corresponding standard deviation in hopping amplitude
        ϵ_mean = [0.], # set mean on-site energy for each orbital in unit cell
        ϵ_std = [0.], # set standard deviation of on-site energy or each orbital in unit cell
        μ  = μ # set chemical potential
    )

    # Define the Hubbard interaction in the model.
    hubbard_model = HubbardModel(
        ph_sym_form = true, # if particle-hole symmetric form for Hubbard interaction is used.
        U_orbital = [1], # orbitals in unit cell with Hubbard interaction.
        U_mean = [U], # mean Hubbard interaction strength for corresponding orbital species in unit cell.
        U_std = [0.], # standard deviation of Hubbard interaction strength for corresponding orbital species in unit cell.
    )

    # Write model summary TOML file specifying Hamiltonian that will be simulated.
    model_summary(
        simulation_info = simulation_info,
        β = β, Δτ = Δτ,
        model_geometry = model_geometry,
        tight_binding_model = tight_binding_model,
        interactions = (hubbard_model,)
    )

    # Initialize tight-binding parameters.
    tight_binding_parameters = TightBindingParameters(
        tight_binding_model = tight_binding_model,
        model_geometry = model_geometry,
        rng = rng
    )

    # Initialize Hubbard interaction parameters.
    hubbard_parameters = HubbardParameters(
        model_geometry = model_geometry,
        hubbard_model = hubbard_model,
        rng = rng
    )

    # Apply Spin Hirsch Hubbard-Stratonovich (HS) transformation to decouple the Hubbard interaction,
    # and initialize the corresponding HS fields that will be sampled in the DQMC simulation.
    hst_parameters = HubbardSpinHirschHST(
        β = β, Δτ = Δτ,
        hubbard_parameters = hubbard_parameters,
        rng = rng
    )

    # Initialize the container that measurements will be accumulated into.
    measurement_container = initialize_measurement_container(model_geometry, β, Δτ)

    # Initialize the tight-binding model related measurements, like the hopping energy.
    initialize_measurements!(measurement_container, tight_binding_model)

    # Initialize the Hubbard interaction related measurements.
    initialize_measurements!(measurement_container, hubbard_model)

    # Initialize the single-particle electron Green's function measurement.
    initialize_correlation_measurements!(
        measurement_container = measurement_container,
        model_geometry = model_geometry,
        correlation = "greens",
        time_displaced = true,
        pairs = [(1, 1)]
    )

    # Initialize density correlation function measurement.
    initialize_correlation_measurements!(
        measurement_container = measurement_container,
        model_geometry = model_geometry,
        correlation = "density",
        time_displaced = false,
        integrated = true,
        pairs = [(1, 1)]
    )

    # Initialize the pair correlation function measurement.
    initialize_correlation_measurements!(
        measurement_container = measurement_container,
        model_geometry = model_geometry,
        correlation = "pair",
        time_displaced = false,
        integrated = true,
        pairs = [(1, 1)]
    )

    # Initialize the spin-z correlation function measurement.
    initialize_correlation_measurements!(
        measurement_container = measurement_container,
        model_geometry = model_geometry,
        correlation = "spin_z",
        time_displaced = false,
        integrated = true,
        pairs = [(1, 1)]
    )

    # Initialize the d-wave pair susceptibility measurement.
    initialize_composite_correlation_measurement!(
        measurement_container = measurement_container,
        model_geometry = model_geometry,
        name = "d-wave",
        correlation = "pair",
        ids = [bond_px_id, bond_nx_id, bond_py_id, bond_ny_id],
        coefficients = [0.5, 0.5, -0.5, -0.5],
        time_displaced = false,
        integrated = true
    )

    # Allocate FermionPathIntegral type for spin-up electrons.
    fermion_path_integral_up = FermionPathIntegral(
        tight_binding_parameters = tight_binding_parameters, β = β, Δτ = Δτ,
        forced_complex_potential = (U < 0),
        forced_complex_kinetic = false
    )

    # Allocate FermionPathIntegral type for spin-down electrons.
    fermion_path_integral_dn = FermionPathIntegral(
        tight_binding_parameters = tight_binding_parameters, β = β, Δτ = Δτ,
        forced_complex_potential = (U < 0),
        forced_complex_kinetic = false
    )

    # Initialize FermionPathIntegral type for both the spin-up and spin-down electrons to account for Hubbard interaction.
    initialize!(fermion_path_integral_up, fermion_path_integral_dn, hubbard_parameters)

    # Initialize FermionPathIntegral type for both the spin-up and spin-down electrons to account for the current
    # Hubbard-Stratonovich field configuration.
    initialize!(fermion_path_integral_up, fermion_path_integral_dn, hst_parameters)

    # Initialize imaginary-time propagators for all imaginary-time slices for spin-up and spin-down electrons.
    Bup = initialize_propagators(fermion_path_integral_up, symmetric=symmetric, checkerboard=checkerboard)
    Bdn = initialize_propagators(fermion_path_integral_dn, symmetric=symmetric, checkerboard=checkerboard)

    # Initialize FermionGreensCalculator type for spin-up and spin-down electrons.
    fermion_greens_calculator_up = dqmcf.FermionGreensCalculator(Bup, β, Δτ, n_stab)
    fermion_greens_calculator_dn = dqmcf.FermionGreensCalculator(Bdn, β, Δτ, n_stab)

    # Initialize alternate FermionGreensCalculator type for performing reflection updates.
    fermion_greens_calculator_up_alt = dqmcf.FermionGreensCalculator(fermion_greens_calculator_up)
    fermion_greens_calculator_dn_alt = dqmcf.FermionGreensCalculator(fermion_greens_calculator_dn)

    # Allocate matrices for spin-up and spin-down electron Green's function matrices.
    Gup = zeros(eltype(Bup[1]), size(Bup[1]))
    Gdn = zeros(eltype(Bdn[1]), size(Bdn[1]))

    # Initialize the spin-up and spin-down electron Green's function matrices, also
    # calculating their respective determinants as the same time.
    logdetGup, sgndetGup = dqmcf.calculate_equaltime_greens!(Gup, fermion_greens_calculator_up)
    logdetGdn, sgndetGdn = dqmcf.calculate_equaltime_greens!(Gdn, fermion_greens_calculator_dn)

    # Allocate matrices for various time-displaced Green's function matrices.
    Gup_ττ = similar(Gup) # Gup(τ,τ)
    Gup_τ0 = similar(Gup) # Gup(τ,0)
    Gup_0τ = similar(Gup) # Gup(0,τ)
    Gdn_ττ = similar(Gdn) # Gdn(τ,τ)
    Gdn_τ0 = similar(Gdn) # Gdn(τ,0)
    Gdn_0τ = similar(Gdn) # Gdn(0,τ)

    # Initialize diagnostic parameters to asses numerical stability.
    δG = zero(logdetGup)
    δθ = zero(logdetGup)

    # Iterate over number of thermalization updates to perform.
    for n in 1:N_therm

        # Perform reflection update for HS fields with randomly chosen site.
        (accepted, logdetGup, sgndetGup, logdetGdn, sgndetGdn) = reflection_update!(
            Gup, logdetGup, sgndetGup, Gdn, logdetGdn, sgndetGdn,
            hst_parameters,
            fermion_path_integral_up = fermion_path_integral_up,
            fermion_path_integral_dn = fermion_path_integral_dn,
            fermion_greens_calculator_up = fermion_greens_calculator_up,
            fermion_greens_calculator_dn = fermion_greens_calculator_dn,
            fermion_greens_calculator_up_alt = fermion_greens_calculator_up_alt,
            fermion_greens_calculator_dn_alt = fermion_greens_calculator_dn_alt,
            Bup = Bup, Bdn = Bdn, rng = rng
        )

        # Record whether reflection update was accepted or not.
        metadata["reflection_acceptance_rate"] += accepted

        # Perform sweep all imaginary-time slice and orbitals, attempting an update to every HS field.
        (acceptance_rate, logdetGup, sgndetGup, logdetGdn, sgndetGdn, δG, δθ) = local_updates!(
            Gup, logdetGup, sgndetGup, Gdn, logdetGdn, sgndetGdn,
            hst_parameters,
            fermion_path_integral_up = fermion_path_integral_up,
            fermion_path_integral_dn = fermion_path_integral_dn,
            fermion_greens_calculator_up = fermion_greens_calculator_up,
            fermion_greens_calculator_dn = fermion_greens_calculator_dn,
            Bup = Bup, Bdn = Bdn, δG_max = δG_max, δG = δG, δθ = δθ, rng = rng,
            update_stabilization_frequency = false
        )

        # Record acceptance rate for sweep.
        metadata["local_acceptance_rate"] += acceptance_rate
    end

    # Reset diagnostic parameters used to monitor numerical stability to zero.
    δG = zero(logdetGup)
    δθ = zero(logdetGup)

    # Calculate the bin size.
    bin_size = N_measurements ÷ N_bins

    # Iterate over measurements.
    for measurement in 1:N_measurements

        # Iterate over updates between measurements.
        for update in 1:N_updates

            # Perform reflection update for HS fields with randomly chosen site.
            (accepted, logdetGup, sgndetGup, logdetGdn, sgndetGdn) = reflection_update!(
                Gup, logdetGup, sgndetGup, Gdn, logdetGdn, sgndetGdn,
                hst_parameters,
                fermion_path_integral_up = fermion_path_integral_up,
                fermion_path_integral_dn = fermion_path_integral_dn,
                fermion_greens_calculator_up = fermion_greens_calculator_up,
                fermion_greens_calculator_dn = fermion_greens_calculator_dn,
                fermion_greens_calculator_up_alt = fermion_greens_calculator_up_alt,
                fermion_greens_calculator_dn_alt = fermion_greens_calculator_dn_alt,
                Bup = Bup, Bdn = Bdn, rng = rng
            )

            # Record whether reflection update was accepted or not.
            metadata["reflection_acceptance_rate"] += accepted

            # Perform sweep all imaginary-time slice and orbitals, attempting an update to every HS field.
            (acceptance_rate, logdetGup, sgndetGup, logdetGdn, sgndetGdn, δG, δθ) = local_updates!(
                Gup, logdetGup, sgndetGup, Gdn, logdetGdn, sgndetGdn,
                hst_parameters,
                fermion_path_integral_up = fermion_path_integral_up,
                fermion_path_integral_dn = fermion_path_integral_dn,
                fermion_greens_calculator_up = fermion_greens_calculator_up,
                fermion_greens_calculator_dn = fermion_greens_calculator_dn,
                Bup = Bup, Bdn = Bdn, δG_max = δG_max, δG = δG, δθ = δθ, rng = rng,
                update_stabilization_frequency = false
            )

            # Record acceptance rate.
            metadata["local_acceptance_rate"] += acceptance_rate
        end

        # Make measurements.
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
            coupling_parameters = (hubbard_parameters, hst_parameters)
        )

        # Write the bin-averaged measurements to file if update ÷ bin_size == 0.
        write_measurements!(
            measurement_container = measurement_container,
            simulation_info = simulation_info,
            model_geometry = model_geometry,
            measurement = measurement,
            bin_size = bin_size,
            Δτ = Δτ
        )
    end

    # Merge binned data into a single HDF5 file.
    merge_bins(simulation_info)

    # Normalize acceptance rate.
    metadata["local_acceptance_rate"] /=  (N_therm + N_measurements * N_updates)
    metadata["reflection_acceptance_rate"] /= (N_therm + N_measurements * N_updates)

    # Record final stabilization period used at the end of the simulation.
    metadata["n_stab_final"] = fermion_greens_calculator_up.n_stab

    # Record largest numerical error.
    metadata["dG"] = δG

    # Write simulation summary TOML file.
    save_simulation_info(simulation_info, metadata)

    # Process the simulation results, calculating final error bars for all measurements.
    # writing final statistics to CSV files.
    process_measurements(
        datafolder = simulation_info.datafolder,
        n_bins = N_bins,
        export_to_csv = true,
        scientific_notation = false,
        decimals = 7,
        delimiter = " "
    )

    # Calculate AFM correlation ratio.
    Rafm, ΔRafm = compute_correlation_ratio(
        datafolder = simulation_info.datafolder,
        correlation = "spin_z",
        type = "equal-time",
        id_pairs = [(1, 1)],
        id_pair_coefficients = [1.0],
        q_point = (L÷2, L÷2),
        q_neighbors = [
            (L÷2+1, L÷2), (L÷2-1, L÷2),
            (L÷2, L÷2+1), (L÷2, L÷2-1)
        ]
    )

    # Record the AFM correlation ratio mean and standard deviation.
    metadata["Rafm_mean_real"] = real(Rafm)
    metadata["Rafm_mean_imag"] = imag(Rafm)
    metadata["Rafm_std"] = ΔRafm

    # Write simulation summary TOML file.
    save_simulation_info(simulation_info, metadata)

    return nothing
end # end of run_simulation function

# Only execute if the script is run directly from the command line.
if abspath(PROGRAM_FILE) == @__FILE__

    # Run the simulation, reading in command line arguments.
    run_simulation(;
        sID = parse(Int, ARGS[1]), # Simulation ID.
        U = parse(Float64, ARGS[2]), # Hubbard interaction strength.
        t′ = parse(Float64, ARGS[3]), # Next-nearest-neighbor hopping amplitude.
        μ = parse(Float64, ARGS[4]), # Chemical potential.
        L = parse(Int, ARGS[5]), # Lattice size.
        β = parse(Float64, ARGS[6]), # Inverse temperature.
        N_therm = parse(Int, ARGS[7]), # Number of thermalization sweeps.
        N_measurements = parse(Int, ARGS[8]), # Number of measurement to make.
        N_bins = parse(Int, ARGS[9]), # Number of measurement bins.
        N_updates = parse(Int, ARGS[10]) # Number of updates per measurement.
    )
end
