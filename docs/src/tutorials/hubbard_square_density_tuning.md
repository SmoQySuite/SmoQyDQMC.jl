```@meta
EditURL = "../../../tutorials/hubbard_square_density_tuning.jl"
```

# 1d) Square Hubbard Model with Density Tuning
Download this example as a [Julia script](../assets/scripts/tutorials/hubbard_square_density_tuning.jl).

In this example we demonstrate how to introduce chemical potential and density tuning to the previous
[1c) Square Hubbard Model with Checkpointing](@ref) tutorial.
Specifically, we show how to use the algorithm introduced in
[Phys. Rev. E 105, 045311](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.105.045311)
for dynamically adjusting the chemical potential during the simulation in order to achieve a target
electron density or filling fraction.

Note that when you dope the Hubbard model away from half-filling a sign problem is introduced.
As with making measurements, if the sign problem becomes severe the density tuning algorithm will
become very inefficient as simply providing an accurate measurement of the density and
compressibility (which is used to adjust the chemical potential) will become challenging.

## Import Packages
Compared to the previous [1c) Square Hubbard Model with Checkpointing](@ref) tutorial,
we now need to import the [MuTuner.jl](https://github.com/cohensbw/MuTuner.jl.git)
package, which is reexported by [SmoQyDQMC.jl](https://github.com/SmoQySuite/SmoQyDQMC.jl.git)

````julia
using SmoQyDQMC
import SmoQyDQMC.LatticeUtilities as lu
import SmoQyDQMC.JDQMCFramework as dqmcf
import SmoQyDQMC.MuTuner as mt

using Random
using Printf
using MPI
````

## Specify simulation parameters
Here we introduce the keyword argument `n` to the `run_simulation` function
which specifies the target electron density we want to achieve in the simulation.
Now the `μ` argument specifies the initial chemical potential we begin the simulation with,
but of course it will be adjusted during the simulation to achieve the target density `n`.

````julia
# Top-level function to run simulation.
function run_simulation(
    comm::MPI.Comm; # MPI communicator.
    # KEYWORD ARGUMENTS
    sID, # Simulation ID.
    U, # Hubbard interaction.
    t′, # Next-nearest-neighbor hopping amplitude.
    n, # Target density.
    μ, # Initial chemical potential.
    L, # System size.
    β, # Inverse temperature.
    N_therm, # Number of thermalization updates.
    N_measurements, # Number of measurements to make.
    N_bins, # Number of times bin-averaged measurements are written to file.
    N_updates, # Number of updates between measurements.
    checkpoint_freq, # Frequency with which checkpoint files are written in hours.
    runtime_limit = Inf, # Simulation runtime limit in hours.
    Δτ = 0.05, # Discretization in imaginary time.
    n_stab = 10, # Numerical stabilization period in imaginary-time slices.
    δG_max = 1e-6, # Threshold for numerical error corrected by stabilization.
    symmetric = false, # Whether symmetric propagator definition is used.
    checkerboard = false, # Whether checkerboard approximation is used.
    seed = abs(rand(Int)), # Seed for random number generator.
    filepath = "." # Filepath to where data folder will be created.
)
````

## Initialize simulation
No changes need to made to this section of the code from the previous
[1c) Square Hubbard Model with Checkpointing](@ref) tutorial.

````julia
    # Record when the simulation began.
    start_timestamp = time()

    # Convert runtime limit from hours to seconds.
    runtime_limit = runtime_limit * 60.0^2

    # Convert checkpoint frequency from hours to seconds.
    checkpoint_freq = checkpoint_freq * 60.0^2

    # Construct the foldername the data will be written to.
    datafolder_prefix = @sprintf "hubbard_square_U%.2f_tp%.2f_n%.2f_L%d_b%.2f" U t′ n L β

    # Get MPI process ID.
    pID = MPI.Comm_rank(comm)

    # Initialize simulation info.
    simulation_info = SimulationInfo(
        filepath = filepath,
        datafolder_prefix = datafolder_prefix,
        write_bins_concurrent = (L > 10),
        sID = sID,
        pID = pID
    )

    # Initialize the directory the data will be written to if one does not already exist.
    initialize_datafolder(comm, simulation_info)
````

## Initialize simulation metadata
Here it is useful to record the initial chemical potential `μ` used during the simulation
in the metadata dictionary.

````julia
    # If starting a new simulation i.e. not resuming a previous simulation.
    if !simulation_info.resuming

        # Begin thermalization updates from start.
        n_therm = 1

        # Begin measurement updates from start.
        n_measurements = 1

        # Initialize random number generator
        rng = Xoshiro(seed)

        # Initialize metadata dictionary
        metadata = Dict()

        # Record simulation parameters.
        metadata["mu"] = μ
        metadata["N_therm"] = N_therm
        metadata["N_measurements"] = N_measurements
        metadata["N_bins"] = N_bins
        metadata["N_updates"] = N_updates
        metadata["n_stab_init"] = n_stab
        metadata["dG_max"] = δG_max
        metadata["symmetric"] = symmetric
        metadata["checkerboard"] = checkerboard
        metadata["seed"] = seed
        metadata["local_acceptance_rate"] = 0.0
        metadata["reflection_acceptance_rate"] = 0.0
````

## Initialize Model
No changes need to made to this section of the code from the previous
[1c) Square Hubbard Model with Checkpointing](@ref) tutorial.

````julia
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

        # Define the next-nearest-neighbor bond in +x+y direction.
        bond_pxpy = lu.Bond(
            orbitals = (1,1),
            displacement = [1, 1]
        )

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

        # Set neartest-neighbor hopping amplitude to unity,
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
            μ = μ # set chemical potential
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
````

## Initialize model parameters
In this section we need to make use of the [MuTuner.jl](https://github.com/cohensbw/MuTuner.jl.git)
package, initializing an instance of the
[`MuTuner.MuTunerLogger`](https://cohensbw.github.io/MuTuner.jl/stable/api/)
type using the
[`MuTuner.init_mutunerlogger`](https://cohensbw.github.io/MuTuner.jl/stable/api/)
function. Note that we use the [`LatticeUtilities.nsites`](@extref)
function to calculate the total number of orbitals in our system.

````julia
        # Initialize tight-binding parameters.
        tight_binding_parameters = TightBindingParameters(
            tight_binding_model = tight_binding_model,
            model_geometry = model_geometry,
            rng = rng
        )

        # Initialize Hubbard interaction parameters.
        hubbard_params = HubbardParameters(
            model_geometry = model_geometry,
            hubbard_model = hubbard_model,
            rng = rng
        )

        # Apply Hubbard-Stratonovich (HS) transformation to decouple the Hubbard interaction,
        # and initialize the corresponding HS fields that will be sampled in the DQMC simulation.
        hst_parameters = HubbardSpinHirschHST(
            β = β, Δτ = Δτ,
            hubbard_parameters = hubbard_params,
            rng = rng
        )

        # Initialize MuTunerLogger type that will be used to dynamically adjust the
        # chemical potential during the simulation.
        chemical_potential_tuner = mt.init_mutunerlogger(
            target_density = n,
            inverse_temperature = β,
            system_size = lu.nsites(unit_cell, lattice),
            initial_chemical_potential = μ,
            complex_sign_problem = false
        )
````

## Initialize measurements
No changes need to made to this section of the code from the previous
[1c) Square Hubbard Model with Checkpointing](@ref) tutorial.

````julia
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
````

## Write first checkpoint
Here we need to add the
[`MuTuner.MuTunerLogger`](https://cohensbw.github.io/MuTuner.jl/stable/api/)
instance `chemical_potential_tuner` to the checkpoint file.

````julia
        # Write initial checkpoint file.
        checkpoint_timestamp = write_jld2_checkpoint(
            comm,
            simulation_info;
            checkpoint_freq = checkpoint_freq,
            start_timestamp = start_timestamp,
            runtime_limit = runtime_limit,
            # Contents of checkpoint file below.
            n_therm, n_measurements,
            tight_binding_parameters, hubbard_params, hst_parameters,
            chemical_potential_tuner, measurement_container, model_geometry, metadata, rng
        )
````

## Load checkpoint
Here we need to make sure to load the [`MuTuner.MuTunerLogger`](https://cohensbw.github.io/MuTuner.jl/stable/api/)
instance `chemical_potential_tuner` from the checkpoint file.

````julia
    # If resuming a previous simulation.
    else

        # Load the checkpoint file.
        checkpoint, checkpoint_timestamp = read_jld2_checkpoint(simulation_info)

        # Unpack contents of checkpoint dictionary.
        tight_binding_parameters = checkpoint["tight_binding_parameters"]
        hubbard_params = checkpoint["hubbard_params"]
        hst_parameters = checkpoint["hst_parameters"]
        chemical_potential_tuner = checkpoint["chemical_potential_tuner"]
        measurement_container = checkpoint["measurement_container"]
        model_geometry = checkpoint["model_geometry"]
        metadata = checkpoint["metadata"]
        rng = checkpoint["rng"]
        n_therm = checkpoint["n_therm"]
        n_measurements = checkpoint["n_measurements"]
    end
````

## Setup DQMC simulation
No changes need to made to this section of the code from the previous
[1c) Square Hubbard Model with Checkpointing](@ref) tutorial.

````julia
    # Allocate FermionPathIntegral type for both the spin-up and spin-down electrons.
    fermion_path_integral_up = FermionPathIntegral(tight_binding_parameters = tight_binding_parameters, β = β, Δτ = Δτ)
    fermion_path_integral_dn = FermionPathIntegral(tight_binding_parameters = tight_binding_parameters, β = β, Δτ = Δτ)

    # Initialize FermionPathIntegral type for both the spin-up and spin-down electrons to account for Hubbard interaction.
    initialize!(fermion_path_integral_up, fermion_path_integral_dn, hubbard_params)

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
````

## Thermalize system
Here we need to add a call to the [`update_chemical_potential!`](@ref) function
after completing the updates but before writing the checkpoint file is written.
And again, we need to make sure the include the `chemical_potential_tuner` in the checkpoint file.

````julia
    # Iterate over number of thermalization updates to perform.
    for update in n_therm:N_therm

        # Update the chemical potential to achieve the target density.
        (logdetGup, sgndetGup, logdetGdn, sgndetGdn) = update_chemical_potential!(
            Gup, logdetGup, sgndetGup,
            Gdn, logdetGdn, sgndetGdn;
            chemical_potential_tuner = chemical_potential_tuner,
            tight_binding_parameters = tight_binding_parameters,
            fermion_path_integral_up = fermion_path_integral_up,
            fermion_path_integral_dn = fermion_path_integral_dn,
            fermion_greens_calculator_up = fermion_greens_calculator_up,
            fermion_greens_calculator_dn = fermion_greens_calculator_dn,
            Bup = Bup, Bdn = Bdn
        )

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
            update_stabilization_frequency = true
        )

        # Record acceptance rate for sweep.
        metadata["local_acceptance_rate"] += acceptance_rate

        # Write checkpoint file.
        checkpoint_timestamp = write_jld2_checkpoint(
            comm,
            simulation_info;
            checkpoint_timestamp = checkpoint_timestamp,
            checkpoint_freq = checkpoint_freq,
            start_timestamp = start_timestamp,
            runtime_limit = runtime_limit,
            # Contents of checkpoint file below.
            n_therm = update + 1,
            n_measurements = 1,
            tight_binding_parameters, hubbard_params, hst_parameters,
            chemical_potential_tuner, measurement_container, model_geometry, metadata, rng
        )
    end
````

## Make measurements
Here we need to add a call to the [`update_chemical_potential!`](@ref) function
after making and writing measurements but before writing the checkpoint file is written.
And again, we need to make sure the include the `chemical_potential_tuner` in the checkpoint file.

````julia
    # Reset diagnostic parameters used to monitor numerical stability to zero.
    δG = zero(logdetGup)
    δθ = zero(logdetGup)

    # Calculate the bin size.
    bin_size = N_measurements ÷ N_bins

    # Iterate over measurements.
    for measurement in n_measurements:N_measurements

        # Iterate over number of updates between measurements.
        for update in 1:N_updates

            # Update the chemical potential to achieve the target density.
            (logdetGup, sgndetGup, logdetGdn, sgndetGdn) = update_chemical_potential!(
                Gup, logdetGup, sgndetGup,
                Gdn, logdetGdn, sgndetGdn;
                chemical_potential_tuner = chemical_potential_tuner,
                tight_binding_parameters = tight_binding_parameters,
                fermion_path_integral_up = fermion_path_integral_up,
                fermion_path_integral_dn = fermion_path_integral_dn,
                fermion_greens_calculator_up = fermion_greens_calculator_up,
                fermion_greens_calculator_dn = fermion_greens_calculator_dn,
                Bup = Bup, Bdn = Bdn
            )

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
                update_stabilization_frequency = true
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
            coupling_parameters = (hubbard_params, hst_parameters)
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

        # Write checkpoint file.
        checkpoint_timestamp = write_jld2_checkpoint(
            comm,
            simulation_info;
            checkpoint_timestamp = checkpoint_timestamp,
            checkpoint_freq = checkpoint_freq,
            start_timestamp = start_timestamp,
            runtime_limit = runtime_limit,
            # Contents of checkpoint file below.
            n_therm  = N_therm + 1,
            n_measurements = measurement + 1,
            tight_binding_parameters, hubbard_params, hst_parameters,
            chemical_potential_tuner, measurement_container, model_geometry, metadata, rng
        )
    end
````

## Merge binned data
No changes need to made to this section of the code from the previous [1a) Square Hubbard Model](@ref) tutorial.

````julia
    # Merge binned data into a single HDF5 file.
    merge_bins(simulation_info)
````

## Record simulation metadata
Here we can add a call to the [`save_density_tuning_profile`](@ref), which records the full history
of the chemical potential and density tuning process.

````julia
    # Normalize acceptance rate.
    metadata["local_acceptance_rate"] /=  (N_therm + N_measurements * N_updates)
    metadata["reflection_acceptance_rate"] /=  (N_therm + N_measurements * N_updates)

    # Record final stabilization frequency used at end of simulation.
    metadata["n_stab_final"] = fermion_greens_calculator_up.n_stab

    # Record largest numerical error.
    metadata["dG"] = δG

    # Write simulation summary TOML file.
    save_simulation_info(simulation_info, metadata)

    # Save the density tuning profile to file.
    save_density_tuning_profile(
        simulation_info, chemical_potential_tuner,
        export_to_h5 = true,
        export_to_csv = false
    )
````

## Post-process results
No changes need to made to this section of the code from the previous
[1c) Square Hubbard Model with Checkpointing](@ref) tutorial.

````julia
    # Process the simulation results, calculating final error bars for all measurements.
    # writing final statistics to CSV files.
    process_measurements(
        comm;
        datafolder = simulation_info.datafolder,
        n_bins = N_bins,
        export_to_csv = true,
        scientific_notation = false,
        decimals = 7,
        delimiter = " "
    )

    # Calculate AFM correlation ratio.
    Rafm, ΔRafm = compute_correlation_ratio(
        comm;
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

    # Rename the data folder to indicate the simulation is complete.
    simulation_info = rename_complete_simulation(
        comm, simulation_info,
        delete_jld2_checkpoints = true
    )

    return nothing
end # end of run_simulation function
````

## Execute script
Here we add an additional command line argument to specify the target density `n` we want to achieve in the simulation.
Now the `μ` command line argument specifies the initial chemical potential we begin the simulation with.
For instance, a simulation can be run with the command
```bash
mpiexecjl -n 16 julia hubbard_square_density_tuning.jl 1 5.0 -0.25 0.8 0.0 4 4.0 2000 2000 40 5 1.0
```
or
```bash
srun julia hubbard_square_density_tuning.jl 1 5.0 -0.25 0.8 0.0 4 4.0 2000 2000 40 5 1.0
```
where the target density is ``\langle n \rangle = 0.8`` and the initial chemical potential is ``\mu = 0.0``.

````julia
if abspath(PROGRAM_FILE) == @__FILE__

    # Initialize MPI
    MPI.Init()

    # Initialize the MPI communicator.
    comm = MPI.COMM_WORLD

    # Run the simulation, reading in command line arguments.
    run_simulation(
        comm;
        sID = parse(Int, ARGS[1]), # Simulation ID.
        U = parse(Float64, ARGS[2]), # Hubbard interaction.
        t′ = parse(Float64, ARGS[3]), # Next-nearest-neighbor hopping amplitude.
        n = parse(Float64, ARGS[4]), # Target density.
        μ = parse(Float64, ARGS[5]), # Initial chemical potential.
        L = parse(Int, ARGS[6]), # System size.
        β = parse(Float64, ARGS[7]), # Inverse temperature.
        N_therm = parse(Int, ARGS[8]), # Number of thermalization updates.
        N_measurements = parse(Int, ARGS[9]), # Number of measurements to make.
        N_bins = parse(Int, ARGS[10]), # Number of times bin-averaged measurements are written to file.
        N_updates = parse(Int, ARGS[11]), # Number of updates between measurements.
        checkpoint_freq = parse(Float64, ARGS[12]) # Frequency with which checkpoint files are written in hours.
    )

    # Finalize MPI.
    MPI.Finalize()
end
````

