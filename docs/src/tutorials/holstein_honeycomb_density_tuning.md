```@meta
EditURL = "../../../tutorials/holstein_honeycomb_density_tuning.jl"
```

# 2d) Honeycomb Holstein Model with Density Tuning
Download this example as a [Julia script](../assets/scripts/tutorials/holstein_honeycomb_density_tuning.jl).

In this example we demonstrate how to introduce chemical potential and density tuning to the previous
[2c) Honeycomb Holstein Model with Checkpointing](@ref) tutorial.
Specifically, we show how to use the algorithm introduced in
[Phys. Rev. E 105, 045311](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.105.045311)
for dynamically adjusting the chemical potential during the simulation in order to achieve a target
electron density or filling fraction.

## Import Packages
Compared to the previouse [1c) Square Hubbard Model with Checkpointing](@ref) tutorial,
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
    Ω, # Phonon energy.
    α, # Electron-phonon coupling.
    n, # Target density.
    μ, # Initial chemical potential.
    L, # System size.
    β, # Inverse temperature.
    N_therm, # Number of thermalization updates.
    N_updates, # Total number of measurements and measurement updates.
    N_bins, # Number of times bin-averaged measurements are written to file.
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
[2c) Honeycomb Holstein Model with Checkpointing](@ref) tutorial.

````julia
    # Record when the simulation began.
    start_timestamp = time()

    # Convert runtime limit from hours to seconds.
    runtime_limit = runtime_limit * 60.0^2

    # Convert checkpoint frequency from hours to seconds.
    checkpoint_freq = checkpoint_freq * 60.0^2

    # Construct the foldername the data will be written to.
    datafolder_prefix = @sprintf "holstein_honeycomb_w%.2f_a%.2f_mu%.2f_L%d_b%.2f" Ω α μ L β

    # Get MPI process ID.
    pID = MPI.Comm_rank(comm)

    # Initialize simulation info.
    simulation_info = SimulationInfo(
        filepath = filepath,
        datafolder_prefix = datafolder_prefix,
        sID = sID,
        pID = pID
    )

    # Initialize the directory the data will be written to.
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
        n_updates = 1

        # Initialize random number generator
        rng = Xoshiro(seed)

        # Initialize additiona_info dictionary
        metadata = Dict()

        # Record simulation parameters.
        metadata["mu"] = μ
        metadata["N_therm"] = N_therm
        metadata["N_updates"] = N_updates
        metadata["N_bins"] = N_bins
        metadata["n_stab"] = n_stab
        metadata["dG_max"] = δG_max
        metadata["symmetric"] = symmetric
        metadata["checkerboard"] = checkerboard
        metadata["seed"] = seed
        metadata["hmc_acceptance_rate"] = 0.0
        metadata["reflection_acceptance_rate"] = 0.0
        metadata["swap_acceptance_rate"] = 0.0
````

## Initialize model
No changes need to made to this section of the code from the previous
[2c) Honeycomb Holstein Model with Checkpointing](@ref) tutorial.

````julia
        # Define the unit cell.
        unit_cell = lu.UnitCell(
            lattice_vecs = [[3/2,√3/2],
                            [3/2,-√3/2]],
            basis_vecs   = [[0.,0.],
                            [1.,0.]]
        )

        # Define finite lattice with periodic boundary conditions.
        lattice = lu.Lattice(
            L = [L, L],
            periodic = [true, true]
        )

        # Initialize model geometry.
        model_geometry = ModelGeometry(unit_cell, lattice)

        # Define the first nearest-neighbor bond in a honeycomb lattice.
        bond_1 = lu.Bond(orbitals = (1,2), displacement = [0,0])

        # Add the first nearest-neighbor bond in a honeycomb lattice to the model.
        bond_1_id = add_bond!(model_geometry, bond_1)

        # Define the second nearest-neighbor bond in a honeycomb lattice.
        bond_2 = lu.Bond(orbitals = (1,2), displacement = [-1,0])

        # Add the second nearest-neighbor bond in a honeycomb lattice to the model.
        bond_2_id = add_bond!(model_geometry, bond_2)

        # Define the third nearest-neighbor bond in a honeycomb lattice.
        bond_3 = lu.Bond(orbitals = (1,2), displacement = [0,-1])

        # Add the third nearest-neighbor bond in a honeycomb lattice to the model.
        bond_3_id = add_bond!(model_geometry, bond_3)

        # Set neartest-neighbor hopping amplitude to unity,
        # setting the energy scale in the model.
        t = 1.0

        # Define the honeycomb tight-binding model.
        tight_binding_model = TightBindingModel(
            model_geometry = model_geometry,
            t_bonds        = [bond_1, bond_2, bond_3], # defines hopping
            t_mean         = [t, t, t], # defines corresponding hopping amplitude
            μ              = μ, # set chemical potential
            ϵ_mean         = [0.0, 0.0] # set the (mean) on-site energy
        )

        # Initialize a null electron-phonon model.
        electron_phonon_model = ElectronPhononModel(
            model_geometry = model_geometry,
            tight_binding_model = tight_binding_model
        )

        # Define a dispersionless electron-phonon mode to live on each site in the lattice.
        phonon_1 = PhononMode(orbital = 1, Ω_mean = Ω)

        # Add the phonon mode definition to the electron-phonon model.
        phonon_1_id = add_phonon_mode!(
            electron_phonon_model = electron_phonon_model,
            phonon_mode = phonon_1
        )

        # Define a dispersionless electron-phonon mode to live on each site in the lattice.
        phonon_2 = PhononMode(orbital = 2, Ω_mean = Ω)

        # Add the phonon mode definition to the electron-phonon model.
        phonon_2_id = add_phonon_mode!(
            electron_phonon_model = electron_phonon_model,
            phonon_mode = phonon_2
        )

        # Define first local Holstein coupling for first phonon mode.
        holstein_coupling_1 = HolsteinCoupling(
            model_geometry = model_geometry,
            phonon_mode = phonon_1_id,
            # Couple the first phonon mode to first orbital in the unit cell.
            bond = lu.Bond(orbitals = (1,1), displacement = [0, 0]),
            α_mean = α
        )

        # Add the first local Holstein coupling definition to the model.
        holstein_coupling_1_id = add_holstein_coupling!(
            electron_phonon_model = electron_phonon_model,
            holstein_coupling = holstein_coupling_1,
            model_geometry = model_geometry
        )

        # Define first local Holstein coupling for first phonon mode.
        holstein_coupling_2 = HolsteinCoupling(
            model_geometry = model_geometry,
            phonon_mode = phonon_2_id,
            # Couple the second phonon mode to second orbital in the unit cell.
            bond = lu.Bond(orbitals = (2,2), displacement = [0, 0]),
            α_mean = α
        )

        # Add the first local Holstein coupling definition to the model.
        holstein_coupling_2_id = add_holstein_coupling!(
            electron_phonon_model = electron_phonon_model,
            holstein_coupling = holstein_coupling_2,
            model_geometry = model_geometry
        )

        # Write model summary TOML file specifying Hamiltonian that will be simulated.
        model_summary(
            simulation_info = simulation_info,
            β = β, Δτ = Δτ,
            model_geometry = model_geometry,
            tight_binding_model = tight_binding_model,
            interactions = (electron_phonon_model,)
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

        # Initialize electron-phonon parameters.
        electron_phonon_parameters = ElectronPhononParameters(
            β = β, Δτ = Δτ,
            electron_phonon_model = electron_phonon_model,
            tight_binding_parameters = tight_binding_parameters,
            model_geometry = model_geometry,
            rng = rng
        )

        # Initialize MuTunerLogger type that will be used to dynamically adjust the
        # chemicaml potential during the simulation.
        chemical_potential_tuner = mt.init_mutunerlogger(
            target_density = n,
            inverse_temperature = β,
            system_size = lu.nsites(unit_cell, lattice),
            initial_chemical_potential = μ,
            complex_sign_problem = false
        )
````

## Initialize meuasurements
No changes need to made to this section of the code from the previous
[2c) Honeycomb Holstein Model with Checkpointing](@ref) tutorial.

````julia
        # Initialize the container that measurements will be accumulated into.
        measurement_container = initialize_measurement_container(model_geometry, β, Δτ)

        # Initialize the tight-binding model related measurements, like the hopping energy.
        initialize_measurements!(measurement_container, tight_binding_model)

        # Initialize the electron-phonon interaction related measurements.
        initialize_measurements!(measurement_container, electron_phonon_model)

        # Initialize the single-particle electron Green's function measurement.
        initialize_correlation_measurements!(
            measurement_container = measurement_container,
            model_geometry = model_geometry,
            correlation = "greens",
            time_displaced = true,
            pairs = [
                # Measure green's functions for all pairs or orbitals.
                (1, 1), (2, 2), (1, 2)
            ]
        )

        # Initialize the single-particle electron Green's function measurement.
        initialize_correlation_measurements!(
            measurement_container = measurement_container,
            model_geometry = model_geometry,
            correlation = "phonon_greens",
            time_displaced = true,
            pairs = [
                # Measure green's functions for all pairs of modes.
                (1, 1), (2, 2), (1, 2)
            ]
        )

        # Initialize density correlation function measurement.
        initialize_correlation_measurements!(
            measurement_container = measurement_container,
            model_geometry = model_geometry,
            correlation = "density",
            time_displaced = false,
            integrated = true,
            pairs = [
                (1, 1), (2, 2),
            ]
        )

        # Initialize the pair correlation function measurement.
        initialize_correlation_measurements!(
            measurement_container = measurement_container,
            model_geometry = model_geometry,
            correlation = "pair",
            time_displaced = false,
            integrated = true,
            pairs = [
                # Measure local s-wave pair susceptibility associated with
                # each orbital in the unit cell.
                (1, 1), (2, 2)
            ]
        )

        # Initialize the spin-z correlation function measurement.
        initialize_correlation_measurements!(
            measurement_container = measurement_container,
            model_geometry = model_geometry,
            correlation = "spin_z",
            time_displaced = false,
            integrated = true,
            pairs = [
                (1, 1), (2, 2)
            ]
        )

        # Initialize CDW correlation measurement.
        initialize_composite_correlation_measurement!(
            measurement_container = measurement_container,
            model_geometry = model_geometry,
            name = "cdw",
            correlation = "density",
            ids = [1, 2],
            coefficients = [1.0, -1.0],
            time_displaced = false,
            integrated = true
        )

        # Initialize the sub-directories to which the various measurements will be written.
        initialize_measurement_directories(comm, simulation_info, measurement_container)
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
            n_therm, n_updates,
            tight_binding_parameters, electron_phonon_parameters, chemical_potential_tuner,
            measurement_container, model_geometry, metadata, rng
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
        tight_binding_parameters    = checkpoint["tight_binding_parameters"]
        electron_phonon_parameters  = checkpoint["electron_phonon_parameters"]
        chemical_potential_tuner    = checkpoint["chemical_potential_tuner"]
        measurement_container       = checkpoint["measurement_container"]
        model_geometry              = checkpoint["model_geometry"]
        metadata                    = checkpoint["metadata"]
        rng                         = checkpoint["rng"]
        n_therm                     = checkpoint["n_therm"]
        n_updates                   = checkpoint["n_updates"]
    end
````

## Setup DQMC simulation
No changes need to made to this section of the code from the previous
[2c) Honeycomb Holstein Model with Checkpointing](@ref) tutorial.

````julia
    # Allocate a single FermionPathIntegral for both spin-up and down electrons.
    fermion_path_integral = FermionPathIntegral(tight_binding_parameters = tight_binding_parameters, β = β, Δτ = Δτ)

    # Initialize FermionPathIntegral type to account for electron-phonon interaction.
    initialize!(fermion_path_integral, electron_phonon_parameters)

    # Initialize imaginary-time propagators for all imaginary-time slices.
    B = initialize_propagators(fermion_path_integral, symmetric=symmetric, checkerboard=checkerboard)

    # Initialize FermionGreensCalculator type.
    fermion_greens_calculator = dqmcf.FermionGreensCalculator(B, β, Δτ, n_stab)

    # Initialize alternate fermion greens calculator required for performing EFA-HMC, reflection and swap updates below.
    fermion_greens_calculator_alt = dqmcf.FermionGreensCalculator(fermion_greens_calculator)

    # Allcoate equal-time electron Green's function matrix.
    G = zeros(eltype(B[1]), size(B[1]))

    # Initialize electron Green's function matrx, also calculating the matrix determinant as the same time.
    logdetG, sgndetG = dqmcf.calculate_equaltime_greens!(G, fermion_greens_calculator)

    # Allocate matrices for various time-displaced Green's function matrices.
    G_ττ = similar(G) # G(τ,τ)
    G_τ0 = similar(G) # G(τ,0)
    G_0τ = similar(G) # G(0,τ)

    # Initialize diagonostic parameters to asses numerical stability.
    δG = zero(logdetG)
    δθ = zero(sgndetG)
````

## Setup EFA-HMC Updates
No changes need to made to this section of the code from the previous
[2c) Honeycomb Holstein Model with Checkpointing](@ref) tutorial.

````julia
    # Number of fermionic time-steps in HMC update.
    Nt = 10

    # Fermionic time-step used in HMC update.
    Δt = π/(2*Ω*Nt)

    # Initialize Hamitlonian/Hybrid monte carlo (HMC) updater.
    hmc_updater = EFAHMCUpdater(
        electron_phonon_parameters = electron_phonon_parameters,
        G = G, Nt = Nt, Δt = Δt
    )
````

## Thermalize system
Here we need to add a call to the [`update_chemical_potential!`](@ref) function
after completeing the updates but before writing the checkpoint file is written.
And again, we need to make sure the include the `chemical_potential_tuner` in the checkpoint file.

````julia
    # Iterate over number of thermalization updates to perform.
    for update in n_therm:N_therm

        # Perform a reflection update.
        (accepted, logdetG, sgndetG) = reflection_update!(
            G, logdetG, sgndetG, electron_phonon_parameters,
            fermion_path_integral = fermion_path_integral,
            fermion_greens_calculator = fermion_greens_calculator,
            fermion_greens_calculator_alt = fermion_greens_calculator_alt,
            B = B, rng = rng
        )

        # Record whether the reflection update was accepted or rejected.
        metadata["reflection_acceptance_rate"] += accepted

        # Perform a swap update.
        (accepted, logdetG, sgndetG) = swap_update!(
            G, logdetG, sgndetG, electron_phonon_parameters,
            fermion_path_integral = fermion_path_integral,
            fermion_greens_calculator = fermion_greens_calculator,
            fermion_greens_calculator_alt = fermion_greens_calculator_alt,
            B = B, rng = rng
        )

        # Record whether the reflection update was accepted or rejected.
        metadata["swap_acceptance_rate"] += accepted

        # Perform an HMC update.
        (accepted, logdetG, sgndetG, δG, δθ) = hmc_update!(
            G, logdetG, sgndetG, electron_phonon_parameters, hmc_updater,
            fermion_path_integral = fermion_path_integral,
            fermion_greens_calculator = fermion_greens_calculator,
            fermion_greens_calculator_alt = fermion_greens_calculator_alt,
            B = B, δG_max = δG_max, δG = δG, δθ = δθ, rng = rng
        )

        # Record whether the HMC update was accepted or rejected.
        metadata["hmc_acceptance_rate"] += accepted

        # Update the chemical potential to achieve the target density.
        (logdetG, sgndetG) = update_chemical_potential!(
            G, logdetG, sgndetG;
            chemical_potential_tuner = chemical_potential_tuner,
            tight_binding_parameters = tight_binding_parameters,
            fermion_path_integral = fermion_path_integral,
            fermion_greens_calculator = fermion_greens_calculator,
            B = B
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
            n_therm  = update + 1,
            n_updates = 1,
            tight_binding_parameters, electron_phonon_parameters, chemical_potential_tuner,
            measurement_container, model_geometry, metadata, rng
        )
    end
````

## Make measurements
Here we need to add a call to the [`update_chemical_potential!`](@ref) function
after making and writing measurements but before writing the checkpoint file is written.
And again, we need to make sure the include the `chemical_potential_tuner` in the checkpoint file.

````julia
    # Reset diagonostic parameters used to monitor numerical stability to zero.
    δG = zero(logdetG)
    δθ = zero(sgndetG)

    # Calculate the bin size.
    bin_size = N_updates ÷ N_bins

    # Iterate over updates and measurements.
    for update in n_updates:N_updates

        # Perform a reflection update.
        (accepted, logdetG, sgndetG) = reflection_update!(
            G, logdetG, sgndetG, electron_phonon_parameters,
            fermion_path_integral = fermion_path_integral,
            fermion_greens_calculator = fermion_greens_calculator,
            fermion_greens_calculator_alt = fermion_greens_calculator_alt,
            B = B, rng = rng
        )

        # Record whether the reflection update was accepted or rejected.
        metadata["reflection_acceptance_rate"] += accepted

        # Perform a swap update.
        (accepted, logdetG, sgndetG) = swap_update!(
            G, logdetG, sgndetG, electron_phonon_parameters,
            fermion_path_integral = fermion_path_integral,
            fermion_greens_calculator = fermion_greens_calculator,
            fermion_greens_calculator_alt = fermion_greens_calculator_alt,
            B = B, rng = rng
        )

        # Record whether the reflection update was accepted or rejected.
        metadata["swap_acceptance_rate"] += accepted

        # Perform an HMC update.
        (accepted, logdetG, sgndetG, δG, δθ) = hmc_update!(
            G, logdetG, sgndetG, electron_phonon_parameters, hmc_updater,
            fermion_path_integral = fermion_path_integral,
            fermion_greens_calculator = fermion_greens_calculator,
            fermion_greens_calculator_alt = fermion_greens_calculator_alt,
            B = B, δG_max = δG_max, δG = δG, δθ = δθ, rng = rng
        )

        # Record whether the HMC update was accepted or rejected.
        metadata["hmc_acceptance_rate"] += accepted

        # Make measurements.
        (logdetG, sgndetG, δG, δθ) = make_measurements!(
            measurement_container,
            logdetG, sgndetG, G, G_ττ, G_τ0, G_0τ,
            fermion_path_integral = fermion_path_integral,
            fermion_greens_calculator = fermion_greens_calculator,
            B = B, δG_max = δG_max, δG = δG, δθ = δθ,
            model_geometry = model_geometry, tight_binding_parameters = tight_binding_parameters,
            coupling_parameters = (electron_phonon_parameters,)
        )

        # Write the bin-averaged measurements to file if update ÷ bin_size == 0.
        write_measurements!(
            measurement_container = measurement_container,
            simulation_info = simulation_info,
            model_geometry = model_geometry,
            update = update,
            bin_size = bin_size,
            Δτ = Δτ
        )

        # Update the chemical potential to achieve the target density.
        (logdetG, sgndetG) = update_chemical_potential!(
            G, logdetG, sgndetG;
            chemical_potential_tuner = chemical_potential_tuner,
            tight_binding_parameters = tight_binding_parameters,
            fermion_path_integral = fermion_path_integral,
            fermion_greens_calculator = fermion_greens_calculator,
            B = B
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
            n_updates = update + 1,
            tight_binding_parameters, electron_phonon_parameters, chemical_potential_tuner,
            measurement_container, model_geometry, metadata, rng
        )
    end
````

## Record simulation metadata
Here we can add a call to the [`save_density_tuning_profile`](@ref), which records the full history
of the chemical potential and density tuning process.

````julia
    # Calculate acceptance rates.
    metadata["hmc_acceptance_rate"] /= (N_updates + N_therm)
    metadata["reflection_acceptance_rate"] /= (N_updates + N_therm)
    metadata["swap_acceptance_rate"] /= (N_updates + N_therm)

    # Record largest numerical error encountered during simulation.
    metadata["dG"] = δG

    # Write simulation metadata to simulation_info.toml file.
    save_simulation_info(simulation_info, metadata)

    # Save the density tuning profile to file.
    save_density_tuning_profile(simulation_info, chemical_potential_tuner)
````

## Post-process results
No changes need to made to this section of the code from the previous
[2c) Honeycomb Holstein Model with Checkpointing](@ref) tutorial.

````julia
    # Process the simulation results, calculating final error bars for all measurements,
    # writing final statisitics to CSV files.
    process_measurements(comm, simulation_info.datafolder, N_bins, time_displaced = true)

    # Merge binary files containing binned data into a single file.
    compress_jld2_bins(comm, folder = simulation_info.datafolder)

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
mpiexecjl -n 16 julia holstein_honeycomb_density_tuning.jl 1 1.0 1.5 0.8 0.0 3 4.0 5000 10000 100 0.5
```
or
```bash
srun julia holstein_honeycomb_density_tuning.jl 1 1.0 1.5 0.8 0.0 3 4.0 5000 10000 100 0.5
```
where the target density is ``\langle n \rangle = 0.8`` and the initial chemical potential is ``\mu = 0.0``.

````julia
# Only excute if the script is run directly from the command line.
if abspath(PROGRAM_FILE) == @__FILE__

    # Initialize MPI
    MPI.Init()

    # Initialize the MPI communicator.
    comm = MPI.COMM_WORLD

    # Run the simulation.
    run_simulation(
        comm;
        sID             = parse(Int,     ARGS[1]),  # Simulation ID.
        Ω               = parse(Float64, ARGS[2]),  # Phonon energy.
        α               = parse(Float64, ARGS[3]),  # Electron-phonon coupling.
        n               = parse(Float64, ARGS[4]),  # Target density.
        μ               = parse(Float64, ARGS[5]),  # Initial chemical potential.
        L               = parse(Int,     ARGS[6]),  # System size.
        β               = parse(Float64, ARGS[7]),  # Inverse temperature.
        N_therm         = parse(Int,     ARGS[8]),  # Number of thermalization updates.
        N_updates       = parse(Int,     ARGS[9]),  # Total number of measurements and measurement updates.
        N_bins          = parse(Int,     ARGS[10]), # Number of times bin-averaged measurements are written to file.
        checkpoint_freq = parse(Float64, ARGS[11]), # Frequency with which checkpoint files are written in hours.
    )

    # Finalize MPI.
    MPI.Finalize()
end
````

