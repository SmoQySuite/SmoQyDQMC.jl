```@meta
EditURL = "../../../tutorials/hubbard_square_mpi.jl"
```

# 1b) Square Hubbard Model with MPI Parallelization
Download this example as a [Julia script](../assets/scripts/tutorials/hubbard_square_mpi.jl).

This tutorial will build on the previous [1a) Square Hubbard Model](@ref) tutorial, demonstrating
how to add parallelization with MPI using the [MPI.jl](https://github.com/JuliaParallel/MPI.jl.git) package.
By this we mean that each MPI process will act as independent walker, running it's own independent DQMC simulation,
with the final reported estimates for measured quantities being the average across all walkers.

The exposition in this tutorial will focus on the changes that need to be made to the [1a) Square Hubbard Model](@ref)
tutorial to introduce MPI parallelization, omitting a more comprehensive discussion of other parts of the code that
were included in the previous tutorial.

## Import Packages
We now need to import the [MPI.jl](https://github.com/JuliaParallel/MPI.jl.git) package as well.

````julia
using SmoQyDQMC
import SmoQyDQMC.LatticeUtilities as lu
import SmoQyDQMC.JDQMCFramework as dqmcf

using Random
using Printf
using MPI
````

## Specify simulation parameters
Here we have introduced the `comm` argument to the `run_simulation` function, which is a type exported by the
[MPI.jl](https://github.com/JuliaParallel/MPI.jl.git) package to facilitate communication and synchronization
between the different MPI processes.

````julia
# Top-level function to run simulation.
function run_simulation(
    comm::MPI.Comm; # MPI communicator.
    # KEYWORD ARGUMENTS
    sID, # Simulation ID.
    U, # Hubbard interaction.
    t′, # Next-nearest-neighbor hopping amplitude.
    μ, # Chemical potential.
    L, # System size.
    β, # Inverse temperature.
    N_therm, # Number of thermalization updates.
    N_updates, # Total number of measurements and measurement updates.
    N_bins, # Number of times bin-averaged measurements are written to file.
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
Now when initializing the [`SimulationInfo`](@ref) type, we also need to include the
MPI process ID `pID`, which can be retrieved using the
[`MPI.Comm_rank`](https://juliaparallel.org/MPI.jl/stable/reference/comm/#MPI.Comm_rank)
function.

We also the [`initialize_datafolder`](@ref) function such that it takes the `comm` as the
first argument. This ensures that all the MPI processes remained synchronized, and none
try proceeding beyond this point until the data folder has been initialized.

````julia
    # Construct the foldername the data will be written to.
    datafolder_prefix = @sprintf "hubbard_square_U%.2f_tp%.2f_mu%.2f_L%d_b%.2f" U t′ μ L β

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
No changes need to made to this section of the code from the previous [1a) Square Hubbard Model](@ref) tutorial.

````julia
    # Initialize random number generator
    rng = Xoshiro(seed)

    # Initialize additiona_info dictionary
    metadata = Dict()

    # Record simulation parameters.
    metadata["N_therm"] = N_therm
    metadata["N_updates"] = N_updates
    metadata["N_bins"] = N_bins
    metadata["n_stab_init"] = n_stab
    metadata["dG_max"] = δG_max
    metadata["symmetric"] = symmetric
    metadata["checkerboard"] = checkerboard
    metadata["seed"] = seed
    metadata["avg_acceptance_rate"] = 0.0
````

## Initialize model
No changes need to made to this section of the code from the previous [1a) Square Hubbard Model](@ref) tutorial.

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
        t_mean  = [t, t, t′, t′], # defines corresponding mean hopping amplitude
        t_std   = [0., 0., 0., 0.], # defines corresponding standard deviation in hopping amplitude
        ϵ_mean  = [0.], # set mean on-site energy for each orbital in unit cell
        ϵ_std   = [0.], # set standard deviation of on-site energy or each orbital in unit cell
        μ       = μ # set chemical potential
    )

    # Define the Hubbard interaction in the model.
    hubbard_model = HubbardModel(
        shifted   = false, # if true, then Hubbard interaction is instead parameterized as U⋅nup⋅ndn
        U_orbital = [1], # orbitals in unit cell with Hubbard interaction.
        U_mean    = [U], # mean Hubbard interaction strength for corresponding orbital species in unit cell.
        U_std     = [0.], # standard deviation of Hubbard interaction strength for corresponding orbital species in unit cell.
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
No changes need to made to this section of the code from the previous [1a) Square Hubbard Model](@ref) tutorial.

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

    # Apply Ising Hubbard-Stranonvich (HS) transformation to decouple the Hubbard interaction,
    # and initialize the corresponding HS fields that will be sampled in the DQMC simulation.
    hubbard_stratonovich_params = HubbardIsingHSParameters(
        β = β, Δτ = Δτ,
        hubbard_parameters = hubbard_params,
        rng = rng
    )
````

## Initialize meuasurements
The only change we need to make to this section of the code from the previous [1a) Square Hubbard Model](@ref) tutorial
is to add the `comm` as the first argument to the [`initialize_measurement_directories`](@ref) function.
The ensures that not of the MPI processes proceed beyond that point until the directory structure has been initialized.

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

    # Initialize the sub-directories to which the various measurements will be written.
    initialize_measurement_directories(comm, simulation_info, measurement_container)
````

## Setup DQMC simulation
No changes need to made to this section of the code from the previous [1a) Square Hubbard Model](@ref) tutorial.

````julia
    # Allocate FermionPathIntegral type for both the spin-up and spin-down electrons.
    fermion_path_integral_up = FermionPathIntegral(tight_binding_parameters = tight_binding_parameters, β = β, Δτ = Δτ)
    fermion_path_integral_dn = FermionPathIntegral(tight_binding_parameters = tight_binding_parameters, β = β, Δτ = Δτ)

    # Initialize FermionPathIntegral type for both the spin-up and spin-down electrons to account for Hubbard interaction.
    initialize!(fermion_path_integral_up, fermion_path_integral_dn, hubbard_params)

    # Initialize FermionPathIntegral type for both the spin-up and spin-down electrons to account for the current
    # Hubbard-Stratonovich field configuration.
    initialize!(fermion_path_integral_up, fermion_path_integral_dn, hubbard_stratonovich_params)

    # Initialize imaginary-time propagators for all imaginary-time slices for spin-up and spin-down electrons.
    Bup = initialize_propagators(fermion_path_integral_up, symmetric=symmetric, checkerboard=checkerboard)
    Bdn = initialize_propagators(fermion_path_integral_dn, symmetric=symmetric, checkerboard=checkerboard)

    # Initialize FermionGreensCalculator type for spin-up and spin-down electrons.
    fermion_greens_calculator_up = dqmcf.FermionGreensCalculator(Bup, β, Δτ, n_stab)
    fermion_greens_calculator_dn = dqmcf.FermionGreensCalculator(Bdn, β, Δτ, n_stab)

    # Allcoate matrices for spin-up and spin-down electron Green's function matrices.
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

    # Initialize diagonostic parameters to asses numerical stability.
    δG = zero(logdetGup)
    δθ = zero(sgndetGup)
````

## Thermalize system
No changes need to made to this section of the code from the previous [1a) Square Hubbard Model](@ref) tutorial.

````julia
    # Iterate over number of thermalization updates to perform.
    for n in 1:N_therm

        # Perform sweep all imaginary-time slice and orbitals, attempting an update to every HS field.
        (acceptance_rate, logdetGup, sgndetGup, logdetGdn, sgndetGdn, δG, δθ) = local_updates!(
            Gup, logdetGup, sgndetGup, Gdn, logdetGdn, sgndetGdn,
            hubbard_stratonovich_params,
            fermion_path_integral_up = fermion_path_integral_up,
            fermion_path_integral_dn = fermion_path_integral_dn,
            fermion_greens_calculator_up = fermion_greens_calculator_up,
            fermion_greens_calculator_dn = fermion_greens_calculator_dn,
            Bup = Bup, Bdn = Bdn, δG_max = δG_max, δG = δG, δθ = δθ, rng = rng,
            update_stabilization_frequency = true
        )

        # Record acceptance rate for sweep.
        metadata["avg_acceptance_rate"] += acceptance_rate
    end
````

## Make measurements
No changes need to made to this section of the code from the previous [1a) Square Hubbard Model](@ref) tutorial.

````julia
    # Reset diagonostic parameters used to monitor numerical stability to zero.
    δG = zero(logdetGup)
    δθ = zero(sgndetGup)

    # Calculate the bin size.
    bin_size = N_updates ÷ N_bins

    # Iterate over updates and measurements.
    for update in 1:N_updates

        # Perform sweep all imaginary-time slice and orbitals, attempting an update to every HS field.
        (acceptance_rate, logdetGup, sgndetGup, logdetGdn, sgndetGdn, δG, δθ) = local_updates!(
            Gup, logdetGup, sgndetGup, Gdn, logdetGdn, sgndetGdn,
            hubbard_stratonovich_params,
            fermion_path_integral_up = fermion_path_integral_up,
            fermion_path_integral_dn = fermion_path_integral_dn,
            fermion_greens_calculator_up = fermion_greens_calculator_up,
            fermion_greens_calculator_dn = fermion_greens_calculator_dn,
            Bup = Bup, Bdn = Bdn, δG_max = δG_max, δG = δG, δθ = δθ, rng = rng,
            update_stabilization_frequency = true
        )

        # Record acceptance rate.
        metadata["avg_acceptance_rate"] += acceptance_rate

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
            coupling_parameters = (hubbard_params, hubbard_stratonovich_params)
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
    end
````

## Record simulation metadata
No changes need to made to this section of the code from the previous [1a) Square Hubbard Model](@ref) tutorial.

````julia
    # Normalize acceptance rate.
    metadata["avg_acceptance_rate"] /=  (N_therm + N_updates)

    metadata["n_stab_final"] = fermion_greens_calculator_up.n_stab

    # Record largest numerical error.
    metadata["dG"] = δG

    # Write simulation summary TOML file.
    save_simulation_info(simulation_info, metadata)
````

## Post-rocess results
The main change we need to make from the previos [1a) Square Hubbard Model](@ref) tutorial is to call
the [`process_measurements`](@ref), [`compute_correlation_ratio`](@ref) and [`compress_jld2_bins`](@ref) function
such that the first argument is the `comm` object, thereby ensuring a parallelized version of each method is called.

````julia
    # Set the number of bins used to calculate the error in measured observables.
    n_bins = N_bins

    # Process the simulation results, calculating final error bars for all measurements,
    # writing final statisitics to CSV files.
    process_measurements(comm, simulation_info.datafolder, n_bins, time_displaced = false)

    # Calculate AFM correlation ratio.
    Rafm, ΔRafm = compute_correlation_ratio(
        comm;
        folder = simulation_info.datafolder,
        correlation = "spin_z",
        type = "equal-time",
        id_pairs = [(1, 1)],
        coefs = [1.0],
        k_point = (L÷2, L÷2), # Corresponds to Q_afm = (π/a, π/a).
        num_bins = n_bins
    )

    # Record the AFM correlation ratio mean and standard deviation.
    metadata["Rafm_real_mean"] = real(Rafm)
    metadata["Rafm_imag_mean"] = imag(Rafm)
    metadata["Rafm_std"]       = ΔRafm

    # Write simulation summary TOML file.
    save_simulation_info(simulation_info, metadata)

    # Merge binary files containing binned data into a single file.
    compress_jld2_bins(comm, folder = simulation_info.datafolder)

    return nothing
end # end of run_simulation function
````

## Execute script
Here we first need to initialize MPI using the
[`MPI.Init`](https://juliaparallel.org/MPI.jl/stable/reference/environment/#MPI.Init) command.
Then, we need to make sure to pass the `comm = MPI.COMM_WORLD` to the `run_simulation` function.
At the very end of simulation it is good practice to run the `MPI.Finalize()` function even though
it is typically not strictly required.

Only excute if the script is run directly from the command line.

````julia
if abspath(PROGRAM_FILE) == @__FILE__

    # Initialize MPI
    MPI.Init()

    # Initialize the MPI communicator.
    comm = MPI.COMM_WORLD

    # Run the simulation, reading in command line arguments.
    run_simulation(
        comm;
        sID       = parse(Int,     ARGS[1]), # Simulation ID.
        U         = parse(Float64, ARGS[2]), # Hubbard interaction.
        t′        = parse(Float64, ARGS[3]), # Next-nearest-neighbor hopping amplitude.
        μ         = parse(Float64, ARGS[4]), # Chemical potential.
        L         = parse(Int,     ARGS[5]), # System size.
        β         = parse(Float64, ARGS[6]), # Inverse temperature.
        N_therm   = parse(Int,     ARGS[7]), # Number of thermalization updates.
        N_updates = parse(Int,     ARGS[8]), # Total number of measurements and measurement updates.
        N_bins    = parse(Int,     ARGS[9])  # Number of times bin-averaged measurements are written to file.
    )

    # Finalize MPI.
    MPI.Finalize()
end
````

Here is an example of what the command to run this script might look like:
```bash
mpiexecjl -n 16 julia hubbard_square_mpi.jl 1 5.0 -0.25 -2.0 4 4.0 2500 10000 100
```
This will 16 MPI processes, each running and independent simulation using a different random seed
the final results arrived at by averaging over all 16 walkers.
Here `mpiexecjl` is the MPI exectuable that can be easily install using the directions
found [here](https://juliaparallel.org/MPI.jl/stable/usage/#Julia-wrapper-for-mpiexec) in the
[MPI.jl](https://github.com/JuliaParallel/MPI.jl) documentation. However, you can substitute a
different MPI executable here if one is already configured on your system.

Also, when submitting jobs via [SLURM](https://slurm.schedmd.com/documentation.html)
on a High-Performance Computing (HPC) cluster, if a default MPI exectuable
is already configured on the system, as is frequently the case, then the script can likely be run inside the
`*.sh` job file using the [`srun`](https://slurm.schedmd.com/srun.html) command:
```bash
srun julia hubbard_square_mpi.jl 1 5.0 -0.25 -2.0 4 4.0 2500 10000 100
```
The `srun` command should automatically detect the number of available cores requested by the job and run
the script using the MPI executable with the appropriate number of processes.

