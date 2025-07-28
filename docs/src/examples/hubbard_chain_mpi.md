```@meta
EditURL = "../../../examples/hubbard_chain_mpi.jl"
```

# Hubbard Chain with MPI
Download this example as a [Julia script](../assets/scripts/examples/hubbard_chain_mpi.jl).

In this example we simulate the same system as in example 1, the half-filled Hubbard chain.
However, in this example we use MPI, via the [`MPI.jl`](https://github.com/JuliaParallel/MPI.jl.git),
to perform multiple simulations in parallel, with the results getting automatically averaged over
at the end of the simulation.
A short test simulation using the script associated with this example can be run as
```
> mpiexecjl -n 8 julia hubbard_chain_mpi.jl 1 6.0 0.0 8.0 16 2000 10000 50
```
This would result in `8` identical simulations being run in parallel.
It possible that this command may need to be modified slightly depending on how MPI is set up on your system.
For more information I recommend you refer to the
[`MPI.jl documenation`](https://juliaparallel.org/MPI.jl/stable/).

Below you will find the source code from the julia script linked at the top of this page,
but with additional comments giving more detailed explanations for what certain parts of the code are doing.

````julia
using LinearAlgebra
using Random
using Printf
using MPI

using SmoQyDQMC
import SmoQyDQMC.LatticeUtilities  as lu
import SmoQyDQMC.JDQMCFramework    as dqmcf
import SmoQyDQMC.JDQMCMeasurements as dqmcm

# initialize MPI
MPI.Init()

# Define top-level function for running DQMC simulation
function run_hubbard_chain_simulation(sID, U, μ, β, L, N_burnin, N_updates, N_bins; filepath = ".")

    # Initialize the MPI communicator.
    comm = MPI.COMM_WORLD

    # Construct the foldername the data will be written to.
    datafolder_prefix = @sprintf "hubbard_chain_U%.2f_mu%.2f_L%d_b%.2f" U μ L β

    # Get the MPI comm rank, which fixes the process ID (pID).
    pID = MPI.Comm_rank(comm)

    # Initialize an instance of the SimulationInfo type.
    simulation_info = SimulationInfo(
        filepath = filepath,
        datafolder_prefix = datafolder_prefix,
        sID = sID,
        pID = pID
    )

    # Synchronize all the MPI processes.
    # Here we need to make sure the data folder is initialized before letting
    # all the various processes move beyond this point.
    MPI.Barrier(comm)

    # Initialize a random number generator that will be used throughout the simulation.
    seed = abs(rand(Int))
    rng = Xoshiro(seed)

    # Set the discretization in imaginary time for the DQMC simulation.
    Δτ = 0.10

    # Calculate the length of the imaginary time axis, Lτ = β/Δτ.
    Lτ = dqmcf.eval_length_imaginary_axis(β, Δτ)

    # This flag indicates whether or not to use the checkboard approximation to
    # represent the exponentiated hopping matrix exp(-Δτ⋅K)
    checkerboard = false

    # Whether the propagator matrices should be represented using the
    # symmetric form B = exp(-Δτ⋅K/2)⋅exp(-Δτ⋅V)⋅exp(-Δτ⋅K/2)
    # or the asymetric form B = exp(-Δτ⋅V)⋅exp(-Δτ⋅K)
    symmetric = false

    # Set the initial period in imaginary time slices with which the Green's function matrices
    # will be recomputed using a numerically stable procedure.
    n_stab = 10

    # Specify the maximum allowed error in any element of the Green's function matrix that is
    # corrected by performing numerical stabiliziation.
    δG_max = 1e-6

    # Calculate the bins size.
    bin_size = div(N_updates, N_bins)

    # Initialize a dictionary to store additional information about the simulation.
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

    #######################
    ### DEFINE THE MODEL ##
    #######################

    # Initialize an instance of the type UnitCell.
    unit_cell = lu.UnitCell(lattice_vecs = [[1.0]],
                            basis_vecs   = [[0.0]])

    # Initialize an instance of the type Lattice.
    lattice = lu.Lattice(
        L = [L],
        periodic = [true]
    )

    # Initialize an instance of the ModelGeometry type.
    model_geometry = ModelGeometry(unit_cell, lattice)

    # Define the nearest-neighbor bond for a 1D chain.
    bond = lu.Bond(orbitals = (1,1), displacement = [1])

    # Add this bond to the model, by adding it to the ModelGeometry type.
    bond_id = add_bond!(model_geometry, bond)

    # Define nearest-neighbor hopping amplitude, setting the energy scale for the system.
    t = 1.0

    # Define the tight-binding model
    tight_binding_model = TightBindingModel(
        model_geometry = model_geometry,
        t_bonds = [bond], # defines hopping
        t_mean = [t],     # defines corresponding hopping amplitude
        μ = μ,            # set chemical potential
        ϵ_mean = [0.]     # set the (mean) on-site energy
    )

    # Initialize the Hubbard interaction in the model.
    hubbard_model = HubbardModel(
        shifted = false, # If true, Hubbard interaction instead parameterized as U⋅nup⋅ndn
        U_orbital = [1],
        U_mean = [U],
    )

    # Initialize the directory the data will be written to.
    initialize_datafolder(simulation_info)

    # Write the model summary to file.
    model_summary(
        simulation_info = simulation_info,
        β = β, Δτ = Δτ,
        model_geometry = model_geometry,
        tight_binding_model = tight_binding_model,
        interactions = (hubbard_model,)
    )

    #########################################
    ### INITIALIZE FINITE MODEL PARAMETERS ##
    #########################################


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

    # Apply Ising Hubbard-Stranonvich (HS) transformation, and initialize
    # corresponding HS fields that will be sampled in DQMC simulation.
    hubbard_ising_parameters = HubbardIsingHSParameters(
        β = β, Δτ = Δτ,
        hubbard_parameters = hubbard_parameters,
        rng = rng
    )

    ##############################
    ### INITIALIZE MEASUREMENTS ##
    ##############################

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

    # Initialize the sub-directories to which the various measurements will be written.
    initialize_measurement_directories(
        simulation_info = simulation_info,
        measurement_container = measurement_container
    )

    # Synchronize all the MPI processes.
    # We need to ensure the sub-directories the measurements will be written are created
    # prior to letting any of the processes move beyond this point.
    MPI.Barrier(comm)

    #############################
    ### SET-UP DQMC SIMULATION ##
    #############################


    # Allocate FermionPathIntegral type for both the spin-up and spin-down electrons.
    fermion_path_integral_up = FermionPathIntegral(tight_binding_parameters = tight_binding_parameters, β = β, Δτ = Δτ)
    fermion_path_integral_dn = FermionPathIntegral(tight_binding_parameters = tight_binding_parameters, β = β, Δτ = Δτ)

    # Initialize the FermionPathIntegral type for both the spin-up and spin-down electrons.
    initialize!(fermion_path_integral_up, fermion_path_integral_dn, hubbard_parameters)
    initialize!(fermion_path_integral_up, fermion_path_integral_dn, hubbard_ising_parameters)

    # Initialize the imaginary-time propagators for each imaginary-time slice for both the
    # spin-up and spin-down electrons.
    Bup = initialize_propagators(fermion_path_integral_up, symmetric=symmetric, checkerboard=checkerboard)
    Bdn = initialize_propagators(fermion_path_integral_dn, symmetric=symmetric, checkerboard=checkerboard)

    # Initialize FermionGreensCalculator for the spin-up and spin-down electrons.
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
    Gup_ττ = similar(Gup) # G↑(τ,τ)
    Gup_τ0 = similar(Gup) # G↑(τ,0)
    Gup_0τ = similar(Gup) # G↑(0,τ)
    Gdn_ττ = similar(Gdn) # G↓(τ,τ)
    Gdn_τ0 = similar(Gdn) # G↓(τ,0)
    Gdn_0τ = similar(Gdn) # G↓(0,τ)

    # Initialize variables to keep track of the largest numerical error in the
    # Green's function matrices corrected by numerical stabalization.
    δG = zero(typeof(logdetGup))
    δθ = zero(typeof(sgndetGup))

    ####################################
    ### BURNIN/THERMALIZATION UPDATES ##
    ####################################

    # Iterate over burnin/thermalization updates.
    for n in 1:N_burnin

        # Perform a sweep through the lattice, attemping an update to each Ising HS field.
        (acceptance_rate, logdetGup, sgndetGup, logdetGdn, sgndetGdn, δG, δθ) = local_updates!(
            Gup, logdetGup, sgndetGup, Gdn, logdetGdn, sgndetGdn,
            hubbard_ising_parameters,
            fermion_path_integral_up = fermion_path_integral_up,
            fermion_path_integral_dn = fermion_path_integral_dn,
            fermion_greens_calculator_up = fermion_greens_calculator_up,
            fermion_greens_calculator_dn = fermion_greens_calculator_dn,
            Bup = Bup, Bdn = Bdn, δG_max = δG_max, δG = δG, δθ = δθ, rng = rng
        )

        # Record the acceptance rate for the attempted local updates to the HS fields.
        additional_info["local_acceptance_rate"] += acceptance_rate
    end

    ################################
    ### START MAKING MEAUSREMENTS ##
    ################################

    # Re-initialize variables to keep track of the largest numerical error in the
    # Green's function matrices corrected by numerical stabalization.
    δG = zero(typeof(logdetGup))
    δθ = zero(typeof(sgndetGup))

    # Iterate over the number of bin, i.e. the number of time measurements will be dumped to file.
    for bin in 1:N_bins

        # Iterate over the number of updates and measurements performed in the current bin.
        for n in 1:bin_size

            # Perform a sweep through the lattice, attemping an update to each Ising HS field.
            (acceptance_rate, logdetGup, sgndetGup, logdetGdn, sgndetGdn, δG, δθ) = local_updates!(
                Gup, logdetGup, sgndetGup, Gdn, logdetGdn, sgndetGdn,
                hubbard_ising_parameters,
                fermion_path_integral_up = fermion_path_integral_up,
                fermion_path_integral_dn = fermion_path_integral_dn,
                fermion_greens_calculator_up = fermion_greens_calculator_up,
                fermion_greens_calculator_dn = fermion_greens_calculator_dn,
                Bup = Bup, Bdn = Bdn, δG_max = δG_max, δG = δG, δθ = δθ, rng = rng
            )

            # Record the acceptance rate for the attempted local updates to the HS fields.
            additional_info["local_acceptance_rate"] += acceptance_rate

            # Make measurements, with the results being added to the measurement container.
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
                coupling_parameters = (hubbard_parameters, hubbard_ising_parameters)
            )
        end

        # Write the average measurements for the current bin to file.
        write_measurements!(
            measurement_container = measurement_container,
            simulation_info = simulation_info,
            model_geometry = model_geometry,
            bin = bin,
            bin_size = bin_size,
            Δτ = Δτ
        )
    end

    # Calculate acceptance rate for local updates.
    additional_info["local_acceptance_rate"] /= (N_updates + N_burnin)

    # Record the final numerical stabilization period that the simulation settled on.
    additional_info["n_stab_final"] = fermion_greens_calculator_up.n_stab

    # Record the maximum numerical error corrected by numerical stablization.
    additional_info["dG"] = δG

    # Write simulation summary TOML file.
    save_simulation_info(simulation_info, additional_info)

    #################################
    ### PROCESS SIMULATION RESULTS ##
    #################################

    # Synchronize all the MPI processes.
    # Before we prcoess the binned data to get the final averages and error bars
    # we need to make sure all the simulations running in parallel have run to
    # completion.
    MPI.Barrier(comm)

    # Have the primary MPI process calculate the final error bars for all measurements,
    # writing final statisitics to CSV files.
    if iszero(simulation_info.pID)
        process_measurements(simulation_info.datafolder, N_bins)
    end

    return nothing
end


# Only excute if script is run directly from the command line.
if abspath(PROGRAM_FILE) == @__FILE__

    # Read in the command line arguments.
    sID = parse(Int, ARGS[1]) # simulation ID
    U = parse(Float64, ARGS[2])
    μ = parse(Float64, ARGS[3])
    β = parse(Float64, ARGS[4])
    L = parse(Int, ARGS[5])
    N_burnin = parse(Int, ARGS[6])
    N_updates = parse(Int, ARGS[7])
    N_bins = parse(Int, ARGS[8])

    # Run the simulation.
    run_hubbard_chain_simulation(sID, U, μ, β, L, N_burnin, N_updates, N_bins)

    # Finalize MPI (not strictly required).
    MPI.Finalize()
end
````

