```@meta
EditURL = "../../../examples/hubbard_chain_checkpoint.jl"
```

# Hubbard Chain with Checkpointing
Download this example as a [Julia script](../assets/scripts/examples/hubbard_chain_checkpoint.jl).

In this script we take the script from the previous example and introduce
checkpointing, so that if the simulation is killed at some point it can resumed
from the previous checkpoint. It is important to note that how checkpointing
is introduced in this script is not unique, and other checkpointing schemes could be implemented
in a script. For instance, in this script the checkpointing is implemented such that the
number of checkpoints written to file during the simulation is a fixed number at the start of the simulation.
It is possible, though slightly more involved, to implement a checkpointing scheme that instead writes checkpoints
to file based on the wall clock and the amount of time that has passed since the previous checkpoint was written to file.

To write the checkpoints we use the package
[`JLD2`](https://github.com/JuliaIO/JLD2.jl.git), which allows the checkpoint files to be
written to file as binary files that are HDF5 compatible.
In this script, following the thermalziation/burnin updates, a checkpoint is written to file
whenever measurements are written to file, so a total of `N_bins` checkpoints are written following the
initial thermalization/burnin period of the simulation. During the thermalization/burnin updates, checkpoints
are written with the same frequency as they are going to be once measurements start getting made.

Below you will find the source code from the julia script linked at the top of this page,
but with additional comments giving more detailed explanations for what certain parts of the code are doing.

````julia
using LinearAlgebra
using Random
using Printf
using MPI

# Import JLD2 package for write checkpoints during the simulation
# to file as a binary file.
using JLD2

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

    # Define checkpoint filename.
    # We implement three checkpoint files, an old, current and new one,
    # that get cycled through to ensure a checkpoint file always exists in the off
    # chance that the simulation is killed while a checkpoint is getting written to file.
    # Additionally, each simulation that is running in parallel with MPI will have their own
    # checkpoints written to file.
    datafolder = simulation_info.datafolder
    sID        = simulation_info.sID
    pID        = simulation_info.pID
    checkpoint_name_old          = @sprintf "checkpoint_sID%d_pID%d_old.jld2" sID pID
    checkpoint_filename_old      = joinpath(datafolder, checkpoint_name_old)
    checkpoint_name_current      = @sprintf "checkpoint_sID%d_pID%d_current.jld2" sID pID
    checkpoint_filename_current  = joinpath(datafolder, checkpoint_name_current)
    checkpoint_name_new          = @sprintf "checkpoint_sID%d_pID%d_new.jld2" sID pID
    checkpoint_filename_new      = joinpath(datafolder, checkpoint_name_new)

    ######################################################
    ### DEFINE SOME RELEVANT DQMC SIMULATION PARAMETERS ##
    ######################################################

    # Set the discretization in imaginary time for the DQMC simulation.
    Δτ = 0.10

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

    # Initialize variables to keep track of the largest numerical error in the
    # Green's function matrices corrected by numerical stabalization.
    δG = 0.0
    δθ = 0.0

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

    #######################################################
    ### BRANCHING BEHAVIOR BASED ON WHETHER STARTING NEW ##
    ### SIMULAIOTN OR RESUMING PREVIOUS SIMULATION.      ##
    #######################################################

    # Synchronize all the MPI processes.
    MPI.Barrier(comm)

    # If starting a new simulation.
    if !simulation_info.resuming

        # Initialize a random number generator that will be used throughout the simulation.
        seed = abs(rand(Int))
        rng = Xoshiro(seed)

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

        #############################
        ### WRITE FIRST CHECKPOINT ##
        #############################

        # Calculate the bin size.
        bin_size = div(N_updates, N_bins)

        # Calculate the number of thermalization/burnin bins.
        # This determines the number times the simulations checkpoints
        # during the initial thermalziation/burnin period.
        N_bins_burnin = div(N_burnin, bin_size)

        # Initialize variable to keep track of the current burnin bin.
        n_bin_burnin = 1

        # Initialize variable to keep track of the current bin.
        n_bin = 1

        # Write an initial checkpoint to file.
        JLD2.jldsave(
            checkpoint_filename_current;
            rng, additional_info,
            N_burnin, N_updates, N_bins,
            bin_size, N_bins_burnin, n_bin_burnin, n_bin,
            measurement_container,
            model_geometry,
            tight_binding_parameters,
            hubbard_parameters,
            dG = δG, dtheta = δθ, n_stab = n_stab
        )

    # If resuming simulation from previous checkpoint.
    else

        # Initialize checkpoint to nothing before it is loaded.
        checkpoint = nothing

        # Try loading in the new checkpoint.
        if isfile(checkpoint_filename_new)
            try
                # Load the new checkpoint.
                checkpoint = JLD2.load(checkpoint_filename_new)
            catch
                nothing
            end
        end

        # Try loading in the current checkpoint.
        if isfile(checkpoint_filename_current) && isnothing(checkpoint)
            try
                # Load the current checkpoint.
                checkpoint = JLD2.load(checkpoint_filename_current)
            catch
                nothing
            end
        end

        # Try loading in the current checkpoint.
        if isfile(checkpoint_filename_old) && isnothing(checkpoint)
            try
                # Load the old checkpoint.
                checkpoint = JLD2.load(checkpoint_filename_old)
            catch
                nothing
            end
        end

        # Throw an error if no checkpoint was succesfully loaded.
        if isnothing(checkpoint)
            error("Failed to load checkpoint successfully!")
        end

        # Unpack the contents of the checkpoint.
        rng                      = checkpoint["rng"]
        additional_info          = checkpoint["additional_info"]
        N_burnin                 = checkpoint["N_burnin"]
        N_updates                = checkpoint["N_updates"]
        N_bins                   = checkpoint["N_bins"]
        bin_size                 = checkpoint["bin_size"]
        N_bins_burnin            = checkpoint["N_bins_burnin"]
        n_bin_burnin             = checkpoint["n_bin_burnin"]
        n_bin                    = checkpoint["n_bin"]
        model_geometry           = checkpoint["model_geometry"]
        measurement_container    = checkpoint["measurement_container"]
        tight_binding_parameters = checkpoint["tight_binding_parameters"]
        hubbard_parameters       = checkpoint["hubbard_parameters"]
        hubbard_ising_parameters = checkpoint["hubbard_ising_parameters"]
        δG                       = checkpoint["dG"]
        δθ                       = checkpoint["dtheta"]
        n_stab                   = checkpoint["n_stab"]
    end

    # Synchronize all the MPI processes.
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

    ####################################
    ### BURNIN/THERMALIZATION UPDATES ##
    ####################################

    # Iterate over burnin/thermalization bins.
    for bin in n_bin_burnin:N_bins_burnin

        # Iterate over updates in current bin.
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
        end

        # Write the new checkpoint to file.
        JLD2.jldsave(
            checkpoint_filename_new;
            rng, additional_info,
            N_burnin, N_updates, N_bins,
            bin_size, N_bins_burnin,
            n_bin_burnin = bin + 1,
            n_bin = 1,
            measurement_container,
            model_geometry,
            tight_binding_parameters,
            hubbard_parameters,
            hubbard_ising_parameters,
            dG = δG, dtheta = δθ, n_stab = n_stab
        )
        # Make the current checkpoint the old checkpoint.
        mv(checkpoint_filename_current, checkpoint_filename_old, force = true)
        # Make the new checkpoint the current checkpoint.
        mv(checkpoint_filename_new, checkpoint_filename_current, force = true)
    end

    ################################
    ### START MAKING MEAUSREMENTS ##
    ################################

    # Iterate over the number of bin, i.e. the number of time measurements will be dumped to file.
    for bin in n_bin:N_bins

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

        # Write the new checkpoint to file.
        JLD2.jldsave(
            checkpoint_filename_new;
            rng, additional_info,
            N_burnin, N_updates, N_bins,
            bin_size, N_bins_burnin,
            n_bin_burnin = N_bins_burnin+1,
            n_bin = bin + 1,
            measurement_container,
            model_geometry,
            tight_binding_parameters,
            hubbard_parameters,
            hubbard_ising_parameters,
            dG = δG, dtheta = δθ, n_stab = n_stab
        )
        # Make the current checkpoint the old checkpoint.
        mv(checkpoint_filename_current, checkpoint_filename_old, force = true)
        # Make the new checkpoint the current checkpoint.
        mv(checkpoint_filename_new, checkpoint_filename_current, force = true)
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

