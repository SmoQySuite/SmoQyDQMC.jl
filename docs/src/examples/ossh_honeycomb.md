```@meta
EditURL = "../../../examples/ossh_honeycomb.jl"
```

# Honeycomb Optical Su-Schrieffer-Heeger Model
Download this example as a [Julia script](../assets/scripts/examples/ossh_honeycomb.jl).

This script simulates the optical Su-Schrieffer-Heeger (oSSH) Model on a honeycomb lattice, as defined in
in [Phys. Rev. B 110, 115130](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.110.115130).
This script can be used to reproduce the results presented in this paper, which investigated the emergence of
Kekulé valence bond solid order in the honeycomb oSSH model at half-filling via DQMC simulations performed
using the [SmoQyDQMC.jl](https://github.com/SmoQySuite/SmoQyDQMC.jl.git) package.

````julia
using SmoQyDQMC
import SmoQyDQMC.LatticeUtilities as lu
import SmoQyDQMC.JDQMCFramework as dqmcf

using Random
using Printf
using MPI

# Top-level function to run simulation.
function run_simulation(
    comm::MPI.Comm; # MPI communicator.
    # KEYWORD ARGUMENTS
    sID, # Simulation ID.
    Ω, # Phonon energy.
    λ, # Electron-phonon coupling.
    μ, # Chemical potential.
    L, # System size.
    β, # Inverse temperature.
    N_therm, # Number of thermalization updates.
    N_measurements, # Total number of measurements and measurement updates.
    N_bins, # Number of times bin-averaged measurements are written to file.
    checkpoint_freq, # Frequency with which checkpoint files are written in hours.
    runtime_limit = Inf, # Simulation runtime limit in hours.
    Nt = 4, # Number of time-steps in HMC update.
    Δτ = 0.05, # Discretization in imaginary time.
    n_stab = 5, # Numerical stabilization period in imaginary-time slices.
    δG_max = 1e-6, # Threshold for numerical error corrected by stabilization.
    update_stabilization_frequency = true, # Whether to dynamically update stabilization frequency.
    symmetric = true, # Whether symmetric propagator definition is used.
    checkerboard = true, # Whether checkerboard approximation is used.
    seed = abs(rand(Int)), # Seed for random number generator.
    filepath = "." # Filepath to where data folder will be created.
)

    # Record when the simulation began.
    start_timestamp = time()

    # Convert runtime limit from hours to seconds.
    runtime_limit = runtime_limit * 60.0^2

    # Convert checkpoint frequency from hours to seconds.
    checkpoint_freq = checkpoint_freq * 60.0^2

    # Construct the foldername the data will be written to.
    datafolder_prefix = @sprintf "ossh_honeycomb_w%.2f_l%.2f_mu%.2f_L%d_b%.2f" Ω λ μ L β

    # Get MPI process ID.
    pID = MPI.Comm_rank(comm)

    # Initialize simulation info.
    simulation_info = SimulationInfo(
        filepath = filepath,
        datafolder_prefix = datafolder_prefix,
        write_bins_concurrent = (L > 7),
        sID = sID,
        pID = pID
    )

    # Initialize the directory the data will be written to.
    initialize_datafolder(comm, simulation_info)

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
        metadata["Nt"] = Nt
        metadata["N_therm"] = N_therm
        metadata["N_measurements"] = N_measurements
        metadata["N_bins"] = N_bins
        metadata["n_stab"] = n_stab
        metadata["dG_max"] = δG_max
        metadata["symmetric"] = symmetric
        metadata["checkerboard"] = checkerboard
        metadata["seed"] = seed
        metadata["hmc_acceptance_rate"] = 0.0
        metadata["radial_acceptance_rate"] = 0.0
        metadata["swap_acceptance_rate"] = 0.0

        # label the sublattice A and B
        A, B = 1, 2

        # Define lattice vectors.
        a1 = [+3/2, +√3/2]
        a2 = [+3/2, -√3/2]

        # Define basis vectors for two orbitals in the honeycomb unit cell.
        rA = [0.0, 0.0] # Location of sublattice A orbital in unit cell.
        rB = [1.0, 0.0] # Location of sublattice B orbital in unit cell.

        # Define the unit cell.
        unit_cell = lu.UnitCell(
            lattice_vecs = [a1, a2],
            basis_vecs = [rA, rB]
        )

        # Define finite lattice with periodic boundary conditions.
        lattice = lu.Lattice(
            L = [L, L],
            periodic = [true, true]
        )

        # Initialize model geometry.
        model_geometry = ModelGeometry(unit_cell, lattice)

        # Define the first nearest-neighbor bond in a honeycomb lattice.
        bond_AB_1 = lu.Bond(orbitals = (A,B), displacement = [0,0])

        # Add the first nearest-neighbor bond in a honeycomb lattice to the model.
        bond_AB_1_id = add_bond!(model_geometry, bond_AB_1)

        # Define the second nearest-neighbor bond in a honeycomb lattice.
        bond_AB_2 = lu.Bond(orbitals = (A,B), displacement = [-1,0])

        # Add the second nearest-neighbor bond in a honeycomb lattice to the model.
        bond_AB_2_id = add_bond!(model_geometry, bond_AB_2)

        # Define the third nearest-neighbor bond in a honeycomb lattice.
        bond_AB_3 = lu.Bond(orbitals = (A,B), displacement = [0,-1])

        # Add the third nearest-neighbor bond in a honeycomb lattice to the model.
        bond_AB_3_id = add_bond!(model_geometry, bond_AB_3)

        # Set nearest-neighbor hopping amplitude to unity,
        # setting the energy scale in the model.
        t = 1.0

        # Define the honeycomb tight-binding model.
        tight_binding_model = TightBindingModel(
            model_geometry = model_geometry,
            t_bonds = [bond_AB_1, bond_AB_2, bond_AB_3],
            t_mean = [t, t, t],
            μ  = μ,
            ϵ_mean = [0.0, 0.0]
        )

        # Initialize a null electron-phonon model.
        electron_phonon_model = ElectronPhononModel(
            model_geometry = model_geometry,
            tight_binding_model = tight_binding_model
        )

        # Define the sublattice A x-direction displacement phonon.
        phonon_A_x = PhononMode(
            basis_vec = rA,
            Ω_mean = Ω
        )

        # Add the sublattice A x-direction displacement phonon to the electron-phonon model.
        phonon_A_x_id = add_phonon_mode!(
            electron_phonon_model = electron_phonon_model,
            phonon_mode = phonon_A_x
        )

        # Define the sublattice A y-direction displacement phonon.
        phonon_A_y = PhononMode(
            basis_vec = rA,
            Ω_mean = Ω
        )

        # Add the sublattice A y-direction displacement phonon to the electron-phonon model.
        phonon_A_y_id = add_phonon_mode!(
            electron_phonon_model = electron_phonon_model,
            phonon_mode = phonon_A_y
        )

        # Define the sublattice B x-direction displacement phonon.
        phonon_B_x = PhononMode(
            basis_vec = rB,
            Ω_mean = Ω
        )

        # Add the sublattice B x-direction displacement phonon to the electron-phonon model.
        phonon_B_x_id = add_phonon_mode!(
            electron_phonon_model = electron_phonon_model,
            phonon_mode = phonon_B_x
        )

        # Define the sublattice B y-direction displacement phonon.
        phonon_B_y = PhononMode(
            basis_vec = rB,
            Ω_mean = Ω
        )

        # Add the sublattice B y-direction displacement phonon to the electron-phonon model.
        phonon_B_y_id = add_phonon_mode!(
            electron_phonon_model = electron_phonon_model,
            phonon_mode = phonon_B_y
        )

        # calculate microscopic coupling constant λ = α²/(M⋅Ω²⋅t) with ħ = Kb = a = M = t = 1
        α = Ω * sqrt(λ)

        # Defines x-direction SSH modulation of first A to B nearest-neighbor hopping amplitude.
        ossh_AB_1_x_coupling = SSHCoupling(
            model_geometry = model_geometry,
            tight_binding_model = tight_binding_model,
            phonon_ids = (phonon_A_x_id, phonon_B_x_id),
            bond = bond_AB_1,
            α_mean = α
        )

        # Add x-direction SSH modulation of first A to B nearest-neighbor hopping amplitude to e-ph model.
        ossh_AB_1_x_coupling_id = add_ssh_coupling!(
            electron_phonon_model = electron_phonon_model,
            ssh_coupling = ossh_AB_1_x_coupling,
            tight_binding_model = tight_binding_model
        )

        # Defines x-direction SSH modulation of second A to B nearest-neighbor hopping amplitude.
        ossh_AB_2_x_coupling = SSHCoupling(
            model_geometry = model_geometry,
            tight_binding_model = tight_binding_model,
            phonon_ids = (phonon_A_x_id, phonon_B_x_id),
            bond = bond_AB_2,
            α_mean = -α*cos(π/3)
        )

        # Add x-direction SSH modulation of second A to B nearest-neighbor hopping amplitude to e-ph model.
        ossh_AB_2_x_coupling_id = add_ssh_coupling!(
            electron_phonon_model = electron_phonon_model,
            ssh_coupling = ossh_AB_2_x_coupling,
            tight_binding_model = tight_binding_model
        )

        # Defines y-direction SSH modulation of second A to B nearest-neighbor hopping amplitude.
        ossh_AB_2_y_coupling = SSHCoupling(
            model_geometry = model_geometry,
            tight_binding_model = tight_binding_model,
            phonon_ids = (phonon_A_y_id, phonon_B_y_id),
            bond = bond_AB_2,
            α_mean = -α*cos(π/6)
        )

        # Add y-direction SSH modulation of second A to B nearest-neighbor hopping amplitude to e-ph model.
        ossh_AB_2_y_coupling_id = add_ssh_coupling!(
            electron_phonon_model = electron_phonon_model,
            ssh_coupling = ossh_AB_2_y_coupling,
            tight_binding_model = tight_binding_model
        )

        # Defines x-direction SSH modulation of third A to B nearest-neighbor hopping amplitude.
        ossh_AB_3_x_coupling = SSHCoupling(
            model_geometry = model_geometry,
            tight_binding_model = tight_binding_model,
            phonon_ids = (phonon_A_x_id, phonon_B_x_id),
            bond = bond_AB_3,
            α_mean = -α*cos(π/3)
        )

        # Add x-direction SSH modulation of third A to B nearest-neighbor hopping amplitude to e-ph model.
        ossh_AB_3_x_coupling_id = add_ssh_coupling!(
            electron_phonon_model = electron_phonon_model,
            ssh_coupling = ossh_AB_3_x_coupling,
            tight_binding_model = tight_binding_model
        )

        # Defines y-direction SSH modulation of third A to B nearest-neighbor hopping amplitude.
        ossh_AB_3_y_coupling = SSHCoupling(
            model_geometry = model_geometry,
            tight_binding_model = tight_binding_model,
            phonon_ids = (phonon_A_y_id, phonon_B_y_id),
            bond = bond_AB_3,
            α_mean = α*cos(π/6)
        )

        # Add y-direction SSH modulation of third A to B nearest-neighbor hopping amplitude to e-ph model.
        ossh_AB_3_y_coupling_id = add_ssh_coupling!(
            electron_phonon_model = electron_phonon_model,
            ssh_coupling = ossh_AB_3_y_coupling,
            tight_binding_model = tight_binding_model
        )

        # Write model summary TOML file specifying Hamiltonian that will be simulated.
        model_summary(
            simulation_info = simulation_info,
            β = β, Δτ = Δτ,
            model_geometry = model_geometry,
            tight_binding_model = tight_binding_model,
            interactions = (electron_phonon_model,)
        )

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
                (A, A), (B, B), (A, B), (B, A)
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
                (phonon_A_x_id, phonon_A_x_id),
                (phonon_A_y_id, phonon_A_y_id),
                (phonon_B_x_id, phonon_B_x_id),
                (phonon_B_y_id, phonon_B_y_id),
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
                (A, A), (B, B), (A, B), (B, A)
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
                (A, A), (B, B), (A, B), (B, A)
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
                (A, A), (B, B), (A, B), (B, A)
            ]
        )

        # Initialize the spin-z correlation function measurement.
        initialize_correlation_measurements!(
            measurement_container = measurement_container,
            model_geometry = model_geometry,
            correlation = "bond",
            time_displaced = false,
            integrated = true,
            pairs = [
                (bond_AB_1_id, bond_AB_1_id), (bond_AB_1_id, bond_AB_2_id), (bond_AB_1_id, bond_AB_3_id),
                (bond_AB_2_id, bond_AB_1_id), (bond_AB_2_id, bond_AB_2_id), (bond_AB_2_id, bond_AB_3_id),
                (bond_AB_3_id, bond_AB_1_id), (bond_AB_3_id, bond_AB_2_id), (bond_AB_3_id, bond_AB_3_id),
            ]
        )

        # Initialize measurement of electron Green's function traced
        # over both orbitals in the unit cell.
        initialize_composite_correlation_measurement!(
            measurement_container = measurement_container,
            model_geometry = model_geometry,
            name = "tr_greens",
            correlation = "greens",
            id_pairs = [(A, A), (B, B)],
            coefficients = [1.0, 1.0],
            time_displaced = true,
        )

        # Initialize CDW correlation measurement.
        initialize_composite_correlation_measurement!(
            measurement_container = measurement_container,
            model_geometry = model_geometry,
            name = "cdw",
            correlation = "density",
            ids = [A, B],
            coefficients = [1.0, -1.0],
            time_displaced = false,
            integrated = true
        )

        # Initialize C3 BOW correlation measurement
        initialize_composite_correlation_measurement!(
            measurement_container = measurement_container,
            model_geometry = model_geometry,
            name = "C3_bond",
            correlation = "bond",
            ids = [bond_AB_1_id, bond_AB_2_id, bond_AB_3_id],
            coefficients = [1.0, exp(-1im*2π/3), exp(-1im*4π/3)],
            time_displaced = false,
            integrated = true
        )

        # Initialize alternate C3 BOW correlation measurement
        initialize_composite_correlation_measurement!(
            measurement_container = measurement_container,
            model_geometry = model_geometry,
            name = "C3_alt_bond",
            correlation = "bond",
            id_pairs = [
                (bond_AB_1_id, bond_AB_1_id), (bond_AB_2_id, bond_AB_2_id), (bond_AB_3_id, bond_AB_3_id),
                (bond_AB_1_id, bond_AB_2_id), (bond_AB_2_id, bond_AB_1_id),
                (bond_AB_1_id, bond_AB_3_id), (bond_AB_3_id, bond_AB_1_id),
                (bond_AB_2_id, bond_AB_3_id), (bond_AB_3_id, bond_AB_2_id)
            ],
            coefficients = [
                2.0, 2.0, 2.0,
                -1.0, -1.0,
                -1.0, -1.0,
                -1.0, -1.0
            ],
            time_displaced = false,
            integrated = true
        )

        # Initialize C3 phonon green's correlation measurement
        initialize_composite_correlation_measurement!(
            measurement_container = measurement_container,
            model_geometry = model_geometry,
            name = "tr_phonon_greens",
            correlation = "phonon_greens",
            id_pairs = [
                (phonon_A_x_id, phonon_A_x_id), (phonon_A_y_id, phonon_A_y_id),
                (phonon_B_x_id, phonon_B_x_id), (phonon_B_y_id, phonon_B_y_id)
            ],
            coefficients = [1.0, 1.0, 1.0, 1.0],
            time_displaced = false,
            integrated = true
        )

        # Write initial checkpoint file.
        checkpoint_timestamp = write_jld2_checkpoint(
            comm,
            simulation_info;
            checkpoint_freq = checkpoint_freq,
            start_timestamp = start_timestamp,
            runtime_limit = runtime_limit,
            # Contents of checkpoint file below.
            n_therm, n_measurements,
            tight_binding_parameters, electron_phonon_parameters,
            measurement_container, model_geometry, metadata, rng
        )

    # If resuming a previous simulation.
    else

        # Load the checkpoint file.
        checkpoint, checkpoint_timestamp = read_jld2_checkpoint(simulation_info)

        # Unpack contents of checkpoint dictionary.
        tight_binding_parameters = checkpoint["tight_binding_parameters"]
        electron_phonon_parameters = checkpoint["electron_phonon_parameters"]
        measurement_container = checkpoint["measurement_container"]
        model_geometry = checkpoint["model_geometry"]
        metadata = checkpoint["metadata"]
        rng = checkpoint["rng"]
        n_therm = checkpoint["n_therm"]
        n_measurements = checkpoint["n_measurements"]
    end

    # calculate length of imaginary time axis.
    Lτ = round(Int, β/Δτ)

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

    # Allocate equal-time electron Green's function matrix.
    G = zeros(eltype(B[1]), size(B[1]))

    # Initialize electron Green's function matrix, also calculating the matrix determinant as the same time.
    logdetG, sgndetG = dqmcf.calculate_equaltime_greens!(G, fermion_greens_calculator)

    # Allocate matrices for various time-displaced Green's function matrices.
    G_ττ = similar(G) # G(τ,τ)
    G_τ0 = similar(G) # G(τ,0)
    G_0τ = similar(G) # G(0,τ)

    # Initialize diagnostic parameters to asses numerical stability.
    δG = zero(logdetG)
    δθ = zero(logdetG)

    # Initialize Hamiltonian/Hybrid monte carlo (HMC) updater.
    hmc_updater = EFAHMCUpdater(
        electron_phonon_parameters = electron_phonon_parameters,
        G = G, Nt = Nt, Δt = π/(2*Nt)
    )

    # Iterate over number of thermalization updates to perform.
    for update in n_therm:N_therm

        # Perform a reflection update.
        (accepted, logdetG, sgndetG) = radial_update!(
            G, logdetG, sgndetG, electron_phonon_parameters,
            fermion_path_integral = fermion_path_integral,
            fermion_greens_calculator = fermion_greens_calculator,
            fermion_greens_calculator_alt = fermion_greens_calculator_alt,
            B = B, rng = rng, σ = 1.0
        )

        # Record whether the reflection update was accepted or rejected.
        metadata["radial_acceptance_rate"] += accepted

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
            B = B, δG_max = δG_max, δG = δG, δθ = δθ, rng = rng,
            update_stabilization_frequency = update_stabilization_frequency
        )

        # Record whether the HMC update was accepted or rejected.
        metadata["hmc_acceptance_rate"] += accepted

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
            tight_binding_parameters, electron_phonon_parameters,
            measurement_container, model_geometry, metadata, rng
        )
    end

    # Reset diagnostic parameters used to monitor numerical stability to zero.
    δG = zero(logdetG)
    δθ = zero(logdetG)

    # Calculate the bin size.
    bin_size = N_measurements ÷ N_bins

    # Iterate over updates and measurements.
    for measurement in n_measurements:N_measurements

        # println("measurement = ", measurement)

        # Perform a reflection update.
        (accepted, logdetG, sgndetG) = radial_update!(
            G, logdetG, sgndetG, electron_phonon_parameters,
            fermion_path_integral = fermion_path_integral,
            fermion_greens_calculator = fermion_greens_calculator,
            fermion_greens_calculator_alt = fermion_greens_calculator_alt,
            B = B, rng = rng, σ = 1.0
        )

        # Record whether the reflection update was accepted or rejected.
        metadata["radial_acceptance_rate"] += accepted

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
            B = B, δG_max = δG_max, δG = δG, δθ = δθ, rng = rng,
            update_stabilization_frequency = update_stabilization_frequency
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
            update_stabilization_frequency = update_stabilization_frequency,
            coupling_parameters = (electron_phonon_parameters,)
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
            n_therm = N_therm + 1,
            n_measurements = measurement + 1,
            tight_binding_parameters, electron_phonon_parameters,
            measurement_container, model_geometry, metadata, rng
        )
    end

    # Merge binned data into a single HDF5 file.
    merge_bins(simulation_info)

    # Calculate acceptance rates.
    metadata["hmc_acceptance_rate"] /= (N_measurements + N_therm)
    metadata["radial_acceptance_rate"] /= (N_measurements + N_therm)
    metadata["swap_acceptance_rate"] /= (N_measurements + N_therm)

    # Record largest numerical error encountered during simulation.
    metadata["dG"] = δG

    # Write simulation metadata to simulation_info.toml file.
    save_simulation_info(simulation_info, metadata)

    # Process the simulation results, calculating final error bars for all measurements.
    # writing final statistics to CSV files.
    process_measurements(
        comm,
        datafolder = simulation_info.datafolder,
        n_bins = N_bins,
        export_to_csv = true,
        scientific_notation = true,
        decimals = 7,
        delimiter = " "
    )

    process_measurements(
        pIDs = pID,
        datafolder = simulation_info.datafolder,
        n_bins = N_bins,
        export_to_csv = true,
        scientific_notation = true,
        decimals = 7,
        delimiter = " "
    )

    # KVBS correlation ratio.
    Rkvbs, ΔRkvbs = compute_composite_correlation_ratio(
        comm;
        datafolder = simulation_info.datafolder,
        name = "C3_bond",
        type = "equal-time",
        q_point = (L÷3, 2L÷3),
        q_neighbors = [
            (L÷3+1, 2L÷3+0), (L÷3+0, 2L÷3+1), (L÷3+1, 2L÷3+1),
            (L÷3-1, 2L÷3+0), (L÷3+0, 2L÷3-1), (L÷3-1, 2L÷3-1)
        ]
    )

    # Record the KVBS correlation ratio mean and standard deviation.
    metadata["Rkvbs_mean_real"] = real(Rkvbs)
    metadata["Rkvbs_mean_imag"] = imag(Rkvbs)
    metadata["Rkvbs_std"] = ΔRkvbs

    # KVBS alternate correlation ratio.
    Rkvbs_alt, ΔRkvbs_alt = compute_composite_correlation_ratio(
        comm;
        datafolder = simulation_info.datafolder,
        name = "C3_alt_bond",
        type = "equal-time",
        q_point = (L÷3, 2L÷3),
        q_neighbors = [
            (L÷3+1, 2L÷3+0), (L÷3+0, 2L÷3+1), (L÷3+1, 2L÷3+1),
            (L÷3-1, 2L÷3+0), (L÷3+0, 2L÷3-1), (L÷3-1, 2L÷3-1)
        ]
    )

    # Record the KVBS alternate correlation ratio mean and standard deviation.
    metadata["Rkvbs_alt_mean_real"] = real(Rkvbs_alt)
    metadata["Rkvbs_alt_mean_imag"] = imag(Rkvbs_alt)
    metadata["Rkvbs_alt_std"] = ΔRkvbs_alt

    # Write simulation summary TOML file.
    save_simulation_info(simulation_info, metadata)

    # Rename the data folder to indicate the simulation is complete.
    simulation_info = rename_complete_simulation(
        comm, simulation_info,
        delete_jld2_checkpoints = true
    )

    return nothing
end # end of run_simulation function

# Only execute if the script is run directly from the command line.
if abspath(PROGRAM_FILE) == @__FILE__

    # Initialize MPI
    MPI.Init()

    # Initialize the MPI communicator.
    comm = MPI.COMM_WORLD

    # Run the simulation.
    run_simulation(
        comm;
        sID = parse(Int, ARGS[1]), # Simulation ID.
        Ω = parse(Float64, ARGS[2]), # Phonon energy.
        λ = parse(Float64, ARGS[3]), # Electron-phonon coupling.
        μ = parse(Float64, ARGS[4]), # Chemical potential.
        L = parse(Int, ARGS[5]), # System size.
        β = parse(Float64, ARGS[6]), # Inverse temperature.
        N_therm = parse(Int, ARGS[7]), # Number of thermalization updates.
        N_measurements = parse(Int, ARGS[8]), # Total number of measurements and measurement updates.
        N_bins = parse(Int, ARGS[9]), # Number of times bin-averaged measurements are written to file.
        checkpoint_freq = parse(Float64, ARGS[10]), # Frequency with which checkpoint files are written in hours.
        runtime_limit = checkbounds(Bool, ARGS, 11) ? parse(Float64, ARGS[11]) : Float64(Inf) # runtime limit in hours.
    )

    # Finalize MPI.
    MPI.Finalize()
end
````

