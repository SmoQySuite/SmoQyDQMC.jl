# # Square Holstein-Hubbard Model
#
# In this example we simulate the Holstein-Hubbard model on a square lattice, with a Hamiltonian given by
# ```math
# \begin{align*}
# \hat{H} & = \sum_i \left( \frac{1}{2M}\hat{P}_i^2 + \frac{1}{2} M \Omega^2 \hat{X}_i^2 \right) \\
# & - t \sum_{\sigma,\langle i,j \rangle} \left( \hat{c}_{\sigma,i}^{\dagger} \hat{c}_{\sigma,j}^{\phantom\dagger} + \hat{c}_{\sigma,j}^{\dagger} \hat{c}_{\sigma,i}^{\phantom\dagger} \right) - \mu \sum_{\sigma,i} \hat{n}_{\sigma,i} \\
# & + U \sum_i \left( \hat{n}_{\uparrow,i} - \tfrac{1}{2} \right)\left( \hat{n}_{\downarrow,j} + \tfrac{1}{2} \right) + \alpha \sum_{\sigma,i} \hat{X}_i \left( \hat{n}_{\sigma,i} - \tfrac{1}{2} \right)
# \end{align*}
# ```
# in which the fluctuations in the position of dispersionless phonon modes placed on each site in the lattice modulate the on-site energy.
# In the above expression ``\hat{c}^\dagger_{\sigma,i} \ (\hat{c}^{\phantom \dagger}_{\sigma,i})`` creation (annihilation) operator
# a spin ``\sigma`` electron on site ``i`` in the lattice, and ``\hat{n}_{\sigma,i} = \hat{c}^\dagger_{\sigma,i} \hat{c}^{\phantom \dagger}_{\sigma,i}``
# is corresponding electron number operator. Here the sum over ``\langle i,j \rangle`` runs over all nearest-neighbor pairs of sites in the lattice,
# and ``t`` and ``\mu`` are the nearest-neighbor hopping amplitude and chemical potential, respectively.
# The phonon position (momentum) operator for the dispersionless phonon mode on site ``i``
# is given by ``\hat{X}_{i} \ (\hat{P}_{i})``, where ``\Omega`` and ``M`` are the phonon frequency and associated ion mass respectively.
# Therefore, the strength of the electron-phonon coupling is controlled by the parameter ``\alpha``.
# Lastly, the parameter ``U`` controls the strength of the on-site Hubbard interaction.

# Note that this example script comes with all the bells and whistles so to speak, including support for MPI parallelization as well as checkpointing.

using SmoQyDQMC
import SmoQyDQMC.LatticeUtilities as lu
import SmoQyDQMC.JDQMCFramework as dqmcf

using Random
using Printf
using MPI

## Top-level function to run simulation.
function run_simulation(
    comm::MPI.Comm; # MPI communicator.
    ## KEYWORD ARGUMENTS
    sID, # Simulation ID.
    U, # Hubbard interaction strength.
    Ω, # Phonon energy.
    α, # Electron-phonon coupling.
    μ, # Chemical potential.
    L, # System size.
    β, # Inverse temperature.
    N_therm, # Number of thermalization updates.
    N_measurements, # Total number of measurements.
    N_bins, # Number of times bin-averaged measurements are written to file.
    N_local_updates, # Number of local update sweeps per HMC update and measurement.
    checkpoint_freq, # Frequency with which checkpoint files are written in hours.
    runtime_limit = Inf, # Simulation runtime limit in hours.
    Nt = 8, # Number of time-steps in HMC update.
    Δτ = 0.05, # Discretization in imaginary time.
    n_stab = 10, # Numerical stabilization period in imaginary-time slices.
    δG_max = 1e-6, # Threshold for numerical error corrected by stabilization.
    symmetric = false, # Whether symmetric propagator definition is used.
    checkerboard = false, # Whether checkerboard approximation is used.
    seed = abs(rand(Int)), # Seed for random number generator.
    filepath = "." # Filepath to where data folder will be created.
)

    ## Record when the simulation began.
    start_timestamp = time()

    ## Convert runtime limit from hours to seconds.
    runtime_limit = runtime_limit * 60.0^2

    ## Convert checkpoint frequency from hours to seconds.
    checkpoint_freq = checkpoint_freq * 60.0^2

    ## Construct the foldername the data will be written to.
    datafolder_prefix = @sprintf "square_hol_hub_U%.2f_w%.2f_a%.2f_mu%.2f_L%d_b%.2f" U Ω α μ L β

    ## Get MPI process ID.
    pID = MPI.Comm_rank(comm)

    ## Initialize simulation info.
    simulation_info = SimulationInfo(
        filepath = filepath,
        datafolder_prefix = datafolder_prefix,
        write_bins_concurrent = (L > 10),
        sID = sID,
        pID = pID
    )

    ## Initialize the directory the data will be written to.
    initialize_datafolder(comm, simulation_info)

    ## If starting a new simulation i.e. not resuming a previous simulation.
    if !simulation_info.resuming

        ## Begin thermalization updates from start.
        n_therm = 1

        ## Begin measurement updates from start.
        n_measurements = 1

        ## Initialize random number generator
        rng = Xoshiro(seed)

        ## Initialize metadata dictionary
        metadata = Dict()

        ## Record simulation parameters.
        metadata["Nt"] = Nt
        metadata["N_therm"] = N_therm
        metadata["N_measurements"] = N_measurements
        metadata["N_bins"] = N_bins
        metadata["n_stab"] = n_stab
        metadata["dG_max"] = δG_max
        metadata["symmetric"] = symmetric
        metadata["checkerboard"] = checkerboard
        metadata["seed"] = seed
        metadata["reflection_acceptance_rate"] = 0.0
        metadata["hmc_acceptance_rate"] = 0.0
        metadata["local_acceptance_rate"] = 0.0

        ## Initialize an instance of the type UnitCell.
        unit_cell = lu.UnitCell(
            lattice_vecs = [[1.0,0.0],
                            [0.0,1.0]],
            basis_vecs = [[0.0,0.0]]
        )

        ## Initialize an instance of the type Lattice.
        lattice = lu.Lattice(
            L = [L,L],
            periodic = [true,true]
        )

        ## Get the number of sites in the lattice.
        N = lu.nsites(unit_cell, lattice)

        ## Initialize an instance of the ModelGeometry type.
        model_geometry = ModelGeometry(unit_cell, lattice)

        ## Define the nearest-neighbor bond in the x-direction.
        bond_px = lu.Bond(orbitals = (1,1), displacement = [1,0])

        ## Add this bond in x-direction to the model geometry.
        bond_px_id = add_bond!(model_geometry, bond_px)

        ## Define the nearest-neighbor bond in the y-direction.
        bond_py = lu.Bond(orbitals = (1,1), displacement = [0,1])

        ## Add this bond in y-direction to the model geometry.
        bond_py_id = add_bond!(model_geometry, bond_py)

        ## Define the nearest-neighbor bond in the -x-direction.
        bond_nx = lu.Bond(orbitals = (1,1), displacement = [-1,0])

        ## Add this bond in +x-direction to the model geometry.
        bond_nx_id = add_bond!(model_geometry, bond_nx)

        ## Define the nearest-neighbor bond in the -y-direction.
        bond_ny = lu.Bond(orbitals = (1,1), displacement = [0,-1])

        ## Add this bond in +y-direction to the model geometry.
        bond_ny_id = add_bond!(model_geometry, bond_ny)

        ## Define nearest-neighbor hopping amplitude, setting the energy scale for the system.
        t = 1.0

        ## Define the tight-binding model
        tight_binding_model = TightBindingModel(
            model_geometry = model_geometry,
            t_bonds = [bond_px, bond_py], # defines hopping
            t_mean = [t, t], # defines corresponding hopping amplitude
            μ = μ, # set chemical potential
            ϵ_mean = [0.] # set the (mean) on-site energy
        )

        ## Define the Hubbard interaction in the model.
        hubbard_model = HubbardModel(
            ph_sym_form = true, # if particle-hole symmetric form for Hubbard interaction is used.
            U_orbital   = [1], # orbitals in unit cell with Hubbard interaction.
            U_mean      = [U], # mean Hubbard interaction strength for corresponding orbital species in unit cell.
            U_std       = [0.], # standard deviation of Hubbard interaction strength for corresponding orbital species in unit cell.
        )

        ## Initialize a null electron-phonon model.
        electron_phonon_model = ElectronPhononModel(
            model_geometry = model_geometry,
            tight_binding_model = tight_binding_model
        )

        ## Define a dispersionless phonon mode.
        phonon = PhononMode(
            basis_vec = [0.0,0.0],
            Ω_mean = Ω
        )

        ## Add dispersionless phonon mode phonon to the model.
        phonon_id = add_phonon_mode!(
            electron_phonon_model = electron_phonon_model,
            phonon_mode = phonon
        )

        ## Define first local Holstein coupling for first phonon mode.
        holstein_coupling = HolsteinCoupling(
            model_geometry = model_geometry,
            phonon_id = phonon_id,
            orbital_id = 1,
            displacement = [0, 0],
            α_mean = α,
            ph_sym_form = true,
        )

        ## Add Holstein coupling to the model.
        holstein_coupling_id = add_holstein_coupling!(
            electron_phonon_model = electron_phonon_model,
            holstein_coupling = holstein_coupling,
            model_geometry = model_geometry
        )

        ## Write a model summary to file.
        model_summary(
            simulation_info = simulation_info,
            β = β, Δτ = Δτ,
            model_geometry = model_geometry,
            tight_binding_model = tight_binding_model,
            interactions = (electron_phonon_model,)
        )

        ## Initialize tight-binding parameters.
        tight_binding_parameters = TightBindingParameters(
            tight_binding_model = tight_binding_model,
            model_geometry = model_geometry,
            rng = rng
        )

        ## Initialize Hubbard interaction parameters.
        hubbard_parameters = HubbardParameters(
            model_geometry = model_geometry,
            hubbard_model = hubbard_model,
            rng = rng
        )

        ## Apply Spin Channel Hirsch Hubbard-Stratonovich transformation.
        hst_parameters = HubbardSpinHirschHST(
            β = β, Δτ = Δτ,
            hubbard_parameters = hubbard_parameters,
            rng = rng
        )

        ## Initialize electron-phonon parameters.
        electron_phonon_parameters = ElectronPhononParameters(
            β = β, Δτ = Δτ,
            electron_phonon_model = electron_phonon_model,
            tight_binding_parameters = tight_binding_parameters,
            model_geometry = model_geometry,
            rng = rng
        )

        ## Initialize the container that measurements will be accumulated into.
        measurement_container = initialize_measurement_container(model_geometry, β, Δτ)

        ## Initialize the tight-binding model related measurements, like the hopping energy.
        initialize_measurements!(measurement_container, tight_binding_model)

        ## Initialize the electron-phonon interaction related measurements.
        initialize_measurements!(measurement_container, electron_phonon_model)

        ## Initialize the hubbard interaction related measurements.
        initialize_measurements!(measurement_container, hubbard_model)

        ## Initialize the single-particle electron Green's function measurement.
        initialize_correlation_measurements!(
            measurement_container = measurement_container,
            model_geometry = model_geometry,
            correlation = "greens",
            time_displaced = true,
            pairs = [
                ## Measure green's functions for all pairs or orbitals.
                (1, 1),
            ]
        )

        ## Initialize the single-particle electron Green's function measurement.
        initialize_correlation_measurements!(
            measurement_container = measurement_container,
            model_geometry = model_geometry,
            correlation = "phonon_greens",
            time_displaced = true,
            pairs = [
                (phonon_id, phonon_id)
            ]
        )

        ## Initialize density correlation function measurement.
        initialize_correlation_measurements!(
            measurement_container = measurement_container,
            model_geometry = model_geometry,
            correlation = "density",
            time_displaced = false,
            integrated = true,
            pairs = [
                (1, 1),
            ]
        )

        ## Initialize the pair correlation function measurement.
        initialize_correlation_measurements!(
            measurement_container = measurement_container,
            model_geometry = model_geometry,
            correlation = "pair",
            time_displaced = false,
            integrated = true,
            pairs = [
                (1, 1),
            ]
        )

        ## Initialize the spin-z correlation function measurement.
        initialize_correlation_measurements!(
            measurement_container = measurement_container,
            model_geometry = model_geometry,
            correlation = "spin_z",
            time_displaced = false,
            integrated = true,
            pairs = [
                (1, 1),
            ]
        )

        ## Initialize the d-wave pair susceptibility measurement.
        initialize_composite_correlation_measurement!(
            measurement_container = measurement_container,
            model_geometry = model_geometry,
            name = "d-wave",
            correlation = "pair",
            ids = [bond_px_id, bond_nx_id, bond_py_id, bond_ny_id],
            coefficients = [0.5, 0.5, -0.5, -0.5],
            time_displaced = true,
            integrated = true
        )

        ## Initialize trace of bond correlation measurements.
        initialize_composite_correlation_measurement!(
            measurement_container = measurement_container,
            model_geometry = model_geometry,
            name = "tr_bonds",
            correlation = "bond",
            id_pairs = [
                (bond_px_id, bond_px_id),
                (bond_py_id, bond_py_id)
            ],
            coefficients = [
                +1.0, +1.0
            ],
            time_displaced = false,
            integrated = true
        )

        ## Write initial checkpoint file.
        checkpoint_timestamp = write_jld2_checkpoint(
            comm,
            simulation_info;
            checkpoint_freq = checkpoint_freq,
            start_timestamp = start_timestamp,
            runtime_limit = runtime_limit,
            exit_code = 13,
            ## Contents of checkpoint file below.
            n_therm, n_measurements,
            tight_binding_parameters, electron_phonon_parameters,
            hubbard_parameters, hst_parameters,
            measurement_container, model_geometry, metadata, rng
        )

    ## If resuming a previous simulation.
    else

        ## Load the checkpoint file.
        checkpoint, checkpoint_timestamp = read_jld2_checkpoint(simulation_info)

        ## Unpack contents of checkpoint dictionary.
        tight_binding_parameters = checkpoint["tight_binding_parameters"]
        hubbard_parameters = checkpoint["hubbard_parameters"]
        hst_parameters = checkpoint["hst_parameters"]
        electron_phonon_parameters = checkpoint["electron_phonon_parameters"]
        measurement_container = checkpoint["measurement_container"]
        model_geometry = checkpoint["model_geometry"]
        metadata = checkpoint["metadata"]
        rng = checkpoint["rng"]
        n_therm = checkpoint["n_therm"]
        n_measurements = checkpoint["n_measurements"]
    end

    ## Allocate FermionPathIntegral type for both the spin-up and spin-down electrons.
    fermion_path_integral_up = FermionPathIntegral(tight_binding_parameters = tight_binding_parameters, β = β, Δτ = Δτ)
    fermion_path_integral_dn = FermionPathIntegral(tight_binding_parameters = tight_binding_parameters, β = β, Δτ = Δτ)

    ## Initialize FermionPathIntegral type for both the spin-up and spin-down electrons to account for Hubbard interaction.
    initialize!(fermion_path_integral_up, fermion_path_integral_dn, hubbard_parameters)

    ## Initialize FermionPathIntegral type for both the spin-up and spin-down electrons to account for the current
    ## Hubbard-Stratonovich field configuration.
    initialize!(fermion_path_integral_up, fermion_path_integral_dn, hst_parameters)

    ## Initialize FermionPathIntegral type to account for electron-phonon interaction.
    initialize!(fermion_path_integral_up, fermion_path_integral_dn, electron_phonon_parameters)

    ## Initialize imaginary-time propagators for all imaginary-time slices for spin-up and spin-down electrons.
    Bup = initialize_propagators(fermion_path_integral_up, symmetric=symmetric, checkerboard=checkerboard)
    Bdn = initialize_propagators(fermion_path_integral_dn, symmetric=symmetric, checkerboard=checkerboard)

    ## Initialize FermionGreensCalculator type for spin-up and spin-down electrons.
    fermion_greens_calculator_up = dqmcf.FermionGreensCalculator(Bup, β, Δτ, n_stab)
    fermion_greens_calculator_dn = dqmcf.FermionGreensCalculator(Bdn, β, Δτ, n_stab)

    ## Initialize alternate FermionGreensCalculator type for performing reflection updates.
    fermion_greens_calculator_up_alt = dqmcf.FermionGreensCalculator(fermion_greens_calculator_up)
    fermion_greens_calculator_dn_alt = dqmcf.FermionGreensCalculator(fermion_greens_calculator_dn)

    ## Allocate matrices for spin-up and spin-down electron Green's function matrices.
    Gup = zeros(eltype(Bup[1]), size(Bup[1]))
    Gdn = zeros(eltype(Bdn[1]), size(Bdn[1]))

    ## Initialize the spin-up and spin-down electron Green's function matrices, also
    ## calculating their respective determinants as the same time.
    logdetGup, sgndetGup = dqmcf.calculate_equaltime_greens!(Gup, fermion_greens_calculator_up)
    logdetGdn, sgndetGdn = dqmcf.calculate_equaltime_greens!(Gdn, fermion_greens_calculator_dn)

    ## Allocate matrices for various time-displaced Green's function matrices.
    Gup_ττ = similar(Gup) # Gup(τ,τ)
    Gup_τ0 = similar(Gup) # Gup(τ,0)
    Gup_0τ = similar(Gup) # Gup(0,τ)
    Gdn_ττ = similar(Gdn) # Gdn(τ,τ)
    Gdn_τ0 = similar(Gdn) # Gdn(τ,0)
    Gdn_0τ = similar(Gdn) # Gdn(0,τ)

    ## Initialize diagnostic parameters to asses numerical stability.
    δG = zero(logdetGup)
    δθ = zero(logdetGup)

    ## Initialize Hamiltonian/Hybrid monte carlo (HMC) updater.
    hmc_updater = EFAHMCUpdater(
        electron_phonon_parameters = electron_phonon_parameters,
        G = Gup, Nt = Nt, Δt = π/(2*Nt)
    )

    ## Iterate over number of thermalization updates to perform.
    for update in n_therm:N_therm

        ## Perform a reflection update.
        (accepted, logdetGup, sgndetGup, logdetGdn, sgndetGdn) = reflection_update!(
            Gup, logdetGup, sgndetGup,
            Gdn, logdetGdn, sgndetGdn,
            electron_phonon_parameters,
            fermion_path_integral_up = fermion_path_integral_up,
            fermion_path_integral_dn = fermion_path_integral_dn,
            fermion_greens_calculator_up = fermion_greens_calculator_up,
            fermion_greens_calculator_dn = fermion_greens_calculator_dn,
            fermion_greens_calculator_up_alt = fermion_greens_calculator_up_alt,
            fermion_greens_calculator_dn_alt = fermion_greens_calculator_dn_alt,
            Bup = Bup, Bdn = Bdn, rng = rng
        )

        ## Record whether the reflection update was accepted or rejected.
        metadata["reflection_acceptance_rate"] += accepted

        ## Perform an HMC update.
        (accepted, logdetGup, sgndetGup, logdetGdn, sgndetGdn, δG, δθ) = hmc_update!(
            Gup, logdetGup, sgndetGup,
            Gdn, logdetGdn, sgndetGdn,
            electron_phonon_parameters, hmc_updater,
            fermion_path_integral_up = fermion_path_integral_up,
            fermion_path_integral_dn = fermion_path_integral_dn,
            fermion_greens_calculator_up = fermion_greens_calculator_up,
            fermion_greens_calculator_dn = fermion_greens_calculator_dn,
            fermion_greens_calculator_up_alt = fermion_greens_calculator_up_alt,
            fermion_greens_calculator_dn_alt = fermion_greens_calculator_dn_alt,
            Bup = Bup, Bdn = Bdn, δG_max = δG_max, δG = δG, δθ = δθ, rng = rng
        )

        ## Record whether the HMC update was accepted or rejected.
        metadata["hmc_acceptance_rate"] += accepted

        ## Iterate over number of local updates to perform.
        for local_update in 1:N_local_updates

            ## Perform local update.
            (acceptance_rate, logdetGup, sgndetGup, logdetGdn, sgndetGdn, δG, δθ) = local_updates!(
                Gup, logdetGup, sgndetGup,
                Gdn, logdetGdn, sgndetGdn,
                hst_parameters,
                fermion_path_integral_up = fermion_path_integral_up,
                fermion_path_integral_dn = fermion_path_integral_dn,
                fermion_greens_calculator_up = fermion_greens_calculator_up,
                fermion_greens_calculator_dn = fermion_greens_calculator_dn,
                Bup = Bup, Bdn = Bdn, δG_max = δG_max, δG = δG, δθ = δθ, rng = rng,
                update_stabilization_frequency = true
            )

            ## Record acceptance rate for sweep.
            metadata["local_acceptance_rate"] += acceptance_rate
        end

        ## Write checkpoint file.
        checkpoint_timestamp = write_jld2_checkpoint(
            comm,
            simulation_info;
            checkpoint_timestamp = checkpoint_timestamp,
            checkpoint_freq = checkpoint_freq,
            start_timestamp = start_timestamp,
            runtime_limit = runtime_limit,
            exit_code = 13,
            ## Contents of checkpoint file below.
            n_therm = update + 1,
            n_measurements = 1,
            tight_binding_parameters, electron_phonon_parameters,
            hubbard_parameters, hst_parameters,
            measurement_container, model_geometry, metadata, rng
        )
    end

    ## Reset diagnostic parameters used to monitor numerical stability to zero.
    δG = zero(logdetGup)
    δθ = zero(logdetGup)

    ## Calculate the bin size.
    bin_size = N_measurements ÷ N_bins

    ## Iterate over measurements.
    for measurement in n_measurements:N_measurements

        ## Perform a reflection update.
        (accepted, logdetGup, sgndetGup, logdetGdn, sgndetGdn) = reflection_update!(
            Gup, logdetGup, sgndetGup,
            Gdn, logdetGdn, sgndetGdn,
            electron_phonon_parameters,
            fermion_path_integral_up = fermion_path_integral_up,
            fermion_path_integral_dn = fermion_path_integral_dn,
            fermion_greens_calculator_up = fermion_greens_calculator_up,
            fermion_greens_calculator_dn = fermion_greens_calculator_dn,
            fermion_greens_calculator_up_alt = fermion_greens_calculator_up_alt,
            fermion_greens_calculator_dn_alt = fermion_greens_calculator_dn_alt,
            Bup = Bup, Bdn = Bdn, rng = rng
        )

        ## Record whether the reflection update was accepted or rejected.
        metadata["reflection_acceptance_rate"] += accepted

        ## Perform an HMC update.
        (accepted, logdetGup, sgndetGup, logdetGdn, sgndetGdn, δG, δθ) = hmc_update!(
            Gup, logdetGup, sgndetGup,
            Gdn, logdetGdn, sgndetGdn,
            electron_phonon_parameters, hmc_updater,
            fermion_path_integral_up = fermion_path_integral_up,
            fermion_path_integral_dn = fermion_path_integral_dn,
            fermion_greens_calculator_up = fermion_greens_calculator_up,
            fermion_greens_calculator_dn = fermion_greens_calculator_dn,
            fermion_greens_calculator_up_alt = fermion_greens_calculator_up_alt,
            fermion_greens_calculator_dn_alt = fermion_greens_calculator_dn_alt,
            Bup = Bup, Bdn = Bdn, δG_max = δG_max, δG = δG, δθ = δθ, rng = rng
        )

        ## Record whether the HMC update was accepted or rejected.
        metadata["hmc_acceptance_rate"] += accepted

        ## Iterate over number of local updates to perform.
        for local_update in 1:N_local_updates

            ## Perform local update.
            (acceptance_rate, logdetGup, sgndetGup, logdetGdn, sgndetGdn, δG, δθ) = local_updates!(
                Gup, logdetGup, sgndetGup,
                Gdn, logdetGdn, sgndetGdn,
                hst_parameters,
                fermion_path_integral_up = fermion_path_integral_up,
                fermion_path_integral_dn = fermion_path_integral_dn,
                fermion_greens_calculator_up = fermion_greens_calculator_up,
                fermion_greens_calculator_dn = fermion_greens_calculator_dn,
                Bup = Bup, Bdn = Bdn, δG_max = δG_max, δG = δG, δθ = δθ, rng = rng,
                update_stabilization_frequency = true
            )

            ## Record acceptance rate for sweep.
            metadata["local_acceptance_rate"] += acceptance_rate
        end

        ## Make measurements.
        (logdetGup, sgndetGup, logdetGdn, sgndetGdn, δG, δθ) = make_measurements!(
            measurement_container,
            logdetGup, sgndetGup, Gup, Gup_ττ, Gup_τ0, Gup_0τ,
            logdetGdn, sgndetGdn, Gdn, Gdn_ττ, Gdn_τ0, Gdn_0τ,
            fermion_path_integral_up = fermion_path_integral_up,
            fermion_path_integral_dn = fermion_path_integral_dn,
            fermion_greens_calculator_up = fermion_greens_calculator_up,
            fermion_greens_calculator_dn = fermion_greens_calculator_dn,
            Bup = Bup, Bdn = Bdn, δG_max = δG_max, δG = δG, δθ = δθ,
            model_geometry = model_geometry,
            tight_binding_parameters = tight_binding_parameters,
            coupling_parameters = (hubbard_parameters, hst_parameters, electron_phonon_parameters)
        )

        ## Write the bin-averaged measurements to file if update ÷ bin_size == 0.
        write_measurements!(
            measurement_container = measurement_container,
            simulation_info = simulation_info,
            model_geometry = model_geometry,
            measurement = measurement,
            bin_size = bin_size,
            Δτ = Δτ
        )

        ## Write checkpoint file.
        checkpoint_timestamp = write_jld2_checkpoint(
            comm,
            simulation_info;
            checkpoint_timestamp = checkpoint_timestamp,
            checkpoint_freq = checkpoint_freq,
            start_timestamp = start_timestamp,
            runtime_limit = runtime_limit,
            exit_code = 13,
            ## Contents of checkpoint file below.
            n_therm = N_therm + 1,
            n_measurements = measurement + 1,
            tight_binding_parameters, electron_phonon_parameters,
            hubbard_parameters, hst_parameters,
            measurement_container, model_geometry, metadata, rng
        )
    end

    ## Merge binned data into a single HDF5 file.
    merge_bins(simulation_info)

    ## Calculate acceptance rates.
    metadata["reflection_acceptance_rate"] /= (N_measurements + N_therm)
    metadata["hmc_acceptance_rate"] /= (N_measurements + N_therm)
    metadata["local_acceptance_rate"] /= N_local_updates * (N_measurements + N_therm)

    ## Record largest numerical error encountered during simulation.
    metadata["dG"] = δG

    ## Write simulation metadata to simulation_info.toml file.
    save_simulation_info(simulation_info, metadata)

    ## Process the simulation results, calculating final error bars for all measurements.
    ## writing final statistics to CSV files.
    process_measurements(
        comm;
        datafolder = simulation_info.datafolder,
        n_bins = N_bins,
        export_to_csv = true,
        scientific_notation = true,
        decimals = 6,
        delimiter = " "
    )

    ## Calculate CDW correlation ratio.
    Rcdw, ΔRcdw = compute_correlation_ratio(
        datafolder = simulation_info.datafolder,
        correlation = "density",
        type = "equal-time",
        id_pairs = [(1, 1)],
        id_pair_coefficients = [1.0],
        q_point = (L÷2, L÷2),
        q_neighbors = [
            (L÷2+1, L÷2), (L÷2-1, L÷2),
            (L÷2, L÷2+1), (L÷2, L÷2-1)
        ]
    )

    ## Record the correlation ratio.
    metadata["Rcdw_mean"] = real(Rcdw)
    metadata["Rcdw_std"] = ΔRcdw

    ## Calculate AFM correlation ratio.
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

    ## Record the AFM correlation ratio mean and standard deviation.
    metadata["Rafm_mean"] = real(Rafm)
    metadata["Rafm_std"] = ΔRafm

    ## Write simulation summary TOML file.
    save_simulation_info(simulation_info, metadata)

    ## Rename the data folder to indicate the simulation is complete.
    simulation_info = rename_complete_simulation(
        comm, simulation_info,
        delete_jld2_checkpoints = true
    )

    return nothing
end # end of run_simulation function

## Only execute if the script is run directly from the command line.
if abspath(PROGRAM_FILE) == @__FILE__

    ## Initialize MPI
    MPI.Init()

    ## Initialize the MPI communicator.
    comm = MPI.COMM_WORLD

    ## Run the simulation.
    run_simulation(
        comm;
        sID = parse(Int, ARGS[1]), # Simulation ID.
        U = parse(Float64, ARGS[2]), # Hubbard interaction strength.
        Ω = parse(Float64, ARGS[3]), # Phonon energy.
        α = parse(Float64, ARGS[4]), # Electron-phonon coupling.
        μ = parse(Float64, ARGS[5]), # Chemical potential.
        L = parse(Int, ARGS[6]), # System size.
        β = parse(Float64, ARGS[7]), # Inverse temperature.
        N_therm = parse(Int, ARGS[8]), # Number of thermalization updates.
        N_measurements = parse(Int, ARGS[9]), # Total number of measurements.
        N_bins = parse(Int, ARGS[10]), # Number of times bin-averaged measurements are recorded.
        N_local_updates = parse(Int, ARGS[11]), # Number of local update sweeps per HMC update and measurement.
        checkpoint_freq = parse(Float64, ARGS[12]), # Frequency with which checkpoint files are written in hours.
    )

    ## Finalize MPI.
    MPI.Finalize()
end
