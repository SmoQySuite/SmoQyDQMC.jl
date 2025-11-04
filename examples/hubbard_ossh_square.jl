# # Square Optical Su-Schrieffer-Heeger Model
#
# In this example we simulate the optical Su-Schrieffer-Heeger (OSSH)-Hubbard model on a square lattice, with a Hamiltonian given by
# ```math
# \begin{align*}
# \hat{H} & = \sum_{\mathbf{i}}\left(\frac{1}{2M}\hat{P}_{\mathbf{i},x}^{2}+\frac{1}{2}M\Omega^{2}\hat{X}_{\mathbf{i}}^{2}\right) + \sum_{\mathbf{i}}\left(\frac{1}{2M}\hat{P}_{\mathbf{i},y}^{2}+\frac{1}{2}M\Omega^{2}\hat{Y}_{\mathbf{i}}^{2}\right) \\
#         & + U \sum_{\mathbf{i}}(\hat{n}_{\uparrow,\mathbf{i}}-\tfrac{1}{2}) (\hat{n}_{\downarrow,\mathbf{i}}-\tfrac{1}{2}) - \mu\sum_{\mathbf{i},\sigma}\hat{n}_{\sigma,\mathbf{i}}\\
#         & - \sum_{\mathbf{i},\sigma}\left[t-\alpha\left(\hat{X}_{\mathbf{i}+\mathbf{x}}-\hat{X}_{\mathbf{i}}\right)\right]\left(\hat{c}_{\sigma,\mathbf{i}+\mathbf{x}}^{\dagger}\hat{c}_{\sigma,\mathbf{i}}^{\phantom{\dagger}}+\hat{c}_{\sigma,\mathbf{i}}^{\dagger}\hat{c}_{\sigma,\mathbf{i}+\mathbf{x}}^{\phantom{\dagger}}\right) \\
#         & - \sum_{\mathbf{i},\sigma}\left[t-\alpha\left(\hat{Y}_{\mathbf{i}+\mathbf{y}}-\hat{Y}_{\mathbf{i}}\right)\right]\left(\hat{c}_{\sigma,\mathbf{i}+\mathbf{y}}^{\dagger}\hat{c}_{\sigma,\mathbf{i}}^{\phantom{\dagger}}+\hat{c}_{\sigma,\mathbf{i}}^{\dagger}\hat{c}_{\sigma,\mathbf{i}+\mathbf{y}}^{\phantom{\dagger}}\right),
# \end{align*}
# ```
# in which the fluctuations in the position of dispersionless phonon modes placed on each site in the lattice modulate the hopping amplitude between neighboring sites.
# In the above expression ``\hat{c}^\dagger_{\sigma,\mathbf{i}} \ (\hat{c}^{\phantom \dagger}_{\sigma,\mathbf{i}})`` creation (annihilation) operator
# a spin ``\sigma`` electron on site ``\mathbf{i}`` in the lattice, and ``\hat{n}_{\sigma,\mathbf{i}} = \hat{c}^\dagger_{\sigma,\mathbf{i}} \hat{c}^{\phantom \dagger}_{\sigma,\mathbf{i}}``
# is corresponding electron number operator. The phonon position (momentum) operator for the dispersionless phonon mode on site ``\mathbf{i}``
# is given by ``\hat{X}_{\mathbf{i}} \ (\hat{P}_{\mathbf{i}})``, where ``\Omega`` and ``M`` are the phonon frequency and associated ion mass respectively.
# Therefore, the strength of the electron-phonon coupling is controlled by the parameter ``\alpha``. Lastly, the parameter ``U`` controls the strength of the on-site Hubbard interaction.

# Note that this example scipt comes with all the bells and whistles so to speak, including support for MPI parallelizaiton as well as checkpointing.

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
    N_updates, # Total number of measurements and measurement updates.
    N_bins, # Number of times bin-averaged measurements are written to file.
    checkpoint_freq, # Frequency with which checkpoint files are written in hours.
    runtime_limit = Inf, # Simulation runtime limit in hours.
    Nt = 10, # Number of time-steps in HMC update.
    Δτ = 0.05, # Discretization in imaginary time.
    n_stab = 10, # Numerical stabilization period in imaginary-time slices.
    δG_max = 1e-6, # Threshold for numerical error corrected by stabilization.
    symmetric = false, # Whether symmetric propagator definition is used.
    checkerboard = false, # Whether checkerboard approximation is used.
    write_bins_concurrent = true, # Whether to write the HDF5 bins files during the simulation.
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
    datafolder_prefix = @sprintf "square_hubbard_ossh_U%2.f_w%.2f_a%.2f_mu%.2f_L%d_b%.2f" U Ω α μ L β

    ## Get MPI process ID.
    pID = MPI.Comm_rank(comm)

    ## Initialize simulation info.
    simulation_info = SimulationInfo(
        filepath = filepath,
        datafolder_prefix = datafolder_prefix,
        write_bins_concurrent = write_bins_concurrent,
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
        n_updates = 1

        ## Initialize random number generator
        rng = Xoshiro(seed)

        ## Initialize additiona_info dictionary
        metadata = Dict()

        ## Record simulation parameters.
        metadata["Nt"] = Nt
        metadata["N_therm"] = N_therm
        metadata["N_updates"] = N_updates
        metadata["N_bins"] = N_bins
        metadata["n_stab"] = n_stab
        metadata["dG_max"] = δG_max
        metadata["symmetric"] = symmetric
        metadata["checkerboard"] = checkerboard
        metadata["seed"] = seed
        metadata["hmc_acceptance_rate"] = 0.0
        metadata["local_acceptance_rate"] = 0.0
        metadata["swap_acceptance_rate"] = 0.0
        metadata["ph_ref_acceptance_rate"] = 0.0
        metadata["hst_ref_acceptance_rate"] = 0.0

        ## Initialize an instance of the type UnitCell.
        unit_cell = lu.UnitCell(
            lattice_vecs = [[1.0,0.0],
                            [0.0,1.0]],
            basis_vecs   = [[0.0,0.0]]
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

        ## Define a dispersionless phonon mode to represent vibrations in the x-direction.
        phonon_x = PhononMode(
            basis_vec = [0.0,0.0],
            Ω_mean = Ω
        )

        ## Add x-direction optical ssh phonon to electron-phonon model.
        phonon_x_id = add_phonon_mode!(
            electron_phonon_model = electron_phonon_model,
            phonon_mode = phonon_x
        )

        ## Define a dispersionless phonon mode to represent vibrations in the y-direction.
        phonon_y = PhononMode(
            basis_vec = [0.0,0.0],
            Ω_mean = Ω
        )

        ## Add y-direction optical ssh phonon to electron-phonon model.
        phonon_y_id = add_phonon_mode!(
            electron_phonon_model = electron_phonon_model,
            phonon_mode = phonon_y
        )

        ## Defines ssh e-ph coupling such that total effective hopping.
        ossh_x_coupling = SSHCoupling(
            model_geometry = model_geometry,
            tight_binding_model = tight_binding_model,
            phonon_ids = (phonon_x_id, phonon_x_id),
            bond = bond_px,
            α_mean = α
        )

        ## Add x-direction optical SSH coupling to the electron-phonon model.
        ossh_x_coupling_id = add_ssh_coupling!(
            electron_phonon_model = electron_phonon_model,
            ssh_coupling = ossh_x_coupling,
            tight_binding_model = tight_binding_model
        )

        ## Defines ssh e-ph coupling such that total effective hopping.
        ossh_y_coupling = SSHCoupling(
            model_geometry = model_geometry,
            tight_binding_model = tight_binding_model,
            phonon_ids = (phonon_y_id, phonon_y_id),
            bond = bond_py,
            α_mean = α
        )

        ## Add y-direction optical SSH coupling to the electron-phonon model.
        ossh_y_coupling_id = add_ssh_coupling!(
            electron_phonon_model = electron_phonon_model,
            ssh_coupling = ossh_y_coupling,
            tight_binding_model = tight_binding_model
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

        ## Apply Ising Hubbard-Stratonovich (HS) transformation to decouple the Hubbard interaction,
        ## and initialize the corresponding HS fields that will be sampled in the DQMC simulation.
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
                (phonon_x_id, phonon_x_id),
                (phonon_y_id, phonon_y_id)
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
                ## Measure local s-wave pair susceptibility associated with
                ## each orbital in the unit cell.
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

        ## Initialize the bond correlation measurement
        initialize_correlation_measurements!(
            measurement_container = measurement_container,
            model_geometry = model_geometry,
            correlation = "bond",
            time_displaced = false,
            integrated = true,
            pairs = [
                (bond_px_id, bond_px_id),
                (bond_py_id, bond_py_id),
                (bond_px_id, bond_py_id),
            ]
        )

        ## Measure composite bond correlation for detecting a bond ordered wave (BOW)
        ## that breaks a C4 rotation symmetry.
        initialize_composite_correlation_measurement!(
            measurement_container = measurement_container,
            model_geometry = model_geometry,
            name = "BOW_C4",
            correlation = "bond",
            ids = [bond_px_id, bond_py_id, bond_nx_id, bond_ny_id],
            coefficients = [+1.0, +1.0im, -1.0, -1.0im],
            displacement_vecs = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            time_displaced = false,
            integrated = true
        )

        ## Measure composite bond correlation for detecting a bond ordered wave (BOW)
        ## that breaks a C2 rotation symmetry.
        initialize_composite_correlation_measurement!(
            measurement_container = measurement_container,
            model_geometry = model_geometry,
            name = "BOW_C2",
            correlation = "bond",
            ids = [bond_px_id, bond_py_id, bond_nx_id, bond_ny_id],
            coefficients = [+1.0, -1.0, +1.0, -1.0],
            displacement_vecs = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
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
            ## Contents of checkpoint file below.
            n_therm, n_updates,
            tight_binding_parameters, electron_phonon_parameters,
            hubbard_parameters, hst_parameters,
            measurement_container, model_geometry, metadata, rng
        )

    ## If resuming a previous simulation.
    else

        ## Load the checkpoint file.
        checkpoint, checkpoint_timestamp = read_jld2_checkpoint(simulation_info)

        ## Unpack contents of checkpoint dictionary.
        tight_binding_parameters    = checkpoint["tight_binding_parameters"]
        hubbard_parameters          = checkpoint["hubbard_parameters"]
        hst_parameters              = checkpoint["hst_parameters"]
        electron_phonon_parameters  = checkpoint["electron_phonon_parameters"]
        measurement_container       = checkpoint["measurement_container"]
        model_geometry              = checkpoint["model_geometry"]
        metadata                    = checkpoint["metadata"]
        rng                         = checkpoint["rng"]
        n_therm                     = checkpoint["n_therm"]
        n_updates                   = checkpoint["n_updates"]
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

    ## Initialize diagonostic parameters to asses numerical stability.
    δG = zero(logdetGup)
    δθ = zero(logdetGup)

    ## Initialize Hamitlonian/Hybrid monte carlo (HMC) updater.
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
        metadata["ph_ref_acceptance_rate"] += accepted

        ## Perform a swap update.
        (accepted, logdetGup, sgndetGup, logdetGdn, sgndetGdn) = swap_update!(
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
        metadata["swap_acceptance_rate"] += accepted

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

        ## Perform reflection update for HS fields with randomly chosen site.
        (accepted, logdetGup, sgndetGup, logdetGdn, sgndetGdn) = reflection_update!(
            Gup, logdetGup, sgndetGup,
            Gdn, logdetGdn, sgndetGdn,
            hst_parameters,
            fermion_path_integral_up = fermion_path_integral_up,
            fermion_path_integral_dn = fermion_path_integral_dn,
            fermion_greens_calculator_up = fermion_greens_calculator_up,
            fermion_greens_calculator_dn = fermion_greens_calculator_dn,
            fermion_greens_calculator_up_alt = fermion_greens_calculator_up_alt,
            fermion_greens_calculator_dn_alt = fermion_greens_calculator_dn_alt,
            Bup = Bup, Bdn = Bdn, rng = rng
        )

        ## Record whether reflection update was accepted or not.
        metadata["hst_ref_acceptance_rate"] += accepted

        ## Perform sweep all imaginary-time slice and orbitals, attempting an update to every HS field.
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

        ## Write checkpoint file.
        checkpoint_timestamp = write_jld2_checkpoint(
            comm,
            simulation_info;
            checkpoint_timestamp = checkpoint_timestamp,
            checkpoint_freq = checkpoint_freq,
            start_timestamp = start_timestamp,
            runtime_limit = runtime_limit,
            ## Contents of checkpoint file below.
            n_therm  = update + 1,
            n_updates = 1,
            tight_binding_parameters, electron_phonon_parameters,
            hubbard_parameters, hst_parameters,
            measurement_container, model_geometry, metadata, rng
        )
    end

    ## Reset diagonostic parameters used to monitor numerical stability to zero.
    δG = zero(logdetGup)
    δθ = zero(logdetGup)

    ## Calculate the bin size.
    bin_size = N_updates ÷ N_bins

    ## Iterate over updates and measurements.
    for update in n_updates:N_updates

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
        metadata["ph_ref_acceptance_rate"] += accepted

        ## Perform a swap update.
        (accepted, logdetGup, sgndetGup, logdetGdn, sgndetGdn) = swap_update!(
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
        metadata["swap_acceptance_rate"] += accepted

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

        ## Perform reflection update for HS fields with randomly chosen site.
        (accepted, logdetGup, sgndetGup, logdetGdn, sgndetGdn) = reflection_update!(
            Gup, logdetGup, sgndetGup,
            Gdn, logdetGdn, sgndetGdn,
            hst_parameters,
            fermion_path_integral_up = fermion_path_integral_up,
            fermion_path_integral_dn = fermion_path_integral_dn,
            fermion_greens_calculator_up = fermion_greens_calculator_up,
            fermion_greens_calculator_dn = fermion_greens_calculator_dn,
            fermion_greens_calculator_up_alt = fermion_greens_calculator_up_alt,
            fermion_greens_calculator_dn_alt = fermion_greens_calculator_dn_alt,
            Bup = Bup, Bdn = Bdn, rng = rng
        )

        ## Record whether reflection update was accepted or not.
        metadata["hst_ref_acceptance_rate"] += accepted

        ## Perform sweep all imaginary-time slice and orbitals, attempting an update to every HS field.
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
            update = update,
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
            ## Contents of checkpoint file below.
            n_therm  = N_therm + 1,
            n_updates = update + 1,
            tight_binding_parameters, electron_phonon_parameters,
            hubbard_parameters, hst_parameters,
            measurement_container, model_geometry, metadata, rng
        )
    end

    ## Merge binned data into a single HDF5 file.
    merge_bins(simulation_info)

    ## Calculate acceptance rates.
    metadata["hmc_acceptance_rate"] /= (N_updates + N_therm)
    metadata["ph_ref_acceptance_rate"] /= (N_updates + N_therm)
    metadata["swap_acceptance_rate"] /= (N_updates + N_therm)
    metadata["local_acceptance_rate"] /= (N_updates + N_therm)
    metadata["hst_ref_acceptance_rate"] /= (N_updates + N_therm)

    ## Record largest numerical error encountered during simulation.
    metadata["dG"] = δG

    ## Write simulation metadata to simulation_info.toml file.
    save_simulation_info(simulation_info, metadata)

    ## Process the simulation results, calculating final error bars for all measurements.
    ## writing final statisitics to CSV files.
    process_measurements(
        comm;
        datafolder = simulation_info.datafolder,
        n_bins = N_bins,
        export_to_csv = true,
        scientific_notation = false,
        decimals = 9,
        delimiter = " "
    )

    ## Calculate C4 BOW q=(π,π) correlation ratio.
    Rbow, ΔRbow = compute_composite_correlation_ratio(
        comm;
        datafolder = simulation_info.datafolder,
        name = "BOW_C4",
        type = "equal-time",
        q_point = (L÷2, L÷2),
        q_neighbors = [
            (L÷2+1, L÷2), (L÷2, L÷2+1),
            (L÷2-1, L÷2), (L÷2, L÷2-1)
        ]
    )

    # Record the correlation ratio.
    metadata["Rbow_mean_real"] = real(Rbow)
    metadata["Rbow_mean_imag"] = imag(Rbow)
    metadata["Rbow_std"] = ΔRbow

    ## Write simulation summary TOML file.
    save_simulation_info(simulation_info, metadata)

    ## Rename the data folder to indicate the simulation is complete.
    simulation_info = rename_complete_simulation(
        comm, simulation_info,
        delete_jld2_checkpoints = true
    )

    return nothing
end # end of run_simulation function

## Only excute if the script is run directly from the command line.
if abspath(PROGRAM_FILE) == @__FILE__

    ## Initialize MPI
    MPI.Init()

    ## Initialize the MPI communicator.
    comm = MPI.COMM_WORLD

    ## Run the simulation.
    run_simulation(
        comm;
        sID             = parse(Int,     ARGS[1]),  # Simulation ID.
        U               = parse(Float64, ARGS[2]),  # Hubbard interaction strength.
        Ω               = parse(Float64, ARGS[3]),  # Phonon energy.
        α               = parse(Float64, ARGS[4]),  # Electron-phonon coupling.
        μ               = parse(Float64, ARGS[5]),  # Chemical potential.
        L               = parse(Int,     ARGS[6]),  # System size.
        β               = parse(Float64, ARGS[7]),  # Inverse temperature.
        N_therm         = parse(Int,     ARGS[8]),  # Number of thermalization updates.
        N_updates       = parse(Int,     ARGS[9]),  # Total number of measurements and measurement updates.
        N_bins          = parse(Int,     ARGS[10]),  # Number of times bin-averaged measurements are written to file.
        checkpoint_freq = parse(Float64, ARGS[11]), # Frequency with which checkpoint files are written in hours.
    )

    ## Finalize MPI.
    MPI.Finalize()
end
