```@meta
EditURL = "../../../examples/hubbard_holstein_square.jl"
```

# Square Hubbard-Holstein Model
Download this example as a [Julia script](../assets/scripts/examples/hubbard_holstein_square.jl).

In this example we write a script to simulate the Hubbard-Holstein model on a square lattice, with a Hamiltonian given by
```math
\begin{align*}
\hat{H} = & -t \sum_{\sigma,\langle i, j \rangle} (\hat{c}^{\dagger}_{\sigma,i}, \hat{c}^{\phantom \dagger}_{\sigma,j} + {\rm h.c.})
            -\mu \sum_{\sigma,i}\hat{n}_{\sigma,i}\\
          & + U \sum_{i} (\hat{n}_{\uparrow,i}-\tfrac{1}{2})(\hat{n}_{\downarrow,i}-\tfrac{1}{2})
            + \alpha \sum_{\sigma,i} \hat{X}_i (\hat{n}_{\sigma,i} - \tfrac{1}{2}) \\
          & + \sum_i \left( \frac{1}{2M}\hat{P}_i^2 + \frac{1}{2}M\Omega^2\hat{X}_i^2 \right),
\end{align*}
```
where ``\hat{c}^\dagger_{\sigma,i} \ (\hat{c}^{\phantom \dagger}_{\sigma,i})`` creates (annihilates) a spin ``\sigma``
electron on site ``i`` in the lattice, and ``\hat{n}_{\sigma,i} = \hat{c}^\dagger_{\sigma,i} \hat{c}^{\phantom \dagger}_{\sigma,i}``
is the spin-``\sigma`` electron number operator for site ``i``. The nearest-neighbor hopping amplitude is ``t`` and ``\mu`` is the
chemical potential. The strength of the repulsive Hubbard interaction is controlled by ``U>0``. ``\hat{X}_i \ (\hat{P}_i)``
is the phonon position (momentum) operator for a dispersionless mode placed on site ``i`` with phonon frequency ``\Omega`` and
corresponding ion mass ``M``. The stength of the Holstein electron-phonon is controlled by the parameter ``\alpha``.

A short test simulation using the script associated with this example can be run as
```
> julia hubbard_holstein_square.jl 0 6.0 0.1 0.1 0.0 4.0 4 1000 5000 50
```
which simulates the Hubbard-Holstein model on a ``L = 4`` square lattice, with ``U = 6.0``, ``\Omega = 0.1``, ``\alpha = 0.1``
and ``\mu = 0.0`` at an inverse temperature of ``\beta = 4.0``. In this simulation the Hubbard-Stranonovich and phonon fields
are thermalized with `N_burnin = 1000` rounds of updates, followed by `N_udpates = 5000` rounds of updates with measurements
being made. Bin averaged measurements are written to file `N_bins = 50` during the simulation.

Below you will find the source code from the julia script linked at the top of this page,
but with additional comments giving more detailed explanations for what certain parts of the code are doing.
Additionally, this script demonstrates how to calculate the extended s-wave and d-wave pair susceptibilities.

````julia
using LinearAlgebra
using Random
using Printf

using SmoQyDQMC
import SmoQyDQMC.LatticeUtilities  as lu
import SmoQyDQMC.JDQMCFramework    as dqmcf
import SmoQyDQMC.JDQMCMeasurements as dqmcm

# Define top-level function for running DQMC simulation
function run_hubbard_holstein_square_simulation(sID, U, Ω, α, μ, β, L, N_burnin, N_updates, N_bins; filepath = ".")

    # Construct the foldername the data will be written to.
    datafolder_prefix = @sprintf "hubbard_holstein_square_U%.2f_w%.2f_a%.2f_mu%.2f_L%d_b%.2f" U Ω α μ L β

    # Initialize an instance of the SimulationInfo type.
    simulation_info = SimulationInfo(
        filepath = filepath,
        datafolder_prefix = datafolder_prefix,
        sID = sID
    )

    # Initialize the directory the data will be written to.
    initialize_datafolder(simulation_info)

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

    # Number of fermionic time-steps in HMC update.
    Nt = 4

    # Fermionic time-step used in HMC update.
    Δt = π/(2*Ω)/Nt

    # Initialize a dictionary to store additional information about the simulation.
    additional_info = Dict(
        "dG_max" => δG_max,
        "N_burnin" => N_burnin,
        "N_updates" => N_updates,
        "N_bins" => N_bins,
        "bin_size" => bin_size,
        "local_acceptance_rate" => 0.0,
        "hmc_acceptance_rate" => 0.0,
        "reflection_acceptance_rate" => 0.0,
        "n_stab_init" => n_stab,
        "symmetric" => symmetric,
        "checkerboard" => checkerboard,
        "seed" => seed,
        "Nt" => Nt,
        "dt" => Δt,
    )

    #######################
    ### DEFINE THE MODEL ##
    #######################

    # Initialize an instance of the type UnitCell.
    unit_cell = lu.UnitCell(lattice_vecs = [[1.0, 0.0],
                                            [0.0, 1.0]],
                            basis_vecs   = [[0.0, 0.0]])

    # Initialize an instance of the type Lattice.
    lattice = lu.Lattice(
        L = [L, L],
        periodic = [true, true]
    )

    # Initialize an instance of the ModelGeometry type.
    model_geometry = ModelGeometry(unit_cell, lattice)

    # Get the number of orbitals in the lattice.
    N = lu.nsites(unit_cell, lattice)

    # Define the nearest-neighbor bond in the +x direction.
    bond_px = lu.Bond(orbitals = (1,1), displacement = [1,0])

    # Add nearest-neighbor bond in the +x direction.
    bond_px_id = add_bond!(model_geometry, bond_px)

    # Define the nearest-neighbor bond in the +y direction.
    bond_py = lu.Bond(orbitals = (1,1), displacement = [0,1])

    # Add the nearest-neighbor bond in the +y direction.
    bond_py_id = add_bond!(model_geometry, bond_py)

    # Here we define bonds to points in the negative x and y directions respectively.
    # We do this in order to be able to measure all the pairing channels we need
    # in order to reconstruct the extended s-wave and d-wave pair susceptibilities.

    # Define the nearest-neighbor bond in the -x direction.
    bond_nx = lu.Bond(orbitals = (1,1), displacement = [-1,0])

    # Add nearest-neighbor bond in the -x direction.
    bond_nx_id = add_bond!(model_geometry, bond_nx)

    # Define the nearest-neighbor bond in the -y direction.
    bond_ny = lu.Bond(orbitals = (1,1), displacement = [0,-1])

    # Add the nearest-neighbor bond in the -y direction.
    bond_ny_id = add_bond!(model_geometry, bond_ny)

    # Define nearest-neighbor hopping amplitude, setting the energy scale for the system.
    t = 1.0

    # Define the tight-binding model
    tight_binding_model = TightBindingModel(
        model_geometry = model_geometry,
        t_bonds = [bond_px, bond_py], # defines hopping
        t_mean = [t, t],            # defines corresponding hopping amplitude
        μ = μ,                      # set chemical potential
        ϵ_mean = [0.]               # set the (mean) on-site energy
    )

    # Initialize the Hubbard interaction in the model.
    hubbard_model = HubbardModel(
        shifted = false,
        U_orbital = [1],
        U_mean = [U],
    )

    # Initialize a null electron-phonon model.
    electron_phonon_model = ElectronPhononModel(
        model_geometry = model_geometry,
        tight_binding_model = tight_binding_model
    )

    # Define a dispersionless electron-phonon mode to live on each site in the lattice.
    phonon = PhononMode(orbital = 1, Ω_mean = Ω)

    # Add the phonon mode definition to the electron-phonon model.
    phonon_id = add_phonon_mode!(
        electron_phonon_model = electron_phonon_model,
        phonon_mode = phonon
    )

    # Define a on-site Holstein coupling between the electron and the local dispersionless phonon mode.
    holstein_coupling = HolsteinCoupling(
    	model_geometry = model_geometry,
    	phonon_mode = phonon_id,
    	bond = lu.Bond(orbitals = (1,1), displacement = [0,0]),
    	α_mean = α
    )

    # Add the Holstein coupling definition to the model.
    holstein_coupling_id = add_holstein_coupling!(
    	electron_phonon_model = electron_phonon_model,
    	holstein_coupling = holstein_coupling,
    	model_geometry = model_geometry
    )

    # Write the model summary to file.
    model_summary(
        simulation_info = simulation_info,
        β = β, Δτ = Δτ,
        model_geometry = model_geometry,
        tight_binding_model = tight_binding_model,
        interactions = (hubbard_model, electron_phonon_model)
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

    # Initialize electron-phonon parameters.
    electron_phonon_parameters = ElectronPhononParameters(
        β = β, Δτ = Δτ,
        electron_phonon_model = electron_phonon_model,
        tight_binding_parameters = tight_binding_parameters,
        model_geometry = model_geometry,
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

    # Initialize the electron-phonon interaction related measurements.
    initialize_measurements!(measurement_container, electron_phonon_model)

    # Initialize the single-particle electron Green's function measurement.
    initialize_correlation_measurements!(
        measurement_container = measurement_container,
        model_geometry = model_geometry,
        correlation = "greens",
        time_displaced = true,
        pairs = [(1, 1)]
    )

    # measure equal-times green's function for all τ
    initialize_correlation_measurements!(
        measurement_container = measurement_container,
        model_geometry = model_geometry,
        correlation = "greens_tautau",
        time_displaced = false,
        integrated = true,
        pairs = [(1, 1)]
    )

    # Initialize the phonon Green's function measurement.
    initialize_correlation_measurements!(
        measurement_container = measurement_container,
        model_geometry = model_geometry,
        correlation = "phonon_greens",
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

    # Initialize the spin-z correlation function measurement.
    initialize_correlation_measurements!(
        measurement_container = measurement_container,
        model_geometry = model_geometry,
        correlation = "spin_z",
        time_displaced = false,
        integrated = true,
        pairs = [(1, 1)]
    )

    # Measure all possible combinations of bond pairing channels
    # for the bonds we have defined. We will need each of these
    # pairs channels measured in order to reconstruct the extended
    # s-wave and d-wave pair susceptibilities.
    # Initialize the pair correlation function measurement.
    initialize_correlation_measurements!(
        measurement_container = measurement_container,
        model_geometry = model_geometry,
        correlation = "pair",
        time_displaced = false,
        integrated = true,
        pairs = [(1, 1),
                 (bond_px_id, bond_px_id), (bond_px_id, bond_nx_id),
                 (bond_nx_id, bond_px_id), (bond_nx_id, bond_nx_id),
                 (bond_py_id, bond_py_id), (bond_py_id, bond_ny_id),
                 (bond_ny_id, bond_py_id), (bond_ny_id, bond_ny_id),
                 (bond_px_id, bond_py_id), (bond_px_id, bond_ny_id),
                 (bond_nx_id, bond_py_id), (bond_nx_id, bond_ny_id),
                 (bond_py_id, bond_px_id), (bond_py_id, bond_nx_id),
                 (bond_ny_id, bond_px_id), (bond_ny_id, bond_nx_id)]
    )

    # Initialize the current correlation function measurement
    initialize_correlation_measurements!(
        measurement_container = measurement_container,
        model_geometry = model_geometry,
        correlation = "current",
        time_displaced = false,
        integrated = true,
        pairs = [(1, 1), # hopping ID pair for x-direction hopping
                 (2, 2)] # hopping ID pair for y-direction hopping
    )

    # Initialize the sub-directories to which the various measurements will be written.
    initialize_measurement_directories(
        simulation_info = simulation_info,
        measurement_container = measurement_container
    )

    #############################
    ### SET-UP DQMC SIMULATION ##
    #############################

    # Allocate FermionPathIntegral type for both the spin-up and spin-down electrons.
    fermion_path_integral_up = FermionPathIntegral(tight_binding_parameters = tight_binding_parameters, β = β, Δτ = Δτ)
    fermion_path_integral_dn = FermionPathIntegral(tight_binding_parameters = tight_binding_parameters, β = β, Δτ = Δτ)

    # Initialize the FermionPathIntegral type for both the spin-up and spin-down electrons.
    initialize!(fermion_path_integral_up, fermion_path_integral_dn, hubbard_parameters)
    initialize!(fermion_path_integral_up, fermion_path_integral_dn, hubbard_ising_parameters)

    # Initialize the fermion path integral type with respect to electron-phonon interaction.
    initialize!(fermion_path_integral_up, fermion_path_integral_dn, electron_phonon_parameters)

    # Initialize the imaginary-time propagators for each imaginary-time slice for both the
    # spin-up and spin-down electrons.
    Bup = initialize_propagators(fermion_path_integral_up, symmetric=symmetric, checkerboard=checkerboard)
    Bdn = initialize_propagators(fermion_path_integral_dn, symmetric=symmetric, checkerboard=checkerboard)

    # Initialize FermionGreensCalculator for the spin-up and spin-down electrons.
    fermion_greens_calculator_up = dqmcf.FermionGreensCalculator(Bup, β, Δτ, n_stab)
    fermion_greens_calculator_dn = dqmcf.FermionGreensCalculator(Bdn, β, Δτ, n_stab)

    # Initialize alternate fermion greens calculator required for performing various global updates.
    fermion_greens_calculator_up_alt = dqmcf.FermionGreensCalculator(fermion_greens_calculator_up)
    fermion_greens_calculator_dn_alt = dqmcf.FermionGreensCalculator(fermion_greens_calculator_dn)

    # Allcoate matrices for spin-up and spin-down electron Green's function matrices.
    Gup = zeros(eltype(Bup[1]), size(Bup[1]))
    Gdn = zeros(eltype(Bdn[1]), size(Bdn[1]))

    # Initialize the spin-up and spin-down electron Green's function matrices, also
    # calculating their respective determinants as the same time.
    logdetGup, sgndetGup = dqmcf.calculate_equaltime_greens!(Gup, fermion_greens_calculator_up)
    logdetGdn, sgndetGdn = dqmcf.calculate_equaltime_greens!(Gdn, fermion_greens_calculator_dn)

    # Initialize Hamitlonian/Hybrid monte carlo (HMC) updater.
    hmc_updater = EFAHMCUpdater(
        electron_phonon_parameters = electron_phonon_parameters,
        G = Gup, Nt = Nt, Δt = Δt
    )

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

        # Perform a reflection update.
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
            Bup = Bup, Bdn = Bdn, rng = rng, phonon_types = (phonon_id,)
        )

        # Record whether the reflection update was accepted or rejected.
        additional_info["reflection_acceptance_rate"] += accepted

        # Perform an HMC update.
        (accepted, logdetGup, sgndetGup, logdetGdn, sgndetGdn, δG, δθ) = hmc_update!(
            Gup, logdetGup, sgndetGup,
            Gdn, logdetGdn, sgndetGdn,
            electron_phonon_parameters,
            hmc_updater,
            fermion_path_integral_up = fermion_path_integral_up,
            fermion_path_integral_dn = fermion_path_integral_dn,
            fermion_greens_calculator_up = fermion_greens_calculator_up,
            fermion_greens_calculator_dn = fermion_greens_calculator_dn,
            fermion_greens_calculator_up_alt = fermion_greens_calculator_up_alt,
            fermion_greens_calculator_dn_alt = fermion_greens_calculator_dn_alt,
            Bup = Bup, Bdn = Bdn,
            δG_max = δG_max, δG = δG, δθ = δθ, rng = rng
        )

        # Record whether the HMC update was accepted or rejected.
        additional_info["hmc_acceptance_rate"] += accepted

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

            # Perform a reflection update.
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
                Bup = Bup, Bdn = Bdn, rng = rng, phonon_types = (phonon_id,)
            )

            # Record whether the reflection update was accepted or rejected.
            additional_info["reflection_acceptance_rate"] += accepted

            # Perform an HMC update.
            (accepted, logdetGup, sgndetGup, logdetGdn, sgndetGdn, δG, δθ) = hmc_update!(
                Gup, logdetGup, sgndetGup,
                Gdn, logdetGdn, sgndetGdn,
                electron_phonon_parameters,
                hmc_updater,
                fermion_path_integral_up = fermion_path_integral_up,
                fermion_path_integral_dn = fermion_path_integral_dn,
                fermion_greens_calculator_up = fermion_greens_calculator_up,
                fermion_greens_calculator_dn = fermion_greens_calculator_dn,
                fermion_greens_calculator_up_alt = fermion_greens_calculator_up_alt,
                fermion_greens_calculator_dn_alt = fermion_greens_calculator_dn_alt,
                Bup = Bup, Bdn = Bdn,
                δG_max = δG_max, δG = δG, δθ = δθ, rng = rng
            )

            # Record whether the HMC update was accepted or rejected.
            additional_info["hmc_acceptance_rate"] += accepted

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
                coupling_parameters = (
                    hubbard_parameters,
                    hubbard_ising_parameters,
                    electron_phonon_parameters
                )
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

    # Calculate acceptance rates.
    additional_info["hmc_acceptance_rate"] /= (N_updates + N_burnin)
    additional_info["reflection_acceptance_rate"] /= (N_updates + N_burnin)
    additional_info["local_acceptance_rate"] /= (N_updates + N_burnin)

    # Record the final numerical stabilization period that the simulation settled on.
    additional_info["n_stab_final"] = fermion_greens_calculator_up.n_stab

    # Record the maximum numerical error corrected by numerical stablization.
    additional_info["dG"] = δG

    #################################
    ### PROCESS SIMULATION RESULTS ##
    #################################

    # Process the simulation results, calculating final error bars for all measurements,
    # writing final statisitics to CSV files.
    process_measurements(simulation_info.datafolder, N_bins)

    # Here we use the `composite_correlation_stats` to reconstruct the extended
    # s-wave and d-wave pair susceptibilities. We also subract off the background
    # signal associated with the zero-momentum transfer charge susceptibility.
    # Behind the scenes, uses the binning
    # method to calculate the error bars by calculating both susceptibilities for
    # each bin of data that was written to file.

    # Measure the extended s-wave pair susceptibility.
    Pes, ΔPes = composite_correlation_stat(
        folder = simulation_info.datafolder,
        correlations = ["pair", "pair", "pair", "pair",
                        "pair", "pair", "pair", "pair",
                        "pair", "pair", "pair", "pair",
                        "pair", "pair", "pair", "pair"],
        spaces = ["momentum", "momentum", "momentum", "momentum",
                  "momentum", "momentum", "momentum", "momentum",
                  "momentum", "momentum", "momentum", "momentum",
                  "momentum", "momentum", "momentum", "momentum"],
        types = ["integrated", "integrated", "integrated", "integrated",
                 "integrated", "integrated", "integrated", "integrated",
                 "integrated", "integrated", "integrated", "integrated",
                 "integrated", "integrated", "integrated", "integrated"],
        ids = [(bond_px_id, bond_px_id), (bond_nx_id, bond_nx_id), (bond_px_id, bond_nx_id), (bond_nx_id, bond_px_id),
               (bond_py_id, bond_py_id), (bond_ny_id, bond_ny_id), (bond_py_id, bond_ny_id), (bond_ny_id, bond_py_id),
               (bond_px_id, bond_py_id), (bond_nx_id, bond_ny_id), (bond_px_id, bond_ny_id), (bond_nx_id, bond_py_id),
               (bond_py_id, bond_px_id), (bond_ny_id, bond_nx_id), (bond_py_id, bond_nx_id), (bond_ny_id, bond_px_id)],
        locs = [(0,0), (0,0), (0,0), (0,0),
                (0,0), (0,0), (0,0), (0,0),
                (0,0), (0,0), (0,0), (0,0),
                (0,0), (0,0), (0,0), (0,0)],
        num_bins = N_bins,
        f = (P_px_px, P_nx_nx, P_px_nx, P_nx_px,
             P_py_py, P_ny_ny, P_py_ny, P_ny_py,
             P_px_py, P_nx_ny, P_px_ny, P_nx_py,
             P_py_px, P_ny_nx, P_py_nx, P_ny_px) -> (P_px_px + P_nx_nx + P_px_nx + P_nx_px +
                                                     P_py_py + P_ny_ny + P_py_ny + P_ny_py +
                                                     P_px_py + P_nx_ny + P_px_ny + P_nx_py +
                                                     P_py_px + P_ny_nx + P_py_nx + P_ny_px)/4
    )
    additional_info["P_ext-s_avg"] = Pes
    additional_info["P_ext-s_err"] = ΔPes

    # Measure the d-wave pair susceptibility.
    Pd, ΔPd = composite_correlation_stat(
        folder = simulation_info.datafolder,
        correlations = ["pair", "pair", "pair", "pair",
                        "pair", "pair", "pair", "pair",
                        "pair", "pair", "pair", "pair",
                        "pair", "pair", "pair", "pair"],
        spaces = ["momentum", "momentum", "momentum", "momentum",
                  "momentum", "momentum", "momentum", "momentum",
                  "momentum", "momentum", "momentum", "momentum",
                  "momentum", "momentum", "momentum", "momentum"],
        types = ["integrated", "integrated", "integrated", "integrated",
                 "integrated", "integrated", "integrated", "integrated",
                 "integrated", "integrated", "integrated", "integrated",
                 "integrated", "integrated", "integrated", "integrated"],
        ids = [(bond_px_id, bond_px_id), (bond_nx_id, bond_nx_id), (bond_px_id, bond_nx_id), (bond_nx_id, bond_px_id),
               (bond_py_id, bond_py_id), (bond_ny_id, bond_ny_id), (bond_py_id, bond_ny_id), (bond_ny_id, bond_py_id),
               (bond_px_id, bond_py_id), (bond_nx_id, bond_ny_id), (bond_px_id, bond_ny_id), (bond_nx_id, bond_py_id),
               (bond_py_id, bond_px_id), (bond_ny_id, bond_nx_id), (bond_py_id, bond_nx_id), (bond_ny_id, bond_px_id)],
        locs = [(0,0), (0,0), (0,0), (0,0),
                (0,0), (0,0), (0,0), (0,0),
                (0,0), (0,0), (0,0), (0,0),
                (0,0), (0,0), (0,0), (0,0)],
        num_bins = N_bins,
        f = (P_px_px, P_nx_nx, P_px_nx, P_nx_px,
             P_py_py, P_ny_ny, P_py_ny, P_ny_py,
             P_px_py, P_nx_ny, P_px_ny, P_nx_py,
             P_py_px, P_ny_nx, P_py_nx, P_ny_px) -> (P_px_px + P_nx_nx + P_px_nx + P_nx_px +
                                                     P_py_py + P_ny_ny + P_py_ny + P_ny_py -
                                                     P_px_py - P_nx_ny - P_px_ny - P_nx_py -
                                                     P_py_px - P_ny_nx - P_py_nx - P_ny_px)/4
    )
    additional_info["P_d_avg"] = Pd
    additional_info["P_d_err"] = ΔPd

    # Calculate the charge susceptibility for zero momentum transfer (q=0)
    # with the net charge background signal subtracted off.

    # Calculate the Cu-Cu charge susceptibility at q=0 with the background signal removed.
    C0, ΔC0 = composite_correlation_stat(
        folder = simulation_info.datafolder,
        correlations = ["density", "greens_tautau", "greens"],
        spaces = ["momentum", "position", "position"],
        types = ["integrated", "integrated", "time-displaced"],
        ids = [(1,1), (1,1), (1,1)],
        locs = [(0,0), (0,0), (0,0)],
        Δls = [0, 0, 0],
        num_bins = N_bins,
        f = (x, y, z) -> x - (L^2)*4*(β-y)*(1-z)
    )
    additional_info["Chi_C_q0_avg"] = C0
    additional_info["Chi_C_q0_err"] = ΔC0

    # Write simulation summary TOML file.
    save_simulation_info(simulation_info, additional_info)

    return nothing
end


# Only excute if the script is run directly from the command line.
if abspath(PROGRAM_FILE) == @__FILE__

    # Read in the command line arguments.
    sID = parse(Int, ARGS[1]) # simulation ID
    U = parse(Float64, ARGS[2])
    Ω = parse(Float64, ARGS[3])
    α = parse(Float64, ARGS[4])
    μ = parse(Float64, ARGS[5])
    β = parse(Float64, ARGS[6])
    L = parse(Int, ARGS[7])
    N_burnin = parse(Int, ARGS[8])
    N_updates = parse(Int, ARGS[9])
    N_bins = parse(Int, ARGS[10])

    # Run the simulation.
    run_hubbard_holstein_square_simulation(sID, U, Ω, α, μ, β, L, N_burnin, N_updates, N_bins)
end
````

