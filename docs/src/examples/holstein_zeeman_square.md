```@meta
EditURL = "../../../examples/holstein_zeeman_square.jl"
```

# Square Holstein Model with Zeeman Splitting
Download this example as a [Julia script](../assets/scripts/examples/holstein_zeeman_square.jl).

In this example we write a script to simulate the Holstein model on a square lattice in an external applied magentic field
perpendicular to the lattice, manifesting as a Zeeman splitting in the on-site energies between the spin up and down electrons.
More generally, this examples demonstrates how to simulate models with explicit spin-dependence appearing in the Hamiltonian.
The Hamiltonian is given by
```math
\begin{align*}
\hat{H} = & -t \sum_{\sigma,\langle i, j \rangle} (\hat{c}^{\dagger}_{\sigma,i}, \hat{c}^{\phantom \dagger}_{\sigma,j} + {\rm h.c.})
            + \sum_{\sigma,i}(\epsilon_\sigma - \mu) \hat{n}_{\sigma,i}\\
          & + \alpha \sum_{\sigma,i} \hat{X}_i (\hat{n}_{\sigma,i} - \tfrac{1}{2}) \\
          & + \sum_i \left( \frac{1}{2M}\hat{P}_i^2 + \frac{1}{2}M\Omega^2\hat{X}_i^2 \right),
\end{align*}
```
where ``\hat{c}^\dagger_{\sigma,i} \ (\hat{c}^{\phantom \dagger}_{\sigma,i})`` creates (annihilates) a spin ``\sigma``
electron on site ``i`` in the lattice, and ``\hat{n}_{\sigma,i} = \hat{c}^\dagger_{\sigma,i} \hat{c}^{\phantom \dagger}_{\sigma,i}``
is the spin-``\sigma`` electron number operator for site ``i``. The nearest-neighbor hopping amplitude is ``t`` and ``\mu`` is the
chemical potential.
The Zeeman splitting is reflected in the on-site energies taking on spin-resolved values ``\epsilon_\pm = \pm \Delta\epsilon/2``.
The strength of the repulsive Hubbard interaction is controlled by ``U>0``. ``\hat{X}_i \ (\hat{P}_i)``
is the phonon position (momentum) operator for a dispersionless mode placed on site ``i`` with phonon frequency ``\Omega`` and
corresponding ion mass ``M``. The stength of the Holstein electron-phonon is controlled by the parameter ``\alpha``.

A short test simulation using the script associated with this example can be run as
```
> julia hubbard_holstein_square.jl 0 1.0 0.1 0.1 0.0 4.0 4 1000 5000 50
```
which simulates the Holstein model on a ``L = 4`` square lattice, with ``\Delta\epsilon = 1.0``, ``\Omega = 0.1``, ``\alpha = 0.1``
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
function run_holstein_zeeman_square_simulation(sID, Δϵ, Ω, α, μ, β, L, N_burnin, N_updates, N_bins; filepath = ".")

    # Construct the foldername the data will be written to.
    datafolder_prefix = @sprintf "holstein_zeeman_square_ez%.2f_w%.2f_a%.2f_mu%.2f_L%d_b%.2f" Δϵ Ω α μ L β

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

    # Next we define the various Hamiltonian parameters for spin-up and spin-down seperately.
    # Here, only the on-site energy will be different between the two spin species, but in
    # general any other parameter appearing in the tight-binding model could as well, including
    # including the mircoscropic interaction constants.

    # Define nearest-neighbor hopping amplitude, setting the energy scale for the system.
    t = 1.0

    # Define the spin-up tight-binding model
    tight_binding_model_up = TightBindingModel(
        model_geometry = model_geometry,
        t_bonds = [bond_px, bond_py],
        t_mean = [t, t],
        μ = μ,
        ϵ_mean = [+Δϵ]
    )

    # Define the spin-down tight-binding model
    tight_binding_model_dn = TightBindingModel(
        model_geometry = model_geometry,
        t_bonds = [bond_px, bond_py],
        t_mean = [t, t],
        μ = μ,
        ϵ_mean = [-Δϵ]
    )

    # Initialize a null electron-phonon model.
    electron_phonon_model = ElectronPhononModel(
        model_geometry = model_geometry,
        tight_binding_model_up = tight_binding_model_up,
        tight_binding_model_dn = tight_binding_model_up
    )

    # Define a dispersionless electron-phonon mode to live on each site in the lattice.
    phonon = PhononMode(orbital = 1, Ω_mean = Ω)

    # Add the phonon mode definition to the electron-phonon model.
    phonon_id = add_phonon_mode!(
        electron_phonon_model = electron_phonon_model,
        phonon_mode = phonon
    )

    # Define a spin-up on-site Holstein coupling between the electron and the local dispersionless phonon mode.
    holstein_coupling_up = HolsteinCoupling(
    	model_geometry = model_geometry,
    	phonon_mode = phonon_id,
    	bond = lu.Bond(orbitals = (1,1), displacement = [0,0]),
    	α_mean = α
    )

    # Define a spin-down on-site Holstein coupling between the electron and the local dispersionless phonon mode.
    holstein_coupling_dn = HolsteinCoupling(
    	model_geometry = model_geometry,
    	phonon_mode = phonon_id,
    	bond = lu.Bond(orbitals = (1,1), displacement = [0,0]),
    	α_mean = α
    )

    # Add the Holstein coupling definition to the model.
    holstein_coupling_id = add_holstein_coupling!(
    	electron_phonon_model = electron_phonon_model,
    	holstein_coupling_up = holstein_coupling_up,
        holstein_coupling_dn = holstein_coupling_dn,
    	model_geometry = model_geometry
    )

    # Write the model summary to file.
    model_summary(
        simulation_info = simulation_info,
        β = β, Δτ = Δτ,
        model_geometry = model_geometry,
        tight_binding_model_up = tight_binding_model_up,
        tight_binding_model_dn = tight_binding_model_dn,
        interactions = (electron_phonon_model,)
    )

    #########################################
    ### INITIALIZE FINITE MODEL PARAMETERS ##
    #########################################

    # Initialize spin-up tight-binding parameters.
    tight_binding_parameters_up = TightBindingParameters(
        tight_binding_model = tight_binding_model_up,
        model_geometry = model_geometry,
        rng = rng
    )

    # Initialize spin-down tight-binding parameters.
    tight_binding_parameters_dn = TightBindingParameters(
        tight_binding_model = tight_binding_model_dn,
        model_geometry = model_geometry,
        rng = rng
    )

    # Initialize electron-phonon parameters.
    electron_phonon_parameters = ElectronPhononParameters(
        β = β, Δτ = Δτ,
        electron_phonon_model = electron_phonon_model,
        tight_binding_parameters_up = tight_binding_parameters_up,
        tight_binding_parameters_dn = tight_binding_parameters_dn,
        model_geometry = model_geometry,
        rng = rng
    )

    ##############################
    ### INITIALIZE MEASUREMENTS ##
    ##############################

    # Initialize the container that measurements will be accumulated into.
    measurement_container = initialize_measurement_container(model_geometry, β, Δτ)

    # Initialize the tight-binding model related measurements, like the hopping energy.
    initialize_measurements!(measurement_container, tight_binding_model_up, tight_binding_model_up)

    # Initialize the electron-phonon interaction related measurements.
    initialize_measurements!(measurement_container, electron_phonon_model)

    # Now we define the various correlation function measurements we would like to make.
    # Note that relative to the other examples, we include spin-resolved correlation
    # measurements as the underlying model being simulated is spin-dependent.

    # Initialize the single-particle spin-up electron Green's function measurement.
    initialize_correlation_measurements!(
        measurement_container = measurement_container,
        model_geometry = model_geometry,
        correlation = "greens_up",
        time_displaced = true,
        pairs = [(1, 1)]
    )

    # Initialize the single-particle spin-down electron Green's function measurement.
    initialize_correlation_measurements!(
        measurement_container = measurement_container,
        model_geometry = model_geometry,
        correlation = "greens_dn",
        time_displaced = true,
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

    # Initialize total density correlation function measurement.
    initialize_correlation_measurements!(
        measurement_container = measurement_container,
        model_geometry = model_geometry,
        correlation = "density",
        time_displaced = false,
        integrated = true,
        pairs = [(1, 1)]
    )

    # Initialize spin-up density correlation function measurement.
    initialize_correlation_measurements!(
        measurement_container = measurement_container,
        model_geometry = model_geometry,
        correlation = "density_upup",
        time_displaced = false,
        integrated = true,
        pairs = [(1, 1)]
    )

    # Initialize spin-down density correlation function measurement.
    initialize_correlation_measurements!(
        measurement_container = measurement_container,
        model_geometry = model_geometry,
        correlation = "density_dndn",
        time_displaced = false,
        integrated = true,
        pairs = [(1, 1)]
    )

    # Initialize mixed-spin density correlation function measurement.
    initialize_correlation_measurements!(
        measurement_container = measurement_container,
        model_geometry = model_geometry,
        correlation = "density_updn",
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

    # Initialize the total current correlation function measurement
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
    fermion_path_integral_up = FermionPathIntegral(tight_binding_parameters = tight_binding_parameters_up, β = β, Δτ = Δτ)
    fermion_path_integral_dn = FermionPathIntegral(tight_binding_parameters = tight_binding_parameters_dn, β = β, Δτ = Δτ)

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
                model_geometry = model_geometry,
                tight_binding_parameters_up = tight_binding_parameters_up,
                tight_binding_parameters_dn = tight_binding_parameters_dn,
                coupling_parameters = (electron_phonon_parameters,)
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

    # Write simulation summary TOML file.
    save_simulation_info(simulation_info, additional_info)

    return nothing
end


# Only excute if the script is run directly from the command line.
if abspath(PROGRAM_FILE) == @__FILE__

    # Read in the command line arguments.
    sID = parse(Int, ARGS[1]) # simulation ID
    Δϵ = parse(Float64, ARGS[2])
    Ω = parse(Float64, ARGS[3])
    α = parse(Float64, ARGS[4])
    μ = parse(Float64, ARGS[5])
    β = parse(Float64, ARGS[6])
    L = parse(Int, ARGS[7])
    N_burnin = parse(Int, ARGS[8])
    N_updates = parse(Int, ARGS[9])
    N_bins = parse(Int, ARGS[10])

    # Run the simulation.
    run_holstein_zeeman_square_simulation(sID, Δϵ, Ω, α, μ, β, L, N_burnin, N_updates, N_bins)
end
````

