```@meta
EditURL = "../../../examples/holstein_chain.jl"
```

# Holstein Chain with Density Tuning
Download this example as a [Julia script](../assets/scripts/examples/holstein_chain.jl).

In this example we simulate the Holstein model for electron-phonon models on a 1D chain given by
```math
\hat{H} = \hat{H}_e + \hat{H}_{\rm ph} + \hat{H}_{e\textrm{-ph}}.
```
The first term describes bare tight-binding Hamiltonian
```math
\hat{H}_e = -t \sum_{\sigma,i} (\hat{c}^{\dagger}_{\sigma,i+1}, \hat{c}^{\phantom \dagger}_{\sigma,i} + {\rm h.c.})
            -\mu \sum_{\sigma,i} \hat{n}_{\sigma,i},
```
where ``\hat{c}^\dagger_{\sigma,i} \ (\hat{c}^{\phantom \dagger}_{\sigma,i})`` creates (annihilates) a spin ``\sigma``
electron on site ``i`` in the lattice, and ``\hat{n}_{\sigma,i} = \hat{c}^\dagger_{\sigma,i} \hat{c}^{\phantom \dagger}_{\sigma,i}``
is the spin-``\sigma`` electron number operator for site ``i``.
The second terms is the bare phonon Hamiltonian
```math
\hat{H}_{\rm ph} = \sum_i \left( \frac{1}{2M}\hat{P}_i^2 + \frac{1}{2}M\Omega^2\hat{X}_i^2 \right)
```
which describes a dispersionless phonon modes on each site in the lattice, where ``\hat{X}_i \ (\hat{P}_i)`` is the
position (momentum) operator for the phonon mode on site ``i`` in the lattice. The phonon frequency is given by ``\Omega``,
and ``M`` is the corresponding ion mass associated with the phonon mode.
Lastly, the third term introduces the Holstein electron-phonon interaction
```math
\hat{H}_{e\textrm{-ph}} = \alpha \sum_{\sigma,i} \hat{X}_i \left( \hat{n}_{\sigma,i} - \frac{1}{2} \right),
```
where the strength of the interaction is controlled by the parameter ``\alpha``.

The example script [`example_scripts/holstein_chain.jl`](https://github.com/SmoQySuite/SmoQyDQMC.jl/blob/main/example_scripts/holstein_chain.jl)
additionally simulate the Holstein model with an algorithm recently recently introduced in
[`Phys. Rev. E 105, 045311`](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.105.045311)
that dynamically tunes the chemical potential during the simulation to achieve a target electron density.

A short test simulation using the script associated with this example can be run as
```
> julia holstein_chain.jl 0 0.1 0.1 0.5 0.0 4.0 16 2000 10000 20
```
Here a Holstein chain of length ``L=16`` is simulate with ``\Omega = 0.1``, ``\alpha = 0.1`` and inverse temperature ``\beta = 4.0``.
The chemical potential is initialized to ``\mu = 0.0``, and then tuned to achieve are target electron density of ``\langle n \rangle = 0.5``.
In this example `N_burnin = 2000` thermalizatoin HMC and refleciton updates are performed, followed by an additional `N_updates = 10000`
such updates, during which time an equivalent number of measurements are made. Bin averaged measurements are written to
file `N_bins = 20` during the simulation.

Below you will find the source code from the julia script linked at the top of this page,
but with additional comments giving more detailed explanations for what certain parts of the code are doing.

````julia
using LinearAlgebra
using Random
using Printf

using SmoQyDQMC
import SmoQyDQMC.LatticeUtilities  as lu
import SmoQyDQMC.JDQMCFramework    as dqmcf
import SmoQyDQMC.JDQMCMeasurements as dqmcm
# Import the MuTuner module that implements the chemical potential tuning algorithm.
import SmoQyDQMC.MuTuner           as mt

# Define top-level function for running the DQMC simulation.
function run_holstein_chain_simulation(sID, Ω, α, n, μ, β, L, N_burnin, N_updates, N_bins; filepath = ".")

    # Construct the foldername the data will be written to.
    datafolder_prefix = @sprintf "holstein_chain_w%.2f_a%.2f_n%.2f_L%d_b%.2f" Ω α n L β

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
    Δτ = 0.05

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

    # Calculate the bin size.
    bin_size = div(N_updates, N_bins)

    # To update the phonon degrees of freedom in this code we primarily perform
    # hybrid/hamiltonian Monte Carlo (HMC) updates. Below we specify some of the
    # parameters associated with these HMC updates.

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
        "radial_acceptance_rate" => 0.0,
        "n_stab_init" => n_stab,
        "symmetric" => symmetric,
        "checkerboard" => checkerboard,
        "Nt" => Nt,
        "dt" => Δt,
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

    # Get the number of sites in the lattice.
    N = lu.nsites(unit_cell, lattice)

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
    	bond = lu.Bond(orbitals = (1,1), displacement = [0]),
    	α_mean = α
    )

    # Add the Holstein coupling definition to the model.
    holstein_coupling_id = add_holstein_coupling!(
    	electron_phonon_model = electron_phonon_model,
    	holstein_coupling = holstein_coupling,
    	model_geometry = model_geometry
    )

    # Write a model summary to file.
    model_summary(
        simulation_info = simulation_info,
        β = β, Δτ = Δτ,
        model_geometry = model_geometry,
        tight_binding_model = tight_binding_model,
        interactions = (electron_phonon_model,)
    )

    #################################################
    ### INITIALIZE FINITE LATTICE MODEL PARAMETERS ##
    #################################################

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

    ##############################
    ### INITIALIZE MEASUREMENTS ##
    ##############################

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
        pairs = [(1, 1)]
    )

    # Initialize time-displaced phonon Green's function measurement.
    initialize_correlation_measurements!(
        measurement_container = measurement_container,
        model_geometry = model_geometry,
        correlation = "phonon_greens",
        time_displaced = true,
        pairs = [(phonon_id, phonon_id)]
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
    ### SET-UP DQMC SIMULATION ##
    #############################

    # Note that the spin-up and spin-down electron sectors are equivalent in the Holstein model
    # without Hubbard interaction. Therefore, there is only a single Fermion determinant
    # that needs to be calculated. This fact is reflected in the code below.

    # Allocate fermion path integral type.
    fermion_path_integral = FermionPathIntegral(tight_binding_parameters = tight_binding_parameters, β = β, Δτ = Δτ)

    # Initialize the fermion path integral type with respect to electron-phonon interaction.
    initialize!(fermion_path_integral, electron_phonon_parameters)

    # Allocate and initialize propagators for each imaginary time slice.
    B = initialize_propagators(fermion_path_integral, symmetric=symmetric, checkerboard=checkerboard)

    # Initialize fermion greens calculator.
    fermion_greens_calculator = dqmcf.FermionGreensCalculator(B, β, Δτ, n_stab)

    # Initialize alternate fermion greens calculator required for performing various global updates.
    fermion_greens_calculator_alt = dqmcf.FermionGreensCalculator(fermion_greens_calculator)

    # Allocate equal-time Green's function matrix.
    G = zeros(eltype(B[1]), size(B[1]))

    # Initialize equal-time Green's function matrix
    logdetG, sgndetG = dqmcf.calculate_equaltime_greens!(G, fermion_greens_calculator)

    # Allocate matrices for various time-displaced Green's function matrices.
    G_ττ = similar(G) # G(τ,τ)
    G_τ0 = similar(G) # G(τ,0)
    G_0τ = similar(G) # G(0,τ)

    # Initialize variables to keep track of the largest numerical error in the
    # Green's function matrices corrected by numerical stabalization.
    δG = zero(typeof(logdetG))
    δθ = zero(typeof(sgndetG))

    # Initialize Hamitlonian/Hybrid monte carlo (HMC) updater.
    hmc_updater = EFAHMCUpdater(
        electron_phonon_parameters = electron_phonon_parameters,
        G = G, Nt = Nt, Δt = Δt
    )

    # Initialize the density/chemical potential tuner.
    # This type facilitates the tuning of the chemical potential to achieve
    # at target electron density.
    chemical_potential_tuner = mt.MuTunerLogger(n₀ = n, β = β, V = N, u₀ = 1.0, μ₀ = μ, c = 0.5)

    ####################################
    ### BURNIN/THERMALIZATION UPDATES ##
    ####################################

    # Iterate over burnin/thermalization updates.
    for n in 1:N_burnin

        # Perform a reflection update.
        # This update randomly selects a phonon mode in the lattice and reflects
        # all the associated phonon about the origin, (xᵢ → -xᵢ).
        # This updates all the phonon fields to cross the on-site energy barrier
        # associated with bipolaron formation, helping reduce autocorrelation times.
        (accepted, logdetG, sgndetG) = reflection_update!(
            G, logdetG, sgndetG, electron_phonon_parameters,
            fermion_path_integral = fermion_path_integral,
            fermion_greens_calculator = fermion_greens_calculator,
            fermion_greens_calculator_alt = fermion_greens_calculator_alt,
            B = B, rng = rng, phonon_types = (phonon_id,)
        )

        # Record whether the reflection update was accepted or rejected.
        additional_info["reflection_acceptance_rate"] += accepted

        # Perform a radial update.
        (accepted, logdetG, sgndetG) = radial_update!(
            G, logdetG, sgndetG, electron_phonon_parameters,
            fermion_path_integral = fermion_path_integral,
            fermion_greens_calculator = fermion_greens_calculator,
            fermion_greens_calculator_alt = fermion_greens_calculator_alt,
            B = B, rng = rng
        )

        # Record whether the reflection update was accepted or rejected.
        additional_info["radial_acceptance_rate"] += accepted

        # Perform an HMC update.
        (accepted, logdetG, sgndetG, δG, δθ) = hmc_update!(
            G, logdetG, sgndetG, electron_phonon_parameters, hmc_updater,
            fermion_path_integral = fermion_path_integral,
            fermion_greens_calculator = fermion_greens_calculator,
            fermion_greens_calculator_alt = fermion_greens_calculator_alt,
            B = B, δG_max = δG_max, δG = δG, δθ = δθ, rng = rng
        )

        # Record whether the HMC update was accepted or rejected.
        additional_info["hmc_acceptance_rate"] += accepted

        # Update the chemical potential.
        logdetG, sgndetG = update_chemical_potential!(
            G, logdetG, sgndetG,
            chemical_potential_tuner = chemical_potential_tuner,
            tight_binding_parameters = tight_binding_parameters,
            fermion_path_integral = fermion_path_integral,
            fermion_greens_calculator = fermion_greens_calculator,
            B = B
        )
    end

    ################################
    ### START MAKING MEAUSREMENTS ##
    ################################

    # Re-initialize variables to keep track of the largest numerical error in the
    # Green's function matrices corrected by numerical stabalization.
    δG = zero(typeof(logdetG))
    δθ = zero(typeof(sgndetG))

    # Iterate over the number of bin, i.e. the number of time measurements will be dumped to file.
    for bin in 1:N_bins

        # Iterate over the number of updates and measurements performed in the current bin.
        for n in 1:bin_size

            # Perform a reflection update.
            (accepted, logdetG, sgndetG) = reflection_update!(
                G, logdetG, sgndetG, electron_phonon_parameters,
                fermion_path_integral = fermion_path_integral,
                fermion_greens_calculator = fermion_greens_calculator,
                fermion_greens_calculator_alt = fermion_greens_calculator_alt,
                B = B, rng = rng, phonon_types = (phonon_id,)
            )

            # Record whether the reflection update was accepted or rejected.
            additional_info["reflection_acceptance_rate"] += accepted

            # Perform a radial update.
            (accepted, logdetG, sgndetG) = radial_update!(
                G, logdetG, sgndetG, electron_phonon_parameters,
                fermion_path_integral = fermion_path_integral,
                fermion_greens_calculator = fermion_greens_calculator,
                fermion_greens_calculator_alt = fermion_greens_calculator_alt,
                B = B, rng = rng
            )

            # Record whether the reflection update was accepted or rejected.
            additional_info["radial_acceptance_rate"] += accepted

            # Perform an HMC update.
            (accepted, logdetG, sgndetG, δG, δθ) = hmc_update!(
                G, logdetG, sgndetG, electron_phonon_parameters, hmc_updater,
                fermion_path_integral = fermion_path_integral,
                fermion_greens_calculator = fermion_greens_calculator,
                fermion_greens_calculator_alt = fermion_greens_calculator_alt,
                B = B, δG_max = δG_max, δG = δG, δθ = δθ, rng = rng
            )

            # Record whether the HMC update was accepted or rejected.
            additional_info["hmc_acceptance_rate"] += accepted

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

            # Update the chemical potential.
            logdetG, sgndetG = update_chemical_potential!(
                G, logdetG, sgndetG,
                chemical_potential_tuner = chemical_potential_tuner,
                tight_binding_parameters = tight_binding_parameters,
                fermion_path_integral = fermion_path_integral,
                fermion_greens_calculator = fermion_greens_calculator,
                B = B
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

    # Calculate HMC acceptance rate.
    additional_info["hmc_acceptance_rate"] /= (N_updates + N_burnin)

    # Calculate reflection update acceptance rate.
    additional_info["reflection_acceptance_rate"] /= (N_updates + N_burnin)

    # Calculate radial update acceptance rate.
    additional_info["radial_acceptance_rate"] /= (N_updates + N_burnin)

    # Record the final numerical stabilization period that the simulation settled on.
    additional_info["n_stab_final"] = fermion_greens_calculator.n_stab

    # Record the maximum numerical error corrected by numerical stablization.
    additional_info["dG"] = δG

    # Save the density tuning profile.
    save_density_tuning_profile(simulation_info, chemical_potential_tuner)

    # Write simulation summary TOML file.
    save_simulation_info(simulation_info, additional_info)

    #################################
    ### PROCESS SIMULATION RESULTS ##
    #################################

    # Process the simulation results, calculating final error bars for all measurements,
    # writing final statisitics to CSV files.
    process_measurements(simulation_info.datafolder, N_bins)

    return nothing
end

# Only excute if the script is run directly from the command line.
if abspath(PROGRAM_FILE) == @__FILE__

    # Read in the command line arguments.
    sID = parse(Int, ARGS[1]) # simulation ID
    Ω = parse(Float64, ARGS[2])
    α = parse(Float64, ARGS[3])
    n = parse(Float64, ARGS[4]) # target electorn density
    μ = parse(Float64, ARGS[5]) # intial chemical potential
    β = parse(Float64, ARGS[6])
    L = parse(Int, ARGS[7])
    N_burnin = parse(Int, ARGS[8])
    N_updates = parse(Int, ARGS[9])
    N_bins = parse(Int, ARGS[10])

    # Run the simulation.
    run_holstein_chain_simulation(sID, Ω, α, n, μ, β, L, N_burnin, N_updates, N_bins)
end
````

