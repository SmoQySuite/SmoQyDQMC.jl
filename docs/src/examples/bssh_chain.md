```@meta
EditURL = "../../../examples/bssh_chain.jl"
```

# Bond Su-Schrieffer-Heeger Chain
Download this example as a [Julia script](../assets/scripts/examples/bssh_chain.jl).

In this example we simulate the bond Su-Schrieffer-Heeger (BSSH) model on a 1D chain,
with a Hamiltonian given by
```math
\begin{align*}
\hat{H} = \sum_i \left( \frac{1}{2M}\hat{P}_{\langle i+1, i \rangle}^2 + \frac{1}{2}M\Omega^2\hat{X}_{\langle i+1, i \rangle}^2 \right)
          - \sum_{\sigma,i} [t-\alpha \hat{X}_{\langle i+1, i \rangle}] (\hat{c}^{\dagger}_{\sigma,i+1}, \hat{c}^{\phantom \dagger}_{\sigma,i} + {\rm h.c.})
          - \mu \sum_{\sigma,i} \hat{n}_{\sigma,i},
\end{align*}
```
in which dispersionless phonon modes are placed on each *bond*, and their positions modulates only that single corresponding hopping amplitude.
In the above expression ``\hat{c}^\dagger_{\sigma,i} \ (\hat{c}^{\phantom \dagger}_{\sigma,i})`` creation (annihilation) operator
a spin ``\sigma`` electron on site ``i`` in the lattice, and ``\hat{n}_{\sigma,i} = \hat{c}^\dagger_{\sigma,i} \hat{c}^{\phantom \dagger}_{\sigma,i}``
is corresponding electron number operator. The phonon position (momentum) operator for the dispersionless phonon mode on the
bond connecting sites ``i`` and ``i+1`` is given by ``\hat{X}_{\langle i+1, i \rangle} \ (\hat{P}_{\langle i+1, i \rangle})``,
where ``\Omega`` and ``M`` are the phonon frequency and associated ion mass respectively.
Lastly, the strength of the electron-phonon coupling is controlled by the parameter ``\alpha``.

A short test simulation using the script associated with this example can be run as
```
> julia bssh_chain.jl 0 1.0 0.5 0.0 4.0 16 1000 5000 20
```
which simulates an ``L=16`` chain with ``\Omega = 1.0``, ``\alpha = 0.5`` at half-filling ``(\mu = 0.0)`` and
an inverse temperature of ``\beta = 4.0``. In this example `N_burnin = 1000` HMC thermalization updates are performed,
followed an additional `N_updates = 5000` HMC updates, after each of which measurements are made.
Bin averaged measurements are then written to file `N_bins = 20` during the simulation.

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

# Define top-level function for running the DQMC simulation.
function run_bssh_chain_simulation(sID, Ω, α, μ, β, L, N_burnin, N_updates, N_bins; filepath = ".")

    # Construct the foldername the data will be written to.
    datafolder_prefix = @sprintf "bssh_chain_w%.2f_a%.2f_mu%.2f_L%d_b%.2f" Ω α μ L β

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

    # For performance reasons it is important that we represent the exponentiated hopping
    # matrix with the checkerboard approximation when simulating an SSH model, where the
    # phonons modulate the hopping amplitudes. Without the checkerboard approximation,
    # each time a phonon field is updated the kinetic energy matrix would need to be diagonalized
    # to calculate its exponential, which is very computationally expensive.

    # This flag indicates whether or not to use the checkboard approximation to
    # represent the exponentiated hopping matrix exp(-Δτ⋅K)
    checkerboard = true

    # As we are using the checkboard approximation, using a symmetric definition for the propagator
    # matrices is important as it significantly improves the accuracy of approximation.

    # Whether the propagator matrices should be represented using the
    # symmetric form B = exp(-Δτ⋅K/2)⋅exp(-Δτ⋅V)⋅exp(-Δτ⋅K/2)
    # or the asymetric form B = exp(-Δτ⋅V)⋅exp(-Δτ⋅K)
    symmetric = true

    # Set the initial period in imaginary time slices with which the Green's function matrices
    # will be recomputed using a numerically stable procedure.
    n_stab = 10

    # Specify the maximum allowed error in any element of the Green's function matrix that is
    # corrected by performing numerical stabiliziation.
    δG_max = 1e-6

    # Calculate the bin size.
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
        "swap_acceptance_rate" => 0.0,
        "n_stab_init" => n_stab,
        "symmetric" => symmetric,
        "checkerboard" => checkerboard,
        "Nt" => Nt,
        "dt" => Δt,
        "seed" => seed,
    )

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

    # Unlike in the optical SSH model in the previous example, here we need to
    # introduce two types of phonon modes. One of these phonon modes will have
    # infinite ion mass, resulting in the associated phonon fields remaining
    # pinned at zero. The means that when we couple these two types of phonon
    # modes to the electrons with a SSH-like coupling mechanism, this effectively
    # results in defining a phonon modes associated with a single bond/hopping
    # in the lattice.

    # Define a dispersionless electron-phonon mode to live on each site in the lattice.
    phonon = PhononMode(orbital = 1, Ω_mean = Ω, M = 1.0)

    # Add optical ssh phonon to electron-phonon model.
    phonon_id = add_phonon_mode!(
        electron_phonon_model = electron_phonon_model,
        phonon_mode = phonon
    )

    # Define a frozen phonon mode.
    frozen_phonon = PhononMode(orbital = 1, Ω_mean = Ω, M = Inf)

    # Add frozen phonon mode to electron-phonon model.
    frozen_phonon_id = add_phonon_mode!(
        electron_phonon_model = electron_phonon_model,
        phonon_mode = frozen_phonon
    )

    # Define bond SSH coupling.
    # Defines total effective hopping amplitude given by t_eff = t-α⋅X(i+1,i).
    bssh_coupling = SSHCoupling(
        model_geometry = model_geometry,
        tight_binding_model = tight_binding_model,
        phonon_modes = (frozen_phonon_id, phonon_id),
        bond = bond,
        α_mean = α
    )

    # Add bond SSH coupling to the electron-phonon model.
    bssh_coupling_id = add_ssh_coupling!(
        electron_phonon_model = electron_phonon_model,
        ssh_coupling = bssh_coupling,
        tight_binding_model = tight_binding_model
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

    # Initialize the measurement container.
    measurement_container = initialize_measurement_container(model_geometry, β, Δτ)

    # Initialize the measurements associated with the tight-binding model.
    initialize_measurements!(measurement_container, tight_binding_model)

    # Initialize the measurements associated with the electron-phonon model.
    initialize_measurements!(measurement_container, electron_phonon_model)

    # Initialize time-displaced Green's function measurement.
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

    # Initialize the density correlation function measurement.
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

    # Initialize the pair correlation function measurements.
    initialize_correlation_measurements!(
        measurement_container = measurement_container,
        model_geometry = model_geometry,
        correlation = "pair",
        time_displaced = false,
        integrated = true,
        pairs = [(1, 1), (bond_id, bond_id)]
    )

    # Initialize the bond correlation function measurement.
    initialize_correlation_measurements!(
        measurement_container = measurement_container,
        model_geometry = model_geometry,
        correlation = "bond",
        time_displaced = false,
        integrated = true,
        pairs = [(bond_id, bond_id)]
    )

    # Initialize current-current correlation function measurement.
    initialize_correlation_measurements!(
        measurement_container = measurement_container,
        model_geometry = model_geometry,
        correlation = "current",
        time_displaced = false,
        integrated = true,
        pairs = [(1, 1)] # Hopping ID pair.
    )

    # Initialize the sub-directories the various measurements will be written to.
    initialize_measurement_directories(
        simulation_info = simulation_info,
        measurement_container = measurement_container
    )

    #############################
    ### SET-UP DQMC SIMULATION ##
    #############################

    # Allocate FermionPathIntegral type.
    fermion_path_integral = FermionPathIntegral(tight_binding_parameters = tight_binding_parameters, β = β, Δτ = Δτ)

    # Initialize the FermionPathIntegral type
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

    ####################################
    ### BURNIN/THERMALIZATION UPDATES ##
    ####################################

    # Iterate over burnin/thermalization updates.
    for n in 1:N_burnin

        # Perform a swap update.
        # In a swap update, two phonon modes are randomly selected in the lattice
        # and their phonon fields are exchanged for all imaginary time slices.
        (accepted, logdetG, sgndetG) = swap_update!(
            G, logdetG, sgndetG, electron_phonon_parameters,
            fermion_path_integral = fermion_path_integral,
            fermion_greens_calculator = fermion_greens_calculator,
            fermion_greens_calculator_alt = fermion_greens_calculator_alt,
            B = B, rng = rng, phonon_type_pairs = ((phonon_id, phonon_id),)
        )

        # Record whether the swap update was accepted or rejected.
        additional_info["swap_acceptance_rate"] += accepted

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

            # Perform a swap update..
            (accepted, logdetG, sgndetG) = swap_update!(
                G, logdetG, sgndetG, electron_phonon_parameters,
                fermion_path_integral = fermion_path_integral,
                fermion_greens_calculator = fermion_greens_calculator,
                fermion_greens_calculator_alt = fermion_greens_calculator_alt,
                B = B, rng = rng, phonon_type_pairs = ((phonon_id, phonon_id),)
            )

            # Record whether the swap update was accepted or rejected.
            additional_info["swap_acceptance_rate"] += accepted

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

    # Calculate swap update acceptance rate.
    additional_info["swap_acceptance_rate"] /= (N_updates + N_burnin)

    # Record the final numerical stabilization period that the simulation settled on.
    additional_info["n_stab_final"] = fermion_greens_calculator.n_stab

    # Record the maximum numerical error corrected by numerical stablization.
    additional_info["dG"] = δG

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
    μ = parse(Float64, ARGS[4])
    β = parse(Float64, ARGS[5])
    L = parse(Int, ARGS[6])
    N_burnin = parse(Int, ARGS[7])
    N_updates = parse(Int, ARGS[8])
    N_bins = parse(Int, ARGS[9])

    # Run the simulation.
    run_bssh_chain_simulation(sID, Ω, α, μ, β, L, N_burnin, N_updates, N_bins)
end
````

