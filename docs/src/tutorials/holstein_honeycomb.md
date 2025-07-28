```@meta
EditURL = "../../../tutorials/holstein_honeycomb.jl"
```

# 2a) Honeycomb Holstein Model
Download this example as a [Julia script](../assets/scripts/tutorials/holstein_honeycomb.jl).

In this example we will work through simulating the Holstein model on a honeycomb lattice.
The Holstein Hamiltonian is given by
```math
\begin{align*}
\hat{H} = & -t \sum_{\langle i, j \rangle, \sigma} (\hat{c}^{\dagger}_{\sigma,i}, \hat{c}^{\phantom \dagger}_{\sigma,j} + {\rm h.c.})
- \mu \sum_{i,\sigma} \hat{n}_{\sigma,i} \\
& + \frac{1}{2} M \Omega^2 \sum_{i} \hat{X}_i^2 + \sum_i \frac{1}{2M} \hat{P}_i^2
+ \alpha \sum_i \hat{X}_i (\hat{n}_{\uparrow,i} + \hat{n}_{\downarrow,i} - 1)
\end{align*}
```
where ``\hat{c}^\dagger_{\sigma,i} \ (\hat{c}^{\phantom \dagger}_{\sigma,i})`` creates (annihilates) a spin ``\sigma``
electron on site ``i`` in the lattice, and ``\hat{n}_{\sigma,i} = \hat{c}^\dagger_{\sigma,i} \hat{c}^{\phantom \dagger}_{\sigma,i}``
is the spin-``\sigma`` electron number operator for site ``i``. Here ``\mu`` is the chemical potential and  ``t`` is the nearest-neighbor
hopping amplitude, with the sum over ``\langle i,j \rangle`` denoting a sum over all nearest-neighbor pairs of sites.
A local dispersionless phonon mode is then placed on each site in the lattice, with ``\hat{X}_i`` and ``\hat{P}_i`` the corresponding
phonon position and momentum operator on site ``i`` in the lattice. The phonon mass and energy are denoted ``M`` and ``\Omega`` respectively.
Lastly, the phonon displacement ``\hat{X}_i`` couples to the total local density ``\hat{n}_{\uparrow,i} + \hat{n}_{\downarrow,i},`` with the
parameter ``\alpha`` controlling the strength of this coupling.

## Import packages
As in the previous tutorial, we begin by importing the necessary packages;
for more details refer to [here.](@ref hubbard_square_import_packages)

````julia
using SmoQyDQMC
import SmoQyDQMC.LatticeUtilities as lu
import SmoQyDQMC.JDQMCFramework as dqmcf

using Random
using Printf
````

## Specify simulation parameters
The entire main body of the simulation we will wrapped in a top-level function named `run_simulation`
that will take as keyword arguments various model and simulation parameters that we may want to change.
The function arguments with default values are ones that are typically left unchanged between simulations.

````julia
# Top-level function to run simulation.
function run_simulation(;
    # KEYWORD ARGUMENTS
    sID, # Simulation ID.
    Ω, # Phonon energy.
    α, # Electron-phonon coupling.
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
In this first part of the script we name and initialize our simulation, record important metadata about the simulation
and create the data folder our simulation results will be written to.
For more information refer to [here.](@ref hubbard_square_initialize_simulation)

````julia
    # Construct the foldername the data will be written to.
    datafolder_prefix = @sprintf "holstein_honeycomb_w%.2f_a%.2f_mu%.2f_L%d_b%.2f" Ω α μ L β

    # Initialize simulation info.
    simulation_info = SimulationInfo(
        filepath = filepath,
        datafolder_prefix = datafolder_prefix,
        sID = sID
    )

    # Initialize the directory the data will be written to.
    initialize_datafolder(simulation_info)
````

## Initialize simulation metadata
In this section of the code we record important metadata about the simulation, including initializing the random number
generator that will be used throughout the simulation.
The important metadata within the simulation will be recorded in the `metadata` dictionary.

````julia
    # Initialize random number generator
    rng = Xoshiro(seed)

    # Initialize additiona_info dictionary
    metadata = Dict()

    # Record simulation parameters.
    metadata["N_therm"] = N_therm
    metadata["N_updates"] = N_updates
    metadata["N_bins"] = N_bins
    metadata["n_stab"] = n_stab
    metadata["dG_max"] = δG_max
    metadata["symmetric"] = symmetric
    metadata["checkerboard"] = checkerboard
    metadata["seed"] = seed
````

Here we also update variables to keep track of the acceptance rates for the various types of Monte Carlo updates
that will be performed during the simulation. This will be discussed in more detail in later sections of the tutorial.

````julia
    metadata["hmc_acceptance_rate"] = 0.0
    metadata["reflection_acceptance_rate"] = 0.0
    metadata["swap_acceptance_rate"] = 0.0
````

## Initialize model
The next step is define the model we wish to simulate.
In this example the relevant model parameters the phonon energy ``\Omega`` (`Ω`), electron-phonon coupling ``\alpha`` (`α`),
chemical potential ``\mu`` (`μ`), and lattice size ``L`` (`L`).
The neasrest-neighbor hopping amplitude and phonon mass are normalized to unity, ``t = M = 1``.

````julia
    # Define the unit cell.
    unit_cell = lu.UnitCell(
        lattice_vecs = [[3/2,√3/2],
                        [3/2,-√3/2]],
        basis_vecs   = [[0.,0.],
                        [1.,0.]]
    )

    # Define finite lattice with periodic boundary conditions.
    lattice = lu.Lattice(
        L = [L, L],
        periodic = [true, true]
    )

    # Initialize model geometry.
    model_geometry = ModelGeometry(unit_cell, lattice)

    # Define the first nearest-neighbor bond in a honeycomb lattice.
    bond_1 = lu.Bond(orbitals = (1,2), displacement = [0,0])

    # Add the first nearest-neighbor bond in a honeycomb lattice to the model.
    bond_1_id = add_bond!(model_geometry, bond_1)

    # Define the second nearest-neighbor bond in a honeycomb lattice.
    bond_2 = lu.Bond(orbitals = (1,2), displacement = [-1,0])

    # Add the second nearest-neighbor bond in a honeycomb lattice to the model.
    bond_2_id = add_bond!(model_geometry, bond_2)

    # Define the third nearest-neighbor bond in a honeycomb lattice.
    bond_3 = lu.Bond(orbitals = (1,2), displacement = [0,-1])

    # Add the third nearest-neighbor bond in a honeycomb lattice to the model.
    bond_3_id = add_bond!(model_geometry, bond_3)
````

Next we specify the Honeycomb tight-binding term in our Hamiltonian with the [`TightBindingModel`](@ref) type.

````julia
    # Set neartest-neighbor hopping amplitude to unity,
    # setting the energy scale in the model.
    t = 1.0

    # Define the honeycomb tight-binding model.
    tight_binding_model = TightBindingModel(
        model_geometry = model_geometry,
        t_bonds        = [bond_1, bond_2, bond_3], # defines hopping
        t_mean         = [t, t, t], # defines corresponding hopping amplitude
        μ              = μ, # set chemical potential
        ϵ_mean         = [0.0, 0.0] # set the (mean) on-site energy
    )
````

Now we need to initialize the electron-phonon part of the Hamiltonian with the [`ElectronPhononModel`](@ref) type.

````julia
    # Initialize a null electron-phonon model.
    electron_phonon_model = ElectronPhononModel(
        model_geometry = model_geometry,
        tight_binding_model = tight_binding_model
    )
````

Then we need to define and add two types phonon modes to the model, one for each orbital in the Honeycomb unit cell,
using the [`PhononMode`](@ref) type and [`add_phonon_mode!`](@ref) function.

````julia
    # Define a dispersionless electron-phonon mode to live on each site in the lattice.
    phonon_1 = PhononMode(orbital = 1, Ω_mean = Ω)

    # Add the phonon mode definition to the electron-phonon model.
    phonon_1_id = add_phonon_mode!(
        electron_phonon_model = electron_phonon_model,
        phonon_mode = phonon_1
    )

    # Define a dispersionless electron-phonon mode to live on each site in the lattice.
    phonon_2 = PhononMode(orbital = 2, Ω_mean = Ω)

    # Add the phonon mode definition to the electron-phonon model.
    phonon_2_id = add_phonon_mode!(
        electron_phonon_model = electron_phonon_model,
        phonon_mode = phonon_2
    )
````

Now we need to define and add a local Holstein couplings to our model for each of the two phonon modes
in each unit cell using the [`HolsteinCoupling`](@ref) type and [`add_holstein_coupling!`](@ref) function.

````julia
    # Define first local Holstein coupling for first phonon mode.
    holstein_coupling_1 = HolsteinCoupling(
        model_geometry = model_geometry,
        phonon_mode = phonon_1_id,
        # Couple the first phonon mode to first orbital in the unit cell.
        bond = lu.Bond(orbitals = (1,1), displacement = [0, 0]),
        α_mean = α
    )

    # Add the first local Holstein coupling definition to the model.
    holstein_coupling_1_id = add_holstein_coupling!(
        electron_phonon_model = electron_phonon_model,
        holstein_coupling = holstein_coupling_1,
        model_geometry = model_geometry
    )

    # Define first local Holstein coupling for first phonon mode.
    holstein_coupling_2 = HolsteinCoupling(
        model_geometry = model_geometry,
        phonon_mode = phonon_2_id,
        # Couple the second phonon mode to second orbital in the unit cell.
        bond = lu.Bond(orbitals = (2,2), displacement = [0, 0]),
        α_mean = α
    )

    # Add the first local Holstein coupling definition to the model.
    holstein_coupling_2_id = add_holstein_coupling!(
        electron_phonon_model = electron_phonon_model,
        holstein_coupling = holstein_coupling_2,
        model_geometry = model_geometry
    )
````

Lastly, the [`model_summary`](@ref) function is used to write a `model_summary.toml` file,
completely specifying the Hamiltonian that will be simulated.

````julia
    # Write model summary TOML file specifying Hamiltonian that will be simulated.
    model_summary(
        simulation_info = simulation_info,
        β = β, Δτ = Δτ,
        model_geometry = model_geometry,
        tight_binding_model = tight_binding_model,
        interactions = (electron_phonon_model,)
    )
````

## Initialize model parameters
The next step is to initialize our model parameters given the size of our finite lattice.
To clarify, both the [`TightBindingModel`](@ref) and [`ElectronPhononModel`](@ref) types are agnostic to the size of the lattice being simulated,
defining the model in a translationally invariant way. As [SmoQyDQMC.jl](https://github.com/SmoQySuite/SmoQyDQMC.jl.git) supports
random disorder in the terms appearing in the Hamiltonian, it is necessary to initialize seperate parameter values for each unit cell in the lattice.
For instance, we need to initialize a seperate number to represent the on-site energy for each orbital in our finite lattice.

````julia
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
````

## Initialize meuasurements
Having initialized both our model and the corresponding model parameters,
the next step is to initialize the various measurements we want to make during our DQMC simulation.
For more information refer to [here.](@ref hubbard_square_initialize_measurements)

````julia
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
            (1, 1), (2, 2), (1, 2)
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
            (1, 1), (2, 2), (1, 2)
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
            (1, 1), (2, 2),
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
            (1, 1), (2, 2)
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
            (1, 1), (2, 2)
        ]
    )
````

It is also useful to initialize more specialized composite correlation function measurements.
Specifically, to detect the formation of charge-density wave order where the electrons preferentially
localize on one of the two sub-lattices of the honeycomb lattice, it is useful to measure the correlation function
```math
C_\text{cdw}(\mathbf{r},\tau) = \frac{1}{L^2}\sum_{\mathbf{i}} \langle \hat{\Phi}^{\dagger}_{\mathbf{i}+\mathbf{r}}(\tau) \hat{\Phi}^{\phantom\dagger}_{\mathbf{i}}(0) \rangle,
```
where
```math
\hat{\Phi}_{\mathbf{i}}(\tau) = \hat{n}_{\mathbf{i},A}(\tau) - \hat{n}_{\mathbf{i},B}(\tau)
```
and ``\hat{n}_{\mathbf{i},\gamma} = (\hat{n}_{\uparrow,\mathbf{i},o} + \hat{n}_{\downarrow,\mathbf{i},o})`` is the total electron number
operator for orbital ``\gamma \in \{A,B\}`` in unit cell ``\mathbf{i}``.
It is then also useful to calculate the corresponding structure factor ``S_\text{cdw}(\mathbf{q},\tau)`` and susceptibility ``\chi_\text{cdw}(\mathbf{q}).``
This can all be easily calculated using the [`initialize_composite_correlation_measurement!`](@ref) function, as shown below.

````julia
    # Initialize CDW correlation measurement.
    initialize_composite_correlation_measurement!(
        measurement_container = measurement_container,
        model_geometry = model_geometry,
        name = "cdw",
        correlation = "density",
        ids = [1, 2],
        coefficients = [1.0, -1.0],
        time_displaced = false,
        integrated = true
    )
````

The [`initialize_measurement_directories`](@ref) can now be used used to initialize the various subdirectories
in the data folder that the measurements will be written to.
Again, for more information refer to the [Simulation Output Overview](@ref) page.

````julia
    # Initialize the sub-directories to which the various measurements will be written.
    initialize_measurement_directories(simulation_info, measurement_container)
````

## Setup DQMC simulation
This section of the code sets up the DQMC simulation by allocating the initializing the relevant types and arrays we will need in the simulation.

This section of code is perhaps the most opaque and difficult to understand, and will be discussed in more detail once written.
That said, you do not need to fully comprehend everything that goes on in this section as most of it is fairly boilerplate,
and will not need to be changed much once written.
This is true even if you want to modify this script to perform a DQMC simulation for a different Hamiltonian.
For more information refer to [here](@ref hubbard_square_setup_dqmc).

````julia
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

    # Allcoate equal-time electron Green's function matrix.
    G = zeros(eltype(B[1]), size(B[1]))

    # Initialize electron Green's function matrx, also calculating the matrix determinant as the same time.
    logdetG, sgndetG = dqmcf.calculate_equaltime_greens!(G, fermion_greens_calculator)

    # Allocate matrices for various time-displaced Green's function matrices.
    G_ττ = similar(G) # G(τ,τ)
    G_τ0 = similar(G) # G(τ,0)
    G_0τ = similar(G) # G(0,τ)

    # Initialize diagonostic parameters to asses numerical stability.
    δG = zero(logdetG)
    δθ = zero(sgndetG)
````

## [Setup EFA-HMC Updates](@id holstein_square_efa-hmc_updates)
Before we begin the simulation, we also want to initialize an instance of the
[`EFAHMCUpdater`](@ref) type, which will be used to perform hybrid Monte Carlo (HMC)
udpates to the phonon fields that use exact fourier acceleration (EFA)
to further reduce autocorrelation times.

The two main parameters that need to be specified are the time-step size ``\Delta t`` and number of time-steps ``N_t``
performed in the HMC update, with the corresponding integrated trajectory time then equalling ``T_t = N_t \cdot \Delta t.``
Note that the computational cost of an HMC update is linearly proportional to ``N_t,`` while the acceptance rate is inversely
proportional to ``\Delta t.``

[Previous studies](https://arxiv.org/abs/2404.09723) have shown that a good place to start
with the integrated trajectory time ``T_t`` is a quarter the period of the bare phonon mode,
``T_t \approx \frac{1}{4} \left( \frac{2\pi}{\Omega} \right) = \pi/(2\Omega).``
It is also important to keep the acceptance rate for the HMC updates above ``\sim 90\%`` to help prevent numerical instabilities from occuring.

Based on user experience, a good (conservative) starting place is to set the number of time-step to ``N_t \approx 10,``
and then set the time-step size to ``\Delta t \approx \pi/(2\Omega N_t),``
effectively setting the integrated trajectory time to ``T_t = \pi/(2\Omega).``
Then, if the acceptance rate is too low you increase ``N_t,`` which results in a reduction of ``\Delta t.``
Conversely, if the acceptance rate is very high ``(\gtrsim 99 \% )`` it can be useful to decrease ``N_t``,
thereby increasing ``\Delta t,`` as this will reduce the computational cost of performing an EFA-HMC update.

````julia
    # Number of fermionic time-steps in HMC update.
    Nt = 10

    # Fermionic time-step used in HMC update.
    Δt = π/(2*Ω*Nt)

    # Initialize Hamitlonian/Hybrid monte carlo (HMC) updater.
    hmc_updater = EFAHMCUpdater(
        electron_phonon_parameters = electron_phonon_parameters,
        G = G, Nt = Nt, Δt = Δt
    )
````

## Thermalize system
The next section of code performs updates to thermalize the system prior to beginning measurements.
In addition to EFA-HMC updates that will be performed using the [`EFAHMCUpdater`](@ref) type initialized above and
the [`hmc_update!`](@ref) function below, we will also perform reflection and swap updates using the
[`reflection_update!`](@ref) and [`swap_update!`](@ref) functions respectively.

````julia
    # Iterate over number of thermalization updates to perform.
    for n in 1:N_therm

        # Perform a reflection update.
        (accepted, logdetG, sgndetG) = reflection_update!(
            G, logdetG, sgndetG, electron_phonon_parameters,
            fermion_path_integral = fermion_path_integral,
            fermion_greens_calculator = fermion_greens_calculator,
            fermion_greens_calculator_alt = fermion_greens_calculator_alt,
            B = B, rng = rng
        )

        # Record whether the reflection update was accepted or rejected.
        metadata["reflection_acceptance_rate"] += accepted

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
            B = B, δG_max = δG_max, δG = δG, δθ = δθ, rng = rng
        )

        # Record whether the HMC update was accepted or rejected.
        metadata["hmc_acceptance_rate"] += accepted
    end
````

## Make measurements
In this next section of code we continue to sample the phonon fields as above,
but will also begin making measurements as well. For more discussion on the overall
structure of this part of the code, refer to [here](@ref hubbard_square_make_measurements).

````julia
    # Reset diagonostic parameters used to monitor numerical stability to zero.
    δG = zero(logdetG)
    δθ = zero(sgndetG)

    # Calculate the bin size.
    bin_size = N_updates ÷ N_bins

    # Iterate over updates and measurements.
    for update in 1:N_updates

        # Perform a reflection update.
        (accepted, logdetG, sgndetG) = reflection_update!(
            G, logdetG, sgndetG, electron_phonon_parameters,
            fermion_path_integral = fermion_path_integral,
            fermion_greens_calculator = fermion_greens_calculator,
            fermion_greens_calculator_alt = fermion_greens_calculator_alt,
            B = B, rng = rng
        )

        # Record whether the reflection update was accepted or rejected.
        metadata["reflection_acceptance_rate"] += accepted

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
            B = B, δG_max = δG_max, δG = δG, δθ = δθ, rng = rng
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
            coupling_parameters = (electron_phonon_parameters,)
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
At this point we are done sampling and taking measurements.
Next, we want to calculate the final acceptance rate for the various types of
udpates we performed, as well as write the simulation metadata to file,
including the contents of the `metadata` dictionary.

````julia
    # Calculate acceptance rates.
    metadata["hmc_acceptance_rate"] /= (N_updates + N_therm)
    metadata["reflection_acceptance_rate"] /= (N_updates + N_therm)
    metadata["swap_acceptance_rate"] /= (N_updates + N_therm)

    # Record largest numerical error encountered during simulation.
    metadata["dG"] = δG

    # Write simulation metadata to simulation_info.toml file.
    save_simulation_info(simulation_info, metadata)
````

## Post-process results
In this final section of code we post-process the binned data.
This includes calculating final estimates for the mean and error of all measured observables.
The final statistics are written to CSV files using the function [`process_measurements`](@ref) function.
For more information refer to [here](@ref hubbard_square_process_results).

````julia
    # Process the simulation results, calculating final error bars for all measurements,
    # writing final statisitics to CSV files.
    process_measurements(simulation_info.datafolder, N_bins, time_displaced = true)

    # Merge binary files containing binned data into a single file.
    compress_jld2_bins(folder = simulation_info.datafolder)

    return nothing
end # end of run_simulation function
````

## Execute script

DQMC simulations are typically run from the command line as jobs on a computing cluster.
With this in mind, the following block of code only executes if the Julia script is run from the command line,
also reading in additional command line arguments.

````julia
# Only excute if the script is run directly from the command line.
if abspath(PROGRAM_FILE) == @__FILE__

    # Run the simulation.
    run_simulation(;
        sID       = parse(Int,     ARGS[1]), # Simulation ID.
        Ω         = parse(Float64, ARGS[2]), # Phonon energy.
        α         = parse(Float64, ARGS[3]), # Electron-phonon coupling.
        μ         = parse(Float64, ARGS[4]), # Chemical potential.
        L         = parse(Int,     ARGS[5]), # System size.
        β         = parse(Float64, ARGS[6]), # Inverse temperature.
        N_therm   = parse(Int,     ARGS[7]), # Number of thermalization updates.
        N_updates = parse(Int,     ARGS[8]), # Total number of measurements and measurement updates.
        N_bins    = parse(Int,     ARGS[9])  # Number of times bin-averaged measurements are written to file.
    )
end
````

For instance, the command
```
> julia holstein_honeycomb.jl 1 1.0 1.5 0.0 3 4.0 5000 10000 100
```
runs a DQMC simulation of a Holstein model on a ``3 \times 3`` unit cell (`N = 2 \times 3^2 = 18` site) honeycomb lattice
at half-filling ``(\mu = 0)`` and inverse temperature ``\beta = 4.0``.
The phonon energy is set to ``\Omega = 1.0`` and the electron-phonon coupling is set to ``\alpha = 1.5.``
In the DQMC simulation, 5,000 EFA-HMC, reflection and swap updates are performed to thermalize the system.
Then an additional 10,000 such udpates are performed, after each of set of which measurements are made.
During the simulation, bin-averaged measurements are written to file 100 times,
with each bin of data containing the average of 10,000/100 = 100 sequential measurements.

