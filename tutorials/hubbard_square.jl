# # 1a) Square Hubbard Model

# In this example we will work through simulating the repulsive Hubbard model on a square lattice.
# The Hubbard Hamiltonian for a square lattice given by
# ```math
# \begin{align}
# \hat{H} = &
# -t \sum_{\langle i, j \rangle, \sigma} (\hat{c}^{\dagger}_{\sigma,i}, \hat{c}^{\phantom \dagger}_{\sigma,j} + {\rm h.c.})
# -t^{\prime} \sum_{\langle\langle i, j \rangle\rangle, \sigma} (\hat{c}^{\dagger}_{\sigma,i}, \hat{c}^{\phantom \dagger}_{\sigma,j} + {\rm h.c.}) \\
# & + U \sum_i (\hat{n}_{\uparrow,i}-\tfrac{1}{2})(\hat{n}_{\downarrow,i}-\tfrac{1}{2})
# - \mu \sum_{i,\sigma} \hat{n}_{\sigma,i},
# \end{align}
# ```
# where ``\hat{c}^\dagger_{\sigma,i} \ (\hat{c}^{\phantom \dagger}_{\sigma,i})`` creates (annihilates) a spin ``\sigma``
# electron on site ``i`` in the lattice, and ``\hat{n}_{\sigma,i} = \hat{c}^\dagger_{\sigma,i} \hat{c}^{\phantom \dagger}_{\sigma,i}``
# is the spin-``\sigma`` electron number operator for site ``i``. In the above Hamiltonian ``(t^{\prime}) \ t`` is the (next-) nearest-neighbor hopping amplitude
# and ``U > 0`` controls the strength of the on-site Hubbard repulsion.
# Lastly, we note the system is half-filled and particle-hole symmetric when the next-nearest-neighbor hopping amplitude
# and the chemical potential is zero ``(t^{\prime} = \mu = 0.0),`` in which case there is no sign problem.

# ## [Import packages](@id hubbard_square_import_packages)
# Let us begin by importing [SmoQyDQMC.jl](https://github.com/SmoQySuite/SmoQyDQMC.jl.git), and its relevant submodules.

using SmoQyDQMC
import SmoQyDQMC.LatticeUtilities as lu
import SmoQyDQMC.JDQMCFramework as dqmcf

# The [SmoQyDQMC](https://github.com/SmoQySuite/SmoQyDQMC.jl.git) package re-exports several other packages that we will make use of in this tutorial.
# The first one is [LatticeUtilities](https://github.com/SmoQySuite/LatticeUtilities.jl.git), which we will use to define the lattice geometry for our model.
# The second submodule is the [JDQMCFramework](https://github.com/SmoQySuite/JDQMCFramework.jl.git) package, which exports useful types and methods for writing
# a determinant quantum Monte Carlo (DQMC) code, taking care of things like numerical stabilization.
# We will see how to leverage both these packages in this tutorial.

# We will also  use the Standard Library packages [Random](https://docs.julialang.org/en/v1/stdlib/Random/)
# and [Printf](https://docs.julialang.org/en/v1/stdlib/Printf/) for random number generation and C-style string
# formatting, respectively.

using Random
using Printf

# ## Specify simulation parameters

# The entire main body of the simulation we will wrapped in a top-level function named `run_simulation`
# that will take as keyword arguments various model and simulation parameters that we may want to change.
# The function arguments with default values are ones that are typically left unchanged between simulations.
# The specific meaning of each argument will be discussed in more detail in later sections of the tutorial.

## Top-level function to run simulation.
function run_simulation(;
    ## KEYWORD ARGUMENTS
    sID, # Simulation ID.
    U, # Hubbard interaction.
    t′, # Next-nearest-neighbor hopping amplitude.
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
    write_bins_concurrent = true, # Whether to write binned data to file during simulation or hold it in memory.
    seed = abs(rand(Int)), # Seed for random number generator.
    filepath = "." # Filepath to where data folder will be created.
)

# ## [Initialize simulation](@id hubbard_square_initialize_simulation)
# In this first part of the script we name and initialize our simulation, creating the data folder our simulation results will be written to.
# This is done by initializing an instances of the [`SimulationInfo`](@ref) type, and then calling the [`initialize_datafolder`](@ref) function.

# Note that the `write_bins_concurrent` keyword arguments controls whether or not binned simulation measurement data
# is written to file during the simulation, or held in memory and only written to file once the simulation is complete.
# The default behavior is to `write_bins_concurrent = true` to mitigate concerns regarding the simulation using too much memory.
# However, when performing simulations of small systems that do not take very long, writing data to file too frequently can
# sometimes cause network latency issues on some clusters and HPC systems, in which case it may be advisable to set `write_bins_concurrent = false`.

    ## Construct the foldername the data will be written to.
    datafolder_prefix = @sprintf "hubbard_square_U%.2f_tp%.2f_mu%.2f_L%d_b%.2f" U t′ μ L β

    ## Initialize simulation info.
    simulation_info = SimulationInfo(
        filepath = filepath,                     
        datafolder_prefix = datafolder_prefix,
        write_bins_concurrent = write_bins_concurrent,
        sID = sID
    )

    ## Initialize the directory the data will be written to.
    initialize_datafolder(simulation_info)

# ## Initialize simulation metadata
# In this section of the code we record important metadata about the simulation, including initializing the random number
# generator that will be used throughout the simulation.
# The important metadata within the simulation will be recorded in the `metadata` dictionary.

    ## Initialize random number generator
    rng = Xoshiro(seed)

    ## Initialize additiona_info dictionary
    metadata = Dict()

    ## Record simulation parameters.
    metadata["N_therm"] = N_therm
    metadata["N_updates"] = N_updates
    metadata["N_bins"] = N_bins
    metadata["n_stab_init"] = n_stab
    metadata["dG_max"] = δG_max
    metadata["symmetric"] = symmetric
    metadata["checkerboard"] = checkerboard
    metadata["seed"] = seed
    metadata["local_acceptance_rate"] = 0.0
    metadata["reflection_acceptance_rate"] = 0.0

# In the above, `sID` stands for simulation ID, which is used to distinguish simulations that would otherwise be identical i.e. to
# distinguish simulations that use the same parameters and are only different in the random seed used to initialize the simulation.
# A valid `sID` is any positive integer greater than zero, and is used when naming the data folder the simulation results will be written to.
# Specifically, the actual data folder created above will be `"$(filepath)/$(datafolder_prefix)-$(sID)"`.
# Note that if you set `sID = 0`, then it will instead be assigned smallest previously unused integer value. For instance, suppose the directory
# `"$(filepath)/$(datafolder_prefix)-1"` already exits. Then if you pass `sID = 0` to [`SimulationInfo`](@ref), then the simulation ID
# `sID = 2` will be used instead, and a directory `"$(filepath)/$(datafolder_prefix)-2"` will be created.

# Another useful resource in the documentation is the [Simulation Output Overview](@ref) page, which describes the output written to the
# data folder generated during a [SmoQyDQMC](https://github.com/SmoQySuite/SmoQyDQMC.jl.git) simulation.

# ## Initialize model

# The next step is define the model we wish to simulate.
# In this example the relevant model parameters are the Hubbard interaction strength ``U`` (`U`), chemical potential ``\mu`` (`μ`),
# next-nearest-neighbor hopping amplitude ``t^\prime`` (`t′`) and lattice size ``L`` (`L`).

# First we define the lattice geometry for our model, relying on the
# [LatticeUtilities](https://github.com/SmoQySuite/LatticeUtilities.jl.git) package to do so.
# We define a the unit cell and size of our finite lattice using the [`UnitCell`](https://smoqysuite.github.io/LatticeUtilities.jl/stable/api/#LatticeUtilities.UnitCell)
# and [`Lattice`](https://smoqysuite.github.io/LatticeUtilities.jl/stable/api/#LatticeUtilities.Lattice) types, respectively.
# Lastly, we define various instances of the [`Bond`](https://smoqysuite.github.io/LatticeUtilities.jl/stable/api/#LatticeUtilities.Bond) type to represent the
# the nearest-neighbor and next-nearest-neighbor bonds.
# All of this information regarding the lattice geometry is then stored in an instance of the [`ModelGeometry`](@ref) type.
# Further documentation, with usage examples, for [LatticeUtilities](https://github.com/SmoQySuite/LatticeUtilities.jl.git) package
# can be found [here](https://smoqysuite.github.io/LatticeUtilities.jl/stable/).

    ## Define unit cell.
    unit_cell = lu.UnitCell(
        lattice_vecs = [[1.0, 0.0],
                        [0.0, 1.0]],
        basis_vecs = [[0.0, 0.0]]
    )

    ## Define finite lattice with periodic boundary conditions.
    lattice = lu.Lattice(
        L = [L, L],
        periodic = [true, true]
    )

    ## Initialize model geometry.
    model_geometry = ModelGeometry(
        unit_cell, lattice
    )

    ## Define the nearest-neighbor bond in +x direction.
    bond_px = lu.Bond(
        orbitals = (1,1),
        displacement = [1, 0]
    )

    ## Add this bond definition to the model, by adding it the model_geometry.
    bond_px_id = add_bond!(model_geometry, bond_px)

    ## Define the nearest-neighbor bond in +y direction.
    bond_py = lu.Bond(
        orbitals = (1,1),
        displacement = [0, 1]
    )

    ## Add this bond definition to the model, by adding it the model_geometry.
    bond_py_id = add_bond!(model_geometry, bond_py)

    ## Define the next-nearest-neighbor bond in +x+y direction.
    bond_pxpy = lu.Bond(
        orbitals = (1,1),
        displacement = [1, 1]
    )

    ## Define the nearest-neighbor bond in -x direction.
    ## Will be used to make measurements later in this tutorial.
    bond_nx = lu.Bond(
        orbitals = (1,1),
        displacement = [-1, 0]
    )

    ## Add this bond definition to the model, by adding it the model_geometry.
    bond_nx_id = add_bond!(model_geometry, bond_nx)

    ## Define the nearest-neighbor bond in -y direction.
    ## Will be used to make measurements later in this tutorial.
    bond_ny = lu.Bond(
        orbitals = (1,1),
        displacement = [0, -1]
    )

    ## Add this bond definition to the model, by adding it the model_geometry.
    bond_ny_id = add_bond!(model_geometry, bond_ny)

    ## Define the next-nearest-neighbor bond in +x+y direction.
    bond_pxpy = lu.Bond(
        orbitals = (1,1),
        displacement = [1, 1]
    )

    ## Add this bond definition to the model, by adding it the model_geometry.
    bond_pxpy_id = add_bond!(model_geometry, bond_pxpy)

    ## Define the next-nearest-neighbor bond in +x-y direction.
    bond_pxny = lu.Bond(
        orbitals = (1,1),
        displacement = [1, -1]
    )

    ## Add this bond definition to the model, by adding it the model_geometry.
    bond_pxny_id = add_bond!(model_geometry, bond_pxny)

# Next we specify the non-interacting tight-binding term in our Hamiltonian with the [`TightBindingModel`](@ref) type.

    ## Set neartest-neighbor hopping amplitude to unity,
    ## setting the energy scale in the model.
    t = 1.0

    ## Define the non-interacting tight-binding model.
    tight_binding_model = TightBindingModel(
        model_geometry = model_geometry,
        t_bonds = [bond_px, bond_py, bond_pxpy, bond_pxny], # defines hopping
        t_mean  = [t, t, t′, t′], # defines corresponding mean hopping amplitude
        t_std   = [0., 0., 0., 0.], # defines corresponding standard deviation in hopping amplitude
        ϵ_mean  = [0.], # set mean on-site energy for each orbital in unit cell
        ϵ_std   = [0.], # set standard deviation of on-site energy or each orbital in unit cell
        μ       = μ # set chemical potential
    )

# Finally, we define the Hubbard interaction with the [`HubbardModel`](@ref) type.
# Here the boolean `ph_sym_form` keyword argument determines whether the particle-hole symemtric form (`ph_sym_form = true`) for the Hubbard
# interaction ``U(\hat{n}_{\uparrow,i}-\tfrac{1}{2})(\hat{n}_{\downarrow,i}-\tfrac{1}{2})`` is used, or the form
# ``U\hat{n}_{\uparrow,i}\hat{n}_{\downarrow,i}`` is used (`ph_sym_form = false`) instead.

    ## Define the Hubbard interaction in the model.
    hubbard_model = HubbardModel(
        ph_sym_form = true, # if particle-hole symmetric form for Hubbard interaction is used.
        U_orbital   = [1], # orbitals in unit cell with Hubbard interaction.
        U_mean      = [U], # mean Hubbard interaction strength for corresponding orbital species in unit cell.
        U_std       = [0.], # standard deviation of Hubbard interaction strength for corresponding orbital species in unit cell.
    )

# Note that most terms in our model can support random disorder.
# However, we have suppressed this behavior by setting all relevant standard deviations in model values to zero.
# If these standard deviations were not specified they would have also defaulted to zero
# We explicitly set them to zero here to simply highlight the presence of this functionality even though we are not using it.

# Lastly, the [`model_summary`](@ref) function is used to write a `model_summary.toml` file,
# completely specifying the Hamiltonian that will be simulated.

    ## Write model summary TOML file specifying Hamiltonian that will be simulated.
    model_summary(
        simulation_info = simulation_info,
        β = β, Δτ = Δτ,
        model_geometry = model_geometry,
        tight_binding_model = tight_binding_model,
        interactions = (hubbard_model,)
    )

# ## Initialize model parameters
# The next step is to initialize our model parameters given the size of our finite lattice.
# To clarify, both the [`TightBindingModel`](@ref) and [`HubbardModel`](@ref) types are agnostic to the size of the lattice being simulated,
# defining the model in a translationally invariant way. As [SmoQyDQMC.jl](https://github.com/SmoQySuite/SmoQyDQMC.jl.git) supports
# random disorder in the terms appearing in the Hamiltonian, it is necessary to initialize separate parameter values for each unit cell in the lattice.
# For instance, we need to initialize a separate number to represent the on-site energy for each orbital in our finite lattice.
# To do so we need to initialize an instance of the [`TightBindingParameters`](@ref) and [`HubbardParameters`](@ref) types.

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

# Having initialized the Hubbard Hamiltonian parameters above, we now need to decouple the interaction by applying a Hubbard-Stratonovich (HS) transformation
# to decouple the local Hubbard interaction, thereby introducing the HS fields that will be sampled during the simulation.
# The [SmoQyDQMC.jl](https://github.com/SmoQySuite/SmoQyDQMC.jl.git) package implements four types of Hubbard-Stratonovich transformations
# for decoupling the local Hubbard interaction, represented by the four types [`HubbardDensityHirschHST`](@ref), [`HubbardDensityGaussHermiteHST`](@ref),
# [`HubbardSpinHirschHST`](@ref) and [`HubbardSpinGaussHermiteHST`]. In this example we will use the [`HubbardSpinHirschHST`](@ref) type
# to decouple the local Hubabrd interactions.

    ## Apply Spin Hirsch Hubbard-Stratonovich (HS) transformation to decouple the Hubbard interaction,
    ## and initialize the corresponding HS fields that will be sampled in the DQMC simulation.
    hst_parameters = HubbardSpinHirschHST(
        β = β, Δτ = Δτ,
        hubbard_parameters = hubbard_parameters,
        rng = rng
    )

# ## [Initialize measurement](@id hubbard_square_initialize_measurements)

# Having initialized both our model and the corresponding model parameters,
# the next step is to initialize the various measurements we want to make during our DQMC simulation.
# This includes defining the various types of correlation measurements that will be made, which is primarily done
# using the [`initialize_correlation_measurements!`](@ref) function.

# Here the arguments `β` and `Δτ` correspond to the inverse temperature and imaginary-time axis discretization constant,
# which were passed as arguments to the `run_simulation` function.

    ## Initialize the container that measurements will be accumulated into.
    measurement_container = initialize_measurement_container(model_geometry, β, Δτ)

    ## Initialize the tight-binding model related measurements, like the hopping energy.
    initialize_measurements!(measurement_container, tight_binding_model)

    ## Initialize the Hubbard interaction related measurements.
    initialize_measurements!(measurement_container, hubbard_model)

    ## Initialize the single-particle electron Green's function measurement.
    initialize_correlation_measurements!(
        measurement_container = measurement_container,
        model_geometry = model_geometry,
        correlation = "greens",
        time_displaced = true,
        pairs = [(1, 1)]
    )

    ## Initialize density correlation function measurement.
    initialize_correlation_measurements!(
        measurement_container = measurement_container,
        model_geometry = model_geometry,
        correlation = "density",
        time_displaced = false,
        integrated = true,
        pairs = [(1, 1)]
    )

    ## Initialize the pair correlation function measurement.
    initialize_correlation_measurements!(
        measurement_container = measurement_container,
        model_geometry = model_geometry,
        correlation = "pair",
        time_displaced = false,
        integrated = true,
        pairs = [(1, 1)]
    )

    ## Initialize the spin-z correlation function measurement.
    initialize_correlation_measurements!(
        measurement_container = measurement_container,
        model_geometry = model_geometry,
        correlation = "spin_z",
        time_displaced = false,
        integrated = true,
        pairs = [(1, 1)]
    )

# We also want to define define what we term a composite correlation measurement to measure
# d-wave pairing tendencies in our Hubbard model. Specifically, we would like to measure the d-wave pair susceptibility
# ```math
# \chi_d(\mathbf{q}) = \frac{1}{L^2} \int_0^\beta d\tau \sum_{\mathbf{r}, \mathbf{i}} e^{-\text{i}\mathbf{q}\cdot\mathbf{r}}
# \langle \hat{\Delta}^{\phantom\dagger}_{d,\mathbf{i}+\mathbf{r}}(\tau) \hat{\Delta}^{\dagger}_{d,\mathbf{i}}(0) \rangle
# ```
# for all scattering momentum ``\mathbf{q}``, where
# ```math
# \hat{\Delta}^{\dagger}_{d,\mathbf{i}}(\tau) = \frac{1}{2}\left[
# (
#    \hat{c}^\dagger_{\uparrow,\mathbf{i}+\mathbf{x}} + \hat{c}^\dagger_{\uparrow,\mathbf{i}-\mathbf{x}}
#  - \hat{c}^\dagger_{\uparrow,\mathbf{i}+\mathbf{y}} - \hat{c}^\dagger_{\uparrow,\mathbf{i}-\mathbf{y}}
# )
# \hat{c}^\dagger_{\downarrow,\mathbf{i}}
# \right]
# ```
# is the d-wave pair creation operator. We do this using the [`initialize_composite_correlation_measurement!`](@ref) function.

    ## Initialize the d-wave pair susceptibility measurement.
    initialize_composite_correlation_measurement!(
        measurement_container = measurement_container,
        model_geometry = model_geometry,
        name = "d-wave",
        correlation = "pair",
        ids = [bond_px_id, bond_nx_id, bond_py_id, bond_ny_id],
        coefficients = [0.5, 0.5, -0.5, -0.5],
        time_displaced = false,
        integrated = true
    )

# ## [Setup DQMC simulation](@id hubbard_square_setup_dqmc)
# This section of the code sets up the DQMC simulation by allocating the initializing the relevant types and arrays we will need in the simulation.

# This portion of code is perhaps the most opaque and difficult to understand, and quite a bit of exposition is included describing it.
# That said, you do not need to fully comprehend everything that goes on in this section as most of it is fairly boilerplate,
# and will not need to be changed much once written.
# This is true even if you want to modify this script to perform a DQMC simulation for a different Hamiltonian.

# To start, two instances of the [`FermionPathIntegral`](@ref) type are allocated, one for each electron spin species.
# Recall that after performing a HS transformation to decouple the Hubbard interaction, the resulting
# Hamiltonian is quadratic in fermion creation and annihilation operators, but fluctuates in imaginary-time as a result of introducing the HS fields.
# Therefore, this Hamiltonian may be expressed as
# ```math
# \hat{H}_l = \sum_\sigma \hat{\mathbf{c}}_\sigma^\dagger \left[ H_{\sigma,l} \right] \hat{\mathbf{c}}_\sigma
# = \sum_\sigma \hat{\mathbf{c}}_\sigma^\dagger \left[ K_{\sigma,l} + V_{\sigma,l} \right] \hat{\mathbf{c}}_\sigma,
# ```
# at imaginary-time ``\tau = \Delta\tau \cdot l``.
# where ``\hat{\mathbf{c}}_\sigma \ (\hat{\mathbf{c}}_\sigma^\dagger)`` is a column (row) vector of spin-``\sigma`` electron annihilation (creation) operators for each orbital in the lattice.
# Here ``H_{\sigma,l}`` is the spin-``\sigma`` Hamiltonian matrix for imaginary-time ``\tau``, which can be expressed as the sum of the
# electron kinetic and potential energy matrices ``K_{\sigma,l}`` and ``V_{\sigma,l}``, respectively.
# The purpose of the [`FermionPathIntegral`](@ref) type is to contain the minimal information required to reconstruct each ``K_{\sigma,l}`` and ``V_{\sigma,l}`` matrix.
# Each instance of the [`FermionPathIntegral`](@ref) type is first allocated and initialized to just reflect the non-interacting component of the Hamiltonian.
# Then the two subsequent `initialize!` calls modify the [`FermionPathIntegral`](@ref) type to reflect the contributions from the Hubbard interaction and initial HS field configuration.

# Note that if the keyword arguments `forced_complex_potential` and `forced_complex_kinetic` are set to `true` then the ``V_{\sigma,l}``
# and ``K_{\sigma,l}`` matrices are forced to be complex matrices, respectively. If instead set to false then their type is inferred from
# the instance of `tight_binding_parameters` passed during initialization.
# This is important, as the Hubbard-Stratonovich transform represented by the
# [`HubbardSpinHirschHST`](@ref) type used in this example is real if ``U \ge 0`` and complex if ``U < 0``. Therefore, we need to set
# `forced_complex_potential = (U < 0)` to account for this fact if we want this example to work for both repulsive and attractive Hubbard interactions.
# The opposite would be true if we had instead used [`HubbardDensityHirschHST`](@ref) decoupling scheme instead.

    ## Allocate FermionPathIntegral type for spin-up electrons.
    fermion_path_integral_up = FermionPathIntegral(
        tight_binding_parameters = tight_binding_parameters, β = β, Δτ = Δτ,
        forced_complex_potential = (U < 0),
        forced_complex_kinetic = false
    )

    ## Allocate FermionPathIntegral type for spin-down electrons.
    fermion_path_integral_dn = FermionPathIntegral(
        tight_binding_parameters = tight_binding_parameters, β = β, Δτ = Δτ,
        forced_complex_potential = (U < 0),
        forced_complex_kinetic = false
    )

    ## Initialize FermionPathIntegral type for both the spin-up and spin-down electrons to account for Hubbard interaction.
    initialize!(fermion_path_integral_up, fermion_path_integral_dn, hubbard_parameters)

    ## Initialize FermionPathIntegral type for both the spin-up and spin-down electrons to account for the current
    ## Hubbard-Stratonovich field configuration.
    initialize!(fermion_path_integral_up, fermion_path_integral_dn, hst_parameters)

# Next, the [`initialize_propagators`](@ref) function allocates and initializes the ``B_{\sigma,l}`` propagator matrices
# to reflect the current state of the ``K_{\sigma,l}`` and ``V_{\sigma,l}`` matrices as represented by the [`FermionPathIntegral`](@ref) type.
# If `symmetric = true`, then the propagator matrices take the form
# ```math
# B_{\sigma,l} = \left[ e^{-\Delta\tau K_{\sigma,l}/2} \right]^\dagger \cdot e^{-\Delta\tau V_{\sigma,l}} \cdot e^{-\Delta\tau K_{\sigma,l}/2},
# ```
# whereas if `symmetric = false` then 
# ```math
# B_{\sigma,l} = e^{-\Delta\tau V_{\sigma,l}} \cdot e^{-\Delta\tau K_{\sigma,l}}.
# ```
# If `checkerboard = true`, then the exponentiated kinetic energy matrices ``e^{-\Delta\tau K_{\sigma,l}} \ \left( \text{ or } e^{-\Delta\tau K_{\sigma,l}/2} \right)``
# are represented using the sparse checkerboard approximation, otherwise they are computed exactly.

    ## Initialize imaginary-time propagators for all imaginary-time slices for spin-up and spin-down electrons.
    Bup = initialize_propagators(fermion_path_integral_up, symmetric=symmetric, checkerboard=checkerboard)
    Bdn = initialize_propagators(fermion_path_integral_dn, symmetric=symmetric, checkerboard=checkerboard)

# Next, four instances of the [`FermionGreensCalculator`](https://smoqysuite.github.io/JDQMCFramework.jl/stable/api/#JDQMCFramework.FermionGreensCalculator)
# type are initialized, which are used to take care of numerical stabilization behind the scenes in the DQMC simulation.
# Here `n_stab` is the period in imaginary-time with which numerical stabilization is performed, and is typically on the order of ``n_{\rm stab} \sim 10.``

    ## Initialize FermionGreensCalculator type for spin-up and spin-down electrons.
    fermion_greens_calculator_up = dqmcf.FermionGreensCalculator(Bup, β, Δτ, n_stab)
    fermion_greens_calculator_dn = dqmcf.FermionGreensCalculator(Bdn, β, Δτ, n_stab)

    ## Initialize alternate FermionGreensCalculator type for performing reflection updates.
    fermion_greens_calculator_up_alt = dqmcf.FermionGreensCalculator(fermion_greens_calculator_up)
    fermion_greens_calculator_dn_alt = dqmcf.FermionGreensCalculator(fermion_greens_calculator_dn)

# Now we allocate and initialize the equal-time Green's function matrix ``G_\sigma(0,0)`` for both spin species (`Gup` and `Gdn`).
# The initialization process also returns ``\log | \det G_\sigma(0,0) |`` (`logdetGup` and `logdetGdn`) and ``{\rm sgn} \det G_\sigma(0,0)`` (`sgndetGup` and `sgndetGdn`).

    ## Allcoate matrices for spin-up and spin-down electron Green's function matrices.
    Gup = zeros(eltype(Bup[1]), size(Bup[1]))
    Gdn = zeros(eltype(Bdn[1]), size(Bdn[1]))

    ## Initialize the spin-up and spin-down electron Green's function matrices, also
    ## calculating their respective determinants as the same time.
    logdetGup, sgndetGup = dqmcf.calculate_equaltime_greens!(Gup, fermion_greens_calculator_up)
    logdetGdn, sgndetGdn = dqmcf.calculate_equaltime_greens!(Gdn, fermion_greens_calculator_dn)

# We also need to allocate matrices to represent the equal-time and time-displaced Green's function matrices ``G_\sigma(\tau,\tau)`` (`Gup_ττ` and `Gdn_ττ`),
# ``G_\sigma(\tau, 0)`` (`Gup_τ0` and `Gdn_τ0`), and ``G_\sigma(0,\tau)`` (`Gup_0τ` and `Gdn_0τ`) for ``\tau \ne 0``.
# All of these various Green's function matrices are required if we want to make time-displaced correlation function measurements.

    ## Allocate matrices for various time-displaced Green's function matrices.
    Gup_ττ = similar(Gup) # Gup(τ,τ)
    Gup_τ0 = similar(Gup) # Gup(τ,0)
    Gup_0τ = similar(Gup) # Gup(0,τ)
    Gdn_ττ = similar(Gdn) # Gdn(τ,τ)
    Gdn_τ0 = similar(Gdn) # Gdn(τ,0)
    Gdn_0τ = similar(Gdn) # Gdn(0,τ)

# Lastly, we initialize two diagnostic parameters `δG` and `δθ` to asses numerical stability during the simulation.
# The `δG`  parameter is particularly important to keep track of during the simulation, and is defined as
# ```math
# \delta G = \max \left( | G^{\rm stab.}_\sigma(0,0) - G^{\rm naive}_\sigma(0,0) | \right),
# ```
# i.e. the maximum magnitude numerical error corrected by numerical stabilization for any Green's function matrix element.
# The ``\delta \theta`` diagnostic parameter reports the error in the phase of the fermion determinant as it can in general be complex,
# but this is less important to keep track of in most situations.

    ## Initialize diagnostic parameters to asses numerical stability.
    δG = zero(logdetGup)
    δθ = zero(logdetGup)

# ## Thermalize system
# The next section of code performs updates to thermalize the system prior to beginning measurements.
# The structure of this function should be fairly intuitive, mainly consisting of a loop inside of which we perform updates to the Hubbard-Stratonovich fields.
# In this example we perform two types of updates.

# We use the [`local_updates!`](@ref) function to attempt a local update to every HS field.
# Additionally, if the [`local_updates!`](@ref) argument `update_stabilization_frequency = true`, then the `δG_max` parameter acts a maximum threshold for `δG`.
# If `δG` exceeds `δG_max`, then `n_stab` is decrement by one (the frequency of numerical stabilization is increased) and `δG` is reset to zero.
# In the case `update_stabilization_frequency = false`,
# then `δG_max` doesn't do anything and `n_stab` remains unchanged during the simulation,
# with `δG` is simply reporting the maximum observed numerical error during the simulation.

# We also use the [`reflection_update!`](@ref) function to perform a reflection update on a randomly selected sites in the lattice
# whereby we flip the HS fields on every imaginary-time slice for that site
# i.e. propose an update such that ``s_{i,l} \rightarrow -s_{i,l}`` for all ``l \in [1, L_\tau]``.
# This type of reflection update [has been shown](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.44.10502)
# to be important for mitigating ergodicity issues in large Hubbard ``U`` simulations.

# In the block of code below, `N_therm` is the number of local sweeps and reflection updates that are performed to thermalize the system.

    ## Iterate over number of thermalization updates to perform.
    for n in 1:N_therm

        ## Perform reflection update for HS fields with randomly chosen site.
        (accepted, logdetGup, sgndetGup, logdetGdn, sgndetGdn) = reflection_update!(
            Gup, logdetGup, sgndetGup, Gdn, logdetGdn, sgndetGdn,
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
        metadata["reflection_acceptance_rate"] += accepted

        ## Perform sweep all imaginary-time slice and orbitals, attempting an update to every HS field.
        (acceptance_rate, logdetGup, sgndetGup, logdetGdn, sgndetGdn, δG, δθ) = local_updates!(
            Gup, logdetGup, sgndetGup, Gdn, logdetGdn, sgndetGdn,
            hst_parameters,
            fermion_path_integral_up = fermion_path_integral_up,
            fermion_path_integral_dn = fermion_path_integral_dn,
            fermion_greens_calculator_up = fermion_greens_calculator_up,
            fermion_greens_calculator_dn = fermion_greens_calculator_dn,
            Bup = Bup, Bdn = Bdn, δG_max = δG_max, δG = δG, δθ = δθ, rng = rng,
            update_stabilization_frequency = false
        )

        ## Record acceptance rate for sweep.
        metadata["local_acceptance_rate"] += acceptance_rate
    end

# ## [Make measurements](@id hubbard_square_make_measurements)
# In this next section of code we continue to sample the HS field with [`local_updates!`](@ref) function, but begin making measurements as well.
# Here, `N_updates` refers to the number of times [`local_updates!`](@ref) is called,
# as well as the number of times measurements are made using the [`make_measurements!`](@ref) function.
# The parameter `N_bins` then controls the number of times bin-averaged measurements are written to binary
# [JLD2](https://github.com/JuliaIO/JLD2.jl.git) files, subject to the constraint that `(N_updates % N_bins) == 0`.
# Therefore, the number of measurements that are averaged over per bin is given by `bin_size = N_updates ÷ N_bins`.
# The bin-averaged measurements are written to file once `bin_size` measurements are accumulated using the [`write_measurements!`](@ref) function.

    ## Reset diagonostic parameters used to monitor numerical stability to zero.
    δG = zero(logdetGup)
    δθ = zero(logdetGup)

    ## Calculate the bin size.
    bin_size = N_updates ÷ N_bins

    ## Iterate over updates and measurements.
    for update in 1:N_updates

        ## Perform reflection update for HS fields with randomly chosen site.
        (accepted, logdetGup, sgndetGup, logdetGdn, sgndetGdn) = reflection_update!(
            Gup, logdetGup, sgndetGup, Gdn, logdetGdn, sgndetGdn,
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
        metadata["reflection_acceptance_rate"] += accepted

        ## Perform sweep all imaginary-time slice and orbitals, attempting an update to every HS field.
        (acceptance_rate, logdetGup, sgndetGup, logdetGdn, sgndetGdn, δG, δθ) = local_updates!(
            Gup, logdetGup, sgndetGup, Gdn, logdetGdn, sgndetGdn,
            hst_parameters,
            fermion_path_integral_up = fermion_path_integral_up,
            fermion_path_integral_dn = fermion_path_integral_dn,
            fermion_greens_calculator_up = fermion_greens_calculator_up,
            fermion_greens_calculator_dn = fermion_greens_calculator_dn,
            Bup = Bup, Bdn = Bdn, δG_max = δG_max, δG = δG, δθ = δθ, rng = rng,
            update_stabilization_frequency = false
        )

        ## Record acceptance rate.
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
            model_geometry = model_geometry, tight_binding_parameters = tight_binding_parameters,
            coupling_parameters = (hubbard_parameters, hst_parameters)
        )

        ## Write the bin-averaged measurements to file if update ÷ bin_size == 0.
        write_measurements!(
            measurement_container = measurement_container,
            simulation_info = simulation_info,
            model_geometry = model_geometry,
            measurement = update,
            bin_size = bin_size,
            Δτ = Δτ
        )
    end

# ## Merge binned data
# At this point the simulation is essentially complete, with all updates and measurements having been performed.
# However, the binned measurement data resides in many seperate HDF5 files currently.
# Here we will merge these seperate HDF5 files into a single file containing all the binned data
# using the [`merge_bins`](@ref) function.

    ## Merge binned data into a single HDF5 file.
    merge_bins(simulation_info)

# ## Record simulation metadata
# At this point we are done sampling and taking measurements.
# Next, we want to calculate the final acceptance rate for the Monte Carlo updates,
# as well as write the simulation metadata to file, including the contents of the `metadata` dictionary.
# This is done using the [`save_simulation_info`](@ref) function.

    ## Normalize acceptance rate.
    metadata["local_acceptance_rate"] /=  (N_therm + N_updates)
    metadata["reflection_acceptance_rate"] /= (N_therm + N_updates)

    ## Record final stabilization period used at the end of the simulation.
    metadata["n_stab_final"] = fermion_greens_calculator_up.n_stab

    ## Record largest numerical error.
    metadata["dG"] = δG

    ## Write simulation summary TOML file.
    save_simulation_info(simulation_info, metadata)

# ## [Post-process results](@id hubbard_square_process_results)
# In this final section of code we post-process the binned data.
# This includes calculating the final estimates for the mean and error of all measured observables,
# which will be written to an HDF5 file using the [`process_measurements`](@ref) function.
# Inside this function the binned data gets further rebinned into `n_bins`,
# where `n_bins` is any positive integer satisfying the constraints `(N_bins ≥ n_bin)` and `(N_bins % n_bins == 0)`.
# Note that the [`process_measurements`](@ref) function has many additional keyword arguments that can be used to control the output.
# For instance, in this example in addition to writing the statistics to an HDF5 file, we also export the statistics to CSV files
# by setting `export_to_csv = true`, with additional keyword arguments controlling the formatting of the CSV files.
# Again, for more information on how to interpret the output refer the [Simulation Output Overview](@ref) page.

    ## Process the simulation results, calculating final error bars for all measurements.
    ## writing final statisitics to CSV files.
    process_measurements(
        datafolder = simulation_info.datafolder,
        n_bins = N_bins,
        export_to_csv = true,
        scientific_notation = false,
        decimals = 7,
        delimiter = " "
    )

# A common measurement that needs to be computed at the end of a DQMC simulation is something called the correlation
# ratio with respect to the ordering wave-vector for a specified type of structure factor measured during the simulation.
# In the case of the square Hubbard model, we are interested in measureing the correlation ratio
# ```math
# R_z(\mathbf{Q}_\text{afm}) = 1 - \frac{1}{4} \sum_{\delta\mathbf{q}} \frac{S_z(\mathbf{Q}_\text{afm} + \delta\mathbf{q})}{S_z(\mathbf{Q}_\text{afm})}
# ```
# with respect to the equal-time antiferromagnetic (AFM) structure factor ``S_z(\mathbf{Q}_\text{afm})``, where ``S_z(\mathbf{q})`` is the spin-``z``
# equal-time structure factor and ``\mathbf{Q}_\text{afm} = (\pi/a, \pi/a)`` is the AFM ordering wave-vector.
# The sum over ``\delta\mathbf{q}`` runs over the four wave-vectors that neigboring ``\mathbf{Q}_\text{afm}.``

# Here we use the [`compute_correlation_ratio`](@ref) function to compute to compute this correlation ratio.
# Note that the ``\mathbf{Q}_\text{afm}`` is specified using the `q_point` keyword argument, and the four neighboring wave-vectors
# ``\mathbf{Q}_\text{afm} + \delta\mathbf{q}`` are specified using the `q_neighbors` keyword argument.
# These wave-vectors are specified using the convention described [here](@ref vector_reporting_conventions) in the [Simulation Output Overview](@ref) page.

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

# Next, we record the measurement in the `metadata` dictionary, and then write a new version of the simulation summary TOML file that
# contains this new information using the [`save_simulation_info`](@ref) function.

    ## Record the AFM correlation ratio mean and standard deviation.
    metadata["Rafm_mean_real"] = real(Rafm)
    metadata["Rafm_mean_imag"] = imag(Rafm)
    metadata["Rafm_std"]       = ΔRafm

    ## Write simulation summary TOML file.
    save_simulation_info(simulation_info, metadata)

# Note that as long as the binned data persists the [`process_measurements`](@ref) and [`compute_correlation_ratio`](@ref)
# functions can be rerun to recompute the final statistics for the measurements without needing to rerun the simulation.

    return nothing
end # end of run_simulation function

# ## Execute script

# DQMC simulations are typically run from the command line as jobs on a computing cluster.
# With this in mind, the following block of code only executes if the Julia script is run from the command line,
# also reading in additional command line arguments.

## Only excute if the script is run directly from the command line.
if abspath(PROGRAM_FILE) == @__FILE__

    ## Run the simulation, reading in command line arguments.
    run_simulation(;
        sID       = parse(Int,     ARGS[1]), # Simulation ID.
        U         = parse(Float64, ARGS[2]), # Hubbard interaction strength.
        t′        = parse(Float64, ARGS[3]), # Next-nearest-neighbor hopping amplitude.
        μ         = parse(Float64, ARGS[4]), # Chemical potential.
        L         = parse(Int,     ARGS[5]), # Lattice size.
        β         = parse(Float64, ARGS[6]), # Inverse temperature.
        N_therm   = parse(Int,     ARGS[7]), # Number of thermalization sweeps.
        N_updates = parse(Int,     ARGS[8]), # Number of measurement sweeps.
        N_bins    = parse(Int,     ARGS[9])  # Number times binned data is written to file.
    )
end

# For instance, the command
# ```
# > julia hubbard_square.jl 1 5.0 -0.25 -2.0 4 4.0 2500 10000 100
# ```
# runs a DQMC simulation of a ``N = 4 \times 4`` doped square Hubbard model at inverse temperature ``\beta = 4.0``
# with interaction strength ``U = 5.0,`` chemical potential ``\mu = -2.0`` and next-nearest-neighbor hopping amplitude ``t^\prime = -0.25``.
# In the DQMC simulation, ``2500`` sweeps through the lattice are be performed to thermalize the system.
# Then an additional ``10,000`` sweeps are performed, after each of which measurements are made.
# During the simulation, bin-averaged measurements are written to file ``100`` times,
# with each bin of data containing the average of ``10,000/100 = 100`` sequential measurements.