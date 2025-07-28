```@meta
EditURL = "../../../examples/hubbard_threeband.jl"
```

# Three-Band Hubbard Model
Download this example as a [Julia script](../assets/scripts/examples/hubbard_threeband.jl).

In this example we simulate an effective two-dimensional 3-band Hubbard model
meant to represent a copper-oxide plane in the superconducting cuprates, with a Hamiltonian
written in hole language given by
```math
\begin{align*}
\hat{H}= & \sum_{\sigma,\langle i,j,\alpha\rangle}t_{pd}^{i,j,\alpha}(\hat{d}_{\sigma,i}^{\dagger}\hat{p}_{\sigma,j,\alpha}^{\phantom{\dagger}}+{\rm h.c.})
           + \sum_{\sigma,\langle i,\alpha,j,\alpha'\rangle}t_{pp}^{i,\alpha',j,\alpha}(\hat{p}_{\sigma,i,\alpha'}^{\dagger}\hat{p}_{\sigma,j,\alpha}^{\phantom{\dagger}}+{\rm h.c.})\\
         & +(\epsilon_{d}-\mu)\sum_{\sigma,i}\hat{n}_{\uparrow,i}^{d}+(\epsilon_{p}-\mu)\sum_{\sigma,j}\hat{n}_{\sigma,j,\alpha}^{p}\\
         & +U_{d}\sum_{i}\hat{n}_{\uparrow,i}^{d}\hat{n}_{\downarrow,i}^{d}+U_{p}\sum_{j,\alpha}\hat{n}_{\uparrow,j,\alpha}^{p}\hat{n}_{\downarrow,j,\alpha}^{p}.
\end{align*}
```
The operator ``\hat{d}^{\dagger}_{\sigma, i} \ (\hat{d}^{\phantom \dagger}_{\sigma, i})`` creates (annihilates) a spin-``\sigma`` hole on a Cu-``3d_{x^2-y^2}``
orbital in unit ``i`` in the lattice.
The ``\hat{p}^{\dagger}_{\sigma,i,\alpha} \ (\hat{p}^{\phantom \dagger}_{\sigma,i,\alpha})`` operator creates (annihilates) a spin-``\sigma`` hole on a
O-``2p_\alpha`` orbital in unit cell ``i``, where ``\alpha = x \ {\rm or} \ y.`` The corresponding spin-``\sigma`` hole number operators for the
Cu-``3d_{x^2-y^2}`` and O-``2p_\alpha`` orbitals in unit cell ``i`` are ``\hat{n}^d_{\sigma,i}`` and ``\hat{n}^p_{\sigma,i,\alpha}``.
The hopping integrals between the Cu-``3d_{x^2-y^2}`` orbitals and nearest-neighbor O-``2p_\alpha`` are parameterized as
``t_{pd}^{i,j,\alpha} = P_{pd}^{i,j,\alpha} t_{pd}`` where `` P_{pd}^{i,j,\alpha} = \pm 1`` is a overall phase factor.
Similarly, the hopping integral between nearest-neighbor O-``2p_x`` and O-``2p_y`` orbitals is parameterized as
``t_{pp}^{i,\alpha',j,\alpha} = P_{pp}^{i,\alpha',j,\alpha} t_{pp}``, where again ``P_{pp}^{i,\alpha',j,\alpha} t_{pp} = \pm 1`` is an overall phase factor.
Refer to Fig. 1 in [`PhysRevB.103.144514`](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.103.144514) to see a figure detailing these phase factor conventions.
The on-site energies ``\epsilon_d`` and ``\epsilon_p`` are for the Cu-``3d_{x^2-y^2}`` and O-``2p_\alpha`` orbitals respectively,
and ``\mu`` is the global chemical potential. Finally, ``U_d`` and ``U_p`` are the on-site Hubbard interactions for the
Cu-``3d_{x^2-y^2}`` and O-``2p_\alpha`` orbitals respectively.

A short test simulation using the script associated with this example can be run as
```
> julia hubbard_threeband.jl 0 8.5 4.1 1.13 0.49 0.0 3.24 0.0 4.0 8 2 2000 10000 50
```
In this example we are simulating the three-band Hubbard model on a ``8 \times 2`` unit cell finite lattice at inverse temperature ``\beta = 4.0``.
The on-site Hubbard interaction on the Cu-``3d_{x^2-y^2}`` and O-``2p_\alpha`` is ``U_d = 8.5`` and ``U_p = 4.1`` respectively.
The nearest-neighbor hopping integral amplitude between the Cu-``3d_{x^2-y^2}`` and O-``2p_\alpha`` orbitals is ``t_{pd} = 1.13``,
while it is ``t_{pp} = 0.49`` between the nearest-neighbor O-``2p_x`` and O-``2p_y`` orbitals.
The on-site energy for the Cu-``3d_{x^2-y^2}`` and O-``2p_\alpha`` orbitals ``\epsilon_d = 0.0`` and ``\epsilon_p = 3.25``.
Lastly, the global chemical potential is set to ``\mu = 0.0``.
In this simulation `N_burnin = 2000` sweeps through the lattice updating the Hubbard-Stratonovich fields are performed to thermalize the system,
followed by `N_udpates = 10000` sweeps, after each of which measurements are made. Bin averaged measurements are written to file
`N_bins = 50` times during the simulation.

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

# Define top-level function for running DQMC simulation
function run_hubbard_threeband_simulation(sID, Ud, Up, tpd, tpp, ϵd, ϵp, μ, β, Lx, Ly, N_burnin, N_updates, N_bins; filepath = ".")

    # Construct the foldername the data will be written to.
    datafolder_prefix = @sprintf "hubbard_threeband_Ud%.2f_Up%.2f_tpd%.2f_tpp%.2f_ed%.2f_ep%.2f_mu%.2f_b%.2f_Lx%d_Ly%d" Ud Up tpd tpp ϵd ϵp μ β Lx Ly

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

    # Initialize a dictionary to store additional information about the simulation.
    additional_info = Dict(
        "dG_max" => δG_max,
        "N_burnin" => N_burnin,
        "N_updates" => N_updates,
        "N_bins" => N_bins,
        "bin_size" => bin_size,
        "local_acceptance_rate" => 0.0,
        "n_stab_init" => n_stab,
        "symmetric" => symmetric,
        "checkerboard" => checkerboard,
        "seed" => seed,
    )

    #######################
    ### DEFINE THE MODEL ##
    #######################

    # Initialize an instance of the type UnitCell.
    unit_cell = lu.UnitCell(
        lattice_vecs = [[1.0, 0.0], [0.0, 1.0]],
        basis_vecs   = [[0.0, 0.0], # Orbital ID = 1 <==> Cu-3d
                        [0.5, 0.0], # Orbital ID = 2 <==> O-2px
                        [0.0, 0.5]] # Orbital ID = 3 <==> O-2py
    )

    # Initialize variables to map orbitals to orbital ID.
    (Cu_3d, O_2px, O_2py) = (1, 2, 3)

    # Initialize an instance of the type Lattice.
    lattice = lu.Lattice(
        L = [Lx, Ly],
        periodic = [true, true]
    )

    # Initialize an instance of the ModelGeometry type.
    model_geometry = ModelGeometry(unit_cell, lattice)

    # Define bond going from Cu-3d to O-2px in +x direction.
    bond_3d_2px_px = lu.Bond(orbitals = (Cu_3d, O_2px), displacement = [0,0])
    bond_3d_2px_px_id = add_bond!(model_geometry, bond_3d_2px_px)

    # Define bond going from Cu-3d to O-2py in +y direction.
    bond_3d_2py_py = lu.Bond(orbitals = (Cu_3d, O_2py), displacement = [0,0])
    bond_3d_2py_py_id = add_bond!(model_geometry, bond_3d_2py_py)

    # Define bond going from Cu-3d to O-2px in -x direction.
    bond_2px_3d_nx = lu.Bond(orbitals = (Cu_3d, O_2px), displacement = [-1,0])
    bond_2px_3d_nx_id = add_bond!(model_geometry, bond_2px_3d_nx)

    # Define bond going from Cu-3d to O-2py in -y direction.
    bond_2py_3d_ny = lu.Bond(orbitals = (Cu_3d, O_2py), displacement = [0,-1])
    bond_2py_3d_ny_id = add_bond!(model_geometry, bond_2py_3d_ny)

    # Define bond going from O-2px to O-2py in the (-x+y)/√2 direction.
    bond_2px_2py_nxpy = lu.Bond(orbitals = (O_2px, O_2py), displacement = [0,0])
    bond_2px_2py_nxpy_id = add_bond!(model_geometry, bond_2px_2py_nxpy)

    # Define bond going to O-2px to O-2py in the (-x-y)/√2 direction.
    bond_2px_2py_nxny = lu.Bond(orbitals = (O_2px, O_2py), displacement = [0,-1])
    bond_2px_2py_nxny_id = add_bond!(model_geometry, bond_2px_2py_nxny)

    # Define bond going from O-2px to O-2py in the (+x+y)/√2 direction.
    bond_2px_2py_pxpy = lu.Bond(orbitals = (O_2px, O_2py), displacement = [1,0])
    bond_2px_2py_pxpy_id = add_bond!(model_geometry, bond_2px_2py_pxpy)

    # Define bond going from O-2px to O-2py in the (+x-y)/√2 direction.
    bond_2px_2py_pxny = lu.Bond(orbitals = (O_2px, O_2py), displacement = [1,-1])
    bond_2px_2py_pxny_id = add_bond!(model_geometry, bond_2px_2py_pxny)

    # These nexts bonds are needed to measuring a pairing channel needed to
    # reconstruct the d-wave pair susceptibility.

    # Define bond going from Cu-3d to Cu-3d in +x direction.
    bond_3d_3d_px = lu.Bond(orbitals = (Cu_3d, Cu_3d), displacement = [1, 0])
    bond_3d_3d_px_id = add_bond!(model_geometry, bond_3d_3d_px)

    # Define bond going from Cu-3d to Cu-3d in -x direction.
    bond_3d_3d_nx = lu.Bond(orbitals = (Cu_3d, Cu_3d), displacement = [-1, 0])
    bond_3d_3d_nx_id = add_bond!(model_geometry, bond_3d_3d_nx)

    # Define bond going from Cu-3d to Cu-3d in +y direction.
    bond_3d_3d_py = lu.Bond(orbitals = (Cu_3d, Cu_3d), displacement = [0, 1])
    bond_3d_3d_py_id = add_bond!(model_geometry, bond_3d_3d_py)

    # Define bond going from Cu-3d to Cu-3d in -y direction.
    bond_3d_3d_ny = lu.Bond(orbitals = (Cu_3d, Cu_3d), displacement = [0, -1])
    bond_3d_3d_ny_id = add_bond!(model_geometry, bond_3d_3d_ny)

    # Define tight binding model
    tight_binding_model = TightBindingModel(
        model_geometry = model_geometry,
        μ = μ,
        ϵ_mean  = [ϵd, ϵp, ϵp],
        t_bonds = [bond_3d_2px_px, bond_3d_2py_py, bond_2px_3d_nx, bond_2py_3d_ny,
                   bond_2px_2py_nxpy, bond_2px_2py_nxny, bond_2px_2py_pxpy, bond_2px_2py_pxny],
        t_mean  = [tpd, -tpd, -tpd, tpd, -tpp, tpp, tpp, -tpp]
    )

    # Initialize a finite Hubbard interaction just on copper orbitals.
    if iszero(Up)
        hubbard_model = HubbardModel(
            shifted   = true, # If true, Hubbard interaction instead parameterized as U⋅nup⋅ndn
            U_orbital = [1, 2, 3],
            U_mean    = [Ud, Up, Up],
        )
    # Initialize the Hubbard interaction on copper and oxygen orbitals.
    else
        hubbard_model = HubbardModel(
            shifted   = true, # If true, Hubbard interaction instead parameterized as U⋅nup⋅ndn
            U_orbital = [1],
            U_mean    = [Ud],
        )
    end

    # Write the model summary to file.
    model_summary(
        simulation_info = simulation_info,
        β = β, Δτ = Δτ,
        model_geometry = model_geometry,
        tight_binding_model = tight_binding_model,
        interactions = (hubbard_model,)
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
        pairs = [(Cu_3d, Cu_3d), (O_2px, O_2px), (O_2py, O_2py),
                 (Cu_3d, O_2px), (Cu_3d, O_2py), (O_2px, O_2py)]
    )

    # Initialize density correlation function measurement.
    initialize_correlation_measurements!(
        measurement_container = measurement_container,
        model_geometry = model_geometry,
        correlation = "density",
        time_displaced = false,
        integrated = true,
        pairs = [(Cu_3d, Cu_3d), (O_2px, O_2px), (O_2py, O_2py),
                 (Cu_3d, O_2px), (Cu_3d, O_2py), (O_2px, O_2py)]
    )

    # Initialize the pair correlation function measurement.
    initialize_correlation_measurements!(
        measurement_container = measurement_container,
        model_geometry = model_geometry,
        correlation = "pair",
        time_displaced = false,
        integrated = true,
        pairs = [(Cu_3d, Cu_3d), (O_2px, O_2px), (O_2py, O_2py),
                 (Cu_3d, O_2px), (Cu_3d, O_2py), (O_2px, O_2py)]
    )

    # Initialize the spin-z correlation function measurement.
    initialize_correlation_measurements!(
        measurement_container = measurement_container,
        model_geometry = model_geometry,
        correlation = "spin_z",
        time_displaced = false,
        integrated = true,
        pairs = [(Cu_3d, Cu_3d), (O_2px, O_2px), (O_2py, O_2py),
                 (Cu_3d, O_2px), (Cu_3d, O_2py), (O_2px, O_2py)]
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
        pairs = [(Cu_3d, Cu_3d), (O_2px, O_2px), (O_2py, O_2py),
                 (bond_3d_3d_px_id, bond_3d_3d_px_id), (bond_3d_3d_px_id, bond_3d_3d_nx_id),
                 (bond_3d_3d_nx_id, bond_3d_3d_px_id), (bond_3d_3d_nx_id, bond_3d_3d_nx_id),
                 (bond_3d_3d_py_id, bond_3d_3d_py_id), (bond_3d_3d_py_id, bond_3d_3d_ny_id),
                 (bond_3d_3d_ny_id, bond_3d_3d_py_id), (bond_3d_3d_ny_id, bond_3d_3d_ny_id),
                 (bond_3d_3d_px_id, bond_3d_3d_py_id), (bond_3d_3d_px_id, bond_3d_3d_ny_id),
                 (bond_3d_3d_nx_id, bond_3d_3d_py_id), (bond_3d_3d_nx_id, bond_3d_3d_ny_id),
                 (bond_3d_3d_py_id, bond_3d_3d_px_id), (bond_3d_3d_py_id, bond_3d_3d_nx_id),
                 (bond_3d_3d_ny_id, bond_3d_3d_px_id), (bond_3d_3d_ny_id, bond_3d_3d_nx_id)]
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

    # Initialize variables to keep track of the largest numerical error in the
    # Green's function matrices corrected by numerical stabalization.
    δG = zero(typeof(logdetGup))
    δθ = zero(typeof(sgndetGup))

    ####################################
    ### BURNIN/THERMALIZATION UPDATES ##
    ####################################

    # Iterate over burnin/thermalization updates.
    for n in 1:N_burnin

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

    # Process the simulation results, calculating final error bars for all measurements,
    # writing final statisitics to CSV files.
    process_measurements(simulation_info.datafolder, N_bins)

    # Measure the d-wave pair suspcetibility.
    P_d, ΔP_d = composite_correlation_stat(
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
        ids = [(bond_3d_3d_px_id, bond_3d_3d_px_id), (bond_3d_3d_px_id, bond_3d_3d_nx_id),
               (bond_3d_3d_nx_id, bond_3d_3d_px_id), (bond_3d_3d_nx_id, bond_3d_3d_nx_id),
               (bond_3d_3d_py_id, bond_3d_3d_py_id), (bond_3d_3d_py_id, bond_3d_3d_ny_id),
               (bond_3d_3d_ny_id, bond_3d_3d_py_id), (bond_3d_3d_ny_id, bond_3d_3d_ny_id),
               (bond_3d_3d_px_id, bond_3d_3d_py_id), (bond_3d_3d_px_id, bond_3d_3d_ny_id),
               (bond_3d_3d_nx_id, bond_3d_3d_py_id), (bond_3d_3d_nx_id, bond_3d_3d_ny_id),
               (bond_3d_3d_py_id, bond_3d_3d_px_id), (bond_3d_3d_py_id, bond_3d_3d_nx_id),
               (bond_3d_3d_ny_id, bond_3d_3d_px_id), (bond_3d_3d_ny_id, bond_3d_3d_nx_id)],
        locs = [(0,0), (0,0),
                (0,0), (0,0),
                (0,0), (0,0),
                (0,0), (0,0),
                (0,0), (0,0),
                (0,0), (0,0),
                (0,0), (0,0),
                (0,0), (0,0)],
        num_bins = N_bins,
        f = (P_px_px, P_px_nx,
             P_nx_px, P_nx_nx,
             P_py_py, P_py_ny,
             P_ny_py, P_ny_ny,
             P_px_py, P_px_ny,
             P_nx_py, P_nx_ny,
             P_py_px, P_py_nx,
             P_ny_px, P_ny_nx) -> (P_px_px + P_px_nx +
                                   P_nx_px + P_nx_nx +
                                   P_py_py + P_py_ny +
                                   P_ny_py + P_ny_ny -
                                   P_px_py - P_px_ny -
                                   P_nx_py - P_nx_ny -
                                   P_py_px - P_py_nx -
                                   P_ny_px - P_ny_nx)/4
    )

    # Record the d-wave pair suspcetibility.
    additional_info["P_d_mean"] = real(P_d)
    additional_info["P_d_std"]  = ΔP_d

    # Write simulation summary TOML file.
    save_simulation_info(simulation_info, additional_info)

    return nothing
end


# Only excute if the script is run directly from the command line.
if abspath(PROGRAM_FILE) == @__FILE__

    # Read in the command line arguments.
    sID = parse(Int, ARGS[1]) # simulation ID
    Ud = parse(Float64, ARGS[2])
    Up = parse(Float64, ARGS[3])
    tpd = parse(Float64, ARGS[4])
    tpp = parse(Float64, ARGS[5])
    ϵd = parse(Float64, ARGS[6])
    ϵp = parse(Float64, ARGS[7])
    μ = parse(Float64, ARGS[8])
    β = parse(Float64, ARGS[9])
    Lx = parse(Int, ARGS[10])
    Ly = parse(Int, ARGS[11])
    N_burnin = parse(Int, ARGS[12])
    N_updates = parse(Int, ARGS[13])
    N_bins = parse(Int, ARGS[14])

    # Run the simulation.
    run_hubbard_threeband_simulation(sID, Ud, Up, tpd, tpp, ϵd, ϵp, μ, β, Lx, Ly, N_burnin, N_updates, N_bins)
end
````

