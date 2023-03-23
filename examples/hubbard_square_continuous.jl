using LinearAlgebra
using Random
using Printf

using SmoQyDQMC
import SmoQyDQMC.LatticeUtilities  as lu
import SmoQyDQMC.JDQMCFramework    as dqmcf
import SmoQyDQMC.JDQMCMeasurements as dqmcm
import SmoQyDQMC.MuTuner           as mt

# top level function to run simulation
function run_simulation()

    # set inverse temperature
    β = 3.0

    # system size
    L = 8

    # nearest-neighbor hopping amplitude
    t = 1.0

    # hubbard U
    U = 8.0

    # initial chemical potential
    μ = -2.0

    # continuous hs parameter
    p = 4.0

    # define simulation name
    datafolder_prefix = @sprintf "hubbard_square_U%.2f_mu%.2f_L%d_b%.2f" U μ L β

    # initialize simulation info
    simulation_info = SimulationInfo(
        filepath = ".",                     
        datafolder_prefix = datafolder_prefix,
        sID = 1
    )

    # initialize data folder
    initialize_datafolder(simulation_info)

    ##################################
    ## DEFINE SIMULATION PARAMETERS ##
    ##################################

    # initialize random seed
    seed = abs(rand(Int))

    # initialize random number generator
    rng = Xoshiro(seed)

    # discretization in imaginary time
    Δτ = 0.10

    # evaluate length of imaginary time axis
    Lτ = dqmcf.eval_length_imaginary_axis(β, Δτ)

    # whether to use the checkerboard approximation
    checkerboard = false

    # whether to use symmetric propagator defintion 
    symmetric = false

    # initial stabilization frequency
    n_stab = 10

    # max allowed error in green's function
    δG_max = 1e-5

    # langevin monte carlo timestep
    Δt = 0.10

    # number of thermalization/burnin updates
    N_burnin = 6_400

    # number of simulation updates
    N_updates = 32_000

    # number of bins/number of time 
    N_bins = 128

    # bin size
    bin_size = div(N_updates, N_bins)

    # initialize addition simulation information dictionary
    additional_info = Dict(
        "dG_max" => δG_max,
        "N_burnin" => N_burnin,
        "N_updates" => N_updates,
        "N_bins" => N_bins,
        "bin_size" => bin_size,
        "acceptance_rate" => 0.0,
        "n_stab_init" => n_stab,
        "symmetric" => symmetric,
        "checkerboard" => checkerboard,
        "seed" => seed,
    )

    ##################
    ## DEFINE MODEL ##
    ##################

    # calculate length of imaginary time axis
    Lτ = dqmcf.eval_length_imaginary_axis(β, Δτ)

    # define kagome unit cell
    unit_cell = lu.UnitCell(lattice_vecs = [[1.0, 0.0], [0.0, 1.0]],
                            basis_vecs   = [[0.0, 0.0]])

    # define size of lattice (only supports periodic b.c. for now)
    lattice = lu.Lattice(
        L = [L, L],
        periodic = [true, true] # must be true for now
    )

    # define model geometry
    model_geometry = ModelGeometry(unit_cell, lattice)

    # calculate number of orbitals in the lattice
    N = lu.nsites(unit_cell, lattice)

    # define bond in x-direction
    bond_x = lu.Bond(orbitals = (1,1), displacement = [1,0])
    bond_x_id = add_bond!(model_geometry, bond_x)

    # define bond in y-direction
    bond_y = lu.Bond(orbitals = (1,1), displacement = [0,1])
    bond_y_id = add_bond!(model_geometry, bond_y)

    # define non-interacting tight binding model
    tight_binding_model = TightBindingModel(
        model_geometry = model_geometry,
        t_bonds = [bond_x, bond_y],
        t_mean = [t, t],
        μ = μ,
        ϵ_mean = [0.]
    )

    # define hubbard model
    hubbard_model = HubbardModel(
        shifted = false, # means coupling of the form U⋅(nup-1/2)⋅(ndn-1/2)
        U_orbital = [1],
        U_mean = [U],
    )

    # write model summary to file
    model_summary(
        simulation_info = simulation_info,
        β = β, Δτ = Δτ,
        model_geometry = model_geometry,
        tight_binding_model = tight_binding_model,
        interactions = (hubbard_model,)
    )

    ####################################################
    ## INITIALIZE MODEL PARAMETERS FOR FINITE LATTICE ##
    ####################################################

    # define tight binding parameters for finite lattice based on tight binding model
    tight_binding_parameters = TightBindingParameters(
        tight_binding_model = tight_binding_model,
        model_geometry = model_geometry,
        rng = rng
    )

    # define hubbard parameters
    hubbard_parameters = HubbardParameters(
        model_geometry = model_geometry,
        hubbard_model = hubbard_model,
        rng = rng
    )

    # define ising hubbard-stratonovich transformation to decouple hubbard interaction
    hubbard_hs_parameters = HubbardContinuousHSParameters(
        β = β, Δτ = Δτ, p = p,
        hubbard_parameters = hubbard_parameters,
        rng = rng
    )

    ######################################
    ## DEFINE AND INIALIZE MEASUREMENTS ##
    ######################################

    # initialize measurement container
    measurement_container = initialize_measurement_container(model_geometry, β, Δτ)

    # initializing tight-binding model measurements
    initialize_measurements!(measurement_container, tight_binding_model)

    # initialize hubbard model measurements
    initialize_measurements!(measurement_container, hubbard_model)

    # measure time-displaced spin-up green's function
    initialize_correlation_measurements!(
        measurement_container = measurement_container,
        model_geometry = model_geometry,
        correlation = "greens",
        time_displaced = false,
        pairs = [(1, 1)]
    )

    # measure time-displaced density-density correlation function
    initialize_correlation_measurements!(
        measurement_container = measurement_container,
        model_geometry = model_geometry,
        correlation = "density",
        time_displaced = false,
        integrated = false,
        pairs = [(1, 1)]
    )

    # measure time-displaced pair correlation function
    initialize_correlation_measurements!(
        measurement_container = measurement_container,
        model_geometry = model_geometry,
        correlation = "pair",
        time_displaced = false,
        integrated = false,
        pairs = [(1, 1)]
    )

    # measure time-displaced spin-spin correlation function in z direction
    initialize_correlation_measurements!(
        measurement_container = measurement_container,
        model_geometry = model_geometry,
        correlation = "spin_z",
        time_displaced = false,
        integrated = false,
        pairs = [(1, 1)]
    )

    # initialize measurement sub-directories
    initialize_measurement_directories(
        simulation_info = simulation_info,
        measurement_container = measurement_container
    )

    ###################################################
    ## SET-UP & INITIALIZE DQMC SIMULATION FRAMEWORK ##
    ###################################################

    # initialize a fermion path integral according non-interacting tight-binding model
    fermion_path_integral_up = FermionPathIntegral(tight_binding_parameters = tight_binding_parameters, β = β, Δτ = Δτ)
    fermion_path_integral_dn = FermionPathIntegral(tight_binding_parameters = tight_binding_parameters, β = β, Δτ = Δτ)

    # initialize fermion path integrals to hubbard parameters
    initialize!(fermion_path_integral_up, fermion_path_integral_dn, hubbard_parameters)

    # initialize fermion path integrals to hubbard continuous hubbard-stratonovich transformation parameters
    initialize!(fermion_path_integral_up, fermion_path_integral_dn, hubbard_hs_parameters)

    # allocate initialize propagators for each imaginary time slice
    Bup = initialize_propagators(fermion_path_integral_up, symmetric=symmetric, checkerboard=checkerboard)
    Bdn = initialize_propagators(fermion_path_integral_dn, symmetric=symmetric, checkerboard=checkerboard)

    # initialize fermion greens calculator
    fermion_greens_calculator_up = dqmcf.FermionGreensCalculator(Bup, β, Δτ, n_stab)
    fermion_greens_calculator_dn = dqmcf.FermionGreensCalculator(Bdn, β, Δτ, n_stab)

    # initialize alternate fermion greens calculators for performing various global updates
    fermion_greens_calculator_up_alt = dqmcf.FermionGreensCalculator(fermion_greens_calculator_up)
    fermion_greens_calculator_dn_alt = dqmcf.FermionGreensCalculator(fermion_greens_calculator_dn)

    # calculate/initialize equal-time Green's function matrices
    Gup = zeros(eltype(Bup[1]), size(Bup[1]))
    Gdn = zeros(eltype(Bdn[1]), size(Bdn[1]))
    logdetGup, sgndetGup = dqmcf.calculate_equaltime_greens!(Gup, fermion_greens_calculator_up)
    logdetGdn, sgndetGdn = dqmcf.calculate_equaltime_greens!(Gdn, fermion_greens_calculator_dn)

    # initialize G(τ,τ), G(τ,0) and G(0,τ) Green's function matrices for both spin species
    Gup_ττ = similar(Gup)
    Gup_τ0 = similar(Gup)
    Gup_0τ = similar(Gup)
    Gdn_ττ = similar(Gdn)
    Gdn_τ0 = similar(Gdn)
    Gdn_0τ = similar(Gdn)

    ############################
    ## PERFORM BURNIN UPDATES ##
    ############################

    # intialize errors associated with numerical instability to zero
    δG = zero(typeof(logdetGup))
    δθ = zero(typeof(sgndetGup))

    # perform thermalization/burnin updates
    for n in 1:N_burnin

        # perform sweep through lattice performing local updates to ising hubbard-stratonovich fields
        (acceptance_rate, logdetGup, sgndetGup, logdetGdn, sgndetGdn, δG, δθ) = lmc_update!(
            Gup, logdetGup, sgndetGup, Gdn, logdetGdn, sgndetGdn, hubbard_hs_parameters,
            Δt = Δt,
            fermion_path_integral_up = fermion_path_integral_up,
            fermion_path_integral_dn = fermion_path_integral_dn,
            fermion_greens_calculator_up = fermion_greens_calculator_up,
            fermion_greens_calculator_dn = fermion_greens_calculator_dn,
            fermion_greens_calculator_up_alt = fermion_greens_calculator_up_alt,
            fermion_greens_calculator_dn_alt = fermion_greens_calculator_dn_alt,
            Bup = Bup, Bdn = Bdn, δG_max = δG_max, δG = δG, δθ = δθ,
            rng = rng, initialize_force = false
        )

        # record acceptance rate of sweep through lattice
        additional_info["acceptance_rate"] += acceptance_rate
    end

    ##################################################################
    ## PERFORM SIMULATION/MEASUREMENT UPDATES AND MAKE MEASUREMENTS ##
    ##################################################################

    # intialize errors associated with numerical instability to zero
    δG = zero(typeof(logdetGup))
    δθ = zero(typeof(sgndetGup))

    # iterate of measurement bins
    for bin in 1:N_bins

        # iterate over updates per bin
        for n in 1:bin_size

            # perform sweep through lattice performing local updates to ising hubbard-stratonovich fields
            (acceptance_rate, logdetGup, sgndetGup, logdetGdn, sgndetGdn, δG, δθ) = lmc_update!(
                Gup, logdetGup, sgndetGup, Gdn, logdetGdn, sgndetGdn, hubbard_hs_parameters,
                Δt = Δt,
                fermion_path_integral_up = fermion_path_integral_up,
                fermion_path_integral_dn = fermion_path_integral_dn,
                fermion_greens_calculator_up = fermion_greens_calculator_up,
                fermion_greens_calculator_dn = fermion_greens_calculator_dn,
                fermion_greens_calculator_up_alt = fermion_greens_calculator_up_alt,
                fermion_greens_calculator_dn_alt = fermion_greens_calculator_dn_alt,
                Bup = Bup, Bdn = Bdn, δG_max = δG_max, δG = δG, δθ = δθ,
                rng = rng, initialize_force = false
            )

            # record acceptance rate of sweep through lattice
            additional_info["acceptance_rate"] += acceptance_rate

            # make measurements
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
                coupling_parameters = (hubbard_parameters,)
            )
        end

        # write measurements to file
        write_measurements!(
            measurement_container = measurement_container,
            simulation_info = simulation_info,
            model_geometry = model_geometry,
            bin = bin,
            bin_size = bin_size,
            Δτ = Δτ
        )
    end

    # normalize acceptance rate measurements
    additional_info["acceptance_rate"] /= (N_updates + N_burnin)

    # record final max stabilization error that was correct and frequency of stabilization
    additional_info["n_stab_final"] = fermion_greens_calculator_up.n_stab
    additional_info["dG"] = δG

    # write simulation information to file
    save_simulation_info(simulation_info, additional_info)

    # process measurements
    process_measurements(simulation_info.datafolder, 32)

    return nothing
end

# run the simulation
run_simulation()