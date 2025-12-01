using SmoQyDQMC
import SmoQyDQMC.LatticeUtilities as lu
import SmoQyDQMC.JDQMCFramework as dqmcf

using Random
using Printf

# Top-level function to run simulation.
function run_simulation(;
    U, # Hubbard interaction.
    t′, # Next-nearest-neighbor hopping amplitude.
    μ, # Chemical potential.
    L, # System size.
    β, # Inverse temperature.
    N_updates, # Total number of measurements and measurement updates.
    Δτ = 0.05, # Discretization in imaginary time.
    n_stab = 10, # Numerical stabilization period in imaginary-time slices.
    δG_max = 1e-6, # Threshold for numerical error corrected by stabilization.
    symmetric = false, # Whether symmetric propagator definition is used.
    checkerboard = false, # Whether checkerboard approximation is used.
    seed = abs(rand(Int)), # Seed for random number generator.
)

    # Initialize random number generator
    rng = Xoshiro(seed)

    # Define unit cell.
    unit_cell = lu.UnitCell(
        lattice_vecs = [[1.0, 0.0],
                        [0.0, 1.0]],
        basis_vecs = [[0.0, 0.0]]
    )

    # Define finite lattice with periodic boundary conditions.
    lattice = lu.Lattice(
        L = [L, L],
        periodic = [true, true]
    )

    # Initialize model geometry.
    model_geometry = ModelGeometry(
        unit_cell, lattice
    )

    # Define the nearest-neighbor bond in +x direction.
    bond_px = lu.Bond(
        orbitals = (1,1),
        displacement = [1, 0]
    )

    # Add this bond definition to the model, by adding it the model_geometry.
    bond_px_id = add_bond!(model_geometry, bond_px)

    # Define the nearest-neighbor bond in +y direction.
    bond_py = lu.Bond(
        orbitals = (1,1),
        displacement = [0, 1]
    )

    # Add this bond definition to the model, by adding it the model_geometry.
    bond_py_id = add_bond!(model_geometry, bond_py)

    # Define the next-nearest-neighbor bond in +x+y direction.
    bond_pxpy = lu.Bond(
        orbitals = (1,1),
        displacement = [1, 1]
    )

    # Define the nearest-neighbor bond in -x direction.
    # Will be used to make measurements later in this tutorial.
    bond_nx = lu.Bond(
        orbitals = (1,1),
        displacement = [-1, 0]
    )

    # Add this bond definition to the model, by adding it the model_geometry.
    bond_nx_id = add_bond!(model_geometry, bond_nx)

    # Define the nearest-neighbor bond in -y direction.
    # Will be used to make measurements later in this tutorial.
    bond_ny = lu.Bond(
        orbitals = (1,1),
        displacement = [0, -1]
    )

    # Add this bond definition to the model, by adding it the model_geometry.
    bond_ny_id = add_bond!(model_geometry, bond_ny)

    # Define the next-nearest-neighbor bond in +x+y direction.
    bond_pxpy = lu.Bond(
        orbitals = (1,1),
        displacement = [1, 1]
    )

    # Add this bond definition to the model, by adding it the model_geometry.
    bond_pxpy_id = add_bond!(model_geometry, bond_pxpy)

    # Define the next-nearest-neighbor bond in +x-y direction.
    bond_pxny = lu.Bond(
        orbitals = (1,1),
        displacement = [1, -1]
    )

    # Add this bond definition to the model, by adding it the model_geometry.
    bond_pxny_id = add_bond!(model_geometry, bond_pxny)

    # Set neartest-neighbor hopping amplitude to unity,
    # setting the energy scale in the model.
    t = 1.0

    # Define the non-interacting tight-binding model.
    tight_binding_model = TightBindingModel(
        model_geometry = model_geometry,
        t_bonds = [bond_px, bond_py, bond_pxpy, bond_pxny], # defines hopping
        t_mean  = [t, t, t′, t′], # defines corresponding mean hopping amplitude
        t_std   = [0., 0., 0., 0.], # defines corresponding standard deviation in hopping amplitude
        ϵ_mean  = [0.], # set mean on-site energy for each orbital in unit cell
        ϵ_std   = [0.], # set standard deviation of on-site energy or each orbital in unit cell
        μ       = μ # set chemical potential
    )

    # Define the Hubbard interaction in the model.
    hubbard_model = HubbardModel(
        ph_sym_form = true, # if particle-hole symmetric form for Hubbard interaction is used.
        U_orbital   = [1], # orbitals in unit cell with Hubbard interaction.
        U_mean      = [U], # mean Hubbard interaction strength for corresponding orbital species in unit cell.
        U_std       = [0.], # standard deviation of Hubbard interaction strength for corresponding orbital species in unit cell.
    )

    # Initialize tight-binding parameters.
    tight_binding_parameters = TightBindingParameters(
        tight_binding_model = tight_binding_model,
        model_geometry = model_geometry,
        rng = rng
    )

    # Initialize Hubbard interaction parameters.
    hubbard_params = HubbardParameters(
        model_geometry = model_geometry,
        hubbard_model = hubbard_model,
        rng = rng
    )

    # Apply Hubbard-Stratonovich (HS) transformation to decouple the Hubbard interaction,
    # and initialize the corresponding HS fields that will be sampled in the DQMC simulation.
    hst_parameters = HubbardDensityHirschHST(
        β = β, Δτ = Δτ,
        hubbard_parameters = hubbard_params,
        rng = rng
    )

    # Initialize FermionPathIntegral type for both the spin-up and spin-down electrons to account for Hubbard interaction.
    initialize!(fermion_path_integral, hubbard_params)

    # Initialize FermionPathIntegral type for both the spin-up and spin-down electrons to account for the current
    # Hubbard-Stratonovich field configuration.
    initialize!(fermion_path_integral, hst_parameters)

    # Initialize imaginary-time propagators for all imaginary-time slices for spin-up and spin-down electrons.
    B = initialize_propagators(fermion_path_integral, symmetric=symmetric, checkerboard=checkerboard)

    # Initialize FermionGreensCalculator type for spin-up and spin-down electrons.
    fermion_greens_calculator = dqmcf.FermionGreensCalculator(B, β, Δτ, n_stab)

    # Initialize alternate FermionGreensCalculator type for performing reflection updates.
    fermion_greens_calculator_alt = dqmcf.FermionGreensCalculator(fermion_greens_calculator)

    # Allocate matrices for spin-up and spin-down electron Green's function matrices.
    G = zeros(eltype(B[1]), size(B[1]))

    # Initialize the spin-up and spin-down electron Green's function matrices, also
    # calculating their respective determinants as the same time.
    logdetG, sgndetG = dqmcf.calculate_equaltime_greens!(G, fermion_greens_calculator)

    # Initialize diagnostic parameters to asses numerical stability.
    δG = zero(logdetG)
    δθ = zero(logdetG)

    # Iterate over updates and measurements.
    for update in 1:N_updates

        # Perform reflection update for HS fields with randomly chosen site.
        (accepted, logdetG, sgndetG) = reflection_update!(
            G, logdetG, sgndetG, hst_parameters,
            fermion_path_integral = fermion_path_integral,
            fermion_greens_calculator = fermion_greens_calculator,
            fermion_greens_calculator_alt = fermion_greens_calculator_alt,
            B = B, rng = rng
        )

        # Perform reflection update for HS fields with randomly chosen site.
        (accepted, logdetG, sgndetG) = swap_update!(
            G, logdetG, sgndetG, hst_parameters,
            fermion_path_integral = fermion_path_integral,
            fermion_greens_calculator = fermion_greens_calculator,
            fermion_greens_calculator_alt = fermion_greens_calculator_alt,
            B = B, rng = rng
        )

        # Perform sweep all imaginary-time slice and orbitals, attempting an update to every HS field.
        (acceptance_rate, logdetG, sgndetG, δG, δθ) = local_updates!(
            G, logdetG, sgndetG, hst_parameters,
            fermion_path_integral = fermion_path_integral,
            fermion_greens_calculator = fermion_greens_calculator,
            B = B, δG_max = δG_max, δG = δG, δθ = δθ, rng = rng,
            update_stabilization_frequency = false
        )
    end

    return nothing
end # end of run_simulation function
