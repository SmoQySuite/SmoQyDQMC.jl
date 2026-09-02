# Tips & Tricks

## Accessing the Phonon Fields Directly

In certain situations it is advantageous to directly access the phonon field configurations in a DQMC simulation.
For instance, if you know the ground state is ordered in a certain way — such as the charge density wave (CDW) phase in the half-filled Holstein model — it can be helpful to initialize the phonon configuration in a simulation to begin in a staggered pattern that reflects the expected CDW phase, as this can significantly reduce the number of updates required to thermalize the system.

This is just one example situation where accessing the phonon field configuration directly can be useful.
Now let us discuss how to do so in practice.
Let us suppose you have written a simulation script, and in this script the following variables have been initialized:
- `lattice`: Instance of the [`Lattice`](https://smoqysuite.github.io/LatticeUtilities.jl/stable/api/#LatticeUtilities.Lattice) type.
- `electron_phonon_model`: Instance of the [`ElectronPhononModel`](@ref) type.
- `electron_phonon_parameters`: Instance of the [`ElectronPhononParameters`](@ref) type.
The phonon field configuration is stored in the matrix `electron_phonon_parameters.x` matrix, which has dimension
``(N_\text{ph}, L_\tau)``, where ``N_\text{ph}`` is the total number of phonon modes in the finite lattice,
and ``L_\tau`` is the length of the discretized imaginary-time axis.

Now I will describe how to reshape this matrix to make accessing desired fields simpler.
The extent of the finite lattice being simulated in the direction of each lattice vector is stored in the vector
of integers `lattice.L` of length ``D``, where ``D \in \{1, 2, 3 \}`` is the dimension of the system being simulated.
The number of phonon modes ``n_\text{ph}`` in each unit cell is given by `n_ph = length(electron_phonon_model.phonon_modes)`.
Additionally, the length of the discretized imaginary-time axis ``L_\tau`` is stored in the struct field `electron_phonon_parameters.Lτ`.
Therefore, it is useful to reshape the array containing the phonon fields in the following manner:
```julia
x = reshape(electron_phonon_parameters.x, (lattice.L..., n_ph, electron_phonon_parameters.Lτ))
```
Now it is relatively straightforward to maps from a specified imaginary-time slice, phonon mode species and unit cell location to a specific phonon field.

As an example, consider the phonon field `x[2,1,3,37]`.
This phonon field is associated with the 37'th imaginary-time slice.
It corresponds to the third phonon mode type found in each unit cell, which corresponds to `PHONON_ID = 3`; refer to [Model Summary](@ref) for more information.
Lastly, this phonon field lives in a unit cell displaced from the origin/reference unit cell by one unit cell in the direction of the first lattice vector ``(2 - 1 = 1)`` and zero unit cells in the direction of the second lattice vector ``(1 - 1 = 0)``, given that Julia indexes from one.