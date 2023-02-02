```@meta
CurrentModule = SmoQyDQMC
```

# SmoQyDQMC

Documentation for [SmoQyDQMC.jl](https://github.com/SmoQySuite/SmoQyDQMC.jl).
This package implements determinant quantum Monte Carlo (DQMC) method for Hubbard,
and electron-phonon interactions, including both Holstein and Su-Schrieffer-Heeger (SSH) style
electron-phonon coupling.

**This code is currently in the experimental phase of development.**

## Funding

The development of this code was supported by the U.S. Department of Energy, Office of Science, Basic Energy Sciences,
under Award Number DE-SC0022311.

## Installation

**NOTE**: This package is in the experimental phase of development and is not yet published to the Julia [`General`](https://github.com/JuliaRegistries/General.git) registry.
The instruction for installation below will be updated once that package is registered.

### Method 1

First clone the [`SmoQyDQMC.jl`](https://github.com/SmoQySuite/SmoQyDQMC.jl) repository onto your machine:
```
git clone https://github.com/SmoQySuite/SmoQyDQMC.jl
```
Then navigate into the [`SmoQyDQMC.jl`](https://github.com/SmoQySuite/SmoQyDQMC.jl) repository directory and open a Julia REPL environment and run the following command:
```julia
] dev .
```

### Method 2

Open a Julia REPL environment and run the following command:
```julia
] dev https://github.com/SmoQySuite/SmoQyDQMC.jl
```
This command clones the [`SmoQyDQMC.jl`](https://github.com/SmoQySuite/SmoQyDQMC.jl) repository to the hidden direcotry `.julia/dev` that exists in the same directory where Julia is installed.

## Supported Hamiltonians

The [`SmoQyDQMC.jl`](https://github.com/SmoQySuite/SmoQyDQMC.jl) currently support Hamitonians of the form

```math
\hat{H} = \hat{H}_{0}+\hat{H}_{\textrm{hub.}}+\hat{H}_{e\textrm{-ph}}
```
for arbitrary lattice geometries. The term ``\hat{H}_{0}`` corresponds to the non-interacting tight-binding model given by
```math
\hat{H}_{0} = \sum_{\langle(\mathbf{i},\nu),(\mathbf{j},\kappa)\rangle,\sigma}\big(-t_{(\mathbf{i},\nu),(\mathbf{j},\kappa)}\hat{c}_{\sigma,\mathbf{i},\nu}^{\dagger}\hat{c}_{\sigma,\mathbf{j},\kappa}+\textrm{h.c.}\big)+\sum_{\mathbf{i},\nu,\sigma}\big(\epsilon_{\mathbf{i},\nu}-\mu\big)\hat{n}_{\sigma,\mathbf{i},\nu},
```
where ``\sigma`` is the electron spin, ``\mathbf{i}`` and ``\mathbf{j}`` specify the unit cell, and ``\kappa`` and ``\nu`` specify the orbital species in the unit cell.

The ``\hat{H}_{\textrm{hub.}}`` term corresponds to the Hubbard interaction, which may be represented in either the form
```math
\hat{H}_{\textrm{hub.}} = \sum_{\mathbf{i},\nu}U_{\mathbf{i},\nu}\hat{n}_{\uparrow,\mathbf{i},\nu}\hat{n}_{\downarrow,\mathbf{i},\nu}
```
or
```math
\hat{H}_{\textrm{hub.}} = \sum_{\mathbf{i},\nu}U_{\mathbf{i},\nu}\big(\hat{n}_{\uparrow,\mathbf{i},\nu}-\tfrac{1}{2}\big)\big(\hat{n}_{\downarrow,\mathbf{i},\nu}-\tfrac{1}{2}\big).
```
The code supports both repulsive ``(U>0)`` and attractive ``(U<0)`` Hubbard interaction.

The term
```math
\hat{H}_{e\textrm{-ph}} = \hat{H}_{\textrm{ph.}}+\hat{H}_{\textrm{hol.}}+\hat{H}_{\textrm{ssh}}+\hat{H}_{\textrm{disp.}}
```
represents the electron-phonon interaction component of the Hamiltonian.
Here ``\hat{H}_{\textrm{ph.}}`` defines a population of non-interacting local phonon modes given by
```math
\hat{H}_{\textrm{ph.}} = \sum_{\mathbf{i},\nu}\bigg(\frac{1}{2M_{\mathbf{i},\nu}}\hat{P}_{\mathbf{i},\nu}^{2}+\frac{1}{2}M_{\mathbf{i},\nu}\Omega_{\mathbf{i},\nu}^{2}\hat{X}_{\mathbf{i},\nu}^{2}+\frac{1}{24}M_{\mathbf{i},\nu}\Omega_{4,\mathbf{i},\nu}^{2}\hat{X}_{\mathbf{i},\nu}^{4}\bigg),
```
where the index ``\nu`` runs over all phonon modes in each unit-cell. Note that any number independent of phonon modes can be placed on each orbital in the unit cell.

Next, the ``\hat{H}_{\textrm{hol.}}`` term describes the Holstein ``e``-ph couplings in the model, that may be either local or long-ranged, given by
```math
\hat{H}_{\textrm{hol.}} = \sum_{\mathbf{i},\sigma}\left[\sum_{n=1}^{4}\alpha_{n,\mathbf{i},(\kappa,\nu,\mathbf{r})}\hat{X}_{\mathbf{i},\kappa}^{n}\right]\big(\hat{n}_{\sigma,\mathbf{i}+\mathbf{r},\nu}-\tfrac{1}{2}\big),
```
with non-linear coupling out to fourth order supported.
Similarly, the ``\hat{H}_{\textrm{ssh}}`` term describes the Su–Schrieffer–Heeger (SSH) ``e``-ph couplings in the model, and is given by
```math
\hat{H}_{\textrm{ssh}} = \sum_{\mathbf{i},\sigma}\left[\sum_{n=1}^{4}\alpha_{n,\mathbf{i},(\kappa,\nu,\mathbf{r})}\Big(\hat{X}_{\mathbf{i}+\mathbf{r},\nu}-\hat{X}_{\mathbf{k},\kappa}\Big)^{n}\right]\big(\hat{c}_{\sigma,\mathbf{i}+\mathbf{r},\nu}^{\dagger}\hat{c}_{\sigma,\mathbf{i},\kappa}+\textrm{h.c.}\big),
```
where again non-linear coupling out to fourth order are supported.
Lastly, the ``\hat{H}_{\textrm{disp.}}`` term describes dispersive coupling between phonon modes, and is given by
```math
\hat{H}_{\textrm{disp.}} = \sum_{\mathbf{i}}\bigg(\frac{M_{\mathbf{i},\kappa}M_{\mathbf{i}+\mathbf{r},\nu}}{M_{\mathbf{i},\kappa}+M_{\mathbf{i}+\mathbf{r},\nu}}\bigg)\left[\Omega_{\mathbf{i},(\kappa,\nu,\mathbf{r})}^{2}\Big(\hat{X}_{\mathbf{i}+\mathbf{r},\nu}-\hat{X}_{\mathbf{k},\kappa}\Big)^{2}+\frac{1}{12}\Omega_{4,\mathbf{i},(\kappa,\nu,\mathbf{r})}^{2}\Big(\hat{X}_{\mathbf{i}+\mathbf{r},\nu}-\hat{X}_{\mathbf{k},\kappa}\Big)^{4}\right].
```

## Notable Package Dependencies

This section reviews some of the more notable and interesting packages package dependencies.

### Re-exported Packages

The [SmoQyDQMC.jl](https://github.com/SmoQySuite/SmoQyDQMC.jl) re-exports certain packages using
the [`Reexport.jl`](https://github.com/simonster/Reexport.jl.git) package in order to simplify the installation process.

- [`LatticeUtilties.jl`](https://github.com/cohensbw/LatticeUtilities.jl.git): Used to represent arbitrary lattice geometries.
- [`JDQMCFramework.jl`](https://github.com/SmoQySuite/JDQMCFramework.jl.git): Impelements and exports the basic framework for running a DQMC simulation.
- [`JDQMCMeasurements.jl`](https://github.com/SmoQySuite/JDQMCMeasurements.jl.git): Implements various global, local and correlation measurements for a DQMC simulation.
- [`MuTuner.jl`](https://github.com/cohensbw/MuTuner.jl.git): Impelments and exports an algorithm for tuning the chemical potential to achieve a target density in grand canonical Monte Carlo simulations.

### External Dependencies

- [`StableLinearAlgebra.jl`](https://github.com/cohensbw/StableLinearAlgebra.jl.git): Implements optimized numerical stabilizaiton methods required by DQMC simulations.
- [`Checkerboard.jl`](https://github.com/cohensbw/Checkerboard.jl.git): Implements and exports the checkerboard method for approximating exponentiated hopping matrices by a sparse matrix.
- [`JLD2.jl`](https://github.com/JuliaIO/JLD2.jl.git): Package used to write data to binary files in an HDF5 compatible format. It is also recommended this package be used at the scripting level to implement checkpointing in a simulation.
- [`BinningAnalysis.jl`](https://github.com/carstenbauer/BinningAnalysis.jl.git): Export method impelementing the jackknife algorithm for calculating error bars.

## Contact Us

For question and comments regarding this package, please email either Dr. Benjamin Cohen-Stead at [bcohenst@utk.edu](mailto:bcohenst@utk.edu) or Professor Steven Johnston at [sjohn145@utk.edu](mailto:sjohn145@utk.edu).