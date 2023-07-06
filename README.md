# SmoQyDQMC

<!-- [![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://SmoQySuite.github.io/SmoQyDQMC.jl/stable/) -->
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://SmoQySuite.github.io/SmoQyDQMC.jl/dev/)
[![Build Status](https://github.com/SmoQySuite/SmoQyDQMC.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/SmoQySuite/SmoQyDQMC.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/SmoQySuite/SmoQyDQMC.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/SmoQySuite/SmoQyDQMC.jl)
![](https://img.shields.io/badge/Lifecycle-Maturing-007EC6g)

This package implements the determinant quantum Monte Carlo (DQMC) method for Hubbard,
and electron-phonon interactions, including both Holstein and Su-Schrieffer-Heeger (SSH) style
electron-phonon coupling.

This package is currently in the experimental phase of development.

## Funding

The development of this code was supported by the U.S. Department of Energy, Office of Science, Basic Energy Sciences,
under Award Number DE-SC0022311.

## Documentation

- [`DEV`](https://SmoQySuite.github.io/SmoQyDQMC.jl/dev/): Documentation associated with most recent commit to the main branch.

## Notable Package Dependencies

This section reviews some notable package dependencies.

### Re-exported Packages

The [SmoQyDQMC.jl](https://github.com/SmoQySuite/SmoQyDQMC.jl) re-exports certain packages using
the [`Reexport.jl`](https://github.com/simonster/Reexport.jl.git) package in order to simplify the installation process.

- [`LatticeUtilties.jl`](https://github.com/cohensbw/LatticeUtilities.jl.git): Used to represent arbitrary lattice geometries.
- [`JDQMCFramework.jl`](https://github.com/SmoQySuite/JDQMCFramework.jl.git): Implements and exports the basic framework for running a DQMC simulation.
- [`JDQMCMeasurements.jl`](https://github.com/SmoQySuite/JDQMCMeasurements.jl.git): Implements various global, local and correlation measurements for a DQMC simulation.
- [`MuTuner.jl`](https://github.com/cohensbw/MuTuner.jl.git): Impelments and exports an algorithm for tuning the chemical potential to achieve a target density in grand canonical Monte Carlo simulations.

### External Dependencies

- [`StableLinearAlgebra.jl`](https://github.com/cohensbw/StableLinearAlgebra.jl.git): Implements optimized numerical stabilizaiton methods required by DQMC simulations.
- [`Checkerboard.jl`](https://github.com/cohensbw/Checkerboard.jl.git): Implements and exports the checkerboard method for approximating exponentiated hopping matrices by a sparse matrix.
- [`JLD2.jl`](https://github.com/JuliaIO/JLD2.jl.git): Package used to write data to binary files in an HDF5 compatible format. It is also recommended this package be used at the scripting level to implement checkpointing in a simulation.
- [`BinningAnalysis.jl`](https://github.com/carstenbauer/BinningAnalysis.jl.git): Export method impelementing the jackknife algorithm for calculating error bars.

## Contact Us

For question and comments regarding this package, please email either Dr. Benjamin Cohen-Stead at [bcohenst@utk.edu](mailto:bcohenst@utk.edu) or Professor Steven Johnston at [sjohn145@utk.edu](mailto:sjohn145@utk.edu).