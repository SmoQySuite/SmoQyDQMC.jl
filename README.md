# SmoQyDQMC

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://SmoQySuite.github.io/SmoQyDQMC.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://SmoQySuite.github.io/SmoQyDQMC.jl/dev/)
[![Build Status](https://github.com/SmoQySuite/SmoQyDQMC.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/SmoQySuite/SmoQyDQMC.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/SmoQySuite/SmoQyDQMC.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/SmoQySuite/SmoQyDQMC.jl)
![](https://img.shields.io/badge/Lifecycle-Maturing-007EC6g)

This package implements the determinant quantum Monte Carlo (DQMC) method for Hubbard,
and electron-phonon interactions, including both Holstein and Su-Schrieffer-Heeger (SSH) style
electron-phonon coupling.

## Funding

The development of this code was supported by the U.S. Department of Energy, Office of Science, Basic Energy Sciences,
under Award Number DE-SC0022311.

## Installation

To install the [SmoQyDQMC.jl](https://github.com/SmoQySuite/SmoQyDQMC.jl),
simply open the Julia REPL and run the command
```julia
julia> ]
pkg> add SmoQyDQMC
```
or equivalently via `Pkg` do
```julia
julia> using Pkg; Pkg.add("SmoQyDQMC")
```

## Documentation

- [`STABLE`](https://SmoQySuite.github.io/SmoQyDQMC.jl/stable/): Documentation for the latest version of the code published to the Julia [General](https://github.com/JuliaRegistries/General.git) registry.
- [`DEV`](https://SmoQySuite.github.io/SmoQyDQMC.jl/dev/): Documentation associated with most recent commit to the main branch.

## Publication List

Follow this [link](https://smoqysuite.github.io/SmoQyDQMC.jl/dev/#Publication-List)
to see a list of some of the publications that report results generated using
the [SmoQyDQMC.jl](https://github.com/SmoQySuite/SmoQyDQMC.jl) package.

## Notable Package Dependencies

This section reviews some notable package dependencies.

### Re-exported Packages

The [SmoQyDQMC.jl](https://github.com/SmoQySuite/SmoQyDQMC.jl) re-exports certain packages using
the [Reexport.jl](https://github.com/simonster/Reexport.jl.git) package in order to simplify the installation process.

- [LatticeUtilties.jl](https://github.com/SmoQySuite/LatticeUtilities.jl.git): Used to represent arbitrary lattice geometries.
- [JDQMCFramework.jl](https://github.com/SmoQySuite/JDQMCFramework.jl.git): Implements and exports the basic framework for running a DQMC simulation.
- [JDQMCMeasurements.jl](https://github.com/SmoQySuite/JDQMCMeasurements.jl.git): Implements various global, local and correlation measurements for a DQMC simulation.
- [MuTuner.jl](https://github.com/cohensbw/MuTuner.jl.git): Impelments and exports an algorithm for tuning the chemical potential to achieve a target density in grand canonical Monte Carlo simulations.

### External Dependencies

- [StableLinearAlgebra.jl](https://github.com/SmoQySuite/StableLinearAlgebra.jl.git): Implements optimized numerical stabilizaiton methods required by DQMC simulations.
- [Checkerboard.jl](https://github.com/SmoQySuite/Checkerboard.jl.git): Implements and exports the checkerboard method for approximating exponentiated hopping matrices by a sparse matrix.
- [JLD2.jl](https://github.com/JuliaIO/JLD2.jl.git): Package used to write data to binary files in an HDF5 compatible format. It is also recommended this package be used at the scripting level to implement checkpointing in a simulation.

## Citation

If you found this library to be useful in the course of academic work, please consider citing us:

```bibtex
@misc{SmoQyDQMC,
      title={SmoQyDQMC.jl: A flexible implementation of determinant quantum Monte Carlo for Hubbard and electron-phonon interactions}, 
      author={Benjamin Cohen-Stead and Sohan Malkaruge Costa and James Neuhaus and Andy Tanjaroon Ly and Yutan Zhang and Richard Scalettar and Kipton Barros and Steven Johnston},
      year={2023},
      eprint={2311.09395},
      archivePrefix={arXiv},
      primaryClass={cond-mat.str-el},
      url={https://arxiv.org/abs/2311.09395}
}
```

## Publications

A list of some of the publications that report results generated using [SmoQyDQMC.jl](https://github.com/SmoQySuite/SmoQyDQMC.jl)
can be found [here](https://smoqysuite.github.io/SmoQyDQMC.jl/stable/#Publication-List).

## Contact Us

For question and comments regarding this package, please email either Dr. Benjamin Cohen-Stead at [bcohenst@utk.edu](mailto:bcohenst@utk.edu) or Professor Steven Johnston at [sjohn145@utk.edu](mailto:sjohn145@utk.edu).