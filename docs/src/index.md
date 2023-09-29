```@meta
CurrentModule = SmoQyDQMC
```

# SmoQyDQMC

Documentation for [SmoQyDQMC.jl](https://github.com/SmoQySuite/SmoQyDQMC.jl).
This package implements the determinant quantum Monte Carlo (DQMC) method for Hubbard,
and electron-phonon interactions, including both Holstein and Su-Schrieffer-Heeger (SSH) style
electron-phonon coupling.

**This code is currently in the experimental phase of development.**

## Funding

The development of this code was supported by the U.S. Department of Energy, Office of Science, Basic Energy Sciences,
under Award Number DE-SC0022311.

## Installation

To install the [SmoQyDQMC.jl](https://github.com/SmoQySuite/SmoQyDQMC.jl),
simply open the Julia REPL and run the following command:
```julia
julia> ]
pkg> add SmoQyDQMC
```
or equivalently via `Pkg` do
```julia
julia> using Pkg; Pkg.add("SmoQyDQMC")
```

## Supported Hamiltonians

This section describes the class of Hamiltonians [`SmoQyDQMC.jl`](https://github.com/SmoQySuite/SmoQyDQMC.jl) currently supports,
and how the various terms appearing in the Hamiltonian are parameterized within the code.
We start by partitioning the full Hamiltonian as 
```math
\begin{align*}
    \hat{\mathcal{H}} = \hat{\mathcal{U}} + \hat{\mathcal{K}} + \hat{\mathcal{V}},
\end{align*}
```
where ``\hat{\mathcal{U}}`` is the bare lattice energy, ``\hat{\mathcal{K}}`` the total electron kinetic energy, and ``\hat{\mathcal{V}}`` the total electron potential energy. In the discussion that follows we apply the normalization ``\hbar = 1`` throughout.

The bare lattice term is further decomposed into
```math
\begin{align*}
    \hat{\mathcal{U}} = \hat{\mathcal{U}}_{\rm ph} + \hat{\mathcal{U}}_{\rm disp},
\end{align*}
```
where
```math
\begin{align*}
    \hat{\mathcal{U}}_{\rm ph} =& \sum_{\mathbf{i},\nu}\sum_{n_{\mathbf{i},\nu}}
        \left[
            \frac{1}{2M_{n_{\mathbf{i},\nu}}}\hat{P}_{n_{\mathbf{i},\nu}}
            + \frac{1}{2}M_{n_{\mathbf{i},\nu}}\Omega_{0,n_{\mathbf{i},\nu}}^2\hat{X}_{n_{\mathbf{i},\nu}}^2
            + \frac{1}{24}M_{n_{\mathbf{i},\nu}}\Omega_{a,n_{\mathbf{i},\nu}}^2\hat{X}_{n_{\mathbf{i},\nu}}^4
        \right]
\end{align*}
```
describes the placement of local dispersionless phonon (LDP) modes in the lattice, i.e. an Einstein solid, and
```math
\begin{align*}
    \hat{\mathcal{U}}_{\rm disp} =& \sum_{\substack{\mathbf{i},\nu \\ \mathbf{j},\gamma}}\sum_{\substack{n_{\mathbf{i},\nu} \\ n_{\mathbf{j},\gamma}}}
        \frac{M_{n_{\mathbf{i},\alpha}}M_{n_{\mathbf{j},\gamma}}}{M_{n_{\mathbf{i},\alpha}}+M_{n_{\mathbf{j},\gamma}}}\left[
            \tilde{\Omega}^2_{0,n_{\mathbf{i},\alpha},n_{\mathbf{j},\gamma}}(\hat{X}_{n_{\mathbf{i},\nu}}-\hat{X}_{n_{\mathbf{j},\gamma}})^2
            + \frac{1}{12}\tilde{\Omega}^2_{a,n_{\mathbf{i},\alpha},n_{\mathbf{j},\gamma}}(\hat{X}_{n_{\mathbf{i},\nu}}-\hat{X}_{n_{\mathbf{j},\gamma}})^4
        \right]
\end{align*}
```
introduces dispersion between the LDP modes. The sums over ``\mathbf{i} \ (\mathbf{j})`` and ``\nu \ (\gamma)`` run over unit cells in the lattice and orbitals within each unit cell respectively. A sum over ``n_{\mathbf{i},\nu} \ (n_{\mathbf{j},\gamma})`` then runs over the LDP modes placed on a given orbital in the lattice.

The position and momentum operators for each LPD mode are given by ``\hat{X}_{n_{\mathbf{i},\nu}}`` and ``\hat{P}_{n_{\mathbf{i},\nu}}`` respectively, with corresponding phonon mass ``M_{n_{\mathbf{i},\nu}}``. The spring constant is ``K_{n_{\mathbf{i},\nu}} = M_{n_{\mathbf{i},\nu}} \Omega_{0,n_{n_{\mathbf{i},\nu}}}^2``, with ``\Omega_{0,n_{n_{\mathbf{i},\nu}}}`` specifying the phonon frequency. The ``U_{\rm ph}`` also supports an anharmonic ``\hat{X}_{n_{\mathbf{i},\nu}}^4`` contribution to the LDP potential energy that is controlled by the parameter ``\Omega_{a,n_{n_{\mathbf{i},\nu}}}``. Similary, ``\tilde{\Omega}_{0,n_{\mathbf{i},\alpha},n_{\mathbf{j},\gamma}} \ (\tilde{\Omega}_{a,n_{\mathbf{i},\alpha},n_{\mathbf{j},\gamma}})`` is the coefficient controlling harmonic (anhmaronic) dispersion between LDP modes.

Next we trace out the phonon degrees of freedom 

The electron kinetic energy is decomposed as
```math
\begin{align*}
    \hat{\mathcal{K}} = \hat{\mathcal{K}}_0 + \hat{\mathcal{K}}_{\rm ssh},
\end{align*}
```
where
```math
\begin{align*}
    \hat{\mathcal{K}}_0 =& -\sum_\sigma\sum_{\substack{\mathbf{i},\nu \\ \mathbf{j},\gamma}}
        \left[
            t_{(\mathbf{i},\nu),(\mathbf{j},\gamma)} \hat{c}^\dagger_{\sigma,\mathbf{i},\nu}\hat{c}_{\sigma,\mathbf{j},\gamma} + {\rm h.c.}
        \right]
\end{align*}
```
is the non-interacting electron kinetic energy, and
```math
\begin{align*}
    \hat{\mathcal{K}}_{\rm ssh} =& \sum_\sigma\sum_{\substack{\mathbf{i},\nu \\ \mathbf{j},\gamma}}\sum_{\substack{n_{\mathbf{i},\nu} \\ n_{\mathbf{j},\gamma}}}\sum_{m=1}^4
        (\hat{X}_{n_{\mathbf{i},\nu}}-\hat{X}_{n_{\mathbf{j},\gamma}})^m\left[
            \alpha_{m,n_{\mathbf{i},\nu},n_{\mathbf{j},\gamma}} \hat{c}^\dagger_{\sigma,\mathbf{i},\nu}\hat{c}_{\sigma,\mathbf{j},\gamma} + {\rm h.c.}
        \right]
\end{align*}
```
is describes the interaction between the lattice degrees of freedom and the electron kinetic energy via a Su-Schrieffer-Heeger (SSH)-like coupling mechanism. The hopping integral between from orbital ``\gamma`` in unit cell ``\mathbf{j}`` to orbital ``\nu`` in unit cell ``\mathbf{i}`` is given by ``t_{(\mathbf{i},\nu),(\mathbf{j},\gamma)}``, and may in general be complex. The modulations to this hopping integral are controlled by the parameters ``\alpha_{m,(\mathbf{i},\nu),(\mathbf{j},\gamma)}``, where ``m\in [1,4]`` specifies the order of the difference in the phonon positions that modulates the hopping integral.

Lastly, the electron potential energy is broken down into the three terms
```math
\begin{align*}
    \hat{\mathcal{V}} = \hat{\mathcal{V}}_0 + \hat{\mathcal{V}}_{\rm hol} + \hat{\mathcal{V}}_{\rm hub},
\end{align*}
```
where
```math
\begin{align*}
    \hat{\mathcal{V}}_0 =& \sum_\sigma\sum_{\mathbf{i},\nu}
        \left[
            (\epsilon_{\mathbf{i},\nu} - \mu) \hat{n}_{\sigma,\mathbf{i},\nu}
        \right]
\end{align*}
```
is the non-interacting electron potential energy,
```math
\begin{align*}
    \hat{\mathcal{V}}_{\rm hol} =& \sum_\sigma\sum_{\substack{\mathbf{i},\nu \\ \mathbf{j},\gamma}}\sum_{n_{\mathbf{i},\nu}}\sum_{m=1}^4
        \hat{X}^m_{n_{\mathbf{i},\nu}} \left[
            \tilde{\alpha}_{m,n_{\mathbf{i},\nu},(\mathbf{j},\gamma)} (\hat{n}_{\sigma,\mathbf{j},\gamma}-\tfrac{1}{2})
        \right]
\end{align*}
```
is the contribution to the electron potential energy that results from a Holstein-like coupling to the lattice degrees of freedom, and
```math
\begin{align*}
    \hat{\mathcal{V}}_{{\rm hub}}=&
    \begin{cases}
        \sum_{\mathbf{i},\nu}U_{\mathbf{i},\nu}\big(\hat{n}_{\uparrow,\mathbf{i},\nu}-\tfrac{1}{2}\big)\big(\hat{n}_{\downarrow,\mathbf{i},\nu}-\tfrac{1}{2}\big)\\
        \sum_{\mathbf{i},\nu}U_{\mathbf{i},\nu}\hat{n}_{\uparrow,\mathbf{i},\nu}\hat{n}_{\downarrow,\mathbf{i},\nu}
    \end{cases}
\end{align*}
```
is the on-site Hubbard interaction contribution to the electron potential energy. In ``\hat{\mathcal{V}}_0`` the chemical potential is given by ``\mu``, and ``\epsilon_{\mathbf{i},\nu}`` is the on-site energy, the parameter ``\tilde{\alpha}_{m,n_{\mathbf{i},\nu},(\mathbf{j},\gamma)}`` controls the strength of the Holstein-like coupling in ``\hat{\mathcal{V}}_{\rm ph}``, and ``U_{\mathbf{i},\nu}`` is the on-site Hubbard interaction strength in ``\hat{\mathcal{V}}_{\rm hub}``. Note that either functional form for ``V_{\rm hub}`` can used in the code.

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