# Examples

In this section we link to and discuss some example scripts for running DQMC simulations of various systems.

## Optical SSH Chain

In the script [`examples/ossh_chain.jl`](https://github.com/SmoQySuite/SmoQyDQMC.jl/blob/main/examples/ossh_chain.jl)
we set up a simulation of an optical SSH model on a one-dimensional chain, where the Hamiltonian is given by
```math
\begin{align}
\hat{H}_{\textrm{o-ssh}} = & -\sum_{i,\sigma}[t-\alpha(\hat{X}_{i+1}-\hat{X}_{i})](\hat{c}_{i+1,\sigma}^{\dagger}\hat{c}_{i,\sigma}+\textrm{h.c.})-\mu\sum_{i,\sigma}\hat{n}_{i,\sigma}\\
                           & +\sum_{i}\bigg[\frac{1}{2M}\hat{P}_{i}^{2}+\frac{1}{2}M\Omega^{2}\hat{X}_{i}^{2}\bigg].
\end{align}
```
The simulation is run with the command
```
> julia -O3 ossh_chain.jl
```

## Bond SSH Chain

In the script [`examples/bssh_chain.jl`](https://github.com/SmoQySuite/SmoQyDQMC.jl/blob/main/examples/bssh_chain.jl)
we set up a simulation of an bond SSH model on a one-dimensional chain, where the Hamiltonian is given by
```math
\begin{align}
\hat{H}_{\textrm{b-ssh}} = & -\sum_{i,\sigma}[t-\alpha(\hat{X}_{\langle i+1,i\rangle})](\hat{c}_{i+1,\sigma}^{\dagger}\hat{c}_{i,\sigma}+\textrm{h.c.})-\mu\sum_{i,\sigma}\hat{n}_{i,\sigma}\\
                           & +\sum_{i}\bigg[\frac{1}{2M}\hat{P}_{\langle i+1,i\rangle}^{2}+\frac{1}{2}M\Omega^{2}\hat{X}_{\langle i+1,i\rangle}^{2}\bigg],
\end{align}
```
such that the phonon modes effectively live on the bonds between orbitals rather than on the orbitals themselves. In practice in the code this is achieved by
defining two phonon modes per orbital and setting the mass of one of the two phonon modes to infinity. In this way, the phonon modes with finite mass
effectively lives on the bond.
The simulation is run with the command
```
> julia -O3 bssh_chain.jl
```

## Bond SSH Chain with MPI

In the script [`examples/bssh_chain_mpi.jl`](https://github.com/SmoQySuite/SmoQyDQMC.jl/blob/main/examples/bssh_chain_mpi.jl) run simulations identical to the previous section,
except we use [`MPI.jl`](https://github.com/JuliaParallel/MPI.jl.git) to run identical simulations in parallel.
The simulation may be run with either the command
```
> mpiexec -n n julia -O3 bssh_chain_mpi.jl
```
or
```
> mpiexecjl -n n julia -O3 bssh_chain_mpi.jl
```
depending on how your system is configured, where `n` specifies the number of simulations/processes that are run in parallel using.
For more information on how to set-up and use MPI on your system refer to the (very good)
documentation for [`MPI.jl`](https://github.com/JuliaParallel/MPI.jl.git).

## Kagome Holstein Model with Density Tuning

In the script [`examples/holstein_kagome.jl`](https://github.com/SmoQySuite/SmoQyDQMC.jl/blob/main/examples/holstein_kagome.jl)
we simulate the kagome lattice Holstein model
```math
\begin{align}
\hat{H}_{\textrm{kag-hol}} = & -t\sum_{\langle i,j\rangle,\sigma}(\hat{c}_{i,\sigma}^{\dagger}\hat{c}_{j,\sigma}+\textrm{h.c.})-\mu\sum_{i,\sigma}\hat{n}_{i,\sigma}\\
                             & +\sum_{i}\bigg[\frac{1}{2M}\hat{P}_{i}^{2}+\frac{1}{2}M\Omega^{2}\hat{X}_{i}^{2}\bigg]+\alpha\sum_{i,\sigma}\hat{X}_{i}(\hat{n}_{i,\sigma}-\tfrac{1}{2}),
\end{align}
```
where the sum over ``\langle i,j \rangle`` runs over all pairs of nearest-neighbor orbitals in the kagome lattice. Additionally, in the simulation the chemical potential ``\mu``
is tuned to achieve a target density of ``\langle n \rangle = 2/3.`` The simulation is run with the command
```
> julia -O3 holstein_kagome.jl
```

## Square Lattice Hubbard Model

In the script [`examples/hubbard_square.jl`](https://github.com/SmoQySuite/SmoQyDQMC.jl/blob/main/examples/hubbard_square.jl)
we simulate the half-filled, particle-hole symmetric repulsive Hubbard model on a square lattice, given by
```math
\begin{align*}
\hat{H}_{\textrm{sq-hub}} = & -t\sum_{\langle i,j\rangle,\sigma}(\hat{c}_{i,\sigma}^{\dagger}\hat{c}_{j,\sigma}+\textrm{h.c.})-\mu\sum_{i,\sigma}\hat{n}_{i,\sigma}\\
                            & +U\sum_{i}(\hat{n}_{i,\uparrow}-\tfrac{1}{2})(\hat{n}_{i,\downarrow}-\tfrac{1}{2}),
\end{align*}
```
where ``\mu = 0``. The simulation is run with the command
```
> julia -O3 hubbard_sqaure.jl
```