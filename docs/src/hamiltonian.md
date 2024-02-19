# Supported Hamiltonians

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
    \hat{\mathcal{K}} = \sum_{\sigma=\uparrow,\downarrow} \left[ \hat{\mathcal{K}}_{\sigma,0} + \hat{\mathcal{K}}_{\sigma,{\rm ssh}} \right],
\end{align*}
```
where
```math
\begin{align*}
    \hat{\mathcal{K}}_{\sigma,0} =& -\sum_{\substack{\mathbf{i},\nu \\ \mathbf{j},\gamma}}
        \left[
            t_{\sigma,(\mathbf{i},\nu),(\mathbf{j},\gamma)} \hat{c}^\dagger_{\sigma,\mathbf{i},\nu}\hat{c}_{\sigma,\mathbf{j},\gamma} + {\rm h.c.}
        \right]
\end{align*}
```
is the non-interacting spin-``\sigma`` electron kinetic energy, and
```math
\begin{align*}
    \hat{\mathcal{K}}_{\sigma,{\rm ssh}} =& \sum_{\substack{\mathbf{i},\nu \\ \mathbf{j},\gamma}}\sum_{\substack{n_{\mathbf{i},\nu} \\ n_{\mathbf{j},\gamma}}}\sum_{m=1}^4
        (\hat{X}_{n_{\mathbf{i},\nu}}-\hat{X}_{n_{\mathbf{j},\gamma}})^m\left[
            \alpha_{\sigma,m,n_{\mathbf{i},\nu},n_{\mathbf{j},\gamma}} \hat{c}^\dagger_{\sigma,\mathbf{i},\nu}\hat{c}_{\sigma,\mathbf{j},\gamma} + {\rm h.c.}
        \right]
\end{align*}
```
is describes the interaction between the lattice degrees of freedom and the spin-``\sigma`` electron kinetic energy via a Su-Schrieffer-Heeger (SSH)-like coupling mechanism. The hopping integral between from orbital ``\gamma`` in unit cell ``\mathbf{j}`` to orbital ``\nu`` in unit cell ``\mathbf{i}`` is given by ``t_{(\mathbf{i},\nu),(\mathbf{j},\gamma)}``, and may in general be complex. The modulations to this hopping integral are controlled by the parameters ``\alpha_{m,(\mathbf{i},\nu),(\mathbf{j},\gamma)}``, where ``m\in [1,4]`` specifies the order of the difference in the phonon positions that modulates the hopping integral.

Lastly, the electron potential energy is broken down into the three terms
```math
\begin{align*}
    \hat{\mathcal{V}} = \sum_{\sigma=\uparrow,\downarrow} \left[ \hat{\mathcal{V}}_{\sigma,0} + \hat{\mathcal{V}}_{\sigma,{\rm hol}} \right] + \hat{\mathcal{V}}_{\rm hub},
\end{align*}
```
where
```math
\begin{align*}
    \hat{\mathcal{V}}_{\sigma,0} =& \sum_{\mathbf{i},\nu}
        \left[
            (\epsilon_{\sigma,\mathbf{i},\nu} - \mu) \hat{n}_{\sigma,\mathbf{i},\nu}
        \right]
\end{align*}
```
is the non-interacting spin-``\sigma`` electron potential energy,
```math
\begin{align*}
    \hat{\mathcal{V}}_{\sigma,{\rm hol}} =&
    \begin{cases}
        \sum_{\mathbf{i},\nu} \sum{\mathbf{j},\gamma} \sum_{n_{\mathbf{i},\nu}} \left[\sum_{m=1,3}\tilde{\alpha}_{\sigma,m,n_{\mathbf{i},\nu},(\mathbf{j},\gamma)} \ \hat{X}^m_{n_{\mathbf{i},\nu}}(\hat{n}_{\sigma,\mathbf{j},\gamma} - \tfrac{1}{2}) + \sum_{m=2,4}\tilde{\alpha}_{\sigma,m,n_{\mathbf{i},\nu},(\mathbf{j},\gamma)} \ \hat{X}^m_{n_{\mathbf{i},\nu}}\hat{n}_{\sigma,\mathbf{j},\gamma}\right] \\
        \sum_{\mathbf{i},\nu} \sum{\mathbf{j},\gamma} \sum_{n_{\mathbf{i},\nu}} \sum_{m=1}^4 \tilde{\alpha}_{\sigma,m,n_{\mathbf{i},\nu},(\mathbf{j},\gamma)} \ \hat{X}^m_{n_{\mathbf{i},\nu}} \hat{n}_{\sigma,\mathbf{j},\gamma}
    \end{cases}
\end{align*}
```
is the contribution to the spin-``\sigma`` electron potential energy that results from a Holstein-like coupling to the lattice degrees of freedom, and
```math
\begin{align*}
    \hat{\mathcal{V}}_{{\rm hub}}=&
    \begin{cases}
        \sum_{\mathbf{i},\nu}U_{\mathbf{i},\nu}\big(\hat{n}_{\uparrow,\mathbf{i},\nu}-\tfrac{1}{2}\big)\big(\hat{n}_{\downarrow,\mathbf{i},\nu}-\tfrac{1}{2}\big)\\
        \sum_{\mathbf{i},\nu}U_{\mathbf{i},\nu}\hat{n}_{\uparrow,\mathbf{i},\nu}\hat{n}_{\downarrow,\mathbf{i},\nu}
    \end{cases}
\end{align*}
```
is the on-site Hubbard interaction contribution to the electron potential energy. In ``\hat{\mathcal{V}}_0`` the chemical potential is given by ``\mu``, and ``\epsilon_{\mathbf{i},\nu}`` is the on-site energy, the parameter ``\tilde{\alpha}_{m,n_{\mathbf{i},\nu},(\mathbf{j},\gamma)}`` controls the strength of the Holstein-like coupling in ``\hat{\mathcal{V}}_{\rm ph}``, and ``U_{\mathbf{i},\nu}`` is the on-site Hubbard interaction strength in ``\hat{\mathcal{V}}_{\rm hub}``.
Note that either functional form for ``\hat{\mathcal{V}}_{\rm hub}`` and ``\hat{\mathcal{V}}_{\sigma, {\rm hub}}`` can be used in the code.
Note that the two possible parameterizations for ``\hat{\mathcal{V}}_{\sigma, {\rm hub}}`` are inequivalent! 