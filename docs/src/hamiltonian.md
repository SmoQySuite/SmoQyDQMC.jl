# Supported Hamiltonians

This section describes the class of Hamiltonians [`SmoQyDQMC.jl`](https://github.com/SmoQySuite/SmoQyDQMC.jl) currently supports,
and how the various terms appearing in the Hamiltonian are parameterized within the code.
We start by partitioning the full Hamiltonian as
```math
\begin{align*}
    \hat{\mathcal{H}} = \hat{\mathcal{U}} + \hat{\mathcal{K}} + \hat{\mathcal{V}},
\end{align*}
```
where ``\hat{\mathcal{U}}`` describes the non-interacting lattice (phonon) degrees of freedom and ``\hat{\mathcal{K}}`` and ``\hat{\mathcal{V}}`` are the total electron kinetic and potential energies, respectively. Note that both ``\hat{\mathcal{K}}`` and ``\hat{\mathcal{V}}`` can depend on the dynamical lattice coordinates, leading to an electron-phonon (``e``-ph) coupling that is either diagonal or off-diagonal in the orbital basis. ``\hat{\mathcal{V}}`` also includes any contributions from local intra- and inter-orbital Hubbard repulsion or extended Hubbard interactions. 

The non-interacting lattice terms are further subdivided into the sum of three terms
```math
\begin{align*}
    \hat{\mathcal{U}} = \hat{\mathcal{U}}_{\rm qho} + \hat{\mathcal{U}}_{\rm anh} + \hat{\mathcal{U}}_{\rm disp}. 
\end{align*}
```
The first term 
```math
\begin{align*}
    \hat{\mathcal{U}}_{\rm qho} =& \sum_{\mathbf{i},\nu}
        \left[
            \frac{1}{2M_{\mathbf{i},\nu}}\hat{P}^2_{\mathbf{i},\nu}
            + \frac{1}{2}M_{\mathbf{i},\nu}\Omega_{0,\mathbf{i},\nu}^2\hat{X}_{\mathbf{i},\nu}^2
        \right]
\end{align*}
```
describes the placement of local quantum harmonic oscillator (QHO) modes in a cluster, i.e. an Einstein solid, while the second term
```math
\begin{align*}
    \hat{\mathcal{U}}_{\rm anh} =& \sum_{\mathbf{i},\nu}
        \left[
            \frac{1}{24}M_{\mathbf{i},\nu}\Omega_{a,\mathbf{i},\nu}^2\hat{X}_{\mathbf{i},\nu}^4
        \right]
\end{align*}
```
introduces anharmonic contributions to the oscillator potential. The third term
```math
\begin{align*}
    \hat{\mathcal{U}}_{\rm disp} =& \sum_{\substack{\mathbf{i},\nu \\ \mathbf{j},\gamma}}
        \frac{M_{\mathbf{i},\alpha}M_{\mathbf{j},\gamma}}{M_{\mathbf{i},\alpha}+M_{\mathbf{j},\gamma}}\left[
            \tilde{\Omega}^2_{0,(\mathbf{i},\alpha),(\mathbf{j},\gamma)}(\hat{X}_{\mathbf{i},\nu}-\zeta_{(\mathbf{i},\alpha),(\mathbf{j},\gamma)}\hat{X}_{\mathbf{j},\gamma})^2
            + \frac{1}{12}\tilde{\Omega}^2_{a,(\mathbf{i},\alpha),(\mathbf{j},\gamma)}(\hat{X}_{\mathbf{i},\nu}-\zeta_{(\mathbf{i},\alpha),(\mathbf{j},\gamma)}\hat{X}_{\mathbf{j},\gamma})^4
        \right]
\end{align*}
```
introduces coupling (or dispersion) between the QHO modes. The sums over ``\mathbf{i} \ (\mathbf{j})`` and ``\nu \ (\gamma)`` run over unit cells in the lattice and phonon modes within each unit cell.

The position and momentum operators for each QHO mode are ``\hat{X}_{\mathbf{i},\nu}`` and ``\hat{P}_{\mathbf{i},\nu}`` respectively, with corresponding phonon mass ``M_{\mathbf{i},\nu}``. The spring constant is ``K_{\mathbf{i},\nu} = M_{\mathbf{i},\nu} \Omega_{0,\mathbf{i},\nu}^2``, with ``\Omega_{0,\mathbf{i},\nu}`` specifying the phonon frequency.  ``\hat{\mathcal{U}}_{\rm anh}`` then introduces an anharmonic ``\hat{X}_{\mathbf{i},\nu}^4`` contribution to the QHO potential energy that is controlled by the parameter ``\Omega_{a,\mathbf{i},\nu}``. Similarly, ``\tilde{\Omega}_{0,(\mathbf{i},\alpha),(\mathbf{j},\gamma)} \ (\tilde{\Omega}_{a,(\mathbf{i},\alpha),(\mathbf{j},\gamma)})`` is the coefficient controlling harmonic (anharmonic) dispersion between QHO modes.
The parameter ``\zeta_{(\mathbf{i},\alpha),(\mathbf{j},\gamma)} = \pm 1`` determines whether the difference or sum appears of pairs of phonon displacements
appears the dispersive coupling terms.
Note that unlike harmonic parameters ``\Omega_{0,\mathbf{i},\nu}`` and ``\tilde{\Omega}_{0,(\mathbf{i},\alpha),(\mathbf{j},\gamma)}``, the anharmonic parameters ``\Omega_{a,\mathbf{i},\nu}`` and ``\tilde{\Omega}_{a,(\mathbf{i},\alpha),(\mathbf{j},\gamma)}`` do not have units of frequency, but instead include an additional factor of inverse length squared.

The electron kinetic energy is conveniently expressed as
```math
\begin{align*}
    \hat{\mathcal{K}} = \hat{\mathcal{K}}_{0} + \hat{\mathcal{K}}_{{\rm ssh}} = \sum_\sigma \hat{\mathcal{K}}_{0,\sigma} + \sum_\sigma\hat{\mathcal{K}}_{{\rm ssh},\sigma}.
\end{align*}
```
The first term describes the non-interacting spin-``\sigma`` electron kinetic energy 
```math
\begin{align*}
    \hat{\mathcal{K}}_{0,\sigma} =& -\sum_{\substack{\mathbf{i},\nu \\ \mathbf{j},\gamma}}
        \left[
            t_{\sigma,(\mathbf{i},\nu),(\mathbf{j},\gamma)} \hat{c}^\dagger_{\sigma,\mathbf{i},\nu}
            \hat{c}^{\phantom{\dagger}}_{\sigma,\mathbf{j},\gamma} + {\rm h.c.}
        \right], 
\end{align*}
```
where ``t_{\sigma,(\mathbf{i},\nu),(\mathbf{j},\gamma)}`` is the spin-``\sigma`` hopping integral from orbital ``\gamma`` in unit cell ``\mathbf{j}`` to orbital ``\nu`` in unit cell ``\mathbf{i}``, and may be real or complex. The second term describes the interaction between the lattice degrees of freedom and the spin-``\sigma`` electron kinetic energy via a Su-Schrieffer-Heeger (SSH)-like coupling mechanism
```math
\begin{align*}
    \hat{\mathcal{K}}_{{\rm ssh},\sigma} =& \sum_{\substack{\mathbf{i},\nu, \eta \\ \mathbf{j},\gamma, \rho}}\sum_{m=1}^4
        (\hat{X}_{\mathbf{i},\nu}-\hat{X}_{\mathbf{j},\gamma})^m\left[ \alpha_{\sigma,m,(\mathbf{i},\nu,\eta),(\mathbf{j},\gamma,\rho)} \hat{c}^\dagger_{\sigma,\mathbf{i},\eta}\hat{c}^{\phantom{\dagger}}_{\sigma,\mathbf{j},\rho} + {\rm h.c.}
        \right]. 
\end{align*}
```
Here, the modulations of the spin-``\sigma`` hopping integrals to ``m``\textsuperscript{th} ``(=1-4)`` order in displacement are controlled by the parameters ``\alpha_{\sigma,m,(\mathbf{i},\nu,\eta),(\mathbf{j},\gamma,\rho)}``. Note that if the corresponding bare hopping amplitude ``t_{\sigma,(\mathbf{i},\eta),(\mathbf{j},\rho)}`` is complex, then the ``\alpha_{\sigma,m,(\mathbf{i},\nu,\eta),(\mathbf{j},\gamma,\rho)}`` parameter is defined to share the same complex phase. This convention ensures that ``e``-ph interaction only modulates the magnitude of the hopping amplitude and not the phase.

Lastly, the electron potential energy is expressed as
```math
\begin{align*}
    \hat{\mathcal{V}} = \hat{\mathcal{V}}_0 + \hat{\mathcal{V}}_{\rm hol} + \hat{\mathcal{V}}_{\rm hub} + \hat{\mathcal{V}}_\text{exh} = \sum_\sigma \hat{\mathcal{V}}_{0,\sigma} + \sum_\sigma \hat{\mathcal{V}}_{{\rm hol},\sigma} + \hat{\mathcal{V}}_{\rm hub}+ \hat{\mathcal{V}}_\text{exh},
\end{align*}
```
where
```math
\begin{align*}
    \hat{\mathcal{V}}_{0,\sigma} =& \sum_{\mathbf{i},\nu}
        \left[
            (\epsilon_{\sigma, \mathbf{i},\nu} - \mu_\sigma) \hat{n}_{\sigma,\mathbf{i},\nu}
        \right]
\end{align*}
```
is the non-interacting spin-``\sigma`` electron potential energy. Here, ``\mu_\sigma`` is the spin-``\sigma`` chemical potential and ``\epsilon_{\sigma, \mathbf{i},\nu}`` is the spin-``\sigma`` on-site energy for orbital ``\nu`` in unit cell ``{\bf i}``. 

The second term 
```math
\begin{align*}
    \hat{\mathcal{V}}_{{\rm hol},\sigma} =&
    \begin{cases}
        \sum_{\substack{\mathbf{i},\nu \\ \mathbf{j},\gamma}} \left[\sum_{m=1,3}\kappa_{\sigma,m,(\mathbf{i},\nu),(\mathbf{j},\gamma)} \ \hat{X}^m_{\mathbf{i},\nu}(\hat{n}_{\sigma,\mathbf{j},\gamma} - \tfrac{1}{2}) + \sum_{m=2,4}\kappa_{\sigma,m,(\mathbf{i},\nu),(\mathbf{j},\gamma)} \ \hat{X}^m_{\mathbf{i},\nu}\hat{n}_{\sigma,\mathbf{j},\gamma}\right] \\
        \sum_{\substack{\mathbf{i},\nu \\ \mathbf{j},\gamma}}\sum_{m=1}^4 \kappa_{\sigma,m,(\mathbf{i},\nu),(\mathbf{j},\gamma)} \ \hat{X}^m_{\mathbf{i},\nu} \hat{n}_{\sigma,\mathbf{j},\gamma}
    \end{cases}
\end{align*}
```
is the contribution to the spin-``\sigma`` electron potential energy that results from a Holstein- or Fr{\"o}hlich-like coupling to the lattice degrees of freedom. The parameter ``\kappa_{\sigma,m,(\mathbf{i},\nu),(\mathbf{j},\gamma)}`` controls the strength of this coupling in the ``\hat{\mathcal{V}}_{\text{hol},\sigma}`` term. It is important to note that the two parametrizations that are available in [`SmoQyDQMC.jl`](https://github.com/SmoQySuite/SmoQyDQMC.jl)  are inequivalent, with the first being particle-hole symmetric in the atomic limit.

The third term
```math
\begin{align*}
    \hat{\mathcal{V}}_{{\rm hub}}=&
    \begin{cases}
        \sum_{\mathbf{i},\nu}U_{\mathbf{i},\nu}\big(\hat{n}_{\uparrow,\mathbf{i},\nu}-\tfrac{1}{2}\big)\big(\hat{n}_{\downarrow,\mathbf{i},\nu}-\tfrac{1}{2}\big)\\
        \sum_{\mathbf{i},\nu}U_{\mathbf{i},\nu}\hat{n}_{\uparrow,\mathbf{i},\nu}\hat{n}_{\downarrow,\mathbf{i},\nu}
    \end{cases}
\end{align*}
```
defines the intra-orbtial/local Hubbard interaction, where ``U_{\mathbf{i},\nu}`` is the interaction strength. Note that [`SmoQyDQMC.jl`](https://github.com/SmoQySuite/SmoQyDQMC.jl)  allows the user to parameterize the Hubbard interaction using either functional form for ``\hat {\mathcal V}_{\rm hub}``. 
The top-most is particle-hole symmetric and is often useful at half-filling.

Lastly, the fourth term
```math
\begin{equation*}
    \begin{aligned}
        \hat{\mathcal{V}}_\text{exh} & =
            \begin{cases}
                \sum_{\substack{\mathbf{i},\nu,\sigma \\ \mathbf{j},\gamma,\sigma'}} V_{(\mathbf{i},\nu),(\mathbf{j},\gamma)}\big(\hat{n}_{\sigma,\mathbf{i},\nu}-\frac{1}{2}\big)\big(\hat{n}_{\sigma',\mathbf{j},\gamma}-\frac{1}{2}\big)\\
                \sum_{\substack{\mathbf{i},\nu,\sigma \\ \mathbf{j},\gamma,\sigma'}} V_{(\mathbf{i},\nu),(\mathbf{j},\gamma)}\hat{n}_{\sigma,\mathbf{i},\nu}\hat{n}_{\sigma',\mathbf{j},\gamma}
            \end{cases}\\
    \end{aligned}
\end{equation*}
```
introduces extended Hubbard interactions with ``V_{(\mathbf{i},\nu),(\mathbf{j},\gamma)}`` subject to the constraint ``V_{(\mathbf{i},\nu),(\mathbf{i},\nu)} = 0``. Note, however, that local inter-orbital Hubbard interactions can still be treated by defining an interaction within a single unit cell ``\mathbf{i} = \mathbf{j}`` between a pair of orbitals ``\nu \ne \gamma`` that share the same basis vector
``\mathbf{r}_\nu = \mathbf{r}_\gamma``. In this case, the parameter ``V_{(\mathbf{i},\nu),(\mathbf{j},\gamma)}`` is typically denoted ``U_{\mathbf{i}, \nu, \gamma}``. 
As with the local Hubbard interaction, users can parameterize the extended Hubbard interaction using either a particle-hole symmetric (top) or asymmetric (bottom) form.