@doc raw"""
    PhononMode{E<:AbstractFloat}

Defines a phonon mode ``\nu`` at location `\mathbf{r}_\nu` in the unit cell. Specifically, it defines the phonon Hamiltonian terms
```math
\hat{H}_{{\rm ph}} = \sum_{\mathbf{i}}
  \left[
      \frac{1}{2} M_{\mathbf{i},\nu}\Omega_{\mathbf{i},\nu}^{2}\hat{X}_{\mathbf{i},\nu}^{2}
    + \frac{1}{12}M_{\mathbf{i},\nu}\Omega_{4,\mathbf{i},\nu}^{2}\hat{X}_{\mathbf{i},\nu}^{4}
    + \frac{1}{2M_{\mathbf{i},\nu}}\hat{P}_{\mathbf{i},\nu}^{2}
  \right],
```
where the sum runs over unit cell ``\mathbf{i}``, ``\mathbf{r}_\nu`` denotes the location of the phonon mode in the unit cell,
``M_{\mathbf{i},\nu}`` is the phonon mass `M`, ``\Omega_{\mathbf{i},\nu}`` is the phonon frequency that is distributed according
to a normal distribution with mean `Ω_mean` and standard deviation `Ω_std`. Lastly, ``\Omega_{4,\mathbf{i},\nu}`` is the anharmonic
coefficient, and is distributed according to a normal distribution with mean `Ω4_mean` and standard deviation `Ω4_std`.

# Fields

- `basis_vec::SVector{D,E}`: Location ``\mathbf{r}_\nu`` of phonon mode in unit cell.
- `M::E`:: The phonon mass ``M_{\mathbf{i},\nu}.``
- `Ω_mean::E`: Mean of normal distribution the phonon frequency ``\Omega_{\mathbf{i},\nu}`` is sampled from.
- `Ω_std::E`: Standard deviation of normal distribution the phonon frequency ``\Omega_{\mathbf{i},\nu}`` is sampled from.
- `Ω4_mean::E`: Mean of normal distribution the anharmonic coefficient ``\Omega_{4,\mathbf{i},\nu}`` is sampled from.
- `Ω4_std::E`: Standard deviation of normal distribution the anharmonic coefficient ``\Omega_{4,\mathbf{i},\nu}`` is sampled from.
"""
struct PhononMode{E<:AbstractFloat, D}

    # basis vector associated with phonon mode
    basis_vec::SVector{D,E}

    # phonon mass
    M::E

    # mean phonon frequency
    Ω_mean::E

    # standard deviation of phonon frequency
    Ω_std::E

    # mean anharmonic coefficient
    Ω4_mean::E

    # standard deviation of anharmonic coefficient
    Ω4_std::E
end

@doc raw"""
    PhononMode(;
        # KEYWORD ARGUMENTS
        basis_vec::AbstractVector{E},
        Ω_mean::E,
        Ω_std::E = 0.,
        M::E = 1.,
        Ω4_mean::E = 0.0,
        Ω4_std::E = 0.0,
    ) where {E<:AbstractFloat}

Initialize and return a instance of [`PhononMode`](@ref).
"""
function PhononMode(;
    # KEYWORD ARGUMENTS
    basis_vec::AbstractVector{E},
    Ω_mean::E,
    Ω_std::E = 0.,
    M::E = 1.,
    Ω4_mean::E = 0.0,
    Ω4_std::E = 0.0,
) where {E<:AbstractFloat}

    D = length(basis_vec)
    r = SVector{D,E}(basis_vec)
    return PhononMode{E,D}(r, M, Ω_mean, Ω_std, Ω4_mean, Ω4_std)
end


@doc raw"""
    HolsteinCoupling{E<:AbstractFloat, D}

Defines a Holstein coupling between a specified phonon mode and orbital density.
Specifically, if `ph_sym_form = true` then a the particle-hole symmetric form of the Holstein coupling given by
```math
\begin{align*}
H = \sum_{\mathbf{i}} \Big[ 
        & (\alpha_{\mathbf{i},(\mathbf{r},\kappa,\nu)} \hat{X}_{\mathbf{i},\nu}
        + \alpha_{3,\mathbf{i},(\mathbf{r},\kappa,\nu)} \hat{X}^3_{\mathbf{i},\nu}) \ (\hat{n}_{\sigma,\mathbf{i}+\mathbf{r},\kappa}-\tfrac{1}{2})\\
        & + (\alpha_{2,\mathbf{i},(\mathbf{r},\kappa,\nu)} \hat{X}^2_{\mathbf{i},\nu}
        + \alpha_{4,\mathbf{i},(\mathbf{r},\kappa,\nu)} \hat{X}^4_{\mathbf{i},\nu}) \ \hat{n}_{\sigma,\mathbf{i}+\mathbf{r},\kappa} 
\Big]
\end{align*},
```
is used, whereas if `ph_sym_form = false` Holstein coupling is given by
```math
\begin{align*}
H = \sum_{\mathbf{i}} \Big[ 
        & (\alpha_{\mathbf{i},(\mathbf{r},\kappa,\nu)} \hat{X}_{\mathbf{i},\nu}
        + \alpha_{3,\mathbf{i},(\mathbf{r},\kappa,\nu)} \hat{X}^3_{\mathbf{i},\nu}) \ \hat{n}_{\sigma,\mathbf{i}+\mathbf{r},\kappa}\\
        & + (\alpha_{2,\mathbf{i},(\mathbf{r},\kappa,\nu)} \hat{X}^2_{\mathbf{i},\nu}
        + \alpha_{4,\mathbf{i},(\mathbf{r},\kappa,\nu)} \hat{X}^4_{\mathbf{i},\nu}) \ \hat{n}_{\sigma,\mathbf{i}+\mathbf{r},\kappa} 
\Big]
\end{align*}.
```
In the above, ``\sigma`` specifies the sum, and the sum over ``\mathbf{i}`` runs over unit cells in the lattice.
In the above ``\nu`` and ``\kappa`` specify the phonon mode orbital species IDs respectively, and ``\mathbf{r}`` is a static
displacement in unit cells.

# Fields

- `ph_sym_form::Bool`: If particle-hole symmetric form is used for Holstein coupling.
- `phonon_id::Int`: The ID ``\nu`` specifying phonon mode getting coupled to.
- `orbital_id::Int`: The ID ``\kappa`` specifying orbital species the phonon mode getting coupled to.
- `displacement::SVector{D,Int}`: Static displacement ``r`` in unit cells in the direction of each lattice vector.
- `α_mean::E`: Mean of the linear Holstein coupling coefficient ``\alpha_{1,\mathbf{i},(\mathbf{r},\kappa,\nu)}.``
- `α_std::E`: Standard deviation of the linear Holstein coupling coefficient ``\alpha_{1,\mathbf{i},(\mathbf{r},\kappa,\nu)}.``
- `α2_mean::E`: Mean of the squared Holstein coupling coefficient ``\alpha_{2,\mathbf{i},(\mathbf{r},\kappa,\nu)}.``
- `α2_std::E`: Standard deviation of the squared Holstein coupling coefficient ``\alpha_{2,\mathbf{i},(\mathbf{r},\kappa,\nu)}.``
- `α3_mean::E`: Mean of the cubic Holstein coupling coefficient ``\alpha_{3,\mathbf{i},(\mathbf{r},\kappa,\nu)}.``
- `α3_std::E`: Standard deviation of the cubic Holstein coupling coefficient ``\alpha_{3,\mathbf{i},(\mathbf{r},\kappa,\nu)}.``
- `α4_mean::E`: Mean of the quartic Holstein coupling coefficient ``\alpha_{4,\mathbf{i},(\mathbf{r},\kappa,\nu)}.``
- `α4_std::E`: Standard deviation of the quartic Holstein coupling coefficient ``\alpha_{4,\mathbf{i},(\mathbf{r},\kappa,\nu)}.``

# Comment

Note that the initial orbital `bond.orbital[1]` must match the orbital species associated with phonon mode [`PhononMode`](@ref) getting coupled to.
"""
struct HolsteinCoupling{E<:AbstractFloat, D}

    # phonon id
    phonon_id::Int

    # orbital id
    orbital_id::Int

    # displacement in unit cells
    displacement::SVector{D,Int}

    # mean linear (X) coupling coefficient
    α_mean::E

    # standard deviation of coupling coefficient
    α_std::E

    # mean squared (X²) coupling coefficient
    α2_mean::E

    # standard deviation of squared (X²) coupling coefficient
    α2_std::E

    # mean cubic (X³) coupling coefficient
    α3_mean::E

    # standard deviation of cubic (X³) coupling coefficient
    α3_std::E

    # mean quartic (X⁴) coupling coefficient
    α4_mean::E

    # standard deviation of quartic (X⁴) coupling coefficient
    α4_std::E

    # whether particle-hole symmetric form is used
    ph_sym_form::Bool
end

@doc raw"""
    HolsteinCoupling(;
        # KEYWORD ARGUMENTS
        model_geometry::ModelGeometry{D,E},
        phonon_id::Int,
        orbital_id::Int,
        displacement::AbstractVector{Int},
        α_mean::E,
        α_std::E  = 0.0,
        α2_mean::E = 0.0,
        α2_std::E = 0.0,
        α3_mean::E = 0.0,
        α3_std::E = 0.0,
        α4_mean::E = 0.0,
        α4_std::E = 0.0,
        ph_sym_form::Bool = true
    ) where {D, E<:AbstractFloat}

Initialize and return a instance of [`HolsteinCoupling`](@ref).
"""
function HolsteinCoupling(;
    # KEYWORD ARGUMENTS
    model_geometry::ModelGeometry{D,E},
    phonon_id::Int,
    orbital_id::Int,
    displacement::AbstractVector{Int},
    α_mean::E,
    α_std::E  = 0.0,
    α2_mean::E = 0.0,
    α2_std::E = 0.0,
    α3_mean::E = 0.0,
    α3_std::E = 0.0,
    α4_mean::E = 0.0,
    α4_std::E = 0.0,
    ph_sym_form::Bool = true
) where {D, E<:AbstractFloat}

    r = SVector{D,Int}(displacement)
    return HolsteinCoupling(phonon_id, orbital_id, r, α_mean, α_std, α2_mean, α2_std, α3_mean, α3_std, α4_mean, α4_std, ph_sym_form)
end


@doc raw"""
    SSHCoupling{T<:Number, E<:AbstractFloat, D}

Defines a Su-Schrieffer-Heeger (SSH) coupling between a pair of phonon modes.
Specifically, it defines the SSH interaction term
```math
\hat{H}_{{\rm ssh}} = -\sum_{\sigma,\mathbf{i}}
    \left[ t_{\mathbf{i},(\mathbf{r},\kappa,\nu)} - \left( \sum_{n=1}^{4}\alpha_{n,\mathbf{i},(\mathbf{r},\kappa',\nu')}
    \left( \hat{X}_{\mathbf{i}+\mathbf{r},\kappa'} - \hat{X}_{\mathbf{i},\nu'}\right)^{n}\right) \right]
    \left( \hat{c}_{\sigma,\mathbf{i}+\mathbf{r},\kappa}^{\dagger}\hat{c}_{\sigma,\mathbf{i},\nu}+{\rm h.c.} \right),
```
where ``\sigma`` specifies the sum, and the sum over ``\mathbf{i}`` runs over unit cells in the lattice.
In the above ``\nu`` and ``\kappa`` IDs specify orbital species in the unit cell,
and ``\kappa'`` and ``\nu'`` IDs specify the phonon modes getting coupled to.
Finally, ``\mathbf{r}`` is a static displacement in unit cells in the direction of each lattice vector.
In that above expression ``t_{\mathbf{i},(\mathbf{r},\kappa,\nu)}`` is the bare hopping amplitude, which is not specified here.

# Fields

- `phonon_ids::NTuple{2,Int}`: Pair of phonon modes getting coupled together.
- `bond::Bond{D}`: Bond seperating the two orbitals getting coupled to, which are seperated by ``\mathbf{r} + (\mathbf{r}_\kappa - \mathbf{r}_\nu)``.
- `bond_id::Int`: Bond ID associated with the `bond` field.
- `α_mean::T`: Mean of the linear SSH coupling constant ``\alpha_{1,\mathbf{i},(\mathbf{r},\kappa,\nu)}.``
- `α_std::T`: Standard deviation of the linear SSH coupling constant ``\alpha_{1,\mathbf{i},(\mathbf{r},\kappa,\nu)}.``
- `α2_mean::T`: Mean of the quadratic SSH coupling constant ``\alpha_{2,\mathbf{i},(\mathbf{r},\kappa,\nu)}.``
- `α2_std::T`: Standard deviation of the quadratic SSH coupling constant ``\alpha_{2,\mathbf{i},(\mathbf{r},\kappa,\nu)}.``
- `α3_mean::T`: Mean of the cubic SSH coupling constant ``\alpha_{3,\mathbf{i},(\mathbf{r},\kappa,\nu)}.``
- `α3_std::T`: Standard deviation of the cubic SSH coupling constant ``\alpha_{3,\mathbf{i},(\mathbf{r},\kappa,\nu)}.``
- `α4_mean::T`: Mean of the quartic SSH coupling constant ``\alpha_{4,\mathbf{i},(\mathbf{r},\kappa,\nu)}.``
- `α4_std::T`: Standard deviation of the quartic SSH coupling constant ``\alpha_{4,\mathbf{i},(\mathbf{r},\kappa,\nu)}.``
- `expniϕ::T`: Twisted boundary conditions phase factor.
"""
struct SSHCoupling{T<:Number, E<:AbstractFloat, D}

    # phonon modes getting coupled
    phonon_ids::NTuple{2,Int}

    # bond/hopping associated with bond
    bond::Bond{D}

    # bond ID corresponding to bond above
    bond_id::Int

    # mean linear ssh coupling
    α_mean::T

    # standard deviation of linear ssh coupling
    α_std::E

    # mean of squared ssh coupling
    α2_mean::T

    # standard deviation of squared ssh coupling
    α2_std::E

    # mean cubic ssh coupling
    α3_mean::T

    # standard deviation of cubic ssh coupling
    α3_std::E

    # mean quartic ssh coupling
    α4_mean::T

    # standard deviation of quartic ssh coupling
    α4_std::E

    # twist phase
    expniϕ::T
end

@doc raw"""
    SSHCoupling(;
        # KEYWORD ARGUMENTS
        model_geometry::ModelGeometry{D,E},
        tight_binding_model::TightBindingModel{T,E,D},
        phonon_ids::NTuple{2,Int},
        bond::Bond{D},
        α_mean::Union{T,E},
        α_std::E  = 0.0,
        α2_mean::Union{T,E} = 0.0,
        α2_std::E = 0.0,
        α3_mean::Union{T,E} = 0.0,
        α3_std::E = 0.0,
        α4_mean::Union{T,E} = 0.0,
        α4_std::E = 0.0
    ) where {D, T<:Number, E<:AbstractFloat}

Initialize and return a instance of [`SSHCoupling`](@ref).
"""
function SSHCoupling(;
    # KEYWORD ARGUMENTS
    model_geometry::ModelGeometry{D,E},
    tight_binding_model::TightBindingModel{T,E,D},
    phonon_ids::NTuple{2,Int},
    bond::Bond{D},
    α_mean::Union{T,E},
    α_std::E  = 0.0,
    α2_mean::Union{T,E} = 0.0,
    α2_std::E = 0.0,
    α3_mean::Union{T,E} = 0.0,
    α3_std::E = 0.0,
    α4_mean::Union{T,E} = 0.0,
    α4_std::E = 0.0
) where {D, T<:Number, E<:AbstractFloat}

    # make sure there is already a hopping definition for the tight binding model corresponding to the ssh coupling
    @assert bond in tight_binding_model.t_bonds

    # get the bond ID
    bond_id = add_bond!(model_geometry, bond)

    # get the hopping ID associated with SSH coupling
    hopping_id = findfirst(b -> b == bond, tight_binding_model.t_bonds)

    # get the twist-angle phase factor associatedw with the hopping
    expniϕ = tight_binding_model.expniϕ[hopping_id]

    # determine the type of the hopping
    H = isa(expniϕ, Complex) ? Complex{E} : T

    return SSHCoupling{H,E,D}(
        phonon_ids, bond, bond_id,
        H(α_mean),  E(α_std),
        H(α2_mean), E(α2_std),
        H(α3_mean), E(α3_std),
        H(α4_mean), E(α4_std),
        H(expniϕ)
    )
end


@doc raw"""
    PhononDispersion{E<:AbstractFloat, D}

Defines a dispersive phonon coupling between phonon modes. Specifically, it defines the dispersive phonon term
```math
\hat{H}_{{\rm disp}} = \sum_{\mathbf{i}}
    \left(
        \frac{M_{\mathbf{i}+\mathbf{r},\kappa}M_{\mathbf{i},\nu}}{M_{\mathbf{i}+\mathbf{r},\kappa}+M_{\mathbf{i},\nu}}
    \right)
    \bigg[
                    \Omega_{\mathbf{i},(\mathbf{r},\kappa,\nu)}^{2}\Big(\hat{X}_{\mathbf{i}+\mathbf{r},\kappa}-\hat{X}_{\mathbf{i},\nu}\Big)^{2}
       +\frac{1}{12}\Omega_{4,\mathbf{i},(\mathbf{r},\kappa,\nu)}^{2}\Big(\hat{X}_{\mathbf{i}+\mathbf{r},\kappa}-\hat{X}_{\mathbf{i},\nu}\Big)^{4}
    \bigg]
```
where the sum over ``\mathbf{i}`` runs over unit cells in the lattice.
In the above ``\nu`` and ``\kappa`` IDs specify the phonon modes in the unit cell, and ``\mathbf{r}`` is a static displacement in unit cells.

# Fields

- `phonon_ids::NTuple{2,Int}`: ID's for pair of phonon modes getting coupled together.
- `displacement::SVector{D,Int}`: Static displacement ``\mathbf{r}`` in unit cells separating the two phonon modes getting coupled.
- `Ω_mean::E`: Mean dispersive phonon frequency ``\Omega_{\mathbf{i},(\mathbf{r},\kappa,\nu)}.``
- `Ω_std::E`: Standard deviation of dispersive phonon frequency ``\Omega_{\mathbf{i},(\mathbf{r},\kappa,\nu)}.``
- `Ω4_mean::E`: Mean quartic dispersive phonon coefficient ``\Omega_{4,\mathbf{i},(\mathbf{r},\kappa,\nu)}.``
- `Ω4_std::E`: Standard deviation of quartic dispersive phonon coefficient ``\Omega_{4,\mathbf{i},(\mathbf{r},\kappa,\nu)}.``
"""
struct PhononDispersion{E<:AbstractFloat, D}

    # pair of phonon modes getting coupled
    phonon_ids::NTuple{2,Int}

    # static displacment in unit cells
    displacement::SVector{D,Int}

    # mean harmonic frequency
    Ω_mean::E

    # standard deviation of harmonic frequency
    Ω_std::E

    # mean of quartic coefficient
    Ω4_mean::E

    # standard deviation of quartic coefficient
    Ω4_std::E
end

@doc raw"""
    PhononDispersion(;
        # KEYWORD ARGUMENTS
        model_geometry::ModelGeometry{D,E},
        phonon_ids::NTuple{2,Int},
        displacement::AbstractVector{Int},
        Ω_mean::E,
        Ω_std::E=0.0,
        Ω4_mean::E=0.0,
        Ω4_std::E=0.0
    ) where {E<:AbstractFloat, D}

Initialize and return a instance of [`PhononDispersion`](@ref).
"""
function PhononDispersion(;
    # KEYWORD ARGUMENTS
    model_geometry::ModelGeometry{D,E},
    phonon_ids::NTuple{2,Int},
    displacement::AbstractVector{Int},
    Ω_mean::E,
    Ω_std::E=0.0,
    Ω4_mean::E=0.0,
    Ω4_std::E=0.0
) where {E<:AbstractFloat, D}

    r = SVector{D,Int}(displacement)
    return PhononDispersion(phonon_ids, r, Ω_mean, Ω_std, Ω4_mean, Ω4_std)
end


@doc raw"""
    ElectronPhononModel{T<:Number, E<:AbstractFloat, D}

Defines an electron-phonon model.

# Fields

- `phonon_modes::Vector{PhononModes{E,D}}`: A vector of [`PhononMode`](@ref) definitions.
- `phonon_dispersions::Vector{PhononDispersion{E,D}}`: A vector of [`PhononDispersion`](@ref) defintions.
- `holstein_couplings_up::Vector{HolsteinCoupling{E,D}}`: A vector of [`HolsteinCoupling`](@ref) definitions for spin-up.
- `holstein_couplings_dn::Vector{HolsteinCoupling{E,D}}`: A vector of [`HolsteinCoupling`](@ref) definitions for spin-down.
- `ssh_couplings_up::Vector{SSHCoupling{T,E,D}}`: A vector of [`SSHCoupling`](@ref) defintions for spin-up.
- `ssh_couplings_dn::Vector{SSHCoupling{T,E,D}}`: A vector of [`SSHCoupling`](@ref) defintions for spin-down.
"""
struct ElectronPhononModel{T<:Number, E<:AbstractFloat, D}
    
    # phonon modes
    phonon_modes::Vector{PhononMode{E,D}}

    # phonon dispersion
    phonon_dispersions::Vector{PhononDispersion{E,D}}

    # holstein couplings for spin up
    holstein_couplings_up::Vector{HolsteinCoupling{E,D}}

    # holstein couplings for spin down
    holstein_couplings_dn::Vector{HolsteinCoupling{E,D}}

    # ssh couplings
    ssh_couplings_up::Vector{SSHCoupling{T,E,D}}

    # ssh couplings
    ssh_couplings_dn::Vector{SSHCoupling{T,E,D}}
end

@doc raw"""
    ElectronPhononModel(;
        # KEYWORD ARGUMENTS
        model_geometry::ModelGeometry{D,E},
        tight_binding_model::Union{TightBindingModel{T,E,D}, Nothing} = nothing,
        tight_binding_model_up::Union{TightBindingModel{T,E,D}, Nothing} = nothing,
        tight_binding_model_dn::Union{TightBindingModel{T,E,D}, Nothing} = nothing
    ) where {T<:Number, E<:AbstractFloat, D}

Initialize and return a null (empty) instance of [`ElectronPhononModel`](@ref).
Note that either `tight_binding_model` or `tight_binding_model_up` and `tight_binding_model_dn`
needs to be specified.
"""
function ElectronPhononModel(;
    # KEYWORD ARGUMENTS
    model_geometry::ModelGeometry{D,E},
    tight_binding_model::Union{TightBindingModel{T,E,D}, Nothing} = nothing,
    tight_binding_model_up::Union{TightBindingModel{T,E,D}, Nothing} = nothing,
    tight_binding_model_dn::Union{TightBindingModel{T,E,D}, Nothing} = nothing
) where {T<:Number, E<:AbstractFloat, D}

    if isnothing(tight_binding_model) && isnothing(tight_binding_model_up) && isnothing(tight_binding_model_dn)
        error("Tight Binding Model Improperly Specified.")
    end

    phonon_modes = PhononMode{E,D}[]
    phonon_dispersions = PhononDispersion{E,D}[]
    holstein_couplings_up = HolsteinCoupling{E,D}[]
    holstein_couplings_dn = HolsteinCoupling{E,D}[]
    ssh_coupldings_up = SSHCoupling{T,E,D}[]
    ssh_coupldings_dn = SSHCoupling{T,E,D}[]

    return ElectronPhononModel(
        phonon_modes,
        phonon_dispersions,
        holstein_couplings_up, holstein_couplings_dn,
        ssh_coupldings_up, ssh_coupldings_dn
    )
end


# print struct info in TOML format
function Base.show(io::IO, ::MIME"text/plain", elphm::ElectronPhononModel{T,E,D}) where {T<:AbstractFloat,E,D}

    @printf io "[ElectronPhononModel]\n\n"
    for (i, phonon_mode) in enumerate(elphm.phonon_modes)
        @printf io "[[ElectronPhononModel.PhononMode]]\n\n"
        @printf io "PHONON_ID    = %d\n" i
        @printf io "basis_vec = [%s]\n" string(round.(phonon_mode.basis_vec, digits=6))
        if isfinite(phonon_mode.M)
            @printf io "mass         = %.6f\n" phonon_mode.M
        else
            @printf io "mass         = inf\n"
        end
        @printf io "omega_mean   = %.6f\n" phonon_mode.Ω_mean
        @printf io "omega_std    = %.6f\n" phonon_mode.Ω_std
        @printf io "omega_4_mean = %.6f\n" phonon_mode.Ω4_mean
        @printf io "omega_4_std  = %.6f\n\n" phonon_mode.Ω4_std
    end
    for (i, dispersion) in enumerate(elphm.phonon_dispersions)
        @printf io "[[ElectronPhononModel.PhononDispersion]]\n\n"
        @printf io "DISPERSION_ID = %d\n" i
        @printf io "PHONON_IDS    = [%d, %d]\n" dispersion.phonon_ids[1] dispersion.phonon_ids[2]
        @printf io "displacement  = %s\n" string(dispersion.displacement)
        @printf io "omega_mean    = %.6f\n" dispersion.Ω_mean
        @printf io "omega_std     = %.6f\n" dispersion.Ω_std
        @printf io "omega_4_mean  = %.6f\n" dispersion.Ω4_mean
        @printf io "omega_4_std   = %.6f\n\n" dispersion.Ω4_std
    end
    for i in eachindex(elphm.holstein_couplings_up)

        holstein_coupling_up = elphm.holstein_couplings_up[i]
        @printf io "[[ElectronPhononModel.HolsteinCouplingUp]]\n\n"
        @printf io "HOLSTEIN_ID     = %d\n" i
        @printf io "PHONON_ID       = %d\n" holstein_coupling_up.phonon_id
        @printf io "ORBITAL_ID      = %d\n" holstein_coupling_up.orbital_id
        @printf io "displacement    = %s\n" string(holstein_coupling_up.displacement)
        @printf io "alpha_mean      = %.6f\n" holstein_coupling_up.α_mean
        @printf io "alpha_std       = %.6f\n" holstein_coupling_up.α_std
        @printf io "alpha2_mean     = %.6f\n" holstein_coupling_up.α2_mean
        @printf io "alpha2_std      = %.6f\n" holstein_coupling_up.α2_std
        @printf io "alpha3_mean     = %.6f\n" holstein_coupling_up.α3_mean
        @printf io "alpha3_std      = %.6f\n" holstein_coupling_up.α3_std
        @printf io "alpha4_mean     = %.6f\n" holstein_coupling_up.α4_mean
        @printf io "alpha4_std      = %.6f\n\n" holstein_coupling_up.α4_std

        holstein_coupling_dn = elphm.holstein_couplings_dn[i]
        @printf io "[[ElectronPhononModel.HolsteinCouplingDown]]\n\n"
        @printf io "HOLSTEIN_ID     = %d\n" i
        @printf io "PHONON_ID       = %d\n" holstein_coupling_dn.phonon_id
        @printf io "ORBITAL_ID      = %d\n" holstein_coupling_dn.orbital_id
        @printf io "displacement    = %s\n" string(holstein_coupling_dn.displacement)
        @printf io "alpha_mean      = %.6f\n" holstein_coupling_dn.α_mean
        @printf io "alpha_std       = %.6f\n" holstein_coupling_dn.α_std
        @printf io "alpha2_mean     = %.6f\n" holstein_coupling_dn.α2_mean
        @printf io "alpha2_std      = %.6f\n" holstein_coupling_dn.α2_std
        @printf io "alpha3_mean     = %.6f\n" holstein_coupling_dn.α3_mean
        @printf io "alpha3_std      = %.6f\n" holstein_coupling_dn.α3_std
        @printf io "alpha4_mean     = %.6f\n" holstein_coupling_dn.α4_mean
        @printf io "alpha4_std      = %.6f\n\n" holstein_coupling_dn.α4_std
    end
    for i in eachindex(elphm.ssh_couplings_up)

        ssh_coupling_up = elphm.ssh_couplings_up[i]
        bond = ssh_coupling_up.bond
        @printf io "[[ElectronPhononModel.SSHCouplingUp]]\n\n"
        @printf io "SSH_ID       = %d\n" i
        @printf io "PHONON_IDS   = [%d, %d]\n" ssh_coupling_up.phonon_ids[1] ssh_coupling_up.phonon_ids[2]
        @printf io "BOND_ID      = %d\n" ssh_coupling_up.bond_id
        @printf io "orbitals     = [%d, %d]\n" bond.orbitals[1] bond.orbitals[2]
        @printf io "displacement = %s\n" string(bond.displacement)
        @printf io "alpha_mean   = %.6f\n" ssh_coupling_up.α_mean
        @printf io "alpha_std    = %.6f\n" ssh_coupling_up.α_std
        @printf io "alpha2_mean  = %.6f\n" ssh_coupling_up.α2_mean
        @printf io "alpha2_std   = %.6f\n" ssh_coupling_up.α2_std
        @printf io "alpha3_mean  = %.6f\n" ssh_coupling_up.α3_mean
        @printf io "alpha3_std   = %.6f\n" ssh_coupling_up.α3_std
        @printf io "alpha4_mean  = %.6f\n" ssh_coupling_up.α4_mean
        @printf io "alpha4_std   = %.6f\n\n" ssh_coupling_up.α4_std

        ssh_coupling_dn = elphm.ssh_couplings_dn[i]
        bond = ssh_coupling_dn.bond
        @printf io "[[ElectronPhononModel.SSHCouplingDown]]\n\n"
        @printf io "SSH_ID       = %d\n" i
        @printf io "PHONON_IDS   = [%d, %d]\n" ssh_coupling_dn.phonon_ids[1] ssh_coupling_dn.phonon_ids[2]
        @printf io "BOND_ID      = %d\n" ssh_coupling_dn.bond_id
        @printf io "orbitals     = [%d, %d]\n" bond.orbitals[1] bond.orbitals[2]
        @printf io "displacement = %s\n" string(bond.displacement)
        @printf io "alpha_mean   = %.6f\n" ssh_coupling_dn.α_mean
        @printf io "alpha_std    = %.6f\n" ssh_coupling_dn.α_std
        @printf io "alpha2_mean  = %.6f\n" ssh_coupling_dn.α2_mean
        @printf io "alpha2_std   = %.6f\n" ssh_coupling_dn.α2_std
        @printf io "alpha3_mean  = %.6f\n" ssh_coupling_dn.α3_mean
        @printf io "alpha3_std   = %.6f\n" ssh_coupling_dn.α3_std
        @printf io "alpha4_mean  = %.6f\n" ssh_coupling_dn.α4_mean
        @printf io "alpha4_std   = %.6f\n\n" ssh_coupling_dn.α4_std
    end

    return nothing
end

# print struct info in TOML format
function Base.show(io::IO, ::MIME"text/plain", elphm::ElectronPhononModel{T,E,D}) where {T<:Complex,E,D}

    @printf io "[ElectronPhononModel]\n\n"
    for (i, phonon_mode) in enumerate(elphm.phonon_modes)
        @printf io "[[ElectronPhononModel.PhononMode]]\n\n"
        @printf io "PHONON_ID    = %d\n" i
        @printf io "basis_vec = [%s]\n" string(round.(phonon_mode.basis_vec, digits=6))
        if isfinite(phonon_mode.M)
            @printf io "mass         = %.6f\n" phonon_mode.M
        else
            @printf io "mass         = inf\n"
        end
        @printf io "omega_mean   = %.6f\n" phonon_mode.Ω_mean
        @printf io "omega_std    = %.6f\n" phonon_mode.Ω_std
        @printf io "omega_4_mean = %.6f\n" phonon_mode.Ω4_mean
        @printf io "omega_4_std  = %.6f\n\n" phonon_mode.Ω4_std
    end
    for (i, dispersion) in enumerate(elphm.phonon_dispersions)
        @printf io "[[ElectronPhononModel.PhononDispersion]]\n\n"
        @printf io "DISPERSION_ID = %d\n" i
        @printf io "PHONON_IDS    = [%d, %d]\n" dispersion.phonon_ids[1] dispersion.phonon_ids[2]
        @printf io "displacement  = %s\n" string(dispersion.displacement)
        @printf io "omega_mean    = %.6f\n" dispersion.Ω_mean
        @printf io "omega_std     = %.6f\n" dispersion.Ω_std
        @printf io "omega_4_mean  = %.6f\n" dispersion.Ω4_mean
        @printf io "omega_4_std   = %.6f\n\n" dispersion.Ω4_std
    end
    for i in eachindex(elphm.holstein_couplings_up)

        holstein_coupling_up = elphm.holstein_couplings_up[i]
        @printf io "[[ElectronPhononModel.HolsteinCouplingUp]]\n\n"
        @printf io "HOLSTEIN_ID     = %d\n" i
        @printf io "PHONON_ID       = %d\n" holstein_coupling_up.phonon_id
        @printf io "ORBITAL_ID      = %d\n" holstein_coupling_up.orbital_id
        @printf io "displacement    = %s\n" string(holstein_coupling_up.displacement)
        @printf io "alpha_mean      = %.6f\n" holstein_coupling_up.α_mean
        @printf io "alpha_std       = %.6f\n" holstein_coupling_up.α_std
        @printf io "alpha2_mean     = %.6f\n" holstein_coupling_up.α2_mean
        @printf io "alpha2_std      = %.6f\n" holstein_coupling_up.α2_std
        @printf io "alpha3_mean     = %.6f\n" holstein_coupling_up.α3_mean
        @printf io "alpha3_std      = %.6f\n" holstein_coupling_up.α3_std
        @printf io "alpha4_mean     = %.6f\n" holstein_coupling_up.α4_mean
        @printf io "alpha4_std      = %.6f\n\n" holstein_coupling_up.α4_std

        holstein_coupling_dn = elphm.holstein_couplings_dn[i]
        @printf io "[[ElectronPhononModel.HolsteinCouplingDown]]\n\n"
        @printf io "HOLSTEIN_ID     = %d\n" i
        @printf io "PHONON_ID       = %d\n" holstein_coupling_dn.phonon_id
        @printf io "ORBITAL_ID      = %d\n" holstein_coupling_dn.orbital_id
        @printf io "displacement    = %s\n" string(holstein_coupling_dn.displacement)
        @printf io "alpha_mean      = %.6f\n" holstein_coupling_dn.α_mean
        @printf io "alpha_std       = %.6f\n" holstein_coupling_dn.α_std
        @printf io "alpha2_mean     = %.6f\n" holstein_coupling_dn.α2_mean
        @printf io "alpha2_std      = %.6f\n" holstein_coupling_dn.α2_std
        @printf io "alpha3_mean     = %.6f\n" holstein_coupling_dn.α3_mean
        @printf io "alpha3_std      = %.6f\n" holstein_coupling_dn.α3_std
        @printf io "alpha4_mean     = %.6f\n" holstein_coupling_dn.α4_mean
        @printf io "alpha4_std      = %.6f\n\n" holstein_coupling_dn.α4_std
    end
    for i in eachindex(elphm.ssh_couplings_up)

        ssh_coupling_up = elphm.ssh_couplings_up[i]
        bond = ssh_coupling_up.bond
        @printf io "[[ElectronPhononModel.SSHCouplingUp]]\n\n"
        @printf io "SSH_ID           = %d\n" i
        @printf io "PHONON_IDS       = [%d, %d]\n" ssh_coupling_up.phonon_ids[1] ssh_coupling_up.phonon_ids[2]
        @printf io "BOND_ID          = %d\n" ssh_coupling_up.bond_id
        @printf io "orbitals         = [%d, %d]\n" bond.orbitals[1] bond.orbitals[2]
        @printf io "displacement     = %s\n" string(bond.displacement)
        @printf io "alpha_mean_real  = %.6f\n" real(ssh_coupling_up.α_mean)
        @printf io "alpha_mean_imag  = %.6f\n" imag(ssh_coupling_up.α_mean)
        @printf io "alpha_std        = %.6f\n" ssh_coupling_up.α_std
        @printf io "alpha2_mean_real = %.6f\n" real(ssh_coupling_up.α2_mean)
        @printf io "alpha2_mean_imag = %.6f\n" imag(ssh_coupling_up.α2_mean)
        @printf io "alpha2_std       = %.6f\n" ssh_coupling_up.α2_std
        @printf io "alpha3_mean_real = %.6f\n" real(ssh_coupling_up.α3_mean)
        @printf io "alpha3_mean_imag = %.6f\n" imag(ssh_coupling_up.α3_mean)
        @printf io "alpha3_std       = %.6f\n" ssh_coupling_up.α3_std
        @printf io "alpha4_mean_real = %.6f\n" real(ssh_coupling_up.α4_mean)
        @printf io "alpha4_mean_imag = %.6f\n" imag(ssh_coupling_up.α4_mean)
        @printf io "alpha4_std       = %.6f\n\n" ssh_coupling_up.α4_std

        ssh_coupling_dn = elphm.ssh_couplings_dn[i]
        bond = ssh_coupling_dn.bond
        @printf io "[[ElectronPhononModel.SSHCouplingDown]]\n\n"
        @printf io "SSH_ID           = %d\n" i
        @printf io "PHONON_IDS       = [%d, %d]\n" ssh_coupling_dn.phonon_ids[1] ssh_coupling_dn.phonon_ids[2]
        @printf io "BOND_ID          = %d\n" ssh_coupling_dn.bond_id
        @printf io "orbitals         = [%d, %d]\n" bond.orbitals[1] bond.orbitals[2]
        @printf io "displacement     = %s\n" string(bond.displacement)
        @printf io "alpha_mean_real  = %.6f\n" real(ssh_coupling_dn.α_mean)
        @printf io "alpha_mean_imag  = %.6f\n" imag(ssh_coupling_dn.α_mean)
        @printf io "alpha_std        = %.6f\n" ssh_coupling_dn.α_std
        @printf io "alpha2_mean_real = %.6f\n" real(ssh_coupling_dn.α2_mean)
        @printf io "alpha2_mean_imag = %.6f\n" imag(ssh_coupling_dn.α2_mean)
        @printf io "alpha2_std       = %.6f\n" ssh_coupling_dn.α2_std
        @printf io "alpha3_mean_real = %.6f\n" real(ssh_coupling_dn.α3_mean)
        @printf io "alpha3_mean_imag = %.6f\n" imag(ssh_coupling_dn.α3_mean)
        @printf io "alpha3_std       = %.6f\n" ssh_coupling_dn.α3_std
        @printf io "alpha4_mean_real = %.6f\n" real(ssh_coupling_dn.α4_mean)
        @printf io "alpha4_mean_imag = %.6f\n" imag(ssh_coupling_dn.α4_mean)
        @printf io "alpha4_std       = %.6f\n\n" ssh_coupling_dn.α4_std
    end

    return nothing
end


@doc raw"""
    add_phonon_mode!(;
        # KEYWORD ARGUMENTS
        electron_phonon_model::ElectronPhononModel{T,E,D},
        phonon_mode::PhononMode{E,D}
    ) where {T<:Number, E<:AbstractFloat, D}

Add a [`PhononMode`](@ref) to an [`ElectronPhononModel`](@ref).
"""
function add_phonon_mode!(;
    # KEYWORD ARGUMENTS
    electron_phonon_model::ElectronPhononModel{T,E,D},
    phonon_mode::PhononMode{E,D}
) where {T<:Number, E<:AbstractFloat, D}

    # record phonon mode
    push!(electron_phonon_model.phonon_modes, phonon_mode)

    return length(electron_phonon_model.phonon_modes)
end


@doc raw"""
    add_phonon_dispersion!(;
        # KEYWORD ARGUMENTS
        electron_phonon_model::ElectronPhononModel{T,E,D},
        phonon_dispersion::PhononDispersion{E,D},
        model_geometry::ModelGeometry{D,E}
    ) where {T,E,D}

Add a [`PhononDispersion`](@ref) to an [`ElectronPhononModel`](@ref).
"""
function add_phonon_dispersion!(;
    # KEYWORD ARGUMENTS
    electron_phonon_model::ElectronPhononModel{T,E,D},
    phonon_dispersion::PhononDispersion{E,D},
    model_geometry::ModelGeometry{D,E}
) where {T,E,D}

    # get initial and final phonon modes that are coupled
    phonon_id_init, phonon_id_final = phonon_dispersion.phonon_ids
    N_ph = length(electron_phonon_model.phonon_modes)
    @assert phonon_id_init > 0 && phonon_id_init <= N_ph "Initial phonon mode ID out of bounds."
    @assert phonon_id_final > 0 && phonon_id_final <= N_ph "Final phonon mode ID out of bounds."

    # record the phonon dispersion
    phonon_dispersions::Vector{PhononDispersion{E,D}} = electron_phonon_model.phonon_dispersions
    push!(phonon_dispersions, phonon_dispersion)

    return length(phonon_dispersions)
end


@doc raw"""
    add_holstein_coupling!(;
        # KEYWORD ARGUMENTS
        model_geometry::ModelGeometry{D,E},
        electron_phonon_model::ElectronPhononModel{T,E,D},
        holstein_coupling::Union{HolsteinCoupling{E,D}, Nothing} = nothing,
        holstein_coupling_up::Union{HolsteinCoupling{E,D}, Nothing} = nothing,
        holstein_coupling_dn::Union{HolsteinCoupling{E,D}, Nothing} = nothing
    ) where {T,E,D}

Add the [`HolsteinCoupling`](@ref) to an [`ElectronPhononModel`](@ref). Note that either `holstein_coupling`
or `holstein_coupling_up` and `holstein_coupling_dn` must be specified.
"""
function add_holstein_coupling!(;
    # KEYWORD ARGUMENTS
    model_geometry::ModelGeometry{D,E},
    electron_phonon_model::ElectronPhononModel{T,E,D},
    holstein_coupling::Union{HolsteinCoupling{E,D}, Nothing} = nothing,
    holstein_coupling_up::Union{HolsteinCoupling{E,D}, Nothing} = nothing,
    holstein_coupling_dn::Union{HolsteinCoupling{E,D}, Nothing} = nothing
) where {T,E,D}

    (; unit_cell) = model_geometry
    (; holstein_couplings_up, holstein_couplings_dn) = electron_phonon_model

    # if spin-symmetric holstein coupling
    if !isnothing(holstein_coupling_up) && !isnothing(holstein_coupling_dn)

        @assert holstein_coupling_up.orbital_id == holstein_coupling_dn.orbital_id
        @assert holstein_coupling_up.phonon_id == holstein_coupling_dn.phonon_id
        @assert holstein_coupling_up.displacement == holstein_coupling_dn.displacement
    else

        holstein_coupling_up = holstein_coupling
        holstein_coupling_dn = holstein_coupling
    end

    # get number of orbitals in unit cell
    N_orbitals = unit_cell.n

    # get number of phonon modes
    N_ph = length(electron_phonon_model.phonon_modes)

    # make sure phonon and orbital IDs are valid
    @assert holstein_coupling_up.phonon_id > 0 && holstein_coupling_up.phonon_id <= N_ph "Phonon ID for Holstein coupling out of bounds."
    @assert holstein_coupling_up.orbital_id > 0 && holstein_coupling_up.orbital_id <= N_orbitals "Orbital ID for Holstein coupling out of bounds."

    # record the holstein coupling
    push!(holstein_couplings_up, holstein_coupling_up)
    push!(holstein_couplings_dn, holstein_coupling_dn)

    return length(holstein_couplings_up)
end


@doc raw"""
    add_ssh_coupling!(;
        # KEYWORD ARGUMENTS
        electron_phonon_model::ElectronPhononModel{T,E,D},
        tight_binding_model::Union{TightBindingModel{T,E,D}, Nothing} = nothing,
        ssh_coupling::Union{SSHCoupling{T,E,D}, Nothing} = nothing,
        tight_binding_model_up::Union{TightBindingModel{T,E,D}, Nothing} = nothing,
        tight_binding_model_dn::Union{TightBindingModel{T,E,D}, Nothing} = nothing,
        ssh_coupling_up::Union{SSHCoupling{T,E,D}, Nothing} = nothing,
        ssh_coupling_dn::Union{SSHCoupling{T,E,D}, Nothing} = nothing
    ) where {T,E,D}

Add a [`SSHCoupling`](@ref) to an [`ElectronPhononModel`](@ref).
Note that either `ssh_coupling` and `tight_binding_model` or
`ssh_coupling_up`, `ssh_coupling_dn`, `tight_binding_model_up` and
`tight_binding_model_dn` need to be specified.
"""
function add_ssh_coupling!(;
    # KEYWORD ARGUMENTS
    electron_phonon_model::ElectronPhononModel{T,E,D},
    tight_binding_model::Union{TightBindingModel{T,E,D}, Nothing} = nothing,
    ssh_coupling::Union{SSHCoupling{T,E,D}, Nothing} = nothing,
    tight_binding_model_up::Union{TightBindingModel{T,E,D}, Nothing} = nothing,
    tight_binding_model_dn::Union{TightBindingModel{T,E,D}, Nothing} = nothing,
    ssh_coupling_up::Union{SSHCoupling{T,E,D}, Nothing} = nothing,
    ssh_coupling_dn::Union{SSHCoupling{T,E,D}, Nothing} = nothing
) where {T,E,D}

    if (!isnothing(ssh_coupling_up)        && !isnothing(ssh_coupling_dn) &&
        !isnothing(tight_binding_model_up) && !isnothing(tight_binding_model_dn))

        @assert ssh_coupling_up.bond == ssh_coupling_dn.bond "Spin Up and Down SSH Coupling Bonds Do Not Match."
        @assert ssh_coupling_up.phonon_ids == ssh_coupling_dn.phonon_ids "Spin Up and Down SSH Coupling Phonon IDs Do Not Match."

    elseif !isnothing(ssh_coupling) && !isnothing(tight_binding_model)

        tight_binding_model_up = tight_binding_model
        tight_binding_model_dn = tight_binding_model
        ssh_coupling_up = ssh_coupling
        ssh_coupling_dn = ssh_coupling
    
    else

        error("SSH Coupling Not Consistently Specified.")
    end

    ssh_couplings_up = electron_phonon_model.ssh_couplings_up
    ssh_couplings_dn = electron_phonon_model.ssh_couplings_dn
    tbm_bonds_up = tight_binding_model_up.t_bonds
    tbm_bonds_dn = tight_binding_model_dn.t_bonds
    ssh_bond = ssh_coupling_up.bond

    # make sure a hopping already exists in the tight binding model for the ssh coupling
    @assert ssh_bond in tbm_bonds_up "There is no corresponding hopping in the spin-up tight binding model for the SSH coupling."
    @assert ssh_bond in tbm_bonds_dn "There is no corresponding hopping in the spin-down tight binding model for the SSH coupling."

    # record the ssh_bond
    push!(ssh_couplings_up, ssh_coupling_up)
    push!(ssh_couplings_dn, ssh_coupling_dn)

    return length(ssh_couplings_up)
end