@doc raw"""
    PhononMode{E<:AbstractFloat}

Defines a phonon mode on the orbital species `orbital` in the unit cell. Specifically, it defines the phonon Hamiltonian terms
```math
\hat{H}_{{\rm ph}} = \sum_{\mathbf{i}}
  \left[
      \frac{1}{2} M_{\mathbf{i},\nu}\Omega_{\mathbf{i},\nu}^{2}\hat{X}_{\mathbf{i},\nu}^{2}
    + \frac{1}{12}M_{\mathbf{i},\nu}\Omega_{4,\mathbf{i},\nu}^{2}\hat{X}_{\mathbf{i},\nu}^{4}
    + \frac{1}{2M_{\mathbf{i},\nu}}\hat{P}_{\mathbf{i},\nu}^{2}
  \right],
```
where the sum runs over unit cell ``\mathbf{i}``, ``\nu`` denotes the orbital species `orbital` in the unit cell,
``M_{\mathbf{i},\nu}`` is the phonon mass `M`, ``\Omega_{\mathbf{i},\nu}`` is the phonon frequency that is distributed according
to a normal distribution with mean `Ω_mean` and standard deviation `Ω_std`. Lastly, ``\Omega_{4,\mathbf{i},\nu}`` is the anhmaronic
coefficient, and is distributed according to a normal distribution with mean `Ω4_mean` and standard deviation `Ω4_std`.

# Fields

- `orbital::Int`: Orbital species ``\nu`` in the unit cell.
- `M::E`:: The phonon mass ``M_{\mathbf{i},\nu}.``
- `Ω_mean::E`: Mean of normal distribution the phonon frequency ``\Omega_{\mathbf{i},\nu}`` is sampled from.
- `Ω_std::E`: Standard deviation of normal distribution the phonon frequency ``\Omega_{\mathbf{i},\nu}`` is sampled from.
- `Ω4_mean::E`: Mean of normal distribution the anharmonic coefficient ``\Omega_{4,\mathbf{i},\nu}`` is sampled from.
- `Ω4_std::E`: Standard deviation of normal distribution the anharmonic coefficient ``\Omega_{4,\mathbf{i},\nu}`` is sampled from.
"""
struct PhononMode{E<:AbstractFloat}

    # orbital species
    orbital::Int

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
    PhononMode(; orbital::Int, Ω_mean::E, Ω_std::E=0., M::E=1., Ω4_mean::E=0., Ω4_std::E=0.) where {E<:AbstractFloat}

Initialize and return a instance of [`PhononMode`](@ref).
"""
function PhononMode(; orbital::Int, Ω_mean::E, Ω_std::E=0., M::E=1., Ω4_mean::E=0., Ω4_std::E=0.) where {E<:AbstractFloat}

    return PhononMode(orbital, M, Ω_mean, Ω_std, Ω4_mean, Ω4_std)
end


@doc raw"""
    HolsteinCoupling{E<:AbstractFloat, D}

Defines a Holstein coupling between a specified phonon mode and orbital density.
Specifically, it defines the (extended) Holstein Hamiltonian interaction term
```math
\hat{H}_{{\rm hol}} = \sum_{\sigma,\mathbf{i}}
    \left[ \sum_{n=1}^{4}\alpha_{n,\mathbf{i},(\mathbf{r},\kappa,\nu)}\hat{X}_{\mathbf{i},\nu}^{n} \right]
    \left( \hat{n}_{\sigma,\mathbf{i}+\mathbf{r},\kappa}-\tfrac{1}{2} \right),
```
where ``\sigma`` specifies the sum, and the sum over ``\mathbf{i}`` runs over unit cells in the lattice.
In the above ``\nu`` and ``\kappa`` specify orbital species in the unit cell, and ``\mathbf{r}`` is a static
displacement in unit cells.

# Fields

- `phonon_mode::Int`: The phonon mode getting coupled to.
- `bond::Bond{D}`: Static displacement from ``\hat{X}_{\mathbf{i},\nu}`` to ``\hat{n}_{\sigma,\mathbf{i}+\mathbf{r},\kappa}.``
- `bond_id::Int`: Bond ID associtated with `bond` field.
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

    # phonon mode of coupling
    phonon_mode::Int

    # displacement vector to density phonon mode is coupled to
    bond::Bond{D}

    # bond id
    bond_id::Int

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
end

@doc raw"""
    HolsteinCoupling(; model_geometry::ModelGeometry{D,E}, phonon_mode::Int,
                     bond::Bond{D}, α_mean::E, α_std::E=0.,
                     α2_mean::E=0., α2_std::E=0., α3_mean::E=0., α3_std::E=0.,
                     α4_mean::E=0., α4_std::E=0.) where {E,D}

Initialize and return a instance of [`HolsteinCoupling`](@ref).
"""
function HolsteinCoupling(; model_geometry::ModelGeometry{D,E}, phonon_mode::Int,
                          bond::Bond{D}, α_mean::E, α_std::E=0.,
                          α2_mean::E=0., α2_std::E=0., α3_mean::E=0., α3_std::E=0.,
                          α4_mean::E=0., α4_std::E=0.) where {E,D}

    bond_id = add_bond!(model_geometry, bond)
    return HolsteinCoupling(phonon_mode, bond, bond_id, α_mean, α_std, α2_mean, α2_std, α3_mean, α3_std, α4_mean, α4_std)
end


@doc raw"""
    SSHCoupling{T<:Number, E<:AbstractFloat, D}

Defines a Su-Schrieffer-Heeger (SSH) coupling between a pair of phonon modes.
Specifically, it defines the SSH interaction term
```math
\hat{H}_{{\rm ssh}} = -\sum_{\sigma,\mathbf{i}}
    \left[ t_{\mathbf{i},(\mathbf{r},\kappa,\nu)} - \left( \sum_{n=1}^{4}\alpha_{n,\mathbf{i},(\mathbf{r},\kappa,\nu)}
    \left( \hat{X}_{\mathbf{i}+\mathbf{r},\kappa} - \hat{X}_{\mathbf{i},\nu}\right)^{n}\right) \right]
    \left( \hat{c}_{\sigma,\mathbf{i}+\mathbf{r},\kappa}^{\dagger}\hat{c}_{\sigma,\mathbf{i},\nu}+{\rm h.c.} \right),
```
where ``\sigma`` specifies the sum, and the sum over ``\mathbf{i}`` runs over unit cells in the lattice.
In the above ``\nu`` and ``\kappa`` specify orbital species in the unit cell, and ``\mathbf{r}`` is a static
displacement in unit cells. In that above expression ``t_{\mathbf{i},(\mathbf{r},\kappa,\nu)}`` is the bare
hopping amplitude, which is not specified here.

# Fields

- `phonon_modes::NTuple{2,Int}`: Pair of phonon modes getting coupled together.
- `bond::Bond{D}`: Static displacement seperating the two phonon modes getting coupled.
- `bond_id::Int`: Bond ID associated with the `bond` field.
- `α_mean::T`: Mean of the linear SSH coupling constant ``\alpha_{1,\mathbf{i},(\mathbf{r},\kappa,\nu)}.``
- `α_std::T`: Standard deviation of the linear SSH coupling constant ``\alpha_{1,\mathbf{i},(\mathbf{r},\kappa,\nu)}.``
- `α2_mean::T`: Mean of the quadratic SSH coupling constant ``\alpha_{2,\mathbf{i},(\mathbf{r},\kappa,\nu)}.``
- `α2_std::T`: Standard deviation of the quadratic SSH coupling constant ``\alpha_{2,\mathbf{i},(\mathbf{r},\kappa,\nu)}.``
- `α3_mean::T`: Mean of the cubic SSH coupling constant ``\alpha_{3,\mathbf{i},(\mathbf{r},\kappa,\nu)}.``
- `α3_std::T`: Standard deviation of the cubic SSH coupling constant ``\alpha_{3,\mathbf{i},(\mathbf{r},\kappa,\nu)}.``
- `α4_mean::T`: Mean of the quartic SSH coupling constant ``\alpha_{4,\mathbf{i},(\mathbf{r},\kappa,\nu)}.``
- `α4_std::T`: Standard deviation of the quartic SSH coupling constant ``\alpha_{4,\mathbf{i},(\mathbf{r},\kappa,\nu)}.``

# Comment

The pair of orbitals appearing in `bond.orbitals` must correspond to the orbital species associated with the two coupling phonon modes
specified by `phonon_modes`.
"""
struct SSHCoupling{T<:Number, E<:AbstractFloat, D}

    # phonon modes getting coupled
    phonon_modes::NTuple{2,Int}

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
end

@doc raw"""
    SSHCoupling(; model_geometry::ModelGeometry{D,E}, tight_binding_model::TightBindingModel{T,E,D},
                phonon_modes::NTuple{2,Int}, bond::Bond{D},
                α_mean::T, α_std::E=0., α2_mean::T=0., α2_std::E=0., α3_mean::T=0., α3_std::E=0.,
                α4_mean::T=0., α4_std::E=0.) where {D, T<:Number, E<:AbstractFloat}

Initialize and return a instance of [`SSHCoupling`](@ref).
"""
function SSHCoupling(; model_geometry::ModelGeometry{D,E}, tight_binding_model::TightBindingModel{T,E,D},
                     phonon_modes::NTuple{2,Int}, bond::Bond{D},
                     α_mean::T, α_std::E=0., α2_mean::T=0., α2_std::E=0., α3_mean::T=0., α3_std::E=0.,
                     α4_mean::T=0., α4_std::E=0.) where {D, T<:Number, E<:AbstractFloat}

    # make sure there is already a hopping definition for the tight binding model corresponding to the ssh coupling
    @assert bond in tight_binding_model.t_bonds

    # get the bond ID
    bond_id = add_bond!(model_geometry, bond)

    return SSHCoupling(phonon_modes, bond, bond_id, α_mean, α_std, α2_mean, α2_std, α3_mean, α3_std, α4_mean, α4_std)
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
In the above ``\nu`` and ``\kappa`` specify orbital species in the unit cell, and ``\mathbf{r}`` is a static
displacement in unit cells.

# Fields

- `phonon_modes::NTuple{2,Int}`: Pair of phonon modes getting coupled together.
- `bond::Bond{D}`: Static displacement seperating the two phonon modes getting coupled.
- `bond_id::Int`: Bond ID associated with the `bond` field.
- `Ω_mean::E`: Mean dispersive phonon frequency ``\Omega_{\mathbf{i},(\mathbf{r},\kappa,\nu)}.``
- `Ω_std::E`: Standard deviation of dispersive phonon frequency ``\Omega_{\mathbf{i},(\mathbf{r},\kappa,\nu)}.``
- `Ω4_mean::E`: Mean quartic dispersive phonon coefficient ``\Omega_{4,\mathbf{i},(\mathbf{r},\kappa,\nu)}.``
- `Ω4_std::E`: Standard deviation of quartic dispersive phonon coefficient ``\Omega_{4,\mathbf{i},(\mathbf{r},\kappa,\nu)}.``

# Comment

The pair of orbitals appearing in `bond.orbitals` must correspond to the orbital species associated with the two coupling phonon modes
specified by `phonon_modes`.
"""
struct PhononDispersion{E<:AbstractFloat, D}

    # pair of phonon modes getting coupled
    phonon_modes::NTuple{2,Int}

    # static displacment between phonon modes getting coupled
    bond::Bond{D}

    # bond ID associated with bond above
    bond_id::Int

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
    PhononDispersion(; model_geometry::ModelGeometry{D,E}, phonon_modes::NTuple{2,Int}, bond::Bond{D},
                     Ω_mean::E, Ω_std::E=0., Ω4_mean::E=0., Ω4_std::E=0.) where {E<:AbstractFloat, D}

Initialize and return a instance of [`PhononDispersion`](@ref).
"""
function PhononDispersion(; model_geometry::ModelGeometry{D,E}, phonon_modes::NTuple{2,Int}, bond::Bond{D},
                          Ω_mean::E, Ω_std::E=0., Ω4_mean::E=0., Ω4_std::E=0.) where {E<:AbstractFloat, D}

    # get bond ID
    bond_id = add_bond!(model_geometry, bond)

    return PhononDispersion(phonon_modes, bond, bond_id, Ω_mean, Ω_std, Ω4_mean, Ω4_std)
end


@doc raw"""
    ElectronPhononModel{T<:Number, E<:AbstractFloat, D}

Defines an electron-phonon model.

# Fields

- `phonon_modes::Vector{PhononModes{E}}`: A vector of [`PhononMode`](@ref) definitions.
- `holstein_couplings::Vector{HolsteinCoupling{E,D}}`: A vector of [`HolsteinCoupling`](@ref) definitions.
- `ssh_couplings::Vector{SSHCoupling{T,E,D}}`: A vector of [`SSHCoupling`](@ref) defintions.
- `phonon_dispersions::Vector{PhononDispersion{E,D}}`: A vector of [`PhononDispersion`](@ref) defintions.
"""
struct ElectronPhononModel{T<:Number, E<:AbstractFloat, D}
    
    # phonon modes
    phonon_modes::Vector{PhononMode{E}}

    # holstein couplings
    holstein_couplings::Vector{HolsteinCoupling{E,D}}

    # ssh couplings
    ssh_couplings::Vector{SSHCoupling{T,E,D}}

    # phonon dispersion
    phonon_dispersions::Vector{PhononDispersion{E,D}}
end

@doc raw"""
    ElectronPhononModel(; model_geometry::ModelGeometry{D,E},
                        tight_binding_model::TightBindingModel{T,E,D}) where {T<:Number, E<:AbstractFloat, D}

Initialize and return a null (empty) instance of [`ElectronPhononModel`](@ref) given `model_geometry` and `tight_binding_model`.
"""
function ElectronPhononModel(; model_geometry::ModelGeometry{D,E},
                             tight_binding_model::TightBindingModel{T,E,D}) where {T<:Number, E<:AbstractFloat, D}

    phonon_modes = PhononMode{E}[]
    holstein_couplings = HolsteinCoupling{E,D}[]
    ssh_coupldings = SSHCoupling{T,E,D}[]
    phonon_dispersions = PhononDispersion{E,D}[]

    return ElectronPhononModel(phonon_modes, holstein_couplings, ssh_coupldings, phonon_dispersions)
end


# print struct info in TOML format
function Base.show(io::IO, ::MIME"text/plain", elphm::ElectronPhononModel{T,E,D}) where {T<:AbstractFloat,E,D}

    @printf io "[ElectronPhononModel]\n\n"
    for (i, phonon_mode) in enumerate(elphm.phonon_modes)
        @printf io "[[ElectronPhononModel.PhononMode]]\n\n"
        @printf io "ID           = %d\n" i
        @printf io "orbital      = %d\n" phonon_mode.orbital
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
    for (i, holstein_coupling) in enumerate(elphm.holstein_couplings)
        bond::Bond{D} = holstein_coupling.bond
        @printf io "[[ElectronPhononModel.HolsteinCoupling]]\n\n"
        @printf io "ID              = %d\n" i
        @printf io "phonon_id       = %d\n" holstein_coupling.phonon_mode
        @printf io "bond_id         = %d\n" holstein_coupling.bond_id
        @printf io "phonon_orbital  = %d\n" bond.orbitals[1]
        @printf io "density_orbital = %d\n" bond.orbitals[2]
        @printf io "displacement    = %s\n" string(bond.displacement)
        @printf io "alpha_mean      = %.6f\n" holstein_coupling.α_mean
        @printf io "alpha_std       = %.6f\n" holstein_coupling.α_std
        @printf io "alpha2_mean     = %.6f\n" holstein_coupling.α2_mean
        @printf io "alpha2_std      = %.6f\n" holstein_coupling.α2_std
        @printf io "alpha3_mean     = %.6f\n" holstein_coupling.α3_mean
        @printf io "alpha3_std      = %.6f\n" holstein_coupling.α3_std
        @printf io "alpha4_mean     = %.6f\n" holstein_coupling.α4_mean
        @printf io "alpha4_std      = %.6f\n\n" holstein_coupling.α4_std
    end
    for (i, ssh_coupling) in enumerate(elphm.ssh_couplings)
        bond::Bond{D} = ssh_coupling.bond
        @printf io "[[ElectronPhononModel.SSHCoupling]]\n\n"
        @printf io "ID           = %d\n" i
        @printf io "phonon_ids   = [%d, %d]\n" ssh_coupling.phonon_modes[1] ssh_coupling.phonon_modes[2]
        @printf io "bond_id      = %d\n" ssh_coupling.bond_id
        @printf io "orbitals     = [%d, %d]\n" bond.orbitals[1] bond.orbitals[2]
        @printf io "displacement = %s\n" string(bond.displacement)
        @printf io "alpha_mean   = %.6f\n" ssh_coupling.α_mean
        @printf io "alpha_std    = %.6f\n" ssh_coupling.α_std
        @printf io "alpha2_mean  = %.6f\n" ssh_coupling.α2_mean
        @printf io "alpha2_std   = %.6f\n" ssh_coupling.α2_std
        @printf io "alpha3_mean  = %.6f\n" ssh_coupling.α3_mean
        @printf io "alpha3_std   = %.6f\n" ssh_coupling.α3_std
        @printf io "alpha4_mean  = %.6f\n" ssh_coupling.α4_mean
        @printf io "alpha4_std   = %.6f\n\n" ssh_coupling.α4_std
    end
    for (i, dispersion) in enumerate(elphm.phonon_dispersions)
        bond::Bond{D} = dispersion.bond
        @printf io "[[ElectronPhononModel.PhononDispersion]]\n\n"
        @printf io "ID           = %d\n" i
        @printf io "phonon_ids   = [%d, %d]\n" dispersion.phonon_modes[1] dispersion.phonon_modes[2]
        @printf io "bond_id      = %d\n" ssh_coupling.bond_id
        @printf io "orbitals     = [%d, %d]\n" bond.orbitals[1] bond.orbitals[2]
        @printf io "displacement = %s\n" string(bond.displacement)
        @printf io "omega_mean   = %.6f\n" dispersion.Ω_mean
        @printf io "omega_std    = %.6f\n" dispersion.Ω_std
        @printf io "omega_4_mean = %.6f\n" dispersion.Ω4_mean
        @printf io "omega_4_std  = %.6f\n\n" dispersion.Ω4_std
    end

    return nothing
end

# print struct info in TOML format
function Base.show(io::IO, ::MIME"text/plain", elphm::ElectronPhononModel{T,E,D}) where {T<:Complex,E,D}

    @printf io "[ElectronPhononModel]\n\n"
    for (i, phonon_mode) in enumerate(elphm.phonon_modes)
        @printf io "[[ElectronPhononModel.PhononMode]]\n\n"
        @printf io "ID           = %d\n" i
        @printf io "orbital      = %d\n" phonon_mode.orbital
        if isfinite(phonon_mode.M)
            @printf io "mass         = %.6f\n" phonon_mode.M
        else
            @printf io "mass         = inf\n" phonon_mode.M
        end
        @printf io "omega_mean   = %.6f\n" phonon_mode.Ω_mean
        @printf io "omega_std    = %.6f\n" phonon_mode.Ω_std
        @printf io "omega_4_mean = %.6f\n" phonon_mode.Ω4_mean
        @printf io "omega_4_std  = %.6f\n\n" phonon_mode.Ω4_std
    end
    for (i, holstein_coupling) in enumerate(elphm.holstein_couplings)
        bond::Bond{D} = holstein_coupling.bond
        @printf io "[[ElectronPhononModel.HolsteinCoupling]]\n\n"
        @printf io "ID              = %d\n" i
        @printf io "phonon_id       = %d\n" holstein_coupling.phonon_mode
        @printf io "bond_id         = %d\n" holstein_coupling.bond_id
        @printf io "phonon_orbital  = %d\n" bond.orbitals[1]
        @printf io "density_orbital = %d\n" bond.orbitals[2]
        @printf io "displacement    = %s\n" string(bond.displacement)
        @printf io "alpha_mean      = %.6f\n" holstein_coupling.α_mean
        @printf io "alpha_std       = %.6f\n" holstein_coupling.α_std
        @printf io "alpha2_mean     = %.6f\n" holstein_coupling.α2_mean
        @printf io "alpha2_std      = %.6f\n" holstein_coupling.α2_std
        @printf io "alpha3_mean     = %.6f\n" holstein_coupling.α3_mean
        @printf io "alpha3_std      = %.6f\n" holstein_coupling.α3_std
        @printf io "alpha4_mean     = %.6f\n" holstein_coupling.α4_mean
        @printf io "alpha4_std      = %.6f\n\n" holstein_coupling.α4_std
    end
    for (i, ssh_coupling) in enumerate(elphm.ssh_couplings)
        bond::Bond{D} = ssh_coupling.bond
        @printf io "[[ElectronPhononModel.SSHCoupling]]\n\n"
        @printf io "ID               = %d\n" i
        @printf io "phonon_ids       = [%d, %d]\n" ssh_coupling.phonon_modes[1] ssh_coupling.phonon_modes[2]
        @printf io "bond_id          = %d\n" ssh_coupling.bond_id
        @printf io "orbitals         = [%d, %d]\n" bond.orbitals[1] bond.orbitals[2]
        @printf io "displacement     = %s\n" string(bond.displacement)
        @printf io "alpha_mean_real  = %.6f\n" real(ssh_coupling.α_mean)
        @printf io "alpha_mean_imag  = %.6f\n" imag(ssh_coupling.α_mean)
        @printf io "alpha_std        = %.6f\n" ssh_coupling.α_std
        @printf io "alpha2_mean_real = %.6f\n" real(ssh_coupling.α2_mean)
        @printf io "alpha2_mean_imag = %.6f\n" imag(ssh_coupling.α2_mean)
        @printf io "alpha2_std       = %.6f\n" ssh_coupling.α2_std
        @printf io "alpha3_mean_real = %.6f\n" real(ssh_coupling.α3_mean)
        @printf io "alpha3_mean_imag = %.6f\n" imag(ssh_coupling.α3_mean)
        @printf io "alpha3_std       = %.6f\n" ssh_coupling.α3_std
        @printf io "alpha4_mean_real = %.6f\n" real(ssh_coupling.α4_mean)
        @printf io "alpha4_mean_imag = %.6f\n" imag(ssh_coupling.α4_mean)
        @printf io "alpha4_std       = %.6f\n\n" ssh_coupling.α4_std
    end
    for (i, dispersion) in enumerate(elphm.phonon_dispersions)
        bond::Bond{D} = dispersion.bond
        @printf io "[[ElectronPhononModel.PhononDispersion]]\n\n"
        @printf io "ID           = %d\n" i
        @printf io "phonon_ids   = [%d, %d]\n" dispersion.phonon_modes[1] dispersion.phonon_modes[2]
        @printf io "bond_id      = %d\n" ssh_coupling.bond_id
        @printf io "orbitals     = [%d, %d]\n" bond.orbitals[1] bond.orbitals[2]
        @printf io "displacement = %s\n" string(bond.displacement)
        @printf io "omega_mean   = %.6f\n" dispersion.Ω_mean
        @printf io "omega_std    = %.6f\n" dispersion.Ω_std
        @printf io "omega_4_mean = %.6f\n" dispersion.Ω4_mean
        @printf io "omega_4_std  = %.6f\n\n" dispersion.Ω4_std
    end

    return nothing
end


@doc raw"""
    add_phonon_mode!(; electron_phonon_model::ElectronPhononModel{T,E,D},
                     phonon_mode::PhononMode{E}) where {T<:Number, E<:AbstractFloat, D}

Add a [`PhononMode`](@ref) to an [`ElectronPhononModel`](@ref).
"""
function add_phonon_mode!(; electron_phonon_model::ElectronPhononModel{T,E,D},
                          phonon_mode::PhononMode{E}) where {T<:Number, E<:AbstractFloat, D}

    # record phonon mode
    push!(electron_phonon_model.phonon_modes, phonon_mode)

    return length(electron_phonon_model.phonon_modes)
end


@doc raw"""
    add_holstein_coupling!(; electron_phonon_model::ElectronPhononModel{T,E,D},
                           holstein_coupling::HolsteinCoupling{E,D},
                           model_geometry::ModelGeometry{D,E}) where {T,E,D}

Add the [`HolsteinCoupling`](@ref) to an [`ElectronPhononModel`](@ref).
"""
function add_holstein_coupling!(; electron_phonon_model::ElectronPhononModel{T,E,D},
                                holstein_coupling::HolsteinCoupling{E,D},
                                model_geometry::ModelGeometry{D,E}) where {T,E,D}

    # get the phonon mode getting coupled to
    phonon_modes::Vector{PhononMode{E}} = electron_phonon_model.phonon_modes
    phonon_mode = phonon_modes[holstein_coupling.phonon_mode]

    # get the bond associated with holstein coupling
    holstein_bond::Bond{D} = holstein_coupling.bond

    # make sure the initial bond orbital matches the orbital species of the phonon mode
    @assert phonon_mode.orbital == holstein_bond.orbitals[1]

    # record the bond definition associated with the holstein coupling if not already recorded
    bond_id = add_bond!(model_geometry, holstein_bond)

    # record the holstein coupling
    holstein_couplings::Vector{HolsteinCoupling{E,D}} = electron_phonon_model.holstein_couplings
    push!(holstein_couplings, holstein_coupling)

    return length(holstein_couplings)
end


@doc raw"""
    add_ssh_coupling!(; electron_phonon_model::ElectronPhononModel{T,E,D},
                      ssh_coupling::SSHCoupling{T,E,D},
                      tight_binding_model::TightBindingModel{T,E,D}) where {T,E,D}

Add a [`SSHCoupling`](@ref) to an [`ElectronPhononModel`](@ref).
"""
function add_ssh_coupling!(; electron_phonon_model::ElectronPhononModel{T,E,D},
                           ssh_coupling::SSHCoupling{T,E,D},
                           tight_binding_model::TightBindingModel{T,E,D}) where {T,E,D}

    phonon_modes::Vector{PhononMode{E}} = electron_phonon_model.phonon_modes
    ssh_couplings::Vector{SSHCoupling{T,E,D}} = electron_phonon_model.ssh_couplings
    tbm_bonds = tight_binding_model.t_bonds
    ssh_bond::Bond{D} = ssh_coupling.bond

    # get initial and final phonon modes that are coupled
    phonon_mode_init = phonon_modes[ssh_coupling.phonon_modes[1]]
    phonon_mode_final = phonon_modes[ssh_coupling.phonon_modes[2]]

    # make sure a hopping already exists in the tight binding model for the ssh coupling
    @assert ssh_bond in tbm_bonds

    # make the the staring and ending orbitals of the ssh bond match the orbital species of the phonon modes getting coupled
    @assert ssh_bond.orbitals[1] == phonon_mode_init.orbital
    @assert ssh_bond.orbitals[2] == phonon_mode_final.orbital

    # record the ssh_bond
    push!(ssh_couplings, ssh_coupling)

    return length(ssh_couplings)
end


@doc raw"""
    add_phonon_dispersion!(elphm::ElectronPhononModel{T,E,D}, pd::PhononDispersion{E,D}, mg::ModelGeometry{D,E}) where {T,E,D}

Add a [`PhononDispersion`](@ref) to an [`ElectronPhononModel`](@ref).
"""
function add_phonon_dispersion!(; electron_phonon_model::ElectronPhononModel{T,E,D},
                                phonon_dispersion::PhononDispersion{E,D},
                                model_geometry::ModelGeometry{D,E}) where {T,E,D}

    # get initial and final phonon modes that are coupled
    phonon_modes::Vector{PhononMode{E}} = electron_phonon_model.phonon_modes
    phonon_mode_init = phonon_modes[pd.phonon_modes[1]]
    phonon_mode_final = phonon_modes[pd.phonon_modes[2]]

    # get the bond defining the phonon dispersion
    dispersion_bond = phonon_dispersion.bond

    # make the the staring and ending orbitals of the ssh bond match the orbital species of the phonon modes getting coupled
    @assert dispersion_bond.orbitals[1] == phonon_mode_init.orbital
    @assert dispersion_bond.orbitals[2] == phonon_mode_final.orbital

    # record the bond definition associated with the holstein coupling if not already recorded
    bond_id = add_bond!(model_geometry, dispersion_bond)

    # record the phonon dispersion
    phonon_dispersions::Vector{HolsteinCoupling{E,D}} = electron_phonon_model.phonon_dispersions
    push!(phonon_dispersions, phonon_dispersion)

    return length(phonon_dispersions)
end