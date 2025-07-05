@doc raw"""
    ExtendedHubbardModel{T<:AbstractFloat}

A type to represent extended Hubbard interactions.

If the type field `ph_sym_form = false` then the particle-hole asymmetric form of the extended Hubbard interaction
```math
\begin{align*}
\hat{H}_{V} = \sum_{\mathbf{j},\mathbf{r},\nu,\eta}V_{(\mathbf{j}+\mathbf{r},\nu),(\mathbf{j},\eta)} & \hat{n}_{\mathbf{j}+\mathbf{r},\nu}\hat{n}_{\mathbf{j},\eta} \\
    = \sum_{\mathbf{j},\mathbf{r},\nu,\eta}V_{(\mathbf{j}+\mathbf{r},\nu),(\mathbf{j},\eta)} & \bigg[\tfrac{1}{2}(\hat{n}_{\mathbf{j}+\mathbf{r},\nu}+\hat{n}_{\mathbf{j},\eta}-2)^{2}-1 \\
    & -\hat{n}_{\mathbf{j}+\mathbf{r},\nu,\uparrow}\hat{n}_{\mathbf{j}+\mathbf{r},\nu\downarrow}-\hat{n}_{\mathbf{j},\eta,\uparrow}\hat{n}_{\mathbf{j},\eta\downarrow}+\tfrac{3}{2}\hat{n}_{\mathbf{j}+\mathbf{r},\nu}+\tfrac{3}{2}\hat{n}_{\mathbf{j},\eta}\bigg]
\end{align*}
```
is used, where ``\mathbf{j}`` specifies a unit cell in the lattice, ``\mathbf{r}`` is a displacement in units, and ``\nu`` and ``\eta`` specify the orbital in a given unit cell.
Here, ``\hat{n}_{\mathbf{j},\eta} = (\hat{n}_{\uparrow,\mathbf{j},\eta} + \hat{n}_{\downarrow,\mathbf{j},\eta})`` is the electron number operator for orbital
``\eta`` in unit cell ``\mathbf{j}`` in the lattice. Therefore, ``V_{(\mathbf{j}+\mathbf{r},\nu),(\mathbf{j},\eta)}`` controls the strength of the
extended Hubbard interaction between orbital ``\eta`` in unit cell ``\mathbf{j}`` and orbital ``\nu`` in unit cell ``\mathbf{j}+\mathbf{r}``.

If the type field `ph_sym_form = true`, then the particle-hole symmetric for the extended Hubbard interaction
```math
\begin{align*}
\hat{H}_{V}=\sum_{\mathbf{j},\mathbf{r},\nu,\eta}V_{(\mathbf{j}+\mathbf{r},\nu),(\mathbf{j},\eta)}&(\hat{n}_{\mathbf{j}+\mathbf{r},\nu}-1)(\hat{n}_{\mathbf{j},\eta}-1) \\
    = \sum_{\mathbf{j},\mathbf{r},\nu,\eta}V_{(\mathbf{j}+\mathbf{r},\nu),(\mathbf{j},\eta)}&\bigg[\tfrac{1}{2}(\hat{n}_{\mathbf{j}+\mathbf{r},\nu}+\hat{n}_{\mathbf{j},\eta}-2)^{2}+\tfrac{1}{2} \\
    & -(\hat{n}_{\mathbf{j}+\mathbf{r},\nu,\uparrow}-\tfrac{1}{2})(\hat{n}_{\mathbf{j}+\mathbf{r},\nu\downarrow}-\tfrac{1}{2})-(\hat{n}_{\mathbf{j},\eta,\uparrow}-\tfrac{1}{2})(\hat{n}_{\mathbf{j},\eta\downarrow}-\tfrac{1}{2})\bigg]
\end{align*}
```
is used instead.

# Fields

- `ph_sym_form::Bool`: Whether the particle-hole symmetric form of the extended Hubbard interaction is used.
- `V_bond_ids::Vector{Int}`: Bond IDs specifying bond definition that separates a pair of orbitals with an extended Hubbard interaction between them.
- `V_mean::Vector{T}`: Average extended Hubbard interaction strength ``V_{(\mathbf{j}+\mathbf{r},\nu),(\mathbf{j},\eta)}`` associated with bond definition.
- `V_mean::Vector{T}`: Standard deviation of extended Hubbard interaction strength ``V_{(\mathbf{j}+\mathbf{r},\nu),(\mathbf{j},\eta)}`` associated with bond definition.
"""
struct ExtendedHubbardModel{T<:AbstractFloat}

    # whether particle-hole symmetric form for interaction is used
    ph_sym_form::Bool

    # bond IDs
    V_bond_ids::Vector{Int}

    # average extend Hubbard interaction V
    V_mean::Vector{T}

    # standard deviation of extended Hubbard interaction V
    V_std::Vector{T}
end

@doc raw"""
    ExtendedHubbardModel(;
        # KEYWORD ARGUMENTS
        ph_sym_form::Bool,
        V_bonds::Vector{Bond{D}},
        V_mean::Vector{T},
        V_std::Vector{T},
        model_geometry::ModelGeometry{D,T}
    ) where {T<:AbstractFloat, D}

Initialize and return an instance of the type [`ExtendedHubbardModel`](@ref).
"""
function ExtendedHubbardModel(;
    # KEYWORD ARGUMENTS
    ph_sym_form::Bool,
    V_bonds::Vector{Bond{D}},
    V_mean::Vector{T},
    V_std::Vector{T},
    model_geometry::ModelGeometry{D,T}
) where {T<:AbstractFloat, D}

    V_bonds_ids = [add_bond!(bond, model_geometry) for bond in V_bonds]

    return ExtendedHubbardModel{T,D}(ph_sym_form, V_bonds_ids, V_mean, V_std)
end

# show struct info as TOML formatted string
function Base.show(io::IO, ::MIME"text/plain", ehm::ExtendedHubbardModel)

    (; V_bond_ids, V_mean, V_std, ph_sym_form) = ehm
    @printf io "[ExtendedHubbardModel]\n\n"
    @printf io "EXT_HUB_IDS = %s\n" string(collect(1:length(V_bond_ids)))
    @printf io "BOND_IDS    = %s\n" string(V_bond_ids)
    @printf io "V_mean      = %s\n" string(round.(V_mean, digits=6))
    @printf io "V_std       = %s\n" string(round.(V_std, digits=6))
    @printf io "ph_sym_form = %s\n\n" string(ph_sym_form)

    return nothing
end