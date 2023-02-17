@doc raw"""
    HubbardModel{T<:AbstractFloat}

If `shifted = true`, then a Hubbard interaction of the form
```math
\hat{H}_{U}=\sum_{\mathbf{i},\nu}U_{\nu,\mathbf{i}}\hat{n}_{\uparrow,\nu,\mathbf{i}}\hat{n}_{\downarrow,\nu,\mathbf{i}}
```
is assumed, where ``\mathbf{i}`` specifies the unit cell, and ``\nu`` denotes the orbital in the unit cell.
For a bipartite lattice with only nearest neighbor hopping, the on-site energy corresponding to half-filling
and particle-hole symmetry is ``\epsilon_{\nu,\mathbf{i}} = -U_{\nu,\mathbf{i}}/2.``

If `shifted = false`, then a Hubbard interaction of the form
```math
\hat{H}_{U}=\sum_{\mathbf{i},\nu}U_{\nu,\mathbf{i}}(\hat{n}_{\uparrow,\nu,\mathbf{i}}-\tfrac{1}{2})(\hat{n}_{\downarrow,\nu,\mathbf{i}}-\tfrac{1}{2})
```
is assumed. In this case, for a bipartite lattice with only nearest neighbor hopping, the on-site energy corresponding to half-filling
and particle-hole symmetry is ``\epsilon_{\nu,\mathbf{i}} = 0.``

# Fields

- `shifted::Bool`: Determines which form for Hubbard interaction is used, and whether the on-site energies need to be shifted.
- `U_orbital::Vector{Int}`: Orbital species in unit cell with finite Hubbard interaction.
- `U_mean::Vector{T}`: Average Hubbard ``U_\nu`` for a given orbital species in the lattice.
- `U_std::Vector{T}`: Standard deviation of Hubbard ``U_\nu`` for a given orbital species in the lattice.
"""
struct HubbardModel{T<:AbstractFloat}

    # whether zero on-site energy corresponds to half-filling in atomic limit
    shifted::Bool

    # orbital species
    U_orbital::Vector{Int}

    # average Hubbard U
    U_mean::Vector{T}

    # standard deviation of Hubbard U
    U_std::Vector{T}
end

@doc raw"""
    HubbardModel(; shifted::Bool, U_orbital::AbstractVector{Int}, U_mean::AbstractVector{T},
                 U_std::AbstractVector{T} = zeros(eltype(U_mean), length(U_mean))) where {T<:AbstractFloat}

Initialize and return an instance of the type [`HubbardModel`](@ref).
"""
function HubbardModel(; shifted::Bool, U_orbital::AbstractVector{Int}, U_mean::AbstractVector{T},
                      U_std::AbstractVector{T} = zeros(eltype(U_mean), length(U_mean))) where {T<:AbstractFloat}
    
    return HubbardModel(shifted, U_orbital, U_mean, U_std)
end


# show struct info as TOML formatted string
function Base.show(io::IO, ::MIME"text/plain", hm::HubbardModel)

    (; U_orbital, U_mean, U_std, shifted) = hm

    @printf io "[HubbardModel]\n\n"
    @printf io "U_orbital = %s\n" string(U_orbital)
    @printf io "U_mean    = %s\n" string(round.(U_mean, digits=6))
    @printf io "U_std     = %s\n" string(round.(U_std, digits=6))
    @printf io "shifted   = %s\n\n" string(shifted)

    return nothing
end


@doc raw"""
    HubbardParameters{T<:AbstractFloat}

Hubbard parameters for finite lattice.

# Fields

- `U::Vector{T}`: On-site Hubbard interaction for each site with finite Hubbard interaction.
- `sites::Vector{Int}`: Site index associated with each finite Hubbard `U` interaction.
- `orbitals::Vector{Int}`: Orbital species in unit cell with finite Hubbard interaction.
- `shifted::Bool`: Convention used for Hubbard interaction, refer to [`HubbardModel`](@ref) for more information.
"""
struct HubbardParameters{T<:AbstractFloat}

    # Hubbard U for each orbital in the lattice
    U::Vector{T}

    # site index associated with each Hubbard U
    sites::Vector{Int}

    # orbital species in unit cell with finite hubbard interaction
    orbitals::Vector{Int}

    # whether zero on-site energy corresponds to half-filling in atomic limit
    shifted::Bool
end

@doc raw"""
    HubbardParameters(; hubbard_model::HubbardModel{T},
                      model_geometry::ModelGeometry{D,T},
                      rng::AbstractRNG) where {D,T<:AbstractFloat}

Initialize an instance of [`HubbardParameters`](@ref).
"""
function HubbardParameters(; hubbard_model::HubbardModel{T},
                           model_geometry::ModelGeometry{D,T},
                           rng::AbstractRNG) where {D, T<:AbstractFloat}

    (; U_orbital, U_mean, U_std, shifted) = hubbard_model
    (; lattice, unit_cell) = model_geometry

    # number of orbitals with finite hubbard interaction in unit cell
    n_hubbard = length(hubbard_model.U_orbital)

    # number of unit cell in lattice
    N_unitcells = lattice.N

    # get the number of HS transformations per time-slice τ that is applied
    N_hubbard = N_unitcells * n_hubbard

    # allocate arrays
    U     = zeros(T, N_hubbard)
    sites = zeros(Int, N_hubbard)

    # reshape the allocated arrays
    U′     = reshape(U, (N_unitcells, n_hubbard))
    sites′ = reshape(sites, (N_unitcells, n_hubbard))

    # iterate over orbitals in the unit cell with finite hubbard interaction
    for (n,o) in enumerate(U_orbital)
        # iterate over unit cells in the lattice
        for u in 1:N_unitcells
            # calculate the site associated with the hubbard interaction
            sites′[u,n] = loc_to_site(u, o, unit_cell)
            # get the Hubbard U interaction on the site
            U′[u,n] = U_mean[n] + U_std[n] * randn(rng)
        end
    end

    return HubbardParameters(U, sites, U_orbital, shifted)
end


@doc raw"""
    initialize!(fermion_path_integral_up::FermionPathIntegral{T,E},
                fermion_path_integral_dn::FermionPathIntegral{T,E},
                hubbard_parameters::HubbardParameters{E}) where {T,E}

    initialize!(fermion_path_integral::FermionPathIntegral{T,E},
                hubbard_parameters::HubbardParameters{E}) where {T,E}

Initialize the contribution from the Hubbard interaction to a [`FermionPathIntegral`](@ref) instance.
"""
function initialize!(fermion_path_integral_up::FermionPathIntegral{T,E},
                     fermion_path_integral_dn::FermionPathIntegral{T,E},
                     hubbard_parameters::HubbardParameters{E}) where {T,E}

    initialize!(fermion_path_integral_up, hubbard_parameters)
    initialize!(fermion_path_integral_dn, hubbard_parameters)

    return nothing
end

function initialize!(fermion_path_integral::FermionPathIntegral{T,E},
                     hubbard_parameters::HubbardParameters{E}) where {T,E}

    (; shifted, U, sites) = hubbard_parameters
    (; V) = fermion_path_integral

    # shift on-site energies if necessary
    if shifted
        for i in eachindex(U)
            # shift on-site energies by +U/2
            @views @. V[sites[i],:] = V[sites[i],:] + U[i]/2
        end
    end

    return nothing
end