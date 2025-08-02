######################################
## INITIALIZE MEASUREMENT CONTAINER ##
######################################

@doc raw"""
    initialize_measurement_container(
        model_geometry::ModelGeometry{D,T,N},
        β::T, Δτ::T
    ) where {T<:AbstractFloat, D, N}

Initialize and return a measurement container of type `NamedTuple`.
"""
function initialize_measurement_container(
    model_geometry::ModelGeometry{D,T,N},
    β::T, Δτ::T
) where {T<:AbstractFloat, D, N}

    lattice   = model_geometry.lattice::Lattice{D}
    unit_cell = model_geometry.unit_cell::UnitCell{D,T,N}
    bonds     = model_geometry.bonds::Vector{Bond{D}}

    # number of orbitals per unit cell
    norbitals = unit_cell.n

    # length of imaginary time axis
    Lτ = eval_length_imaginary_axis(β, Δτ)

    # size of lattice in unit cells in direction of each lattice vector
    L = lattice.L

    # initialize global measurements
    global_measurements = Dict{String, Complex{T}}(k => zero(Complex{T}) for k in GLOBAL_MEASUREMENTS)

    # initialize local measurement
    local_measurements = Dict{String, Vector{Complex{T}}}(
        "density"    => zeros(Complex{T}, norbitals), # average density for each orbital species
        "density_up" => zeros(Complex{T}, norbitals), # average density for each orbital species
        "density_dn" => zeros(Complex{T}, norbitals), # average density for each orbital species
        "double_occ" => zeros(Complex{T}, norbitals), # average double occupancy for each orbital
    )

    # initialize time displaced correlation measurement dictionary
    equaltime_correlations = Dict{String, CorrelationContainer{D,T}}()

    # initialize time displaced correlation measurement dictionary
    time_displaced_correlations = Dict{String, CorrelationContainer{D+1,T}}()

    # initialize integrated correlation measurement dictionary
    integrated_correlations = Dict{String, CorrelationContainer{D,T}}()

    # initialize time displaced correlation measurement dictionary
    equaltime_composite_correlations = Dict{String, CompositeCorrelationContainer{D,D,T}}()

    # initialize time displaced correlation measurement dictionary
    time_displaced_composite_correlations = Dict{String, CompositeCorrelationContainer{D,D+1,T}}()

    # initialize integrated correlation measurement dictionary
    integrated_composite_correlations = Dict{String, CompositeCorrelationContainer{D,D,T}}()

    # initialize measurement container
    measurement_container = (
        global_measurements                   = global_measurements,
        local_measurements                    = local_measurements,
        equaltime_correlations                = equaltime_correlations,
        time_displaced_correlations           = time_displaced_correlations,
        integrated_correlations               = integrated_correlations,
        equaltime_composite_correlations      = equaltime_composite_correlations,
        time_displaced_composite_correlations = time_displaced_composite_correlations,
        integrated_composite_correlations     = integrated_composite_correlations,
        hopping_to_bond_id          = Int[],
        phonon_basis_vecs           = Vector{T}[],
        L                           = L,
        Lτ                          = Lτ,
        a                           = zeros(Complex{T}, L..., Lτ),
        a′                          = zeros(Complex{T}, L..., Lτ),
        a″                          = zeros(Complex{T}, L..., Lτ),
    )

    return measurement_container
end


############################################################
## INITIALIZE MEASUREMENTS ASSOCIATED WITH VARIOUS MODELS ##
############################################################

@doc raw"""
    initialize_measurements!(
        measurement_container::NamedTuple,
        tight_binding_model_up::TightBindingModel{T,E},
        tight_binding_model_dn::TightBindingModel{T,E},
    ) where {T<:Number, E<:AbstractFloat}

    initialize_measurements!(
        measurement_container::NamedTuple,
        tight_binding_model::TightBindingModel{T,E}
    ) where {T<:Number, E<:AbstractFloat}

Initialize tight-binding model related measurements.

# Initialized Measurements

- `onsite_energy`: Refer to [`measure_onsite_energy`](@ref).
- `onsite_energy_up`: Refer to [`measure_onsite_energy`](@ref).
- `onsite_energy_dn`: Refer to [`measure_onsite_energy`](@ref).
- `hopping_energy`: Refer to [`measure_hopping_energy`](@ref).
- `hopping_energy_up`: Refer to [`measure_hopping_energy`](@ref).
- `hopping_energy_dn`: Refer to [`measure_hopping_energy`](@ref).
"""
function initialize_measurements!(
    measurement_container::NamedTuple,
    tight_binding_model_up::TightBindingModel{T,E},
    tight_binding_model_dn::TightBindingModel{T,E},
) where {T<:Number, E<:AbstractFloat}

    initialize_measurements!(measurement_container, tight_binding_model_up)

    return nothing
end

function initialize_measurements!(
    measurement_container::NamedTuple,
    tight_binding_model::TightBindingModel{T,E}
) where {T<:Number, E<:AbstractFloat}

    (; local_measurements, global_measurements, hopping_to_bond_id) = measurement_container

    # number of orbitals per unit cell
    norbital = length(tight_binding_model.ϵ_mean)

    # number of types of hoppings
    nhopping = length(tight_binding_model.t_bond_ids)

    # initialize chemical potential as global measurement
    global_measurements["chemical_potential"] = zero(Complex{E})

    # initialize on-site energy measurement
    local_measurements["onsite_energy"]    = zeros(Complex{E}, norbital)
    local_measurements["onsite_energy_up"] = zeros(Complex{E}, norbital)
    local_measurements["onsite_energy_dn"] = zeros(Complex{E}, norbital)

    # initialize hopping related measurements
    if nhopping > 0

        local_measurements["hopping_energy"]           = zeros(Complex{E}, nhopping)
        local_measurements["hopping_energy_up"]        = zeros(Complex{E}, nhopping)
        local_measurements["hopping_energy_dn"]        = zeros(Complex{E}, nhopping)

        local_measurements["bare_hopping_energy"]      = zeros(Complex{E}, nhopping)
        local_measurements["bare_hopping_energy_up"]   = zeros(Complex{E}, nhopping)
        local_measurements["bare_hopping_energy_dn"]   = zeros(Complex{E}, nhopping)

        local_measurements["hopping_amplitude"]        = zeros(Complex{E}, nhopping)
        local_measurements["hopping_amplitude_up"]     = zeros(Complex{E}, nhopping)
        local_measurements["hopping_amplitude_dn"]     = zeros(Complex{E}, nhopping)

        local_measurements["hopping_inversion"]        = zeros(Complex{E}, nhopping)
        local_measurements["hopping_inversion_up"]     = zeros(Complex{E}, nhopping)
        local_measurements["hopping_inversion_dn"]     = zeros(Complex{E}, nhopping)

        local_measurements["hopping_inversion_avg"]    = zeros(Complex{E}, nhopping)
        local_measurements["hopping_inversion_avg_up"] = zeros(Complex{E}, nhopping)
        local_measurements["hopping_inversion_avg_dn"] = zeros(Complex{E}, nhopping)
    end

    # record bond ID associated with each hopping ID
    for id in tight_binding_model.t_bond_ids
        push!(hopping_to_bond_id, id)
    end

    return nothing
end


@doc raw"""
    initialize_measurements!(
        measurement_container::NamedTuple,
        hubbard_model::HubbardModel{T}
    ) where {T<:AbstractFloat}

Initialize Hubbard model related measurements.

# Initialized Measurements:

- `hubbard_energy`: Refer to [`measure_hopping_energy`](@ref).
"""
function initialize_measurements!(
    measurement_container::NamedTuple,
    hubbard_model::HubbardModel{T}
) where {T<:AbstractFloat}

    (; local_measurements) = measurement_container
    (; U_orbital_ids) = hubbard_model

    # number of orbitals in unit cell
    n_hubbard = length(U_orbital_ids)

    # initialize hubbard energy measurement U⋅nup⋅ndn
    local_measurements["hubbard_energy"] = zeros(Complex{T}, n_hubbard)

    return nothing
end


@doc raw"""
    initialize_measurements!(
        measurement_container::NamedTuple,
        extended_hubbard_model::ExtendedHubbardModel{T}
    ) where {T<:AbstractFloat}

Initialize Extended Hubbard model related measurements.

# Initialized Measurements:

- `ext_hubbard_energy`: Refer to [`measure_ext_hub_energy`](@ref).
"""
function initialize_measurements!(
    measurement_container::NamedTuple,
    extended_hubbard_model::ExtendedHubbardModel{T}
) where {T<:AbstractFloat}

    (; local_measurements) = measurement_container
    (; V_bond_ids) = extended_hubbard_model

    # number of extended hubbard interactions
    n_ehi = length(V_bond_ids)

    # initialize extended hubbard energy measurement
    local_measurements["ext_hub_energy"] = zeros(Complex{T}, n_ehi)

    return nothing
end


@doc raw"""
    initialize_measurements!(
        measurement_container::NamedTuple,
        electron_phonon_model::ElectronPhononModel{T, E, D}
    ) where {T<:Number, E<:AbstractFloat, D}

Initialize electron-phonon model related measurements.

# Initialized Measurements:

- `phonon_kinetic_energy`: Refer to [`measure_phonon_kinetic_energy`](@ref).
- `phonon_potential_energy`: Refer to [`measure_phonon_potential_energy`](@ref).
- `X`: Measure ``\langle \hat{X} \rangle``, refer to [`measure_phonon_position_moment`](@ref).
- `X2`: Measure ``\langle \hat{X}^2 \rangle``, refer to [`measure_phonon_position_moment`](@ref).
- `X3`: Measure ``\langle \hat{X}^3 \rangle``, refer to [`measure_phonon_position_moment`](@ref).
- `X4`: Measure ``\langle \hat{X}^4 \rangle``, refer to [`measure_phonon_position_moment`](@ref).
- `holstein_energy`: Refer to [`measure_holstein_energy`](@ref).
- `holstein_energy_up`: Refer to [`measure_holstein_energy`](@ref).
- `holstein_energy_dn`: Refer to [`measure_holstein_energy`](@ref).
- `ssh_energy`: Refer to [`measure_ssh_energy`](@ref).
- `ssh_energy_up`: Refer to [`measure_ssh_energy`](@ref).
- `ssh_energy_dn`: Refer to [`measure_ssh_energy`](@ref).
- `dispersion_energy`: Refer to [`measure_dispersion_energy`](@ref).
"""
function initialize_measurements!(
    measurement_container::NamedTuple,
    electron_phonon_model::ElectronPhononModel{T, E, D}
) where {T<:Number, E<:AbstractFloat, D}

    (; local_measurements, phonon_basis_vecs) = measurement_container
    (; phonon_modes, holstein_couplings_up, ssh_couplings_up, phonon_dispersions) = electron_phonon_model

    _initialize_measurements!(local_measurements, phonon_modes)
    _initialize_measurements!(local_measurements, holstein_couplings_up)
    _initialize_measurements!(local_measurements, ssh_couplings_up)
    _initialize_measurements!(local_measurements, phonon_dispersions)

    # Record the basis vector for each type of phonon mode
    for phonon_mode in phonon_modes
        push!(phonon_basis_vecs, phonon_mode.basis_vec)
    end

    return nothing
end

# phonon mode related measurements
function _initialize_measurements!(
    local_measurements::Dict{String, Vector{Complex{T}}},
    phonon_modes::Vector{PhononMode{T,D}}
) where {T<:AbstractFloat, D}

    # number of phonon modes
    n_modes = length(phonon_modes)

    # add measurements
    local_measurements["phonon_kin_energy"] = zeros(Complex{T}, n_modes)
    local_measurements["phonon_pot_energy"] = zeros(Complex{T}, n_modes)
    local_measurements["X"]  = zeros(Complex{T}, n_modes)
    local_measurements["X2"] = zeros(Complex{T}, n_modes)
    local_measurements["X3"] = zeros(Complex{T}, n_modes)
    local_measurements["X4"] = zeros(Complex{T}, n_modes)

    return nothing
end

# holstein related coupling measurements
function _initialize_measurements!(local_measurements::Dict{String, Vector{Complex{T}}},
                                   holstein_couplings::Vector{HolsteinCoupling{T,D}}) where {T<:AbstractFloat, D}

    # number of phonon modes
    n_couplings = length(holstein_couplings)

    # add measurements
    local_measurements["holstein_energy"]    = zeros(Complex{T}, n_couplings)
    local_measurements["holstein_energy_up"] = zeros(Complex{T}, n_couplings)
    local_measurements["holstein_energy_dn"] = zeros(Complex{T}, n_couplings)

    return nothing
end

# ssh coupling related measurements
function _initialize_measurements!(local_measurements::Dict{String, Vector{Complex{E}}},
                                   ssh_couplings::Vector{SSHCoupling{T,E,D}}) where {T<:Number, E<:AbstractFloat, D}

    # number of phonon modes
    n_couplings = length(ssh_couplings)

    # add measurements
    local_measurements["ssh_energy"]        = zeros(Complex{E}, n_couplings)
    local_measurements["ssh_energy_up"]     = zeros(Complex{E}, n_couplings)
    local_measurements["ssh_energy_dn"]     = zeros(Complex{E}, n_couplings)

    return nothing
end

# phonon dispersion related measurements
function _initialize_measurements!(local_measurements::Dict{String, Vector{Complex{T}}},
                                   phonon_dispersions::Vector{PhononDispersion{T,D}}) where {T<:AbstractFloat, D}

    # number of phonon modes
    n_couplings = length(phonon_dispersions)

    # add measurements
    local_measurements["dispersion_energy"] = zeros(Complex{T}, n_couplings)

    return nothing
end


#########################################
## INITIALIZE CORRELATION MEASUREMENTS ##
#########################################

@doc raw"""
    initialize_correlation_measurements!(;
        measurement_container::NamedTuple,
        model_geometry::ModelGeometry{D,T,N},
        correlation::String,
        pairs::AbstractVector{NTuple{2,Int}},
        time_displaced::Bool,
        integrated::Bool = false
    )  where {T<:AbstractFloat, D, N}

Initialize measurements of `correlation` for all ID pairs; refer to [`CORRELATION_FUNCTIONS`](@ref) for ID type associated
with each correlation measurement.
The name `correlation` must therefore also appear in [`CORRELATION_FUNCTIONS`]@ref.
If `time_displaced = true` then time-displaced and integrated correlation measurements are made.
If `time_displaced = false` and `integrated = false`, then just equal-time correlation measurements are made.
If `time_displaced = false` and `integrated = true`, then both equal-time and integrated correlation measurements are made.
"""
function initialize_correlation_measurements!(;
    measurement_container::NamedTuple,
    model_geometry::ModelGeometry{D,T,N},
    correlation::String,
    pairs::AbstractVector{NTuple{2,Int}},
    time_displaced::Bool,
    integrated::Bool = false
)  where {T<:AbstractFloat, D, N}

    # set integrated to false for electron green's function
    if (integrated == true) && (correlation ∈ ("greens", "greens_up", "greens_dn"))
        integrated = false
    # if time-displaced measurements are being made then also make integrated measurements
    elseif time_displaced == true
        integrated = true
    end

    # iterate over all bond/orbial ID pairs
    for pair in pairs
        initialize_correlation_measurement!(
            measurement_container = measurement_container,
            model_geometry = model_geometry,
            correlation = correlation,
            pair = pair,
            time_displaced = time_displaced,
            integrated = integrated
        )
    end

    return nothing
end

# initialize a single correlation measurement
function initialize_correlation_measurement!(;
    measurement_container::NamedTuple,
    model_geometry::ModelGeometry{D,T,N},
    correlation::String,
    pair::NTuple{2,Int},
    time_displaced::Bool,
    integrated::Bool = false
)  where {T<:AbstractFloat, D, N}

    (; time_displaced_correlations, integrated_correlations, equaltime_correlations) = measurement_container

    # check to make sure valid correlation measurement
    @assert correlation in keys(CORRELATION_FUNCTIONS)

    # extent of lattice in unit cells
    L = measurement_container.L

    # length of imaginary time axis
    Lτ = measurement_container.Lτ

    # if time displaced or integrated measurement should be made
    if time_displaced || integrated

        # add time-displaced and integrated correlation key if not present
        if !haskey(time_displaced_correlations, correlation)
            time_displaced_correlations[correlation] = CorrelationContainer(D+1, T, time_displaced)
            integrated_correlations[correlation] = CorrelationContainer(D, T, false)
        end

        # add time-dispalced correlation measurement
        push!(time_displaced_correlations[correlation].id_pairs, pair)
        push!(time_displaced_correlations[correlation].correlations, zeros(Complex{T}, L..., Lτ+1))

        # add integrated correlation measurement
        push!(integrated_correlations[correlation].id_pairs, pair)
        push!(integrated_correlations[correlation].correlations, zeros(Complex{T}, L...))
    end

    # if equal-time measurement should be made
    if !time_displaced

        # add equal-time correlation key if not present
        if !haskey(equaltime_correlations, correlation)
            equaltime_correlations[correlation] = CorrelationContainer(D, T, false)
        end

        # add equal-time correlation measurement
        push!(equaltime_correlations[correlation].id_pairs, pair)
        push!(equaltime_correlations[correlation].correlations, zeros(Complex{T}, L...))
    end

    return nothing
end


###################################################
## INITIALIZE COMPOSITE CORRELATION MEASUREMENTS ##
###################################################

@doc raw"""
    initialize_composite_correlation_measurement!(;
        measurement_container::NamedTuple,
        model_geometry::ModelGeometry{D,T,N},
        name::String,
        correlation::String,
        ids::Union{Nothing,Vector{Int}} = nothing,
        id_pairs::Union{Nothing,Vector{NTuple{2,Int}}} = nothing,
        coefficients,
        displacement_vecs = nothing,
        time_displaced::Bool,
        integrated::Bool = false
    )  where {T<:AbstractFloat, D, N}

Initialize a composite correlation measurement called `name` based
on a linear combination of local operators used in a standard `correlation` measurement.

If the keyword `ids` is passed and `id_pairs = nothing`, then the composite correlation function is given by
```math
\begin{align*}
    C_{\mathbf{r}}(\tau) & = \frac{1}{N}\sum_{\mathbf{i}}\langle\hat{\Phi}_{\mathbf{i}+\mathbf{r}}^{\dagger}(\tau)\hat{\Phi}_{\mathbf{i}}^{\phantom{\dagger}}(0)\rangle \\
                         & = \frac{1}{N}\sum_{\eta,\nu}\sum_{\mathbf{i}}c_{\eta}^{*}c_{\nu}\langle\hat{O}_{\mathbf{i}+\mathbf{r},\eta}^{\dagger}(\tau)\hat{O}_{\mathbf{i},\nu}^{\phantom{\dagger}}(0)\rangle \\
                         & = \sum_{\eta,\nu}c_{\eta}^{*}c_{\nu}C_{\mathbf{r}}^{\eta,\nu}(\tau)
\end{align*}
```
where the composite operator is
```math
\hat{\Phi}_{\mathbf{\mathbf{r}}}(\tau)=\sum_{\nu}c_{\nu}\hat{O}_{\mathbf{r},\nu}(\tau).
```
The sum over ``\mathbf{i}`` runs over all unit cells, ``\mathbf{r}`` denotes a displacement in unit cells and ``N`` is the number unit cells.
The operator type ``\hat{O}^{\nu}`` and corresponding correlation function type ``C_{\mathbf{r}}^{\eta,\nu}(\tau)`` are specified by the `correlation` keyword,
while ``\nu`` corresponds to labels/IDs specified by the `ids` keyword argument.
Lastly, the ``c_\nu`` coefficients are specified using the `coefficients` keyword arguments.
The corresponding fourier transform of this composite correlation function measurement is given by
```math
S_{\mathbf{q}}(\tau)=\sum_{\eta,\nu}\sum_{\mathbf{r}}e^{-{\rm i}\mathbf{q}\cdot(\mathbf{r}+\mathbf{r}_{\eta}-\mathbf{r}_{\nu})}C_{\mathbf{r}}^{\eta,\nu}(\tau),
```
where the static vectors ``\mathbf{r}_\nu`` are specified using the `displacement_vecs` keyword arguments.
If `displacement_vecs = nothing` then ``\mathbf{r}_\nu = 0`` for all label/ID values ``\nu``.

On the other hand, if `id_pairs` is passed and `ids = nothing`, then the composite correlation function is given by
```math
\begin{align*}
    C_{\mathbf{r}}(\tau) & = \sum_{n}c_{n}C_{\mathbf{r}}^{\eta_{n},\nu_{n}}(\tau) \\
                         & = \frac{1}{N}\sum_{n}\sum_{\mathbf{i}}c_{n}\langle\hat{O}_{\mathbf{i}+\mathbf{r},\eta_{n}}^{\dagger}(\tau)\hat{O}_{\mathbf{i},\nu_{n}}^{\phantom{\dagger}}(0)\rangle,
\end{align*}
```
where the ``n`` index runs over pairs of labels/IDs ``(\nu_n, \eta_n)`` specified by the `id_pairs` keyword argument.
Note that the order of the label/ID pair ``(\nu_n, \eta_n)`` reflects how each tuple in the `id_pairs` vector will be interpreted.
Once again, operator type ``\hat{O}^{\nu_n}`` and corresponding correlation function type ``C_{\mathbf{r}}^{\eta_n,\nu_n}(\tau)`` are specified by the `correlation` keyword.
The corresponding fourier transform of this composite correlation function measurement is given by
```math
S_{\mathbf{q}}(\tau)=\sum_{n}\sum_{\mathbf{r}}e^{-{\rm i}\mathbf{q}\cdot(\mathbf{r}+\mathbf{r}_{n})}C_{\mathbf{r}}^{\eta_{n},\nu_{n}}(\tau),
```
where the static displacement vectors ``\mathbf{r}_n`` are specified by the `displacement_vecs` keyword argument.
As before, if `displacement_vecs = nothing`, then ``\mathbf{r}_n = 0`` for all ``n``.

Note that the specified correlation type `correlation` needs to correspond to one of the keys in the global
[`CORRELATION_FUNCTIONS`](@ref) dictionary, which lists all the predefined types of correlation functions that can be measured.
"""
function initialize_composite_correlation_measurement!(;
    measurement_container::NamedTuple,
    model_geometry::ModelGeometry{D,T,N},
    name::String,
    correlation::String,
    ids::Union{Nothing,Vector{Int}} = nothing,
    id_pairs::Union{Nothing,Vector{NTuple{2,Int}}} = nothing,
    coefficients,
    displacement_vecs = nothing,
    time_displaced::Bool,
    integrated::Bool = false
)  where {T<:AbstractFloat, D, N}

    (; time_displaced_composite_correlations,
       integrated_composite_correlations,
       equaltime_composite_correlations,
       L, Lτ
    ) = measurement_container

    @assert correlation in keys(CORRELATION_FUNCTIONS)
    @assert(
        !(isnothing(ids) && isnothing(id_pairs)),
        "One of the keywords `ids` or `id_pairs` needs to be assigned."
    )
    @assert(
        !(!isnothing(ids) && !isnothing(id_pairs)),
        "Only one of the keywords `ids` or `id_pairs` should be assigned."
    )

    # set integrated to false for electron green's function
    if (integrated == true) && (correlation ∈ ("greens", "greens_up", "greens_dn"))
        integrated = false
    # if time-displaced measurements are being made then also make integrated measurements
    elseif time_displaced == true
        integrated = true
    end

    if isa(ids, Vector{Int}) && isa(id_pairs, Nothing)
        @assert length(ids) == length(coefficients) "Length of `ids` and `coefficients` do not match."
        displacement_vecs = isnothing(displacement_vecs) ? [zeros(T,D) for i in ids] : displacement_vecs
        @assert length(ids) == length(displacement_vecs) "Length of `ids` and `displacement_vecs` do not match."
        coefs = Complex{T}[]
        id_pairs = NTuple{2,Int}[]
        dvecs = SVector{D,T}[]
        for j in eachindex(ids)
            for i in eachindex(ids)
                push!( coefs, conj(coefficients[i]) * coefficients[j] )
                push!( id_pairs, (ids[j], ids[i]) )
                push!( dvecs, SVector{D,T}(displacement_vecs[i]) - SVector{D,T}(displacement_vecs[j]))
            end
        end
    else
        @assert length(id_pairs) == length(coefficients)
        coefs = Complex{T}[coefficients...]
        dvecs = isnothing(displacement_vecs) ? [SVector{D,T}(zeros(T,D)) for c in coefs] : [SVector{D,T}(v) for v in displacement_vecs]
        @assert length(dvecs) == length(coefs) "Length of `coefficients` and `displacement_vecs` do not match."
    end

    # if time displaced or integrated measurement should be made
    if time_displaced || integrated
        time_displaced_composite_correlations[name] = CompositeCorrelationContainer(
            Lτ, L, correlation, id_pairs, coefs, time_displaced, dvecs
        )
    end

    # if integrated measurement is made
    if integrated
        integrated_composite_correlations[name] = CompositeCorrelationContainer(
            L, correlation, id_pairs, coefs, dvecs
        )
    end

    # if equal-time measurement should be made
    if !time_displaced
        equaltime_composite_correlations[name] = CompositeCorrelationContainer(
            L, correlation, id_pairs, coefs, dvecs
        )
    end

    return nothing
end