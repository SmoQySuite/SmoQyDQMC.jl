##############################################
## DEFINE TYPES OF CORRELATION MEASUREMENTS ##
##############################################

@doc raw"""
    const CORRELATION_FUNCTIONS = (
        "greens",
        "greens_up",
        "greens_dn",
        "phonon_greens",
        "density",
        "pair",
        "spin_x",
        "spin_z",
        "bond",
        "current"
    )

List of all the correlation functions that can be measured.
Correlation functions are well defined in both position and momentum space.
"""
const CORRELATION_FUNCTIONS = (
    "greens",
    "greens_up",
    "greens_dn",
    "phonon_greens",
    "density",
    "pair",
    "spin_x",
    "spin_z",
    "bond",
    "current"
)


@doc raw"""
    CorrelationContainer{D, T<:AbstractFloat}

Container to hold correlation function data.

# Fields

- `pairs::Vector{NTuple{2,Int}}`: Pairs of bond/orbital IDs to measure.
- `correlations::Vector{Array{Complex{T}, D}}`: Vector of arrays, where each array contains the correlation measurements for a bond/orbital ID pair.
- `time_displaced::Bool`: Whether or not the correlation measurement is time-displaced and will also be written to file.
"""
struct CorrelationContainer{D, T<:AbstractFloat}

    # bond/orbital ID pairs to measure correlation function for
    pairs::Vector{NTuple{2,Int}}

    # correlation data for each pair of bond/orbital IDs getting measured
    correlations::Vector{Array{Complex{T}, D}}

    # whether or not the correlation measurement is time-displaced and will also be written to file.
    time_displaced::Bool
end

@doc raw"""
    CorrelationContainer(D::Int, T::DataType, time_displaced::Bool)

Initialize and return an empty instance of  `CorrelationContainer` for containing correlation data
in a `D` dimensional array.
"""
function CorrelationContainer(D::Int, T::DataType, time_displaced::Bool)

    correlation_container = CorrelationContainer(NTuple{2,Int}[], Array{Complex{T},D}[], time_displaced)

    return correlation_container
end


@doc raw"""
    save(fn::String, correlation_container::CorrelationContainer{D,T}) where {D, T<:AbstractFloat}

Write `correlation_container` to a file with the name `fn` using the [`JLD2.jl`](https://github.com/JuliaIO/JLD2.jl.git) package.
"""
function save(fn::String, correlation_container::CorrelationContainer{D,T}) where {D, T<:AbstractFloat}

    jldsave(fn; pairs = correlation_container.pairs,
            correlations = correlation_container.correlations,
            time_displaced = correlation_container.time_displaced)

    return nothing
end


@doc raw"""
    reset!(correlaiton_container::CorrelationContainer{D,T}) where {D,T<:AbstractFloat}

Reset the correlation data stored in `correlaiton_container` to zero.
"""
function reset!(correlaiton_container::CorrelationContainer{D,T}) where {D,T<:AbstractFloat}

    correlations = correlaiton_container.correlations
    for i in eachindex(correlations)
        fill!(correlations[i], zero(Complex{T}))
    end

    return nothing
end


######################################
## INITIALIZE MEASUREMENT CONTAINER ##
######################################

@doc raw"""
    initialize_measurement_container(model_geometry::ModelGeometry{D,T,N}, ??::T, ????::T) where {T<:AbstractFloat, D, N}

Initialize and return a measurement container of type `NamedTuple`.
"""
function initialize_measurement_container(model_geometry::ModelGeometry{D,T,N}, ??::T, ????::T) where {T<:AbstractFloat, D, N}

    lattice   = model_geometry.lattice::Lattice{D}
    unit_cell = model_geometry.unit_cell::UnitCell{D,T,N}
    bonds     = model_geometry.bonds::Vector{Bond{D}}

    # number of orbitals per unit cell
    norbitals = unit_cell.n

    # length of imaginary time axis
    L?? = eval_length_imaginary_axis(??, ????)

    # size of lattice in unit cells in direction of each lattice vector
    L = lattice.L

    # initialize global measurements
    global_measurements = Dict{String, Complex{T}}(
        "density" => zero(Complex{T}), # average total density ???n???
        "double_occ" => zero(Complex{T}),
        "Nsqrd" => zero(Complex{T}), # total particle number square ???N?????
        "sgndetGup" => zero(Complex{T}), # sign(det(Gup))
        "sgndetGdn" => zero(Complex{T}), # sign(det(Gdn))
        "sgn" => zero(Complex{T}), # sign(det(Gup))???sign(det(Gdn))
    )

    # initialize local measurement
    local_measurements = Dict{String, Vector{Complex{T}}}(
        "density" => zeros(Complex{T}, norbitals), # average density for each orbital species
        "double_occ" => zeros(Complex{T}, norbitals), # average double occupancy for each orbital
    )

    # initialize time displaced correlation measurement dictionary
    equaltime_correlations = Dict{String, CorrelationContainer{D,T}}()

    # initialize time displaced correlation measurement dictionary
    time_displaced_correlations = Dict{String, CorrelationContainer{D+1,T}}()

    # initialize integrated correlation measurement dictionary
    integrated_correlations = Dict{String, CorrelationContainer{D,T}}()

    # initialize measurement container
    measurement_container = (
        global_measurements         = global_measurements,
        local_measurements          = local_measurements,
        equaltime_correlations      = equaltime_correlations,
        time_displaced_correlations = time_displaced_correlations,
        integrated_correlations     = integrated_correlations,
        L                           = L,
        L??                          = L??,
        a                           = zeros(Complex{T}, L..., L??),
        a???                          = zeros(Complex{T}, L..., L??),
        a???                          = zeros(Complex{T}, L..., L??),
    )

    return measurement_container
end


############################################################
## INITIALIZE MEASUREMENTS ASSOCIATED WITH VARIOUS MODELS ##
############################################################

@doc raw"""
    initialize_measurements!(measurement_container::NamedTuple,
                             tight_binding_model::TightBindingModel{T,E}) where {T<:Number, E<:AbstractFloat}

Initialize tight-binding model related measurements.

# Initialized Measurements

- `onsite_energy`: Refer to [`measure_onsite_energy`](@ref).
- `hopping_energy`: Refer to [`measure_hopping_energy`](@ref).
"""
function initialize_measurements!(measurement_container::NamedTuple,
                                  tight_binding_model::TightBindingModel{T,E}) where {T<:Number, E<:AbstractFloat}

    (; local_measurements, global_measurements) = measurement_container

    # number of orbitals per unit cell
    norbital = length(tight_binding_model.??_mean)

    # number of types of hoppings
    nhopping = length(tight_binding_model.t_bond_ids)

    # initialize chemical potential as global measurement
    global_measurements["chemical_potential"] = zero(Complex{E})

    # initialize on-site energy measurement
    local_measurements["onsite_energy"] = zeros(Complex{E}, norbital)

    # initialize hopping energy measurement
    if nhopping > 0
        local_measurements["hopping_energy"] = zeros(Complex{E}, nhopping)
    end

    return nothing
end


@doc raw"""
    initialize_measurements!(measurement_container::NamedTuple,
                             hubbard_model::HubbardModel{T}) where {T<:AbstractFloat}

Initialize Hubbard model related measurements.

# Initialized Measurements:

- `hubbard_energy`: Refer to [`measure_hopping_energy`](@ref).
"""
function initialize_measurements!(measurement_container::NamedTuple,
                                  hubbard_model::HubbardModel{T}) where {T<:AbstractFloat}

    (; local_measurements) = measurement_container

    # number of orbitals in unit cell
    norbital = length(local_measurements["density"])

    # initialize hubbard energy measurement U???nup???ndn
    local_measurements["hubbard_energy"] = zeros(Complex{T}, norbital)

    return nothing
end

@doc raw"""
    initialize_measurements!(measurement_container::NamedTuple,
                             electron_phonon_model::ElectronPhononModel{T, E, D}) where {T<:Number, E<:AbstractFloat, D}

Initialize electron-phonon model related measurements.

# Initialized Measurements:

- `phonon_kinetic_energy`: Refer to [`measure_phonon_kinetic_energy`](@ref).
- `phonon_potential_energy`: Refer to [`measure_phonon_potential_energy`](@ref).
- `X`: Measure ``\langle \hat{X} \rangle``, refer to [`measure_phonon_position_moment`](@ref).
- `X2`: Measure ``\langle \hat{X}^2 \rangle``, refer to [`measure_phonon_position_moment`](@ref).
- `X3`: Measure ``\langle \hat{X}^3 \rangle``, refer to [`measure_phonon_position_moment`](@ref).
- `X4`: Measure ``\langle \hat{X}^4 \rangle``, refer to [`measure_phonon_position_moment`](@ref).
- `holstein_energy`: Refer to [`measure_holstein_energy`](@ref).
- `ssh_energy`: Refer to [`measure_ssh_energy`](@ref).
- `ssh_sgn_switch`: Refer to [`measure_ssh_sgn_switch`](@ref).
- `dispersion_energy`: Refer to [`measure_dispersion_energy`](@ref).
"""
function initialize_measurements!(measurement_container::NamedTuple,
                                  electron_phonon_model::ElectronPhononModel{T, E, D}) where {T<:Number, E<:AbstractFloat, D}

    (; local_measurements) = measurement_container
    (; phonon_modes, holstein_couplings, ssh_couplings, phonon_dispersions) = electron_phonon_model

    _initialize_measurements!(local_measurements, phonon_modes)
    _initialize_measurements!(local_measurements, holstein_couplings)
    _initialize_measurements!(local_measurements, ssh_couplings)
    _initialize_measurements!(local_measurements, phonon_dispersions)

    return nothing
end

# phonon mode related measurements
function _initialize_measurements!(local_measurements::Dict{String, Vector{Complex{T}}},
                                   phonon_modes::Vector{PhononMode{T}}) where {T<:AbstractFloat}

    # number of phonon modes
    n_modes = length(phonon_modes)

    # add measurements
    local_measurements["phonon_kinetic_energy"] = zeros(Complex{T}, n_modes)
    local_measurements["phonon_potential_energy"] = zeros(Complex{T}, n_modes)
    local_measurements["X"] = zeros(Complex{T}, n_modes)
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
    local_measurements["holstein_energy"] = zeros(Complex{T}, n_couplings)

    return nothing
end

# ssh coupling related measurements
function _initialize_measurements!(local_measurements::Dict{String, Vector{Complex{E}}},
                                   ssh_couplings::Vector{SSHCoupling{T,E,D}}) where {T<:Number, E<:AbstractFloat, D}

    # number of phonon modes
    n_couplings = length(ssh_couplings)

    # add measurements
    local_measurements["ssh_energy"] = zeros(Complex{E}, n_couplings)
    local_measurements["ssh_sgn_switch"] = zeros(Complex{E}, n_couplings)

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
    initialize_correlation_measurements!(; measurement_container::NamedTuple,
                                         model_geometry::ModelGeometry{D,T,N},
                                         correlation::String, pairs::AbstractVector{NTuple{2,Int}},
                                         time_displaced::Bool,
                                         integrated::Bool = false)  where {T<:AbstractFloat, D, N}

Initialize measurements of `correlation` for all pairs of bond ID's in `pairs`.
The name `correlation` must appear in `CORRELATION_FUNCTIONS`.
If `time-displaced = true` then time-displaced and integrated correlation measurements are made.
If `time-displaced = false` and `integrated = false`, then just equal-time correlation measurements are made.
If `time-displaced = false` and `integrated = true`, then both equal-time and integrated correlation measurements are made.
"""
function initialize_correlation_measurements!(; measurement_container::NamedTuple,
                                              model_geometry::ModelGeometry{D,T,N},
                                              correlation::String, pairs::AbstractVector{NTuple{2,Int}},
                                              time_displaced::Bool,
                                              integrated::Bool = false)  where {T<:AbstractFloat, D, N}

    # iterate over all bond/orbial ID pairs
    for pair in pairs
        initialize_correlation_measurement!(measurement_container = measurement_container,
                                            model_geometry = model_geometry,
                                            correlation = correlation,
                                            pair = pair,
                                            time_displaced = time_displaced,
                                            integrated = integrated)
    end

    return nothing
end

@doc raw"""
    initialize_correlation_measurement!(; measurement_container::NamedTuple,
                                        model_geometry::ModelGeometry{D,T,N},
                                        correlation::String, pair::NTuple{2,Int},
                                        time_displaced::Bool,
                                        integrated::Bool = false)  where {T<:AbstractFloat, D, N}

Initialize a measurement of `correlation` between the pair of bond ID's `pair`.
The name `correlation` must appear in `CORRELATION_FUNCTIONS`.
If `time-displaced = true` then time-displaced and integrated correlation measurements are made.
If `time-displaced = false` and `integrated = false`, then just equal-time correlation measurements are made.
If `time-displaced = false` and `integrated = true`, then both equal-time and integrated correlation measurements are made.
"""
function initialize_correlation_measurement!(; measurement_container::NamedTuple,
                                             model_geometry::ModelGeometry{D,T,N},
                                             correlation::String, pair::NTuple{2,Int},
                                             time_displaced::Bool,
                                             integrated::Bool = false)  where {T<:AbstractFloat, D, N}

    (; time_displaced_correlations, integrated_correlations, equaltime_correlations) = measurement_container

    # check to make sure valid correlation measurement
    @assert correlation in CORRELATION_FUNCTIONS

    # extent of lattice in unit cells
    L = measurement_container.L

    # length of imaginary time axis
    L?? = measurement_container.L??

    # if time displaced or integrated measurement should be made
    if time_displaced || integrated

        # add time-displaced and integrated correlation key if not present
        if !haskey(time_displaced_correlations, correlation)
            time_displaced_correlations[correlation] = CorrelationContainer(D+1, T, time_displaced)
            integrated_correlations[correlation] = CorrelationContainer(D, T, false)
        end

        # add time-dispalced correlation measurement
        push!(time_displaced_correlations[correlation].pairs, pair)
        push!(time_displaced_correlations[correlation].correlations, zeros(Complex{T}, L..., L??+1))

        # add integrated correlation measurement
        push!(integrated_correlations[correlation].pairs, pair)
        push!(integrated_correlations[correlation].correlations, zeros(Complex{T}, L...))
    end

    # if equal-time measurement should be made
    if !time_displaced

        # add equal-time correlation key if not present
        if !haskey(equaltime_correlations, correlation)
            equaltime_correlations[correlation] = CorrelationContainer(D, T, false)
        end

        # add equal-time correlation measurement
        push!(equaltime_correlations[correlation].pairs, pair)
        push!(equaltime_correlations[correlation].correlations, zeros(Complex{T}, L...))
    end

    return nothing
end


################################################
## INITIALIZE MEASUREMENT DIRECTORY STRUCTURE ##
################################################

@doc raw"""
    initialize_measurement_directories(; simulation_info::SimulationInfo,
                                       measurement_container::NamedTuple)

Initialize the measurement directories for simulation.
"""
function initialize_measurement_directories(; simulation_info::SimulationInfo,
                                            measurement_container::NamedTuple)

    (; datafolder, resuming, pID) = simulation_info
    (; time_displaced_correlations, equaltime_correlations, integrated_correlations) = measurement_container

    # only initialize folders if pID = 0
    if iszero(pID) && !resuming

        # make global measurements directory
        global_directory = joinpath(datafolder, "global")
        mkdir(global_directory)

        # make local measurements directory
        local_directory = joinpath(datafolder, "local")
        mkdir(local_directory)

        # make equaltime correlation directory
        eqaultime_directory = joinpath(datafolder, "equal-time")
        mkdir(eqaultime_directory)

        # iterate over equal-time correlation measurements
        for correlation in keys(equaltime_correlations)

            # make directory for each individual eqaul-time correlation measurement
            equaltime_correlation_directory = joinpath(eqaultime_directory, correlation)
            mkdir(equaltime_correlation_directory)

            # create sub-directories for position and momentum space data
            mkdir(joinpath(equaltime_correlation_directory, "position"))
            mkdir(joinpath(equaltime_correlation_directory, "momentum"))
        end

        # make time-displaced correlation directory
        time_displaced_directory = joinpath(datafolder, "time-displaced")
        mkdir(time_displaced_directory)

        # make integrated correlation directory
        integrated_directory = joinpath(datafolder, "integrated")
        mkdir(integrated_directory)

        # iterate over integrated correlation measurements
        for correlation in keys(integrated_correlations)

            # make directory for integrated correlation measurement
            integrated_correlation_directory = joinpath(integrated_directory, correlation)
            mkdir(integrated_correlation_directory)

            # create sub-directories for position and momentum space time-displaced correlation measurements
            mkdir(joinpath(integrated_correlation_directory, "position"))
            mkdir(joinpath(integrated_correlation_directory, "momentum"))

            # check if also a time-displaced measurement should also be made
            if time_displaced_correlations[correlation].time_displaced

                # make directory for time-displaced correlation measurement
                time_displaced_correlation_directory = joinpath(time_displaced_directory, correlation)
                mkdir(time_displaced_correlation_directory)

                # create sub-directories for position and momentum space time-displaced correlation measurements
                mkdir(joinpath(time_displaced_correlation_directory, "position"))
                mkdir(joinpath(time_displaced_correlation_directory, "momentum"))
            end
        end
    end

    return nothing
end