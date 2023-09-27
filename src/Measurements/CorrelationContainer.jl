@doc raw"""
    CorrelationContainer{D, T<:AbstractFloat}

Container to hold correlation function data.

# Fields

- `id_pairs::Vector{NTuple{2,Int}}`: ID pairs corresponding to relevant ID type for correlation measurement.
- `bond_id_pairs::Vector{NTuple{2,Int}}`: Bond ID pair corresponding to correlation measurement.
- `correlations::Vector{Array{Complex{T}, D}}`: Vector of arrays, where each array contains the correlation measurements for a bond/orbital ID pair.
- `time_displaced::Bool`: Whether or not the correlation measurement is time-displaced and will also be written to file.
"""
struct CorrelationContainer{D, T<:AbstractFloat}

    # ID pairs to measure correlation function for
    id_pairs::Vector{NTuple{2,Int}}

    # corresponding bond ID pairs
    bond_id_pairs::Vector{NTuple{2,Int}}

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

    correlation_container = CorrelationContainer(NTuple{2,Int}[], NTuple{2,Int}[], Array{Complex{T},D}[], time_displaced)

    return correlation_container
end


@doc raw"""
    save(fn::String, correlation_container::CorrelationContainer{D,T}) where {D, T<:AbstractFloat}

Write `correlation_container` to a file with the name `fn` using the [`JLD2.jl`](https://github.com/JuliaIO/JLD2.jl.git) package.
"""
function save(fn::String, correlation_container::CorrelationContainer{D,T}) where {D, T<:AbstractFloat}

    jldsave(fn;
            id_pairs = correlation_container.id_pairs,
            bond_id_pairs = correlation_container.bond_id_pairs,
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