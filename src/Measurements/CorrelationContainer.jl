# Container to hold correlation function data.
struct CorrelationContainer{D, T<:AbstractFloat}

    # ID pairs to measure correlation function for
    id_pairs::Vector{NTuple{2,Int}}

    # correlation data for each pair of bond/orbital IDs getting measured
    correlations::Vector{Array{Complex{T}, D}}

    # whether or not the correlation measurement is time-displaced and will also be written to file.
    time_displaced::Bool
end

# Initialize and return an empty instance of CorrelationContainer for containing correlation data
# in a D dimensional array.
function CorrelationContainer(D::Int, T::DataType, time_displaced::Bool)

    correlation_container = CorrelationContainer(NTuple{2,Int}[], Array{Complex{T},D}[], time_displaced)

    return correlation_container
end

# Write correlation_container to a file with the name fn using the JLD2.jl package.
function save(fn::String, correlation_container::CorrelationContainer{D,T}) where {D, T<:AbstractFloat}

    jldsave(fn;
            id_pairs = correlation_container.id_pairs,
            correlations = correlation_container.correlations,
            time_displaced = correlation_container.time_displaced
    )

    return nothing
end

# Reset the correlation data stored in correlaiton_container to zero.
function reset!(correlaiton_container::CorrelationContainer{D,T}) where {D,T<:AbstractFloat}

    correlations = correlaiton_container.correlations
    for i in eachindex(correlations)
        fill!(correlations[i], zero(Complex{T}))
    end

    return nothing
end