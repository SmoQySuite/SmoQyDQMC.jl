# container to hold composite correlation data
struct CompositeCorrelationContainer{D, T<:AbstractFloat}

    # type of correlation function the composite corelation function is based on
    correlation::String

    # IDs of operatators appearing appearing in composite correlation measurement
    ids::Vector{Int}

    # coefficients of operators appearing in correlation measurement
    coefficients::Vector{Complex{T}}

    # array to contain composite correlation measurements
    composite_correlations::Array{Complex{T}, D}

    # whether time-displaced measurement that will be written to file during simulation
    time_displaced::Bool
end

# initialize composite correlation container for time-displaced measurement
function CompositeCorrelationContainer(
    T::DataType,
    Lτ::Int,
    L::AbstractVector{Int},
    correlation::String,
    ids,
    coefficients,
    time_displaced::Bool
)

    return CompositeCorrelationContainer(
        correlation, 
        Vector{Int}[ids...],
        Vector{Complex{T}}[coefficients...], 
        zeros(Complex{T}, L..., Lτ), 
        time_displaced
    )
end

function CompositeCorrelationContainer(
    T::DataType,
    L::AbstractVector{Int},
    correlation::String,
    ids,
    coefficients,
)

    return CompositeCorrelationContainer(
        correlation, 
        Vector{Int}[ids...],
        Vector{Complex{T}}[coefficients...], 
        zeros(Complex{T}, L...), 
        false
    )
end

# Write correlation_container to a file with the name fn using the JLD2.jl package.
function save(fn::String, composite_correlation_container::CompositeCorrelationContainer{D,T}) where {D, T<:AbstractFloat}

    jldsave(fn;
            correlation = composite_correlation_container.correlation,
            ids = composite_correlation_container.ids,
            coefficients = composite_correlation_container.coefficients,
            composite_correlations = composite_correlation_container.composite_correlations,
            time_displaced = composite_correlation_container.time_displaced
    )

    return nothing
end

# Reset the correlation data stored in correlaiton_container to zero.
function reset!(composite_correlation_container::CompositeCorrelationContainer{D,T}) where {D,T<:AbstractFloat}

    fill!(composite_correlation_container.composite_correlations, zero(Complex{T}))

    return nothing
end