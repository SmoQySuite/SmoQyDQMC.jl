# container to hold composite correlation data
struct CompositeCorrelationContainer{D, W, T<:AbstractFloat}

    # type of correlation function the composite corelation function is based on
    correlation::String

    # IDs of operators appearing appearing in composite correlation measurement
    id_pairs::Vector{NTuple{2,Int}}

    # coefficients of operators appearing in correlation measurement
    coefficients::Vector{Complex{T}}

    # array to contain composite correlation measurements
    correlations::Array{Complex{T}, W}

    # array to contain moment space structure factors
    structure_factors::Array{Complex{T}, W}

    # whether time-displaced measurement that will be written to file during simulation
    time_displaced::Bool

    # displacement vectors applied when performing fourier transforms of each term
    # in composite correlation measurement
    displacement_vecs::Vector{SVector{D,T}}
end

# initialize composite correlation container for time-displaced measurement
function CompositeCorrelationContainer(
    Lτ::Int,
    L::AbstractVector{Int},
    correlation::String,
    id_pairs::Vector{NTuple{2,Int}},
    coefficients::Vector{Complex{T}},
    time_displaced::Bool,
    displacement_vecs::Vector{SVector{D,T}}
) where {D, T<:AbstractFloat}

    @assert length(L) == D
    return CompositeCorrelationContainer{D,D+1,T}(
        correlation, 
        id_pairs,
        coefficients, 
        zeros(Complex{T}, L..., Lτ+1),
        zeros(Complex{T}, L..., Lτ+1),
        time_displaced,
        displacement_vecs
    )
end

# initialize composite correlation container for equal-time or integrated measurement
function CompositeCorrelationContainer(
    L::AbstractVector{Int},
    correlation::String,
    id_pairs::Vector{NTuple{2,Int}},
    coefficients::Vector{Complex{T}},
    displacement_vecs::Vector{SVector{D,T}}
) where {D, T<:AbstractFloat}

    @assert length(L) == D
    return CompositeCorrelationContainer{D,D,T}(
        correlation, 
        id_pairs,
        coefficients, 
        zeros(Complex{T}, L...),
        zeros(Complex{T}, L...),
        false,
        displacement_vecs
    )
end

# Reset the correlation data stored in correlation container to zero.
function reset!(composite_correlation_container::CompositeCorrelationContainer{D,W,T}) where {D,W,T<:AbstractFloat}

    fill!(composite_correlation_container.correlations, zero(Complex{T}))
    fill!(composite_correlation_container.structure_factors, zero(Complex{T}))

    return nothing
end