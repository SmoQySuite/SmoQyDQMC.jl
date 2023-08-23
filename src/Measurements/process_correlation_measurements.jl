######################################
## PROCESS CORRELATION MEASUREMENTS ##
######################################

function process_correlation_measurement(
    folder::String,
    correlation::String,
    type::String,
    space::String,
    pIDs::Vector{Int} = Int[]
)

    return nothing
end

function process_correlation_measurement(
    folder::String,
    correlation::String,
    type::String,
    space::String,
    pID::Int
)

    return nothing
end


# read in equal-time or integrated correlation function
function read_correlation_measurement(
    folder::String,
    correlation::String,
    type::String,
    space::String,
    n_pair::Int,
    model_geometry::ModelGeometry{D,T,N},
    pID::Int,
    bin_intervals::Vector{UnitRange{Int}},
) where {D, T<:AbstractFloat, N}

    @assert type in ("equal-time", "integrated")
    @assert space in ("position", "momentum")

    # construct directory name where binary data lives
    correlation_folder = joinpath(folder, type, correlation, space)

    # get lattice size
    lattice = model_geometry.lattice::Lattice{D}
    L = lattice.L

    # number of bins
    N_bins = length(bin_intervals)

    # bin size
    N_binsize = length(bin_intervals[1])

    # container for binned correlation data
    binned_correlation = zeros(Complex{T}, N_bins, L...)

    # iterate over bins
    for bin in 1:N_bins

        # get a specific correlation bin
        correlation_bin = selectdim(binned_correlation, 1, bin)

        # iterate over bin elements
        for i in bin_intervals[bin]

            # load the correlation data
            corr = JLD2.load(@sprintf("%s/bin-%d_pID-%d.jl2", correlation_folder, i, pID))[n_pair]

            # record the correlation data
            @. correlation_bin += corr / N_binsize
        end
    end

    return binned_correlation
end

# read in time-displaced correlation function
function read_correlation_measurement(
    folder::String,
    correlation::String,
    l::Int,
    space::String,
    n_pair::Int,
    model_geometry::ModelGeometry{D,T,N},
    pID::Int,
    bin_intervals::Vector{UnitRange{Int}},
) where {D, T<:AbstractFloat, N}

    @assert space in ("position", "momentum")

    # construct directory name where binary data lives
    correlation_folder = joinpath(folder, "time-displaced", correlation, space)

    # get lattice size
    lattice = model_geometry.lattice::Lattice{D}
    L = lattice.L

    # number of bins
    N_bins = length(bin_intervals)

    # bin size
    N_binsize = length(bin_intervals[1])

    # container for binned correlation data
    binned_correlation = zeros(Complex{T}, N_bins, L...)

    # iterate over bins
    for bin in 1:N_bins

        # get a specific correlation bin
        correlation_bin = selectdim(binned_correlation, 1, bin)

        # iterate over bin elements
        for i in bin_intervals[bin]

            # load the correlation data
            corr = JLD2.load(@sprintf("%s/bin-%d_pID-%d.jl2", correlation_folder, i, pID))[n_pair]

            # get correlation for the specified time-slice
            corr_l = selectdim(corr, ndims(corr), l)

            # record the correlation data
            @. correlation_bin += corr_l / N_binsize
        end
    end

    return binned_correlation
end

# calculate the mean and error for binned correlations for single MPI walker
function analyze_correlations(
    binned_correlations::AbstractArray{Complex{T}},
    binned_sign::Vector{Complex{T}}
) where {T<:AbstractFloat}

    shape = size(correlations)
    correlations_avg = zeros(Complex{T}, shape[2:end])
    correlations_std = zeros(T, shape[2:end])

    # iterate over correlations
    for c in CartesianIndices(correlations_avg)

        # get all binned values
        binned_vals = @view binned_correlations[:, c]

        # calculate correlation stats
        C, ΔC = jackknife(/, binned_vals, binned_sign)
        correlations_avg[c] = C
        correlations_std[c] = ΔC
    end

    return correlations_avg, correlations_std
end


# write equal-time or integrated correlations to file
function write_correlation(
    fout::IO,
    pair::NTuple{2,Int},
    index::Int,
    correlations_avg::AbstractArray{Complex{T}},
    correlations_err::AbstractArray{Complex{T}}
) where {T<:AbstractFloat}

    # iterate over correlation values
    for c in CartesianIndices(correlations_avg)

        # increment index counter
        index += 1

        # write correlation stat to file
        C  = correlations_avg[c]
        ΔC = correlations_err[c]
        _write_correlation(fout, index, pair, c.I, C, ΔC)
    end

    return index
end

# write time-displace correlation to file
function write_correlation(
    fout::IO,
    pair::NTuple{2,Int},
    index::Int,
    l::Int,
    correlations_avg::AbstractArray{Complex{T}},
    correlations_err::AbstractArray{Complex{T}}
) where {T<:AbstractFloat}

    # iterate over correlation values
    for c in CartesianIndices(correlations_avg)

        # increment index counter
        index += 1

        # write correlation stat to file
        C  = correlations_avg[c]
        ΔC = correlations_err[c]
        _write_correlation(fout, index, pair, l, c.I, C, ΔC)
    end

    return index
end

# write equal-time or integrated correlation stat to file for D=1 dimensional system
function _write_correlation(fout::IO, index::Int, pair::NTuple{2,Int}, n::NTuple{1,Int},
                            C::Complex{T}, ΔC::T) where {T<:AbstractFloat}

    @printf(fout, "%d %d %d %d %.8f %.8f %.8f\n", index, pair[2], pair[1], n[1]-1, real(C), imag(C), ΔC)

    return nothing
end

# write equal-time or integrated correlation stat to file for D=2 dimensional system
function _write_correlation(fout::IO, index::Int, pair::NTuple{2,Int}, n::NTuple{2,Int},
                            C::Complex{T}, ΔC::T) where {T<:AbstractFloat}

    @printf(fout, "%d %d %d %d %d %.8f %.8f %.8f\n", index, pair[2], pair[1], n[2]-1, n[1]-1, real(C), imag(C), ΔC)

    return nothing
end

# write equal-time or integrated correlation stat to file for D=3 dimensional system
function _write_correlation(fout::IO, index::Int, pair::NTuple{2,Int}, n::NTuple{3,Int},
                            C::Complex{T}, ΔC::T) where {T<:AbstractFloat}

    @printf(fout, "%d %d %d %d %d %d %.8f %.8f %.8f\n", index, pair[2], pair[1], n[3]-1, n[2]-1, n[1]-1, real(C), imag(C), ΔC)

    return nothing
end

# write time-displaced correlation stat to file for D=1 dimensional system
function _write_correlation(fout::IO, index::Int, pair::NTuple{2,Int}, l::Int, n::NTuple{1,Int},
                            C::Complex{T}, ΔC::T) where {T<:AbstractFloat}

    @printf(fout, "%d %d %d %d %d %.8f %.8f %.8f\n", index, pair[2], pair[1], l-1, n[1]-1, real(C), imag(C), ΔC)

    return nothing
end

# write time-displaced correlation stat to file for D=2 dimensional system
function _write_correlation(fout::IO, index::Int, pair::NTuple{2,Int}, l::Int, n::NTuple{2,Int},
                                      C::Complex{T}, ΔC::T) where {T<:AbstractFloat}

    @printf(fout, "%d %d %d %d %d %d %.8f %.8f %.8f\n", index, pair[2], pair[1], l-1, n[2]-1, n[1]-1, real(C), imag(C), ΔC)

    return nothing
end

# write time-displaced correlation stat to file for D=3 dimensional system
function _write_correlation(fout::IO, index::Int, pair::NTuple{2,Int}, l::Int, n::NTuple{3,Int},
                            C::Complex{T}, ΔC::T) where {T<:AbstractFloat}

    @printf(fout, "%d %d %d %d %d %d %d %.8f %.8f %.8f\n", index, pair[2], pair[1], l-1, n[3]-1, n[2]-1, n[1]-1, real(C), imag(C), ΔC)

    return nothing
end