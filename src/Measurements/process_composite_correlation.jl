@doc raw"""
    composite_correlation_stats(;
        folder::String,
        correlation::String,
        space::String,
        type::String,
        ids::Vector{NTuple{2,Int}},
        locs::Vector{NTuple{D,Int}},
        num_bins::Int,
        l::Int = 0,
        f::Function = identity
    ) where {D}

Calaculate the mean and error for a composite correlation measurement based on the function
Note that `D` indicates the spatial dimension of the system.

# Keyword Arguments

- `folder::String`: The directory all simulations results were written to.
- `correlation::String`: Name of the correlation in question that will be processed.
- `space::String`: The space of the measurement, either "position" or "momentum".
- `type::String`: The type of correlation measurement "eqaul-time", "time-displaced" or "integrated".
- `ids::Vector{NTuple{2,Int}}`: A vector or ID pairs to read.
- `locs::Vector{NTuple{D,Int}}`: A vector specifying either a displacement if `type = "position"`, or k-point if `type = "momentum"`.
- `num_bins::Int`: The number of bins that should be used to calculate the mean and standard deviation with using the jackknife method.
- `l::Int = 0`: Specifies the imaginary time slice, only used if `type = "time-displaced"`.
- `f::Function = identity`: The function used to construct the composite correlation function based on each correlation specified by `ids` and `locs`.

## Comments

For the `locs` argument, for a given location, i.e. `loc = locs[1]`, the `loc[d]` value
corresponds to either a displacement in the direction of the `d` lattice vector,
or corresponds to a k-point index relative to the `d` reciprocal lattice vector.
"""
function composite_correlation_stats(;
    folder::String,
    correlation::String,
    space::String,
    type::String,
    ids::Vector{NTuple{2,Int}},
    locs::Vector{NTuple{D,Int}},
    num_bins::Int,
    l::Int = 0,
    f::Function = identity
) where {D}

    @assert space == "momentum" || space == "position"
    @assert type == "equal-time" || type == "time-displaced" || type == "integrated"
    @assert haskey(CORRELATION_FUNCTIONS, correlation)
    @assert length(locs) == length(ids)
    if f==identity
        @assert length(locs) == 1 "`f = identity`` only works for single correlation function."
    end

    # get the directory the raw correlation data lives in
    dir = joinpath(folder, type, correlation, space)
    @assert isdir(dir) "`$(isdir)` is not a directory."

    # get the number of processes/simulations that ran in parallel
    N_process = _get_num_processes(folder)

    # get the number of files in each measurement directory
    N_files = length(readdir(dir))

    # get the number of measurements per process
    N_measurement = div(N_files, N_process)

    # calculate the bin size per process
    N_binsize = div(N_measurement, num_bins)

    # make sure the number of bins is valid
    @assert (num_bins * N_binsize) == N_measurement "([num_bins = $num_bins] * [N_binsize = $N_binsize]) != [N_measurement = $N_measurement]"

    # get filenames associated with each bin
    bin_to_filenames = _get_filenames_by_bin(num_bins, N_binsize, N_process)

    # get the binned sign data
    binned_sgn = _get_binned_sgn_data(folder, bin_to_filenames)

    # vector to contain all the correlation binned values
    C_bins = zeros(eltype(binned_sgn), num_bins)

    # iterate over bins
    for bin in eachindex(C_bins)

        # get the average value for the bin
        C_bins[bin] = _load_composite_correlation_bin(
            dir, bin_to_filenames[bin],
            type, ids, locs, l, f, Val(eltype(binned_sgn)), Val(length(ids))
        )
    end

    # calculate the mean and error with jackknife
    C_mean, C_std = jackknife(/, C_bins, binned_sgn)

    return C_mean, C_std
end

# read in the a composite correlation bin
function _load_composite_correlation_bin(dir::String, files::Vector{String}, type::String,
                                         ids::Vector{NTuple{2,Int}}, locs::Vector{NTuple{D,Int}}, l::Int,
                                         f::Function, ::Val{T}, ::Val{P}) where {D,T,P}


    # get the index associated with each pair ID
    pairs = @MVector zeros(Int, P)
    id_pairs = JLD2.load(joinpath(dir,files[1]), "id_pairs")
    for p in eachindex(ids)
        indx = findfirst(i -> i == ids[p], id_pairs)
        @assert !isnothing(indx) "The ID pair $(ids[p]) is invalid."
        pairs[p] = indx
    end

    if type == "time-displaced"
        C = _load_composite_correlation_bin(dir, files, pairs, locs, l, f, Val(T), Val(P))
    else
        C = _load_composite_correlation_bin(dir, files, pairs, locs, f, Val(T), Val(P))
    end

    return C
end

# load composite correlation bin for time-displaced measurement
function _load_composite_correlation_bin(dir::String, files::Vector{String},
                                         pairs::AbstractVector{Int}, locs::Vector{NTuple{D,Int}},
                                         l::Int, f::Function, ::Val{T}, ::Val{P}) where {D,T,P}
    
    # initialize binned composite correlation value
    C = zero(T)

    # iterate over files
    for i in eachindex(files)
        C += _load_composite_correlation(dir, files[i], pairs, locs, l, f, Val(T), Val(P))
    end

    # normalize bin value
    C /= length(files)
    
    return C
end

# load composite correlation bin for equal-time/integreated measurement
function _load_composite_correlation_bin(dir::String, files::Vector{String},
                                         pairs::AbstractVector{Int}, locs::AbstractVector{NTuple{D,Int}},
                                         f::Function, ::Val{T}, ::Val{P}) where {D,T,P}
    
    # initialize binned composite correlation value
    C = zero(T)

    # iterate over files
    for i in eachindex(files)
        C += _load_composite_correlation(dir, files[i], pairs, locs, f, Val(T), Val(P))
    end

    # normalize bin value
    C /= length(files)
    
    return C
end

# load time-displaced composite correlation from file
function _load_composite_correlation(dir::String, file::String,
                                     pairs::AbstractVector{Int}, locs::AbstractVector{NTuple{D,Int}}, l::Int,
                                     f::Function, ::Val{T}, ::Val{P}) where {D, T, P}

    # vector to contain each specific correlation value
    c = @MVector zeros(T, P)

    # load the correlation data
    corrs = JLD2.load(joinpath(dir,file), "correlations")::Vector{Array{T,D+1}}

    # extract each correlation
    for p in eachindex(pairs)
        corr = corrs[pairs[p]]::Array{T,D+1}
        corr0 = OffsetArrays.Origin(0)(corr) # index from 0 array
        loc  = locs[p]
        c[p] = corr0[loc..., l]
    end

    # calculate the composite correlation value
    C = f(c...)

    return C
end

# load equal-time or time-displaced composite correlation from file
function _load_composite_correlation(dir::String, file::String,
                                     pairs::AbstractVector{Int}, locs::AbstractVector{NTuple{D,Int}},
                                     f::Function, ::Val{T}, ::Val{P}) where {D, T, P}

    # vector to contain each specific correlation value
    c = @MVector zeros(T, P)

    # load the correlation data
    corrs = JLD2.load(joinpath(dir,file), "correlations")::Vector{Array{T,D}}

    # extract each correlation
    for p in eachindex(pairs)
        corr = corrs[pairs[p]]::Array{T,D}
        corr0 = OffsetArrays.Origin(0)(corr) # index from 0 array
        loc  = locs[p]
        c[p] = corr0[loc...]
    end

    # calculate the composite correlation value
    C = f(c...)

    return C
end