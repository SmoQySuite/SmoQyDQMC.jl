@doc raw"""
    composite_correlation_stat(;
        folder::String,
        correlations::Vector{String},
        spaces::Vector{String},
        types::Vector{String},
        ids::Vector{NTuple{2,Int}},
        locs::Vector{NTuple{D,Int}},
        Δls::Vector{Int} = Int[],
        num_bins::Int = 0,
        pIDs::Vector{Int} = Int[],
        f::Function = identity
    ) where {D}

Calaculate the mean and error for a composite correlation measurement based on the function `f`.
Note that `D` indicates the spatial dimension of the system.


"""
function composite_correlation_stat(;
    folder::String,
    correlations::Vector{String},
    spaces::Vector{String},
    types::Vector{String},
    ids::Vector{NTuple{2,Int}},
    locs::Vector{NTuple{D,Int}},
    Δls::Vector{Int} = Int[],
    num_bins::Int = 0,
    pIDs::Vector{Int} = Int[],
    f::Function = identity
) where {D}

    # set the walkers to iterate over
    if isempty(pIDs)

        # get the number of MPI walkers
        N_walkers = get_num_walkers(folder)

        # get the pIDs
        pIDs = collect(0:(N_walkers-1))
    end

    # calculate composite correlation for first MPI walker
    C, ΔC = _composite_correlation_stat(folder, correlations, spaces, types, ids, locs, Δls, num_bins, pIDs[1], f)

    # if there is more than one MPI walker
    if length[pIDs] > 1
        # calculate the variance of the composite correlation
        VarC = ΔC^2
        # itereate over remaining pIDs
        for pID in pID[2:end]
            # calculate composite correlation stat for current MPI walker
            C′, ΔC′ = _composite_correlation_stat(folder, correlations, spaces, types, ids, locs, Δls, num_bins, pID, f)
            # update composite correlation stat
            C += C′
            VarC += abs2(ΔC′)
        end
        # normalize composite corrrelation stats
        C /= length(pIDs)
        ΔC = sqrt(VarC) / length(pIDs)
    end

    return C, ΔC
end

# calculate composite correlation stat for single MPI walker
function _composite_correlation_stat(
    folder::String,
    correlations::Vector{String},
    spaces::Vector{String},
    types::Vector{String},
    ids::Vector{NTuple{2,Int}},
    locs::Vector{NTuple{D,Int}},
    Δls::Vector{Int},
    num_bins::Int,
    pID::Int,
    f::Function
) where {D}

    # construct directory name containing binary data
    dirs = [joinpath(folder, types[n], correlations[n], spaces[n]) for n in 1:length(correlations)]

    # get the number of files/measurements
    N_files = length(glob(@sprintf("*_pID-%d.jld2", pID), dirs[1]))

    # defaults num_bins to N_files if num_bins is zero
    num_bins = iszero(num_bins) ? N_files : num_bins

    # get bin intervals
    bin_intervals = get_bin_intervals(folder, num_bins, pID)

    # get the binned average sign
    binned_sign = get_average_sign(folder, bin_intervals, pID)

    # allocate arrays to store binned data
    binned_correlations = zeros(eltype(binned_sign), num_bins, length(correlations))

    # iterate over correlation
    for i in eachindex(correlation)
        # get view to hold current binned correlation values
        binned_correlation = @view binned_correlations[:,i]
        # load binned correlation
        if types[i] == "time-displaced"
            _load_binned_correlation!(binned_correlation, bin_intervals, dirs[i], ids[i], locs[i], Δls[i], pID)
        else
            _load_binned_correlation!(binned_correlation, bin_intervals, dirs[i], ids[i], locs[i], -1, pID)
        end
    end

    # initialize version of function that take the sign as an argument
    F(z...) = f((z[n]/z[1] for n in 2:lastindex(z))...)

    # calculate the composite correlation
    C, ΔC = jackknife(F, binned_sign, binned_correlations...)

    return C, ΔC
end

# load binned equal-time or integrated correlation data
function _load_binned_correlation!(
    binned_vals::AbstractVector{Complex{T}},
    bin_intervals::Vector{UnitRange{Int}},
    dir::String,
    id_pair::NTuple{2,Int},
    loc::NTuple{D,Int},
    Δl::Int,
    pID::Int
) where {D, T<:AbstractFloat}

    # find the specified id pair
    id_pairs = JLD2.load(joinpath(dir,@sprintf("bin-1_pID-%d.jld2", pID)), "id_pairs")
    pair = findfirst(i -> i == id_pair, id_pairs)

    # initialize binned values to zero
    fill!(binned_vals, 0)

    # number of bins
    num_bins = length(bin_intervals)

    # calculate the bin size
    bin_size = length(bin_intervals[1])

    # iterate over bins
    for bin in 1:num_bins
        # iterate over bin elements
        for i in bin_intervals[bin]
            # construct filename
            file = joinpath(dir, @sprintf("bin-%d_pID-%d", i, pID))
            # load correlation data
            corr = OffsetArrays.Origin(0)(JLD2.load(file, "correlations")[pair])
            # record the relevant correlations
            if Δl < 0
                binned_vals[bin] += corr[loc...]
            else
                binned_vals[bin] += corr[loc..., Δl]
            end
        end
        # normalize binned data
        binned_vals[bin] /= bin_size
    end

    return nothing
end