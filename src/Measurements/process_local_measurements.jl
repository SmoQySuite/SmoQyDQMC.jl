################################
## PROCESS LOCAL MEASUREMENTS ##
################################

@doc raw"""
    process_local_measurements(folder::String, N_bin::Int, pIDs::Union{Vector{Int},Int} = Int[])

Process local measurents for the specified process IDs, calculating the average and error for all local measurements
and writing the result to CSV file.
If `pIDs` is not specified, then the statistics are calculated using all MPI walker results.
"""
function process_local_measurements(folder::String, N_bin::Int, pIDs::Union{Vector{Int},Int} = Int[])

    _process_local_measurements(folder, N_bin, pIDs)
    return nothing
end

# process local measurements average across multiple MPI walkers
function _process_local_measurements(folder::String, N_bin::Int, pIDs::Vector{Int})

    # set the walkers to iterate over
    if length(pIDs) == 0

        # get the number of MPI walkers
        N_walkers = get_num_walkers(folder)

        # get the pIDs
        pIDs = collect(0:(N_walkers-1))
    end

    # calculate bin intervals
    bin_intervals = get_bin_intervals(folder, N_bin, pIDs[1])

    # get binned sign
    binned_sign = get_average_sign(folder, bin_intervals, pIDs[1])

    # read in binned local measurements for first pID
    binned_local_measurements = read_local_measurements(folder, pIDs[1], bin_intervals)

    # calculate local measurements stats
    local_measurements_avg, local_measurements_std = analyze_local_measurements(binned_local_measurements, binned_sign)

    # convert standard deviations to variance
    local_measurements_var = local_measurements_std
    for key in keys(local_measurements_avg)
        @. local_measurements_var[key] = abs2(local_measurements_std[key])
    end

    # iterate over remaining mpi walkers
    for pID in pIDs[2:end]

        # read in binned local measurements for current pID
        binned_local_measurements = read_local_measurements(folder, pID, bin_intervals)

        # get binned sign
        binned_sign = get_average_sign(folder, bin_intervals, pID)

        # calculate local measurements stats
        walker_local_measurements_avg, walker_local_measurements_std = analyze_local_measurements(binned_local_measurements, binned_sign)

        # iterate over measurements
        for key in keys(local_measurements_avg)

            # record measurement average
            @. local_measurements_avg[key] += walker_local_measurements_avg[key]

            # record measurement variance
            @. local_measurements_var[key] += abs2(walker_local_measurements_std[key])
        end
    end

    # calculate average for each measurement across all walkers and final standard deviation with errors properly propagated
    for key in keys(local_measurements_avg)
        N_id = length(pIDs)
        @. local_measurements_avg[key] /= N_id
        @. local_measurements_std[key]  = sqrt(local_measurements_var[key]) / N_id
    end

    # write the final local measurement stat to file
    write_local_measurements(folder, local_measurements_avg, local_measurements_std)

    return nothing
end

# process local measurements for a single MPI walker
function _process_local_measurements(folder::String, N_bin::Int, pID::Int)

    # calculate bin intervals
    bin_intervals = get_bin_intervals(folder, N_bin, pID)

    # get binned sign
    binned_sign = get_average_sign(folder, bin_intervals, pID)

    # read in binned local measurements for first pID
    binned_local_measurements = read_local_measurements(folder, pID, bin_intervals)

    # calculate local measurements stats
    local_measurements_avg, local_measurements_std = analyze_local_measurements(binned_local_measurements, binned_sign)

    # write the final local measurement stat to file
    write_local_measurements(folder, pID, local_measurements_avg, local_measurements_std)

    return nothing
end


# read in and bin the local measurements for single MPI walker given by pID
function read_local_measurements(folder::String, pID::Int, bin_intervals::Vector{UnitRange{Int}})
    
    # number of bins
    N_bin = length(bin_intervals)

    # bin size
    N_binsize = length(bin_intervals[1])

    # directory containing binary file local measurement data
    local_folder = joinpath(folder, "local")

    # get filename for sample local measurement
    local_measurement_file = @sprintf("%s/bin-1_pID-%d.jld2", local_folder, pID)

    # read in sample local measurement
    local_measurements = JLD2.load(local_measurement_file)

    # get data type
    T = eltype(local_measurements["density"])

    # initialize binned local measurements folder
    binned_local_measurements = Dict{String, Matrix{T}}()
    for key in keys(local_measurements)
        n = length(local_measurements[key])
        binned_local_measurements[key] = zeros(T, N_bin, n)
    end

    # iterate over bins
    for bin in 1:N_bin

        # iterate over bin elements
        for i in bin_intervals[bin]

            # load local measurements
            local_measurements = JLD2.load(@sprintf("%s/bin-%d_pID-%d.jld2", local_folder, i, pID))

            # iterate over local measurements
            for key in keys(local_measurements)

                # record local measurement
                @views @. binned_local_measurements[key][bin,:] +=  local_measurements[key] / N_binsize
            end
        end
    end

    return binned_local_measurements
end


# calculate the mean and error for local measurements for a single MPI walker, as specified by pID
function analyze_local_measurements(
    binned_local_measurements::Dict{String, Matrix{Complex{T}}},
    binned_sign::Vector{Complex{T}}
) where {T<:AbstractFloat}

    # initialize dictionaries to contain local measurement stats
    local_measurements_avg = Dict{String, Vector{Complex{T}}}()
    local_measurements_std = Dict{String, Vector{T}}()

    # iterate over measurements
    for key in keys(binned_local_measurements)
        # number of IDs
        N_id = size(binned_local_measurements[key],2)
        # initialize vector for measurement
        local_measurements_avg[key] = zeros(Complex{T}, N_id)
        local_measurements_std[key] = zeros(T, N_id)
        # iterate over each type of a given local measurement (orbital id as an example)
        for n in 1:N_id
            # get the binned values
            binned_vals = @view binned_local_measurements[key][:,n]
            # calculate stats
            avg, err = jackknife(/, binned_vals, binned_sign)
            # record the statistics
            local_measurements_avg[key][n] = avg
            local_measurements_std[key][n] = err
        end
    end

    return local_measurements_avg, local_measurements_std
end


# write the local measurements
function write_local_measurements(
    folder::String,
    local_measurements_avg::Dict{String, Vector{Complex{T}}},
    local_measurements_std::Dict{String, Vector{T}}
) where {T<:AbstractFloat}

    open(joinpath(folder,"local_stats.csv"), "w") do fout
        _write_local_measurements(fout, local_measurements_avg, local_measurements_std)
    end

    return nothing
end

# write the local measurements
function write_local_measurements(
    folder::String,
    pID::Int,
    local_measurements_avg::Dict{String, Vector{Complex{T}}},
    local_measurements_std::Dict{String, Vector{T}},
) where {T<:AbstractFloat}

    open(joinpath(folder,"local_stats_pID-$(pID).csv"), "w") do fout
        _write_local_measurements(fout, local_measurements_avg, local_measurements_std)
    end

    return nothing
end

# low-level write the local measurements
function _write_local_measurements(
    fout::IO,
    local_measurements_avg::Dict{String, Vector{Complex{T}}},
    local_measurements_std::Dict{String, Vector{T}}
) where {T<:AbstractFloat}

    # write header
    write(fout, "MEASUREMENT ID_TYPE ID MEAN_R MEAN_I STD\n")

    # iterate over measurements
    for key in keys(local_measurements_avg)

        # iterate over ID's associated with measurement
        for id in 1:length(local_measurements_avg[key])

            # write the measurement to file
            avg = local_measurements_avg[key][id]
            err = local_measurements_std[key][id]
            @printf(fout, "%s %s %d %.8f %.8f %.8f\n", key, LOCAL_MEASUREMENTS[key], id, real(avg), imag(avg), err)
        end
    end

    return nothing
end