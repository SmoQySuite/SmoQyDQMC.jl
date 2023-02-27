@doc raw"""
    global_measurement_bins_to_csv(folder::String)

Write the binned global measurements to file.
"""
function global_measurement_bins_to_csv(folder::String)

    # get the number of processes that ran during simulation
    N_process = _get_num_processes(folder)

    # get the number of files in each measurement directory
    N_files = length(readdir(joinpath(folder,"global")))

    # get the number of measurements per process
    N_bin = div(N_files, N_process)

    # directory where binned global data is
    global_folder = joinpath(folder, "global")

    # filename for global data
    filename = joinpath(folder, "global_bins.csv")

    # load sample global measurement
    sample_global_measurements = JLD2.load(joinpath(global_folder, "bin-1_pID-0.jld2"))

    # get the measurements
    measurements = sort(collect(keys(sample_global_measurements)))

    # open csv file
    open(filename, "w") do fout

        # write the file header
        write(fout, "BIN PID")
        for measurement in measurements
            @printf(fout, " %s_R %s_I", uppercase(measurement), uppercase(measurement))
        end
        write(fout, "\n")

        # iterate over measurements
        for bin in 1:N_bin

            # iterate over processes
            for pID in 0:N_process-1

                # write global measurement to file
                _write_global_measurements(fout, global_folder, measurements, bin, pID)
            end
        end
    end

    return nothing
end

# write global measurement to file
function _write_global_measurements(fout::IO, global_folder::String, measurements::Vector{String}, bin::Int, pID::Int)

    # load global measurement
    filename = @sprintf("bin-%d_pID-%d.jld2", bin, pID)
    global_measurement = JLD2.load(joinpath(global_folder, filename))

    # write measurement
    @printf fout "%d %d" bin pID
    for measurement in measurements
        val = global_measurement[measurement]
        @printf fout " %.8f %.8f" real(val) imag(val)
    end
    write(fout, "\n")

    return nothing
end


@doc raw"""
    local_measurement_bins_to_csv(folder::String, measurement::String)

Write the binned values for the local measurement `measurement` to a CSV file.
"""
function local_measurement_bins_to_csv(folder::String, measurement::String)

    # directory where binned global data is located
    global_folder = joinpath(folder, "global")

    # directory where locationl measurement data is located
    local_folder = joinpath(folder, "local")

    # get the number of processes that ran during simulation
    N_process = _get_num_processes(folder)

    # get the number of files in each measurement directory
    N_file = length(readdir(global_folder))

    # get the number of measurements per process
    N_bin = div(N_file, N_process)

    # get the number of IDs
    N_id = length( JLD2.load(joinpath(global_folder, "bin-1_pID-0.jld2"), measurement) )

    # filename for global data
    filename = joinpath(folder, @sprintf("%s_bins.csv", measurement))

    # open csv file
    open(filename, "w") do fout

        # write the file header
        @printf fout "BIN PID ID %s_R %s_I SIGN_R SIGN_I\n" uppercase(measurement) uppercase(measurement)

        # iterate over bins
        for bin in 1:N_bin

            # iterate over processes
            for pID in 0:N_process-1

                # write local measurement
                _write_local_measurement(fout, global_folder, local_folder, measurement, bin, pID, N_id)
            end
        end
    end

    return nothing
end

# write local measurement to file
function _write_local_measurement(fout::IO, global_folder::String, local_folder::String, measurement::String, bin::Int, pID::Int, N_id::Int)

    # get the average sign
    sgn = JLD2.load( joinpath(global_folder, @sprintf("bin-%d_pID-%d.jld2", bin, pID)), "sgn" )

    # load the local measurement
    local_measurement = JLD2.load( joinpath(local_folder, @sprintf("bin-%d_pID-%d.jld2", bin, pID)), measurement )

    # iterate of IDs
    for id in 1:N_id

        # write measurement to file
        @printf fout "%d %d %d %.8f %.8f %.8f %.8f\n" bin pID id real(local_measurement[id]) imag(local_measurement[id]) real(sgn) imag(sgn)
    end

    return nothing
end


@doc raw"""
    correlation_bins_to_csv(; folder::String, correlation::String, type::String, space::String, write_index_key::Bool = true)

Write binned correlation data for `correlation` to a CSV file.
The field `type` must be set equal to `"equal-time"`, `"time-displaced"` or `"integrated"`,
and the field `space` but bet set to either `"position"` or `"momentum"`.
"""
function correlation_bins_to_csv(; folder::String, correlation::String, type::String, space::String, write_index_key::Bool = true)

    @assert (type == "equal-time") || (type == "time-displaced") || (type == "integrated")
    @assert (space == "momentum") || (space == "position")

    # write key for indices
    if write_index_key
        _write_correlation_index_key(folder, correlation, type, space)
    end

    # correlation data directory
    correlation_folder = joinpath(folder, type, correlation)

    # momentum or position space folder holding bins
    space_folder = joinpath(correlation_folder, space)

    # global measurement direcotry
    global_folder = joinpath(folder, "global")

    # key filename
    filename = @sprintf "%s_%s_%s_bins.csv" correlation space type

    # get the number of processes that ran during simulation
    N_process = _get_num_processes(folder)

    # get the number of files in each measurement directory
    N_files = length(readdir(joinpath(folder,"global")))

    # get the number of bins
    N_bin = div(N_files, N_process)

    # open csv file
    open(joinpath(correlation_folder, filename), "w") do fout

        # write header
        @printf fout "BIN PID INDEX %s_R %s_I SIGN_R SIGN_I\n" uppercase(correlation) uppercase(correlation)

        # iterate over bins
        for bin in 1:N_bin

            # iterate over processes
            for pID in 0:N_process-1

                # write correlation measurement to csv
                _write_correlations_to_csv(fout, space_folder, global_folder, bin, pID)
            end
        end
    end

    return nothing
end

# write correlation function index key for time-dependent correlation
function _write_correlation_index_key(folder::String, correlation::String, type::String, space::String)

    # correlation data directory
    correlation_folder = joinpath(folder, type, correlation)

    # momentum or position space folder holding bins
    space_folder = joinpath(correlation_folder, space)

    # key filename
    filename = @sprintf "%s_%s_%s_index_key.csv" correlation space type

    # load sample correlation data
    sample_data = JLD2.load(joinpath(space_folder, "bin-1_pID-0.jld2"))
    correlations = sample_data["correlations"]
    pairs = sample_data["pairs"]

    # get the spatical dimension of the system
    if type == "time-displaced"
        D = ndims(correlations[1]) - 1
    else
        D = ndims(correlations[1])
    end

    # open key file
    open(joinpath(correlation_folder, filename), "w") do fout

        # write header to file
        write(fout, "Index ID_2 ID_1")
        if type == "time-displaced"
            write(fout, " Tau")
        end
        for d in D:-1:1
            if space == "position"
                @printf fout " R_%d" d
            else
                @printf fout " K_%d" d
            end
        end
        write(fout, "\n")

        # index counter initialized to zero
        index = 0

        # iterate of pairs of IDs
        for i in eachindex(pairs)

            # get ID pair and correlations
            pair = pairs[i]
            correlation = correlations[i]

            # iterate over each correlation respectively
            for c in CartesianIndices(correlation)

                # increment index counter
                index += 1

                # write key for index
                @printf fout "%d %d %d" index pair[2] pair[1]
                for n in reverse(eachindex(c.I))
                    @printf fout " %d" (c.I[n]-1)
                end
                write(fout, "\n")
            end
        end
    end

    return nothing
end

# write correlation measurement to csv file
function _write_correlations_to_csv(fout::IO, correlation_folder::String, global_folder::String, bin::Int, pID::Int)

    # read in correlation bin from jld2 file
    filename = @sprintf "bin-%d_pID-%d.jld2" bin pID
    correlation_bin = JLD2.load(joinpath(correlation_folder, filename))
    pairs = correlation_bin["pairs"]
    correlations = correlation_bin["correlations"]

    # initialize index to zero
    index = 0

    # get the sign associated with the measurement
    sgn = JLD2.load(joinpath(global_folder, filename), "sgn")

    # iterate over pairs
    for i in eachindex(pairs)

        # get the ID pair and corresponding correlations
        pair = pairs[i]
        correlation = correlations[i]

        # iterate over correlation elements
        for c in CartesianIndices(correlation)

            # increment the index
            index += 1

            # write the specific correlation value to file
            _write_correlation_to_csv(fout, bin, pID, index, pair, c.I, correlation[c], sgn)
        end
    end

    return nothing
end

# write specific correlaiton to file
function _write_correlation_to_csv(fout::IO, bin::Int, pID::Int, index::Int,
                                   pair::NTuple{2,Int}, loc::NTuple{1,Int},
                                   val::Complex{T}, sgn::Complex{T}) where {T<:AbstractFloat}

    @printf(fout, "%d %d %d%.8f %.8f %.8f %.8f\n",
            bin, pID, index, real(val), imag(val), real(sgn), imag(sgn))

    return nothing
end

function _write_correlation_to_csv(fout::IO, bin::Int, pID::Int, index::Int,
                                   pair::NTuple{2,Int}, loc::NTuple{2,Int},
                                   val::Complex{T}, sgn::Complex{T}) where {T<:AbstractFloat}

    @printf(fout, "%d %d %d %.8f %.8f %.8f %.8f\n",
            bin, pID, index, real(val), imag(val), real(sgn), imag(sgn))

    return nothing
end

function _write_correlation_to_csv(fout::IO, bin::Int, pID::Int, index::Int,
                                   pair::NTuple{2,Int}, loc::NTuple{3,Int},
                                   val::Complex{T}, sgn::Complex{T}) where {T<:AbstractFloat}

    @printf(fout, "%d %d %d %.8f %.8f %.8f %.8f\n",
            bin, pID, index, real(val), imag(val), real(sgn), imag(sgn))

    return nothing
end

function _write_correlation_to_csv(fout::IO, bin::Int, pID::Int, index::Int,
                                   pair::NTuple{2,Int}, loc::NTuple{4,Int},
                                   val::Complex{T}, sgn::Complex{T}) where {T<:AbstractFloat}

    @printf(fout, "%d %d %d %.8f %.8f %.8f %.8f\n",
            bin, pID, index, real(val), imag(val), real(sgn), imag(sgn))

    return nothing
end