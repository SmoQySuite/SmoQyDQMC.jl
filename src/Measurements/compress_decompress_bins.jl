@doc raw"""
    compress_jld2_bins(
        # ARGUMENTS
        comm::MPI.Comm;
        # KEYWORD ARGUMENTS
        folder::String,
        pID::Int = -1,
        compress::Bool = true
    )

    compress_jld2_bins(;
        # KEYWORD ARGUMENTS
        folder::String,
        pID::Int = -1,
        compress::Bool = true
    )

Combine the many JLD2 binary files containing the binned data into a single JLD2 file.
If `pID ≥ 0`, then only the binned files corresponding to the passed pID values merged into a
single file `binned_data_pID-$(pID).jld2`. If `pID = -1`, then all the binned files, regardless
are merged into a single filed called `binned_data.jld2`. Note that if many simulations were run
in parallel, this can take quite a while, and doing it for a single `pID` at a time may be advisable.
"""
function compress_jld2_bins(
    # ARGUMENTS
    comm::MPI.Comm;
    # KEYWORD ARGUMENTS
    folder::String,
    pID::Int = -1,
    compress::Bool = true
)

    if pID ≥ 0

        compress_jld2_bins(
            folder = folder,
            pID = pID,
            compress = compress
        )

    elseif iszero(MPI.Comm_rank(comm))

        compress_jld2_bins(
            folder = folder,
            pID = -1,
            compress = compress
        )

    end

    return nothing
end

function compress_jld2_bins(;
    # KEYWORD ARGUMENTS
    folder::String,
    pID::Int = -1,
    compress::Bool = true
)

    # construct filename
    if pID ≥ 0
        filename = joinpath(folder, "binned_data_pID-$(pID).jld2")
    else
        filename = joinpath(folder, "binned_data.jld2")
    end

    # write binary data to single JLD2 file
    jldopen(filename, "w"; compress = compress) do fout

        # write global binary data to file
        global_group = JLD2.Group(fout, "global")
        _load_bins(global_group, joinpath(folder, "global"), pID)

        # write local binary data to file
        local_group = JLD2.Group(fout, "local")
        _load_bins(local_group, joinpath(folder, "local"), pID)

        # write equal-time correlation binary data to file
        _load_correlation_bins(fout, folder, "equal-time", pID)

        # write equal-time correlation binary data to file
        _load_correlation_bins(fout, folder, "time-displaced", pID)

        # write equal-time correlation binary data to file
        _load_correlation_bins(fout, folder, "integrated", pID)
    end

    # delete all binary files
    delete_jld2_bins(
        folder = folder,
        pID = pID
    )

    return nothing
end


@doc raw"""
    decompress_jld2_bins(;
        # KEYWORD ARGUMENTS
        folder::String,
        pID::Int = -1
    )

Decompress compressed binned data stored in a single JLD2 file into the original
population of many JLD2 files, eaching containing the binned data for a certain type
of measurement. If `pID ≥ 0`, then the file `binary_data_pID-$(pID).jld2` will be
decompressed. If `pID = -1`, then the file `binary_data.jld2` will be decompressed.
"""
function decompress_jld2_bins(;
    # KEYWORD ARGUMENTS
    folder::String,
    pID::Int = -1
)

    # construct filename
    if pID ≥ 0
        filename = joinpath(folder, "binned_data_pID-$(pID).jld2")
    else
        filename = joinpath(folder, "binned_data.jld2")
    end

    # check if file exists
    if isfile(filename)

        # open compressed jld2 file
        jldopen(filename, "r") do fin

            # write global bins to file
            _write_bins(fin, folder, "global")

            # write local bins to file
            _write_bins(fin, folder, "local")

            # write equal-time correlation bins to file
            _write_correlation_bins(fin, folder, "equal-time")

            # write time-displaced correlation bins to file
            _write_correlation_bins(fin, folder, "time-displaced")

            # write integrated correlation bins to file
            _write_correlation_bins(fin, folder, "integrated")
        end

        # delete binary file
        rm(filename)
    end

    return nothing
end


@doc raw"""
    delete_jld2_bins(;
        # KEYWORD ARGUMENTS
        folder::String,
        pID::Int = -1
    )

Go through and delete all the binary files in the data folder directory.
Please be cautious, once done this operation cannot be undone, data will be lost permanently!
"""
function delete_jld2_bins(;
    # KEYWORD ARGUMENTS
    folder::String,
    pID::Int = -1
)

    # delete global binned binary files
    _delete_bins(joinpath(folder, "global"), pID)

    # delete local binned binary files
    _delete_bins(joinpath(folder, "local"), pID)

    # delte equal-time correlation bins
    _delete_correlation_bins(folder, "equal-time", pID)

    # delte time-displaced correlation bins
    _delete_correlation_bins(folder, "time-displaced", pID)

    # delte integrate correlation bins
    _delete_correlation_bins(folder, "integrated", pID)

    return nothing
end


# write jld2 binary file for a certain type of correlation function
function _write_correlation_bins(fin, folder, correlation_type)

    # check if there are equal-time correlation measurements
    if !isempty(fin[correlation_type])
        # iterate over correlation measurements
        for correlation in keys(fin[correlation_type])
            # write momentum space correlation bins
            _write_bins(fin, folder, correlation_type, correlation, "momentum")
            # write position space correlation bins
            _write_bins(fin, folder, correlation_type, correlation, "position")
        end
    end

    return nothing
end


# write jld2 binary file for each bin
function _write_bins(fin, folder, group_path...)

    subgroup = joinpath(group_path...)
    subdir = joinpath(folder, subgroup)
    group = fin[subgroup]
    for filename in keys(group)
        JLD2.save( joinpath(subdir, filename), group[filename] )
    end

    return nothing
end


# load all JLD2 binary files associated with a certain type of correlation function
# i.e. either equal-time, time-displaced or integrated correlation functions
function _load_correlation_bins(fout, folder, correlation_type, pID)

    # create a new group for the current correlation type
    correlation_type_group = JLD2.Group(fout, correlation_type)

    # directory containing correlations measurements of a given type
    correlation_type_folder = joinpath(folder, correlation_type)

    # read in correlation measurements of that type that were made
    correlation_measurements = readdir(correlation_type_folder)

    # check if any correlation measurements of the specified type were made
    if !isempty(correlation_measurements)

        # iterate over correlations measurements of the current type that were made
        for correlation_measurement in correlation_measurements

            # create group for current correlation measurement
            correlation_measurement_group = JLD2.Group(correlation_type_group, correlation_measurement)

            # create group for momentum space correlation measurements and then load them
            momentum_correlation_group = JLD2.Group(correlation_measurement_group, "momentum")
            momentum_correlation_folder = joinpath(correlation_type_folder, correlation_measurement, "momentum")
            _load_bins(momentum_correlation_group, momentum_correlation_folder, pID)

            # create group for momentum space correlation measurements and then load them
            position_correlation_group = JLD2.Group(correlation_measurement_group, "position")
            position_correlation_folder = joinpath(correlation_type_folder, correlation_measurement, "position")
            _load_bins(position_correlation_group, position_correlation_folder, pID)
        end
    end

    return nothing
end


# delete binned correlationd data
function _delete_correlation_bins(folder, correlation_type, pID)

    # directory containing correlations measurements of a given type
    correlation_type_folder = joinpath(folder, correlation_type)

    # read in correlation measurements of that type that were made
    correlation_measurements = readdir(correlation_type_folder)

    # check if any correlation measurements of the specified type were made
    if !isempty(correlation_measurements)

        # iterate over correlations measurements of the current type that were made
        for correlation_measurement in correlation_measurements

            # momentum space directory
            momentum_folder = joinpath(correlation_type_folder, correlation_measurement, "momentum")

            # momentum bin files
            momentum_files = readdir(momentum_folder, join = true)
            if pID ≥ 0
                momentum_files = filter(f -> endswith(f, "pID-$(pID).jld2"), momentum_files)
            end

            # delete momentum bin files
            for file in momentum_files
                rm(file)
            end

            # position space directory
            position_folder = joinpath(correlation_type_folder, correlation_measurement, "position")

            # position bin files
            position_files = readdir(position_folder, join = true)
            if pID ≥ 0
                position_files = filter(f -> endswith(f, "pID-$(pID).jld2"), position_files)
            end

            # delete position bin files
            for file in position_files
                rm(file)
            end
        end
    end

    return nothing
end


# load all the JLD2 binary files in a given directory.
# this function assumes the folder only contains JLD2 file.
function _load_bins(group, folder, pID)

    binary_files = readdir(folder)
    if pID ≥ 0
        binary_files = filter(f -> endswith(f, "pID-$(pID).jld2"), binary_files)
    end

    for file in binary_files
        group[file] = load(joinpath(folder, file))
    end

    return nothing
end


# delete all the JLD2 binary files in a given folder.
# this function assumes the folder only contains JLD2 file.
function _delete_bins(folder, pID)

    binary_files = readdir(folder)
    if pID ≥ 0
        binary_files = filter(f -> endswith(f, "pID-$(pID).jld2"), binary_files)
    end

    for file in binary_files
        rm(joinpath(folder, file))
    end

    return nothing
end