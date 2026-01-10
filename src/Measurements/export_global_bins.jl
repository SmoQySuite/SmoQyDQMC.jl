@doc raw"""
    export_global_bins_to_h5(
        # ARGUMENTS
        comm::MPI.Comm;
        # KEYWORD ARGUMENTS
        datafolder::String,
        filename_prefix::String = "global_bins",
        pIDs::Union{Vector{Int},Int} = Int[],
        global_measurements::Vector{String} = String[] 
    )

    export_global_bins_to_h5(;
        # KEYWORD ARGUMENTS
        datafolder::String,
        filename_prefix::String = "global_bins",
        pIDs::Union{Vector{Int},Int} = Int[],
        global_measurements::Vector{String} = String[]
    )

Export the binned global measurements for specified process IDs `pIDs` to a
single HDF5 file. If `pIDs = Int[]`, then binned global measurements for all
process IDs are exported. You can specify a subset of specific global measurements
using the `global_measurements` keyword argument. If `global_measurements = String[]`,
then all global measurements are exported.
"""
function export_global_bins_to_h5(
    # ARGUMENTS
    comm::MPI.Comm;
    # KEYWORD ARGUMENTS
    datafolder::String,
    filename_prefix::String = "global_bins",
    pIDs::Union{Vector{Int},Int} = Int[],
    global_measurements::Vector{String} = String[]
)

    filename = nothing
    if iszero(MPI.Comm_rank(comm))
        filename = export_global_bins_to_h5(
            datafolder = datafolder,
            filename_prefix = filename_prefix,
            pIDs = pIDs,
            global_measurements = global_measurements
        )
    end
    MPI.Barrier(comm)

    return filename
end

function export_global_bins_to_h5(;
    # KEYWORD ARGUMENTS
    datafolder::String,
    filename_prefix::String = "global_bins",
    pIDs::Union{Vector{Int},Int} = Int[],
    global_measurements::Vector{String} = String[]
)

    # construct path of directory containing HDF5 bin files
    @assert isdir(datafolder)
    bin_folder = joinpath(datafolder, "bins")

    # if single pID is being exported
    if isa(pIDs, Int)
        # construct filename for output HDF5 file
        filename = joinpath(datafolder, @sprintf("%s_pID-%d.h5", filename_prefix, pIDs))
        pIDs = [pIDs,]
    # if multiple pIDs being exported
    else
        # construct filename for output HDF5 file
        filename = joinpath(datafolder, @sprintf("%s.h5", filename_prefix))
        # if binned data for all pIDs is being exported
        if isempty(pIDs)
            pIDs = collect( 0 : length(readdir(bin_folder)) - 1 )
        end
    end

    # HDF5 bin file names
    h5_bin_filenames = [joinpath(bin_folder, @sprintf("bins_pID-%d.h5", pID)) for pID in pIDs]

    # open all the input HDF5 bin files
    H5BinFiles = [h5open(file, "r") for file in h5_bin_filenames]

    # open HDF5 out file
    h5open(filename, "w") do H5GlobalFile

        # create group to contain measurements
        Measurements = create_group(H5GlobalFile, "GLOBAL")

        # record the pIDs
        attributes(H5GlobalFile)["PIDS"] = pIDs

        # dimension labels
        attributes(H5GlobalFile)["DIM_LABELS"] = ["BIN", "PID"]

        # get list of global measurements
        global_measurements = isempty(global_measurements) ? keys(H5BinFiles[1]["GLOBAL"]) : global_measurements

        # make sure the sign is included as a global measurement
        if !in("sgn", global_measurements)
            push!(global_measurements, "sgn")
        end

        # iterate over global measurements
        for key in global_measurements

            # record all bins of data
            Measurements[key] = stack(
                tuple((read(H5BinFile["GLOBAL"][key]) for H5BinFile in H5BinFiles)...)
            )
        end
    end

    # close all the input HDF5 bin files
    close.(H5BinFiles)

    return filename
end

@doc raw"""
    export_global_bins_to_csv(
        # ARGUMENTS
        comm::MPI.Comm;
        # KEYWORD ARGUMENTS
        datafolder::String,
        filename_prefix::String = "global_bins",
        pIDs::Union{Vector{Int},Int} = Int[],
        global_measurements::Vector{String} = String[],
        decimals::Int = 9,
        scientific_notation::Bool = false,
        delimiter::String = " "
    )

    export_global_bins_to_csv(;
        # KEYWORD ARGUMENTS
        datafolder::String,
        filename_prefix::String = "global_bins",
        pIDs::Union{Vector{Int},Int} = Int[],
        global_measurements::Vector{String} = String[],
        decimals::Int = 9,
        scientific_notation::Bool = false,
        delimiter::String = " "
    )

Export the binned global measurements for specified process IDs `pIDs` to a
single CSV file. If `pIDs = Int[]`, then binned global measurements for all
process IDs are exported. You can specify a subset of specific global measurements
using the `global_measurements` keyword argument. If `global_measurements = String[]`,
then all global measurements are exported.
"""
function export_global_bins_to_csv(
    # ARGUMENTS
    comm::MPI.Comm;
    # KEYWORD ARGUMENTS
    datafolder::String,
    filename_prefix::String = "global_bins",
    pIDs::Union{Vector{Int},Int} = Int[],
    global_measurements::Vector{String} = String[],
    decimals::Int = 6,
    scientific_notation::Bool = false,
    delimiter::String = " "
)

    filename = nothing
    if iszero(MPI.Comm_rank(comm))
        filename = export_global_bins_to_csv(
            datafolder = datafolder,
            filename_prefix = filename_prefix,
            pIDs = pIDs,
            global_measurements = global_measurements,
            decimals = decimals,
            scientific_notation = scientific_notation,
            delimiter = delimiter
        )
    end
    MPI.Barrier(comm)
    
    return filename
end


function export_global_bins_to_csv(;
    # KEYWORD ARGUMENTS
    datafolder::String,
    filename_prefix::String = "global_bins",
    pIDs::Union{Vector{Int},Int} = Int[],
    global_measurements::Vector{String} = String[],
    decimals::Int = 6,
    scientific_notation::Bool = false,
    delimiter::String = " "
)

    # construct path of directory containing HDF5 bin files
    @assert isdir(datafolder)
    bin_folder = joinpath(datafolder, "bins")

    # if single pID is being exported
    if isa(pIDs, Int)
        # construct filename for output CSV file
        filename = joinpath(datafolder, @sprintf("%s_pID-%d.csv", filename_prefix, pIDs))
        pIDs = [pIDs,]
    # if multiple pIDs being exported
    else
        # construct filename for output CSV file
        filename = joinpath(datafolder, @sprintf("%s.csv", filename_prefix))
        # if binned data for all pIDs is being exported
        if isempty(pIDs)
            pIDs = collect( 0 : length(readdir(bin_folder)) - 1 )
        end
    end

    # HDF5 bin file names
    h5_bin_filenames = [joinpath(bin_folder, @sprintf("bins_pID-%d.h5", pID)) for pID in pIDs]

    # open all the input HDF5 bin files
    H5BinFiles = [h5open(file, "r") for file in h5_bin_filenames]

    # get list of global measurements
    global_measurements = isempty(global_measurements) ? keys(H5BinFiles[1]["GLOBAL"]) : global_measurements

    # make sure the sign is included as a global measurement
    if !in("sgn", global_measurements)
        push!(global_measurements, "sgn")
    end

    # initialize function to format numbers to strings
    f = num_to_string_formatter(decimals, scientific_notation)

    # open CSV file
    open(filename, "w") do file

        join(file, ["MEASUREMENT", "PID", "BIN", "VALUE_REAL", "VALUE_IMAG"], delimiter)
        write(file, "\n")

        # iterate over measurements
        for measurement in global_measurements

            # iterate of pID bin files
            for i in eachindex(pIDs)

                # get the pID
                pID = pIDs[i]

                # get the measurement dataset
                Measurement = H5BinFiles[i]["GLOBAL"][measurement]

                # iterate over bins
                for bin in axes(Measurement, 1)

                    # write data to CSV
                    val = Measurement[bin]
                    join(file, [ measurement, pID, bin, f(real(val)), f(imag(val)) ], delimiter)
                    # write end of line
                    write(file, "\n")
                end
            end
        end
    end

    # close all the input HDF5 bin files
    close.(H5BinFiles)

    return filename
end