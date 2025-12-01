@doc raw"""
    export_local_bins_to_h5(
        # ARGUMENTS
        comm::MPI.Comm;
        # KEYWORD ARGUMENTS
        datafolder::String,
        filename_prefix::String = "local_bins",
        pIDs::Union{Vector{Int},Int} = Int[],
        local_measurements::Vector{String} = String[],
    )

    export_local_bins_to_h5(;
        # KEYWORD ARGUMENTS
        datafolder::String,
        filename_prefix::String = "local_bins",
        pIDs::Union{Vector{Int},Int} = Int[],
        local_measurements::Vector{String} = String[],
    )

Export the binned local measurements for specified process IDs `pIDs` to a
single HDF5 file. If `pIDs = Int[]`, then binned local measurements for all
process IDs are exported. You can specify a subset of specific local measurements
using the `local_measurements` keyword argument. If `local_measurements = String[]`,
then all local measurements are exported.
"""
function export_local_bins_to_h5(
    # ARGUMENTS
    comm::MPI.Comm;
    # KEYWORD ARGUMENTS
    datafolder::String,
    filename_prefix::String = "local_bins",
    pIDs::Union{Vector{Int},Int} = Int[],
    local_measurements::Vector{String} = String[],
)

    filename = nothing
    if iszero(MPI.Comm_rank(comm))
        filename = export_local_bins_to_h5(
            datafolder = datafolder,
            filename_prefix = filename_prefix,
            pIDs = pIDs,
            local_measurements = local_measurements
        )
    end
    MPI.Barrier(comm)

    return filename
end

function export_local_bins_to_h5(;
    # KEYWORD ARGUMENTS
    datafolder::String,
    filename_prefix::String = "local_bins",
    pIDs::Union{Vector{Int},Int} = Int[],
    local_measurements::Vector{String} = String[],
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

    # get list of local measurements
    local_measurements = isempty(local_measurements) ? keys(H5BinFiles[1]["LOCAL"]) : local_measurements

    # open HDF5 output file
    h5open(filename, "w") do H5LocalFile

        # record the pIDs
        attributes(H5LocalFile)["PIDS"] = pIDs

        # allocate dataset to contain binned sign data
        dims = (size(H5BinFiles[1]["GLOBAL"]["sgn"])..., length(pIDs))
        Sign = create_dataset(H5LocalFile, "SIGN", eltype(H5BinFiles[1]["GLOBAL"]["sgn"]), dims)
        Tdata = eltype(Sign)
        attributes(Sign)["DIM_LABELS"] = ["BIN", "PID"]
        # record the binned sign data
        for i in eachindex(H5BinFiles)
            Sign[:,i] = read(H5BinFiles[i]["GLOBAL"]["sgn"])
        end

        # create group to contain local measurements
        Local = create_group(H5LocalFile, "LOCAL")

        # iterate over local measurements
        for measurement in local_measurements

            # allocate dataset to contain binned local measurement data
            Measurement_In = H5BinFiles[1]["LOCAL"][measurement]
            dims = size(Measurement_In)
            Measurement = create_dataset(Local, measurement, Tdata, (dims[1], length(pIDs), dims[2]))
            attributes(Measurement)["ID_TYPE"] = read_attribute(Measurement_In, "ID_TYPE")
            attributes(Measurement)["DIM_LABELS"] = ["BIN", "PID", "ID"]
            # record the local measurement
            for i in eachindex(H5BinFiles)
                Measurement[:,i,:] = read(H5BinFiles[i]["LOCAL"][measurement])
            end
        end
    end

    # close all the input HDF5 bin files
    close.(H5BinFiles)

    return nothing
end


@doc raw"""
    export_local_bins_to_csv(
        # ARGUMENTS
        comm::MPI.Comm;
        # KEYWORD ARGUMENTS
        datafolder::String,
        filename_prefix::String = "local_bins",
        pIDs::Union{Vector{Int},Int} = Int[],
        local_measurements::Vector{String} = String[],
        decimals::Int = 9,
        scientific_notation::Bool = false,
        delimiter::String = " "
    )

    export_local_bins_to_csv(;
        # KEYWORD ARGUMENTS
        datafolder::String,
        filename_prefix::String = "local_bins",
        pIDs::Union{Vector{Int},Int} = Int[],
        local_measurements::Vector{String} = String[],
        decimals::Int = 9,
        scientific_notation::Bool = false,
        delimiter::String = " "
    )

Export the binned local measurements for specified process IDs `pIDs` to a
single CSV file. If `pIDs = Int[]`, then binned local measurements for all
process IDs are exported. You can specify a subset of specific local measurements
using the `local_measurements` keyword argument. If `local_measurements = String[]`,
then all local measurements are exported.
"""
function export_local_bins_to_csv(
    # ARGUMENTS
    comm::MPI.Comm;
    # KEYWORD ARGUMENTS
    datafolder::String,
    filename_prefix::String = "local_bins",
    pIDs::Union{Vector{Int},Int} = Int[],
    local_measurements::Vector{String} = String[],
    decimals::Int = 6,
    scientific_notation::Bool = false,
    delimiter::String = " "
)

    filename = nothing
    if iszero(MPI.Comm_rank(comm))
        export_local_bins_to_csv(
            datafolder = datafolder,
            filename_prefix = filename_prefix,
            pIDs = pIDs,
            local_measurements = local_measurements,
            decimals = decimals,
            scientific_notation = scientific_notation,
            delimiter = delimiter
        )
    end
    MPI.Barrier(comm)

    return filename
end


function export_local_bins_to_csv(;
    # KEYWORD ARGUMENTS
    datafolder::String,
    filename_prefix::String = "local_bins",
    pIDs::Union{Vector{Int},Int} = Int[],
    local_measurements::Vector{String} = String[],
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

    # get list of local measurements
    local_measurements = isempty(local_measurements) ? keys(H5BinFiles[1]["LOCAL"]) : local_measurements

    # initialize function to format numbers to strings
    f = num_to_string_formatter(decimals, scientific_notation)

    # open CSV output file
    open(filename, "w") do file

        # write header to csv file
        join(file, ["MEASUREMENT", "ID_TYPE", "PID", "ID", "BIN", "VALUE_REAL", "VALUE_IMAGE", "SIGN_REAL", "SIGN_IMAG"], delimiter)
        write(file, "\n")

        # iterate over local measurements
        for measurement in local_measurements

            # get the ID type
            id_type = read_attribute(H5BinFiles[1]["LOCAL"][measurement], "ID_TYPE")

            # iterate of HDF5 bin files
            for i in eachindex(H5BinFiles)

                # get the relevant pID
                pID = pIDs[i]

                # get input binned sign
                Sign_In = H5BinFiles[i]["GLOBAL"]["sgn"]

                # get input measurement dataset
                Measurement_In = H5BinFiles[i]["LOCAL"][measurement]

                # iterate over IDs
                for id in axes(Measurement_In, 2)

                    # iterate over bins
                    for bin in axes(Measurement_In, 1)

                        # get measurement value
                        val = Measurement_In[bin, id]

                        # get the bin averaged sign
                        sgn = Sign_In[bin]

                        # write the data to the CSV file
                        join(
                            file,
                            [measurement, id_type, pID, id, bin, f(real(val)), f(imag(val)), f(real(sgn)), f(imag(sgn))],
                            delimiter
                        )
                        write(file, "\n")
                    end
                end
            end
        end
    end

    # close all the input HDF5 bin files
    close.(H5BinFiles)

    return filename
end