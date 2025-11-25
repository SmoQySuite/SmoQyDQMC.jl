@doc raw"""
    export_global_stats_to_csv(
        # ARGUMENTS
        comm::MPI.Comm;
        # KEYWORD ARGUMENTS
        datafolder::String,
        h5filename::String = "stats.h5",,
        csvfilename_prefix::String = "global",
        measurements::Vector{String} = String[],
        decimals::Int = 6,
        scientific_notation::Bool = false,
        delimiter::String = " "
    )

    export_global_stats_to_csv(;
        # KEYWORD ARGUMENTS
        datafolder::String,
        h5filename::String = "stats.h5",,
        csvfilename_prefix::String = "global",
        measurements::Vector{String} = String[],
        decimals::Int = 6,
        scientific_notation::Bool = false,
        delimiter::String = " "
    )

This function writes the global measurement statistics stored in the `h5filename` HDF5 file
found in the directory `datafolder` to CSV file, returning the name of the CSV file that was written.
The `measurements` keyword argument specifies the measurements to be exported.
If `measurements = String[]`, then all measurements are exported.
"""
function export_global_stats_to_csv(
    # ARGUMENTS
    comm::MPI.Comm;
    # KEYWORD ARGUMENTS
    datafolder::String,
    h5filename::String = "stats.h5",
    csvfilename_prefix::String = "global",
    measurements::Vector{String} = String[],
    decimals::Int = 6,
    scientific_notation::Bool = false,
    delimiter::String = " "
)

    # if root process
    csvfilename = nothing
    if iszero(MPI.Comm_rank(comm))

        # make sure datafolder exists
        @assert isdir(datafolder)

        # initialize function to convert floats to strings
        formatter = num_to_string_formatter(decimals, scientific_notation)

        # open HDF5 stats file
        h5filenamename = joinpath(datafolder, h5filename)
        H5File = h5open(h5filenamename, "r")

        # export global stats
        csvfilename = _export_global_stats_to_csv(
            datafolder, csvfilename_prefix, H5File, measurements, delimiter, formatter
        )

        # close the HDF5 file
        close(H5File)
    end
    MPI.Barrier(comm)

    return csvfilename
end

function export_global_stats_to_csv(;
    # KEYWORD ARGUMENTS
    datafolder::String,
    h5filename::String = "stats.h5",
    csvfilename_prefix::String = "global",
    measurements::Vector{String} = String[],
    decimals::Int = 6,
    scientific_notation::Bool = false,
    delimiter::String = " "
)

    # make sure datafolder exists
    @assert isdir(datafolder)

    # initialize function to convert floats to strings
    formatter = num_to_string_formatter(decimals, scientific_notation)

    # open HDF5 stats file
    h5filenamename = joinpath(datafolder, h5filename)
    H5File = h5open(h5filenamename, "r")

    csvfilename = _export_global_stats_to_csv(
        datafolder, csvfilename_prefix, H5File, measurements, delimiter, formatter
    )

    # close the HDF5 file
    close(H5File)

    return csvfilename
end

# private function
function _export_global_stats_to_csv(
    datafolder::String,
    csvfilename_prefix::String,
    H5File::HDF5.File,
    measurements::Vector{String},
    delimiter::String,
    formatter::Function
)

    # get process IDs from HDF5 stats file
    pIDs = read_attribute(H5File, "PIDS")

    # construct csv filename
    filename = (
        isone(length(pIDs))
        ? @sprintf("%s_stats_pID-%d.csv", csvfilename_prefix, pIDs[1])
        : @sprintf("%s_stats.csv", csvfilename_prefix)
    )
    csvfilename = joinpath(datafolder, filename)

    # open CSV stats file
    CSVFile = open(csvfilename, "w")

    # Get the global measurements group
    Global = H5File["GLOBAL"]

    # Get global measurements
    measurements = isempty(measurements) ? keys(Global) : measurements

    # Write Header to CSV
    join(CSVFile, ["MEASUREMENT", "MEAN_REAL", "MEAN_IMAG", "STD"], delimiter)
    write(CSVFile, "\n")

    # iterate over measurements
    for measurement in measurements
        # write measurement to csv
        avg = read(Global[measurement]["MEAN"])
        err = read(Global[measurement]["STD"])
        join(CSVFile, (measurement, formatter(real(avg)), formatter(imag(avg)), err), delimiter)
        write(CSVFile, "\n")
    end

    # close CSV stats file
    close(CSVFile)

    return csvfilename
end