@doc raw"""
    export_correlation_stats_to_csv(
        # ARGUMENTS
        comm::MPI.Comm;
        # KEYWORD ARGUMENTS
        datafolder::String,
        correlation::String,
        type::String,
        space::String,
        h5filename::HDF5.File = "stats.h5",
        decimals::Int = 6,
        scientific_notation::Bool = false,
        delimiter::String = " "
    )

    export_correlation_stats_to_csv(;
        # KEYWORD ARGUMENTS
        datafolder::String,
        correlation::String,
        type::String,
        space::String,
        h5filename::HDF5.File = "stats.h5",
        decimals::Int = 6,
        scientific_notation::Bool = false,
        delimiter::String = " "
    )

Export statistics for specified type of correlation function from HDF5 file to a CSV file.
"""
function export_correlation_stats_to_csv(
    # ARGUMENTS
    comm::MPI.Comm;
    # KEYWORD ARGUMENTS
    datafolder::String,
    correlation::String,
    type::String,
    space::String,
    h5filename::String = "stats.h5",
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

        csvfilename = _export_correlation_stats_to_csv(
            datafolder, correlation, type, space, H5File, delimiter, formatter
        )

        # close the HDF5 file
        close(H5File)
    end
    MPI.Barrier(comm)

    return csvfilename
end

function export_correlation_stats_to_csv(;
    # KEYWORD ARGUMENTS
    datafolder::String,
    correlation::String,
    type::String,
    space::String,
    h5filename::String = "stats.h5",
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

    csvfile = _export_correlation_stats_to_csv(
        datafolder, correlation, type, space, H5File, delimiter, formatter
    )

    # close the HDF5 file
    close(H5File)

    return csvfile
end

# private function to write correlation data to csv file
function _export_correlation_stats_to_csv(
    datafolder::String,
    correlation::String,
    type::String,
    space::String,
    H5File::HDF5.File,
    delimiter::String,
    formatter::Function
)

    # get pIDs associated with HDF5 stats file
    pIDs = read(H5File, "pIDs")

    # determine whether standard or composite correlation measurement
    category = haskey(CORRELATION_FUNCTIONS, correlation) ? "STANDARD" : "COMPOSITE"

    # construct directory path
    path = joinpath(datafolder, type, correlation)
    mkpath(path)

    # construct filename for csv file if single pID
    if isone(length(pIDs))
        filename = @sprintf("%s_%s_%s_stats_pID-%d.csv", correlation, space, type, pIDs[1])
    else
        filename = @sprintf("%s_%s_%s_stats.csv", correlation, space, type)
    end
    csvfile = joinpath(path, filename)

    # if a composite correlation measurement
    if category == "COMPOSITE"

        # export composite correlation measurement to csv
        _export_composite_correlation_stats_to_csv(
            csvfile, correlation, uppercase(type), uppercase(space), H5File, delimiter, formatter
        )

    # if a standard correlation measurement
    else

        # export standard correlation measurement to csv
        _export_standard_correlation_stats_to_csv(
            csvfile, correlation, uppercase(type), uppercase(space), H5File, delimiter, formatter
        )

    end

    return csvfile
end

# private function to write composite correlation to csv file
function _export_composite_correlation_stats_to_csv(
    filename::String,
    correlation::String,
    type::String,
    space::String,
    H5File::HDF5.File,
    delimiter::String,
    formatter::Function
)

    # extract correlation group
    Correlation = H5File["CORRELATIONS"]["COMPOSITE"][type][correlation][space]
    
    # get mean and std datasets
    Mean = Correlation["MEAN"]
    Std = Correlation["STD"]

    # get dimensions of datasets
    dims = size(Mean)

    # read in the dim labels
    dim_labels = read(Correlation, "DIM_LABELS")

    # number of space-time dimensions
    D = length(dims)

    # vector to represent space-time displacement
    r = zeros(Int, D)

    # open csv file
    open(filename, "w") do file
        
        # write header
        join(
            file,
            ["INDEX", dim_labels[end:-1:1]..., "MEAN_REAL", "MEAN_IMAG", "STD"],
            delimiter
        )
        write(file, "\n")

        # iterate over elements of stats datasets
        for (index, c) in enumerate(CartesianIndices(dims))

            # get the mean value
            avg = Mean[c.I...]
            err = Std[c.I...]

            # get lattice displacement
            @. r = c.I[end:-1:1] - 1

            # write data to file
            join(
                file,
                [index, r..., formatter(real(avg)), formatter(imag(avg)), formatter(err)],
                delimiter
            )
            write(file, "\n")
        end
    end

    return nothing
end

# private function to write standard correlation to csv file
function _export_standard_correlation_stats_to_csv(
    filename::String,
    correlation::String,
    type::String,
    space::String,
    H5File::HDF5.File,
    delimiter::String,
    formatter::Function
)

    # get correlation group
    Correlation_Group = H5File["CORRELATIONS"]["STANDARD"][type][correlation]

    # get id type
    id_type = read(Correlation_Group, "ID_TYPE")

    # get id pairs
    id_pairs = read(Correlation_Group, "ID_PAIRS")

    # extract correlation group
    Correlation = Correlation_Group[space]
    
    # get mean and std datasets
    Mean = Correlation["MEAN"]
    Std = Correlation["STD"]

    # get dimensions of datasets
    dims = size(Mean)

    # read in the dim labels
    dim_labels = read(Correlation, "DIM_LABELS")

    # number of space-time dimensions
    D = length(dims) - 1

    # vector to represent space-time displacement
    r = zeros(Int, D)

    # open csv file
    open(filename, "w") do file

        # get ID headder labels
        id_type_2 = @sprintf("%s_2", id_type)
        id_type_1 = @sprintf("%s_1", id_type)

        # write header
        join(
            file,
            ["INDEX", id_type_2, id_type_1, dim_labels[end-1:-1:1]..., "MEAN_REAL", "MEAN_IMAG", "STD"],
            delimiter
        )
        write(file,"\n")

        # iterate over elements of stats datasets
        for (index, c) in enumerate(CartesianIndices(dims))

            # get the mean value
            avg = Mean[c.I...]
            err = Std[c.I...]

            # get id pair for measurement
            id_1, id_2 = id_pairs[c.I[end]]

            # get lattice displacement
            @. r = c.I[end-1:-1:1] - 1

            # write data to file
            join(
                file,
                [index, id_2, id_1, r..., formatter(real(avg)), formatter(imag(avg)), formatter(err)],
                delimiter
            )
            write(file, "\n")
        end
    end

    return nothing
end