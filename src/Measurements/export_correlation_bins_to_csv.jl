@doc raw"""
    export_correlation_bins_to_csv(
        # ARGUMENTS
        comm::MPI.Comm;
        # KEYWORD ARGUMENTS
        datafolder::String,
        correlation::String,
        type::String,
        space::String,
        pIDs::Union{Vector{Int},Int} = Int[],
        write_index_key::Bool = true,
        decimals::Int = 6,
        scientific_notation::Bool = false,
        delimiter::String = " "
    )

    export_correlation_bins_to_csv(;
        # KEYWORD ARGUMENTS
        datafolder::String,
        correlation::String,
        type::String,
        space::String,
        pIDs::Union{Vector{Int},Int} = Int[],
        write_index_key::Bool = true,
        decimals::Int = 6,
        scientific_notation::Bool = false,
        delimiter::String = " "
    )

Export the binned data for a specified type of correlation to an CSV file living in the directory `/datafolder/type/correlation/space`.
The type of correlation function is specified by `type ∈ ("equal-time", "time-displaced", "integrated")`.
Where the correlation function is in position or momentum space is given by `space ∈ ("momentum", "position")`.
The `pIDs` keyword specifies for which process IDs the binned correlation data is exported.
If `pIDs = Int[]`, then binned local measurements for all process IDs are exported.
If `write_index_key = true`, then another CSV file is written to the `/datafolder/type/correlation/space` directory which provides a key
on how to interpret the `INDEX` column appearing in the CSV file containing the binned data.
"""
function export_correlation_bins_to_csv(
    # ARGUMENTS
    comm::MPI.Comm;
    # KEYWORD ARGUMENTS
    datafolder::String,
    correlation::String,
    type::String,
    space::String,
    pIDs::Union{Vector{Int},Int} = Int[],
    write_index_key::Bool = true,
    decimals::Int = 6,
    scientific_notation::Bool = false,
    delimiter::String = " "
)

    filename = nothing
    if iszero(MPI.Comm_rank(comm))
        export_correlation_bins_to_csv(
            datafolder = datafolder,
            correlation = correlation,
            type = type,
            space = space,
            pIDs = pIDs,
            write_index_key = write_index_key,
            decimals = decimals,
            scientific_notation = scientific_notation,
            delimiter = delimiter
        )
    end
    MPI.Barrier(comm)

    return filename
end

function export_correlation_bins_to_csv(;
    # KEYWORD ARGUMENTS
    datafolder::String,
    correlation::String,
    type::String,
    space::String,
    pIDs::Union{Vector{Int},Int} = Int[],
    write_index_key::Bool = true,
    decimals::Int = 6,
    scientific_notation::Bool = false,
    delimiter::String = " "
)

    correlation, type, space = lowercase(correlation), lowercase(type), lowercase(space)
    Type, Space = uppercase(type), uppercase(space)
    @assert lowercase(type) ∈ ("equal-time", "time-displaced", "integrated")
    @assert lowercase(space) ∈ ("momentum", "position")
    @assert isdir(datafolder)
    bin_folder = joinpath(datafolder, "bins")

    # filepath the HDF5 file will be written to
    filepath = mkpath(joinpath(datafolder, type, correlation, space))

    # correlation category
    Category = haskey(CORRELATION_FUNCTIONS, correlation) ? "STANDARD" : "COMPOSITE"

    # if single pID is being exported
    if isa(pIDs, Int)
        # construct filename for output HDF5 file
        filename = joinpath(filepath, @sprintf("%s_%s_%s_bins_pID-%d.csv", correlation, space, type))
        pIDs = [pIDs,]
    # if multiple pIDs being exported
    else
        # construct filename for output HDF5 file
        filename = joinpath(filepath, @sprintf("%s_%s_%s_bins.csv", correlation, space, type))
        # if binned data for all pIDs is being exported
        if isempty(pIDs)
            pIDs = collect( 0 : length(readdir(bin_folder)) - 1 )
        end
    end

    # HDF5 bin file names
    h5_bin_filenames = [joinpath(bin_folder, @sprintf("bins_pID-%d.h5", pID)) for pID in pIDs]

    # open all the input HDF5 bin files
    H5BinFiles = [h5open(file, "r") for file in h5_bin_filenames]

    # if index key file is getting written
    if write_index_key

        # filename of the index key is written to
        key_filename = joinpath(filepath, @sprintf("%s_%s_%s_index_key.csv", correlation, space, type))

        # if standard correlation
        if Category == "STANDARD"

            # write index key
            write_standard_index_key(
                key_filename, H5BinFiles[1],
                correlation, type, space,
                delimiter
            )

        # if composite correlation
        else

            # write index key
            write_composite_index_key(
                key_filename, H5BinFiles[1],
                correlation, type, space,
                delimiter
            )
        end
    end

    # number of pIDs
    N_pIDs = length(pIDs)

    # initialize function to format numbers to strings
    f = num_to_string_formatter(decimals, scientific_notation)

    # open csv file
    open(filename, "w") do file

        # write header
        join(
            file,
            [
                "PID", "INDEX", "BIN",
                @sprintf("%s_REAL", uppercase(correlation)), @sprintf("%s_IMAG", uppercase(correlation)),
                "SIGN_REAL", "SIGN_IMAG"
            ],
            delimiter
        )
        write(file, "\n")

        # iterate over process IDs
        for i in eachindex(H5BinFiles)

            # get the pID
            pID = pIDs[i]

            # get the HDF5 bin file
            H5BinFile = H5BinFiles[i]

            # get the relevant dataset
            Correlation = H5BinFile["CORRELATIONS"][Category][Type][correlation][Space]

            # get the binned sign data
            Sign = H5BinFile["GLOBAL"]["sgn"]

            # get the dimensions to iterate over
            dims = size(Correlation)

            # iterate over indices
            for (index, c) in enumerate(CartesianIndices(dims[2:end]))

                # iterate over bins
                for bin in 1:dims[1]

                    # get the correlation value
                    val = Correlation[bin,c.I...]

                    # get the sgn
                    sgn = Sign[bin]

                    # write data to file
                    join(
                        file,
                        [pID, index, bin, f(real(val)), f(imag(val)), f(real(sgn)), f(imag(sgn))],
                        delimiter
                    )
                    write(file, "\n")
                end
            end
        end
    end

    # close all the input HDF5 bin files
    close.(H5BinFiles)

    return filename
end

# write index key for standard correlation
function write_standard_index_key(
    filename::String,
    H5BinFile::HDF5.File,
    correlation::String,
    type::String,
    space::String,
    delimiter::String
)

    # correlation group
    Correlation_Group = H5BinFile["CORRELATIONS"]["STANDARD"][uppercase(type)][correlation]

    # get correlation dataset
    Correlation = Correlation_Group[uppercase(space)]

    # get dimension labels
    dim_labels = read_attribute(Correlation, "DIM_LABELS")
    
    # get dimension of output dataset
    dims = size(Correlation)[2:end]

    # get ID pairs
    id_pairs = read_attribute(Correlation_Group, "ID_PAIRS")

    # get ID type
    id_type = read_attribute(Correlation_Group, "ID_TYPE")

    # open csv file
    open(filename, "w") do file

        # write header
        join(
            file,
            ["INDEX", @sprintf("%s_%d", id_type, 2), @sprintf("%s_%d", id_type, 1), dim_labels[end-1:-1:2]...],
            delimiter
        )
        write(file, "\n")

        # iterate over indices
        for (i, c) in enumerate(CartesianIndices(dims))

            # get the ID pair
            id_1, id_2 = id_pairs[c.I[end]]

            # index line
            join(
                file,
                [i, id_2, id_1, c.I[end-1:-1:1]...],
                delimiter
            )
            write(file, "\n")
        end
    end

    return nothing
end

# write index key for standard correlation
function write_composite_index_key(
    filename::String,
    H5BinFile::HDF5.File,
    correlation::String,
    type::String,
    space::String,
    delimiter::String
)

    # correlation group
    Correlation_Group = H5BinFile["CORRELATIONS"]["COMPOSITE"][uppercase(type)][correlation]

    # get correlation dataset
    Correlation = Correlation_Group[uppercase(space)]

    # get dimension labels
    dim_labels = read_attribute(Correlation, "DIM_LABELS")
    
    # get dimension of output dataset
    dims = size(Correlation)[2:end]

    # open csv file
    open(filename, "w") do file

        # write header
        join(
            file,
            ["INDEX", dim_labels[end:-1:2]...],
            delimiter
        )
        write(file, "\n")

        # iterate over indices
        for (i, c) in enumerate(CartesianIndices(dims))

            # index line
            join(
                file,
                [i, c.I[end:-1:2]...],
                delimiter
            )
            write(file, "\n")
        end
    end

    return nothing
end
