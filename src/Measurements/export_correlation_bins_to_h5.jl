@doc raw"""
    export_correlation_bins_to_h5(
        # ARGUMENTS
        comm::MPI.Comm;
        # KEYWORD ARGUMENTS
        datafolder::String,
        correlation::String,
        type::String,
        space::String,
        pIDs::Union{Vector{Int},Int} = Int[]
    )

    export_correlation_bins_to_h5(;
        # KEYWORD ARGUMENTS
        datafolder::String,
        correlation::String,
        type::String,
        space::String,
        pIDs::Union{Vector{Int},Int} = Int[]
    )

Export the binned data for a specified type of correlation to an HDF5 file living in the directory `/datafolder/type/correlation/space`.
The type of correlation function is specified by `type ∈ ("equal-time", "time-displaced", "integrated")`.
Where the correlation function is in position or momentum space is given by `space ∈ ("momentum", "position")`.
The `pIDs` keyword specifies for which process IDs the binned correlation data is exported.
If `pIDs = Int[]`, then binned local measurements for all process IDs are exported.
"""
function export_correlation_bins_to_h5(
    # ARGUMENTS
    comm::MPI.Comm;
    # KEYWORD ARGUMENTS
    datafolder::String,
    correlation::String,
    type::String,
    space::String,
    pIDs::Union{Vector{Int},Int} = Int[]
)

    filename = nothing
    if iszero(MPI.Comm_rank(comm))
        filename = export_correlation_bins_to_h5(
            datafolder = datafolder,
            correlation = correlation,
            type = type,
            space = space,
            pIDs = pIDs
        )
    end
    MPI.Barrier(comm)

    return filename
end

function export_correlation_bins_to_h5(;
    # KEYWORD ARGUMENTS
    datafolder::String,
    correlation::String,
    type::String,
    space::String,
    pIDs::Union{Vector{Int},Int} = Int[]
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
        filename = joinpath(filepath, @sprintf("%s_%s_%s_bins_pID-%d.h5", correlation, space, type))
        pIDs = [pIDs,]
    # if multiple pIDs being exported
    else
        # construct filename for output HDF5 file
        filename = joinpath(filepath, @sprintf("%s_%s_%s_bins.h5", correlation, space, type))
        # if binned data for all pIDs is being exported
        if isempty(pIDs)
            pIDs = collect( 0 : length(readdir(bin_folder)) - 1 )
        end
    end

    # HDF5 bin file names
    h5_bin_filenames = [joinpath(bin_folder, @sprintf("bins_pID-%d.h5", pID)) for pID in pIDs]

    # open all the input HDF5 bin files
    H5BinFiles = [h5open(file, "r") for file in h5_bin_filenames]

    # number of pIDs
    N_pIDs = length(pIDs)

    # open HDF5 file
    h5open(filename, "w") do file

        # record the pIDs
        attributes(file)["PIDS"] = pIDs

        # record info about measured correlation
        attributes(file)["CORRELATION"] = correlation
        attributes(file)["TYPE"] = Type
        attributes(file)["SPACE"] = Space

        # get correlation in group
        Correlation_Group_In = H5BinFiles[1]["CORRELATIONS"][Category][Type][correlation]
        
        # input correlation dataset
        Correlation_In = Correlation_Group_In[Space]

        # record ID type and pairs
        if haskey(attrs(Correlation_In), "ID_TYPE")
            attributes(file)["ID_TYPE"] = read_attribute(Correlation_Group_In, "ID_TYPE")
            attributes(file)["ID_PAIRS"] = read_attribute(Correlation_Group_In, "ID_PAIRS")
        end

        # initialize dataset to contain binned data
        in_dims = size(Correlation_In)
        N_bins = in_dims[1]
        out_dims = (N_bins, N_pIDs, in_dims[2:end]...)
        Data = create_dataset(file, "DATA", eltype(Correlation_In), out_dims)

        # record dimensional labels of dataset
        bin_dim_labels = read_attribute(Correlation_In, "DIM_LABELS")
        attributes(Data)["DIM_LABELS"] = ["BIN", "PID", bin_dim_labels[2:end]...]        

        # get the assignment slice for dataset
        slice = tuple((1:d for d in in_dims[2:end])...)

        # iterate over process IDs
        for i in eachindex(pIDs)

            # get input correlation in dataset
            Correlation_Group_In = H5BinFiles[1]["CORRELATIONS"][Category][Type][correlation][Space]

            # record the data
            Data[:,i,slice...] = read(Correlation_Group_In)
        end
    end

    # close all the input HDF5 bin files
    close.(H5BinFiles)

    return filename
end