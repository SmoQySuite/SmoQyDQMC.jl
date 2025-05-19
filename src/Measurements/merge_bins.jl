@doc raw"""
    merge_bins(
        # ARGUMENTS
        simulation_info::SimulationInfo
    )

    merge_bins(;
        # KEYWORD ARGUMENTS
        datafolder::String,
        pID::Int
    )
"""
function merge_bins(
    # ARGUMENTS
    simulation_info::SimulationInfo
)

    (; datafolder, pID) = simulation_info
    merge_bins(
        datafolder = datafolder,
        pID = pID
    )

    return nothing
end

function merge_bins(;
    # KEYWORD ARGUMENTS
    datafolder::String,
    pID::Int
)

    # get the directory containing the HDF5 bin files
    binfolder = joinpath(datafolder, "bins")

    # get the directory containing the HDF5 bin files for current pID
    pID_binfolder = joinpath(binfolder, "pID-$pID")

    # check HDF5 files still need to be merged
    if isdir(pID_binfolder)

        # construct filename for the merged HDF5 file
        filename = joinpath(binfolder, "bins_pID-$(pID).h5")
        
        # count the number of bins
        N_bins = length(readdir(pID_binfolder))

        # open new HDF5 file to contain all the binned data
        h5open(filename, "w") do fout
            # Record the process ID
            fout["pID"] = pID
            # Record the number of bins
            fout["N_BINS"] = N_bins
            # open first bin file
            h5open(joinpath(pID_binfolder, "bin-1.h5"), "r") do fin
                # initialize/allocate HDF5 file to contain all binned data
                init_hdf5_bins_file(fout, fin, N_bins)
                # copy contents of first bin file over
                copyto_hdf5_bin(fout, fin, 1)
                # record inverse temperature and system size
                fout["BETA"] = read(fin["BETA"])
                fout["N_ORBITALS"] = read(fin["N_ORBITALS"])
            end
            # iterate over remaining bin files
            for bin in 2:N_bins
                # construct bin files
                binfile = @sprintf "bin-%d.h5" bin
                # open HDF5 bin file
                h5open(joinpath(pID_binfolder, binfile), "r") do fin
                    # copy contents of first bin file over
                    copyto_hdf5_bin(fout, fin, bin)
                end
            end
        end

        # delete pID bin folder
        rm(pID_binfolder, recursive = true)
    end

    return nothing
end

function init_hdf5_bins_file(
    fout::HDF5.File,
    fin::HDF5.File,
    N_bins::Int
)

    # initialize global measurements
    Global = create_group(fout, "GLOBAL")
    Global_Bin = fin["GLOBAL"]
    for key in keys(Global_Bin)
        Global_Measurement = create_dataset(Global, key, eltype(Global_Bin[key]), N_bins)
        attributes(Global_Measurement)["DIM_LABELS"] = ["BIN"]
    end

    # initialize local measurements
    Local = create_group(fout, "LOCAL")
    Local_Bin = fin["LOCAL"]
    for key in keys(Local_Bin)
        Local_Measurement = create_dataset(Local, key, eltype(Local_Bin[key]), (N_bins, size(Local_Bin[key])...))
        attributes(Local_Measurement)["DIM_LABELS"] = ["BIN", "ID"]
        attributes(Local_Measurement)["ID_TYPE"] = LOCAL_MEASUREMENTS[key]
    end

    # initialize group structure of correlations
    Correlations = create_group(fout, "CORRELATIONS")
    Standard = create_group(Correlations, "STANDARD")
    StandardEqualTime = create_group(Standard, "EQUAL-TIME")
    StandardTimeDisplaced = create_group(Standard, "TIME-DISPLACED")
    StandardIntegrated = create_group(Standard, "INTEGRATED")
    Composite = create_group(Correlations, "COMPOSITE")
    CompositeEqualTime = create_group(Composite, "EQUAL-TIME")
    CompositeTimeDisplaced = create_group(Composite, "TIME-DISPLACED")
    CompositeIntegrated = create_group(Composite, "INTEGRATED")

    # initialize standard equal-time correlation measurements
    StandardEqualTime_Bin = fin["CORRELATIONS/STANDARD/EQUAL-TIME"]
    for key in keys(StandardEqualTime_Bin)
        Correlation_Bin = StandardEqualTime_Bin[key]
        Correlation = create_group(StandardEqualTime, key)
        Correlation["ID_PAIRS"] = read(Correlation_Bin["ID_PAIRS"])
        Correlation["ID_TYPE"] = read(Correlation_Bin["ID_TYPE"])
        Position_Bin = Correlation_Bin["POSITION"]
        Position = create_dataset(Correlation, "POSITION", eltype(Position_Bin), (N_bins, size(Position_Bin)...))
        Momentum_Bin = Correlation_Bin["MOMENTUM"]
        Momentum = create_dataset(Correlation, "MOMENTUM", eltype(Momentum_Bin), (N_bins, size(Momentum_Bin)...))
        D = ndims(Position_Bin) - 1 # number of spatial dimensions
        attributes(Position)["DIM_LABELS"] = ["BIN", position_column_labels(D)..., "ID_PAIR"]
        attributes(Momentum)["DIM_LABELS"] = ["BIN", momentum_column_labels(D)..., "ID_PAIR"]
    end

    # initialize standard time-displaced correlation measurements
    StandardTimeDisplaced_Bin = fin["CORRELATIONS/STANDARD/TIME-DISPLACED"]
    for key in keys(StandardTimeDisplaced_Bin)
        Correlation_Bin = StandardTimeDisplaced_Bin[key]
        Correlation = create_group(StandardTimeDisplaced, key)
        Correlation["ID_PAIRS"] = read(Correlation_Bin["ID_PAIRS"])
        Correlation["ID_TYPE"] = read(Correlation_Bin["ID_TYPE"])
        Position_Bin = Correlation_Bin["POSITION"]
        Position = create_dataset(Correlation, "POSITION", eltype(Position_Bin), (N_bins, size(Position_Bin)...))
        Momentum_Bin = Correlation_Bin["MOMENTUM"]
        Momentum = create_dataset(Correlation, "MOMENTUM", eltype(Momentum_Bin), (N_bins, size(Momentum_Bin)...))
        D = ndims(Position_Bin) - 2 # number of spatial dimensions
        attributes(Position)["DIM_LABELS"] = ["BIN", position_column_labels(D)..., "TAU", "ID_PAIR"]
        attributes(Momentum)["DIM_LABELS"] = ["BIN", momentum_column_labels(D)..., "TAU", "ID_PAIR"]
    end

    # initialize standard integrated correlation measurements
    StandardIntegrated_Bin = fin["CORRELATIONS/STANDARD/EQUAL-TIME"]
    for key in keys(StandardIntegrated_Bin)
        Correlation_Bin = StandardIntegrated_Bin[key]
        Correlation = create_group(StandardIntegrated, key)
        Correlation["ID_PAIRS"] = read(Correlation_Bin["ID_PAIRS"])
        Correlation["ID_TYPE"] = read(Correlation_Bin["ID_TYPE"])
        Position_Bin = Correlation_Bin["POSITION"]
        Position = create_dataset(Correlation, "POSITION", eltype(Position_Bin), (N_bins, size(Position_Bin)...))
        Momentum_Bin = Correlation_Bin["MOMENTUM"]
        Momentum = create_dataset(Correlation, "MOMENTUM", eltype(Momentum_Bin), (N_bins, size(Momentum_Bin)...))
        D = ndims(Position_Bin) - 1 # number of spatial dimensions
        attributes(Position)["DIM_LABELS"] = ["BIN", position_column_labels(D)..., "ID_PAIR"]
        attributes(Momentum)["DIM_LABELS"] = ["BIN", momentum_column_labels(D)..., "ID_PAIR"]
    end

    # initialize composite equal-time correlation measurements
    CompositeEqualTime_Bin = fin["CORRELATIONS/COMPOSITE/EQUAL-TIME"]
    for key in keys(CompositeEqualTime_Bin)
        Correlation_Bin = CompositeEqualTime_Bin[key]
        Correlation = create_group(CompositeEqualTime, key)
        Position_Bin = Correlation_Bin["POSITION"]
        Position = create_dataset(Correlation, "POSITION", eltype(Position_Bin), (N_bins, size(Position_Bin)...))
        Momentum_Bin = Correlation_Bin["MOMENTUM"]
        Momentum = create_dataset(Correlation, "MOMENTUM", eltype(Momentum_Bin), (N_bins, size(Momentum_Bin)...))
        D = ndims(Position_Bin) # number of spatial dimensions
        attributes(Position)["DIM_LABELS"] = ["BIN", position_column_labels(D)...]
        attributes(Momentum)["DIM_LABELS"] = ["BIN", momentum_column_labels(D)...]
    end

    # initialize composite time-displaced correlation measurements
    CompositeTimeDisplaced_Bin = fin["CORRELATIONS/COMPOSITE/TIME-DISPLACED"]
    for key in keys(CompositeTimeDisplaced_Bin)
        Correlation_Bin = CompositeTimeDisplaced_Bin[key]
        Correlation = create_group(CompositeTimeDisplaced, key)
        Position_Bin = Correlation_Bin["POSITION"]
        Position = create_dataset(Correlation, "POSITION", eltype(Position_Bin), (N_bins, size(Position_Bin)...))
        Momentum_Bin = Correlation_Bin["MOMENTUM"]
        Momentum = create_dataset(Correlation, "MOMENTUM", eltype(Momentum_Bin), (N_bins, size(Momentum_Bin)...))
        D = ndims(Position_Bin) - 1 # number of spatial dimensions
        attributes(Position)["DIM_LABELS"] = ["BIN", position_column_labels(D)..., "TAU"]
        attributes(Momentum)["DIM_LABELS"] = ["BIN", momentum_column_labels(D)..., "TAU"]
    end

    # initialize composite integrated correlation measurements
    CompositeIntegrated_Bin = fin["CORRELATIONS/COMPOSITE/INTEGRATED"]
    for key in keys(CompositeIntegrated_Bin)
        Correlation_Bin = CompositeIntegrated_Bin[key]
        Correlation = create_group(CompositeIntegrated, key)
        Position_Bin = Correlation_Bin["POSITION"]
        Position = create_dataset(Correlation, "POSITION", eltype(Position_Bin), (N_bins, size(Position_Bin)...))
        Momentum_Bin = Correlation_Bin["MOMENTUM"]
        Momentum = create_dataset(Correlation, "MOMENTUM", eltype(Momentum_Bin), (N_bins, size(Momentum_Bin)...))
        D = ndims(Position_Bin) # number of spatial dimensions
        attributes(Position)["DIM_LABELS"] = ["BIN", position_column_labels(D)...]
        attributes(Momentum)["DIM_LABELS"] = ["BIN", momentum_column_labels(D)...]
    end

    return nothing
end

# copy contents of single bin HDF5 file to HDF5 file containing all binned data
function copyto_hdf5_bin(
    fout::HDF5.File,
    fin::HDF5.File,
    bin::Int
)

    # copy global measurements over
    Global_out = fout["GLOBAL"]
    Global_in = fin["GLOBAL"]
    for key in keys(Global_out)
        Global_out[key][bin] = read(Global_in[key])
    end

    # copy local measurements over
    Local_out = fout["LOCAL"]
    Local_in = fin["LOCAL"]
    for key in keys(Local_out)
        Local_out[key][bin,:] = read(Local_in[key])
    end

    # copy standard equal-time correlation measurements
    copy_correlation_bins(
        fout["CORRELATIONS/STANDARD/EQUAL-TIME"],
        fin["CORRELATIONS/STANDARD/EQUAL-TIME"],
        bin
    )

    # copy standard time-displaced correlation measurements
    copy_correlation_bins(
        fout["CORRELATIONS/STANDARD/TIME-DISPLACED"],
        fin["CORRELATIONS/STANDARD/TIME-DISPLACED"],
        bin
    )

    # copy standard integrated correlation measurements
    copy_correlation_bins(
        fout["CORRELATIONS/STANDARD/INTEGRATED"],
        fin["CORRELATIONS/STANDARD/INTEGRATED"],
        bin
    )

    # copy composite equal-time correlation measurements
    copy_correlation_bins(
        fout["CORRELATIONS/COMPOSITE/EQUAL-TIME"],
        fin["CORRELATIONS/COMPOSITE/EQUAL-TIME"],
        bin
    )

    # copy composite time-displaced correlation measurements
    copy_correlation_bins(
        fout["CORRELATIONS/COMPOSITE/TIME-DISPLACED"],
        fin["CORRELATIONS/COMPOSITE/TIME-DISPLACED"],
        bin
    )

    # copy composite integrated correlation measurements
    copy_correlation_bins(
        fout["CORRELATIONS/COMPOSITE/INTEGRATED"],
        fin["CORRELATIONS/COMPOSITE/INTEGRATED"],
        bin
    )

    return nothing
end

# copy binned correlation table over
function copy_correlation_bins(
    Correlations_out::HDF5.Group,
    Correlations_in::HDF5.Group,
    bin::Int
)

    # iterate over correlation measurements
    for key in keys(Correlations_out)

        # copy the specified bin over
        Cout = Correlations_out[key]
        Cin = Correlations_in[key]
        Position_out = Cout["POSITION"]
        Position_in = Cin["POSITION"]
        Momentum_out = Cout["MOMENTUM"]
        Momentum_in = Cin["MOMENTUM"]
        slice = tuple((1:d for d in size(Position_in))...)
        Position_out[bin, slice...] = read(Position_in)
        Momentum_out[bin, slice...] = read(Momentum_in)
    end

    return nothing
end

# axis/column labels for displacement vectors in position and momentum space
position_column_labels(D::Int) = ((@sprintf("R_%d", d) for d in 1:D)...,)
momentum_column_labels(D::Int) = ((@sprintf("K_%d", d) for d in 1:D)...,)


# private merge bins method when MPI is being used
function _merge_bins(
    comm::MPI.Comm,
    datafolder::String,
    pIDs::Vector{Int}
)

    pIDs = isempty(pIDs) ? collect(0:MPI.Comm_size(comm)-1) : pIDs
    pID = pIDs[MPI.Comm_rank(comm) + 1]
    if isdir(joinpath(folder,"bins","pID-$(pID)"))
        merge_bins(datafolder = datafolder, pID = pID)
    end

    return pIDs
end


# private merge bins method when MPI is NOT being used
function _merge_bins(
    datafolder::String,
    pIDs::Union{Vector{Int},Int}
)

    pIDs = isa(pIDs, Int) ? [pIDs,] : pIDs
    pIDs = isempty(pIDs) ? collect(parse(Int,split(f,"-")[end]) for f in filter(isdir,readdir(joinpath(datafolder,"bins")))) : pIDs
    for pID in pIDs
        merge_bins(datafolder = datafolder, pID = pID)
    end

    return pIDs
end


@doc raw"""
    rm_bins(
        comm::MPI.Comm,
        datafolder::String
    )

    rm_bins(
        datafolder::String
    )

Delete the binned data stored in the `datafolder` directory.
This function essentially deletes the directory `datafolder/bins` and its contents.
"""
function rm_bins(
    comm::MPI.Comm,
    datafolder::String
)
    
    pID = MPI.Comm_rank(comm)
    MPI.Barrier(comm)
    rm(joinpath(datafolder, "bins", "pID-$(pID)"), recursive = true, force = true)
    rm(joinpath(datafolder, "bins", "bins_pID-$(pID).h5"), recursive = true, force = true)
    MPI.Barrier(comm)
    if iszero(pID)
        rm(joinpath(datafolder, "bins"), recursive = true, force = true)
    end
    MPI.Barrier(comm)

    return nothing
end

function rm_bins(
    datafolder::String
)
    rm(joinpath(datafolder,"bins"), recursive = true, force = true)
    return nothing
end