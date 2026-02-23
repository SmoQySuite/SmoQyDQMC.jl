# function to process the statistics for a single pID
function _process_measurements(
    folder::String,
    filename::String,
    pIDs::Vector{Int},
    n_bins::Union{Nothing,Int},
    rm_binned_data::Bool,
    process_global_measurements::Bool,
    process_local_measurements::Bool,
    process_all_equal_time_measurements::Bool,
    process_all_time_displaced_measurements::Bool,
    process_all_integrated_measurements::Bool,
    standard_equal_time::Vector{String},
    standard_time_displaced::Vector{String},
    standard_integrated::Vector{String},
    composite_equal_time::Vector{String},
    composite_time_displaced::Vector{String},
    composite_integrated::Vector{String}
)

    # open output HDF5 stats file
    h5_stats_filename = joinpath(folder, filename)
    H5StatsFile = h5open(h5_stats_filename, "w")

    # get directory containing bin folder
    bin_folder = joinpath(folder, "bins")

    # HDF5 bin file names
    h5_bin_filenames = [joinpath(bin_folder, "bins_pID-$(pID).h5") for pID in pIDs]

    # open all the input HDF5 bin files
    H5BinFiles = [h5open(file, "r") for file in h5_bin_filenames]

    # number of HDF5 files containing binned data
    N_pIDs = length(pIDs)

    # get the number of bins per pID
    n_data_bins = read_attribute(H5BinFiles[1], "N_BINS")
    n_bins = isnothing(n_bins) ? n_data_bins : n_bins
    @assert (n_data_bins % n_bins) == 0

    # calculate total number bins across all pIDs
    N_bins = n_bins * N_pIDs

    # get all equal-time correlations if necessary
    if process_all_equal_time_measurements
        standard_equal_time  = keys(H5BinFiles[1]["CORRELATIONS/STANDARD/EQUAL-TIME"])
        composite_equal_time = keys(H5BinFiles[1]["CORRELATIONS/COMPOSITE/EQUAL-TIME"])
    end

    # get all time-displaced correlations if necessary
    if process_all_time_displaced_measurements
        standard_time_displaced = keys(H5BinFiles[1]["CORRELATIONS/STANDARD/TIME-DISPLACED"])
        composite_time_displaced = keys(H5BinFiles[1]["CORRELATIONS/COMPOSITE/TIME-DISPLACED"])
    end

    # get all integrated correlations if necessary
    if process_all_integrated_measurements
        standard_integrated  = keys(H5BinFiles[1]["CORRELATIONS/STANDARD/INTEGRATED"])
        composite_integrated = keys(H5BinFiles[1]["CORRELATIONS/COMPOSITE/INTEGRATED"])
    end

    # initialize the HDF5 stats file
    allocate_stats_file!(
        H5StatsFile, H5BinFiles[1],
        process_global_measurements, process_local_measurements,
        standard_equal_time, standard_time_displaced, standard_integrated,
        composite_equal_time, composite_time_displaced, composite_integrated
    )

    # record pIDs associated with stats HDF5 files
    attributes(H5StatsFile)["PIDS"] = pIDs

    # get system size and inverse temperature
    N_orbitals = read_attribute(H5BinFiles[1], "N_ORBITALS")
    β = read_attribute(H5BinFiles[1], "BETA")
    Δτ = read_attribute(H5BinFiles[1], "DELTA_TAU")
    Lτ = read_attribute(H5BinFiles[1], "L_TAU")

    # record metadata about stats to computes
    attributes(H5StatsFile)["BETA"] = β
    attributes(H5StatsFile)["DELTA_TAU"] = Δτ
    attributes(H5StatsFile)["L_TAU"] = Lτ
    attributes(H5StatsFile)["N_ORBITALS"] = N_orbitals
    attributes(H5StatsFile)["N_BINS"] = N_bins

    # calculate the binned average sign for each HDF5 containing binned data
    binned_sign = vcat((rebin(read(H5BinFile["GLOBAL/sgn"]), n_bins) for H5BinFile in H5BinFiles)...)

    # preallocate arrays for jackknife
    jackknife_sample_means = (similar(binned_sign), similar(binned_sign))
    jackknife_g = similar(binned_sign)

    # type of reported mean and standard deviations
    T_mean = eltype(binned_sign)
    T_std = real(T_mean)

    # get global measurement stats group
    Global_Out = H5StatsFile["GLOBAL"]

    # iterate over global measurements
    for key in keys(Global_Out)
        # read in global measurement
        binned_vals = vcat((rebin(read(H5BinFiles[n]["GLOBAL"][key]),n_bins) for n in 1:N_pIDs)...)
        binned_vals = rebin(binned_vals , N_bins)
        # if a global measurement does not require reweighting
        if startswith(key,"sgn") || startswith(key,"log") || startswith(key,"action") || startswith(key,"chemical_potential")
            average = mean(binned_vals)
            stdev = stdm(binned_vals, average) / sqrt(N_bins)
        # if a global measurement requires reweighting
        else
            average, stdev = jackknife(
                /, binned_vals, binned_sign;
                jackknife_sample_means, jackknife_g
            )
        end
        # create group for global measurement
        Global_Measurement = Global_Out[key]
        # record final mean and standard deviation
        Global_Measurement["MEAN"] = average
        Global_Measurement["STD"] = stdev
    end


    # calculate the compressibility.
    # calculate the compressibility using fluctuation-dissipation theorem
    # such that κ = (β/N)⋅(⟨N²⟩-⟨N⟩²), using the jackknife method
    # with re-weighting to account for the sign problem
    n = vcat((rebin(read(H5BinFiles[i]["GLOBAL/density"]),n_bins) for i in 1:N_pIDs)...)
    N² = vcat((rebin(read(H5BinFiles[i]["GLOBAL/Nsqrd"]), n_bins) for i in 1:N_pIDs)...)
    S = binned_sign
    κ, Δκ = jackknife(
        (n̄, N̄², S̄) -> (β/N_orbitals)*(N̄²/S̄ - (N_orbitals*n̄/S̄)^2),
         n, N², S
    )
    Compressibility = create_group(Global_Out, "compressibility")
    Compressibility["MEAN"] = κ
    Compressibility["STD"] = Δκ

    # iterate over local measurements
    Local_Out = H5StatsFile["LOCAL"]
    for key in keys(Local_Out)
        # get local measurement group
        Measurement_Out = Local_Out[key]
        # get the local measurement values for all IDs
        data = vcat((rebin(read(H5BinFiles[n]["LOCAL"][key]),n_bins) for n in 1:N_pIDs)...)
        # Get the number of IDs associated with local measurement
        N_IDs = size(data, 2)
        # allocate vectors to contain local measurement stats
        average = zeros(T_mean, N_IDs)
        stdev = zeros(T_std, N_IDs)
        # iterate IDs associated with local measurement
        for ID in 1:N_IDs
            # calculate mean and error for local measurement using jackknife reweighting
            # for current local measurement ID
            average[ID], stdev[ID] = jackknife(
                /, view(data, :, ID), binned_sign;
                jackknife_sample_means, jackknife_g
            )
        end
        # record the local measurement stats
        Measurement_Out["MEAN"] = average
        Measurement_Out["STD"] = stdev
    end

    # process standard equal-time correlation measurements
    process_correlations!(
        H5StatsFile, H5BinFiles, "STANDARD/EQUAL-TIME",
        binned_sign, jackknife_sample_means, jackknife_g
    )

    # process standard time-displaced correlation measurements
    process_correlations!(
        H5StatsFile, H5BinFiles, "STANDARD/TIME-DISPLACED",
        binned_sign, jackknife_sample_means, jackknife_g
    )

    # process standard integrated correlation measurements
    process_correlations!(
        H5StatsFile, H5BinFiles, "STANDARD/INTEGRATED",
        binned_sign, jackknife_sample_means, jackknife_g
    )

    # process composite equal-time correlation measurements
    process_correlations!(
        H5StatsFile, H5BinFiles, "COMPOSITE/EQUAL-TIME",
        binned_sign, jackknife_sample_means, jackknife_g
    )

    # process composite time-displaced correlation measurements
    process_correlations!(
        H5StatsFile, H5BinFiles, "COMPOSITE/TIME-DISPLACED",
        binned_sign, jackknife_sample_means, jackknife_g
    )

    # process composite integrated correlation measurements
    process_correlations!(
        H5StatsFile, H5BinFiles, "COMPOSITE/INTEGRATED",
        binned_sign, jackknife_sample_means, jackknife_g
    )

    # close all the input HDF5 bin files
    close.(H5BinFiles)

    # close the output HDF5 stats file
    close(H5StatsFile)

    # delete HDF5 files containing binned data
    if rm_binned_data
        rm_bins(folder)
    end

    return h5_stats_filename
end


# process a certain type of correlation
function process_correlations!(
    H5StatsFile::HDF5.File,
    H5BinFiles::Vector{HDF5.File},
    correlation_type::String,
    binned_sign::Vector{Complex{T}},
    jackknife_sample_means::Tuple{Vector{Complex{T}},Vector{Complex{T}}},
    jackknife_g::Vector{Complex{T}}
) where {T<:AbstractFloat}

    # get output correlations
    Correlations_Out = H5StatsFile["CORRELATIONS"][correlation_type]

    # get the number of pIDs
    N_pIDs = length(H5BinFiles)

    # total number of bins
    N_bins = length(binned_sign)

    # number of bins per pID
    n_bins = N_bins ÷ N_pIDs

    for key in keys(Correlations_Out)

        # get the output group for the position space correlation
        Position_Out = Correlations_Out[key]["POSITION"]

        # get the input binned datasets for the position correlations
        Positions_In = tuple((H5BinFiles[n]["CORRELATIONS"][correlation_type][key]["POSITION"] for n in 1:N_pIDs)...)

        # get the dimensionality of the correlation data.
        # first dimension is cut off as that one corresponds to the bins.
        dims = size(first(Positions_In))[2:end]

        # allocate arrays to contain stats
        average = zeros(Complex{T}, dims)
        stdev = zeros(T, dims)

        # iterate over all correlation displacement vectors
        for c in CartesianIndices(dims)
            # concatenate rebinned data for each pID together
            data = vcat((
                # rebin the data associated with single pID
                rebin(
                    # read in the binned data associated with each pID
                    Positions_In[n][:,c.I...],
                    n_bins
                )
                # iterate over pIDs
                for n in 1:N_pIDs
            )...)
            # calculate the stats doing jackknife reweighting
            average[c], stdev[c] = jackknife(
                /, data, binned_sign;
                jackknife_sample_means, jackknife_g
            )
        end

        # record the final stats
        Position_Out["MEAN"] = average
        Position_Out["STD"] = stdev

        # get the output group for the momentum space correlation
        Momentum_Out = Correlations_Out[key]["MOMENTUM"]

        # get the input binned datasets for the position correlations
        Momentum_In = tuple((H5BinFiles[n]["CORRELATIONS"][correlation_type][key]["MOMENTUM"] for n in 1:N_pIDs)...)

        # iterate over all scatting momentum vectors
        for c in CartesianIndices(dims)
            # concatenate rebinned data for each pID together
            data = vcat((
                # rebin the data associated with single pID
                rebin(
                    # read in the binned data associated with each pID
                    Momentum_In[n][:,c.I...],
                    n_bins
                )
                # iterate over pIDs
                for n in 1:N_pIDs
            )...)
            # calculate the stats doing jackknife reweighting
            average[c], stdev[c] = jackknife(
                /, data, binned_sign;
                jackknife_sample_means, jackknife_g
            )
        end

        # record the final stats
        Momentum_Out["MEAN"] = average
        Momentum_Out["STD"] = stdev
    end

    return nothing
end


# allocate HDF5 stats file
function allocate_stats_file!(
    H5StatsFile::HDF5.File,
    H5BinFile::HDF5.File,
    process_global_measurements::Bool,
    process_local_measurements::Bool,
    standard_equal_time::Vector{String},
    standard_time_displaced::Vector{String},
    standard_integrated::Vector{String},
    composite_equal_time::Vector{String},
    composite_time_displaced::Vector{String},
    composite_integrated::Vector{String}
)

    # initialize global measurements group
    Global_In = H5BinFile["GLOBAL"]
    Global_Out = create_group(H5StatsFile, "GLOBAL")
    if process_global_measurements
        for key in keys(Global_In)
            create_group(Global_Out, key)
        end
    end

    # initialize local measurements group
    Local_In = H5BinFile["LOCAL"]
    Local_Out = create_group(H5StatsFile, "LOCAL")
    if process_local_measurements
        for key in keys(Local_In)
            Local_Measurement_In = Local_In[key]
            Local_Measurement_Out = create_group(Local_Out, key)
            attributes(Local_Measurement_Out)["ID_TYPE"] = read_attribute(Local_Measurement_In, "ID_TYPE")
        end
    end

    # create groups to contain correlation measurements
    Correlations_In = create_group(H5StatsFile, "CORRELATIONS")
    Standard_In = create_group(Correlations_In, "STANDARD")
    Composite_In = create_group(Correlations_In, "COMPOSITE")
    create_group(Standard_In, "EQUAL-TIME")
    create_group(Standard_In, "TIME-DISPLACED")
    create_group(Standard_In, "INTEGRATED")
    create_group(Composite_In, "EQUAL-TIME")
    create_group(Composite_In, "TIME-DISPLACED")
    create_group(Composite_In, "INTEGRATED")

    # initialize standard equal-time correlation measurements
    init_correlation_type_group!(
        H5StatsFile["CORRELATIONS/STANDARD/EQUAL-TIME"],
        H5BinFile["CORRELATIONS/STANDARD/EQUAL-TIME"],
        standard_equal_time
    )

    # initialize standard time-displaced correlation measurements
    init_correlation_type_group!(
        H5StatsFile["CORRELATIONS/STANDARD/TIME-DISPLACED"],
        H5BinFile["CORRELATIONS/STANDARD/TIME-DISPLACED"],
        standard_time_displaced
    )

    # initialize standard integrated correlation measurements
    init_correlation_type_group!(
        H5StatsFile["CORRELATIONS/STANDARD/INTEGRATED"],
        H5BinFile["CORRELATIONS/STANDARD/INTEGRATED"],
        standard_integrated
    )

    # initialize composite equal-time correlation measurements
    init_correlation_type_group!(
        H5StatsFile["CORRELATIONS/COMPOSITE/EQUAL-TIME"],
        H5BinFile["CORRELATIONS/COMPOSITE/EQUAL-TIME"],
        composite_equal_time
    )

    # initialize composite time-displaced correlation measurements
    init_correlation_type_group!(
        H5StatsFile["CORRELATIONS/COMPOSITE/TIME-DISPLACED"],
        H5BinFile["CORRELATIONS/COMPOSITE/TIME-DISPLACED"],
        composite_time_displaced
    )

    # initialize composite integrated correlation measurements
    init_correlation_type_group!(
        H5StatsFile["CORRELATIONS/COMPOSITE/INTEGRATED"],
        H5BinFile["CORRELATIONS/COMPOSITE/INTEGRATED"],
        composite_integrated
    )

    return nothing
end

# initialize a specific type of correlation measurement
function init_correlation_type_group!(
    CorrelationTypeGroup_Out::HDF5.Group,
    CorrelationTypeGroup_In::HDF5.Group,
    correlations::Vector{String}
)

    for key in correlations
        Correlation_In = CorrelationTypeGroup_In[key]
        Correlation_Out = create_group(CorrelationTypeGroup_Out, key)
        if haskey(attrs(Correlation_In), "ID_TYPE")
            attributes(Correlation_Out)["ID_TYPE"] = read_attribute(Correlation_In, "ID_TYPE")
            attributes(Correlation_Out)["ID_PAIRS"] = read_attribute(Correlation_In, "ID_PAIRS")
        end
        Position = create_group(Correlation_Out, "POSITION")
        Momentum = create_group(Correlation_Out, "MOMENTUM")
        attributes(Position)["DIM_LABELS"] = read_attribute(Correlation_In["POSITION"], "DIM_LABELS")[2:end]
        attributes(Momentum)["DIM_LABELS"] = read_attribute(Correlation_In["MOMENTUM"], "DIM_LABELS")[2:end]
    end

    return nothing
end