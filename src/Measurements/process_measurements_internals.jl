# function to process the statistics for a single pID
function _process_measurements(
    folder::String,
    filename::String,
    pIDs::Vector{Int},
    N_bins::Union{Nothing,Int},
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

    # number of HDF5 files containing binned data
    N_pIDs = length(pIDs)

    # get the number of bins
    N_data_bins = read_attribute(H5BinFiles[1], "N_BINS")
    N_bins = isnothing(N_bins) ? N_data_bins : N_bins
    @assert (N_data_bins % N_bins) == 0

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
    binned_sgns = [
        bin_means(read(H5BinFile["GLOBAL/sgn"]), N_bins) for H5BinFile in H5BinFiles
    ]

    # preallocate arrays for jackknife
    jackknife_sample_means = (similar(binned_sgns[1]), similar(binned_sgns[1]))
    jackknife_g = similar(binned_sgns[1])

    # type of reported mean and standard deviations
    T_mean = typeof(binned_sgns[1][1])
    T_std = real(T_mean)

    # get global measurement stats group
    Global_Out = H5StatsFile["GLOBAL"]

    # iterate over global measurements
    for key in keys(Global_Out)
        # initialize mean and variance of measurement to zero
        average, variance = zero(T_mean), zero(T_std)
        # if a global measurement does not require reweighting
        if startswith(key,"sgn") || startswith(key,"log") || startswith(key,"action") || startswith(key,"chemical_potential")
            # iterate over HDF5 files containing binned data
            for n in 1:N_pIDs
                # calculate stats for current files
                vals = read(H5BinFiles[n]["GLOBAL"][key])
                binned_vals = bin_means(vals, N_bins)
                average += mean(binned_vals)
                variance += var(binned_vals) / N_bins
            end
        # if a global measurement requires reweighting
        else
            # iterate over HDF5 files containing binned data
            for n in 1:N_pIDs
                # calculate stats for current file with reweighting
                vals = read(H5BinFiles[n]["GLOBAL"][key])
                avg, err = binning_analysis(vals, binned_sgns[n], jackknife_sample_means, jackknife_g)
                average += avg
                variance += abs2(err)
            end
        end
        # create group for global measurement
        Global_Measurement = Global_Out[key]
        # record final mean and standard deviation
        Global_Measurement["MEAN"] = average / N_pIDs
        Global_Measurement["STD"] = sqrt(variance) / N_pIDs
    end


    # calculate the compressibility
    κ_avg, κ_var = zero(T_mean), zero(T_std)
    for i in 1:N_pIDs
        n = bin_means(read(H5BinFiles[i]["GLOBAL/density"]), N_bins)
        N² = bin_means(read(H5BinFiles[i]["GLOBAL/Nsqrd"]), N_bins)
        S = binned_sgns[i]
        # calculate the compressibility using flucuation dissipation theorem
        # such that κ = (β/N)⋅(⟨N²⟩-⟨N⟩²), using the jackknife method
        # with reweighting to account for the sign problem
        κ, Δκ = jackknife(
            (n̄, N̄², S̄) -> (β/N_orbitals)*(N̄²/S̄ - (N_orbitals*n̄/S̄)^2),
            n, N², S
        )
        κ_avg += κ
        κ_var += Δκ^2
    end
    Compressibility = create_group(Global_Out, "compressibility")
    Compressibility["MEAN"] = κ_avg / N_pIDs
    Compressibility["STD"] = sqrt(κ_var) / N_pIDs

    # iterate over local measurements
    Local_Out = H5StatsFile["LOCAL"]
    for key in keys(Local_Out)
        # get local measurement group
        Measurement_Out = Local_Out[key]
        # initialize array to contain output stats
        dims = size(H5BinFiles[1]["LOCAL"][key])[2:end]
        Mean_Out = zeros(T_mean, dims)
        Var_Out = zeros(T_std, dims)
        # iterate over HDF5 files containing binned data
        for n in 1:N_pIDs
            # Get input local measurement
            Measurement_In = H5BinFiles[n]["LOCAL"][key]
            # iterate over output dataset elements
            for c in CartesianIndices(dims)
                # calculate stats using jackknife reweighting
                avg, err = binning_analysis(Measurement_In[:,c.I...], binned_sgns[n], jackknife_sample_means, jackknife_g)
                Mean_Out[c] += avg
                Var_Out[c] += abs2(err)
            end
        end
        # calculate the final stats
        @. Mean_Out = Mean_Out / N_pIDs
        @. Var_Out = sqrt(Var_Out) / N_pIDs
        Std_Out = Var_Out
        Measurement_Out["MEAN"] = Mean_Out
        Measurement_Out["STD"] = Std_Out
    end

    # process standard equal-time correlation measurements
    process_correlations!(
        H5StatsFile, H5BinFiles, "CORRELATIONS/STANDARD/EQUAL-TIME",
        binned_sgns, jackknife_sample_means, jackknife_g
    )

    # process standard time-displaced correlation measurements
    process_correlations!(
        H5StatsFile, H5BinFiles, "CORRELATIONS/STANDARD/TIME-DISPLACED",
        binned_sgns, jackknife_sample_means, jackknife_g
    )

    # process standard integrated correlation measurements
    process_correlations!(
        H5StatsFile, H5BinFiles, "CORRELATIONS/STANDARD/INTEGRATED",
        binned_sgns, jackknife_sample_means, jackknife_g
    )

    # process composite equal-time correlation measurements
    process_correlations!(
        H5StatsFile, H5BinFiles, "CORRELATIONS/COMPOSITE/EQUAL-TIME",
        binned_sgns, jackknife_sample_means, jackknife_g
    )

    # process composite time-displaced correlation measurements
    process_correlations!(
        H5StatsFile, H5BinFiles, "CORRELATIONS/COMPOSITE/TIME-DISPLACED",
        binned_sgns, jackknife_sample_means, jackknife_g
    )

    # process composite integrated correlation measurements
    process_correlations!(
        H5StatsFile, H5BinFiles, "CORRELATIONS/COMPOSITE/INTEGRATED",
        binned_sgns, jackknife_sample_means, jackknife_g
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
    binned_sgns::Vector{Vector{Complex{T}}},
    jackknife_sample_means = (similar(binned_sgns), similar(binned_sgns)),
    jackknife_g = similar(binned_sgns)
) where {T<:AbstractFloat}

    # get output correlation
    Correlations_Out = H5StatsFile[correlation_type]

    # get the number of pIDs
    N_pIDs = length(H5BinFiles)

    for key in keys(Correlations_Out)
        # get correlation group
        Correlation_Out = Correlations_Out[key]
        # dataset for mean and standard deviation
        Position_Out = Correlation_Out["POSITION"]
        Momentum_Out = Correlation_Out["MOMENTUM"]
        # initialize arrays to contain output correlation dataset
        dims = size(H5BinFiles[1][correlation_type][key]["POSITION"])[2:end]
        Mean_Out = zeros(Complex{T}, dims)
        Var_Out = zeros(T, dims)
        # iterate over HDF5 files containing binned dataset
        for n in 1:N_pIDs
            # get input correlation dataset
            Correlation_Dataset_In = H5BinFiles[n][correlation_type][key]["POSITION"]
            # iterate over output dataset elements
            for c in CartesianIndices(dims)
                # calculate statistics using jackknife reweighting for position-space correlation
                avg, err = binning_analysis(Correlation_Dataset_In[:,c.I...], binned_sgns[n], jackknife_sample_means, jackknife_g)
                Mean_Out[c] += avg
                Var_Out[c] += abs2(err)
            end
        end
        # calculate the final stats
        @. Mean_Out = Mean_Out / N_pIDs
        @. Var_Out = sqrt(Var_Out) / N_pIDs
        Std_Out = Var_Out
        # record the final stats
        Position_Out["MEAN"] = Mean_Out
        Position_Out["STD"] = Std_Out
        # initialize arrays to contain output correlation dataset to zero
        fill!(Mean_Out, zero(Complex{T}))
        fill!(Std_Out, zero(T))
        # iterate over HDF5 files containing binned dataset
        for n in 1:N_pIDs
            # get input correlation dataset
            Correlation_Dataset_In = H5BinFiles[n][correlation_type][key]["MOMENTUM"]
            # iterate over output dataset elements
            for c in CartesianIndices(dims)
                # calculate statistics using jackknife reweighting for position-space correlation
                avg, err = binning_analysis(Correlation_Dataset_In[:,c.I...], binned_sgns[n], jackknife_sample_means, jackknife_g)
                Mean_Out[c] += avg
                Var_Out[c] += abs2(err)
            end
        end
        # calculate the final stats
        @. Mean_Out = Mean_Out / N_pIDs
        @. Var_Out = sqrt(Var_Out) / N_pIDs
        Std_Out = Var_Out
        # record the final stats
        Momentum_Out["MEAN"] = Mean_Out
        Momentum_Out["STD"] = Std_Out
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

# calculate stats using jackknife based on vector
function binning_analysis(
    data::AbstractVector{Complex{T}},
    sgn::AbstractVector{Complex{T}},
    jackknife_sample_means = (similar(sgn), similar(sgn)),
    jackknife_g = similar(sgn)
) where {T<:AbstractFloat}

    N_bins = length(sgn)
    binned_data = bin_means(data, N_bins)
    avg, err = jackknife(
        /, binned_data, sgn,
        jackknife_sample_means = jackknife_sample_means,
        jackknife_g = jackknife_g
    )

    return avg, err
end


# calculate binned means of a vector of data assuming there are N_bins bins
function bin_means(
    data::AbstractVector{T},
    N_bins::Int
) where {T<:Number}

    N_data = length(data)
    @assert iszero(mod(N_data, N_bins))
    if N_bins == N_data
        binned_data = data
    else
        bin_size = N_data ÷ N_bins
        rdata = reshape(data, bin_size, N_bins)
        binned_data = dropdims(mean(rdata, dims = 1), dims = 1)
    end

    return binned_data
end