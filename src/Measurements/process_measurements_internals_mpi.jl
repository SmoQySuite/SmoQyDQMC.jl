# function to process the statistics for a single pID
function _process_measurements(
    comm::MPI.Comm,
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

    # make sure number of pIDs matches number of MPI processes
    @assert length(pIDs) == MPI.Comm_size(comm)

    # number of pID processes
    N_pIDs = length(pIDs)

    # get MPI rank
    rank = MPI.Comm_rank(comm)

    # if is the root process
    isroot = iszero(rank)

    # get the process ID
    pID = pIDs[rank+1]

    # open output HDF5 stats file if root process
    h5_stats_filename = joinpath(folder, filename)
    H5StatsFile = isroot ? h5open(h5_stats_filename, "w") : nothing

    # directory containing binned data
    bin_folder = joinpath(folder, "bins")

    # open bin File
    h5_bin_filename = joinpath(bin_folder, "bins_pID-$(pID).h5")
    H5BinFile = h5open(h5_bin_filename, "r")

    # get the number of bins
    N_data_bins = read_attribute(H5BinFile, "N_BINS")
    N_bins = isnothing(N_bins) ? N_data_bins : N_bins
    @assert (N_data_bins % N_bins) == 0

    # get all equal-time correlations if necessary
    if process_all_equal_time_measurements
        standard_equal_time  = keys(H5BinFile["CORRELATIONS/STANDARD/EQUAL-TIME"])
        composite_equal_time = keys(H5BinFile["CORRELATIONS/COMPOSITE/EQUAL-TIME"])
    end

    # get all time-displaced correlations if necessary
    if process_all_time_displaced_measurements
        standard_time_displaced = keys(H5BinFile["CORRELATIONS/STANDARD/TIME-DISPLACED"])
        composite_time_displaced = keys(H5BinFile["CORRELATIONS/COMPOSITE/TIME-DISPLACED"])
    end

    # get all integrated correlations if necessary
    if process_all_integrated_measurements
        standard_integrated  = keys(H5BinFile["CORRELATIONS/STANDARD/INTEGRATED"])
        composite_integrated = keys(H5BinFile["CORRELATIONS/COMPOSITE/INTEGRATED"])
    end

    # allocate HDF5 stats file if root process
    if isroot
        allocate_stats_file!(
            H5StatsFile, H5BinFile,
            process_global_measurements, process_local_measurements,
            standard_equal_time, standard_time_displaced, standard_integrated,
            composite_equal_time, composite_time_displaced, composite_integrated
        )

        # record pIDs associated with stats HDF5 files
        attributes(H5StatsFile)["PIDS"] = pIDs
    end
    MPI.Barrier(comm)

    # get system size and inverse temperature
    N_orbitals = read_attribute(H5BinFile, "N_ORBITALS")
    β = read_attribute(H5BinFile, "BETA")

    # record metadata about stats
    if isroot
        # record metadata about stats to computes
        attributes(H5StatsFile)["BETA"] = β
        attributes(H5StatsFile)["N_ORBITALS"] = N_orbitals
        attributes(H5StatsFile)["N_BINS"] = N_bins
    end

    # calculate the binned average sign for each HDF5 containing binned data
    binned_sign = bin_means(read(H5BinFile["GLOBAL/sgn"]), N_bins)

    # preallocate arrays for jackknife
    jackknife_sample_means = (similar(binned_sign), similar(binned_sign))
    jackknife_g = similar(binned_sign)

    # type of reported mean and standard deviations
    Tmean = eltype(binned_sign)
    Tstd = real(Tmean)

    # get the output global measurement stats group
    Global_Out = isroot ? H5StatsFile["GLOBAL"] : nothing

    # get the input global measurements group
    Global_In = H5BinFile["GLOBAL"]

    # iterate over global measurements
    for key in keys(Global_In)
        # get the binned values
        binned_vals = bin_means(read(Global_In[key]), N_bins)
        # if a global measurement does not require reweighting
        if startswith(key,"sgn") || startswith(key,"log") || startswith(key,"action") || startswith(key,"chemical_potential")
            avg = mean(binned_vals)
            err = varm(binned_vals, avg)
        # if a global measurement requires reweighting
        else
            avg, err = jackknife(
                /, binned_vals, binned_sign,
                jackknife_sample_means = jackknife_sample_means,
                jackknife_g = jackknife_g
            )
            err = abs2(err)
        end
        # get final stats
        avg = MPI.Reduce(avg, +, comm)
        err = MPI.Reduce(err, +, comm)
        # record final stats
        if isroot
            Global_Out[key]["MEAN"] = avg / N_pIDs
            Global_Out[key]["STD"] = sqrt(err) / N_pIDs
        end
    end

    # calculate the compressibility
    n  = bin_means(read(Global_In["density"]), N_bins)
    N² = bin_means(read(Global_In["Nsqrd"]),   N_bins)
    S  = binned_sign
    κ, Δκ = jackknife(
        (n̄, N̄², S̄) -> (β/N_orbitals)*(N̄²/S̄ - (N_orbitals*n̄/S̄)^2),
        n, N², S
    )
    κ = MPI.Reduce(κ, +, comm)
    varκ = MPI.Reduce(abs2(Δκ), +, comm)
    if isroot
        Compressibility = create_group(Global_Out, "compressibility")
        Compressibility["MEAN"] = κ / N_pIDs
        Compressibility["STD"] = sqrt(varκ) / N_pIDs
    end

    # get the output local measurement group
    Local_Out = isroot ? H5StatsFile["LOCAL"] : nothing

    # get the input local measurements group
    Local_In = H5BinFile["LOCAL"]

    # iterate over local measurements
    for key in keys(Local_In)
        # get the input measurement bins dataset
        Measurement_In = Local_In[key]
        # get the dimensions the output dataset needs to be
        dims = size(Measurement_In)[2:end]
        # array to contain measurement means and variances
        Measurement_Mean = zeros(Tmean, dims)
        Measurement_Std  = zeros(Tstd, dims)
        # iterate over number of given type of local measurement
        for c in CartesianIndices(dims)
            # calculate statistics
            Measurement_Mean[c], Measurement_Std[c] = binning_analysis(
                Measurement_In[:,c.I...], binned_sign,
                jackknife_sample_means, jackknife_g
            )
        end
        # calculate variances
        Measurement_Var = Measurement_Std
        @. Measurement_Var = abs2(Measurement_Std)
        # collect stats across processes
        MPI.Reduce!(Measurement_Mean, +, comm)
        MPI.Reduce!(Measurement_Var, +, comm)
        # if root process
        if isroot
            # calculate final mean and std
            @. Measurement_Mean = Measurement_Mean / N_pIDs
            @. Measurement_Std = sqrt(Measurement_Var) / N_pIDs
            # record the final stats
            Measurement_Out = Local_Out[key]
            Measurement_Out["MEAN"] = Measurement_Mean
            Measurement_Out["STD"] = Measurement_Std
        end
    end

    # process standard equal-time correlations
    process_correlations!(
        comm, H5StatsFile, H5BinFile,
        "CORRELATIONS/STANDARD/EQUAL-TIME",
        standard_equal_time,
        binned_sign, jackknife_sample_means, jackknife_g
    )

    # process standard time-displaced correlations
    process_correlations!(
        comm, H5StatsFile, H5BinFile,
        "CORRELATIONS/STANDARD/TIME-DISPLACED",
        standard_time_displaced,
        binned_sign, jackknife_sample_means, jackknife_g
    )

    # process standard INTEGRATED correlations
    process_correlations!(
        comm, H5StatsFile, H5BinFile,
        "CORRELATIONS/STANDARD/INTEGRATED",
        standard_integrated,
        binned_sign, jackknife_sample_means, jackknife_g
    )

    # process composite equal-time correlations
    process_correlations!(
        comm, H5StatsFile, H5BinFile,
        "CORRELATIONS/COMPOSITE/EQUAL-TIME",
        composite_equal_time,
        binned_sign, jackknife_sample_means, jackknife_g
    )

    # process composite time-displaced correlations
    process_correlations!(
        comm, H5StatsFile, H5BinFile,
        "CORRELATIONS/COMPOSITE/TIME-DISPLACED",
        composite_time_displaced,
        binned_sign, jackknife_sample_means, jackknife_g
    )

    # process composite INTEGRATED correlations
    process_correlations!(
        comm, H5StatsFile, H5BinFile,
        "CORRELATIONS/COMPOSITE/INTEGRATED",
        composite_integrated,
        binned_sign, jackknife_sample_means, jackknife_g
    )

    # delete HDF5 files containing binned data
    if rm_binned_data
        rm_bins(comm, folder)
    end

    return h5_stats_filename
end

function process_correlations!(
    comm::MPI.Comm,
    H5StatsFile::Union{HDF5.File,Nothing},
    H5BinFile::HDF5.File,
    correlation_type::String,
    correlations::Vector{String},
    binned_sign::Vector{Complex{T}},
    jackknife_sample_means = (similar(binned_signs), similar(binned_signs)),
    jackknife_g = similar(binned_signs)
) where {T<:AbstractFloat}

    # get current MPI rank
    rank = MPI.Comm_rank(comm)
    # get input and output groups containing correlation type
    Correlations_In = H5BinFile[correlation_type]
    Correlations_Out = isnothing(H5StatsFile) ? nothing : H5StatsFile[correlation_type]
    # iterate over correlations
    for key in correlations
        # input correlations
        Correlation_In = Correlations_In[key]
        Position_In = Correlation_In["POSITION"]
        Momentum_In = Correlation_In["MOMENTUM"]
        # get dimensions of output correlation group
        dim = size(Position_In)[2:end]
        # initialize arrays to contain mean and variance
        Mean_Out = zeros(Complex{T}, dim)
        Var_Out = zeros(T, dim)
        # iterate over correlations
        for c in CartesianIndices(dim)
            # calculate stats
            avg, err = binning_analysis(
                Position_In[:,c.I...], binned_sign,
                jackknife_sample_means, jackknife_g
            )
            # record stats
            Mean_Out[c] = avg
            Var_Out[c] = abs(err)
        end
        # collect stats from all MPI processes
        MPI.Reduce!(Mean_Out, +, comm)
        MPI.Reduce!(Var_Out, +, comm)
        # if root process
        if iszero(rank)
            # get number MPI processes
            N_pIDs = MPI.Comm_size(comm)
            # calculate final stats
            @. Mean_Out = Mean_Out / N_pIDs
            @. Var_Out = sqrt(Var_Out) / N_pIDs
            Std_Out = Var_Out
            # get output correlation group
            Position_Out = Correlations_Out[key]["POSITION"]
            # record final stats
            Position_Out["MEAN"] = Mean_Out
            Position_Out["STD"] = Std_Out
        end
        # initialize arrays to contain mean and variance
        fill!(Mean_Out, zero(Complex{T}))
        fill!(Var_Out, zero(T))
        # iterate over correlations
        for c in CartesianIndices(dim)
            # calculate stats
            avg, err = binning_analysis(
                Momentum_In[:,c.I...], binned_sign,
                jackknife_sample_means, jackknife_g
            )
            # record stats
            Mean_Out[c] = avg
            Var_Out[c] = abs(err)
        end
        # collect stats from all MPI processes
        MPI.Reduce!(Mean_Out, +, comm)
        MPI.Reduce!(Var_Out, +, comm)
        # if root process
        if iszero(rank)
            # get number MPI processes
            N_pIDs = MPI.Comm_size(comm)
            # calculate final stats
            @. Mean_Out = Mean_Out / N_pIDs
            @. Var_Out = sqrt(Var_Out) / N_pIDs
            Std_Out = Var_Out
            # get output correlation group
            Momentum_Out = Correlations_Out[key]["MOMENTUM"]
            # record final stats
            Momentum_Out["MEAN"] = Mean_Out
            Momentum_Out["STD"] = Std_Out
        end
    end

    return nothing
end