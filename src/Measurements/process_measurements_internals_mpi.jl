# function to process the statistics for a single pID
function _process_measurements(
    comm::MPI.Comm,
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

    # get the number of bins per pID
    n_data_bins = read_attribute(H5BinFile, "N_BINS")
    n_bins = isnothing(n_bins) ? n_data_bins : n_bins
    @assert (n_data_bins % n_bins) == 0

    # calculate total number bins across all pIDs
    N_bins = n_bins * N_pIDs

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
    Δτ = read_attribute(H5BinFile, "DELTA_TAU")
    Lτ = read_attribute(H5BinFile, "L_TAU")
    L = read_attribute(H5BinFile, "L")
    D = length(L)

    # record metadata about stats
    if isroot
        # record metadata about stats to computes
        attributes(H5StatsFile)["BETA"] = β
        attributes(H5StatsFile)["DELTA_TAU"] = Δτ
        attributes(H5StatsFile)["L_TAU"] = Lτ
        attributes(H5StatsFile)["N_ORBITALS"] = N_orbitals
        attributes(H5StatsFile)["L"] = L
        attributes(H5StatsFile)["N_BINS"] = N_bins
    end

    # calculate the binned average sign for each HDF5 containing binned data
    binned_sign = MPI.gather(rebin(read(H5BinFile["GLOBAL/sgn"]), n_bins), comm)
    binned_sign = isroot ? vcat(binned_sign...) : nothing

    # preallocate arrays for jackknife
    jackknife_sample_means = (similar(binned_sign), similar(binned_sign))
    jackknife_g = similar(binned_sign)

    # type of reported mean and standard deviations
    T_mean = eltype(binned_sign)
    T_std = real(Tmean)

    # get the output global measurement stats group
    Global_Out = isroot ? H5StatsFile["GLOBAL"] : nothing

    # get the input global measurements group
    Global_In = H5BinFile["GLOBAL"]

    # iterate over global measurements
    for key in keys(Global_In)
        # read in the global measurement bins for the current pID and rebin it
        binned_vals = MPI.gather(rebin(read(Global_In[key]), n_bins), comm)
        # if root process
        if isroot
            # concatenate all the data together
            binned_vals = vcat(binned_vals...)
            # if a global measurement does not require re-weighting
            if startswith(key,"sgn") || startswith(key,"log") || startswith(key,"action") || startswith(key,"chemical_potential")
                avg = mean(binned_vals)
                stdev = stdm(binned_vals, avg) / sqrt(N_bins)
            # if a global measurement requires re-weighting
            else
                avg, stdev = jackknife(
                    /, binned_vals, binned_sign;
                    jackknife_sample_means, jackknife_g
                )
            end
            # record the global measurement stats
            Global_Out["MEAN"] = avg
            Global_Out["STD"] = stdev
        end
    end

    # calculate the compressibility
    n̄ = rebin(read(Global_In["density"]), n_bins)
    N² = rebin(read(Global_In["Nsqrd"]), n_bins)
    n̄ = MPI.gather(n̄, comm)
    N² = MPI.gather(N², comm)
    if isroot
        n̄ = vcat(n̄...)
        N² = vcat(N²...)
        S = binned_sign
        κ, Δκ = jackknife(
            (n̄, N̄², S̄) -> (β/N_orbitals)*(N̄²/S̄ - (N_orbitals*n̄/S̄)^2),
            n̄, N², S
        )
        Compressibility = create_group(Global_Out, "compressibility")
        Compressibility["MEAN"] = κ
        Compressibility["STD"] = Δκ
    end

    # get the output local measurement group
    Local_Out = isroot ? H5StatsFile["LOCAL"] : nothing

    # get the input local measurements group
    Local_In = H5BinFile["LOCAL"]

    # iterate over local measurements
    for key in keys(Local_In)
        # get the input measurement bins dataset
        Measurement_In = Local_In[key]
        # gather rebinned measurement data
        data = MPI.gather(rebin(read(Measurement_In), n_bins), comm)
        # if root process
        if isroot
            # concatenate all the gathered data across pIDs
            data = vcat(data...)
            # number of IDs associated with local measurement
            N_IDs = size(data, 2)
            # allocate array to contain stats
            average = zeros(T_mean, N_IDs)
            stdev = zeros(T_std, N_IDs)
            # iterate over IDs associated with Local Measurement
            for ID in 1_N_IDs
                # perform jackknife reweighting to calculate stats
                average[ID], stdev[ID] = jackknife(
                    /, view(data, :, ID), binned_sign;
                    jackknife_sample_means, jackknife_g
                )
            end
            # record the local measurement stats
            Measurement_Out = Local_Out[key]
            Measurement_Out["MEAN"] = average
            Measurement_Out["STD"] = stdev
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
    D::Int,
    jackknife_sample_means = (similar(binned_sign), similar(binned_sign)),
    jackknife_g = similar(binned_sign)
) where {T<:AbstractFloat}

    # Get the number of bins
    N_bins = length(binned_sign)
    # Get the number of pIDs
    N_pID = MPI.Comm_size(comm)
    # get the number of bins per pID
    n_bins = N_bins ÷ N_pID
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
        dims = size(Position_In)
        # initialize array to contain stats
        average = iszero(rank) ? zeros(Complex{T}, dims[2:end]) : nothing
        stdev = iszero(rank) ? zeros(T, dims[2:end]) : nothing
        # get extent of lattice size
        L = dims[2:D+1]
        # get index range associated with lattice size
        Ls = tuple((1:l for l in L)...)
        # iterate over imaginary-time slice and ID pairs when relevant
        for n in CartesianIndices(dims[D+2:end])
            # read in the position data
            data = read(Position_In[:,Ls...,n.I...])
            # rebin the position data
            data = rebin(data, n_bins)
            # gather all the data into the root process
            data = MPI.gather(data, comm)
            # if root process
            if iszero(rank)
                # concatenate the bins from each pID together
                data = vcat(data...)
                # iterate over displacement vectors
                for c in CartesianIndices(L)
                    # perform jackknife reweighting
                    average[c.I..., n.I...], stdev[c.I..., n.I...] = jackknife(
                        /, view(data, :, c.I...), binned_sign;
                        jackknife_sample_means, jackknife_g
                    )
                end
            end
        end
        # record the position space correlation stats
        if iszero(rank)
            Correlations_Out[key]["POSITION"]["MEAN"] = average
            Correlations_Out[key]["POSITION"]["STD"] = stdev
        end
        # iterate over imaginary-time slice and ID pairs if relevant
        for n in CartesianIndices(dims[D+2:end])
            # read in the momentum data
            data = read(Momentum_In[:,Ls...,n.I...])
            # rebin the momentum data
            data = rebin(data, n_bins)
            # gather all the data into the root process
            data = MPI.gather(data, comm)
            # if root process
            if iszero(rank)
                # concatenate the bins from each pID together
                data = vcat(data...)
                # iterate over displacement vectors
                for c in CartesianIndices(L)
                    # perform jackknife reweighting
                    average[c.I..., n.I...], stdev[c.I..., n.I...] = jackknife(
                        /, view(data, :, c.I...), binned_sign;
                        jackknife_sample_means, jackknife_g
                    )
                end
            end
        end
        # record the momentum space correlation stats
        if iszero(rank)
            Correlations_Out[key]["MOMENTUM"]["MEAN"] = average
            Correlations_Out[key]["MOMENTUM"]["STD"] = stdev
        end
    end

    return nothing
end