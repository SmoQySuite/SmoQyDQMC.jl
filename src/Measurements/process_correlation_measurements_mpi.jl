function process_correlation_measurements(
    comm::MPI.Comm,
    folder::String,
    N_bins::Int,
    pIDs::Vector{Int} = Int[],
    types::Vector{String} = ["equal-time", "time-displaced", "integrated"],
    spaces::Vector{String} = ["position", "momentum"]
)

    # set the walkers to iterate over
    if isempty(pIDs)

        # get the number of MPI walkers
        N_walkers = MPI.Comm_size(comm)

        # get the pIDs
        pIDs = collect(0:(N_walkers-1))
    end

    # get mpi ID
    mpiID = MPI.Comm_rank(comm)

    # get corresponding pID
    pID = pIDs[mpiID+1]

    # load model summary parameters
    β, Δτ, Lτ, model_geometry = load_model_summary(folder)

    # calculate bin intervals
    bin_intervals = get_bin_intervals(folder, N_bins, pID)

    # get binned sign
    binned_sign = get_average_sign(folder, bin_intervals, pID)

    # process correlation measurements
    _process_correlation_measurements(comm, folder, pID, types, spaces, Lτ, model_geometry, bin_intervals, binned_sign)

    return nothing
end

# process correlation measurements
function _process_correlation_measurements(
    comm::MPI.Comm,
    folder::String,
    pID::Int,
    types::Vector{String},
    spaces::Vector{String},
    Lτ::Int,
    model_geometry::ModelGeometry{D,T,N},
    bin_intervals::Vector{UnitRange{Int}},
    binned_sign::Vector{Complex{T}}
) where {D, T<:AbstractFloat, N}

    # get MPI rank
    mpiID = MPI.Comm_rank(comm)

    # iterate over correlation types
    for type in types

        # get the name of each correlation measurement that was made
        correlation_folder = joinpath(folder, type)

        # get the names of each correlation
        correlations = filter(i -> isdir(joinpath(correlation_folder,i)), readdir(correlation_folder))

        # iterate over correlations
        for correlation in correlations

            # iterate over spaces
            for space in spaces

                # process correlation measurement
                if type == "time-displaced"
                    if haskey(CORRELATION_FUNCTIONS, correlation)
                        _process_correlation_measurement(
                            comm, mpiID, pID, folder, correlation, Lτ, space, binned_sign, bin_intervals, model_geometry
                        )
                    else
                        _process_composite_correlation_measurement(
                            comm, mpiID, pID, folder, correlation, Lτ, space, binned_sign, bin_intervals, model_geometry
                        )
                    end
                else
                    if haskey(CORRELATION_FUNCTIONS, correlation)
                        _process_correlation_measurement(
                            comm, mpiID, pID, folder, correlation, type, space, binned_sign, bin_intervals, model_geometry
                        )
                    else
                        _process_composite_correlation_measurement(
                            comm, mpiID, pID, folder, correlation, type, space, binned_sign, bin_intervals, model_geometry
                        )
                    end
                end
            end
        end
    end

    return nothing
end

# process equal-time/integrated correlation measurement
function _process_correlation_measurement(
    comm::MPI.Comm,
    mpiID::Int,
    pID::Int,
    folder::String,
    correlation::String,
    type::String,
    space::String,
    binned_sign::Vector{Complex{T}},
    bin_intervals::Vector{UnitRange{Int}},
    model_geometry::ModelGeometry{D,T,N}
) where {D, T<:AbstractFloat, N}

    # get the folder the stats will be written to
    write_folder = joinpath(folder, type, correlation)

    # get the read folders
    read_folder = joinpath(write_folder, space)

    # get size of lattice
    lattice = model_geometry.lattice::Lattice{D}
    L = lattice.L

    # get correlation ID pairs
    pairs = JLD2.load(joinpath(read_folder, @sprintf("bin-1_pID-%d.jld2", pID)), "id_pairs")

    # number of bins
    N_bins = length(bin_intervals)

    # container for binned correlation data
    binned_correlation = zeros(Complex{T}, N_bins, L...)
    correlation_avg = zeros(Complex{T}, L...)
    correlation_std = zeros(T, L...)
    correlation_var = correlation_std

    # pre-allocate arrays for jackknife
    jackknife_sample_means = (zeros(Complex{T}, N_bins), zeros(Complex{T}, N_bins))
    jackknife_g = zeros(Complex{T}, N_bins)

    # if root process
    if iszero(mpiID)

        # number of MPI ranks
        N_ranks = MPI.Comm_size(comm)

        # stats filename
        filename = @sprintf("%s_%s_%s_stats.csv", correlation, space, type)

        # open stats files
        open(joinpath(write_folder, filename), "w") do fout

            # get the id type
            id_type = CORRELATION_FUNCTIONS[correlation]

            # write header to file
            if space == "position"
                write(fout, join(("INDEX", "$(id_type)_2", "$(id_type)_1", ("R_$d" for d in D:-1:1)..., "MEAN_R", "MEAN_I", "STD\n"), " "))
            else
                write(fout, join(("INDEX", "$(id_type)_2", "$(id_type)_1", ("K_$d" for d in D:-1:1)..., "MEAN_R", "MEAN_I", "STD\n"), " "))
            end

            # initialize index to zero
            index = 0

            # iterate over id pairs
            for n in eachindex(pairs)

                # read in binned correlation data
                read_correlation_measurement!(binned_correlation, folder, correlation, type, space, n, pID, bin_intervals)

                # calculate average and error for current pID
                fill!(correlation_avg, 0)
                fill!(correlation_var, 0)
                analyze_correlations!(correlation_avg, correlation_var, binned_correlation, binned_sign, jackknife_sample_means, jackknife_g)

                # collect all the results
                MPI.Reduce!(correlation_avg, +, comm)
                MPI.Reduce!(correlation_var, +, comm)

                # normalize the stats

                @. correlation_avg = correlation_avg / N_ranks
                @. correlation_std = sqrt(correlation_var) / N_ranks

                # write the correlation stats to file
                index = write_correlation(fout, pairs[n], index, correlation_avg, correlation_std)
            end
        end
    # if not the root process
    else

        # iterate over id pairs
        for n in eachindex(pairs)

            # read in binned correlation data
            read_correlation_measurement!(binned_correlation, folder, correlation, type, space, n, pID, bin_intervals)

            # calculate average and error for current pID
            # reset correlation containers
            fill!(correlation_avg, 0)
            fill!(correlation_std, 0)
            analyze_correlations!(correlation_avg, correlation_var, binned_correlation, binned_sign, jackknife_sample_means, jackknife_g)

            # collect all the results
            MPI.Reduce!(correlation_avg, +, comm)
            MPI.Reduce!(correlation_var, +, comm)
        end
    end

    return nothing
end


# process equal-time/integrated composite correlation measurement
function _process_composite_correlation_measurement(
    comm::MPI.Comm,
    mpiID::Int,
    pID::Int,
    folder::String,
    correlation::String,
    type::String,
    space::String,
    binned_sign::Vector{Complex{T}},
    bin_intervals::Vector{UnitRange{Int}},
    model_geometry::ModelGeometry{D,T,N}
) where {D, T<:AbstractFloat, N}

    # get the folder the stats will be written to
    write_folder = joinpath(folder, type, correlation)

    # get the read folders
    read_folder = joinpath(write_folder, space)

    # get size of lattice
    lattice = model_geometry.lattice::Lattice{D}
    L = lattice.L

    # number of bins
    N_bins = length(bin_intervals)

    # container for binned correlation data
    binned_correlation = zeros(Complex{T}, N_bins, L...)
    correlation_avg = zeros(Complex{T}, L...)
    correlation_std = zeros(T, L...)
    correlation_var = correlation_std

    # pre-allocate arrays for jackknife
    jackknife_sample_means = (zeros(Complex{T}, N_bins), zeros(Complex{T}, N_bins))
    jackknife_g = zeros(Complex{T}, N_bins)

    # if root process
    if iszero(mpiID)

        # number of MPI ranks
        N_ranks = MPI.Comm_size(comm)

        # stats filename
        filename = @sprintf("%s_%s_%s_stats.csv", correlation, space, type)

        # open stats files
        open(joinpath(write_folder, filename), "w") do fout

            # write header to file
            if space == "position"
                write(fout, join(("INDEX", ("R_$d" for d in D:-1:1)..., "MEAN_R", "MEAN_I", "STD\n"), " "))
            else
                write(fout, join(("INDEX", ("K_$d" for d in D:-1:1)..., "MEAN_R", "MEAN_I", "STD\n"), " "))
            end

            # initialize index to zero
            index = 0

            # read in binned correlation data
            read_correlation_measurement!(binned_correlation, folder, correlation, type, space, pID, bin_intervals)

            # calculate average and error for current pID
            fill!(correlation_avg, 0)
            fill!(correlation_var, 0)
            analyze_correlations!(correlation_avg, correlation_var, binned_correlation, binned_sign, jackknife_sample_means, jackknife_g)

            # collect all the results
            MPI.Reduce!(correlation_avg, +, comm)
            MPI.Reduce!(correlation_var, +, comm)

            # normalize the stats

            @. correlation_avg = correlation_avg / N_ranks
            @. correlation_std = sqrt(correlation_var) / N_ranks

            # write the correlation stats to file
            index = write_correlation(fout, index, correlation_avg, correlation_std)
        end
    # if not the root process
    else

        # read in binned correlation data
        read_correlation_measurement!(binned_correlation, folder, correlation, type, space, pID, bin_intervals)

        # calculate average and error for current pID
        # reset correlation containers
        fill!(correlation_avg, 0)
        fill!(correlation_std, 0)
        analyze_correlations!(correlation_avg, correlation_var, binned_correlation, binned_sign, jackknife_sample_means, jackknife_g)

        # collect all the results
        MPI.Reduce!(correlation_avg, +, comm)
        MPI.Reduce!(correlation_var, +, comm)
    end

    return nothing
end


# process time-displaced correlation measurement
function _process_correlation_measurement(
    comm::MPI.Comm,
    mpiID::Int,
    pID::Int,
    folder::String,
    correlation::String,
    Lτ::Int,
    space::String,
    binned_sign::Vector{Complex{T}},
    bin_intervals::Vector{UnitRange{Int}},
    model_geometry::ModelGeometry{D,T,N},
) where {D, T<:AbstractFloat, N}

    # get the folder the stats will be written to
    write_folder = joinpath(folder, "time-displaced", correlation)

    # get the read folders
    read_folder = joinpath(write_folder, space)

    # get size of lattice
    lattice = model_geometry.lattice::Lattice{D}
    L = lattice.L

    # get correlation ID pairs
    pairs = JLD2.load(joinpath(read_folder, @sprintf("bin-1_pID-%d.jld2", pID)), "id_pairs")

    # number of bins
    N_bins = length(bin_intervals)

    # container for binned correlation data
    binned_correlation = zeros(Complex{T}, N_bins, L...)
    correlation_avg = zeros(Complex{T}, L...)
    correlation_std = zeros(T, L...)
    correlation_var = correlation_std

    # pre-allocate arrays for jackknife
    jackknife_sample_means = (zeros(Complex{T}, N_bins), zeros(Complex{T}, N_bins))
    jackknife_g = zeros(Complex{T}, N_bins)

    # if root process
    if iszero(mpiID)

        # number of MPI ranks
        N_ranks = MPI.Comm_size(comm)

        # stats filename
        filename = @sprintf("%s_%s_%s_stats.csv", correlation, space, "time-displaced")

        # open stats files
        open(joinpath(write_folder, filename), "w") do fout

            # get the id type
            id_type = CORRELATION_FUNCTIONS[correlation]

            # write header to file
            if space == "position"
                write(fout, join(("INDEX", "$(id_type)_2", "$(id_type)_1", "TAU", ("R_$d" for d in D:-1:1)..., "MEAN_R", "MEAN_I", "STD\n"), " "))
            else
                write(fout, join(("INDEX", "$(id_type)_2", "$(id_type)_1", "TAU", ("K_$d" for d in D:-1:1)..., "MEAN_R", "MEAN_I", "STD\n"), " "))
            end

            # initialize index to zero
            index = 0

            # iterate over id pairs
            for n in eachindex(pairs)

                # iterate of imaginary time displacements
                for l in 0:Lτ

                    # read in binned correlation data
                    read_correlation_measurement!(binned_correlation, folder, correlation, l, space, n, pID, bin_intervals)

                    # calculate average and error for current pID
                    fill!(correlation_avg, 0)
                    fill!(correlation_var, 0)
                    analyze_correlations!(correlation_avg, correlation_var, binned_correlation, binned_sign, jackknife_sample_means, jackknife_g)

                    # collect all the results
                    MPI.Reduce!(correlation_avg, +, comm)
                    MPI.Reduce!(correlation_var, +, comm)

                    # normalize the stats
                    @. correlation_avg = correlation_avg / N_ranks
                    @. correlation_std = sqrt(correlation_var) / N_ranks

                    # write the correlation stats to file
                    index = write_correlation(fout, pairs[n], index, l, correlation_avg, correlation_std)
                end
            end
        end
    # if not the root process
    else

        # iterate over id pairs
        for n in eachindex(pairs)

            # iterate of imaginary time displacements
            for l in 0:Lτ

                # read in binned correlation data
                read_correlation_measurement!(binned_correlation, folder, correlation, l, space, n, pID, bin_intervals)

                # calculate average and error for current pID
                fill!(correlation_avg, 0)
                fill!(correlation_var, 0)
                analyze_correlations!(correlation_avg, correlation_var, binned_correlation, binned_sign, jackknife_sample_means, jackknife_g)

                # collect all the results
                MPI.Reduce!(correlation_avg, +, comm)
                MPI.Reduce!(correlation_var, +, comm)
            end
        end
    end

    return nothing
end


# process time-displaced composite correlation measurement
function _process_composite_correlation_measurement(
    comm::MPI.Comm,
    mpiID::Int,
    pID::Int,
    folder::String,
    correlation::String,
    Lτ::Int,
    space::String,
    binned_sign::Vector{Complex{T}},
    bin_intervals::Vector{UnitRange{Int}},
    model_geometry::ModelGeometry{D,T,N},
) where {D, T<:AbstractFloat, N}

    # get the folder the stats will be written to
    write_folder = joinpath(folder, "time-displaced", correlation)

    # get the read folders
    read_folder = joinpath(write_folder, space)

    # get size of lattice
    lattice = model_geometry.lattice::Lattice{D}
    L = lattice.L

    # number of bins
    N_bins = length(bin_intervals)

    # container for binned correlation data
    binned_correlation = zeros(Complex{T}, N_bins, L...)
    correlation_avg = zeros(Complex{T}, L...)
    correlation_std = zeros(T, L...)
    correlation_var = correlation_std

    # pre-allocate arrays for jackknife
    jackknife_sample_means = (zeros(Complex{T}, N_bins), zeros(Complex{T}, N_bins))
    jackknife_g = zeros(Complex{T}, N_bins)

    # if root process
    if iszero(mpiID)

        # number of MPI ranks
        N_ranks = MPI.Comm_size(comm)

        # stats filename
        filename = @sprintf("%s_%s_%s_stats.csv", correlation, space, "time-displaced")

        # open stats files
        open(joinpath(write_folder, filename), "w") do fout

            # write header to file
            if space == "position"
                write(fout, join(("INDEX", "TAU", ("R_$d" for d in D:-1:1)..., "MEAN_R", "MEAN_I", "STD\n"), " "))
            else
                write(fout, join(("INDEX", "TAU", ("K_$d" for d in D:-1:1)..., "MEAN_R", "MEAN_I", "STD\n"), " "))
            end

            # initialize index to zero
            index = 0

            # iterate of imaginary time displacements
            for l in 0:Lτ

                # read in binned correlation data
                read_correlation_measurement!(binned_correlation, folder, correlation, l, space, pID, bin_intervals)

                # calculate average and error for current pID
                fill!(correlation_avg, 0)
                fill!(correlation_var, 0)
                analyze_correlations!(correlation_avg, correlation_var, binned_correlation, binned_sign, jackknife_sample_means, jackknife_g)

                # collect all the results
                MPI.Reduce!(correlation_avg, +, comm)
                MPI.Reduce!(correlation_var, +, comm)

                # normalize the stats
                @. correlation_avg = correlation_avg / N_ranks
                @. correlation_std = sqrt(correlation_var) / N_ranks

                # write the correlation stats to file
                index = write_correlation(fout, index, l, correlation_avg, correlation_std)
            end
        end
    # if not the root process
    else
        # iterate of imaginary time displacements
        for l in 0:Lτ

            # read in binned correlation data
            read_correlation_measurement!(binned_correlation, folder, correlation, l, space, pID, bin_intervals)

            # calculate average and error for current pID
            fill!(correlation_avg, 0)
            fill!(correlation_var, 0)
            analyze_correlations!(correlation_avg, correlation_var, binned_correlation, binned_sign, jackknife_sample_means, jackknife_g)

            # collect all the results
            MPI.Reduce!(correlation_avg, +, comm)
            MPI.Reduce!(correlation_var, +, comm)
        end
    end

    return nothing
end