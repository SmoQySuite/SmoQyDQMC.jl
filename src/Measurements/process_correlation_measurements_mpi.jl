# process time-displaced correlation measurement
function _process_correlation_measurement(
    comm::MPI.Comm,
    mpiID::Int,
    pID::Int,
    folder::String,
    correlation::String,
    space::String,
    binned_sign::Vector{Complex{T}},
    bin_intervals::Vector{UnitRange{Int}},
    model_geometry::ModelGeometry{D,T,N},
    Lτ::Int,
) where {T<:AbstractFloat}

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
    N_bin = length(bin_intervals)

    # container for binned correlation data
    binned_correlation = zeros(Complex{T}, N_bin, L...)
    correlation_avg = zeros(Complex{T}, L...)
    correlation_std = zeros(Complex{T}, L...)
    correlation_var = correlation_std

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
                    analyze_correlations!(correlation_avg, correlation_std, binned_correlation, binned_sign)

                    # collect all the results
                    MPI.Reduce!(correlation_avg, +, comm)
                    @. correlation_var = abs2(correlation_std)
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
                analyze_correlations!(correlation_avg, correlation_std, binned_correlation, binned_sign)

                # collect all the results
                MPI.Reduce!(correlation_avg, +, comm)
                @. correlation_var = abs2(correlation_std)
                MPI.Reduce!(correlation_var, +, comm)
            end
        end
    end

    return nothing
end