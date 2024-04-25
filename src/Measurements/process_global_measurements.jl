#################################
## PROCESS GLOBAL MEASUREMENTS ##
#################################

@doc raw"""
    process_global_measurements(
        folder::String,
        N_bins::Int,
        pIDs::Union{Vector{Int},Int} = Int[]
    )

    process_global_measurements(
        comm::MPI.Comm,
        folder::String,
        N_bins::Int,
        pIDs::Vector{Int} = Int[]
    )

Process global measurents for the specified process IDs, calculating the average and error for all global measurements
and writing the result to CSV file.
If `pIDs` is not specified, then results for all MPI walkers are averaged over.
"""
function process_global_measurements(
    folder::String,
    N_bins::Int,
    pIDs::Union{Vector{Int},Int} = Int[]
)

    # set the walkers to iterate over
    if isempty(pIDs)

        # get the number of MPI walkers
        N_walkers = get_num_walkers(folder)

        # get the pIDs
        pIDs = collect(0:(N_walkers-1))
    end

    # load model summary
    β, Δτ, Lτ, model_geometry = load_model_summary(folder)
    N_sites = nsites(model_geometry.unit_cell, model_geometry.lattice)

    # process the global measurements
    _process_global_measurements(folder, N_bins, pIDs, β, N_sites)

    return nothing
end

function process_global_measurements(
    comm::MPI.Comm,
    folder::String,
    N_bins::Int,
    pIDs::Vector{Int} = Int[]
)

    # set the walkers to iterate over
    if isempty(pIDs)

        # get the number of MPI walkers
        N_walkers = get_num_walkers(folder)

        # get the pIDs
        pIDs = collect(0:(N_walkers-1))
    end

    # get number of MPI processes
    N_mpi = MPI.Comm_size(comm)
    @assert N_mpi = length(pIDs)

    # get mpi ID
    mpiID = MPI.Comm_rank(comm)

    # get corresponding pID
    pID = pIDs[mpiID+1]

    # calculate bin intervals
    bin_intervals = get_bin_intervals(folder, N_bins, pID)

    # load model summary parameters
    β, Δτ, Lτ, model_geometry = load_model_summary(folder)

    # number of sites in lattice
    N_sites = nsites(model_geometry.unit_cell, model_geometry.lattice)

    # process the global measurements
    _process_global_measurements(comm, folder, bin_intervals, pID, β, N_sites)

    return nothing
end

# process global measurements over multiple walkers
function _process_global_measurements(
    folder::String,
    N_bins::Int,
    pIDs::Vector{Int},
    β::T,
    N_sites::Int
) where {T<:AbstractFloat}

    # calculate bin intervals
    bin_intervals = get_bin_intervals(folder, N_bins, pIDs[1])

    # read in binned global measurements for first pID
    binned_gloabl_measurements = read_global_measurements(folder, pIDs[1], bin_intervals)

    # calculate global measurements stats
    global_measurements_avg, global_measurements_std = analyze_global_measurements(binned_gloabl_measurements, β, N_sites)

    # convert standard deviations to variance
    global_measurements_var = global_measurements_std
    for key in keys(global_measurements_std)
        global_measurements_var[key] = abs2(global_measurements_std[key])
    end

    # iterate over remaining MPI walker results
    for pID in pIDs[2:end]

        # read in binned global measurements for current pID
        binned_gloabl_measurements = read_global_measurements(folder, pID, bin_intervals)

        # calculate global measurements stats
        walker_global_measurements_avg, walker_global_measurements_std = analyze_global_measurements(binned_gloabl_measurements, β, N_sites)

        # iterate over measurements
        for key in keys(global_measurements_avg)

            # record measurement average
            global_measurements_avg[key] += walker_global_measurements_avg[key]

            # record measurement variance
            global_measurements_var[key] += abs2(walker_global_measurements_std[key])
        end
    end

    # calculate average for each measurement across all walkers and final standard deviation with errors properly propagated
    for key in keys(global_measurements_avg)
        global_measurements_avg[key] /= length(pIDs)
        global_measurements_std[key]  = sqrt(global_measurements_var[key]) / length(pIDs)
    end

    # write the final global measurement stat to file
    write_global_measurements(folder, global_measurements_avg, global_measurements_std)

    return nothing
end

# process global measurements using MPI
function _process_global_measurements(
    comm::MPI.Comm,
    folder::String,
    bin_intervals::Vector{UnitRange{Int}},
    pID::Int,
    β::T,
    N_sites::Int
) where {T<:AbstractFloat}

    # get mpi Rank
    mpiID = MPI.Comm_rank(comm)

    # read in binned global measurements for current pID
    binned_gloabl_measurements = read_global_measurements(folder, pID, bin_intervals)

    # calculate global measurements stats
    global_measurements_avg, global_measurements_std = analyze_global_measurements(binned_gloabl_measurements, β, N_sites)

    # convert standard deviations to variances
    global_measurements_var = global_measurements_std
    for key in keys(global_measurements_std)
        global_measurements_var[key] = abs2(global_measurements_std[key])
    end

    # if root process
    if iszero(mpiID)

        # number of MPI ranks
        N_mpi = MPI.Comm_size(comm)

        # iterate over global measurements, reducing the result
        for key in keys(global_measurements_avg)
            global_measurements_avg[key] = MPI.Reduce(global_measurements_avg[key], +, comm)
            global_measurements_var[key] = MPI.Reduce(global_measurements_var[key], +, comm)
        end

        # normalize stats, and convert variance to standard deviation
        for key in keys(global_measurements_avg)
            global_measurements_avg[key] /= N_mpi
            global_measurements_std[key]  = sqrt(global_measurements_var[key])/ N_mpi
        end

        # write the final global measurement stat to file
        write_global_measurements(folder, global_measurements_avg, global_measurements_std)

    # if not root process
    else

        # iterate over global measurements, reducing the result
        for key in keys(global_measurements_avg)
            MPI.Reduce(global_measurements_avg[key], +, comm)
            MPI.Reduce(global_measurements_var[key], +, comm)
        end
    end

    return nothing
end


# process global measurements for single MPI walker
function _process_global_measurements(folder::String, N_bins::Int, pID::Int, β::T, N_sites::Int) where {T<:AbstractFloat}

    # calculate bin intervals
    bin_intervals = get_bin_intervals(folder, N_bins, pID)

    # read in binned global measurements for first pID
    binned_gloabl_measurements = read_global_measurements(folder, pID, bin_intervals)

    # calculate global measurements stats
    global_measurements_avg, global_measurements_std = analyze_global_measurements(binned_gloabl_measurements, β, N_sites)

    # write the final global measurement stat to file
    write_global_measurements(folder, pID, global_measurements_avg, global_measurements_std)

    return nothing
end


# read in and bin the global measurements for single MPI walker given by pID
function read_global_measurements(
    folder::String,
    pID::Int,
    bin_intervals::Vector{UnitRange{Int}}
)

    # directory binary file global measurement data lives in
    global_folder = joinpath(folder, "global")

    # get filename for sample global measurement
    global_measurement_file = @sprintf "%s/bin-1_pID-%d.jld2" global_folder pID

    # read in sample global measurement
    global_measurements = JLD2.load(global_measurement_file)

    # get data type of global measurements
    T = typeof(global_measurements["sgn"])

    # number of bins
    N_bins = length(bin_intervals)

    # size of each bin
    N_binsize = length(bin_intervals[1])

    # initiailize contained for binned global measuremenets
    binned_global_measurements = Dict(key => zeros(T, N_bins) for key in keys(global_measurements))

    # iterate over bins
    for bin in 1:N_bins

        # iterate over each bin element
        for i in bin_intervals[bin]

            # load global measurement
            global_measurements = JLD2.load(@sprintf("%s/bin-%d_pID-%d.jld2", global_folder, i, pID))

            # iterate over global measurements
            for key in keys(global_measurements)

                # record measurement
                binned_global_measurements[key][bin] += global_measurements[key] / N_binsize
            end
        end
    end

    return binned_global_measurements
end


# analyze binned global measurement data for a single MPI walker
function analyze_global_measurements(
    binned_global_measurements::Dict{String, Vector{Complex{T}}},
    β::T, N_sites::Int
) where {T<:AbstractFloat}

    # initialize dictionaries to contain global measurements stats
    global_measurements_avg = Dict{String,Complex{T}}()
    global_measurements_std  = Dict{String,T}()

    # get the binned average sgn
    binned_sgn = binned_global_measurements["sgn"]

    # iterate over measurements
    for key in keys(binned_global_measurements)

        if ( 
               startswith(key, "sgn")
            || startswith(key, "log")
            || startswith(key, "action")
            || isequal(key, "chemical_potential")
        )

            # calculate mean and error of measurement
            global_measurements_avg[key] = mean(binned_global_measurements[key])
            global_measurements_std[key] = std(binned_global_measurements[key]) / sqrt(length(binned_sgn))
        else

            # calculate mean and error of measurement
            avg, error = jackknife(/, binned_global_measurements[key], binned_sgn)
            global_measurements_avg[key] = avg
            global_measurements_std[key] = error
        end
    end

    # calculate the compressibility
    n  = binned_global_measurements["density"]
    N² = binned_global_measurements["Nsqrd"]
    S  = binned_sgn
    κ, Δκ = jackknife((n̄, N̄², S̄) -> (β/N_sites)*(N̄²/S̄ - (N_sites*n̄/S̄)^2), n, N², S)
    global_measurements_avg["compressibility"] = κ
    global_measurements_std["compressibility"] = Δκ

    return global_measurements_avg, global_measurements_std
end


# write global measurement data to file averaged across all walkers
function write_global_measurements(folder::String, global_measurements_avg, global_measurements_std)

    open(joinpath(folder,"global_stats.csv"), "w") do fout
        _write_global_measurements(fout, global_measurements_avg, global_measurements_std)
    end

    return nothing
end

# write global measurement data to file for specific walker specified by pID
function write_global_measurements(folder::String, pID::Int, global_measurements_avg, global_measurements_std)

    open(joinpath(folder,"global_stats_pID-$(pID).csv"), "w") do fout
        _write_global_measurements(fout, global_measurements_avg, global_measurements_std)
    end

    return nothing
end

# lowest-level write global measurement data to file
function _write_global_measurements(
    fout::IO,
    global_measurements_avg::Dict{String,Complex{T}},
    global_measurements_std::Dict{String,T}) where {T<:AbstractFloat}

    # write header to file
    write(fout, "MEASUREMENT MEAN_R MEAN_I STD\n")

    # get all measurements and then sort
    measurements = collect(keys(global_measurements_avg))
    sort!(measurements)

    # iterate over measurement
    for key in measurements

        # write measurement to file
        avg = global_measurements_avg[key]
        err = global_measurements_std[key]
        @printf(fout, "%s %.8f %.8f %.8f\n", key, real(avg), imag(avg), err)
    end
    
    return nothing
end