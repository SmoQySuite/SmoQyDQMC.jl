##############################
## PROCESS ALL MEASUREMENTS ##
##############################

@doc raw"""
    process_measurements(
        # ARGUMENTS
        folder::String,
        N_bins::Int,
        pIDs::Union{Vector{Int},Int} = Int[];
        # KEYWORD ARGUMENTS
        time_displaced::Bool = false
    )

    process_measurements(
        # ARGUMENTS
        comm::MPI.Comm,
        folder::String,
        N_bins::Int,
        pIDs::Union{Vector{Int},Int} = Int[];
        # KEYWORD ARGUMENTS
        time_displaced::Bool = false
    )

Process the measurements recorded in the simulation directory `folder`, where `N_bins` is the number of bins the data is grouped into for calculating error bars.
Note that this method will over-write an existing correlation stats file if there already is one.
The boolean flag `time_displaced` determines whether or not to calculate error bars for time-displaced correlation measurements,
as this can take a non-negligible amount of time for large system, especially when many simulations were run in parallel.
Note that using `pIDs` argument you can filter which MPI walker to use when calculting the statistics.
"""
function process_measurements(
    # ARGUMENTS
    folder::String,
    N_bins::Int,
    pIDs::Union{Vector{Int},Int} = Int[];
    # KEYWORD ARGUMENTS
    time_displaced::Bool = false
)

    # set the walkers to iterate over
    if isempty(pIDs)

        # get the number of MPI walkers
        N_walkers = get_num_walkers(folder)

        # get the pIDs
        pIDs = collect(0:(N_walkers-1))
    end

    # load model summary parameters
    β, Δτ, Lτ, model_geometry = load_model_summary(folder)

    # number of sites in lattice
    N_sites = nsites(model_geometry.unit_cell, model_geometry.lattice)

    # process global measurements
    _process_global_measurements(folder, N_bins, pIDs, β, N_sites)

    # process local measurement
    _process_local_measurements(folder, N_bins, pIDs)

    # process correlation measurement
    if time_displaced
        _process_correlation_measurements(folder, N_bins, pIDs, ["equal-time", "time-displaced", "integrated"], ["position", "momentum"], Lτ, model_geometry)
    else
        _process_correlation_measurements(folder, N_bins, pIDs, ["equal-time", "integrated"], ["position", "momentum"], Lτ, model_geometry)
    end

    return nothing
end

# same as above, but parallelizes data processing with MPI
function process_measurements(
    # ARGUMENTS
    comm::MPI.Comm,
    folder::String,
    N_bins::Int,
    pIDs::Union{Vector{Int},Int} = Int[];
    # KEYWORD ARGUMENTS
    time_displaced::Bool = false
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
    @assert N_mpi == length(pIDs)

    # get mpi ID
    mpiID = MPI.Comm_rank(comm)

    # get corresponding pID
    pID = pIDs[mpiID+1]

    # load model summary parameters
    β, Δτ, Lτ, model_geometry = load_model_summary(folder)

    # number of sites in lattice
    N_sites = nsites(model_geometry.unit_cell, model_geometry.lattice)

    # calculate bin intervals
    bin_intervals = get_bin_intervals(folder, N_bins, pID)

    # get binned sign
    binned_sign = get_average_sign(folder, bin_intervals, pID)

    # process global measurements
    _process_global_measurements(comm, folder, bin_intervals, pID, β, N_sites)

    # process local measurement
    _process_local_measurements(comm, folder, bin_intervals, binned_sign, pID)

    # process correlation measurement
    if time_displaced
        _process_correlation_measurements(
            comm, folder, pID,
            ["equal-time", "time-displaced", "integrated"], ["position", "momentum"],
            Lτ, model_geometry, bin_intervals, binned_sign
        )
    else
        _process_correlation_measurements(
            comm, folder, pID,
            ["equal-time", "integrated"], ["position", "momentum"],
            Lτ, model_geometry, bin_intervals, binned_sign
        )
    end

    return nothing
end