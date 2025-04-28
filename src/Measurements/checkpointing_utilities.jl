@doc raw"""
    read_jld2_checkpoint(
        simulation_info::SimulationInfo
    )

Read in a checkpoint file written using the [`write_jld2_checkpoint`](@ref) function
and return its contents as a dictionary. Behind the scenes, the
[JLD2.jl](https://github.com/JuliaIO/JLD2.jl.git) package is used to read (and write)
the checkpoint files.
"""
function read_jld2_checkpoint(
    simulation_info::SimulationInfo
)

    # get datafolder name
    datafolder = simulation_info.datafolder

    # get MPI process ID
    pID = simulation_info.pID

    # flag to indicate whether checkpoint file was loaded successfully
    checkpoint_loaded = false

    # initialize checkpoint to nothing
    checkpoint = nothing

    # attempt to load checkpoint file
    if (checkpoint_fn = joinpath(datafolder, "checkpoint_pID-$(pID).jld2")) |> isfile

        # attempt to load checkpoint file
        try
            # read checkpoint file
            checkpoint = JLD2.load(checkpoint_fn)
            # mark that the checkpoint has been loaded
            checkpoint_loaded = true
        catch
            nothing
        end
    end

    # attempt to load new checkpoint file
    if (!checkpoint_loaded) && ((checkpoint_fn = joinpath(datafolder, "checkpoint_new_pID-$(pID).jld2")) |> isfile)

        # attempt to load checkpoint file
        try
            # read checkpoint file
            checkpoint = JLD2.load(checkpoint_fn)
            # mark that the checkpoint has been loaded
            checkpoint_loaded = true
        catch
            nothing
        end
    end

    # attempt to load old checkpoint file
    if (!checkpoint_loaded) && ((checkpoint_fn = joinpath(datafolder, "checkpoint_old_pID-$(pID).jld2")) |> isfile)

        # attempt to load checkpoint file
        try
            # read checkpoint file
            checkpoint = JLD2.load(checkpoint_fn)
            # mark that the checkpoint has been loaded
            checkpoint_loaded = true
        catch
            nothing
        end
    end

    # if checkpoint was not loaded
    if !checkpoint_loaded

        # then throw an error
        error("No JLD2 checkpoint file was successfully loaded from $(datafolder) for process ID (pID) $(pID).")
    end

    return checkpoint
end


@doc raw"""
    write_jld2_checkpoint(
        # Arguments
        comm::MPI.Comm,
        simulation_info::SimulationInfo;
        # Keyword Arguments
        checkpoint_timestamp::T = 0.0,
        checkpoint_freq::T = 0.0,
        start_timestamp::T = 0.0,
        runtime_limit::T = Inf,
        error_code::Int = 13,
        # Abitrary Keyword Arguments Written to Checkpoint
        kwargs...
    ) where {T<:AbstractFloat}

    write_jld2_checkpoint(
        # Arguments
        simulation_info::SimulationInfo;
        # Keyword Arguments
        checkpoint_timestamp::T = 0.0,
        checkpoint_freq::T = 0.0,
        start_timestamp::T = 0.0,
        runtime_limit::T = Inf,
        error_code::Int = 13,
        # Abitrary Keyword Arguments Written to Checkpoint
        kwargs...
    ) where {T<:AbstractFloat}

Checkpoint a simulation by writing a new checkpoint file if necessary
The checkpoint file is a [JLD2](https://github.com/JuliaIO/JLD2.jl) binary file.

# Arguments

- `comm::MPI.Comm`: (optional) MPI communicator object used to synchronize processes. Ensures all MPI processes remain syncrhonized.
- `simulation_info::SimulationInfo`: Contains datafolder and MPI process ID information.

# Keyword Arguments

- `checkpoint_timestamp::T = 0.0`: (optional) Epoch timestap of previously written checkpoint file.
- `checkpoint_freq::T = 0.0`: (optional) Frequency with with checkpoint files are written; new checkpoint is written only if this many seconds has elapsed since previous checkpoint.
- `start_timestamp::T = 0.0`: (optional) Epoch timestamp of the start time of the simulation.
- `runtime_limit::T = Inf`: (optional) Maximum runtime of simulation in seconds; if this much time has elapsed since the beginning of the simulation after writing new checkpoint file then the simulation is terminated.
- `error_code::Int = 13`: (optional) Error code used to exit simulation if the runtime limit is exceeded.
- `kwargs...`: Additional keyword arguments containing the information that will stored in the checkpoint file; keyword arguments can point to arbitrary Julia objects.
"""
function write_jld2_checkpoint(
    # Arguments
    comm::MPI.Comm,
    simulation_info::SimulationInfo;
    # Keyword Arguments
    checkpoint_timestamp::T = 0.0,
    checkpoint_freq::T = 0.0,
    start_timestamp::T = 0.0,
    runtime_limit::T = Inf,
    error_code::Int = 13,
    # Abitrary Keyword Arguments Written to Checkpoint
    kwargs...
) where {T<:AbstractFloat}

    # write JLD2 checkpoint file
    checkpoint_timestamp = _write_jld2_checkpoint(
        simulation_info, checkpoint_timestamp, checkpoint_freq;
        kwargs...
    )

    # if time limit for simulation has been exceeded
    if (checkpoint_timestamp - start_timestamp) ≥ runtime_limit

        # syncrhonize MPI processes before exiting simulation
        MPI.Barrier(comm)

        # exit simulation
        exit(error_code)
    end

    return checkpoint_timestamp
end

function write_jld2_checkpoint(
    # Arguments
    simulation_info::SimulationInfo;
    # Keyword Arguments
    checkpoint_timestamp::T = 0.0,
    checkpoint_freq::T = 0.0,
    start_timestamp::T = 0.0,
    runtime_limit::T = Inf,
    error_code::Int = 13,
    # Abitrary Keyword Arguments Written to Checkpoint
    kwargs...
) where {T<:AbstractFloat}

    # write JLD2 checkpoint file
    checkpoint_timestamp = _write_jld2_checkpoint(
        simulation_info, checkpoint_timestamp, checkpoint_freq;
        kwargs...
    )

    # if time limit for simulation has been exceeded
    if (checkpoint_timestamp - start_timestamp) ≥ runtime_limit

        # exit simulation
        exit(error_code)
    end

    return checkpoint_timestamp
end


function _write_jld2_checkpoint(
    # Arguments
    simulation_info::SimulationInfo,
    previous_checkpoint_timestamp::T,
    checkpoint_freq::T;
    # Arbitrary Keyword Arguments
    kwargs...
) where {T<:AbstractFloat}

    # get current timestamp
    timestamp = time()

    # if sufficient time has elapsed since previous checkpoint
    if (timestamp - previous_checkpoint_timestamp) ≥ checkpoint_freq

        # get datafolder name
        datafolder = simulation_info.datafolder

        # get MPI process ID
        pID = simulation_info.pID

        # construct checkpoint filenames
        checkpoint_fn = joinpath(datafolder, "checkpoint_pID-$(pID).jld2")

        # if current checkpoint file exists, make it the old one
        if isfile(checkpoint_fn)
            # define old and new checkpoint filenames
            checkpoint_fn_old = joinpath(datafolder, "checkpoint_old_pID-$(pID).jld2")
            checkpoint_fn_new = joinpath(datafolder, "checkpoint_new_pID-$(pID).jld2")
            # save new checkpoint file
            jldsave(checkpoint_fn_new; kwargs...)
            # move current checkpoint to old checkpoint
            mv(checkpoint_fn, checkpoint_fn_old, force = true)
            # make new checkpoint file the current one
            mv(checkpoint_fn_new, checkpoint_fn, force = true)
            # delete old checkpoint file
            rm(checkpoint_fn_old, force = true)
        # if no checkpoint file exists then create new one
        else
            # save new checkpoint file
            jldsave(checkpoint_fn; kwargs...)
        end

        # create new checkpoint timestamp
        checkpoint_timestamp = timestamp
    
    # if no checkpoint is written as not enough time has elapsed
    else
        
        # retain old checkpoint timestamp
        checkpoint_timestamp = previous_checkpoint_timestamp
    end

    return checkpoint_timestamp
end


@doc raw"""
    rename_complete_simulation(
        # Arguments
        comm::MPI.Comm,
        simulation_info::SimulationInfo;
        # Keyword Arguments
        delete_jld2_checkpoints::Bool = true
    )

    rename_complete_simulation(
        # Arguments
        simulation_info::SimulationInfo;
        # Keyword Arguments
        delete_jld2_checkpoints::Bool = true
    )

When a simulation is complete, this function renames the data folder the results
were written to such that the directory name now begins with `"complete_"`, making it
simpler to identify which simulations no longer need to be resumed if checkpointing
is being used. This function also deletes the any checkpoint files written using
the [`write_jld2_checkpoint`](@ref) function if `delect_checkpoints = true`.
"""
function rename_complete_simulation(
    # Arguments
    comm::MPI.Comm,
    simulation_info::SimulationInfo;
    # Keyword Arguments
    delete_jld2_checkpoints::Bool = true
)

    (; datafolder, filepath, datafolder_prefix, sID, pID) = simulation_info

    # rename data folder
    complete_datafolder_prefix = "complete_" * datafolder_prefix

    # initialize new simulation info object
    simulation_info_complete = SimulationInfo(
        datafolder_prefix = complete_datafolder_prefix,
        filepath = filepath,
        sID = sID,
        pID = pID
    )

    # synchronize MPI processes
    MPI.Barrier(comm)

    # have pID = 0 process rename data folder
    if iszero(pID)
        mv(datafolder, simulation_info_complete.datafolder, force = true)
    end

    # synchronize MPI processes
    MPI.Barrier(comm)

    # if deleting checkpoing files
    if delete_jld2_checkpoints

        # get the files in the data folder
        files = readdir(simulation_info.datafolder, join = true)

        # get ending of checkpoint file name based on pID
        ending = "pID-$(pID).jld2"

        # iterate over files, deleting the checkpoint filed
        for file in files
            if startswith(file, "checkpoint") && endswith(file, ending)
                # delete checkpoint file
                rm(file, force = true)
            end
        end
    end

    return simulation_info_complete
end

function rename_complete_simulation(
    # Arguments
    simulation_info::SimulationInfo;
    # Keyword Arguments
    delete_jld2_checkpoints::Bool = true
)

    (; datafolder, filepath, datafolder_prefix, sID, pID) = simulation_info

    # rename data folder
    complete_datafolder_prefix = "complete_" * datafolder_prefix

    # initialize new simulation info object
    simulation_info_complete = SimulationInfo(
        datafolder_prefix = complete_datafolder_prefix,
        filepath = filepath,
        sID = sID,
        pID = pID
    )

    # rename data folder
    mv(datafolder, simulation_info_complete.datafolder, force = true)

    # if deleteing checkpoing files
    if delete_jld2_checkpoints

        # get the files in the data folder
        files = readdir(simulation_info.datafolder, join = true)

        # iterate over files, deleting the checkpoint filed
        for file in files
            if startswith(file, "checkpoint")
                # delete checkpoint file
                rm(file, force = true)
            end
        end
    end

    return simulation_info_complete
end
