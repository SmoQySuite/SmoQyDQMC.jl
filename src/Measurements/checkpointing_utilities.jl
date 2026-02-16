@doc raw"""
    read_jld2_checkpoint(
        simulation_info::SimulationInfo
    )

Read in a checkpoint file written using the [`write_jld2_checkpoint`](@ref) function and return its contents as a dictionary.
This function returns the tuple `(checkpoint, checkpoint_timestamp)` where `checkpoint` is a dictionary containing the contents of the checkpoint file
and `checkpoint_timestamp` is the epoch timestamp corresponding to when the checkpoint file was read in.
Behind the scenes, the [JLD2.jl](https://github.com/JuliaIO/JLD2.jl.git) package is used to read (and write)
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

    # record the checkpoint timestamp as the current time
    checkpoint_timestamp = time()

    # get the bin files and copy the information over to the simulation info struct
    simulation_info.bin_files = checkpoint["bin_files"]

    # re-calculate the forward FFT plan and record and add it to the measurement container.
    (; a, L) = checkpoint["measurement_container"]
    pfft! = plan_fft!(zeros(eltype(a), L...); flags=FFTW.PATIENT)
    checkpoint["measurement_container"] = (; pfft! = pfft!, checkpoint["measurement_container"]...)

    return checkpoint, checkpoint_timestamp
end


@doc raw"""
    write_jld2_checkpoint(
        # ARGUMENTS
        comm::MPI.Comm,
        simulation_info::SimulationInfo;
        # REQUIRED KEYWORD ARGUMENTS
        model_geometry::ModelGeometry,
        measurement_container::NamedTuple,
        # OPTIONAL KEYWORD ARGUMENTS
        checkpoint_timestamp::T = 0.0,
        checkpoint_freq::T = 0.0,
        start_timestamp::T = 0.0,
        runtime_limit::T = Inf,
        error_code::Int = 13,
        # ARBITRARY KEYWORD ARGUMENTS WRITTEN TO CHECKPOINT
        kwargs...
    ) where {T<:AbstractFloat}

    write_jld2_checkpoint(
        # ARGUMENTS
        simulation_info::SimulationInfo;
        # REQUIRED KEYWORD ARGUMENTS
        model_geometry::ModelGeometry,
        measurement_container::NamedTuple,
        # OPTIONAL KEYWORD ARGUMENTS
        checkpoint_timestamp::T = 0.0,
        checkpoint_freq::T = 0.0,
        start_timestamp::T = 0.0,
        runtime_limit::T = Inf,
        error_code::Int = 13,
        # ARBITRARY KEYWORD ARGUMENTS WRITTEN TO CHECKPOINT
        kwargs...
    ) where {T<:AbstractFloat}

Checkpoint a simulation by writing a new checkpoint file if necessary
The checkpoint file is a [JLD2](https://github.com/JuliaIO/JLD2.jl) binary file.

# Arguments

- `comm::MPI.Comm`: (optional) MPI communicator object used to synchronize processes. Ensures all MPI processes remain synchronized.
- `simulation_info::SimulationInfo`: Contains datafolder and MPI process ID information.

# Keyword Arguments

- `checkpoint_timestamp::T = 0.0`: (optional) Epoch timestamp of previously written checkpoint file.
- `checkpoint_freq::T = 0.0`: (optional) Frequency with with checkpoint files are written; new checkpoint is written only if this many seconds has elapsed since previous checkpoint.
- `start_timestamp::T = 0.0`: (optional) Epoch timestamp of the start time of the simulation.
- `runtime_limit::T = Inf`: (optional) Maximum runtime for simulation in seconds; if after writing a new checkpoint file the next checkpoint file that would be written in the future exceeds the runtime limit then exit the simulation.
- `error_code::Int = 13`: (optional) Error code used to exit simulation if the runtime limit is exceeded.
- `kwargs...`: Additional keyword arguments containing the information that will stored in the checkpoint file; keyword arguments can point to arbitrary Julia objects.

# Notes

The default values for the `checkpoint_timestamp`, `checkpoint_freq`, `start_timestamp`, and `runtime_limit` keyword arguments
result in there being no runtime limit for the simulation and a new checkpoint file being written every time this function is called.
"""
function write_jld2_checkpoint(
    # ARGUMENTS
    comm::MPI.Comm,
    simulation_info::SimulationInfo;
    # REQUIRED KEYWORD ARGUMENTS
    model_geometry::ModelGeometry,
    measurement_container::NamedTuple,
    # OPTIONAL KEYWORD ARGUMENTS
    checkpoint_timestamp::T = 0.0,
    checkpoint_freq::T = 0.0,
    start_timestamp::T = 0.0,
    runtime_limit::T = Inf,
    error_code::Int = 13,
    # ARBITRARY KEYWORD ARGUMENTS WRITTEN TO CHECKPOINT
    kwargs...
) where {T<:AbstractFloat}

    # write JLD2 checkpoint file
    checkpoint_timestamp = _write_jld2_checkpoint(
        simulation_info, checkpoint_timestamp, checkpoint_freq;
        model_geometry = model_geometry,
        measurement_container = measurement_container,
        kwargs...
    )

    # if time limit for simulation is exceeded for next future checkpoint
    if (checkpoint_timestamp + checkpoint_freq - start_timestamp) ≥ runtime_limit

        # synchronize MPI processes before exiting simulation
        MPI.Barrier(comm)

        # finalize MPI
        MPI.Finalize()

        # exit the simulation with given specified error code
        exit(error_code)
    end

    return checkpoint_timestamp
end

function write_jld2_checkpoint(
    # ARGUMENTS
    simulation_info::SimulationInfo;
    # REQUIRED KEYWORD ARGUMENTS
    model_geometry::ModelGeometry,
    measurement_container::NamedTuple,
    # OPTIONAL KEYWORD ARGUMENTS
    checkpoint_timestamp::T = 0.0,
    checkpoint_freq::T = 0.0,
    start_timestamp::T = 0.0,
    runtime_limit::T = Inf,
    error_code::Int = 13,
    # ARBITRARY KEYWORD ARGUMENTS WRITTEN TO CHECKPOINT
    kwargs...
) where {T<:AbstractFloat}

    # write JLD2 checkpoint file
    checkpoint_timestamp = _write_jld2_checkpoint(
        simulation_info, checkpoint_timestamp, checkpoint_freq;
        model_geometry = model_geometry,
        measurement_container = measurement_container,
        kwargs...
    )

    # if time limit for simulation is exceeded for next future checkpoint
    if (checkpoint_timestamp + checkpoint_freq - start_timestamp) ≥ runtime_limit

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
    # Required Keyword Arguments
    model_geometry::ModelGeometry,
    measurement_container::NamedTuple,
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

        # get the bin files stored in the simulation_info struct
        bin_files = simulation_info.bin_files

        # if current checkpoint file exists, make it the old one
        if isfile(checkpoint_fn)
            # define old and new checkpoint filenames
            checkpoint_fn_old = joinpath(datafolder, "checkpoint_old_pID-$(pID).jld2")
            checkpoint_fn_new = joinpath(datafolder, "checkpoint_new_pID-$(pID).jld2")
            # save new checkpoint file
            jldsave(
                checkpoint_fn_new;
                bin_files, model_geometry,
                # Note that the FFT plan stored inside the measurement_container cannot
                # be written to file as it is essentially just C-pointers that are invalid
                # when read back in and will result in seg faults. Therefore, I am filtering
                # out the FFT plan `pfft!` field here.
                measurement_container = (;
                    (k => v for (k,v) in pairs(measurement_container) if k != :pfft!)...
                ), kwargs...
            )
            # move current checkpoint to old checkpoint
            mv(checkpoint_fn, checkpoint_fn_old, force = true)
            # make new checkpoint file the current one
            mv(checkpoint_fn_new, checkpoint_fn, force = true)
            # delete old checkpoint file
            rm(checkpoint_fn_old, force = true)
        # if no checkpoint file exists then create new one
        else
            # save new checkpoint file
            jldsave(
                checkpoint_fn;
                bin_files, model_geometry,
                # Note that the FFT plan stored inside the measurement_container cannot
                # be written to file as it is essentially just C-pointers that are invalid
                # when read back in and will result in seg faults. Therefore, I am filtering
                # out the FFT plan `pfft!` field here.
                measurement_container = (;
                    (k => v for (k,v) in pairs(measurement_container) if k != :pfft!)...
                ), kwargs...
            )
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
    rm_jld2_checkpoints(
        # ARGUMENTS
        comm::MPI.Comm,
        simulation_info::SimulationInfo
    )

    rm_jld2_checkpoints(
        # ARGUMENTS
        simulation_info::SimulationInfo
    )

Delete the JLD2 checkpoint files.
"""
function rm_jld2_checkpoints(
    simulation_info::SimulationInfo
)

    (; datafolder, pID) = simulation_info

    # get the files in the data folder
    files = readdir(datafolder, join = true)

    # end of checkpoint filename
    ending = @sprintf("_pID-%d.jld2", pID)

    # iterate over files
    for file in files
        # get file base name
        name = basename(file)
        # if a checkpoint file
        if startswith(name, "checkpoint_") && endswith(name, ending)
            # delete the checkpoint file
            rm(file, force = true)
        end
    end

    return nothing
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
the [`write_jld2_checkpoint`](@ref) function if `delete_jld2_checkpoints = true`.
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

    # if deleting checkpointing files
    if delete_jld2_checkpoints

        # delete checkpoint files
        rm_jld2_checkpoints(simulation_info_complete)
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

    # if deleting checkpointing files
    if delete_jld2_checkpoints

        # delete checkpoint files
        rm_jld2_checkpoints(simulation_info_complete)
    end

    return simulation_info_complete
end
