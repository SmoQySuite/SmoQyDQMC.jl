@doc raw"""
    SimulationInfo

Contains identification information about simulation, including the location data is written to,
the simulation ID, and MPI process ID, and whether this simulation started a new simulation or resumed
a previous simulation.

# Fields

- `filepath::String`: File path to where data folder lives.
- `datafolder_prefix`: Prefix for the data folder name.
- `datafolder_name::String`: The data folder name, given by `$(datafolder_prefix)_$(sID)`.
- `datafolder::String`: The data folder, including filepath, given by `joinpath(filepath, datafolder_name)`.
- `pID::Int`: MPI process ID, defaults to 0 if MPI not being used.
- `sID::Int`: Simulation ID.
- `write_bins_concurrent::Bool`: Whether binned data will be written to HDF5 during the simulation or held in memory until the end of the simulation.
- `bin_files::Vector{Vector{UInt8}}`: Represents the HDF5 files containing the binned data.
- `resuming::Bool`: Whether current simulation is resuming a previous simulation (`true`) or starting a new one (`false`).
- `smoqy_version::VersionNumber`: Version of [SmoQyDQMC.jl](https://github.com/SmoQySuite/SmoQyDQMC.jl) used in simulation.

# Notes

If `write_bins_concurrent = true`, then the elements of `bin_files` correspond to the HDF5 bin filenames, assuming the vector elements are converted to strings.
If `write_bins_concurrent = false`, then the elements of the `bin_files` correspond to a byte vector representation of a HDF5 file containing the binned data.
For small simulations that run very fast setting `write_bins_concurrent = false` can make sense, as it significantly reduces the frequency of file IO during the simulation.
This can cause issues on some clusters with respect to overtaxing the cluster network if data is being written to file too frequently during the simulation.
However, for most larger simulations it is advisable to set `write_bins_concurrent = true` as this significantly reduces the memory footprint of the simulation,
particularly when making time-displaced correlation measurements. Also, setting `write_bins_concurrent = false` dramatically increases the size of the checkpoint
files if checkpointing is occurring during the simulation, as the checkpoint files now need to contain all the binned data collected during the simulation.
"""
mutable struct SimulationInfo

    # filepath to where data directory will live
    filepath::String

    # prefix of data directory name
    datafolder_prefix::String

    # data directory name
    datafolder_name::String

    # data directory including filepath
    datafolder::String

    # process ID number (for MPI)
    pID::Int

    # simulation ID number
    sID::Int

    # if binned data will be held in memory or written to file during the simulation
    write_bins_concurrent::Bool

    # HDF5 bin files
    bin_files::Vector{Vector{UInt8}}

    # whether previous simulation is being resumed or a new one is begininning
    resuming::Bool

    # version of package (= SMOQYDQMC_VERSION)
    smoqy_version::VersionNumber
end

@doc raw"""
    SimulationInfo(;
        # KEYWORD ARGUMENTS
        datafolder_prefix::String,
        filepath::String = ".",
        write_bins_concurrent::Bool = true,
        sID::Int=0,
        pID::Int=0
    )

Initialize and return in instance of the type [`SimulationInfo`](@ref).
"""
function SimulationInfo(;
    # KEYWORD ARGUMENTS
    datafolder_prefix::String,
    filepath::String = ".",
    write_bins_concurrent::Bool = true,
    sID::Int=0,
    pID::Int=0
)

    # initialize data folder names
    datafolder_name = @sprintf "%s-%d" datafolder_prefix sID
    datafolder = joinpath(filepath, datafolder_name)
    complete_datafolder_name = @sprintf "complete_%s-%d" datafolder_prefix sID
    complete_datafolder = joinpath(filepath, complete_datafolder_name)

    # if null data folder id given, determine data name and id
    if sID==0
        while isdir(datafolder) || isdir(complete_datafolder) || sID==0
            sID += 1
            datafolder_name = @sprintf "%s-%d" datafolder_prefix sID
            datafolder = joinpath(filepath, datafolder_name)
            complete_datafolder_name = @sprintf "complete_%s-%d" datafolder_prefix sID
            complete_datafolder = joinpath(filepath, complete_datafolder_name)
        end
    end

    # if directory already exists then must be resuming simulation
    resuming = isdir(datafolder)

    # initialize bin files
    bin_files = Vector{UInt8}[]

    return SimulationInfo(filepath, datafolder_prefix, datafolder_name, datafolder, pID, sID, write_bins_concurrent, bin_files, resuming, SMOQYDQMC_VERSION)
end

# print struct info as TOML format
function Base.show(io::IO, ::MIME"text/plain", sim_info::SimulationInfo)

    @printf io "[simulation_info]\n\n"
    @printf io "name = \"%s\"\n" sim_info.datafolder_prefix
    @printf io "sID = %d\n" sim_info.sID
    @printf io "pID = %d\n" sim_info.pID
    @printf io "smoqydqmc_version = \"%s\"\n" sim_info.smoqy_version
    if isdefined(Main, :SmoQyElPhQMC)
        @printf io "smoqyelphqmc_version = \"%s\"\n" Main.SmoQyElPhQMC.SMOQYELPHQMC_VERSION
    end
    @printf io "julia_version = \"%s\"" VERSION

    return nothing
end


@doc raw"""
    save_simulation_info(
        sim_info::SimulationInfo,
        metadata = nothing,
        filename = @sprintf "simulation_info_sID-%d_pID-%d.toml" sim_info.sID sim_info.pID
    )

Save the contents `sim_info` to a TOML file, and add an optional additional table to the
output file based on the contents of a dictionary `metadata`.
"""
function save_simulation_info(
    sim_info::SimulationInfo,
    metadata = nothing,
    filename = @sprintf "simulation_info_sID-%d_pID-%d.toml" sim_info.sID sim_info.pID
)

    (; datafolder ) = sim_info
    open(joinpath(datafolder, filename), "w") do fout
        show(fout, "text/plain", sim_info)
        if !isnothing(metadata)
            @printf fout "\n\n"
            TOML.print(fout, Dict("metadata" => metadata), sorted = true)
        end
    end

    return nothing
end


@doc raw"""
    initialize_datafolder(comm::MPI.Comm, sim_info::SimulationInfo)

    initialize_datafolder(sim_info::SimulationInfo)

Initialize `sim_info.datafolder` directory if it does not already exist.
If `comm::MPI.Comm` is passed as the first argument, this this function will synchronize
all the MPI processes, ensuring that none proceed beyond this function call until
the data folder that results will be written to is successfully initialized.
"""
function initialize_datafolder(comm::MPI.Comm, sim_info::SimulationInfo)

    MPI.Barrier(comm)
    initialize_datafolder(sim_info)

    return nothing
end

function initialize_datafolder(sim_info::SimulationInfo)

    (; pID, datafolder, write_bins_concurrent) = sim_info

    # make subdirectory for binned data to be written to
    bin_dir = mkpath(joinpath(datafolder, "bins"))
    if write_bins_concurrent
        mkpath(joinpath(bin_dir, "pID-$(pID)"))
    end

    return nothing
end