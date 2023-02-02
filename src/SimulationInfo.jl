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
- `resuming::Bool`: Whether current simulation is resuming a previous simulation (`true`) or starting a new one (`false`).
- `smoqy_version::VersionNumber`: Version of [`SmoQyDQMC`](@ref) used in simulation.
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

    # whether previous simulation is being resumed or a new one is begininning
    resuming::Bool

    # version of package (= SMOQYDQMC_VERSION)
    smoqy_version::VersionNumber
end

@doc raw"""
    SimulationInfo(; datafolder_prefix::String, filepath::String = ".", sID::Int=0, pID::Int=0)

Initialize and return in instance of the type [`SimulationInfo`](@ref).
"""
function SimulationInfo(; datafolder_prefix::String, filepath::String = ".", sID::Int=0, pID::Int=0)

    # initialize data folder names
    datafolder_name = @sprintf "%s-%d" datafolder_prefix sID
    datafolder = joinpath(filepath, datafolder_name)

    # if null data folder id given, determine data name and id
    if sID==0
        while isdir(datafolder) || sID==0
            sID += 1
            datafolder_name = @sprintf "%s-%d" datafolder_prefix sID
            datafolder = joinpath(filepath, datafolder_name)
        end
    end

    # if directory already exists then must be resuming simulation
    resuming = isdir(datafolder)

    return SimulationInfo(filepath, datafolder_prefix, datafolder_name, datafolder, pID, sID, resuming, SMOQYDQMC_VERSION)
end

# print struct info as TOML format
function Base.show(io::IO, ::MIME"text/plain", sim_info::SimulationInfo)

    @printf io "[simulation_info]\n\n"
    @printf io "name  = \"%s\"\n" sim_info.datafolder_prefix
    @printf io "sID   = %d\n" sim_info.sID
    @printf io "pID   = %d\n" sim_info.pID
    @printf io "smoqydqmc_version = \"%s\"\n" sim_info.smoqy_version
    @printf io "julia_version     = \"%s\"" VERSION

    return nothing
end


@doc raw"""
    save_simulation_info(sim_info::SimulationInfo, additional_info = nothing)

Save the contents `sim_info` to a TOML file, and add an optional additional table to the
output file based on the contents of a dictionary `additional_info`.
"""
function save_simulation_info(sim_info::SimulationInfo, additional_info = nothing)

    (; datafolder, pID, sID) = sim_info
    fn = @sprintf "simulation_info_pID%d_sID%d.toml" pID sID
    open(joinpath(datafolder, fn), "w") do fout
        show(fout, "text/plain", sim_info)
        if !isnothing(additional_info)
            @printf fout "\n\n"
            TOML.print(fout, Dict("additional_info" => additional_info), sorted = true)
        end
    end

    return nothing
end


@doc raw"""
    initialize_datafolder(sim_info::SimulationInfo)

Initalize `sim_info.datafolder` directory if it does not already exist.
"""
function initialize_datafolder(sim_info::SimulationInfo)

    (; pID, datafolder, resuming) = sim_info

    # if master process and starting new simulation (not resuming an existing simulation)
    if iszero(pID) && !resuming

        # make data folder diretory
        mkdir(datafolder)
    end

    return nothing
end