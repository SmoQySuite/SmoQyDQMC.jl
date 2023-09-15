##############################
## PROCESS ALL MEASUREMENTS ##
##############################

@doc raw"""
    process_measurements(
        folder::String, N_bin::Int, pIDs::Union{Vector{Int},Int} = Int[];
        time_displaced::Bool = false,
    )

    process_measurements(
        comm::MPI.Comm, folder::String, N_bin::Int, pIDs::Union{Vector{Int},Int} = Int[];
        time_displaced::Bool = false,
    )

    process_measurements(
        comm::MPI.Comm, folder::String, N_bin::Int, pIDs::Union{Vector{Int},Int} = Int[];
        time_displaced::Bool = false,
    )

Process the measurements recorded in the simulation directory `folder`, where `N_bin` is the number of bins the data is grouped into for calculating error bars.
Note that this method will over-write an existing correlation stats file if there already is one.
The boolean flag `time_displaced` determines whether or not to calculate error bars for time-displaced correlation measurements,
as this can take a non-negligible amount of time for large system, especially when many simulations were run in parallel.
Note that using `pIDs` argument you can filter which MPI walker to use when calculting the statistics.
"""
function process_measurements(folder::String, N_bin::Int, pIDs::Union{Vector{Int},Int} = Int[]; time_displaced::Bool = false)

    # load model summary parameters
    β, Δτ, Lτ, model_geometry = load_model_summary(folder)

    # number of sites in lattice
    N_site = nsites(model_geometry.unit_cell, model_geometry.lattice)

    # process global measurements
    _process_global_measurements(folder, N_bin, pIDs, β, N_site)

    # process local measurement
    _process_local_measurements(folder, N_bin, pIDs)

    # process correlation measurement
    if time_displaced
        _process_correlation_measurements(folder, N_bin, pIDs, ["equal-time", "time-displaced", "integrated"], ["position", "momentum"], Lτ, model_geometry)
    else
        _process_correlation_measurements(folder, N_bin, pIDs, ["equal-time", "integrated"], ["position", "momentum"], Lτ, model_geometry)
    end

    return nothing
<<<<<<< Updated upstream
=======
end

# write time-displaced correlation stat to file for D=1 dimensional system
function _write_displaced_correlation_stat(fout::IO, index::Int, pair::NTuple{2,Int}, l::Int, n::NTuple{1,Int},
                                           C::Complex{T}, ΔC::T) where {T<:AbstractFloat}

    @printf(fout, "%d %d %d %d %d %.8f %.8f %.8f\n", index, pair[2], pair[1], l-1, n[1]-1, real(C), imag(C), ΔC)

    return nothing
end

# write time-displaced correlation stat to file for D=2 dimensional system
function _write_displaced_correlation_stat(fout::IO, index::Int, pair::NTuple{2,Int}, l::Int, n::NTuple{2,Int},
                                           C::Complex{T}, ΔC::T) where {T<:AbstractFloat}

    @printf(fout, "%d %d %d %d %d %d %.8f %.8f %.8f\n", index, pair[2], pair[1], l-1, n[2]-1, n[1]-1, real(C), imag(C), ΔC)

    return nothing
end

# write time-displaced correlation stat to file for D=3 dimensional system
function _write_displaced_correlation_stat(fout::IO, index::Int, pair::NTuple{2,Int}, l::Int, n::NTuple{3,Int},
                                           C::Complex{T}, ΔC::T) where {T<:AbstractFloat}

    @printf(fout, "%d %d %d %d %d %d %d %.8f %.8f %.8f\n", index, pair[2], pair[1], l-1, n[3]-1, n[2]-1, n[1]-1, real(C), imag(C), ΔC)

    return nothing
end

# same as above, but parallelizes data processing with MPI
function process_measurements(comm::MPI.Comm, folder::String, N_bin::Int, pIDs::Union{Vector{Int},Int} = Int[]; time_displaced::Bool = false)

    # load model summary parameters
    β, Δτ, Lτ, model_geometry = load_model_summary(folder)

    # number of sites in lattice
    N_site = nsites(model_geometry.unit_cell, model_geometry.lattice)

    # process global measurements
    _process_global_measurements(folder, N_bin, pIDs, β, N_site)

    # process local measurement
    _process_local_measurements(folder, N_bin, pIDs)

    # process correlation measurement
    if time_displaced
        _process_correlation_measurements(comm, folder, N_bin, pIDs, ["equal-time", "time-displaced", "integrated"], ["position", "momentum"], Lτ, model_geometry)
    else
        _process_correlation_measurements(comm, folder, N_bin, pIDs, ["equal-time", "integrated"], ["position", "momentum"], Lτ, model_geometry)
    end

    return nothing
<<<<<<< Updated upstream
>>>>>>> Stashed changes
=======
end

# write time-displaced correlation stat to file for D=1 dimensional system
function _write_displaced_correlation_stat(fout::IO, index::Int, pair::NTuple{2,Int}, l::Int, n::NTuple{1,Int},
                                           C::Complex{T}, ΔC::T) where {T<:AbstractFloat}

    @printf(fout, "%d %d %d %d %d %.8f %.8f %.8f\n", index, pair[2], pair[1], l-1, n[1]-1, real(C), imag(C), ΔC)

    return nothing
end

# write time-displaced correlation stat to file for D=2 dimensional system
function _write_displaced_correlation_stat(fout::IO, index::Int, pair::NTuple{2,Int}, l::Int, n::NTuple{2,Int},
                                           C::Complex{T}, ΔC::T) where {T<:AbstractFloat}

    @printf(fout, "%d %d %d %d %d %d %.8f %.8f %.8f\n", index, pair[2], pair[1], l-1, n[2]-1, n[1]-1, real(C), imag(C), ΔC)

    return nothing
end

# write time-displaced correlation stat to file for D=3 dimensional system
function _write_displaced_correlation_stat(fout::IO, index::Int, pair::NTuple{2,Int}, l::Int, n::NTuple{3,Int},
                                           C::Complex{T}, ΔC::T) where {T<:AbstractFloat}

    @printf(fout, "%d %d %d %d %d %d %d %.8f %.8f %.8f\n", index, pair[2], pair[1], l-1, n[3]-1, n[2]-1, n[1]-1, real(C), imag(C), ΔC)

    return nothing
end

# same as above, but parallelizes data processing with MPI
function process_measurements(comm::MPI.Comm, folder::String, N_bin::Int, pIDs::Union{Vector{Int},Int} = Int[]; time_displaced::Bool = false)

    # load model summary parameters
    β, Δτ, Lτ, model_geometry = load_model_summary(folder)

    # number of sites in lattice
    N_site = nsites(model_geometry.unit_cell, model_geometry.lattice)

    # process global measurements
    _process_global_measurements(folder, N_bin, pIDs, β, N_site)

    # process local measurement
    _process_local_measurements(folder, N_bin, pIDs)

    # process correlation measurement
    if time_displaced
        _process_correlation_measurements(comm, folder, N_bin, pIDs, ["equal-time", "time-displaced", "integrated"], ["position", "momentum"], Lτ, model_geometry)
    else
        _process_correlation_measurements(comm, folder, N_bin, pIDs, ["equal-time", "integrated"], ["position", "momentum"], Lτ, model_geometry)
    end

    return nothing
>>>>>>> Stashed changes
end