######################################################
## HIGHEST LEVEL/EXPORT WRITE MEASUREMENTS FUNCTION ##
######################################################

@doc raw"""
    write_measurements!(; measurement_container::NamedTuple,
                        simulation_info::SimulationInfo,
                        model_geometry::ModelGeometry{D, E, N},
                        bin::Int, bin_size::Int) where {E<:AbstractFloat, D, N}

Write the measurements contained in `measurement_container` to file.
Measurements are written to file in a binary format using the [`JLD2.jl`](https://github.com/JuliaIO/JLD2.jl.git) package.

This function also does a few other things:
1. Normalizes all the measurements by the `bin_size` i.e. the number of measurements that were accumlated into the measurement container.
2. Take position space correlation function measurements and fourier transform them to momentum space.
3. Integrate relevant time-displaced correlation function measurements over imaginary time to get the corresponding zero matsubara frequency correlation function.
4. Reset all the measurements in `measurement_container` to zero after the measurements are written to file.
"""
function write_measurements!(; measurement_container::NamedTuple,
                             simulation_info::SimulationInfo,
                             model_geometry::ModelGeometry{D, E, N},
                             bin::Int, bin_size::Int, Δτ::E) where {D, E<:AbstractFloat, N}

    (; datafolder, pID) = simulation_info
    lattice   = model_geometry.lattice::Lattice{D}
    unit_cell = model_geometry.unit_cell::UnitCell{D,E,N}
    bonds     = model_geometry.bonds::Vector{Bond{D}}

    # construct filename
    fn = @sprintf "bin-%d_pID-%d.jld2" bin pID

    # normalize all measurements by the bin size
    normalize_measurements!(measurement_container::NamedTuple, bin_size::Int)

    # write global measurements to file
    global_measurements = measurement_container.global_measurements::Dict{String, Complex{E}}
    JLD2.save(joinpath(datafolder, "global", fn), global_measurements)

    # reset global measurements to zero
    for measurement in keys(global_measurements)
        global_measurements[measurement] = zero(Complex{E})
    end

    # write local measurements to file
    local_measurements = measurement_container.local_measurements::Dict{String, Vector{Complex{E}}}
    JLD2.save(joinpath(datafolder, "local", fn), local_measurements)

    # reset global measurements to zero
    for measurement in keys(local_measurements)
        fill!(local_measurements[measurement], zero(Complex{E}))
    end

    # iterate over equal-time measurements
    equaltime_correlations = measurement_container.equaltime_correlations
    for correlation in keys(equaltime_correlations)

        # get the correlation container
        correlation_container = equaltime_correlations[correlation]
        pairs = correlation_container.bond_id_pairs::Vector{NTuple{2,Int}}
        correlations = correlation_container.correlations::Vector{Array{Complex{E}, D}}

        # write position space equal-time correlation to file
        save(joinpath(datafolder, "equal-time", correlation, "position", fn), correlation_container)

        # fourier transform correlations to momentum space
        for i in eachindex(pairs)
            # get the pair of orbitals associated with the correlation
            pair = pairs[i]
            a = bonds[pair[1]].orbitals[1]
            b = bonds[pair[2]].orbitals[1]
            # perform fourier transform
            fourier_transform!(correlations[i], a, b, unit_cell, lattice)
        end

        # write momentum space equal-time correlation to file
        save(joinpath(datafolder, "equal-time", correlation, "momentum", fn), correlation_container)

        # set the correlations to zero
        reset!(correlation_container)
    end

    # iterate over time-displaced correlations
    time_displaced_correlations = measurement_container.time_displaced_correlations
    integrated_correlations = measurement_container.integrated_correlations
    for correlation in keys(time_displaced_correlations)

        # get the correlation container
        correlation_container = time_displaced_correlations[correlation]
        pairs = correlation_container.bond_id_pairs::Vector{NTuple{2,Int}}
        correlations = correlation_container.correlations::Vector{Array{Complex{E}, D+1}}

        # write position space time-displaced correlation to file
        if correlation_container.time_displaced
            save(joinpath(datafolder, "time-displaced", correlation, "position", fn), correlation_container)
        end

        # get susceptibility container
        susceptibility_container = integrated_correlations[correlation]
        susceptibilities = susceptibility_container.correlations::Vector{Array{Complex{E}, D}}

        # calculate position space susceptibilies/integrated correlations
        for i in eachindex(pairs)
            # calculate susceptibility
            susceptibility!(susceptibilities[i], correlations[i], Δτ, D+1)
        end

        # write position space susceptibility to file
        save(joinpath(datafolder, "integrated", correlation, "position", fn), susceptibility_container)

        # fourier transform correlations to momentum space
        for i in eachindex(pairs)
            # get the pair of orbitals associated with the correlation
            pair = pairs[i]
            a = bonds[pair[1]].orbitals[1]
            b = bonds[pair[2]].orbitals[1]
            # perform fourier transform
            fourier_transform!(correlations[i], a, b, D+1, unit_cell, lattice)
        end

        # write momentum space time-displaced correlation to file
        if correlation_container.time_displaced
            save(joinpath(datafolder, "time-displaced", correlation, "momentum", fn), correlation_container)
        end

        # calculate momentum space susceptibilies/integrated correlations
        for i in eachindex(pairs)
            # calculate susceptibility
            susceptibility!(susceptibilities[i], correlations[i], Δτ, D+1)
        end

        # write momentum space susceptibility to file
        save(joinpath(datafolder, "integrated", correlation, "momentum", fn), susceptibility_container)

        # set the correlations to zero
        reset!(correlation_container)
    end

    return nothing
end


########################################
## NORMALIZE MEASUREMENTS BY BIN SIZE ##
########################################

# normalize measurements by bin size
function normalize_measurements!(measurement_container::NamedTuple, bin_size::Int)

    # normalize global measurements by bin size
    global_measurements = measurement_container.global_measurements
    normalize_global_measurements!(global_measurements, bin_size)

    # normalize local measurements by bin size
    local_measurements = measurement_container.local_measurements
    normalize_local_measurements!(local_measurements, bin_size)

    # normalize equal-time correlation function measurement
    equaltime_correlations = measurement_container.equaltime_correlations
    normalize_correlation_measurements!(equaltime_correlations, bin_size)

    # normalize time-displaced correlation function measurements
    time_displaced_correlations = measurement_container.time_displaced_correlations
    normalize_correlation_measurements!(time_displaced_correlations, bin_size)

    return nothing
end

# normalize global measurements by bin size
function normalize_global_measurements!(global_measurements::Dict{String, Complex{T}}, bin_size::Int) where {T<:AbstractFloat}

    for global_measurement in keys(global_measurements)
        global_measurements[global_measurement] /= bin_size
    end

    return nothing
end

# normalize local measurements by bin size
function normalize_local_measurements!(local_measurements::Dict{String, Vector{Complex{T}}}, bin_size::Int) where {T<:AbstractFloat}

    for local_measurement in keys(local_measurements)
        @. local_measurements[local_measurement] /= bin_size
    end

    return nothing
end

# normalize correlation measurement
function normalize_correlation_measurements!(correlation_measurements::Dict{String, CorrelationContainer{D, T}}, bin_size::Int) where {D, T<:AbstractFloat}

    for measurement in keys(correlation_measurements)
        correlation_container = correlation_measurements[measurement]
        pairs = correlation_container.id_pairs::Vector{NTuple{2,Int}}
        correlations = correlation_container.correlations::Vector{Array{Complex{T}, D}}
        for i in eachindex(pairs)
            @. correlations[i] /= bin_size
        end
    end

    return nothing
end