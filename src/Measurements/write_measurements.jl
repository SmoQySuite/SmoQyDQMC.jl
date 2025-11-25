######################################################
## HIGHEST LEVEL/EXPORT WRITE MEASUREMENTS FUNCTION ##
######################################################

@doc raw"""
    write_measurements!(;
        measurement_container::NamedTuple,
        simulation_info::SimulationInfo,
        model_geometry::ModelGeometry{D, E, N},
        Δτ::E,
        bin_size::Int,
        measurement::Int = 0,
        bin::Int = measurement ÷ bin_size
    ) where {D, E<:AbstractFloat, N}

Write the measurements contained in `measurement_container` to file if `update % bin_size == 0`.
Measurements are written to file in a binary format using the [`JLD2.jl`](https://github.com/JuliaIO/JLD2.jl.git) package.

This function also does a few other things:
1. Normalizes all the measurements by the `bin_size` i.e. the number of measurements that were accumulated into the measurement container.
2. Take position space correlation function measurements and fourier transform them to momentum space.
3. Integrate relevant time-displaced correlation function measurements over imaginary time to get the corresponding zero Matsubara frequency correlation function.
4. Reset all the measurements in `measurement_container` to zero after the measurements are written to file.
"""
function write_measurements!(;
    measurement_container::NamedTuple,
    simulation_info::SimulationInfo,
    model_geometry::ModelGeometry{D, E, N},
    Δτ::E,
    bin_size::Int,
    measurement::Int = 0,
    bin::Int = measurement ÷ bin_size,
    update::Int = 0 # OLD KEYWORD ARGUMENT, WILL BE DEPRECATED
) where {D, E<:AbstractFloat, N}

    # use old keyword if necessary
    if !iszero(update) && iszero(measurement)
        measurement = update
        bin = !iszero(bin) ? bin : measurement ÷ bin_size
    end

    # check if bin file needs to be written
    if measurement % bin_size == 0

        (; datafolder, pID, write_bins_concurrent, bin_files) = simulation_info
        lattice   = model_geometry.lattice::Lattice{D}
        unit_cell = model_geometry.unit_cell::UnitCell{D,E,N}
        bonds     = model_geometry.bonds::Vector{Bond{D}}

        (; global_measurements, local_measurements,
           equaltime_correlations, equaltime_composite_correlations,
           time_displaced_correlations, time_displaced_composite_correlations,
           integrated_correlations, integrated_composite_correlations, pfft!
        ) = measurement_container

        # construct filename
        filename = joinpath(datafolder, "bins", @sprintf("pID-%d", pID), @sprintf("bin-%d.h5", bin))

        # normalize all measurements by the bin size
        normalize_measurements!(measurement_container, bin_size)

        # get hopping and phonon to bond ID mappings
        hopping_to_bond_id = measurement_container.hopping_to_bond_id
        phonon_basis_vecs = measurement_container.phonon_basis_vecs

        # displacement vector
        r = zeros(E, D)

        # open HDF5 file to write binned data to
        file = write_bins_concurrent ? h5open(filename, "w") : h5open(filename, "w"; driver=Drivers.Core(; backing_store=false))

        # if first bin record system info
        if isone(bin)
            # Length of imaginary time axis
            Lτ = measurement_container.Lτ
            # record inverse temperature
            attributes(file)["BETA"] = Lτ * Δτ
            # record the imaginary-time discretization
            attributes(file)["DELTA_TAU"] = Δτ
            # record the length of the imaginary-time axis
            attributes(file)["L_TAU"] = Lτ
            # record total number of orbitals in lattice
            attributes(file)["N_ORBITALS"] = nsites(unit_cell, lattice)
        end

        # write global measurements to group
        Global = create_group(file, "GLOBAL")
        for (measurement, value) in global_measurements
            Global[measurement] = value
        end

        # reset global measurements to zero
        for measurement in keys(global_measurements)
            global_measurements[measurement] = zero(Complex{E})
        end

        # write local measurements to group
        Local = create_group(file, "LOCAL")
        for (measurement, value) in local_measurements
            Local[measurement] = value
        end

        # reset global measurements to zero
        for measurement in keys(local_measurements)
            fill!(local_measurements[measurement], zero(Complex{E}))
        end

        # create group to contain correlation measurements
        Correlations = create_group(file, "CORRELATIONS")

        # create standard correlation group
        Standard = create_group(Correlations, "STANDARD")

        # create group for standard equal-time correlation measurements
        StandardEqualTime = create_group(Standard, "EQUAL-TIME")

        # iterate over standard equal-time correlation measurements
        for correlation in keys(equaltime_correlations)

            # get the correlation container for current standard equal-time correlation measurement
            correlation_container = equaltime_correlations[correlation]
            id_pairs = correlation_container.id_pairs::Vector{NTuple{2,Int}}
            id_type = CORRELATION_FUNCTIONS[correlation]
            correlations = correlation_container.correlations::Vector{Array{Complex{E}, D}}

            # create a group for correlation measurement
            StandardEqualTimeCorrelation = create_group(StandardEqualTime, correlation)

            # record ID pairs that were measured
            attributes(StandardEqualTimeCorrelation)["ID_PAIRS"] = id_pairs

            # record the ID type corresponding to correlation measurement
            attributes(StandardEqualTimeCorrelation)["ID_TYPE"] = id_type

            # record the position space correlations
            StandardEqualTimeCorrelation["POSITION"] = stack(correlations)

            # fourier transform correlations to momentum space
            for i in eachindex(correlations)
                # get the pair of orbitals associated with the correlation
                if (id_type == "ORBITAL_ID") || (id_type == "BOND_ID")
                    bond_b_id, bond_a_id = id_pairs[i]
                    a = bonds[bond_a_id].orbitals[1]
                    b = bonds[bond_b_id].orbitals[1]
                    # perform fourier transform
                    fourier_transform!(correlations[i], a, b, unit_cell, lattice, pfft!)
                elseif id_type == "HOPPING_ID"
                    hopping_b_id, hopping_a_id = id_pairs[i]
                    bond_a_id = hopping_to_bond_id[hopping_a_id]
                    bond_b_id = hopping_to_bond_id[hopping_b_id]
                    a = bonds[bond_a_id].orbitals[1]
                    b = bonds[bond_b_id].orbitals[1]
                    # perform fourier transform
                    fourier_transform!(correlations[i], a, b, unit_cell, lattice, pfft!)
                elseif id_type == "PHONON_ID"
                    phonon_b_id, phonon_a_id = id_pairs[i]
                    ra = phonon_basis_vecs[phonon_a_id]
                    rb = phonon_basis_vecs[phonon_b_id]
                    @. r = ra - rb
                    # perform fourier transform
                    fourier_transform!(correlations[i], r, unit_cell, lattice, pfft!)
                end
            end

            # record the momentum space correlations
            StandardEqualTimeCorrelation["MOMENTUM"] = stack(correlations)

            # reset the correlation measurements to zero
            reset!(correlation_container)
        end

        # create group for standard time-displaced correlation measurements
        StandardTimeDisplaced = create_group(Standard, "TIME-DISPLACED")

        # create group for standard integrated correlation measurements
        StandardIntegrated = create_group(Standard, "INTEGRATED")

        # iterate over standard time-displaced correlation measurements
        for correlation in keys(time_displaced_correlations)

            # get the standard time-displaced correlation container
            correlation_container = time_displaced_correlations[correlation]
            id_pairs = correlation_container.id_pairs::Vector{NTuple{2,Int}}
            id_type = CORRELATION_FUNCTIONS[correlation]
            correlations = correlation_container.correlations::Vector{Array{Complex{E}, D+1}}
            time_displaced = correlation_container.time_displaced::Bool

            # if standard time-displaced correlation measurement is being written to file
            if time_displaced

                # create a group for standard time-displaced correlation measurement
                StandardTimeDisplacedCorrelation = create_group(StandardTimeDisplaced, correlation)

                # record ID pairs that were measured
                attributes(StandardTimeDisplacedCorrelation)["ID_PAIRS"] = id_pairs

                # record the ID type corresponding to correlation measurement
                attributes(StandardTimeDisplacedCorrelation)["ID_TYPE"] = id_type

                # record the position space correlations
                StandardTimeDisplacedCorrelation["POSITION"] = stack(correlations)
            end

            # if integrated measurement is also being made
            if haskey(integrated_correlations, correlation)

                # get standard susceptibility/integrated correlation container
                susceptibility_container = integrated_correlations[correlation]
                susceptibilities = susceptibility_container.correlations::Vector{Array{Complex{E}, D}}

                # create a group for standard integrated correlation measurement
                StandardIntegratedCorrelation = create_group(StandardIntegrated, correlation)

                # record ID pairs that were measured
                attributes(StandardIntegratedCorrelation)["ID_PAIRS"] = id_pairs

                # record the ID type corresponding to correlation measurement
                attributes(StandardIntegratedCorrelation)["ID_TYPE"] = id_type

                # calculate position-space standard integrated correlation function
                for i in eachindex(correlations)

                    # perform integration of imaginary-time axis
                    susceptibility!(susceptibilities[i], correlations[i], Δτ, D+1)
                end

                # record the position space susceptibilities
                StandardIntegratedCorrelation["POSITION"] = stack(susceptibilities)
            end

            # fourier transform correlations to momentum space
            for i in eachindex(correlations)
                # get the pair of orbitals associated with the correlation
                if (id_type == "ORBITAL_ID") || (id_type == "BOND_ID")
                    bond_b_id, bond_a_id = id_pairs[i]
                    a = bonds[bond_a_id].orbitals[1]
                    b = bonds[bond_b_id].orbitals[1]
                    # perform fourier transform
                    fourier_transform!(correlations[i], a, b, D+1, unit_cell, lattice, pfft!)
                elseif id_type == "HOPPING_ID"
                    hopping_b_id, hopping_a_id = id_pairs[i]
                    bond_a_id = hopping_to_bond_id[hopping_a_id]
                    bond_b_id = hopping_to_bond_id[hopping_b_id]
                    a = bonds[bond_a_id].orbitals[1]
                    b = bonds[bond_b_id].orbitals[1]
                    # perform fourier transform
                    fourier_transform!(correlations[i], a, b, D+1, unit_cell, lattice, pfft!)
                elseif id_type == "PHONON_ID"
                    phonon_b_id, phonon_a_id = id_pairs[i]
                    ra = phonon_basis_vecs[phonon_a_id]
                    rb = phonon_basis_vecs[phonon_b_id]
                    @. r = ra - rb
                    # perform fourier transform
                    fourier_transform!(correlations[i], r, D+1, unit_cell, lattice, pfft!)
                end
            end

            # if standard time-displaced correlation measurement is being written to file
            if time_displaced

                # record the momentum space correlations
                StandardTimeDisplacedCorrelation["MOMENTUM"] = stack(correlations)
            end

            # if integrated measurement is also being made
            if haskey(integrated_correlations, correlation)

                # calculate momentum-space standard integrated correlation function
                for i in eachindex(correlations)

                    # perform integration of imaginary-time axis
                    susceptibility!(susceptibilities[i], correlations[i], Δτ, D+1)
                end

                # record the momentum space susceptibilities
                StandardIntegratedCorrelation["MOMENTUM"] = stack(susceptibilities)
            end

            # reset the correlation measurements to zero
            reset!(correlation_container)
        end

        # create composite correlation group
        Composite = create_group(Correlations, "COMPOSITE")

        # create group for composite equal-time correlation measurements
        CompositeEqualTime = create_group(Composite, "EQUAL-TIME")

        # iterate over composite equal-time correlation measurements
        for correlation in keys(equaltime_composite_correlations)

            # get the composite correlation container
            correlation_container = equaltime_composite_correlations[correlation]
            correlations = correlation_container.correlations::Array{Complex{E}, D}
            structure_factors = correlation_container.structure_factors::Array{Complex{E}, D}

            # create a group for composite equal-time correlation measurement
            CompositeEqualTimeCorrelation = create_group(CompositeEqualTime, correlation)

            # record the position space correlations
            CompositeEqualTimeCorrelation["POSITION"] = correlations

            # record the momentum space correlations
            CompositeEqualTimeCorrelation["MOMENTUM"] = structure_factors

            # reset the correlation measurements to zero
            reset!(correlation_container)
        end 

        # create group for composite time-displaced correlation measurements
        CompositeTimeDisplaced = create_group(Composite, "TIME-DISPLACED")

        # create group for composite integrated correlation measurements
        CompositeIntegrated = create_group(Composite, "INTEGRATED")

        # iterate over composite time-displaced correlation measurements
        for name in keys(time_displaced_composite_correlations)
            
            # get the composite correlation container
            correlation_container = time_displaced_composite_correlations[name]
            correlations = correlation_container.correlations::Array{Complex{E}, D+1}
            structure_factors = correlation_container.structure_factors::Array{Complex{E}, D+1}
            time_displaced = correlation_container.time_displaced::Bool

            # if composite time-displaced correlation measurement is being written to file
            if time_displaced

                # create a group for composite time-displaced correlation measurement
                CompositeTimeDisplacedCorrelation = create_group(CompositeTimeDisplaced, name)

                # record the position space correlations
                CompositeTimeDisplacedCorrelation["POSITION"] = correlations

                # record the momentum space correlations
                CompositeTimeDisplacedCorrelation["MOMENTUM"] = structure_factors
            end

            # if integrated measurement is also being made
            if haskey(integrated_composite_correlations, name)

                # get susceptibility container
                susceptibility_container = integrated_composite_correlations[name]
                susceptibilities_pos = susceptibility_container.correlations::Array{Complex{E}, D}
                susceptibilities_mom = susceptibility_container.structure_factors::Array{Complex{E}, D}

                # create a group for composite integrate correlation measurement
                CompositeIntegratedCorrelation = create_group(CompositeIntegrated, name)

                # calculate the position space susceptibility/integrated correlations
                susceptibility!(susceptibilities_pos, correlations, Δτ, D+1)

                # record the position space correlations
                CompositeIntegratedCorrelation["POSITION"] = susceptibilities_pos

                # calculate momentum space susceptibilies/integrated correlations
                susceptibility!(susceptibilities_mom, structure_factors, Δτ, D+1)

                # record the momentum space correlations
                CompositeIntegratedCorrelation["MOMENTUM"] = susceptibilities_mom
            end

            # reset the correlation measurements to zero
            reset!(correlation_container)
        end

        # record the h5file filename or contents
        file_bytes = write_bins_concurrent ? Vector{UInt8}(filename) : Vector{UInt8}(file)
        push!(bin_files, file_bytes)
        
        # close file
        close(file)
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

    # normalize equal-time composite correlation function measurements
    equaltime_composite_correlations = measurement_container.equaltime_composite_correlations
    normalize_composite_correlation_measurements!(equaltime_composite_correlations, bin_size)

    # normalize time-displaced composite correlation function measurements
    time_displaced_composite_correlations = measurement_container.time_displaced_composite_correlations
    normalize_composite_correlation_measurements!(time_displaced_composite_correlations, bin_size)

    return nothing
end

# normalize global measurements by bin size
function normalize_global_measurements!(
    global_measurements::Dict{String, Complex{T}}, bin_size::Int
) where {T<:AbstractFloat}

    for global_measurement in keys(global_measurements)
        global_measurements[global_measurement] /= bin_size
    end

    return nothing
end

# normalize local measurements by bin size
function normalize_local_measurements!(
    local_measurements::Dict{String, Vector{Complex{T}}}, bin_size::Int
) where {T<:AbstractFloat}

    for local_measurement in keys(local_measurements)
        @. local_measurements[local_measurement] /= bin_size
    end

    return nothing
end

# normalize correlation measurement
function normalize_correlation_measurements!(
    correlation_measurements::Dict{String, CorrelationContainer{D, T}}, bin_size::Int
) where {D, T<:AbstractFloat}

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

# normalize correlation measurement
function normalize_composite_correlation_measurements!(
    composite_correlation_measurements::Dict{String, CompositeCorrelationContainer{D, P, T}}, bin_size::Int
) where {D, P, T<:AbstractFloat}

    for name in keys(composite_correlation_measurements)
        correlation_container = composite_correlation_measurements[name]
        correlations = correlation_container.correlations::Array{Complex{T}, P}
        structure_factors = correlation_container.structure_factors::Array{Complex{T}, P}
        @. correlations /= bin_size
        @. structure_factors /= bin_size
    end

    return nothing
end