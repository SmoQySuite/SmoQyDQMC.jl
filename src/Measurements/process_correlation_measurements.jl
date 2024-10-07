######################################
## PROCESS CORRELATION MEASUREMENTS ##
######################################

@doc raw"""
    process_correlation_measurements(
        folder::String,
        N_bins::Int,
        pIDs::Union{Vector{Int},Int} = Int[],
        types::Vector{String} = ["equal-time", "time-displaced", "integrated"],
        spaces::Vector{String} = ["position", "momentum"]
    )

    function process_correlation_measurements(
        comm::MPI.Comm,
        N_bins::Int,
        pIDs::Vector{Int} = Int[],
        types::Vector{String} = ["equal-time", "time-displaced", "integrated"],
        spaces::Vector{String} = ["position", "momentum"]
    )

Process correlation measurements, calculating the average and errors and writing the result to CSV file.
"""
function process_correlation_measurements(
    folder::String,
    N_bins::Int,
    pIDs::Union{Vector{Int},Int} = Int[],
    types::Vector{String} = ["equal-time", "time-displaced", "integrated"],
    spaces::Vector{String} = ["position", "momentum"]
)
    
    # set the walkers to iterate over
    if isempty(pIDs)

        # get the number of MPI walkers
        N_walkers = get_num_walkers(folder)

        # get the pIDs
        pIDs = collect(0:(N_walkers-1))
    end

    # load model geometry
    β, Δτ, Lτ, model_geometry = load_model_summary(folder)

    # process correlation measurements
    _process_correlation_measurements(folder, N_bins, pIDs, types, spaces, Lτ, model_geometry)

    return nothing
end

# prcoess correlations measurements averaging over all MPI walkers
function _process_correlation_measurements(
    folder::String,
    N_bins::Int,
    pIDs::Vector{Int},
    types::Vector{String},
    spaces::Vector{String},
    Lτ::Int,
    model_geometry::ModelGeometry{D,T,N}
) where {D, T<:AbstractFloat, N}

    # calculate bin intervals
    bin_intervals = get_bin_intervals(folder, N_bins, pIDs[1])

    # get binned sign for each pID
    binned_sign = get_average_sign(folder, bin_intervals, pIDs[1])
    binned_signs = [binned_sign]
    for pID in pIDs[2:end]
        push!(binned_signs, get_average_sign(folder, bin_intervals, pID))
    end

    # iterate over types of correlation measurements to process
    for type in types

        # get the name of each correlation measurement that was made
        correlation_folder = joinpath(folder, type)

        # get the names of each correlation
        correlations = filter(i -> isdir(joinpath(correlation_folder,i)), readdir(correlation_folder))

        # iterate over correlations
        for correlation in correlations

            # iterate over relevant spaces
            for space in spaces

                # process correlation measurement
                if type == "time-displaced"
                    if haskey(CORRELATION_FUNCTIONS, correlation)
                        _process_correlation_measurement(
                            folder, correlation, space, pIDs, bin_intervals, binned_signs, model_geometry, Lτ, false
                        )
                    else
                        _process_composite_correlation_measurement(
                            folder, correlation, space, pIDs, bin_intervals, binned_signs, model_geometry, Lτ, false
                        )
                    end
                else
                    if haskey(CORRELATION_FUNCTIONS, correlation)
                        _process_correlation_measurement(
                            folder, correlation, type, space, pIDs, bin_intervals, binned_signs, model_geometry, false
                        )
                    else
                        _process_composite_correlation_measurement(
                            folder, correlation, type, space, pIDs, bin_intervals, binned_signs, model_geometry, false
                        )
                    end
                end
            end
        end
    end

    return nothing
end

# process correlation measurements using single MPI walker
function _process_correlation_measurements(
    folder::String,
    N_bins::Int,
    pID::Int,
    types::Vector{String},
    spaces::Vector{String},
    Lτ::Int,
    model_geometry::ModelGeometry{D,T,N}
) where {D, T<:AbstractFloat, N}

    # calculate bin intervals
    bin_intervals = get_bin_intervals(folder, N_bins, pID)

    # get binned sign for each pID
    binned_sign = get_average_sign(folder, bin_intervals, pID)
    binned_signs = [binned_sign]

    # iterate over types of correlation measurements to process
    for type in types

        # get the name of each correlation measurement that was made
        correlation_folder = joinpath(folder, type)

        # get the names of each correlation
        correlations = filter(i -> isdir(joinpath(correlation_folder,i)), readdir(correlation_folder))

        # iterate over correlations
        for correlation in correlations

            # iterate over relevant spaces
            for space in spaces

                # process correlation measurement
                if type == "time-displaced"
                    if haskey(CORRELATION_FUNCTIONS, correlation)
                        _process_correlation_measurement(
                            folder, correlation, space, [pID], bin_intervals, binned_signs, model_geometry, Lτ, true
                        )
                    else
                        _process_composite_correlation_measurement(
                            folder, correlation, space, [pID], bin_intervals, binned_signs, model_geometry, Lτ, true
                        )
                    end
                else
                    if haskey(CORRELATION_FUNCTIONS, correlation)
                        _process_correlation_measurement(
                            folder, correlation, type, space, [pID], bin_intervals, binned_signs, model_geometry, true
                        )
                    else
                        _process_composite_correlation_measurement(
                            folder, correlation, type, space, [pID], bin_intervals, binned_signs, model_geometry, true
                        )
                    end
                end
            end
        end
    end

    return nothing
end

@doc raw"""
    process_correlation_measurement(
        folder::String,
        correlation::String,
        type::String,
        space::String,
        N_bins::Int,
        pIDs::Union{Vector{Int}, Int} = Int[]
    ) 

Process results for the specified correlation function, calculating the associated average and error statistics for it and writing
the result to CSV file.
If `pIDs` is not specified, then the calculated statistics are arrived at by averaging over the results for all MPI walkers.
"""
function process_correlation_measurement(
    folder::String,
    correlation::String,
    type::String,
    space::String,
    N_bins::Int,
    pIDs::Union{Vector{Int}, Int} = Int[]
)

    β, Δτ, Lτ, model_geometry = load_model_summary(folder)
    _process_correlation_measurement_high_level(folder, correlation, type, space, N_bins, pIDs, Lτ, model_geometry)

    return nothing
end

# process correlation measurement averaging over all MPI walkers
function _process_correlation_measurement_high_level(
    folder::String,
    correlation::String,
    type::String,
    space::String,
    N_bins::Int,
    pIDs::Vector{Int},
    Lτ::Int,
    model_geometry::ModelGeometry{D,T,N}
) where {D, T<:AbstractFloat, N}

    # set the walkers to iterate over
    if length(pIDs) == 0

        # get the number of MPI walkers
        N_walkers = get_num_walkers(folder)

        # get the pIDs
        pIDs = collect(0:(N_walkers-1))
    end

    # calculate bin intervals
    bin_intervals = get_bin_intervals(folder, N_bins, pIDs[1])

    # get binned sign for each pID
    binned_sign = get_average_sign(folder, bin_intervals, pIDs[1])
    binned_signs = [binned_sign]
    for pID in pIDs[2:end]
        push!(binned_signs, get_average_sign(folder, bin_intervals, pID))
    end

    # process correlation measurement
    if type == "time-displaced"
        if haskey(CORRELATION_FUNCTIONS, correlation)
            _process_correlation_measurement(
                folder, correlation, space, pIDs, bin_intervals, binned_signs, model_geometry, Lτ, false
            )
        else
            _process_composite_correlation_measurement(
                folder, correlation, space, pIDs, bin_intervals, binned_signs, model_geometry, Lτ, false
            )
        end
    else
        if haskey(CORRELATION_FUNCTIONS, correlation)
            _process_correlation_measurement(
                folder, correlation, type, space, pIDs, bin_intervals, binned_signs, model_geometry, false
            )
        else
            _process_composite_correlation_measurement(
                folder, correlation, type, space, pIDs, bin_intervals, binned_signs, model_geometry, false
            )
        end
    end

    return nothing
end

# process correlation measurement for single MPI walker
function _process_correlation_measurement_high_level(
    folder::String,
    correlation::String,
    type::String,
    space::String,
    N_bins::Int,
    pID::Int,
    Lτ::Int,
    model_geometry::ModelGeometry{D,T,N}
) where {D, T<:AbstractFloat, N}

    # calculate bin intervals
    bin_intervals = get_bin_intervals(folder, N_bins, pID)

    # get binned sign for each pID
    binned_sign = get_average_sign(folder, bin_intervals, pID)
    binned_signs = [binned_sign]

    # process correlation measurement
    if type == "time-displaced"
        if haskey(CORRELATION_FUNCTIONS, correlation)
            _process_correlation_measurement(
                folder, correlation, space, [pID], bin_intervals, binned_signs, model_geometry, Lτ, true
            )
        else
            _process_composite_correlation_measurement(
                folder, correlation, space, [pID], bin_intervals, binned_signs, model_geometry, Lτ, true
            )
        end
    else
        if haskey(CORRELATION_FUNCTIONS, correlation)
            _process_correlation_measurement(
                folder, correlation, type, space, [pID], bin_intervals, binned_signs, model_geometry, true
            )
        else
            _process_composite_correlation_measurement(
                folder, correlation, type, space, [pID], bin_intervals, binned_signs, model_geometry, true
            )
        end
    end

    return nothing
end


# process time-displaced correlation
function _process_correlation_measurement(
    folder::String,
    correlation::String,
    space::String,
    pIDs::Vector{Int},
    bin_intervals::Vector{UnitRange{Int}},
    binned_signs::Vector{Vector{Complex{T}}},
    model_geometry::ModelGeometry{D,T,N},
    Lτ::Int,
    single_pID::Bool,
) where {D, T<:AbstractFloat, N}

    # get the folder the stats will be written to
    write_folder = joinpath(folder, "time-displaced", correlation)

    # get the read folders
    read_folder = joinpath(write_folder, space)

    # get size of lattice
    lattice = model_geometry.lattice::Lattice{D}
    L = lattice.L

    # get number of bins
    N_bins = length(bin_intervals)

    # correlation container for stats
    binned_correlation = zeros(Complex{T}, N_bins, L...)
    correlation_avg = zeros(Complex{T}, L...)
    correlation_std = zeros(T, L...)
    correlation_var = correlation_std

    # pre-allocate arrays for jackknife
    jackknife_samples = (zeros(Complex{T}, N_bins), zeros(Complex{T}, N_bins))
    jackknife_g = zeros(Complex{T}, N_bins)

    # get correlation ID pairs
    pairs = JLD2.load(joinpath(read_folder, @sprintf("bin-1_pID-%d.jld2", pIDs[1])), "id_pairs")

    # stats filename
    if single_pID
        filename = @sprintf("%s_%s_%s_pID-%d_stats.csv", correlation, space, "time-displaced", pIDs[1])
    else
        filename = @sprintf("%s_%s_%s_stats.csv", correlation, space, "time-displaced")
    end

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

        # iterate over ID pairs
        for n in eachindex(pairs)

            # iterate over imaginary imaginary time-displacements
            for l in 0:Lτ

                # reset correlation containers
                fill!(correlation_avg, 0)
                fill!(correlation_std, 0)

                # iterate over walkers
                for i in eachindex(pIDs)

                    # get the pID
                    pID = pIDs[i]

                    # read in binned correlation data
                    read_correlation_measurement!(binned_correlation, folder, correlation, l, space, n, pID, bin_intervals)

                    # calculate average and error for current pID
                    analyze_correlations!(correlation_avg, correlation_var, binned_correlation, binned_signs[i], jackknife_samples, jackknife_g)
                end

                # normalize stats
                N_walkers = length(pIDs)
                @. correlation_avg /= N_walkers
                @. correlation_std  = sqrt(correlation_var) / N_walkers

                # write the correlation stats to file
                index = write_correlation(fout, pairs[n], index, l, correlation_avg, correlation_std)
            end
        end
    end

    return nothing
end

# process time-displaced composite correlaiton
function _process_composite_correlation_measurement(
    folder::String,
    name::String,
    space::String,
    pIDs::Vector{Int},
    bin_intervals::Vector{UnitRange{Int}},
    binned_signs::Vector{Vector{Complex{T}}},
    model_geometry::ModelGeometry{D,T,N},
    Lτ::Int,
    single_pID::Bool,
) where {D, T<:AbstractFloat, N}

    # get the folder the stats will be written to
    write_folder = joinpath(folder, "time-displaced", name)

    # get the read folders
    read_folder = joinpath(write_folder, space)

    # get size of lattice
    lattice = model_geometry.lattice::Lattice{D}
    L = lattice.L

    # get number of bins
    N_bins = length(bin_intervals)

    # correlation container for stats
    binned_correlation = zeros(Complex{T}, N_bins, L...)
    correlation_avg = zeros(Complex{T}, L...)
    correlation_std = zeros(T, L...)
    correlation_var = correlation_std

    # pre-allocate arrays for jackknife
    jackknife_samples = (zeros(Complex{T}, N_bins), zeros(Complex{T}, N_bins))
    jackknife_g = zeros(Complex{T}, N_bins)

    # stats filename
    if single_pID
        filename = @sprintf("%s_%s_%s_pID-%d_stats.csv", name, space, "time-displaced", pIDs[1])
    else
        filename = @sprintf("%s_%s_%s_stats.csv", name, space, "time-displaced")
    end

    # open stats files
    open(joinpath(write_folder, filename), "w") do fout

        # write header to file
        if space == "position"
            write(fout, join(("INDEX", "TAU", ("R_$d" for d in D:-1:1)..., "MEAN_R", "MEAN_I", "STD\n"), " "))
        else
            write(fout, join(("INDEX", "TAU", ("K_$d" for d in D:-1:1)..., "MEAN_R", "MEAN_I", "STD\n"), " "))
        end

        # initialize index to zero
        index = 0

        # iterate over imaginary imaginary time-displacements
        for l in 0:Lτ

            # reset correlation containers
            fill!(correlation_avg, 0)
            fill!(correlation_std, 0)

            # iterate over walkers
            for i in eachindex(pIDs)

                # get the pID
                pID = pIDs[i]

                # read in binned correlation data
                read_correlation_measurement!(binned_correlation, folder, name, l, space, pID, bin_intervals)

                # calculate average and error for current pID
                analyze_correlations!(correlation_avg, correlation_var, binned_correlation, binned_signs[i], jackknife_samples, jackknife_g)
            end

            # normalize stats
            N_walkers = length(pIDs)
            @. correlation_avg /= N_walkers
            @. correlation_std  = sqrt(correlation_var) / N_walkers

            # write the correlation stats to file
            index = write_correlation(fout, index, l, correlation_avg, correlation_std)
        end
    end

    return nothing
end

# process equal-time or integrated correlation
function _process_correlation_measurement(
    folder::String,
    correlation::String,
    type::String,
    space::String,
    pIDs::Vector{Int},
    bin_intervals::Vector{UnitRange{Int}},
    binned_signs::Vector{Vector{Complex{T}}},
    model_geometry::ModelGeometry{D,T,N},
    single_pID::Bool
) where {D, T<:AbstractFloat, N}

    # get the folder the stats will be written to
    write_folder = joinpath(folder, type, correlation)

    # get the read folders
    read_folder = joinpath(write_folder, space)

    # get size of lattice
    lattice = model_geometry.lattice::Lattice{D}
    L = lattice.L

    # number of bins
    N_bins = length(bin_intervals)

    # correlation container for stats
    binned_correlation = zeros(Complex{T}, N_bins, L...)
    correlation_avg = zeros(Complex{T}, L...)
    correlation_std = zeros(T, L...)
    correlation_var = correlation_std

    # pre-allocate arrays for jackknife
    jackknife_samples = (zeros(Complex{T}, N_bins), zeros(Complex{T}, N_bins))
    jackknife_g = zeros(Complex{T}, N_bins)

    # get correlation ID pairs
    pairs = JLD2.load(joinpath(read_folder, @sprintf("bin-1_pID-%d.jld2", pIDs[1])), "id_pairs")

    # stats filename
    if single_pID
        filename = @sprintf("%s_%s_%s_pID-%d_stats.csv", correlation, space, type, pIDs[1])
    else
        filename = @sprintf("%s_%s_%s_stats.csv", correlation, space, type)
    end

    # open stats files
    open(joinpath(write_folder, filename), "w") do fout

        # get the id type
        id_type = CORRELATION_FUNCTIONS[correlation]

        # write header to file
        if space == "position"
            write(fout, join(("INDEX", "$(id_type)_2", "$(id_type)_1", ("R_$d" for d in D:-1:1)..., "MEAN_R", "MEAN_I", "STD\n"), " "))
        else
            write(fout, join(("INDEX", "$(id_type)_2", "$(id_type)_1", ("K_$d" for d in D:-1:1)..., "MEAN_R", "MEAN_I", "STD\n"), " "))
        end

        # initialize index to zero
        index = 0

        # iterate over ID pairs
        for n in eachindex(pairs)

            # reset correlation containers
            fill!(correlation_avg, 0)
            fill!(correlation_std, 0)

            # iterate over walkers
            for i in eachindex(pIDs)

                # get the pID
                pID = pIDs[i]

                # read in binned correlation data
                read_correlation_measurement!(binned_correlation, folder, correlation, type, space, n, pID, bin_intervals)

                # calculate average and error for current pID
                analyze_correlations!(correlation_avg, correlation_var, binned_correlation, binned_signs[i], jackknife_samples, jackknife_g)
            end

            # normalize stats
            N_walkers = length(pIDs)
            @. correlation_avg /= N_walkers
            @. correlation_std  = sqrt(correlation_var) / N_walkers

            # write the correlation stats to file
            index = write_correlation(fout, pairs[n], index, correlation_avg, correlation_std)
        end
    end

    return nothing
end

# process equal-time or integrated composite correlation
function _process_composite_correlation_measurement(
    folder::String,
    name::String,
    type::String,
    space::String,
    pIDs::Vector{Int},
    bin_intervals::Vector{UnitRange{Int}},
    binned_signs::Vector{Vector{Complex{T}}},
    model_geometry::ModelGeometry{D,T,N},
    single_pID::Bool
) where {D, T<:AbstractFloat, N}

    # get the folder the stats will be written to
    write_folder = joinpath(folder, type, name)

    # get the read folders
    read_folder = joinpath(write_folder, space)

    # get size of lattice
    lattice = model_geometry.lattice::Lattice{D}
    L = lattice.L

    # number of bins
    N_bins = length(bin_intervals)

    # correlation container for stats
    binned_correlation = zeros(Complex{T}, N_bins, L...)
    correlation_avg = zeros(Complex{T}, L...)
    correlation_std = zeros(T, L...)
    correlation_var = correlation_std

    # pre-allocate arrays for jackknife
    jackknife_samples = (zeros(Complex{T}, N_bins), zeros(Complex{T}, N_bins))
    jackknife_g = zeros(Complex{T}, N_bins)

    # stats filename
    if single_pID
        filename = @sprintf("%s_%s_%s_pID-%d_stats.csv", name, space, type, pIDs[1])
    else
        filename = @sprintf("%s_%s_%s_stats.csv", name, space, type)
    end

    # open stats files
    open(joinpath(write_folder, filename), "w") do fout

        # write header to file
        if space == "position"
            write(fout, join(("INDEX", ("R_$d" for d in D:-1:1)..., "MEAN_R", "MEAN_I", "STD\n"), " "))
        else
            write(fout, join(("INDEX", ("K_$d" for d in D:-1:1)..., "MEAN_R", "MEAN_I", "STD\n"), " "))
        end

        # initialize index to zero
        index = 0

        # reset correlation containers
        fill!(correlation_avg, 0)
        fill!(correlation_std, 0)

        # iterate over walkers
        for i in eachindex(pIDs)

            # get the pID
            pID = pIDs[i]

            # read in binned correlation data
            read_correlation_measurement!(binned_correlation, folder, name, type, space, pID, bin_intervals)

            # calculate average and error for current pID
            analyze_correlations!(correlation_avg, correlation_var, binned_correlation, binned_signs[i], jackknife_samples, jackknife_g)
        end

        # normalize stats
        N_walkers = length(pIDs)
        @. correlation_avg /= N_walkers
        @. correlation_std  = sqrt(correlation_var) / N_walkers

        # write the correlation stats to file
        index = write_correlation(fout, index, correlation_avg, correlation_std)
    end

    return nothing
end


# read in equal-time or integrated correlation function
function read_correlation_measurement(
    folder::String,
    correlation::String,
    type::String,
    space::String,
    n_pair::Int,
    model_geometry::ModelGeometry{D,T,N},
    pID::Int,
    bin_intervals::Vector{UnitRange{Int}},
) where {D, T<:AbstractFloat, N}

    @assert type in ("equal-time", "integrated")
    @assert space in ("position", "momentum")

    # get lattice size
    lattice = model_geometry.lattice::Lattice{D}
    L = lattice.L

    # container for binned correlation data
    binned_correlation = zeros(Complex{T}, N_bins, L...)

    # read in binned correlation
    read_correlation_measurement!(binned_correlation, folder, correlation, type, space, n_pair, pID, bin_intervals)

    return binned_correlation
end

function read_correlation_measurement!(
    binned_correlation::AbstractArray{Complex{T}},
    folder::String,
    correlation::String,
    type::String,
    space::String,
    n_pair::Int,
    pID::Int,
    bin_intervals::Vector{UnitRange{Int}},
) where {T<:AbstractFloat}

    @assert type in ("equal-time", "integrated")
    @assert space in ("position", "momentum")

    # construct directory name where binary data lives
    correlation_folder = joinpath(folder, type, correlation, space)

    # number of bins
    N_bins = length(bin_intervals)

    # bin size
    N_binsize = length(bin_intervals[1])

    # initialize binned correlation to zero
    fill!(binned_correlation, 0)

    # iterate over bins
    for bin in 1:N_bins

        # get a specific correlation bin
        correlation_bin = selectdim(binned_correlation, 1, bin)

        # iterate over bin elements
        for i in bin_intervals[bin]

            # load the correlation data
            corr = JLD2.load(@sprintf("%s/bin-%d_pID-%d.jld2", correlation_folder, i, pID), "correlations")[n_pair]

            # record the correlation data
            @. correlation_bin += corr / N_binsize
        end
    end

    return nothing
end


# read in equal-time or integrated composite correlation function
function read_correlation_measurement(
    folder::String,
    correlation::String,
    type::String,
    space::String,
    model_geometry::ModelGeometry{D,T,N},
    pID::Int,
    bin_intervals::Vector{UnitRange{Int}},
) where {D, T<:AbstractFloat, N}

    @assert type in ("equal-time", "integrated")
    @assert space in ("position", "momentum")

    # get lattice size
    lattice = model_geometry.lattice::Lattice{D}
    L = lattice.L

    # container for binned correlation data
    binned_correlation = zeros(Complex{T}, N_bins, L...)

    # read in binned correlation
    read_correlation_measurement!(binned_correlation, folder, correlation, type, space, pID, bin_intervals)

    return binned_correlation
end

function read_correlation_measurement!(
    binned_correlation::AbstractArray{Complex{T}},
    folder::String,
    correlation::String,
    type::String,
    space::String,
    pID::Int,
    bin_intervals::Vector{UnitRange{Int}},
) where {T<:AbstractFloat}

    @assert type in ("equal-time", "integrated")
    @assert space in ("position", "momentum")

    # construct directory name where binary data lives
    correlation_folder = joinpath(folder, type, correlation, space)

    # number of bins
    N_bins = length(bin_intervals)

    # bin size
    N_binsize = length(bin_intervals[1])

    # initialize binned correlation to zero
    fill!(binned_correlation, 0)

    # iterate over bins
    for bin in 1:N_bins

        # get a specific correlation bin
        correlation_bin = selectdim(binned_correlation, 1, bin)

        # iterate over bin elements
        for i in bin_intervals[bin]

            # load the correlation data
            corr = JLD2.load(@sprintf("%s/bin-%d_pID-%d.jld2", correlation_folder, i, pID), "correlations")

            # record the correlation data
            @. correlation_bin += corr / N_binsize
        end
    end

    return nothing
end


# read in time-displaced correlation function
function read_correlation_measurement(
    folder::String,
    correlation::String,
    l::Int,
    space::String,
    n_pair::Int,
    model_geometry::ModelGeometry{D,T,N},
    pID::Int,
    bin_intervals::Vector{UnitRange{Int}},
) where {D, T<:AbstractFloat, N}

    # get lattice size
    lattice = model_geometry.lattice::Lattice{D}
    L = lattice.L

    # container for binned correlation data
    binned_correlation = zeros(Complex{T}, N_bins, L...)

    # read in binned correlation
    read_correlation_measurement!(binned_correlation, folder, correlation, l, space, n_pair, pID, bin_intervals)

    return binned_correlation
end

function read_correlation_measurement!(
    binned_correlation::AbstractArray{Complex{T}},
    folder::String,
    correlation::String,
    l::Int,
    space::String,
    n_pair::Int,
    pID::Int,
    bin_intervals::Vector{UnitRange{Int}},
) where {T<:AbstractFloat}

    @assert space in ("position", "momentum")

    # construct directory name where binary data lives
    correlation_folder = joinpath(folder, "time-displaced", correlation, space)

    # bin size
    N_binsize = length(bin_intervals[1])

    # initialize binned correlation to zero
    fill!(binned_correlation, 0)

    # iterate over bins
    for bin in eachindex(bin_intervals)

        # get a specific correlation bin
        correlation_bin = selectdim(binned_correlation, 1, bin)

        # iterate over bin elements
        for i in bin_intervals[bin]

            # load the correlation data
            corr = JLD2.load(@sprintf("%s/bin-%d_pID-%d.jld2", correlation_folder, i, pID), "correlations")[n_pair]

            # get correlation for the specified time-slice
            corr_l = selectdim(corr, ndims(corr), l+1)

            # record the correlation data
            @. correlation_bin += corr_l / N_binsize
        end
    end

    return nothing
end


# read in time-displaced composite correlation function
function read_correlation_measurement(
    folder::String,
    correlation::String,
    l::Int,
    space::String,
    model_geometry::ModelGeometry{D,T,N},
    pID::Int,
    bin_intervals::Vector{UnitRange{Int}},
) where {D, T<:AbstractFloat, N}

    # get lattice size
    lattice = model_geometry.lattice::Lattice{D}
    L = lattice.L

    # container for binned correlation data
    binned_correlation = zeros(Complex{T}, N_bins, L...)

    # read in binned correlation
    read_correlation_measurement!(binned_correlation, folder, correlation, l, space, pID, bin_intervals)

    return binned_correlation
end

function read_correlation_measurement!(
    binned_correlation::AbstractArray{Complex{T}},
    folder::String,
    correlation::String,
    l::Int,
    space::String,
    pID::Int,
    bin_intervals::Vector{UnitRange{Int}},
) where {T<:AbstractFloat}

    @assert space in ("position", "momentum")

    # construct directory name where binary data lives
    correlation_folder = joinpath(folder, "time-displaced", correlation, space)

    # bin size
    N_binsize = length(bin_intervals[1])

    # initialize binned correlation to zero
    fill!(binned_correlation, 0)

    # iterate over bins
    for bin in eachindex(bin_intervals)

        # get a specific correlation bin
        correlation_bin = selectdim(binned_correlation, 1, bin)

        # iterate over bin elements
        for i in bin_intervals[bin]

            # load the correlation data
            corr = JLD2.load(@sprintf("%s/bin-%d_pID-%d.jld2", correlation_folder, i, pID), "correlations")

            # get correlation for the specified time-slice
            corr_l = selectdim(corr, ndims(corr), l+1)

            # record the correlation data
            @. correlation_bin += corr_l / N_binsize
        end
    end

    return nothing
end


function analyze_correlations!(
    correlations_avg::AbstractArray{Complex{T}},
    correlations_var::AbstractArray{T},
    binned_correlations::AbstractArray{Complex{T}},
    binned_sign::Vector{Complex{T}},
    jackknife_samples::Tuple{Vector{Complex{T}},Vector{Complex{T}}},
    jackknife_g::Vector{Complex{T}}
) where {T<:AbstractFloat}

    # iterate over correlations
    for c in CartesianIndices(correlations_avg)

        # get all binned values
        binned_vals = @view binned_correlations[:, c]

        # calculate correlation stats
        C, ΔC = jackknife(
            /, binned_vals, binned_sign,
            jackknife_samples = jackknife_samples,
            jackknife_g = jackknife_g
        )
        correlations_avg[c] += C
        correlations_var[c] += ΔC^2
    end

    return nothing
end


# write equal-time or integrated correlations to file
function write_correlation(
    fout::IO,
    pair::NTuple{2,Int},
    index::Int,
    correlations_avg::AbstractArray{Complex{T}},
    correlations_err::AbstractArray{T}
) where {T<:AbstractFloat}

    # iterate over correlation values
    for c in CartesianIndices(correlations_avg)

        # increment index counter
        index += 1

        # write correlation stat to file
        C  = correlations_avg[c]
        ΔC = correlations_err[c]
        _write_correlation(fout, index, pair, c.I, C, ΔC)
    end

    return index
end

# write equal-time or integrated composite correlations to file
function write_correlation(
    fout::IO,
    index::Int,
    correlations_avg::AbstractArray{Complex{T}},
    correlations_err::AbstractArray{T}
) where {T<:AbstractFloat}

    # iterate over correlation values
    for c in CartesianIndices(correlations_avg)

        # increment index counter
        index += 1

        # write correlation stat to file
        C  = correlations_avg[c]
        ΔC = correlations_err[c]
        _write_correlation(fout, index, c.I, C, ΔC)
    end

    return index
end

# write time-displaced correlation to file
function write_correlation(
    fout::IO,
    pair::NTuple{2,Int},
    index::Int,
    l::Int,
    correlations_avg::AbstractArray{Complex{T}},
    correlations_err::AbstractArray{T}
) where {T<:AbstractFloat}

    # iterate over correlation values
    for c in CartesianIndices(correlations_avg)

        # increment index counter
        index += 1

        # write correlation stat to file
        C  = correlations_avg[c]
        ΔC = correlations_err[c]
        _write_correlation(fout, index, pair, l, c.I, C, ΔC)
    end

    return index
end

# write time-displaced composite correlation to file
function write_correlation(
    fout::IO,
    index::Int,
    l::Int,
    correlations_avg::AbstractArray{Complex{T}},
    correlations_err::AbstractArray{T}
) where {T<:AbstractFloat}

    # iterate over correlation values
    for c in CartesianIndices(correlations_avg)

        # increment index counter
        index += 1

        # write correlation stat to file
        C  = correlations_avg[c]
        ΔC = correlations_err[c]
        _write_correlation(fout, index, l, c.I, C, ΔC)
    end

    return index
end

# write equal-time or integrated correlation stat to file for D=1 dimensional system
function _write_correlation(
    fout::IO, index::Int, pair::NTuple{2,Int}, n::NTuple{1,Int}, C::Complex{T}, ΔC::T
) where {T<:AbstractFloat}

    @printf(fout, "%d %d %d %d %.8f %.8f %.8f\n", index, pair[2], pair[1], n[1]-1, real(C), imag(C), ΔC)

    return nothing
end

# write equal-time or integrated composite correlation stat to file for D=1 dimensional system
function _write_correlation(
    fout::IO, index::Int, n::NTuple{1,Int}, C::Complex{T}, ΔC::T
) where {T<:AbstractFloat}

    @printf(fout, "%d %d %.8f %.8f %.8f\n", index, n[1]-1, real(C), imag(C), ΔC)

    return nothing
end

# write equal-time or integrated correlation stat to file for D=2 dimensional system
function _write_correlation(
    fout::IO, index::Int, pair::NTuple{2,Int}, n::NTuple{2,Int}, C::Complex{T}, ΔC::T
) where {T<:AbstractFloat}

    @printf(fout, "%d %d %d %d %d %.8f %.8f %.8f\n", index, pair[2], pair[1], n[2]-1, n[1]-1, real(C), imag(C), ΔC)

    return nothing
end

# write equal-time or integrated composite correlation stat to file for D=2 dimensional system
function _write_correlation(
    fout::IO, index::Int, n::NTuple{2,Int}, C::Complex{T}, ΔC::T
) where {T<:AbstractFloat}

    @printf(fout, "%d %d %d %.8f %.8f %.8f\n", index, n[2]-1, n[1]-1, real(C), imag(C), ΔC)

    return nothing
end

# write equal-time or integrated correlation stat to file for D=3 dimensional system
function _write_correlation(
    fout::IO, index::Int, pair::NTuple{2,Int}, n::NTuple{3,Int}, C::Complex{T}, ΔC::T
) where {T<:AbstractFloat}

    @printf(fout, "%d %d %d %d %d %d %.8f %.8f %.8f\n", index, pair[2], pair[1], n[3]-1, n[2]-1, n[1]-1, real(C), imag(C), ΔC)

    return nothing
end

# write equal-time or integrated composite correlation stat to file for D=3 dimensional system
function _write_correlation(
    fout::IO, index::Int, n::NTuple{3,Int}, C::Complex{T}, ΔC::T
) where {T<:AbstractFloat}

    @printf(fout, "%d %d %d %d %.8f %.8f %.8f\n", index, n[3]-1, n[2]-1, n[1]-1, real(C), imag(C), ΔC)

    return nothing
end

# write time-displaced correlation stat to file for D=1 dimensional system
function _write_correlation(
    fout::IO, index::Int, pair::NTuple{2,Int}, l::Int, n::NTuple{1,Int}, C::Complex{T}, ΔC::T
) where {T<:AbstractFloat}

    @printf(fout, "%d %d %d %d %d %.8f %.8f %.8f\n", index, pair[2], pair[1], l, n[1]-1, real(C), imag(C), ΔC)

    return nothing
end

# write time-displaced composite correlation stat to file for D=1 dimensional system
function _write_correlation(
    fout::IO, index::Int, l::Int, n::NTuple{1,Int}, C::Complex{T}, ΔC::T
) where {T<:AbstractFloat}

    @printf(fout, "%d %d %d %.8f %.8f %.8f\n", index, l, n[1]-1, real(C), imag(C), ΔC)

    return nothing
end

# write time-displaced correlation stat to file for D=2 dimensional system
function _write_correlation(
    fout::IO, index::Int, pair::NTuple{2,Int}, l::Int, n::NTuple{2,Int}, C::Complex{T}, ΔC::T
) where {T<:AbstractFloat}

    @printf(fout, "%d %d %d %d %d %d %.8f %.8f %.8f\n", index, pair[2], pair[1], l, n[2]-1, n[1]-1, real(C), imag(C), ΔC)

    return nothing
end

# write time-displaced composite correlation stat to file for D=2 dimensional system
function _write_correlation(
    fout::IO, index::Int, l::Int, n::NTuple{2,Int}, C::Complex{T}, ΔC::T
) where {T<:AbstractFloat}

    @printf(fout, "%d %d %d %d %.8f %.8f %.8f\n", index, l, n[2]-1, n[1]-1, real(C), imag(C), ΔC)

    return nothing
end

# write time-displaced correlation stat to file for D=3 dimensional system
function _write_correlation(
    fout::IO, index::Int, pair::NTuple{2,Int}, l::Int, n::NTuple{3,Int}, C::Complex{T}, ΔC::T
) where {T<:AbstractFloat}

    @printf(fout, "%d %d %d %d %d %d %d %.8f %.8f %.8f\n", index, pair[2], pair[1], l, n[3]-1, n[2]-1, n[1]-1, real(C), imag(C), ΔC)

    return nothing
end

# write time-displaced composite correlation stat to file for D=3 dimensional system
function _write_correlation(
    fout::IO, index::Int, l::Int, n::NTuple{3,Int}, C::Complex{T}, ΔC::T
) where {T<:AbstractFloat}

    @printf(fout, "%d %d %d %d %d %.8f %.8f %.8f\n", index, l, n[3]-1, n[2]-1, n[1]-1, real(C), imag(C), ΔC)

    return nothing
end