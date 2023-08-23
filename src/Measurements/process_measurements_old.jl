@doc raw"""
    process_measurements(folder::String, N_bin::Int; time_displaced::Bool = false)

Process the measurements recorded in the simulation directory `folder`, where `N_bin` is the number of bins the data is grouped into for calculating error bars.
Note that this method will over-write an existing correlation stats file if there already is one.
The boolean flag `time_displaced` determines whether or not to calculate error bars for time-displaced correlation measurements,
as this can take a non-negligible amount of time for large system, especially when many simulations were run in parallel.
"""
function process_measurements(folder::String, N_bin::Int; time_displaced::Bool = false)

    # make sure valid directory
    @assert isdir(folder)

    # get the number of processes that ran during simulation
    N_process = _get_num_processes(folder)

    # get the number of files in each measurement directory
    N_files = length(readdir(joinpath(folder,"global")))

    # get the number of measurements per process
    N_measurement = div(N_files, N_process)

    # calculate the bin size per process
    N_binsize = div(N_measurement, N_bin)

    # make sure the number of bins is valid
    @assert (N_bin * N_binsize) == N_measurement "([N_bin = $N_bin] * [N_binsize = $N_binsize]) != [N_measurement = $N_measurement]"

    # load model information
    β, Δτ, Lτ, model_geometry = _load_model_summary(folder)

    # get filenames associated with each bin
    bin_to_filenames = _get_filenames_by_bin(N_bin, N_binsize, N_process)

    # get the binned sign data
    binned_sgn = _get_binned_sgn_data(folder, bin_to_filenames)

    # process global measurements
    _process_global_measurements(folder, bin_to_filenames, β, model_geometry, binned_sgn)

    # process local measurement
    _process_local_measurements(folder, bin_to_filenames, binned_sgn)

    # process correlation measurement
    _process_correlation_measurements(folder, bin_to_filenames, binned_sgn, Lτ, model_geometry, time_displaced)

    return nothing
end


@doc raw"""
    process_correlation_measurement(; folder::String, correlation::String, type::String, space::String, N_bin::Int)

Process a single correlation measurement using `N_bin` number of bins to calculate the error bars.
The argument `type` must be set to either `"equal-time"`, `"time-displaced"` or `"integrated"`,
and the argument `space` must be set to either `"position"` or `"momentum"`.
"""
function process_correlation_measurement(; folder::String, correlation::String, type::String, space::String, N_bin::Int)

    @assert (type == "equal-time") || (type == "time-displaced") || (type == "integrated")
    @assert (space == "momentum") || (space == "position")

    # get the number of processes that ran during simulation
    N_process = _get_num_processes(folder)

    # get the number of files in each measurement directory
    N_files = length(readdir(joinpath(folder,"global")))

    # get the number of measurements per process
    N_measurement = div(N_files, N_process)

    # calculate the bin size per process
    N_binsize = div(N_measurement, N_bin)

    # make sure the number of bins is valid
    @assert (N_bin * N_binsize) == N_measurement

    # get filenames associated with each bin
    bin_to_filenames = _get_filenames_by_bin(N_bin, N_binsize, N_process)

    # load model information
    β, Δτ, Lτ, model_geometry = _load_model_summary(folder)

    # get the binned sign data
    binned_sgn = _get_binned_sgn_data(folder, bin_to_filenames)

    if type == "time-displaced"

        # process time-displaced correlation function
        _process_displaced_correlation(correlation, joinpath(folder, type), space, bin_to_filenames, binned_sgn, Lτ, model_geometry)
    else

        # process equal-time or integrated correlation
        _process_equaltime_correlation(correlation, joinpath(folder, type), space, bin_to_filenames, binned_sgn, model_geometry, type)
    end

    return nothing
end


# load relevant model info
function _load_model_summary(folder::String)

    # load full model summary
    model_summary = TOML.parsefile(joinpath(folder, "model_summary.toml"))
    
    # get the inverse temperature
    β = model_summary["beta"]

    # get discretization in imaginary time
    Δτ = model_summary["dtau"]

    # get length of imaginary time axis
    Lτ = model_summary["L_tau"]

    # lattice geometry info table
    geometry_info = model_summary["geometry"]

    # construct unit cell
    unit_cell_info = geometry_info["unit_cell"]
    unit_cell = UnitCell(
        lattice_vecs = collect(unit_cell_info["lattice_vectors"][key] for key in keys(unit_cell_info["lattice_vectors"])),
        basis_vecs = collect(basis_vec_dict["r"] for basis_vec_dict in unit_cell_info["basis_vector"]),
    )

    # construct lattice
    lattice_info = geometry_info["lattice"]
    lattice = Lattice(
        L = lattice_info["L"],
        periodic = lattice_info["periodic"]
    )

    # initialize model geometry struct
    model_geometry = ModelGeometry(unit_cell, lattice)

    # add all bond definitions to model geometry
    for bond_info in geometry_info["bond"]

        # define bond
        bond = Bond(
            orbitals = bond_info["orbitals"],
            displacement = bond_info["displacement"]
        )

        # add bond to model geometry
        bond_id = add_bond!(model_geometry, bond)
    end

    return β, Δτ, Lτ, model_geometry
end


# get the number of processes
function _get_num_processes(folder)

    # initialize the number of processes to zero
    N_process = 0

    # iterate over files in global measurement directory
    for file in readdir(joinpath(folder,"global"))

        # split file name
        atoms = split(file, ('.','-','_'))

        # get process id
        pID = parse(Int, atoms[4])

        # record the max process id
        N_process = max(N_process, pID)
    end

    return N_process + 1
end


# get filenames associated with each bin
function _get_filenames_by_bin(N_bin::Int, N_binsize::Int, N_process::Int)

    # initialize vector to hold filenames associated with each bin
    bin_to_filenames = Vector{String}[]

    # iterate over bins
    for bin in 1:N_bin

        # ininitialize vector to contain filenames associated with each bin
        bin_filenames = String[]

        # iterate over bin size
        for n in (bin-1)*N_binsize + 1 : bin*N_binsize

            # iterate over number of processes
            for pID in 0:N_process-1

                # record filename
                filename = @sprintf "bin-%d_pID-%d.jld2" n pID
                push!(bin_filenames, filename)
            end
        end

        # record all filenames associated with current bin
        push!(bin_to_filenames, bin_filenames)
    end

    return bin_to_filenames
end


# get the binned sgn data
function _get_binned_sgn_data(folder::String, bin_to_filenames::Vector{Vector{String}})

    # global measurements folder
    global_folder = joinpath(folder, "global")

    # get number of bins
    N_bin = length(bin_to_filenames)

    # get the data type
    T = typeof( JLD2.load(joinpath(global_folder, "bin-1_pID-0.jld2"), "sgn") )

    # initialize vector to store average sign by bin
    binned_sgn = zeros(T, N_bin)

    # iterate over bins
    for bin in eachindex(bin_to_filenames)

        # get filenames associated with bin
        filenames = bin_to_filenames[bin]

        # iterate over measreuments in bin
        for n in eachindex(filenames)

            # get the sign
            binned_sgn[bin] += JLD2.load(joinpath(global_folder, filenames[n]), "sgn")
        end

        # normalize measurement
        binned_sgn[bin] /= length(filenames)
    end
    
    return binned_sgn
end


# process global measurements
function _process_global_measurements(folder::String, bin_to_filenames::Vector{Vector{String}},
                                     β::T, model_geometry::ModelGeometry{D, T, N},
                                     binned_sgn::Vector{Complex{T}}) where {D, T<:AbstractFloat, N}

    # get the number of sites/orbitals in the lattice
    unit_cell = model_geometry.unit_cell::UnitCell{D,T,N}
    lattice = model_geometry.lattice::Lattice{D}
    N_site = nsites(unit_cell, lattice)

    # get directory name for global measurements
    global_folder = joinpath(folder, "global")

    # get number of bins
    N_bin = length(bin_to_filenames)

    # initialize container to hold binned global measurements
    global_measurement = JLD2.load(joinpath(global_folder, bin_to_filenames[1][1]))
    binned_global_measurements = Dict(key => zeros(Complex{T}, N_bin) for key in keys(global_measurement))

    # iterate over bins
    for bin in eachindex(bin_to_filenames)

        # get filenames associated with bin
        filenames = bin_to_filenames[bin]

        # read in global measurement bin
        _global_measurement_bin(bin, global_folder, filenames, binned_global_measurements)
    end

    # calculate statics and write them to file
    open(joinpath(folder, "global_stats.csv"), "w") do fout

        # write header to file
        write(fout, "MEASUREMENT MEAN_R MEAN_I STD\n")

        # iterate over measurements
        for measurement in sort(collect(keys(binned_global_measurements)))

            # calculate mean and error for binned global measurement and write to file
            _write_global_measurement(fout, measurement, binned_global_measurements, binned_sgn)
        end

        # measure and record the compressibility using the fluctuation-disipation based relationship κ = β/N⋅(⟨N⟩²-⟨N²⟩)
        n = binned_global_measurements["density"]
        N² = binned_global_measurements["Nsqrd"]
        S = binned_sgn
        κ, Δκ = jackknife((n̄, N̄², S̄) -> (β/N_site)*(N̄²/S̄ - (N_site*n̄/S̄)^2), n, N², S)
        @printf(fout, "compressibility %.8f %.8f %.8f\n", real(κ), imag(κ), Δκ)
    end

    return nothing
end

# read in global measurement bin
function _global_measurement_bin(bin::Int, global_folder::String, filenames::Vector{String}, binned_global_measurements::Dict)

    # iterate over measreuments in bin
    for n in eachindex(filenames)

        # iterate over global measurements
        for measurement in keys(binned_global_measurements)

            binned_global_measurements[measurement][bin] += JLD2.load(joinpath(global_folder, filenames[n]), measurement)
        end
    end

    # normalize measurements
    for measurement in keys(binned_global_measurements)
        binned_global_measurements[measurement][bin] /= length(filenames)
    end

    return nothing
end

# calculate mean and error for binned global measurement and write to file
function _write_global_measurement(fout::IO, measurement::String, binned_global_measurements::Dict, binned_sgn::Vector{Complex{T}}) where {T<:AbstractFloat}

    # get the binned averaged
    binned_vals = binned_global_measurements[measurement]

    # if measurement of sign
    if startswith(measurement, "sgn") || measurement == "chemical_potential"

        # calculate mean and standard deviation of sign
        mean_sgn = mean(binned_vals)
        std_sgn = stdm(binned_vals, mean_sgn) / sqrt(length(binned_vals))

        # write sign stats to file
        @printf(fout, "%s %.8f %.8f %.8f\n", measurement, real(mean_sgn), imag(mean_sgn), std_sgn)

    # if standard measurement
    else

        # calculate mean and standard deviation
        mean_val, std_val = jackknife(/, binned_vals, binned_sgn)

        # write measurement stats to file
        @printf(fout, "%s %.8f %.8f %.8f\n", measurement, real(mean_val), imag(mean_val), std_val)
    end

    return nothing
end


# process local measurements
function _process_local_measurements(folder::String, bin_to_filenames::Vector{Vector{String}},
                                    binned_sgn::Vector{Complex{T}}) where {T<:AbstractFloat}

    # local measurements directory
    local_directory = joinpath(folder, "local")

    # number of bins
    N_bin = length(bin_to_filenames)

    # get template local measurement data
    local_measurements = JLD2.load(joinpath(local_directory, bin_to_filenames[1][1]))

    # construct container to binned local measurements
    binned_local_measurements = Dict{String, Matrix{Complex{T}}}()
    for measurement in keys(local_measurements)
        binned_local_measurements[measurement] = zeros(Complex{T}, N_bin, length(local_measurements[measurement]))
    end

    # iterate over bins
    for bin in eachindex(bin_to_filenames)

        # get filenames associated with bin
        filenames = bin_to_filenames[bin]

        # read in the current bin
        _read_local_measurement_bin(bin, local_directory, filenames, binned_local_measurements)
    end

    # calculate statistic and write them to file
    open(joinpath(folder, "local_stats.csv"), "w") do fout

        # write header
        write(fout, "MEASUREMENT ID_TYPE ID MEAN_R MEAN_I STD\n")

        # iterate over measurements
        for measurement in sort(collect(keys(binned_local_measurements)))

            # calculate mean and error for binned local measurements and write to file
            _write_local_measurement(fout, measurement, binned_local_measurements, binned_sgn)

        end
    end

    return nothing
end

# read local measurement bin
function _read_local_measurement_bin(bin::Int, local_directory::String, filenames::Vector{String}, binned_local_measurements::Dict)

    # iterate over measreuments in bin
    for n in eachindex(filenames)

        # load the data
        local_measurements = JLD2.load(joinpath(local_directory, filenames[n]))

        # iterate over measurements
        for measurement in keys(local_measurements)

            # iterate over ID's associated with measurement
            for id in eachindex(local_measurements[measurement])

                # record the measurement
                binned_local_measurements[measurement][bin, id] += local_measurements[measurement][id]
            end
        end
    end

    # normalize the binned measurements
    for measurement in keys(binned_local_measurements)
        for id in axes(binned_local_measurements[measurement], 2)
            binned_local_measurements[measurement][bin,id] /= length(filenames)
        end
    end

    return nothing
end

# calculate mean and error for binned local measurement and write to file
function _write_local_measurement(fout::IO, measurement::String, binned_local_measurements::Dict, binned_sgn::Vector{Complex{T}}) where {T<:AbstractFloat}

        # iterate over relevant ID's
        for id in axes(binned_local_measurements[measurement], 2)

            # get the binned measurement values
            binned_vals = @view binned_local_measurements[measurement][:,id]

            # calculate mean and standard deviation
            m, Δm = jackknife(/, binned_vals, binned_sgn)

            # write stats to file
            @printf(fout, "%s %s %d %.8f %.8f %.8f\n", measurement, LOCAL_MEASUREMENTS[measurement], id, real(m), imag(m), Δm)
        end

    return nothing
end


# process correlation measurements
function _process_correlation_measurements(folder::String, bin_to_filenames::Vector{Vector{String}}, binned_sgn::Vector{Complex{T}},
                                          Lτ::Int, model_geometry::ModelGeometry{D,T,N}, time_displaced::Bool) where {D, T<:AbstractFloat, N}

    # get directory for each type of correlation measurement
    equaltime_folder = joinpath(folder, "equal-time")
    time_displaced_folder = joinpath(folder, "time-displaced")
    integrated_folder = joinpath(folder, "integrated")

    # get each type of correlation measurement
    equaltime_correlations = filter(i -> isdir(joinpath(equaltime_folder,i)), readdir(equaltime_folder))
    time_displaced_correlations = filter(i -> isdir(joinpath(time_displaced_folder,i)), readdir(time_displaced_folder))
    integrated_correlations = filter(i -> isdir(joinpath(integrated_folder,i)), readdir(integrated_folder))

    # process equal-time correlation measurements
    for correlation in equaltime_correlations

        # process equal-time position space correlation measurement
        _process_equaltime_correlation(correlation, equaltime_folder, "position", bin_to_filenames, binned_sgn, model_geometry, "equal-time")

        # process equal-time momentum space correlation measurement
        _process_equaltime_correlation(correlation, equaltime_folder, "momentum", bin_to_filenames, binned_sgn, model_geometry, "equal-time")
    end

    # process time-dispalced correlation measurements
    if time_displaced
        for correlation in time_displaced_correlations

            # process integrated position space correlation measurement
            _process_displaced_correlation(correlation, time_displaced_folder, "position", bin_to_filenames, binned_sgn, Lτ, model_geometry)

            # process integrated momentum space correlation measurement
            _process_displaced_correlation(correlation, time_displaced_folder, "momentum", bin_to_filenames, binned_sgn, Lτ, model_geometry)
        end
    end

    # process integrated correlation measurements
    for correlation in integrated_correlations

        # process integrated position space correlation measurement
        _process_equaltime_correlation(correlation, integrated_folder, "position", bin_to_filenames, binned_sgn, model_geometry, "integrated")

        # process integrated momentum space correlation measurement
        _process_equaltime_correlation(correlation, integrated_folder, "momentum", bin_to_filenames, binned_sgn, model_geometry, "integrated")
    end

    return nothing
end

# process equal-time/integrated correlation function
function _process_equaltime_correlation(correlation::String, folder::String, space::String,
                                       bin_to_filenames::Vector{Vector{String}}, binned_sgn::Vector{Complex{T}},
                                       model_geometry::ModelGeometry{D,T,N}, correlation_type::String) where {D, T<:AbstractFloat, N}

    unit_cell = model_geometry.unit_cell::UnitCell{D,T,N}
    lattice = model_geometry.lattice::Lattice{D}

    # correlation folder
    correlation_folder = joinpath(folder, correlation)

    # momentum space or position space folder
    space_folder = joinpath(correlation_folder, space)

    # get correlation ID pairs
    pairs = JLD2.load(joinpath(space_folder, bin_to_filenames[1][1]), "id_pairs")

    # get number of pairs
    N_pair = length(pairs)

    # get size of lattice
    L = lattice.L

    # number of bins
    N_bin = length(bin_to_filenames)

    # define arrays to contain correlation bins
    correlation_bins = zeros(Complex{T}, N_bin, L...)

    # filename for correlation stats
    filename = @sprintf("%s_%s_%s_stats.csv", correlation, space, correlation_type)

    # open stats file
    open(joinpath(correlation_folder, filename), "w") do fout

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
        for n in 1:N_pair

            # process equal-time correlation pair
            index = _process_equaltime_correlation_pair(fout, space_folder, bin_to_filenames, correlation_bins, binned_sgn, pairs[n], n, index)
        end
    end

    return nothing
end

function _process_equaltime_correlation_pair(fout::IO, folder::String, bin_to_filenames::Vector{Vector{String}},
                                             correlation_bins::Array{Complex{T},P}, binned_sgn::Vector{Complex{T}},
                                             pair::NTuple{2,Int}, n_pair::Int, index::Int) where {T<:AbstractFloat, P}

    # number of bins
    N_bin = length(bin_to_filenames)

    # reset the correlation container
    fill!(correlation_bins, zero(Complex{T}))

    # iterate over bins
    for bin in 1:N_bin

        # read in the correlation bin data
        _read_equaltime_correlation_bin(bin, folder, bin_to_filenames, correlation_bins, n_pair)
    end

    # iterate over all displacements/k-points
    for c in CartesianIndices(selectdim(correlation_bins,1,1))

        # increment the index counter
        index += 1

        # get a view into the binned values
        binned_vals = @view correlation_bins[:, c]

        # calculate mean and std using jackknife
        C, ΔC = jackknife(/, binned_vals, binned_sgn)

        # write equal-time correlation stat to file
        _write_equaltime_correlation_stat(fout, index, pair, c.I, C, ΔC)
    end

    return index
end

# read in correlation bin for specified bin
function _read_equaltime_correlation_bin(bin::Int, folder::String, bin_to_filenames::Vector{Vector{String}},
                                         correlation_bins::Array{Complex{T},P}, n_pair::Int) where {T<:AbstractFloat, P}

    # get the filenames associated with the current bin
    filenames = bin_to_filenames[bin]

    # construct view into correlation_bins for current bin
    correlation_bin = selectdim(correlation_bins, 1, bin)

    # iterate over files in bin
    for i in eachindex(filenames)

        # read in correlation measurement
        correlations = JLD2.load(joinpath(folder, filenames[i]), "correlations")[n_pair]

        # add measurement to bin
        @. correlation_bin += correlations
    end
    
    # normalize correlation bin
    for i in eachindex(correlation_bin)
        correlation_bin[i] /= length(filenames)
    end

    return nothing
end

# write equal-time correlation stat to file for D=1 dimensional system
function _write_equaltime_correlation_stat(fout::IO, index::Int, pair::NTuple{2,Int}, n::NTuple{1,Int},
                                           C::Complex{T}, ΔC::T) where {T<:AbstractFloat}

    @printf(fout, "%d %d %d %d %.8f %.8f %.8f\n", index, pair[2], pair[1], n[1]-1, real(C), imag(C), ΔC)

    return nothing
end

# write equal-time correlation stat to file for D=2 dimensional system
function _write_equaltime_correlation_stat(fout::IO, index::Int, pair::NTuple{2,Int}, n::NTuple{2,Int},
                                           C::Complex{T}, ΔC::T) where {T<:AbstractFloat}

    @printf(fout, "%d %d %d %d %d %.8f %.8f %.8f\n", index, pair[2], pair[1], n[2]-1, n[1]-1, real(C), imag(C), ΔC)

    return nothing
end

# write equal-time correlation stat to file for D=3 dimensional system
function _write_equaltime_correlation_stat(fout::IO, index::Int, pair::NTuple{2,Int}, n::NTuple{3,Int},
                                           C::Complex{T}, ΔC::T) where {T<:AbstractFloat}

    @printf(fout, "%d %d %d %d %d %d %.8f %.8f %.8f\n", index, pair[2], pair[1], n[3]-1, n[2]-1, n[1]-1, real(C), imag(C), ΔC)

    return nothing
end


# process time-displaced correlation function
function _process_displaced_correlation(correlation::String, folder::String, space::String,
                                       bin_to_filenames::Vector{Vector{String}}, binned_sgn::Vector{Complex{T}},
                                       Lτ::Int, model_geometry::ModelGeometry{D,T,N}) where {D, T<:AbstractFloat, N}

    unit_cell = model_geometry.unit_cell::UnitCell{D,T,N}
    lattice = model_geometry.lattice::Lattice{D}

    # correlation folder
    correlation_folder = joinpath(folder, correlation)

    # momentum space or position space folder
    space_folder = joinpath(correlation_folder, space)

    # get correlation ID pairs
    pairs = JLD2.load(joinpath(space_folder, bin_to_filenames[1][1]), "id_pairs")

    # get number of pairs
    N_pair = length(pairs)

    # get size of lattice
    L = lattice.L

    # number of bins
    N_bin = length(bin_to_filenames)

    # define arrays to contain correlation bins
    correlation_bins = zeros(Complex{T}, N_bin, L...)

    # filename for correlation stats
    filename = @sprintf("%s_%s_time-displaced_stats.csv", correlation, space)

    # open stats file
    open(joinpath(correlation_folder, filename), "w") do fout

        # get id type
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
        for n in 1:N_pair

            # process equal-time correlation pair
            index = _process_displaced_correlation_pair(fout, space_folder, bin_to_filenames, correlation_bins, binned_sgn, pairs[n], n, Lτ, index)
        end
    end

    return nothing
end

function _process_displaced_correlation_pair(fout::IO, folder::String, bin_to_filenames::Vector{Vector{String}},
                                             correlation_bins::Array{Complex{T},P}, binned_sgn::Vector{Complex{T}},
                                             pair::NTuple{2,Int}, n_pair::Int, Lτ::Int, index::Int) where {T<:AbstractFloat, P}

    # iterate over possible imaginary time displacements
    for l in 1:Lτ+1

        index = _process_displaced_correlation_pair_slice(fout, folder, bin_to_filenames, correlation_bins, binned_sgn, pair, n_pair, l, index)
    end

    return index
end

function _process_displaced_correlation_pair_slice(fout::IO, folder::String, bin_to_filenames::Vector{Vector{String}},
                                                   correlation_bins::Array{Complex{T},P}, binned_sgn::Vector{Complex{T}},
                                                   pair::NTuple{2,Int}, n_pair::Int, l::Int, index::Int) where {T<:AbstractFloat, P}

    # number of bins
    N_bin = length(bin_to_filenames)

    # reset the correlation container
    fill!(correlation_bins, zero(Complex{T}))

    # iterate over bins
    for bin in 1:N_bin

        # read in the correlation bin data
        _read_displaced_correlation_bin(bin, folder, bin_to_filenames, correlation_bins, n_pair, l)
    end

    # iterate over all displacements/k-points
    for c in CartesianIndices(selectdim(correlation_bins,1,1))

        # increment the index counter
        index += 1

        # get a view into the binned values
        binned_vals = @view correlation_bins[:, c]

        # calculate mean and std using jackknife
        C, ΔC = jackknife(/, binned_vals, binned_sgn)

        # write equal-time correlation stat to file
        _write_displaced_correlation_stat(fout, index, pair, l, c.I, C, ΔC)
    end

    return index
end

# read in time-displaced correlation bin for specified bin and time slice
function _read_displaced_correlation_bin(bin::Int, folder::String, bin_to_filenames::Vector{Vector{String}},
                                         correlation_bins::Array{Complex{T},P}, n_pair::Int, l::Int) where {T<:AbstractFloat, P}

    # get the filenames associated with the current bin
    filenames = bin_to_filenames[bin]

    # construct view into correlation_bins for current bin
    correlation_bin = selectdim(correlation_bins, 1, bin)

    # iterate over files in bin
    for i in eachindex(filenames)

        # read in correlation measurement
        correlations = JLD2.load(joinpath(folder, filenames[i]), "correlations")[n_pair]

        # get correlation associated with appropriate imaginary time slice
        correlations_l = selectdim(correlations, ndims(correlations), l)

        # add measurement to bin
        @. correlation_bin += correlations_l
    end
    
    # normalize correlation bin
    for i in eachindex(correlation_bin)
        correlation_bin[i] /= length(filenames)
    end

    return nothing
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