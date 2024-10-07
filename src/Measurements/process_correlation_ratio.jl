@doc raw"""
    compute_correlation_ratio(
        comm::MPI.Comm;
        folder::String,
        correlation::String,
        type::String,
        ids,
        id_pairs::Vector{NTuple{2,Int}} = NTuple{2,Int}[],
        coefs,
        k_point,
        num_bins::Int = 0,
        pIDs::Vector{Int} = Int[],
    )

    compute_correlation_ratio(;
        # Keyword Arguments
        folder::String,
        correlation::String,
        type::String,
        ids,
        id_pairs::Vector{NTuple{2,Int}} = NTuple{2,Int}[],
        coefs,
        k_point,
        num_bins::Int = 0,
        pIDs::Vector{Int} = Int[],
    )

Compute the correlation ratio at the ``\mathbf{k}``-point using a linear combination of standard correlation function measurements.
The linear combination of correlation functions used is defined by `id_pairs` and `coefs`. If `id_pairs` is empty,
then all possible combinations of ID pairs are construct based passed `ids`, with coefficients similarly expanded out.
If `type` is `"equal-time"` or `"time-displaced"` then the equal-time correlation ratio is calculated.
If `type` is "integrated" then the integrated correlation ratio is calculated.
"""
function compute_correlation_ratio(
    comm::MPI.Comm;
    # Keyword Arguments
    folder::String,
    correlation::String,
    type::String,
    ids,
    id_pairs::Vector{NTuple{2,Int}} = NTuple{2,Int}[],
    coefs,
    k_point,
    num_bins::Int = 0,
    pIDs::Vector{Int} = Int[],
)
    @assert type ∈ ("equal-time", "time-displaced", "integrated")
    @assert correlation ∈ keys(CORRELATION_FUNCTIONS)

    # set the walkers to iterate over
    if isempty(pIDs)

        # get the number of MPI walkers
        N_walkers = get_num_walkers(folder)

        # get the pIDs
        pIDs = collect(0:(N_walkers-1))
    end

    # get dimension and size of lattice
    β, Δτ, Lτ, model_geometry = load_model_summary(folder)
    lattice = model_geometry.lattice
    L = lattice.L # size of lattice
    D = ndims(lattice) # dimension of lattice

    # calculate relevent k-points
    @assert length(k_point) == ndims(lattice)

    # construct id pairs if not given
    if isempty(id_pairs)
        @assert length(ids) == length(coefs)
        coefficients = Complex{real(eltype(coefs))}[]
        for j in eachindex(ids)
            for i in eachindex(ids)
                push!(id_pairs, (ids[j],ids[i]))
                push!(coefficients, conj(coefs[i]) * coefs[j])
            end
        end
    else
        @assert length(coefs) == length(id_pairs)
        coefficients = Complex{real(eltype(coefs))}[coefs...]
    end

    # get central k-point and 2⋅D neighboring k-points
    k = tuple(k_point...)
    k_neighbors = Vector{NTuple{D,Int}}(undef, 2*D)
    for d in 1:D
        k_neighbors[2*d-1] = tuple(( mod(k[d′]+isequal(d′,d),L[d]) for d′ in eachindex(k) )...)
        k_neighbors[2*d]   = tuple(( mod(k[d′]-isequal(d′,d),L[d]) for d′ in eachindex(k) )...)
    end

    # get the MPI rank
    mpiID = MPI.Comm_rank(comm)

    # get the number of MPI ranks
    N_mpi = MPI.Comm_size(comm)
    @assert length(pIDs) == N_mpi

    # get the pID corresponding to mpiID
    pID = pIDs[mpiID+1]

    # compute correlation ratio for current pID
    R, ΔR = _compute_correlation_ratio(folder, correlation, type, id_pairs, coefficients, k, k_neighbors, num_bins, pID)
    varR = ΔR^2

    # perform an all-reduce to get the average statics calculate on all MPI processes
    R    = MPI.Allreduce(R, +, comm)
    varR = MPI.Allreduce(varR, +, comm)
    R    = R / N_mpi
    ΔR   = sqrt(varR) / N_mpi

    return R, ΔR
end

# compute correlation ratio using single process
function compute_correlation_ratio(;
    # Keyword Arguments
    folder::String,
    correlation::String,
    type::String,
    ids,
    id_pairs::Vector{NTuple{2,Int}} = NTuple{2,Int}[],
    coefs,
    k_point,
    num_bins::Int = 0,
    pIDs::Vector{Int} = Int[],
)
    @assert type ∈ ("equal-time", "time-displaced", "integrated")
    @assert correlation ∈ keys(CORRELATION_FUNCTIONS)

    # set the walkers to iterate over
    if isempty(pIDs)

        # get the number of MPI walkers
        N_walkers = get_num_walkers(folder)

        # get the pIDs
        pIDs = collect(0:(N_walkers-1))
    end

    # get dimension and size of lattice
    β, Δτ, Lτ, model_geometry = load_model_summary(folder)
    lattice = model_geometry.lattice
    L = lattice.L # size of lattice
    D = ndims(lattice) # dimension of lattice

    # calculate relevent k-points
    @assert length(k_point) == ndims(lattice)

    # construct id pairs if not given
    if isempty(id_pairs)
        @assert length(ids) == length(coefs)
        coefficients = Complex{real(eltype(coefs))}[]
        for j in eachindex(ids)
            for i in eachindex(ids)
                push!(id_pairs, (ids[j],ids[i]))
                push!(coefficients, conj(coefs[i]) * coefs[j])
            end
        end
    else
        @assert length(coefs) == length(id_pairs)
        coefficients = Complex{real(eltype(coefs))}[coefs...]
    end

    # get central k-point and 2⋅D neighboring k-points
    k = tuple(k_point...)
    k_neighbors = NTuple{D,Int}[]
    for d in 1:D
        push!(k_neighbors, tuple(( mod(k[d′]+isequal(d′,d),L[d]) for d′ in eachindex(k) )...))
        push!(k_neighbors, tuple(( mod(k[d′]-isequal(d′,d),L[d]) for d′ in eachindex(k) )...))
    end

    # initialize mean and variance of correlation ratio to zero
    R, varR = zero(β), zero(β)

    # iterate over pIDs
    for i in eachindex(pIDs)
        # calcuate correlation ratio for current PID
        Ri, ΔRi = _compute_correlation_ratio(folder, correlation, type, id_pairs, coefficients, k, k_neighbors, num_bins, pIDs[i])
        R += Ri
        varR += abs2(ΔRi)
    end
    # normalize the final measurements, computing the final standard deviation
    R /= length(pIDs)
    ΔR = sqrt(varR) / length(pIDs)

    return R, ΔR
end

# compute the correlation ratio for a single walker (single pID)
function _compute_correlation_ratio(
    folder::String,
    correlation::String,
    type::String,
    id_pairs::Vector{NTuple{2,Int}},
    coefficients::Vector{Complex{T}},
    k_point::NTuple{D,Int},
    kneighbors::Vector{NTuple{D,Int}},
    num_bins::Int,
    pID::Int
) where {D, T<:AbstractFloat}

    # construct directory name containing binary data
    dir = joinpath(folder, type, correlation, "momentum")

    # get the number of files/measurements
    N_files = length(glob(@sprintf("*_pID-%d.jld2", pID), dir))

    # defaults num_bins to N_files if num_bins is zero
    num_bins = iszero(num_bins) ? N_files : num_bins

    # get bin intervals
    bin_intervals = get_bin_intervals(folder, num_bins, pID)

    # get the binned average sign
    binned_sign = get_average_sign(folder, bin_intervals, pID)

    # binned structure factor at k-point
    S_k_point_bins = zeros(Complex{T}, num_bins)

    # binned average structure factor of neighboring k-points
    S_kneighbors_bins = zeros(Complex{T}, num_bins)

    # whether to load time-displaced data or not
    Δl = isequal(type, "time-displaced") ? 0 : -1

    # iterate over ID pairs
    for i in eachindex(id_pairs)

        # load k-point data
        _load_binned_correlation!(S_k_point_bins, bin_intervals, dir, id_pairs[i], k_point, Δl, pID, coefficients[i])

        # iterate over neighboring k-points
        for n in eachindex(kneighbors)
            # load neighboring k-point data
            _load_binned_correlation!(S_kneighbors_bins, bin_intervals, dir, id_pairs[i], kneighbors[n], Δl, pID, coefficients[i]/(2*D))
        end
    end

    # calculate correlation ratio
    R, ΔR = jackknife((Skpdq, Sk, s) -> 1 - abs(Skpdq/s)/abs(Sk/s), S_kneighbors_bins, S_k_point_bins, binned_sign)

    return R, ΔR
end


@doc raw"""
    compute_composite_correlation_ratio(
        comm::MPI.Comm;
        # Keyword Arguments
        folder::String,
        correlation::String,
        type::String,
        k_point,
        num_bins::Int = 0,
        pIDs::Vector{Int} = Int[],
    )

    compute_composite_correlation_ratio(;
        # Keyword Arguments
        folder::String,
        correlation::String,
        type::String,
        k_point,
        num_bins::Int = 0,
        pIDs::Vector{Int} = Int[],
    )

Compute the correlation ratio for a specified ``\mathbf{k}``-point for the specified composite `correlation` function.
If `type` is `"equal-time"` or `"time-displaced"` then the equal-time correlation ratio is calculated.
If `type` is "integrated" then the integrated correlation ratio is calculated.
"""
function compute_composite_correlation_ratio(
    comm::MPI.Comm;
    # Keyword Arguments
    folder::String,
    correlation::String,
    type::String,
    k_point,
    num_bins::Int = 0,
    pIDs::Vector{Int} = Int[],
)
    @assert type ∈ ("equal-time", "time-displaced", "integrated")
    @assert correlation ∉ keys(CORRELATION_FUNCTIONS)

    # set the walkers to iterate over
    if isempty(pIDs)

        # get the number of MPI walkers
        N_walkers = get_num_walkers(folder)

        # get the pIDs
        pIDs = collect(0:(N_walkers-1))
    end

    # get dimension and size of lattice
    β, Δτ, Lτ, model_geometry = load_model_summary(folder)
    lattice = model_geometry.lattice
    L = lattice.L # size of lattice
    D = ndims(lattice) # dimension of lattice

    # calculate relevent k-points
    @assert length(k_point) == ndims(lattice)

    # get central k-point and 2⋅D neighboring k-points
    k = tuple(k_point...)
    k_neighbors = NTuple{D,Int}[]
    for d in 1:D
        push!(k_neighbors, tuple(( mod(k[d′]+isequal(d′,d),L[d]) for d′ in eachindex(k) )...))
        push!(k_neighbors, tuple(( mod(k[d′]-isequal(d′,d),L[d]) for d′ in eachindex(k) )...))
    end

    # get the MPI rank
    mpiID = MPI.Comm_rank(comm)

    # get the number of MPI ranks
    @assert length(pIDs) == MPI.Comm_size(comm)
    N_mpi = MPI.Comm_size(comm)

    # get the pID corresponding to mpiID
    pID = pIDs[mpiID+1]

    # compute correlation ratio for current pID
    R, ΔR = _compute_composite_correlation_ratio(folder, correlation, type, k, k_neighbors, num_bins, pID)
    varR = ΔR^2

    # perform an all-reduce to get the average statics calculate on all MPI processes
    R    = MPI.Allreduce(R, +, comm)
    varR = MPI.Allreduce(varR, +, comm)
    R    = R / N_mpi
    ΔR   = sqrt(varR) / N_mpi

    return R, ΔR
end

# compute correlation ratio using single process
function compute_composite_correlation_ratio(;
    # Keyword Arguments
    folder::String,
    correlation::String,
    type::String,
    k_point,
    num_bins::Int = 0,
    pIDs::Vector{Int} = Int[],
)
    @assert type ∈ ("equal-time", "time-displaced", "integrated")
    @assert correlation ∉ keys(CORRELATION_FUNCTIONS)

    # set the walkers to iterate over
    if isempty(pIDs)

        # get the number of MPI walkers
        N_walkers = get_num_walkers(folder)

        # get the pIDs
        pIDs = collect(0:(N_walkers-1))
    end

    # get dimension and size of lattice
    β, Δτ, Lτ, model_geometry = load_model_summary(folder)
    lattice = model_geometry.lattice
    L = lattice.L # size of lattice
    D = ndims(lattice) # dimension of lattice

    # calculate relevent k-points
    @assert length(k_point) == ndims(lattice)

    # get central k-point and 2⋅D neighboring k-points
    k = tuple(k_point...)
    k_neighbors = NTuple{D,Int}[]
    for d in 1:D
        push!(k_neighbors, tuple(( mod(k[d′]+isequal(d′,d),L[d]) for d′ in eachindex(k) )...))
        push!(k_neighbors, tuple(( mod(k[d′]-isequal(d′,d),L[d]) for d′ in eachindex(k) )...))
    end

    # initialize mean and variance of correlation ratio to zero
    R, varR = zero(β), zero(β)

    # iterate over pIDs
    for i in eachindex(pIDs)
        # calcuate correlation ratio for current PID
        Ri, ΔRi = _compute_composite_correlation_ratio(folder, correlation, type, k, k_neighbors, num_bins, pIDs[i])
        R += Ri
        varR += abs2(ΔRi)
    end
    # normalize the final measurements, computing the final standard deviation
    R /= length(pIDs)
    ΔR = sqrt(varR) / length(pIDs)

    return R, ΔR
end


# compute the correlation ratio for a single walker (single pID)
function _compute_composite_correlation_ratio(
    folder::String,
    correlation::String,
    type::String,
    k_point::NTuple{D,Int},
    kneighbors::Vector{NTuple{D,Int}},
    num_bins::Int,
    pID::Int
) where {D}

    # construct directory name containing binary data
    dir = joinpath(folder, type, correlation, "momentum")

    # get the number of files/measurements
    N_files = length(glob(@sprintf("*_pID-%d.jld2", pID), dir))

    # defaults num_bins to N_files if num_bins is zero
    num_bins = iszero(num_bins) ? N_files : num_bins

    # get bin intervals
    bin_intervals = get_bin_intervals(folder, num_bins, pID)

    # get the binned average sign
    binned_sign = get_average_sign(folder, bin_intervals, pID)

    # get data type
    T = real(eltype(binned_sign))

    # binned structure factor at k-point
    S_k_point_bins = zeros(Complex{T}, num_bins)

    # binned average structure factor of neighboring k-points
    S_kneighbors_bins = zeros(Complex{T}, num_bins)

    # whether to load time-displaced data or not
    Δl = isequal(type, "time-displaced") ? 0 : -1

    # load k-point data
    _load_binned_composite_correlation!(S_k_point_bins, bin_intervals, dir, k_point, Δl, pID, 1.0)

    # iterate over neighboring k-points
    for n in eachindex(kneighbors)
        # load neighboring k-point data
        _load_binned_composite_correlation!(S_kneighbors_bins, bin_intervals, dir, kneighbors[n], Δl, pID, inv(2*D))
    end

    # calculate composite correlation ratio
    R, ΔR = jackknife((Skpdq, Sk, s) -> 1 - abs(Skpdq/s)/abs(Sk/s), S_kneighbors_bins, S_k_point_bins, binned_sign)

    return R, ΔR
end


# load binned equal-time or integrated composite correlation data
function _load_binned_composite_correlation!(
    binned_vals::AbstractVector{Complex{T}},
    bin_intervals::Vector{UnitRange{Int}},
    dir::String,
    loc::NTuple{D,Int},
    Δl::Int,
    pID::Int,
    coef = one(Complex{T})
) where {D, T<:AbstractFloat}

    # number of bins
    num_bins = length(bin_intervals)

    # calculate the bin size
    bin_size = length(bin_intervals[1])

    # iterate over bins
    for bin in 1:num_bins
        # iterate over bin elements
        for i in bin_intervals[bin]
            # construct filename
            file = joinpath(dir, @sprintf("bin-%d_pID-%d.jld2", i, pID))
            # load correlation data
            corr = OffsetArrays.Origin(0)(JLD2.load(file, "correlations"))
            # record the relevant correlations
            if Δl < 0
                binned_vals[bin] += coef * corr[loc...]
            else
                binned_vals[bin] += coef * corr[loc..., Δl]
            end
        end
        # normalize binned data
        binned_vals[bin] /= bin_size
    end

    return nothing
end