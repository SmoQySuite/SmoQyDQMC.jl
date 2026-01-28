@doc raw"""
    compute_correlation_ratio(
        comm::MPI.Comm;
        # KEYWORD ARGUMENTS
        datafolder::String,
        correlation::String,
        type::String,
        id_pairs::Vector{NTuple{2,Int}},
        id_pair_coefficients::Vector{T},
        q_point::NTuple{D,Int},
        q_neighbors::Vector{NTuple{D,Int}},
        num_bins::Int = 0,
        pIDs::Vector{Int} = Int[],
    ) where {D, T<:Number}

    compute_correlation_ratio(;
        # KEYWORD ARGUMENTS
        datafolder::String,
        correlation::String,
        type::String,
        id_pairs::Vector{NTuple{2,Int}},
        id_pair_coefficients::Vector{T},
        q_point::NTuple{D,Int},
        q_neighbors::Vector{NTuple{D,Int}},
        num_bins::Int = 0,
        pIDs::Union{Int,Vector{Int}} = Int[]
    ) where {D, T<:Number}

Compute the correlation ratio at the ``\mathbf{k}``-point using a linear combination of standard correlation function measurements.
The linear combination of correlation functions used is defined by `id_pairs` and `coefs`.
If `type` is `"equal-time"` or `"time-displaced"` then the equal-time correlation ratio is calculated.
If `type` is "integrated" then the integrated correlation ratio is calculated.
"""
function compute_correlation_ratio(
    comm::MPI.Comm;
    # KEYWORD ARGUMENTS
    datafolder::String,
    correlation::String,
    type::String,
    id_pairs::Vector{NTuple{2,Int}},
    id_pair_coefficients::Vector{T},
    q_point::NTuple{D,Int},
    q_neighbors::Vector{NTuple{D,Int}},
    num_bins::Int = 0,
    pIDs::Vector{Int} = Int[],
) where {D, T<:Number}

    # number of MPI processes
    N_mpi = MPI.Comm_size(comm)

    # determine relevant pIDs
    pIDs = isempty(pIDs) ? collect(0:N_mpi-1) : pIDs
    pID = pIDs[MPI.Comm_rank(comm) + 1]

    # compute correlation ratio
    R, ΔR = _compute_correlation_ratio(
        datafolder, correlation, type, id_pairs, id_pair_coefficients,
        q_point, q_neighbors, num_bins, pID
    )
    varR = abs2(ΔR)

    # perform an all-reduce to get the average statics calculate on all MPI processes
    R    = MPI.Allreduce(R, +, comm)
    varR = MPI.Allreduce(varR, +, comm)
    R    = R / N_mpi
    ΔR   = sqrt(varR) / N_mpi

    return R, real(ΔR)
end

# compute correlation ratio using single process
function compute_correlation_ratio(;
    # KEYWORD ARGUMENTS
    datafolder::String,
    correlation::String,
    type::String,
    id_pairs::Vector{NTuple{2,Int}},
    id_pair_coefficients::Vector{T},
    q_point::NTuple{D,Int},
    q_neighbors::Vector{NTuple{D,Int}},
    num_bins::Int = 0,
    pIDs::Union{Int,Vector{Int}} = Int[]
) where {D, T<:Number}

    # determine relevant process IDs
    pIDs = isa(pIDs, Int) ? [pIDs,] : pIDs
    if isempty(pIDs)
        pIDs = collect( 0 : length(readdir(joinpath(datafolder,"bins")))-1 )
    end

    # initialize correlation ratio mean and variance to zero
    R = zero(Complex{real(T)})
    varR = zero(real(T))

    # iterate over process IDs
    for pID in pIDs

        # compute correlation ratio for current process ID
        R′, ΔR′ = _compute_correlation_ratio(
            datafolder, correlation, type, id_pairs, id_pair_coefficients,
            q_point, q_neighbors, num_bins, pID
        )

        # record statistics
        R += R′
        varR += abs2(ΔR′)
    end

    # calculate final stats
    N_pID = length(pIDs)
    R = R / N_pID
    ΔR = sqrt(varR)/ N_pID

    return R, ΔR
end

# compute correlation ration for single process
function _compute_correlation_ratio(
    datafolder::String,
    correlation::String,
    type::String,
    id_pairs::Vector{NTuple{2,Int}},
    id_pair_coefficients::Vector{T},
    q_point::NTuple{D,Int},
    q_neighbors::Vector{NTuple{D,Int}},
    num_bins::Int,
    pID::Int
) where {D, T<:Number}

    @assert(
        in(correlation, keys(CORRELATION_FUNCTIONS)),
        "The $correlation correlation is a composite correlation function when it should be a standard correlation function."
    )

    # construct filename for HDF5 file containing binned data
    filename = joinpath(datafolder, "bins", @sprintf("bins_pID-%d.h5", pID))

    # uppercase type defintion
    Type = uppercase(type)

    # open HDF5 bin file
    H5File = h5open(filename, "r")

    # get all the sgn data
    sgn = read(H5File["GLOBAL/sgn"])

    # set number of bins if necessary
    num_bins = iszero(num_bins) ? length(sgn) : num_bins
    @assert length(sgn) % num_bins == 0

    # initialize vectors to contain structure factors
    Sq_bins = zeros(eltype(sgn), num_bins)
    Sqpdq_bins = zeros(eltype(sgn), num_bins)

    # number of neighboring wavevector to average over
    Ndq = length(q_neighbors)

    # open HDF5 
    Correlation = H5File["CORRELATIONS"]["STANDARD"][Type][correlation]

    # get dataset containing momentum data
    Momentum = Correlation["MOMENTUM"]

    # load all ID pairs
    all_id_pairs = map(p -> tuple(p...), read_attribute(Correlation, "ID_PAIRS"))

    # iterate over ID pairs
    for i in eachindex(id_pairs)

        # get the current id pair
        id_pair = id_pairs[i]

        # get the coefficient associated with id pair
        coef = id_pair_coefficients[i]

        # get the dataset index associated with ID pair
        indx = findfirst(p -> p == id_pair, all_id_pairs)

        # if time-displaced correlation measurement
        if Type == "TIME-DISPLACED"

            # record structure factor corresponding to ordering wave-vector
            Sq_bins += coef * bin_means(Momentum[:,(q_point[d]+1 for d in 1:D)...,1,indx], num_bins)

            # iterate over neighboring wave-vectors
            for n in 1:Ndq

                # get the neighboring wave-vector
                q_neighbor = q_neighbors[n]

                # record structure factor corresponding to ordering wave-vector
                Sqpdq_bins += coef/Ndq * bin_means(Momentum[:,(q_neighbor[d]+1 for d in 1:D)...,1,indx], num_bins)
            end

        # if equal-time or integrated correlation measurement
        else

            # record structure factor corresponding to ordering wave-vector
            Sq_bins += coef * bin_means(Momentum[:,(q_point[d]+1 for d in 1:D)...,indx], num_bins)

            # iterate over neighboring wave-vectors
            for n in 1:Ndq

                # get the neighboring wave-vector
                q_neighbor = q_neighbors[n]

                # record structure factor corresponding to ordering wave-vector
                Sqpdq_bins += coef/Ndq * bin_means(Momentum[:,(q_neighbor[d]+1 for d in 1:D)...,indx], num_bins)
            end
        end
    end

    # calculate correlation ratio
    R, ΔR = jackknife((Sqpdq, Sq) -> 1 - Sqpdq/Sq, Sqpdq_bins, Sq_bins, bias_corrected=false)

    # close HDF5 file
    close(H5File)

    return R, ΔR
end


@doc raw"""
    compute_composite_correlation_ratio(
        comm::MPI.Comm;
        # KEYWORD ARGUMENTS
        datafolder::String,
        name::String,
        type::String,
        q_point::NTuple{D,Int},
        q_neighbors::Vector{NTuple{D,Int}},
        pIDs::Vector{Int} = Int[]
    ) where {D}

    compute_composite_correlation_ratio(;
        # Keyword Arguments
        datafolder::String,
        name::String,
        type::String,
        q_point::NTuple{D,Int},
        q_neighbors::Vector{NTuple{D,Int}},
        num_bins::Int = 0,
        pIDs::Union{Int,Vector{Int}} = Int[]
    ) where {D}


"""
function compute_composite_correlation_ratio(
    comm::MPI.Comm;
    # KEYWORD ARGUMENTS
    datafolder::String,
    name::String,
    type::String,
    q_point::NTuple{D,Int},
    q_neighbors::Vector{NTuple{D,Int}},
    num_bins::Int = 0,
    pIDs::Vector{Int} = Int[]
) where {D}

    # number of MPI processes
    N_mpi = MPI.Comm_size(comm)

    # determine relevant pIDs
    pIDs = isempty(pIDs) ? collect(0:N_mpi-1) : pIDs
    pID = pIDs[MPI.Comm_rank(comm) + 1]

    # compute correlation ratio
    R, ΔR = _compute_composite_correlation_ratio(
        datafolder, name, type, q_point, q_neighbors, num_bins, pID
    )
    varR = abs2(ΔR)

    # perform an all-reduce to get the average statics calculate on all MPI processes
    R    = MPI.Allreduce(R, +, comm)
    varR = MPI.Allreduce(varR, +, comm)
    R    = R / N_mpi
    ΔR   = sqrt(varR) / N_mpi

    return R, ΔR
end


function compute_composite_correlation_ratio(;
    # Keyword Arguments
    datafolder::String,
    name::String,
    type::String,
    q_point::NTuple{D,Int},
    q_neighbors::Vector{NTuple{D,Int}},
    num_bins::Int = 0,
    pIDs::Union{Int,Vector{Int}} = Int[]
) where {D}

    # determine relevant process IDs
    pIDs = isa(pIDs, Int) ? [pIDs,] : pIDs
    if isempty(pIDs)
        pIDs = collect( 0 : length(readdir(joinpath(datafolder,"bins")))-1 )
    end

    # initialize correlation ratio mean and variance to zero
    R = complex(0.0)
    varR = 0.0

    # iterate over process IDs
    for pID in pIDs

        # compute correlation ratio for current process ID
        R′, ΔR′ = _compute_composite_correlation_ratio(
            datafolder, name, type, q_point, q_neighbors, num_bins, pID
        )

        # record statistics
        R += R′
        varR += abs2(ΔR′)
    end

    # calculate final stats
    N_pID = length(pIDs)
    R = R / N_pID
    ΔR = sqrt(varR)/ N_pID

    return R, ΔR
end

# compute composite correlation ratio for single process
function _compute_composite_correlation_ratio(
    datafolder::String,
    correlation::String,
    type::String,
    q_point::NTuple{D,Int},
    q_neighbors::Vector{NTuple{D,Int}},
    num_bins::Int,
    pID::Int
) where {D}

    @assert(
        !in(correlation, keys(CORRELATION_FUNCTIONS)),
        "The $correlation correlation is a standard correlation function when it should be a named composite correlation function."
    )

    # construct filename for HDF5 file containing binned data
    filename = joinpath(datafolder, "bins", @sprintf("bins_pID-%d.h5", pID))

    # uppercase type definition
    Type = uppercase(type)

    # open HDF5 bin file
    H5File = h5open(filename, "r")

    # get all the sgn data
    sgn = read(H5File["GLOBAL/sgn"])

    # set number of bins if necessary
    num_bins = iszero(num_bins) ? length(sgn) : num_bins
    @assert length(sgn) % num_bins == 0

    # initialize vectors to contain structure factors
    Sq_bins = zeros(eltype(sgn), num_bins)
    Sqpdq_bins = zeros(eltype(sgn), num_bins)

    # number of neighboring wave-vector to average over
    Ndq = length(q_neighbors)

    # open HDF5 
    Correlation = H5File["CORRELATIONS"]["COMPOSITE"][Type][correlation]

    # get dataset containing momentum data
    Momentum = Correlation["MOMENTUM"]

    # if time-displaced correlation measurement
    if Type == "TIME-DISPLACED"

        # record structure factor corresponding to ordering wave-vector
        Sq_bins += bin_means(Momentum[:,(q_point[d]+1 for d in 1:D)...,1], num_bins)

        # iterate over neighboring wave-vectors
        for n in 1:Ndq

            # get the neighboring wave-vector
            q_neighbor = q_neighbors[n]

            # record structure factor corresponding to ordering wave-vector
            Sqpdq_bins += bin_means(Momentum[:,(q_neighbor[d]+1 for d in 1:D)...,1], num_bins) / Ndq
        end

    # if equal-time or integrated correlation measurement
    else

        # record structure factor corresponding to ordering wave-vector
        Sq_bins += bin_means(Momentum[:,(q_point[d]+1 for d in 1:D)...], num_bins)

        # iterate over neighboring wave-vectors
        for n in 1:Ndq

            # get the neighboring wave-vector
            q_neighbor = q_neighbors[n]

            # record structure factor corresponding to ordering wave-vector
            Sqpdq_bins += bin_means(Momentum[:,(q_neighbor[d]+1 for d in 1:D)...], num_bins) / Ndq
        end
    end

    # calculate composite correlation ratio
    R, ΔR = jackknife((Sqpdq, Sq) -> 1 - Sqpdq/Sq, Sqpdq_bins, Sq_bins, bias_corrected=false)

    # close HDF5 file
    close(H5File)

    return R, ΔR
end