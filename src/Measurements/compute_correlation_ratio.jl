@doc raw"""
    compute_correlation_ratio(
        comm::MPI.Comm;
        # KEYWORD ARGUMENTS
        folder::String,
        correlation::String,
        type::String,
        id_pairs::Vector{NTuple{2,Int}},
        id_pair_coefficients,
        q_point::NTuple{2,Int},
        q_neighbors::Vector{NTuple{2,Int}},
        num_bins::Int = 0,
        pIDs::Vector{Int} = Int[],
    ) where {D}

    compute_correlation_ratio(;
        # KEYWORD ARGUMENTS
        folder::String,
        correlation::String,
        type::String,
        id_pairs::Vector{NTuple{2,Int}},
        id_pair_coefficients,
        q_point::NTuple{D,Int},
        q_neighbors::Vector{NTuple{D,Int}},
        num_bins::Int = 0,
        pIDs::Vector{Int} = Int[],
    ) where {D}

Compute the correlation ratio at the ``\mathbf{k}``-point using a linear combination of standard correlation function measurements.
The linear combination of correlation functions used is defined by `id_pairs` and `coefs`.
If `type` is `"equal-time"` or `"time-displaced"` then the equal-time correlation ratio is calculated.
If `type` is "integrated" then the integrated correlation ratio is calculated.
"""
function compute_correlation_ratio(
    comm::MPI.Comm;
    # KEYWORD ARGUMENTS
    folder::String,
    correlation::String,
    type::String,
    id_pairs::Vector{NTuple{2,Int}},
    id_pair_coefficients::Vector{T},
    q_point::NTuple{D,Int},
    q_neighbors::Vector{NTuple{D,Int}},
    num_bins::Int = 0,
    pIDs::Vector{Int} = Int[],
) where {D, T<:Number}

    # determine relevant pIDs
    pIDs = isempty(pIDs) ? collect(0:MPI.Comm_size(comm)-1) : pIDs
    pID = pIDs[MPI.Comm_rank(comm) + 1]

    # compute correlation ratio
    R, ΔR = _compute_correlation_ratio(
        folder, correlation, type, id_pairs, id_pair_coefficients,
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
    folder::String,
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
            folder, correlation, type, id_pairs, id_pair_coeffients,
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
    folder::String,
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
    filename = joinpath(folder, @sprintf("bin_pID-%d.h5", pID))

    # uppercase type defintion
    Type = uppercase(type)

    # open HDF5 bin file
    H5File = h5open(filename, "r")

    # get the binned sign
    s_bins = bin_means(read(H5BinFile["GLOBAL/sgn"]), N_bins)

    # initialize vectors to contain structure factors
    Sq_bins = zeros(eltype(s), num_bins)
    Sqpdq_bins = zeros(eltype(s), num_bins)

    # number of neighboring wavevector to average over
    Ndq = length(q_neighbors)

    # open HDF5 
    Correlation = H5File["CORRELATIONS"]["STANDARD"][TYPE][correlation]

    # get dataset containing momentum data
    Momentum = Correlation["MOMENTUM"]

    # get dimensions of dataset
    dataset_dims = size(Momentum)

    # load all ID pairs
    all_id_pairs = read(Correlation, "ID_PAIRS")

    # get the index of each id pair
    id_pair_indices = collect(findfirst(p -> p == p′, all_id_pairs) for p′ in id_pairs)

    # iterate over ID pairs
    for i in eachindex(id_pairs)

        # get the current id pair
        id_pair = id_pairs[i]

        # get the coefficient associated with id pair
        coef = coefficients[i]

        # get the dataset index associated with ID pair
        indx = findfirst(p -> p == id_pair, all_id_pairs)

        # if time-displaced correlation measurement
        if Type == "TIME-DISPLACED"

            # record structure factor corresponding to ordering wave-vector
            Sq_bins += coef * bin_means(Momentum[:,(q_point[d]+1 for d in 1:D)...,1,indx], num_bins)

            # iterate over neighboring wave-vectors
            for n in 1:Ndq

                # get the nieghboring wave-vector
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

                # get the nieghboring wave-vector
                q_neighbor = q_neighbors[n]

                # record structure factor corresponding to ordering wave-vector
                Sqpdq_bins += coef/Ndq * bin_means(Momentum[:,(q_neighbor[d]+1 for d in 1:D)...,indx], num_bins)
            end
        end
    end

    # calculate correlation ratio
    # calculate composite correlation ratio
    R, ΔR = jackknife((Sqpdq, Sq, s) -> 1 - abs(Sqpdq/s)/abs(Sq/s), Sqpdq_bins, Sq_bins, s_bins)

    # close HDF5 file
    close(H5File)

    return R, ΔR
end


function compute_composite_correlation_ratio(
    comm::MPI.Comm;
    # KEYWORD ARGUMENTS
    folder::String,
    correlation::String,
    type::String,
    q_point::NTuple{D,Int},
    q_neighbors::Vector{NTuple{D,Int}},
    pIDs::Vector{Int} = Int[]
) where {D, T<:Number}

    # determine relevant pIDs
    pIDs = isempty(pIDs) ? collect(0:MPI.Comm_size(comm)-1) : pIDs
    pID = pIDs[MPI.Comm_rank(comm) + 1]

    # compute correlation ratio
    R′, ΔR′ = _compute_composite_correlation_ratio(
        folder, correlation, type, q_point, q_neighbors, num_bins, pID
    )
    varR = abs2(ΔR)

    # perform an all-reduce to get the average statics calculate on all MPI processes
    R    = MPI.Allreduce(R, +, comm)
    varR = MPI.Allreduce(varR, +, comm)
    R    = R / N_mpi
    ΔR   = sqrt(varR) / N_mpi

    return nothing
end


function compute_composite_correlation_ratio(;
    # Keyword Arguments
    folder::String,
    correlation::String,
    type::String,
    q_point::NTuple{D,Int},
    q_neighbors::Vector{NTuple{D,Int}},
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
        R′, ΔR′ = _compute_composite_correlation_ratio(
            folder, correlation, type, q_point, q_neighbors, num_bins, pID
        )

        # record statistics
        R += R′
        varR += abs2(ΔR′)
    end

    # calculate final stats
    N_pID = length(pIDs)
    R = R / N_pID
    ΔR = sqrt(varR)/ N_pID

    return nothing
end

# compute composite correlation ratio for single process
function _compute_composite_correlation_ratio(
    folder::String,
    correlation::String,
    type::String,
    q_point::NTuple{D,Int},
    q_neighbors::Vector{NTuple{D,Int}},
    num_bins::Int,
    pID::Int
) where {D, T<:Number}

    @assert(
        !in(correlation, keys(CORRELATION_FUNCTIONS)),
        "The $correlation correlation is a standard correlation function when it should be a composite correlation function."
    )

    # construct filename for HDF5 file containing binned data
    filename = joinpath(folder, @sprintf("bin_pID-%d.h5", pID))

    # uppercase type defintion
    Type = uppercase(type)

    # open HDF5 bin file
    H5File = h5open(filename, "r")

    # get the binned sign
    s_bins = bin_means(read(H5BinFile["GLOBAL/sgn"]), N_bins)

    # initialize vectors to contain structure factors
    Sq_bins = zeros(eltype(s), num_bins)
    Sqpdq_bins = zeros(eltype(s), num_bins)

    # number of neighboring wavevector to average over
    Ndq = length(q_neighbors)

    # open HDF5 
    Correlation = H5File["CORRELATIONS"]["COMPOSITE"][TYPE][correlation]

    # get dataset containing momentum data
    Momentum = Correlation["MOMENTUM"]

    # get dimensions of dataset
    dataset_dims = size(Momentum)


    # iterate over ID pairs
    for i in eachindex(id_pairs)

        # if time-displaced correlation measurement
        if Type == "TIME-DISPLACED"

            # record structure factor corresponding to ordering wave-vector
            Sq_bins += bin_means(Momentum[:,(q_point[d]+1 for d in 1:D)...,1], num_bins)

            # iterate over neighboring wave-vectors
            for n in 1:Ndq

                # get the nieghboring wave-vector
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

                # get the nieghboring wave-vector
                q_neighbor = q_neighbors[n]

                # record structure factor corresponding to ordering wave-vector
                Sqpdq_bins += bin_means(Momentum[:,(q_neighbor[d]+1 for d in 1:D)...], num_bins) / Ndq
            end
        end
    end

    # calculate correlation ratio
    # calculate composite correlation ratio
    R, ΔR = jackknife((Sqpdq, Sq, s) -> 1 - abs(Sqpdq/s)/abs(Sq/s), Sqpdq_bins, Sq_bins, s_bins)

    # close HDF5 file
    close(H5File)

    return R, ΔR
end