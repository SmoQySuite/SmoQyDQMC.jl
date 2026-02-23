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

Compute the correlation ratio at the ``\mathbf{q}``-point using a linear combination of standard correlation function measurements.
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

    # load the relevant data
    Sq, Sqpdq = _load_structure_factor_data(
        datafolder, correlation, type,
        id_pairs, id_pair_coefficients,
        q_point, q_neighbors,
        num_bins, pID
    )

    # gather the data across all pIDs
    Sq = MPI.gather(Sq, comm)
    Sqpdq = MPI.gather(Sqpdq, comm)

    # if root process
    if iszero(MPI.Comm_rank(comm))

        # concatenate all the data together
        Sq = vcat(Sq...)
        Sqpdq = vcat(Sqpdq...)

        # calculate correlation ratio
        R, ΔR = jackknife((Sqpdq, Sq) -> 1 - Sqpdq/Sq, Sqpdq, Sq, bias_corrected=false)
    else

        R, ΔR = nothing, nothing
    end

    # broadcast the number to all processes
    R = MPI.bcast(R, comm)
    ΔR = MPI.bcast(ΔR, comm)

    return R, ΔR
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

    # load the relevant data
    data = tuple((
        _load_structure_factor_data(
            datafolder, correlation, type,
            id_pairs, id_pair_coefficients,
            q_point, q_neighbors,
            num_bins, pID
        ) for pID in pIDs
    )...)

    # extract relevant structure factor data
    Sq = vcat((d[1] for d in data)...)
    Sqpdq = vcat((d[2] for d in data)...)

    # calculate correlation ratio
    R, ΔR = jackknife((Sqpdq, Sq) -> 1 - Sqpdq/Sq, Sqpdq, Sq, bias_corrected=false)

    return R, ΔR
end

# compute correlation ration for single process
function _load_structure_factor_data(
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
        "$correlation is not a defined STANDARD correlation function."
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

    # number of neighboring momentum to average over
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
            Sq_bins += coef * rebin(Momentum[:,(q_point[d]+1 for d in 1:D)...,1,indx], num_bins)

            # iterate over neighboring wave-vectors
            for n in 1:Ndq

                # get the neighboring wave-vector
                q_neighbor = q_neighbors[n]

                # record structure factor corresponding to ordering wave-vector
                Sqpdq_bins += coef/Ndq * rebin(Momentum[:,(q_neighbor[d]+1 for d in 1:D)...,1,indx], num_bins)
            end

        # if equal-time or integrated correlation measurement
        else

            # record structure factor corresponding to ordering wave-vector
            Sq_bins += coef * rebin(Momentum[:,(q_point[d]+1 for d in 1:D)...,indx], num_bins)

            # iterate over neighboring wave-vectors
            for n in 1:Ndq

                # get the neighboring wave-vector
                q_neighbor = q_neighbors[n]

                # record structure factor corresponding to ordering wave-vector
                Sqpdq_bins += coef/Ndq * rebin(Momentum[:,(q_neighbor[d]+1 for d in 1:D)...,indx], num_bins)
            end
        end
    end

    return Sq_bins, Sqpdq_bins
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

Compute the correlation ratio at a specified ``\mathbf{q}``-point for a measured composite correlation measurement.
If `type` is `"equal-time"` or `"time-displaced"` then the equal-time correlation ratio is calculated.
If `type` is "integrated" then the integrated correlation ratio is calculated.
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

    # load the relevant data
    Sq, Sqpdq = _load_composite_structure_factor_data(
        datafolder, name, type,
        q_point, q_neighbors,
        num_bins, pID
    )

    # gather the data across all pIDs
    Sq = MPI.Allgather(Sq, comm)
    Sqpdq = MPI.Allgather(Sqpdq, comm)

    # if root process
    if iszero(MPI.Comm_rank(comm))

        # concatenate all the data together
        Sq = vcat(Sq...)
        Sqpdq = vcat(Sqpdq...)

        # calculate correlation ratio
        R, ΔR = jackknife((Sqpdq, Sq) -> 1 - Sqpdq/Sq, Sqpdq, Sq, bias_corrected=false)
    else

        R, ΔR = nothing, nothing
    end

    # broadcast the number to all processes
    R = MPI.bcast(R, comm)
    ΔR = MPI.bcast(ΔR, comm)

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

    # load the relevant data
    data = tuple((
        _load_composite_structure_factor_data(
            datafolder, name, type,
            q_point, q_neighbors,
            num_bins, pID
        ) for pID in pIDs
    )...)

    # extract relevant structure factor data
    Sq = vcat((d[1] for d in data)...)
    Sqpdq = vcat((d[2] for d in data)...)

    # calculate correlation ratio
    R, ΔR = jackknife((Sqpdq, Sq) -> 1 - Sqpdq/Sq, Sqpdq, Sq, bias_corrected=false)

    return R, ΔR
end

# compute composite correlation ratio for single process
function _load_composite_structure_factor_data(
    datafolder::String,
    correlation::String,
    type::String,
    q_point::NTuple{D,Int},
    q_neighbors::Vector{NTuple{D,Int}},
    num_bins::Int,
    pID::Int
) where {D}

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
        Sq_bins += rebin(Momentum[:,(q_point[d]+1 for d in 1:D)...,1], num_bins)

        # iterate over neighboring wave-vectors
        for n in 1:Ndq

            # get the neighboring wave-vector
            q_neighbor = q_neighbors[n]

            # record structure factor corresponding to ordering wave-vector
            Sqpdq_bins += rebin(Momentum[:,(q_neighbor[d]+1 for d in 1:D)...,1], num_bins) / Ndq
        end

    # if equal-time or integrated correlation measurement
    else

        # record structure factor corresponding to ordering wave-vector
        Sq_bins += rebin(Momentum[:,(q_point[d]+1 for d in 1:D)...], num_bins)

        # iterate over neighboring wave-vectors
        for n in 1:Ndq

            # get the neighboring wave-vector
            q_neighbor = q_neighbors[n]

            # record structure factor corresponding to ordering wave-vector
            Sqpdq_bins += rebin(Momentum[:,(q_neighbor[d]+1 for d in 1:D)...], num_bins) / Ndq
        end
    end

    # close HDF5 file
    close(H5File)

    return Sq_bins, Sqpdq_bins
end