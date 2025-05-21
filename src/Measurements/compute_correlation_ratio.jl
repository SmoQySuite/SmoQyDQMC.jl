@doc raw"""
    compute_correlation_ratio(
        comm::MPI.Comm;
        folder::String,
        correlation::String,
        type::String,
        id_pairs::Vector{NTuple{2,Int}},
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
        id_pairs::Vector{NTuple{2,Int}},
        coefs,
        k_point,
        num_bins::Int = 0,
        pIDs::Vector{Int} = Int[],
    )

Compute the correlation ratio at the ``\mathbf{k}``-point using a linear combination of standard correlation function measurements.
The linear combination of correlation functions used is defined by `id_pairs` and `coefs`.
If `type` is `"equal-time"` or `"time-displaced"` then the equal-time correlation ratio is calculated.
If `type` is "integrated" then the integrated correlation ratio is calculated.
"""
function compute_correlation_ratio(
    comm::MPI.Comm;
    # Keyword Arguments
    folder::String,
    correlation::String,
    type::String,
    id_pairs::Vector{NTuple{2,Int}},
    coefs,
    k_point,
    num_bins::Int = 0,
    pIDs::Vector{Int} = Int[],
)

    return R, real(ΔR)
end

# compute correlation ratio using single process
function compute_correlation_ratio(;
    # Keyword Arguments
    folder::String,
    correlation::String,
    type::String,
    id_pairs::Vector{NTuple{2,Int}},
    coefs,
    k_point,
    num_bins::Int = 0,
    pIDs::Vector{Int} = Int[],
)



    return R, real(ΔR)
end

function _compute_correlation_ratio(
    folder::String,
    correlation::String,
    type::String,
    id_pairs::Vector{NTuple{2,Int}},
    coefficients::Vector{Complex{T}},
    k_point::NTuple{D,Int},
    num_bins::Int,
    pID::Int
) where {D, T<:AbstractFloat}

    # construct filename for HDF5 file containing binned data
    filename = joinpath(folder, @sprintf("bin_pID-%d.h5", pID))

    # open HDF5 bin file
    H5File = h5open(filename, "r")

    # open HDF5 
    Correlation = H5File["CORRELATIONS"]["STANDARD"][uppercase(type)][correlation]

    # get dataset containing momentum data
    Momentum = Correlation["MOMENTUM"]

    # get dimensions of dataset
    dataset_dims = size(Momentum)

    # get dimensions of lattice
    L = dataset_dims[2:D+2]

    # load all ID pairs
    all_id_pairs = read(Correlation, "ID_PAIRS")

    # close HDF5 file
    close(H5File)

    return nothing
end