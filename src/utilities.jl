# Swap the contents of the two arrays `a` and `b`.
function swap!(a::AbstractArray{T}, b::AbstractArray{T}) where {T}

    @fastmath @inbounds for i in eachindex(a)
        tmp = a[i]
        a[i] = b[i]
        b[i] = tmp
    end

    return nothing
end

# if x = 0 then return 1, otherwise just return sign(x)
sign_or_0to1(x::T) where {T<:Number} = iszero(x) ? one(T) : sign(x)

# default bosonic action evaluation method
bosonic_action(some_model_parameters) = 0.0

# sample two unique random numbers from the range 1:N
function draw2(rng, N)
    a = rand(rng, 1:N)
    b = rand(rng, 1:N-1)
    b += (b >= a)
    return a, b
end

# detects a not finite number
@inline notfinite(x::Number, thresh = Inf) = ( (!isfinite(x)) || abs(x) > thresh )

# rebin a data array along a specified array dimension
function rebin(
    data::AbstractArray{T},
    N_bins::Int,
    dim::Int = 1
) where {T<:Number}

    # Get the length of the dimension to be rebinned
    N_data = size(data,dim)
    @assert iszero(mod(N_data, N_bins))
    # if no rebinning is required
    if N_data == N_bins
        # relabel the original data array the rebinned array
        rebinned_data =  data
    # perform rebinning if necessary
    else
        # Calculate the bin size
        N_binsize = N_data ÷ N_bins
        # get size of data array
        data_dims = size(data)
        # calculated reshaped dims
        reshaped_dims = (data_dims[1:dim-1]..., N_binsize, N_bins, data_dims[dim+1:end]...)
        # reshape the data array
        reshaped_data = reshape(data, reshaped_dims)
        # calculate the average of each new bin and write the rebinned data to a new array
        rebinned_data = dropdims(mean(reshaped_data, dims=dim), dims=dim)
    end

    return rebinned_data
end