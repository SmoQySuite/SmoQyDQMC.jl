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