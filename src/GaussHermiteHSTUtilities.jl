# evaluate η(s) function for gauss-hermite hubbard-stratonovich transformation
function eval_η(s::Int)

    return sign(s) * sqrt(6*(1-√6) + 4*√6*abs(s))
end

# evaluate bosonic action for single gauss-hermite hubbard-stratonovich field
function eval_Sgh(s::Int)

    return -log(1+√6*(1-2/3*abs(s)))
end


# evaluate bosonic action for array of gauss-hermite hubbard-stratonovich field
function eval_Sgh(s::AbstractArray{Int})

    return sum(eval_Sgh, s)
end


# given a gauss-hermite hubbard-stratonovich field (ghhsf), sample a different one 
function sample_new_ghhsf(rng::AbstractRNG, s::Int)

    states = (
        (-1,+1,+2), # (s = -2) → s′
        (-2,+1,+2), # (s = -1) → s′
        (-2,-1,+2), # (s = +1) → s′
        (-2,-1,+1)  # (s = +2) → s′
    )
    index = s + 3 - (s > 0) # map field to index
    s′ = rand(rng, states[index]) # sample field

    return s′
end