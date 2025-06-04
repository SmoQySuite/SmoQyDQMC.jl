# @doc raw"""
#     forward_partially_wrap_greens(G::Matrix{T}, B::P,
#                                   M::Matrix{T}=similar(G)) where {T, E, P<:AbstractPropagator{T,E}}

# If the propagator `B` is represented in the symmetric form
# ```math
# B_l = \Gamma_l(\Delta\tau/2) \cdot \Lambda_l(\Delta\tau) \cdot \Gamma_l^\dagger(\Delta\tau/2),
# ```
# where ``\tau = \Delta\tau \cdot l``, ``\Gamma(\Delta\tau/2) = e^{-\Delta\tau K_l/2}`` and ``\Lambda(\Delta\tau) = e^{-\Delta\tau V_l}``,
# then apply the transformation
# ```math
# \tilde{G}(\tau,\tau) = \Gamma^{-1}_l(\Delta\tau/2) \cdot G(\tau,\tau) \Gamma_l(\Delta\tau/2)
# ```
# to the equal-time Green's function matrix `G` in-place.
# """
function forward_partially_wrap_greens(G::Matrix{T}, B::P, M::Matrix{T}=similar(G)) where {T, E, P<:AbstractPropagator{T,E}}

    # only apply transformation if symmetric/hermitian definition for propagator is being used
    _forward_partially_wrap_greens(G, B, M)

    return nothing
end

# perform the G̃(τ,τ) = Γ⁻¹[l]⋅G(τ,τ)⋅Γ[l] transformation in the case the Γ[l]=exp(-Δτ⋅K[l]/2) is the exactly exponentiated hopping matrix
function _forward_partially_wrap_greens(G::Matrix{T}, B::SymExactPropagator{T,E}, M::Matrix{T}) where {T,E}

    (; expmΔτKo2, exppΔτKo2) = B
    mul!(M, G, expmΔτKo2) # G(τ,τ)⋅Γ[l]
    mul!(G, exppΔτKo2, M) # G̃(τ,τ) = Γ⁻¹[l]⋅G(τ,τ)⋅Γ[l]

    return nothing
end

# perform the G̃(τ,τ) = Γ⁻¹[l]⋅G(τ,τ)⋅Γ[l] transformation in the case the Γ[l] is the checkerboard approximation of exp(-Δτ⋅K[l]/2)
function _forward_partially_wrap_greens(G::Matrix{T}, B::SymChkbrdPropagator{T,E}, ignore...) where {T,E}

    (; expmΔτKo2) = B
    rmul!(G, expmΔτKo2) # G(τ,τ)⋅Γ[l]
    ldiv!(expmΔτKo2, G) # G̃(τ,τ) = Γ⁻¹[l]⋅G(τ,τ)⋅Γ[l]

    return nothing
end

# do nothing for asymmetric propagator
function _forward_partially_wrap_greens(G::Matrix{T}, B::AsymExactPropagator{T,E}, ignore...) where {T,E}

    return nothing
end

# do nothing for asymmetric propagator
function _forward_partially_wrap_greens(G::Matrix{T}, B::AsymChkbrdPropagator{T,E}, ignore...) where {T,E}

    return nothing
end

# @doc raw"""
#     reverse_partially_wrap_greens(G::Matrix{T}, B::P,
#                                   M::Matrix{T}=similar(G)) where {T, E, P<:AbstractPropagator{T,E}}

# If the propagator `B` is represented in the symmetric form
# ```math
# B_l = \Gamma_l(\Delta\tau/2) \cdot \Lambda_l(\Delta\tau) \cdot \Gamma_l^\dagger(\Delta\tau/2),
# ```
# where ``\tau = \Delta\tau \cdot l``, ``\Gamma(\Delta\tau/2) = e^{-\Delta\tau K_l/2}`` and ``\Lambda(\Delta\tau) = e^{-\Delta\tau V_l}``,
# then apply the transformation
# ```math
# G(\tau,\tau) = \Gamma_l(\Delta\tau/2) \cdot \tilde{G}(\tau,\tau) \Gamma_l^{-1}(\Delta\tau/2)
# ```
# to the equal-time Green's function matrix `G` in-place.
# """
function reverse_partially_wrap_greens(G::Matrix{T}, B::P, M::Matrix{T}=similar(G)) where {T, E, P<:AbstractPropagator{T,E}}

    # only apply transformation if symmetric/hermitian definition for propagator is being used
    _reverse_partially_wrap_greens(G, B, M)

    return nothing
end

# perform the G(τ,τ) = Γ[l]⋅G̃(τ,τ)⋅Γ⁻¹[l] transformation in the case the Γ[l]=exp(-Δτ⋅K[l]/2) is the exactly exponentiated hopping matrix
function _reverse_partially_wrap_greens(G::Matrix{T}, B::SymExactPropagator{T,E}, M::Matrix{T}) where {T,E}

    (; expmΔτKo2, exppΔτKo2) = B
    mul!(M, G, exppΔτKo2) # G̃(τ,τ)⋅Γ⁻¹[l]
    mul!(G, expmΔτKo2, M) # G(τ,τ) = Γ[l]⋅G̃(τ,τ)⋅Γ⁻¹[l]

    return nothing
end

# perform the G(τ,τ) = Γ[l]⋅G̃(τ,τ)⋅Γ⁻¹[l] transformation in the case the Γ[l] is the checkerboard approximation of exp(-Δτ⋅K[l]/2)
function _reverse_partially_wrap_greens(G::Matrix{T}, B::SymChkbrdPropagator{T,E}, ignore...) where {T,E}

    (; expmΔτKo2) = B
    rdiv!(G, expmΔτKo2) # G̃(τ,τ)⋅Γ⁻¹[l]
    lmul!(expmΔτKo2, G) # G(τ,τ) = Γ[l]⋅G̃(τ,τ)⋅Γ⁻¹[l]

    return nothing
end

# do nothing for asymmetric propagator
function _reverse_partially_wrap_greens(G::Matrix{T}, B::AsymExactPropagator{T,E}, ignore...) where {T,E}

    return nothing
end

# do nothing for asymmetric propagator
function _reverse_partially_wrap_greens(G::Matrix{T}, B::AsymChkbrdPropagator{T,E}, ignore...) where {T,E}

    return nothing
end

# Swap the contents of the two arrays `a` and `b`.
function swap!(a::AbstractArray{T}, b::AbstractArray{T}) where {T}

    @fastmath @inbounds for i in eachindex(a)
        tmp = a[i]
        a[i] = b[i]
        b[i] = tmp
    end

    return nothing
end

sign_or_0to1(x::T) where {T<:Number} = iszero(x) ? one(T) : sign(x)

# default bosonic action evaluation method
bosonic_action(some_model_parameters) = 0.0