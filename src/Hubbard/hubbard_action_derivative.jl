function hubbard_action_derivative!(dSds::AbstractMatrix{E},
                                    hubbard_hs_parameters::AbstractHubbardHS{T},
                                    Gup::Matrix{T}, logdetGup::E, sgndetGup::T,
                                    Gdn::Matrix{T}, logdetGdn::E, sgndetGdn::T,
                                    δG::E, δθ::T,
                                    fermion_greens_calculator_up::FermionGreensCalculator{T,E},
                                    fermion_greens_calculator_dn::FermionGreensCalculator{T,E},
                                    Bup::Vector{P}, Bdn::Vector{P}) where {T<:Number, E<:AbstractFloat, P<:AbstractPropagator{T,E}}
    
    # initialize derivative to zero
    fill!(dSds, zero(E))

    # calculate contribution to derivative from spin up sector
    logdetGup, sgndetGup, δG, δθ = _fermionic_action_derivative!(dSds, hubbard_hs_parameters, Gup, logdetGup, sgndetGup, +1,
                                                                 δG, δθ, fermion_greens_calculator_up, Bup)

    # calculate contribution to derivative from spin down sector
    logdetGdn, sgndetGdn, δG, δθ = _fermionic_action_derivative!(dSds, hubbard_hs_parameters, Gdn, logdetGdn, sgndetGdn, -1,
                                                                 δG, δθ, fermion_greens_calculator_dn, Bdn)

    # calculate bosonic component of derivative
    _bosonic_action_derivative!(dSds, hubbard_hs_parameters)

    return (logdetGup, sgndetGup, logdetGdn, sgndetGdn, δG, δθ)
end

function hubbard_action_derivative!(dSds::AbstractMatrix{E},
                                    hubbard_hs_parameters::AbstractHubbardHS{T},
                                    G::Matrix{T}, logdetG::E, sgndetG::T,
                                    δG::E, δθ::T,
                                    fermion_greens_calculator_up::FermionGreensCalculator{T,E},
                                    B::Vector{P}) where {T<:Number, E<:AbstractFloat, P<:AbstractPropagator{T,E}}

    (; U) = hubbard_hs_parameters
    
    # make sure strictly attractive hubbard interactions
    @assert all(u -> u < 0, U)
    
    # initialize derivative to zero
    fill!(dSds, zero(E))

    # calculate derivative from spin up sector
    logdetG, sgndetG, δG, δθ = _fermionic_action_derivative!(dSds, hubbard_hs_parameters, G, logdetG, sgndetG, +1,
                                                             δG, δθ, fermion_greens_calculator_up, B)

    # scale by two to account for spin degenercy
    @. dSds = 2 * dSds

    # calculate bosonic component of derivative
    _bosonic_action_derivative!(dSds, hubbard_hs_parameters)

    return (logdetG, sgndetG, δG, δθ)
end


function _fermionic_action_derivative!(dSds::AbstractMatrix{E},
                                       hubbard_hs_parameters::AbstractHubbardHS{T},
                                       G::Matrix{T}, logdetG::E, sgndetG::T,
                                       σ::Int, δG::E, δθ::T,
                                       fermion_greens_calculator::FermionGreensCalculator{T,E},
                                       B::Vector{P}) where {T<:Number, E<:AbstractFloat, P<:AbstractPropagator{T,E}}

    # get some temporary storage matrix to work with
    G′ = fermion_greens_calculator.G′::Matrix{T}

    # Iterate over imaginary time τ=Δτ⋅l.
    for l in fermion_greens_calculator

        # get propagator for current time slice
        B_l = B[l]::P

        # Propagate equal-time Green's function matrix to current imaginary time G(τ±Δτ,τ±Δτ) ==> G(τ,τ)
        # depending on whether iterating over imaginary time in the forward or reverse direction
        propagate_equaltime_greens!(G, fermion_greens_calculator, B)

        # apply the transformation G̃(τ,τ) = exp(+Δτ⋅K[l]/2)⋅G(τ,τ)⋅exp(-Δτ⋅K[l]/2)
        # if B[l] = exp(-Δτ⋅K[l]/2)⋅exp(-Δτ⋅V[l])⋅exp(-Δτ⋅K[l]/2),
        # otherwise nothing when B[l] = exp(-Δτ⋅V[l])⋅exp(-Δτ⋅K[l])
        forward_partially_wrap_greens(G, B_l, G′)

        # calculate the derivative of the fermionic action with respect to each phonon field
        # for the current imaginary time slice τ = Δτ⋅l
        _fermionic_action_derivative!(dSds, l, G, σ, hubbard_hs_parameters)

        # apply the transformation G(τ,τ) = exp(-Δτ⋅K[l]/2)⋅G̃(τ,τ)⋅exp(+Δτ⋅K[l]/2)
        # if B[l] = exp(-Δτ⋅K[l]/2)⋅exp(-Δτ⋅V[l])⋅exp(-Δτ⋅K[l]/2),
        # otherwise nothing when B[l] = exp(-Δτ⋅V[l])⋅exp(-Δτ⋅K[l])
        reverse_partially_wrap_greens(G, B_l, G′)

        # Periodically re-calculate the Green's function matrix for numerical stability.
        logdetG, sgndetG, δG′, δθ′ = stabilize_equaltime_greens!(G, logdetG, sgndetG, fermion_greens_calculator, B, update_B̄=false)

        # record maximum error recorded by numerical stabilization
        δG = maximum((δG, δG′))
        δθ = maximum(abs, (δθ, δθ′))
    end

    return (logdetG, sgndetG, δG, δθ)
end

function _fermionic_action_derivative!(dSds::Matrix{E}, l::Int, G::Matrix{T}, σ::Int,
                                       hubbard_hs_parameters::AbstractHubbardHS{E}) where {T<:Number, E<:AbstractFloat}

    (; Δτ, sites, s, U) = hubbard_hs_parameters

    # iterate over HS field
    for i in axes(s,1)
        # get the orbital associated with the HS field
        site = sites[i]
        # evaluate the derivative
        dSds[i,l] = dSds[i,l] + (σ*heaviside(U[i]) + heaviside(-U[i])) * eval_dads(i, l, hubbard_hs_parameters) * (G[site,site] - 1)
    end

    return nothing
end

@doc raw"""
    update!(fermion_path_integral::FermionPathIntegral{T,E},
            hubbard_hs_parameters::AbstractHubbardHS{E},
            σ::Int, sgn::Int) where {T, E}

Update `fermion_path_integral` according to the Hubbard-Stranonovich (HS) fields `hubbard_hs_parameters.s`,
where `σ` specified the spin, and `sgn` specifies whether the effect of the current HS configuration
should be added to (`sgn = +1`) or subtracted from (`sgn = -1`) the `fermion_path_integral`.
"""
function update!(fermion_path_integral::FermionPathIntegral{T,E},
                 hubbard_hs_parameters::AbstractHubbardHS{E},
                 σ::Int, sgn::Int) where {T, E}

    (; s, U, sites, Δτ) = hubbard_hs_parameters
    (; V) = fermion_path_integral

    # iterate over imaginary time axis
    for l in axes(s,2)
        # iterate over fields for current imaginary time slice
        for i in axes(s,1)
            # get the corresponding site/orbital associated with the HS field
            site = sites[i]
            # upate fermion path integral
            V[site,l] = V[site,l] - sgn * (σ*heaviside(U[i]) + heaviside(-U[i]))/Δτ * eval_a(i, l, hubbard_hs_parameters)
        end
    end

    return nothing
end