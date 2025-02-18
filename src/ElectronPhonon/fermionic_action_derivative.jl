# calculate the derivative of the fermionic action
function fermionic_action_derivative!(
    dSdx::AbstractMatrix{E},
    G::Matrix{T}, logdetG::E, sgndetG::T, δG::E, δθ::E,
    electron_phonon_parameters::ElectronPhononParameters{T,E},
    fermion_greens_calculator::FermionGreensCalculator{T,E},
    B::Vector{P};
    spin::Int = +1
) where {T<:Number, E<:AbstractFloat, P<:AbstractPropagator{T,E}}

    # get some temporary storage matrices to work with
    ldr_ws = fermion_greens_calculator.ldr_ws::LDRWorkspace{T}
    G′ = fermion_greens_calculator.G′::Matrix{T}
    G″ = ldr_ws.M′::Matrix{T}
    G‴ = ldr_ws.M″::Matrix{T}

    # Iterate over imaginary time τ=Δτ⋅l.
    for l in fermion_greens_calculator

        # Propagate equal-time Green's function matrix to current imaginary time G(τ±Δτ,τ±Δτ) ==> G(τ,τ)
        # depending on whether iterating over imaginary time in the forward or reverse direction
        propagate_equaltime_greens!(G, fermion_greens_calculator, B)

        # calculate the derivative of the fermionic action with respect to each phonon field
        # for the current imaginary time slice τ = Δτ⋅l
        dSdx_l = @view dSdx[:,l]
        _fermionic_action_derivative!(dSdx_l, l, G, sgndetG, electron_phonon_parameters, B[l], G′, G″, G‴, spin = spin)

        # Periodically re-calculate the Green's function matrix for numerical stability.
        logdetG, sgndetG, δG′, δθ′ = stabilize_equaltime_greens!(G, logdetG, sgndetG, fermion_greens_calculator, B, update_B̄=false)

        # record maximum error recorded by numerical stabilization
        δG = maximum((δG, δG′))
        δθ = maximum(abs, (δθ, δθ′))
    end

    return (logdetG, sgndetG, δG, δθ)
end


# evaluate the derivative of the fermionic action for at time slice l for symmetric propagators
function _fermionic_action_derivative!(
    dSdx::AbstractVector{E}, l::Int, G::Matrix{T}, sgndetG::T,
    electron_phonon_parameters::ElectronPhononParameters{T,E},
    B::Union{SymChkbrdPropagator{T,E},SymExactPropagator{T,E}},
    G′::Matrix{T}, G″::Matrix{T}, G‴::Matrix{T};
    spin::Int
) where {T<:Number, E<:AbstractFloat}

    phonon_parameters = electron_phonon_parameters.phonon_parameters::PhononParameters{E}
    if isone(spin)
        holstein_parameters = electron_phonon_parameters.holstein_parameters_up::HolsteinParameters{E}
        ssh_parameters = electron_phonon_parameters.ssh_parameters_up::SSHParameters{T}
    else
        holstein_parameters = electron_phonon_parameters.holstein_parameters_dn::HolsteinParameters{E}
        ssh_parameters = electron_phonon_parameters.ssh_parameters_dn::SSHParameters{T}
    end

    # get discretization in imaginary time
    Δτ = electron_phonon_parameters.Δτ

    # get the phonon mass
    mass = phonon_parameters.M

    # get the phonon field associated with the current time slice
    x = @view electron_phonon_parameters.x[:,l]

    # define Λ = exp(-Δτ⋅V) and Γ = exp(-Δτ⋅K/2) and Γ⁻¹ = exp(+Δτ⋅K/2)
    Λ   = Diagonal(B.expmΔτV)
    Γ   = B.expmΔτKo2
    Γᵀ  = adjoint(Γ)
    Γ⁻¹ = get_inv_exp_K(B)

    # construct (G - I)
    copyto!(G″, I) # I
    @. G″ = G - G″ # G - I

    # evaluate ∂S/∂x += sgn(det(G))⋅Tr[(∂Γ/∂x)⋅Γ⁻¹⋅(G - I)]
    eval_tr_dΓdx_invΓ_A!(dSdx, x, Δτ/2, sgndetG, G″, ssh_parameters, Γ, mass, G‴)

    # construct (Γ⁻¹⋅G⋅Γ - I)
    mul!(G′, Γ⁻¹, G) # Γ⁻¹⋅G
    mul!(G″, G′, Γ)  # Γ⁻¹⋅G⋅Γ
    copyto!(G′, I)
    @. G′ = G″ - G′  # Γ⁻¹⋅G⋅Γ - I
    
    # evaluate ∂S/∂x += sgn(det(G))⋅Tr[(∂Λ/∂x)⋅Λ⁻¹⋅(Γ⁻¹⋅G⋅Γ - I)]
    eval_tr_dΛdx_invΛ_A!(dSdx, x, Δτ, sgndetG, G′, holstein_parameters)

    # construct Γ⁻ᵀ⋅(I - Λ⁻¹⋅Γ⁻¹⋅G⋅Γ⋅Λ)
    ldiv!(Λ, G″)  # Λ⁻¹⋅Γ⁻¹⋅G⋅Γ
    rmul!(G″, Λ)   # Λ⁻¹⋅Γ⁻¹⋅G⋅Γ⋅Λ
    copyto!(G′, I)
    @. G″ = G″ - G′ # Λ⁻¹⋅Γ⁻¹⋅G⋅Γ⋅Λ - I

    # evaluate ∂S/∂x += sgn(det(G))⋅Tr[(∂Γᵀ/∂x)⋅Γ⁻ᵀ⋅(Λ⁻¹⋅Γ⁻¹⋅G⋅Γ⋅Λ - I)]
    eval_tr_dΓdx_invΓ_A!(dSdx, x, Δτ/2, sgndetG, G″, ssh_parameters, Γᵀ, mass, G‴)

    return nothing
end


# evaluate the derivative of the fermionic action for at time slice l for asymmetric propagators
function _fermionic_action_derivative!(
    dSdx::AbstractVector{E}, l::Int, G::Matrix{T}, sgndetG::T,
    electron_phonon_parameters::ElectronPhononParameters{T,E},
    B::Union{AsymChkbrdPropagator{T,E},AsymExactPropagator{T,E}},
    G′::Matrix{T}, G″::Matrix{T}, G‴::Matrix{T};
    spin::Int
) where {T<:Number, E<:AbstractFloat}

    phonon_parameters = electron_phonon_parameters.phonon_parameters::PhononParameters{E}
    if isone(spin)
        holstein_parameters = electron_phonon_parameters.holstein_parameters_up::HolsteinParameters{E}
        ssh_parameters = electron_phonon_parameters.ssh_parameters_up::SSHParameters{T}
    else
        holstein_parameters = electron_phonon_parameters.holstein_parameters_dn::HolsteinParameters{E}
        ssh_parameters = electron_phonon_parameters.ssh_parameters_dn::SSHParameters{T}
    end

    # get discretization in imaginary time
    Δτ = electron_phonon_parameters.Δτ

    # get the phonon mass
    mass = phonon_parameters.M

    # get the phonon field associated with the current time slice
    x = @view electron_phonon_parameters.x[:,l]

    # define Λ = exp(-Δτ⋅V) and Γ = exp(-Δτ⋅K) and Γ⁻¹ = exp(+Δτ⋅K)
    Λ = Diagonal(B.expmΔτV)
    Γ = B.expmΔτK

    # construct (I - G)
    copyto!(G′, I)
    @. G′ = G - G′ # G - I

    # evaluate ∂S/∂x += sgn(det(G))⋅Tr[(∂Λ/∂x)⋅Λ⁻¹⋅(G - I)]
    eval_tr_dΛdx_invΛ_A!(dSdx, x, Δτ, sgndetG, G′, holstein_parameters)

    # construct (I - Λ⁻¹⋅G⋅Λ)
    mul!(G′, G, Λ) # G⋅Λ
    ldiv!(Λ, G′)   # Λ⁻¹⋅G⋅Λ
    copyto!(G″, I)
    @. G″ = G′ - G″  # Λ⁻¹⋅G⋅Λ - I

    # evaluate ∂S/∂x += sgn(det(G))⋅Tr[(∂Γ/∂x)⋅Γ⁻¹⋅(Λ⁻¹⋅G⋅Λ - I)]
    eval_tr_dΓdx_invΓ_A!(dSdx, x, Δτ, sgndetG, G″, ssh_parameters, Γ, mass, G‴)

    return nothing
end


# evaluate ∂S/∂x += sgn(det(G))⋅Tr[(∂Λ/∂x)⋅Λ⁻¹⋅A]
function eval_tr_dΛdx_invΛ_A!(dSdx::AbstractVector{E}, x::AbstractVector{E},
                              Δτ::E, sgndetG::T, A::Matrix{T},
                              holstein_parameters::HolsteinParameters{E}) where {T,E}

    (; Nholstein, α, α2, α3, α4, neighbor_table, coupling_to_phonon) = holstein_parameters

    # check if finite number of holstein couplings
    if Nholstein > 0
        # iterate over holstein coupling
        for c in 1:Nholstein
            # get the phonon associated with the coupling
            p = coupling_to_phonon[c]
            # get the orbital whose density is getting coupled to
            i = neighbor_table[2,c]
            # calculate the non-zero matrix element ∂Λ/∂x[i,i] associated with current holstein coupling,
            # recalling that Λ = exp(-Δτ⋅V), where V is the diagonal on-site energy matrix.
            # therefore, ∂Λ/∂x⋅Λ⁻¹ = -Δτ⋅(∂V/∂x)
            nΔτdVdt_ii = -Δτ * (α[c] + 2*α2[c]*x[p] + 3*α3[c]*x[p]^2 + 4*α4[c]*x[p]^3)
            # evaluate ∂S/∂x += Tr[(∂Λ/∂x)⋅A]
            dSdx[p] += real(nΔτdVdt_ii * A[i,i])
        end
    end

    return nothing
end


# evaluate ∂S/∂x += Tr[(∂Γ/∂x)⋅Γ⁻¹⋅A]
function eval_tr_dΓdx_invΓ_A!(dSdx::AbstractVector{E}, x::AbstractVector{E},
                              Δτ::E, sgndetG::T, A::Matrix{T}, ssh_parameters::SSHParameters{T},
                              Γ::AbstractMatrix{T}, M::Vector{E}, A′::Matrix{T}) where {T,E}

    (; Nssh, α, α2, α3, α4, coupling_to_phonon, neighbor_table) = ssh_parameters

    # comment: Γ = exp(-Δτ⋅K) ==> ∂Γ/∂x⋅Γ⁻¹ = -Δτ⋅∂K/∂x

    # if finite number of ssh couplints
    if Nssh > 0
        # iterate over SSH couplings
        for c in 1:Nssh
            # get the pair of phonons getting coupled
            p  = coupling_to_phonon[1,c]
            p′ = coupling_to_phonon[2,c]
            # get the pair of orbitals that the coupled phonon live on
            i = neighbor_table[1,c]
            j = neighbor_table[2,c]
            # calculate the difference in phonon position
            Δx = x[p′] - x[p]
            # if mass of initial phonon is finite
            if isfinite(M[p])
                # get off-diagonal matrix element of -Δτ⋅∂K/∂x
                nΔτdKdx_ji = -Δτ * (-α[c] - 2*α2[c]*Δx - 3*α3[c]*Δx^2 - 4*α4[c]*Δx^3)
                # evaluate Tr[-Δτ⋅∂K/∂x⋅A]
                dSdx[p] += real(nΔτdKdx_ji * A[i,j] + conj(nΔτdKdx_ji) * A[j,i])
            end
            # if mass of final phonon is finite
            if isfinite(M[p′])
                # get off-diagonal matrix element of -Δτ⋅∂K/∂x
                nΔτdKdx_ji = -Δτ * (α[c] + 2*α2[c]*Δx + 3*α3[c]*Δx^2 + 4*α4[c]*Δx^3)
                # evaluate Tr[-Δτ⋅∂K/∂x⋅A]
                dSdx[p′] += real(nΔτdKdx_ji * A[i,j] + conj(nΔτdKdx_ji) * A[j,i])
            end
        end
    end

    return nothing
end

# evaluate ∂S/∂x += Tr[(∂Γ/∂x)⋅Γ⁻¹⋅A]
function eval_tr_dΓdx_invΓ_A!(dSdx::AbstractVector{E}, x::AbstractVector{E},
                              Δτ::E, sgndetG::T, A::Matrix{T},
                              ssh_parameters::SSHParameters{T},
                              Γ::CheckerboardMatrix{T}, M::Vector{E}, A′::Matrix{T}) where {T,E}

    (; Nssh, α, α2, α3, α4, coupling_to_phonon, hopping_to_couplings) = ssh_parameters
    (; Ncolors, perm, neighbor_table) = Γ
    color_bounds = Γ.colors

    # check if there are a finite number of ssh couplings
    if Nssh > 0
        # copy original matrix
        copyto!(A′, A)
        # determine the order to iterate over the checkerboard colors in
        color_order = Γ.transposed ? (1:Ncolors) : (Ncolors:-1:1)
        # iterate over checkerboard colors
        for color in color_order
            # iterate over bounds for current color
            start = color_bounds[1,color]
            stop  = color_bounds[2,color]
            for n in start:stop
                # get the hopping ID associated with the 2x2 checkerboard matrix
                h = perm[n]
                # get ssh coupling
                h_to_c = hopping_to_couplings[h]
                # if there is an ssh coupling associated with the hopping
                if !isempty(h_to_c)
                    # iterate over ssh coupling associated with hopping
                    for c in h_to_c
                        # get the pair of phonons getting coupled
                        p  = coupling_to_phonon[1,c]
                        p′ = coupling_to_phonon[2,c]
                        # get the pair of orbitals that the coupled phonons live on
                        i = neighbor_table[1,n]
                        j = neighbor_table[2,n]
                        # calculate the difference in phonon position
                        Δx = x[p′] - x[p]
                        # if mass of initial phonon is finite
                        if isfinite(M[p])
                            # get off-diagonal matrix element of -Δτ⋅∂K/∂x
                            nΔτdKdx_ji = -Δτ * (-α[c] - 2*α2[c]*Δx - 3*α3[c]*Δx^2 - 4*α4[c]*Δx^3)
                            # evaluate Tr[(∂Γ/∂x)⋅A] = Tr[-Δτ⋅∂K/∂x⋅A′]
                            dSdx[p] += real(nΔτdKdx_ji * A′[i,j] + conj(nΔτdKdx_ji) * A′[j,i])
                        end
                        # if mass of final phonon is finite
                        if isfinite(M[p′])
                            # get off-diagonal matrix element of -Δτ⋅∂K/∂x
                            nΔτdKdx_ji = -Δτ * (α[c] + 2*α2[c]*Δx + 3*α3[c]*Δx^2 + 4*α4[c]*Δx^3)
                            # evaluate Tr[(∂Γ/∂x)⋅A] = Tr[-Δτ⋅∂K/∂x⋅A′]
                            dSdx[p′] += real(nΔτdKdx_ji * A′[i,j] + conj(nΔτdKdx_ji) * A′[j,i])
                        end
                    end
                end
            end
            # apply the wrapping transformation A′ := Γₙ⁻¹⋅A′⋅Γₙ
            if color != last(color_order)
                ldiv!(Γ, A′, color)
                rmul!(A′, Γ, color)
            end
        end
    end

    return nothing
end


# return the inverse of the exponentiated hopping matrix for a propagator
get_inv_exp_K(B::SymExactPropagator) = B.exppΔτKo2
get_inv_exp_K(B::AsymExactPropagator) = B.exppΔτK
get_inv_exp_K(B::SymChkbrdPropagator) = inv(B.expmΔτKo2)
get_inv_exp_K(B::AsymChkbrdPropagator) = inv(B.expmΔτK)


# # evaluate ∂S/∂x += Tr[(∂Γ/∂x)⋅Γ⁻¹⋅A]
# function eval_tr_dΓdx_invΓ_A!(dSdx::AbstractVector{E}, x::AbstractVector{E},
#                               Δτ::E, sgndetG::T, A::Matrix{T},
#                               ssh_parameters::SSHParameters{T},
#                               Γ::CheckerboardMatrix{T}, M::Vector{E}, A′::Matrix{T}) where {T,E}

#     (; Nssh, α, α2, α3, α4, coupling_to_phonon, hopping_to_couplings) = ssh_parameters
#     (; Ncolors, perm, neighbor_table) = Γ
#     color_bounds = Γ.colors

#     # check if there are a finite number of ssh couplings
#     if Nssh > 0
#         # iterate over checkerboard colors
#         for color in 1:Ncolors
#             # apply appropriate checkerboard transformation for the current checkerboard color
#             copyto!(A′, A)
#             cyclic_checkerboard_transformation!(A′, Γ, color)
#             # iterate over bounds for current color
#             start = color_bounds[1,color]
#             stop  = color_bounds[2,color]
#             for n in start:stop
#                 # get the hopping ID associated with the 2x2 checkerboard matrix
#                 h = perm[n]
#                 # get ssh coupling
#                 h_to_c = hopping_to_couplings[h]
#                 # if there is an ssh coupling associated with the hopping
#                 if !isempty(h_to_c)
#                     # iterate over ssh coupling associated with hopping
#                     for c in h_to_c
#                         # get the pair of phonons getting coupled
#                         p  = coupling_to_phonon[1,c]
#                         p′ = coupling_to_phonon[2,c]
#                         # get the pair of orbitals that the coupled phonons live on
#                         i = neighbor_table[1,n]
#                         j = neighbor_table[2,n]
#                         # calculate the difference in phonon position
#                         Δx = x[p′] - x[p]
#                         # if mass of initial phonon is finite
#                         if isfinite(M[p])
#                             # get off-diagonal matrix element of -Δτ⋅∂K/∂x
#                             nΔτdKdx_ji = -Δτ * (-α[c] - 2*α2[c]*Δx - 3*α3[c]*Δx^2 - 4*α4[c]*Δx^3)
#                             # evaluate Tr[(∂Γ/∂x)⋅A] = Tr[-Δτ⋅∂K/∂x⋅A′]
#                             dSdx[p] += real(nΔτdKdx_ji * A′[i,j] + conj(nΔτdKdx_ji) * A′[j,i])
#                         end
#                         # if mass of final phonon is finite
#                         if isfinite(M[p′])
#                             # get off-diagonal matrix element of -Δτ⋅∂K/∂x
#                             nΔτdKdx_ji = -Δτ * (α[c] + 2*α2[c]*Δx + 3*α3[c]*Δx^2 + 4*α4[c]*Δx^3)
#                             # evaluate Tr[(∂Γ/∂x)⋅A] = Tr[-Δτ⋅∂K/∂x⋅A′]
#                             dSdx[p′] += real(nΔτdKdx_ji * A′[i,j] + conj(nΔτdKdx_ji) * A′[j,i])
#                         end
#                     end
#                 end
#             end
#         end
#     end

#     return nothing
# end

# # apply cyclic checkerboard transformation
# function cyclic_checkerboard_transformation!(A::Matrix{T}, Γ::CheckerboardMatrix{T}, color::Int) where {T}

#     # note: Γₙ = exp(-Δτ⋅Kₙ) where Γₙ is checkerboard color matrix for color = n

#     # example Ncolors = 4, color = 2, transposed = false
#     # calculate A′ for
#     # Tr[(Γ₄⋅Γ₃⋅∂Γ₂/∂x⋅Γ₁)⋅(Γ₁⁻¹⋅Γ₂⁻¹⋅Γ₃⁻¹⋅Γ₄⁻¹)⋅A]
#     # = -Δτ⋅Tr[(Γ₄⋅Γ₃⋅∂K₂/∂x⋅Γ₂⋅Γ₁)⋅(Γ₁⁻¹⋅Γ₂⁻¹⋅Γ₃⁻¹⋅Γ₄⁻¹)⋅A]
#     # = -Δτ⋅Tr[Γ₄⋅Γ₃⋅∂K₂/∂x⋅Γ₃⁻¹⋅Γ₄⁻¹⋅A]
#     # = -Δτ⋅Tr[∂K₂/∂x⋅Γ₃⁻¹⋅Γ₄⁻¹⋅A⋅Γ₄⋅Γ₃]
#     # = -Δτ⋅Tr[∂K₂/∂x⋅A′], where A′ = Γ₃⁻¹⋅Γ₄⁻¹⋅A⋅Γ₄⋅Γ₃

#     # example Ncolors = 4, color = 3, transposed = true
#     # calculate A′ for
#     # Tr[(Γ₁⋅Γ₂⋅∂Γ₃/∂x⋅Γ₄)⋅(Γ₄⁻¹⋅Γ₃⁻¹⋅Γ₂⁻¹⋅Γ₁⁻¹)⋅A]
#     # = -Δτ⋅Tr[(Γ₁⋅Γ₂⋅∂K₃/∂x⋅Γ₃⋅Γ₄)⋅(Γ₄⁻¹⋅Γ₃⁻¹⋅Γ₂⁻¹⋅Γ₁⁻¹)⋅A]
#     # = -Δτ⋅Tr[Γ₁⋅Γ₂⋅∂K₃/∂x⋅Γ₂⁻¹⋅Γ₁⁻¹⋅A]
#     # = -Δτ⋅Tr[∂K₃/∂x⋅Γ₂⁻¹⋅Γ₁⁻¹⋅A⋅Γ₁⋅Γ₂]
#     # = -Δτ⋅Tr[∂K₃/∂x⋅A′] where A′ = Γ₂⁻¹⋅Γ₁⁻¹⋅A⋅Γ₁⋅Γ₂

#     # whether the checkerboard matrix is transposed
#     transposed = Γ.transposed::Bool

#     # total number of colors
#     Ncolors = Γ.Ncolors::Int

#     # checkerboard colors to iterate over and their order
#     colors = transposed ? (1:color-1) : (Ncolors:-1:(color+1))

#     # perform checkerboard transformation
#     for c in colors
#         # calculate A := Γₙ⁻¹⋅A⋅Γₙ
#         ldiv!(Γ, A, c)
#         rmul!(A, Γ, c)
#     end

#     return nothing
# end