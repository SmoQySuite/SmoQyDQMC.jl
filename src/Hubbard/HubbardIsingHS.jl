@doc raw"""
    HubbardIsingHSParameters{T<:AbstractFloat}

Parameters associated with decoupling the Hubbard interaction using the standard Ising
Hubbard-Stratonovich (HS) transformation.

# Fields

- `β::T`: Inverse temperature.
- `Δτ::T`: Discretization in imaginary time.
- `Lτ::Int`: Length of imaginary time axis.
- `N::Int`: Number of orbitals in the lattice.
- `U::Vector{T}`: Each hubbard interaction.
- `α::Vector{T}`: The constant given by ``\cosh(\alpha_i) = e^{\Delta\tau \vert U_i \vert/2}.``
- `sites::Vector{Int}`: Site index associated with each finite Hubbard `U` interaction.
- `s::Matrix{Int}`: Ising Hubbard-Stratonovich fields.
- `update_perm::Vector{Int}`: Order in which to iterate over HS fields in time slice when performing local updates.
"""
struct HubbardIsingHSParameters{T<:AbstractFloat}

    # inverse temperature
    β::T

    # discretization in imaginary time
    Δτ::T

    # length of imaginary time axis
    Lτ::Int

    # number of orbitals with finite Hubbard U
    N::Int

    # each finite hubbard interaction
    U::Vector{T}
    
    # cosh(α) = exp(Δτ|U|/2)
    α::Vector{T}

    # site index associated with each Hubbard U
    sites::Vector{Int}

    # Ising Hubbard-Stratonovich fields
    s::Matrix{Int}

    # order in which to iterate over orbitals when updating Hubbard-Stratonovich fields.
    update_perm::Vector{Int}
end


@doc raw"""
    HubbardIsingHSParameters(; β::E, Δτ::E,
                             hubbard_parameters::HubbardParameters{E},
                             rng::AbstractRNG) where {E<:AbstractFloat}

Initialize and return an instance of the [`HubbardIsingHSParameters`](@ref) type.
Note that on-site energies `fpi.V` are shifted by ``-U_i/2`` if ``hm.shifted = true``.
"""
function HubbardIsingHSParameters(; β::E, Δτ::E,
                                  hubbard_parameters::HubbardParameters{E},
                                  rng::AbstractRNG) where {E<:AbstractFloat}

    (; U, sites) = hubbard_parameters

    # calcualte length of imaginary time axis
    Lτ = eval_length_imaginary_axis(β, Δτ)

    # get the number of HS transformations per imaginary time-slice
    N = length(U)

    # calculate α constant for each Hubbard U, given by cosh(α) = exp(Δτ⋅|U|/2)
    α = similar(U)
    @. α = acosh(exp(Δτ*abs(U)/2))

    # initialize HS fields
    s = rand(rng, -1:2:1, N, Lτ)

    # initialize update permuation order
    update_perm = collect(1:N)

    return HubbardIsingHSParameters(β, Δτ, Lτ, N, U, α, sites, s, update_perm)
end


@doc raw"""
    initialize!(fermion_path_integral_up::FermionPathIntegral{T,E},
                fermion_path_integral_dn::FermionPathIntegral{T,E},
                hubbard_ising_parameters::HubbardIsingHSParameters{E}) where {T,E}

Initialize the contribution from the Hubbard interaction to the [`FermionPathIntegral`](@ref)
instance `fermion_path_integral_up` for spin up and `fermion_path_integral_dn` spin down.
"""
function initialize!(fermion_path_integral_up::FermionPathIntegral{T,E},
                     fermion_path_integral_dn::FermionPathIntegral{T,E},
                     hubbard_ising_parameters::HubbardIsingHSParameters{E}) where {T,E}
    
    (; α, U, Δτ, s) = hubbard_ising_parameters
    Vup = fermion_path_integral_up.V
    Vdn = fermion_path_integral_dn.V

    # add Ising HS field contribution to diagonal on-site energy matrices
    for l in axes(Vup,2)
        @views @. Vup[:,l] = Vup[:,l] - α/Δτ * s[:,l]
    end

    if !(Vup === Vdn)
        for l in axes(Vdn,2)
            @views @. Vdn[:,l] = Vdn[:,l] + sign(U)*α/Δτ * s[:,l]
        end
    end

    return nothing
end


@doc raw"""
    initialize!(fermion_path_integral::FermionPathIntegral{T,E},
                hubbard_ising_parameters::HubbardIsingHSParameters{E}) where {T,E}

Initialize the contribution from an attractive Hubbard interaction to the [`FermionPathIntegral`](@ref)
instance `fermion_path_integral`.
"""
function initialize!(fermion_path_integral::FermionPathIntegral{T,E},
                     hubbard_ising_parameters::HubbardIsingHSParameters{E}) where {T,E}
    
    (; α, U, Δτ, s) = hubbard_ising_parameters
    V = fermion_path_integral.V

    # make sure its a strictly attractive hubbard interaction
    @assert all(u -> u < 0.0, U)

    # add Ising HS field contribution to diagonal on-site energy matrices
    for l in axes(Vup,2)
        @views @. V[:,l] = Vup[:,l] - α/Δτ * s[:,l]
    end

    return nothing
end


@doc raw"""
    local_updates!(Gup::Matrix{T}, logdetGup::E, sgndetGup::T,
                   Gdn::Matrix{T}, logdetGdn::E, sgndetGdn::T,
                   hubbard_ising_parameters::HubbardIsingHSParameters{E};
                   fermion_path_integral_up::FermionPathIntegral{T,E},
                   fermion_path_integral_dn::FermionPathIntegral{T,E},
                   fermion_greens_calculator_up::FermionGreensCalculator{T,E},
                   fermion_greens_calculator_dn::FermionGreensCalculator{T,E},
                   Bup::Vector{P}, Bdn::Vector{P},
                   δG_max::E, δG::E, δθ::E, rng::AbstractRNG) where {T<:Number, E<:AbstractFloat, P<:AbstractPropagator{T,E}}

Sweep through every imaginary time slice and orbital in the lattice, peforming local updates to every
Ising Hubbard-Stratonovich (HS) field.

This method returns the a tuple containing `(acceptance_rate, logdetGup, sgndetGup, logdetGdn, sgndetGdn, δG, δθ)`.

# Arguments

- `Gup::Matrix{T}`: Spin-up equal-time Green's function matrix.
- `logdetGup::E`: The log of the absolute value of the determinant of the spin-up equal-time Green's function matrix, ``\log \vert \det G_\uparrow(\tau,\tau) \vert.``
- `sgndetGup::T`: The sign/phase of the determinant of the spin-up equal-time Green's function matrix, ``\det G_\uparrow(\tau,\tau) / \vert \det G_\uparrow(\tau,\tau) \vert.``
- `Gdn::Matrix{T}`: Spin-down equal-time Green's function matrix.
- `logdetGdn::E`: The log of the absolute value of the determinant of the spin-down equal-time Green's function matrix, ``\log \vert \det G_\downarrow(\tau,\tau) \vert.``
- `sgndetGdn::T`: The sign/phase of the determinant of the spin-down equal-time Green's function matrix, ``\det G_\downarrow(\tau,\tau) / \vert \det G_\downarrow(\tau,\tau) \vert.``
- `hubbard_ising_parameters::HubbardIsingHSParameters{E}`: Ising Hubbard-Stratonovich fields and associated parameters to update.

## Keyword Arguments

- `fermion_path_integral_up::FermionPathIntegral{T,E}`: An instance of the [`FermionPathIntegral`](@ref) type for spin-up electrons.
- `fermion_path_integral_dn::FermionPathIntegral{T,E}`: An instance of the [`FermionPathIntegral`](@ref) type for spin-down electrons.
- `fermion_greens_calculator_up::FermionGreensCalculator{T,E}`: An instance of the [`FermionGreensCalculator`](https://smoqysuite.github.io/JDQMCFramework.jl/stable/api/#JDQMCFramework.FermionGreensCalculator) type for the spin-up electrons.
- `fermion_greens_calculator_dn::FermionGreensCalculator{T,E}`: An instance of the [`FermionGreensCalculator`](https://smoqysuite.github.io/JDQMCFramework.jl/stable/api/#JDQMCFramework.FermionGreensCalculator) type for the spin-down electrons.
- `Bup::Vector{P}`: Spin-up propagators for each imaginary time slice.
- `Bdn::Vector{P}`: Spin-dn propagators for each imaginary time slice.
- `δG::E`: Previously recorded maximum error in the Green's function corrected by numerical stabilization.
- `δθ::T`: Previously recorded maximum error in the sign/phase of the determinant of the equal-time Green's function matrix corrected by numerical stabilization.
- `rng::AbstractRNG`: Random number generator used in method instead of global random number generator, important for reproducibility.
"""
function local_updates!(Gup::Matrix{T}, logdetGup::E, sgndetGup::T,
                        Gdn::Matrix{T}, logdetGdn::E, sgndetGdn::T,
                        hubbard_ising_parameters::HubbardIsingHSParameters{E};
                        fermion_path_integral_up::FermionPathIntegral{T,E},
                        fermion_path_integral_dn::FermionPathIntegral{T,E},
                        fermion_greens_calculator_up::FermionGreensCalculator{T,E},
                        fermion_greens_calculator_dn::FermionGreensCalculator{T,E},
                        Bup::Vector{P}, Bdn::Vector{P},
                        δG_max::E, δG::E, δθ::E, rng::AbstractRNG) where {T<:Number, E<:AbstractFloat, P<:AbstractPropagator{T,E}}

    Δτ          = hubbard_ising_parameters.Δτ::E
    U           = hubbard_ising_parameters.U::Vector{E}
    α           = hubbard_ising_parameters.α::Vector{E}
    sites       = hubbard_ising_parameters.sites::Vector{Int}
    s           = hubbard_ising_parameters.s::Matrix{Int}
    update_perm = hubbard_ising_parameters.update_perm::Vector{Int}
    u           = fermion_path_integral_up.u::Vector{T}
    v           = fermion_path_integral_dn.v::Vector{T}

    # get temporary storage matrix
    G′ = fermion_greens_calculator_up.G′::Matrix{T}

    # get on-site energy matrices for spin up and down electrons for all time slices
    Vup = fermion_path_integral_up.V::Matrix{T}
    Vdn = fermion_path_integral_dn.V::Matrix{T}

    # counts of the number of proposed spin flips
    proposed_spin_flips = 0

    # counter for the number of accepted spin flips
    accepted_spin_flips = 0

    # Iterate over imaginary time τ=Δτ⋅l.
    for l in fermion_greens_calculator_up

        # Propagate equal-time Green's function matrix to current imaginary time G(τ±Δτ,τ±Δτ) ==> G(τ,τ)
        # depending on whether iterating over imaginary time in the forward or reverse direction
        propagate_equaltime_greens!(Gup, fermion_greens_calculator_up, Bup)
        propagate_equaltime_greens!(Gdn, fermion_greens_calculator_dn, Bdn)

        # get propagators for current time slice
        Bup_l = Bup[l]::P
        Bdn_l = Bdn[l]::P

        # apply the transformation G̃(τ,τ) = exp(+Δτ⋅K[l]/2)⋅G(τ,τ)⋅exp(-Δτ⋅K[l]/2)
        # if B[l] = exp(-Δτ⋅K[l]/2)⋅exp(-Δτ⋅V[l])⋅exp(-Δτ⋅K[l]/2),
        # otherwise nothing when B[l] = exp(-Δτ⋅V[l])⋅exp(-Δτ⋅K[l])
        forward_partially_wrap_greens(Gup, Bup_l, G′)
        forward_partially_wrap_greens(Gdn, Bdn_l, G′)

        # shuffle the order in which orbitals/sites will be iterated over
        shuffle!(rng, update_perm)

        # iterate over orbitals in the lattice
        for i in update_perm

            # increment number of proposed spin flips
            proposed_spin_flips += 1

            # calculate the change in the bosonic action, only non-zero for attractive hubbard interactions
            if U[i] < 0.0
                ΔSb_il = 2 * α[i] * (-s[i,l])
            else
                ΔSb_il = 0.0
            end

            # get the site index associated with the current Hubbard interaction
            site = sites[i]

            # calculate new value of Vup[i,l] and Vdn[i,l] resulting from the
            # Ising HS field having its sign flipped s[i,l] ==> -s[i,l]
            Vup_il′ = -2 *              α[i]/Δτ * (-s[i,l]) + Vup[site,l]
            Vdn_il′ = +2 * sign(U[i]) * α[i]/Δτ * (-s[i,l]) + Vdn[site,l]

            # calculate spin-up determinant ratio associated with Ising HS spin flip
            Rup_il, Δup_il = local_update_det_ratio(Gup, Bup_l, Vup_il′, site, Δτ)
            Rdn_il, Δdn_il = local_update_det_ratio(Gdn, Bdn_l, Vdn_il′, site, Δτ)

            # calculate acceptance probability
            P_il = abs(exp(-ΔSb_il) * Rup_il * Rdn_il)

            # accept or reject proposed upate
            if rand(rng) < P_il

                # increment the cound of accepted spin flips
                accepted_spin_flips += 1

                # flip the spin
                s[i,l] = -s[i,l]

                # update diagonal on-site energy matrix
                Vup[site,l] = Vup_il′
                Vdn[site,l] = Vdn_il′

                # update the spin-up and down Green's function
                logdetGup, sgndetGup = local_update_greens!(Gup, logdetGup, sgndetGup, Bup_l, Rup_il, Δup_il, site, u, v)
                logdetGdn, sgndetGdn = local_update_greens!(Gdn, logdetGdn, sgndetGdn, Bdn_l, Rdn_il, Δdn_il, site, u, v)
            end
        end

        # apply the transformation G(τ,τ) = exp(-Δτ⋅K[l]/2)⋅G̃(τ,τ)⋅exp(+Δτ⋅K[l]/2)
        # if B[l] = exp(-Δτ⋅K[l]/2)⋅exp(-Δτ⋅V[l])⋅exp(-Δτ⋅K[l]/2),
        # otherwise nothing when B[l] = exp(-Δτ⋅V[l])⋅exp(-Δτ⋅K[l])
        reverse_partially_wrap_greens(Gup, Bup_l, G′)
        reverse_partially_wrap_greens(Gdn, Bdn_l, G′)

        # Periodically re-calculate the Green's function matrix for numerical stability.
        logdetGup, sgndetGup, δGup, δθup = stabilize_equaltime_greens!(Gup, logdetGup, sgndetGup, fermion_greens_calculator_up, Bup, update_B̄=true)
        logdetGdn, sgndetGdn, δGdn, δθdn = stabilize_equaltime_greens!(Gdn, logdetGdn, sgndetGdn, fermion_greens_calculator_dn, Bdn, update_B̄=true)

        # record the max errors
        δG = maximum((δG, δGup, δGdn))
        δθ = maximum(abs, (δθ, δθup, δθdn))

        # keep spin-up and spin-down sectors synchronized
        iterate(fermion_greens_calculator_dn, fermion_greens_calculator_up.forward)
    end

    # update stabilization frequency if required
    (logdetGup, sgndetGup, logdetGdn, sgndetGdn, δG, δθ) = update_stabalization_frequency!(
        Gup, logdetGup, sgndetGup,
        Gdn, logdetGdn, sgndetGdn,
        fermion_greens_calculator_up = fermion_greens_calculator_up,
        fermion_greens_calculator_dn = fermion_greens_calculator_dn,
        Bup = Bup, Bdn = Bdn, δG = δG, δθ = δθ, δG_max = δG_max
    )

    # calculate the acceptance rate
    acceptance_rate = accepted_spin_flips / proposed_spin_flips

    return (acceptance_rate, logdetGup, sgndetGup, logdetGdn, sgndetGdn, δG, δθ)
end


@doc raw"""
    local_updates!(G::Matrix{T}, logdetG::E, sgndetG::T,
                   hubbard_ising_parameters::HubbardIsingHSParameters{E};
                   fermion_path_integral::FermionPathIntegral{T,E},
                   fermion_greens_calculator::FermionGreensCalculator{T,E},
                   B::Vector{P}, δG, δθ::E,
                   rng::AbstractRNG) where {T<:Number, E<:AbstractFloat, P<:AbstractPropagator{T,E}}

Sweep through every imaginary time slice and orbital in the lattice, performing local updates to every
Ising Hubbard-Stratonovich (HS) field, assuming strictly attractive Hubbard interactions and perfect spin symmetry.

This method returns the a tuple containing `(acceptance_rate, logdetG, sgndetG, δG, δθ)`.

# Arguments

- `G::Matrix{T}`: Equal-time Green's function matrix.
- `logdetG::E`: The log of the absolute value of the determinant of theequal-time Green's function matrix, ``\log \vert \det G_\uparrow(\tau,\tau) \vert.``
- `sgndetG::T`: The sign/phase of the determinant of the equal-time Green's function matrix, ``\det G_\uparrow(\tau,\tau) / \vert \det G_\uparrow(\tau,\tau) \vert.``
- `hubbard_ising_parameters::HubbardIsingHSParameters{E}`: Ising Hubbard-Stratonovich fields and associated parameters to update.

## Keyword Arguments

- `fermion_path_integral::FermionPathIntegral{T,E}`: An instance of the [`FermionPathIntegral`](@ref) type.
- `fermion_greens_calculator::FermionGreensCalculator{T,E}`: An instance of the [`FermionGreensCalculator`](https://smoqysuite.github.io/JDQMCFramework.jl/stable/api/#JDQMCFramework.FermionGreensCalculator) type.
- `B::Vector{P}`: Propagators for each imaginary time slice.
- `δG::E`: Previously recorded maximum error in the Green's function corrected by numerical stabilization.
- `δθ::T`: Previously recorded maximum error in the sign/phase of the determinant of the equal-time Green's function matrix corrected by numerical stabilization.
- `rng::AbstractRNG`: Random number generator used in method instead of global random number generator, important for reproducibility.
"""
function local_updates!(G::Matrix{T}, logdetG::E, sgndetG::T,
                        hubbard_ising_parameters::HubbardIsingHSParameters{E};
                        fermion_path_integral::FermionPathIntegral{T,E},
                        fermion_greens_calculator::FermionGreensCalculator{T,E},
                        B::Vector{P}, δG, δθ::E,
                        rng::AbstractRNG) where {T<:Number, E<:AbstractFloat, P<:AbstractPropagator{T,E}}

    (; update_perm, U, α, sites, s, Δτ) = hubbard_ising_parameters
    (; u, v) = fermion_path_integral

    # get temporary storage matrix
    G′ = fermion_greens_calculator.G′

    # get on-site energy matrices for spin up and down electrons for all time slices
    V = fermion_path_integral.V

    # counts of the number of proposed spin flips
    proposed_spin_flips = 0

    # counter for the number of accepted spin flips
    accepted_spin_flips = 0

    # Iterate over imaginary time τ=Δτ⋅l.
    for l in fermion_greens_calculator

        # Propagate equal-time Green's function matrix to current imaginary time G(τ±Δτ,τ±Δτ) ==> G(τ,τ)
        # depending on whether iterating over imaginary time in the forward or reverse direction
        propagate_equaltime_greens!(G, fermion_greens_calculator, B)

        # apply the transformation G̃(τ,τ) = exp(+Δτ⋅K[l]/2)⋅G(τ,τ)⋅exp(-Δτ⋅K[l]/2)
        # if B[l] = exp(-Δτ⋅K[l]/2)⋅exp(-Δτ⋅V[l])⋅exp(-Δτ⋅K[l]/2),
        # otherwise nothing when B[l] = exp(-Δτ⋅V[l])⋅exp(-Δτ⋅K[l])
        forward_partially_wrap_greens(G, B[l], G′)

        # shuffle the order in which orbitals/sites will be iterated over
        shuffle!(rng, update_perm)

        # iterate over orbitals in the lattice
        for i in update_perm

            # increment number of proposed spin flips
            proposed_spin_flips += 1

            # calculate the change in the bosonic action, only non-zero for attractive hubbard interactions
            ΔSb_il = -2 * α[i] * s[i,l]

            # get the site index associated with the current Hubbard interaction
            site = sites[i]

            # calculate new/proposed element in diagonal on-site energy matrix for spin up
            V_il′ = 2*α[i]/Δτ * s[i,l] + V[site,l]

            # calculate spin-up determinant ratio associated with Ising HS spin flip
            R_il, Δ_il = local_update_det_ratio(G, B[l], V_il′, site, Δτ)
            # calculate acceptance probability
            P_il = exp(-ΔSb_il) * abs2(R_il)

            # accept or reject proposed upate
            if rand(rng) < P_il

                # increment the cound of accepted spin flips
                accepted_spin_flips += 1

                # flip the spin
                s[i,l] = -s[i,l]

                # update diagonal on-site energy matrix
                V[site,l] = V_il′

                # update the spin-up and down Green's function
                logdetG, sgndetG = local_update_greens!(G, logdetG, sgndetG, B[l], R_il, Δ_il, site, u, v)
            end
        end

        # apply the transformation G(τ,τ) = exp(-Δτ⋅K[l]/2)⋅G̃(τ,τ)⋅exp(+Δτ⋅K[l]/2)
        # if B[l] = exp(-Δτ⋅K[l]/2)⋅exp(-Δτ⋅V[l])⋅exp(-Δτ⋅K[l]/2),
        # otherwise nothing when B[l] = exp(-Δτ⋅V[l])⋅exp(-Δτ⋅K[l])
        reverse_partially_wrap_greens(G, B[l], M)

        # Periodically re-calculate the Green's function matrix for numerical stability.
        logdetG, sgndetG, δG′, δθ′ = stabilize_equaltime_greens!(G, logdetG, sgndetG, fermion_greens_calculator, B, update_B̄=true)

        # record the max errors
        δG = maximum((δG, δG′))
        δθ = maximum(abs, (δθ, δθ′))
    end

    # update stabilization frequency if required
    (logdetG, sgndetG, δG, δθ) = update_stabalization_frequency!(
        G, logdetG, sgndetG,
        fermion_greens_calculator = fermion_greens_calculator,
        B = B, δG = δG, δθ = δθ, δG_max = δG_max
    )

    # calculate the acceptance rate
    acceptance_rate = accepted_spin_flips / proposed_spin_flips

    return (acceptance_rate, logdetG, sgndetG, δG, δθ)
end


@doc raw"""
    reflection_update!(Gup::Matrix{T}, logdetGup::E, sgndetGup::T,
                       Gdn::Matrix{T}, logdetGdn::E, sgndetGdn::T,
                       hubbard_ising_parameters::HubbardIsingHSParameters{E};
                       fermion_path_integral_up::FermionPathIntegral{T,E},
                       fermion_path_integral_dn::FermionPathIntegral{T,E},
                       fermion_greens_calculator_up::FermionGreensCalculator{T,E},
                       fermion_greens_calculator_dn::FermionGreensCalculator{T,E},
                       fermion_greens_calculator_up_alt::FermionGreensCalculator{T,E},
                       fermion_greens_calculator_dn_alt::FermionGreensCalculator{T,E},
                       Bup::Vector{P}, Bdn::Vector{P},
                       rng::AbstractRNG) where {T<:Number, E<:AbstractFloat, P<:AbstractPropagator{T,E}}

Perform a reflection update where the sign of every Ising Hubbard-Stratonovich field on a randomly chosen orbital in the lattice is changed.
This function returns `(accepted, logdetGup, sgndetGup, logdetGdn, sgndetGdn)`.

# Arguments

- `Gup::Matrix{T}`: Spin-up eqaul-time Greens function matrix.
- `logdetGup::E`: Log of the determinant of the spin-up eqaul-time Greens function matrix.
- `sgndetGup::T`: Sign/phase of the determinant of the spin-up eqaul-time Greens function matrix.
- `Gdn::Matrix{T}`: Spin-down eqaul-time Greens function matrix.
- `logdetGdn::E`: Log of the determinant of the spin-down eqaul-time Greens function matrix.
- `sgndetGdn::T`: Sign/phase of the determinant of the spin-down eqaul-time Greens function matrix.
- `hubbard_ising_parameters::HubbardIsingHSParameters{E}`: Ising Hubbard-Stratonovich fields and associated parameters to update.

# Keyword Arguments

- `fermion_path_integral_up::FermionPathIntegral{T,E}`: An instance of [`FermionPathIntegral`](@ref) type for spin-up electrons.
- `fermion_path_integral_dn::FermionPathIntegral{T,E}`: An instance of [`FermionPathIntegral`](@ref) type for spin-down electrons.
- `fermion_greens_calculator_up::FermionGreensCalculator{T,E}`: Contains matrix factorization information for current spin-up sector state.
- `fermion_greens_calculator_dn::FermionGreensCalculator{T,E}`: Contains matrix factorization information for current spin-down sector state.
- `fermion_greens_calculator_up_alt::FermionGreensCalculator{T,E}`: Used to calculate matrix factorizations for proposed spin-up sector state.
- `fermion_greens_calculator_dn_alt::FermionGreensCalculator{T,E}`: Used to calculate matrix factorizations for proposed spin-up sector state.
- `Bup::Vector{P}`: Spin-up propagators for each imaginary time slice.
- `Bdn::Vector{P}`: Spin-down propagators for each imaginary time slice.
- `rng::AbstractRNG`: Random number generator used in method instead of global random number generator, important for reproducibility.
"""
function reflection_update!(Gup::Matrix{T}, logdetGup::E, sgndetGup::T,
                            Gdn::Matrix{T}, logdetGdn::E, sgndetGdn::T,
                            hubbard_ising_parameters::HubbardIsingHSParameters{E};
                            fermion_path_integral_up::FermionPathIntegral{T,E},
                            fermion_path_integral_dn::FermionPathIntegral{T,E},
                            fermion_greens_calculator_up::FermionGreensCalculator{T,E},
                            fermion_greens_calculator_dn::FermionGreensCalculator{T,E},
                            fermion_greens_calculator_up_alt::FermionGreensCalculator{T,E},
                            fermion_greens_calculator_dn_alt::FermionGreensCalculator{T,E},
                            Bup::Vector{P}, Bdn::Vector{P},
                            rng::AbstractRNG) where {T<:Number, E<:AbstractFloat, P<:AbstractPropagator{T,E}}

    (; N, U, α, sites, s, Δτ) = hubbard_ising_parameters
    Gup′ = fermion_greens_calculator_up_alt.G′
    Gdn′ = fermion_greens_calculator_dn_alt.G′

    # pick a random site/orbital in lattice with finite Hubbard U to perform reflection update on
    i     = rand(rng, 1:N)
    site  = sites[i]
    s_i   = @view s[i, :]
    Vup_i = @view fermion_path_integral_up.V[site, :]
    Vdn_i = @view fermion_path_integral_dn.V[site, :]

    # reflect all the HS field on site i
    @. s_i = -s_i

    # calculate change in bosonic action, only non-zero for attractive Hubbard interaction
    ΔSb_i = U[i] > zero(E) ? zero(E) : 2 * α[i] * sum(s_i)

    # update diagonal on-site energy matrix
    @. Vup_i = -2*α[i]/Δτ * s_i + Vup_i
    @. Vdn_i = +2*α[i]/Δτ * s_i + Vdn_i

    # update propagator matrices
    @fastmath @inbounds for l in eachindex(Bup)
        expmΔτVup_l = Bup[l].expmΔτV::Vector{E}
        expmΔτVdn_l = Bdn[l].expmΔτV::Vector{E}
        expmΔτVup_l[site] = exp(-Δτ*Vup_i[l])
        expmΔτVdn_l[site] = exp(-Δτ*Vdn_i[l])
    end

    # calculate new Green's function matrices and determinant of new Green's function matrix
    logdetGup′, sgndetGup′ = calculate_equaltime_greens!(Gup′, fermion_greens_calculator_up_alt, Bup)
    logdetGdn′, sgndetGdn′ = calculate_equaltime_greens!(Gdn′, fermion_greens_calculator_dn_alt, Bdn)

    # calculate acceptance probability P = exp(-ΔS_b)⋅|det(Gup)/det(Gup′)|⋅|det(Gdn)/det(Gdn′)|
    #                                    = exp(-ΔS_b)⋅|det(Mup′)/det(Mup)|⋅|det(Mdn′)/det(Mdn)|
    P_i = exp(-ΔSb_i + logdetGup + logdetGdn - logdetGup′ - logdetGdn′)

    # accept or reject the update
    if rand(rng) < P_i
        logdetGup = logdetGup′
        logdetGdn = logdetGdn′
        sgndetGup = sgndetGup′
        sgndetGdn = sgndetGdn′
        copyto!(Gup, Gup′)
        copyto!(Gdn, Gdn′)
        copyto!(fermion_greens_calculator_up, fermion_greens_calculator_up_alt)
        copyto!(fermion_greens_calculator_dn, fermion_greens_calculator_dn_alt)
        accepted = true
    else
        # flip HS field back
        @. s_i = -s_i
        # revert diagonal on-site energy matrix
        @. Vup_i = -2*α[i]/Δτ * s_i + Vup_i
        @. Vdn_i = +2*α[i]/Δτ * s_i + Vdn_i
        # revert propagator matrices
        @fastmath @inbounds for l in eachindex(Bup)
            expmΔτVup_l = Bup[l].expmΔτV::Vector{E}
            expmΔτVdn_l = Bdn[l].expmΔτV::Vector{E}
            expmΔτVup_l[site] = exp(-Δτ*Vup_i[l])
            expmΔτVdn_l[site] = exp(-Δτ*Vdn_i[l])
        end
        accepted = false
    end

    return (accepted, logdetGup, sgndetGup, logdetGdn, sgndetGdn)
end

@doc raw"""
    reflection_update!(G::Matrix{T}, logdetG::E, sgndetG::T,
                       hubbard_ising_hs_params::HubbardIsingHSParameters{E};
                       fermion_path_integral::FermionPathIntegral{T,E},
                       fermion_greens_calculator::FermionGreensCalculator{T,E},
                       fermion_greens_calculator_alt::FermionGreensCalculator{T,E},
                       B::Vector{P},
                       rng::AbstractRNG) where {T<:Number, E<:AbstractFloat, P<:AbstractPropagator{T,E}}

Perform a reflection update where the sign of every Ising Hubbard-Stratonovich field on a randomly chosen orbital in the lattice is changed.
This function returns `(accepted, logdetG, sgndetG)`. This method assumes strictly attractive Hubbard interactions.

# Arguments

- `G::Matrix{T}`: Eqaul-time Greens function matrix.
- `logdetG::E`: Log of the determinant of the eqaul-time Greens function matrix.
- `sgndetG::T`: Sign/phase of the determinant of the eqaul-time Greens function matrix.
- `hubbard_ising_parameters::HubbardIsingHSParameters{E}`: Ising Hubbard-Stratonovich fields and associated parameters to update.

# Keyword Arguments

- `fermion_path_integral::FermionPathIntegral{T,E}`: An instance of [`FermionPathIntegral`](@ref) type.
- `fermion_greens_calculator::FermionGreensCalculator{T,E}`: Contains matrix factorization information for current state.
- `fermion_greens_calculator_alt::FermionGreensCalculator{T,E}`: Used to calculate matrix factorizations for proposed state.
- `B::Vector{P}`: Propagators for each imaginary time slice.
- `rng::AbstractRNG`: Random number generator used in method instead of global random number generator, important for reproducibility.
"""
function reflection_update!(G::Matrix{T}, logdetG::E, sgndetG::T,
                            hubbard_ising_hs_params::HubbardIsingHSParameters{E};
                            fermion_path_integral::FermionPathIntegral{T,E},
                            fermion_greens_calculator::FermionGreensCalculator{T,E},
                            fermion_greens_calculator_alt::FermionGreensCalculator{T,E},
                            B::Vector{P},
                            rng::AbstractRNG) where {T<:Number, E<:AbstractFloat, P<:AbstractPropagator{T,E}}

    (; N, U, α, s, Δτ) = hubbard_ising_hs_params
    G′ = fermion_greens_calculator_alt.G′

    # pick a random site/orbital in lattice with finite Hubbard U to perform reflection update on
    i    = rand(rng, 1:N)
    site = sites[i]
    s_i  = @view s[i, :]
    V_i  = @view fermion_path_integral.V[site, :]

    # reflect all the HS field on site i
    @. s_i = -s_i

    # calculate change in bosonic action, only non-zero for attractive Hubbard interaction
    ΔSb_i = 2 * α[i] * sum(s_i)

    # update diagonal on-site energy matrix
    @. V_i = -2*α[i]/Δτ * s_i + V_i

    # update propagator matrices
    @fastmath @inbounds for l in eachindex(B)
        expmΔτV_l = B[l].expmΔτV::Vector{E}
        expmΔτV_l[site] = exp(-Δτ*V_i[l])
    end

    # calculate new Green's function matrices and determinant of new Green's function matrix
    logdetG′, sgndetG′ = calculate_equaltime_greens!(G′, fermion_greens_calculator_alt, B)

    # calculate acceptance probability P = exp(-ΔS_b)⋅|det(G)/det(G′)|²
    #                                    = exp(-ΔS_b)⋅|det(M′)/det(M)|²
    P_i = exp(-ΔSb_i + 2*logdetG - 2*logdetG′)

    # accept or reject the update
    if rand(rng) < P_i
        logdetG = logdetG′
        sgndetG = sgndetG′
        copyto!(G, G′)
        copyto!(fermion_greens_calculator, fermion_greens_calculator_alt)
        accepted = true
    else
        # flip HS field back
        @. s_i = -s_i
        # revert diagonal on-site energy matrix
        @. V_i = -2*α[i]/Δτ * s_i + V_i
        # revert propagator matrices
        @fastmath @inbounds for l in eachindex(Bup)
            expmΔτV_l = B[l].expmΔτV::Vector{E}
            expmΔτV_l[site] = exp(-Δτ*V_i[l])
        end
        accepted = false
    end

    return (accepted, logdetG, sgndetG)
end


@doc raw"""
    swap_update!(Gup::Matrix{T}, logdetGup::E, sgndetGup::T,
                 Gdn::Matrix{T}, logdetGdn::E, sgndetGdn::T,
                 hubbard_ising_hs_params::HubbardIsingHSParameters{E};
                 fermion_path_integral_up::FermionPathIntegral{T,E},
                 fermion_path_integral_dn::FermionPathIntegral{T,E},
                 fermion_greens_calculator_up::FermionGreensCalculator{T,E},
                 fermion_greens_calculator_dn::FermionGreensCalculator{T,E},
                 fermion_greens_calculator_up_alt::FermionGreensCalculator{T,E},
                 fermion_greens_calculator_dn_alt::FermionGreensCalculator{T,E},
                 Bup::Vector{P}, Bdn::Vector{P},
                 rng::AbstractRNG) where {T<:Number, E<:AbstractFloat, P<:AbstractPropagator{T,E}}

Perform a swap update where the HS fields associated with two randomly chosen sites in the lattice are exchanged.
This function returns `(accepted, logdetGup, sgndetGup, logdetGdn, sgndetGdn)`.

# Arguments

- `Gup::Matrix{T}`: Spin-up eqaul-time Greens function matrix.
- `logdetGup::E`: Log of the determinant of the spin-up eqaul-time Greens function matrix.
- `sgndetGup::T`: Sign/phase of the determinant of the spin-up eqaul-time Greens function matrix.
- `Gdn::Matrix{T}`: Spin-down eqaul-time Greens function matrix.
- `logdetGdn::E`: Log of the determinant of the spin-down eqaul-time Greens function matrix.
- `sgndetGdn::T`: Sign/phase of the determinant of the spin-down eqaul-time Greens function matrix.
- `hubbard_ising_parameters::HubbardIsingHSParameters{E}`: Ising Hubbard-Stratonovich fields and associated parameters to update.

# Keyword Arguments

- `fermion_path_integral_up::FermionPathIntegral{T,E}`: An instance of [`FermionPathIntegral`](@ref) type for spin-up electrons.
- `fermion_path_integral_dn::FermionPathIntegral{T,E}`: An instance of [`FermionPathIntegral`](@ref) type for spin-down electrons.
- `fermion_greens_calculator_up::FermionGreensCalculator{T,E}`: Contains matrix factorization information for current spin-up sector state.
- `fermion_greens_calculator_dn::FermionGreensCalculator{T,E}`: Contains matrix factorization information for current spin-down sector state.
- `fermion_greens_calculator_up_alt::FermionGreensCalculator{T,E}`: Used to calculate matrix factorizations for proposed spin-up sector state.
- `fermion_greens_calculator_dn_alt::FermionGreensCalculator{T,E}`: Used to calculate matrix factorizations for proposed spin-up sector state.
- `Bup::Vector{P}`: Spin-up propagators for each imaginary time slice.
- `Bdn::Vector{P}`: Spin-down propagators for each imaginary time slice.
- `rng::AbstractRNG`: Random number generator used in method instead of global random number generator, important for reproducibility.
"""
function swap_update!(Gup::Matrix{T}, logdetGup::E, sgndetGup::T,
                      Gdn::Matrix{T}, logdetGdn::E, sgndetGdn::T,
                      hubbard_ising_hs_params::HubbardIsingHSParameters{E};
                      fermion_path_integral_up::FermionPathIntegral{T,E},
                      fermion_path_integral_dn::FermionPathIntegral{T,E},
                      fermion_greens_calculator_up::FermionGreensCalculator{T,E},
                      fermion_greens_calculator_dn::FermionGreensCalculator{T,E},
                      fermion_greens_calculator_up_alt::FermionGreensCalculator{T,E},
                      fermion_greens_calculator_dn_alt::FermionGreensCalculator{T,E},
                      Bup::Vector{P}, Bdn::Vector{P},
                      rng::AbstractRNG) where {T<:Number, E<:AbstractFloat, P<:AbstractPropagator{T,E}}

    (; N, U, α, sites, s, Δτ) = hubbard_ising_hs_params
    Gup′ = fermion_greens_calculator_up_alt.G′
    Gdn′ = fermion_greens_calculator_dn_alt.G′

    # ranomly pick two sites with Hubbard U interaction on them
    i = rand(rng, 1:N)
    j = i
    while j==i
        j = rand(rng, 1:N)
    end

    # get the site index associted with each Hubbard U
    site_i = sites[i]
    site_j = sites[j]

    # get the HS fields associated with each site
    s_i = @view s[i,:]
    s_j = @view s[j,:]
    Vup_i = @view fermion_path_integral_up.V[site_i, :]
    Vdn_i = @view fermion_path_integral_dn.V[site_i, :]
    Vup_j = @view fermion_path_integral_up.V[site_j, :]
    Vdn_j = @view fermion_path_integral_dn.V[site_j, :]

    # calculate the initial bosonic action
    Sb  = U[i] > zero(E) ? zero(E) : 2 * α[i] * sum(s_i)
    Sb += U[j] > zero(E) ? zero(E) : 2 * α[j] * sum(s_j)

    # swap the HS fields
    swap!(s_i, s_j)

    # calculate the final bosonic action
    Sb′  = U[i] > zero(E) ? zero(E) : 2 * α[i] * sum(s_i)
    Sb′ += U[j] > zero(E) ? zero(E) : 2 * α[j] * sum(s_j)

    # calculate the change in the bosonic action
    ΔSb = Sb′ - Sb

    # update diagonal on-site energy matrix
    @. Vup_i = Vup_i - α[i]/Δτ * (s_i - s_j) 
    @. Vdn_i = Vdn_i + α[i]/Δτ * (s_i - s_j) 
    @. Vup_j = Vup_j - α[j]/Δτ * (s_j - s_i) 
    @. Vdn_j = Vdn_j + α[j]/Δτ * (s_j - s_i)

    # update propagator matrices
    @fastmath @inbounds for l in eachindex(Bup)
        expmΔτVup_l = Bup[l].expmΔτV::Vector{E}
        expmΔτVdn_l = Bdn[l].expmΔτV::Vector{E}
        expmΔτVup_l[site_i] = exp(-Δτ*Vup_i[l])
        expmΔτVdn_l[site_i] = exp(-Δτ*Vdn_i[l])
        expmΔτVup_l[site_j] = exp(-Δτ*Vup_j[l])
        expmΔτVdn_l[site_j] = exp(-Δτ*Vdn_j[l])
    end

    # calculate new Green's function matrices and determinant of new Green's function matrix
    logdetGup′, sgndetGup′ = calculate_equaltime_greens!(Gup′, fermion_greens_calculator_up_alt, Bup)
    logdetGdn′, sgndetGdn′ = calculate_equaltime_greens!(Gdn′, fermion_greens_calculator_dn_alt, Bdn)

    # calculate acceptance probability P = exp(-ΔS_b)⋅|det(Gup)/det(Gup′)|⋅|det(Gdn)/det(Gdn′)|
    #                                    = exp(-ΔS_b)⋅|det(Mup′)/det(Mup)|⋅|det(Mdn′)/det(Mdn)|
    P_i = exp(-ΔSb + logdetGup + logdetGdn - logdetGup′ - logdetGdn′)

    # accept or reject the update
    if rand(rng) < P_i
        logdetGup = logdetGup′
        logdetGdn = logdetGdn′
        sgndetGup = sgndetGup′
        sgndetGdn = sgndetGdn′
        copyto!(Gup, Gup′)
        copyto!(Gdn, Gdn′)
        copyto!(fermion_greens_calculator_up, fermion_greens_calculator_up_alt)
        copyto!(fermion_greens_calculator_dn, fermion_greens_calculator_dn_alt)
        accepted = true
    else
        # flip HS fields back
        swap!(s_i, s_j)
        # revert diagonal on-site energy matrix
        @. Vup_i = Vup_i - α[i]/Δτ * (s_i - s_j) 
        @. Vdn_i = Vdn_i + α[i]/Δτ * (s_i - s_j) 
        @. Vup_j = Vup_j - α[j]/Δτ * (s_j - s_i) 
        @. Vdn_j = Vdn_j + α[j]/Δτ * (s_j - s_i)
        # revert propagator matrices
        @fastmath @inbounds for l in eachindex(Bup)
            expmΔτVup_l = Bup[l].expmΔτV::Vector{E}
            expmΔτVdn_l = Bdn[l].expmΔτV::Vector{E}
            expmΔτVup_l[site_i] = exp(-Δτ*Vup_i[l])
            expmΔτVdn_l[site_i] = exp(-Δτ*Vdn_i[l])
            expmΔτVup_l[site_j] = exp(-Δτ*Vup_j[l])
            expmΔτVdn_l[site_j] = exp(-Δτ*Vdn_j[l])
        end
        accepted = false
    end

    return (accepted, logdetGup, sgndetGup, logdetGdn, sgndetGdn)
end


@doc raw"""
    swap_update!(G::Matrix{T}, logdetG::E, sgndetG::T,
                 hubbard_ising_hs_params::HubbardIsingHSParameters{E};
                 fermion_path_integral::FermionPathIntegral{T,E},
                 fermion_greens_calculator::FermionGreensCalculator{T,E},
                 fermion_greens_calculator_alt::FermionGreensCalculator{T,E},
                 B::Vector{P}, rng::AbstractRNG) where {T<:Number, E<:AbstractFloat, P<:AbstractPropagator{T,E}}

For strictly attractive Hubbard interactions, perform a swap update where the HS fields associated with two randomly chosen
sites in the lattice are exchanged. This function returns `(accepted, logdetG, sgndetG)`.

# Arguments

- `G::Matrix{T}`: Eqaul-time Greens function matrix.
- `logdetG::E`: Log of the determinant of the eqaul-time Greens function matrix.
- `sgndetG::T`: Sign/phase of the determinant of the eqaul-time Greens function matrix.
- `hubbard_ising_parameters::HubbardIsingHSParameters{E}`: Ising Hubbard-Stratonovich fields and associated parameters to update.

# Keyword Arguments

- `fermion_path_integral::FermionPathIntegral{T,E}`: An instance of [`FermionPathIntegral`](@ref) type.
- `fermion_greens_calculator::FermionGreensCalculator{T,E}`: Contains matrix factorization information for current state.
- `fermion_greens_calculator_alt::FermionGreensCalculator{T,E}`: Used to calculate matrix factorizations for proposed state.
- `B::Vector{P}`: Propagators for each imaginary time slice.
- `rng::AbstractRNG`: Random number generator used in method instead of global random number generator, important for reproducibility.
"""
function swap_update!(G::Matrix{T}, logdetG::E, sgndetG::T,
                      hubbard_ising_hs_params::HubbardIsingHSParameters{E};
                      fermion_path_integral::FermionPathIntegral{T,E},
                      fermion_greens_calculator::FermionGreensCalculator{T,E},
                      fermion_greens_calculator_alt::FermionGreensCalculator{T,E},
                      B::Vector{P}, rng::AbstractRNG) where {T<:Number, E<:AbstractFloat, P<:AbstractPropagator{T,E}}

    (; N, U, α, sites, s, Δτ) = hubbard_ising_hs_params
    G′ = fermion_greens_calculator_alt.G′

    # ranomly pick two sites with Hubbard U interaction on them
    i = rand(rng, 1:N)
    j = i
    while j==i
        j = rand(rng, 1:N)
    end

    # get the site index associted with each Hubbard U
    site_i = sites[i]
    site_j = sites[j]

    # get the HS fields associated with each site
    s_i = @view s[i,:]
    s_j = @view s[j,:]
    V_i = @view fermion_path_integral.V[site_i, :]
    V_j = @view fermion_path_integral.V[site_j, :]

    # calculate the initial bosonic action
    Sb  = U[i] > zero(E) ? zero(E) : 2 * α[i] * sum(s_i)
    Sb += U[j] > zero(E) ? zero(E) : 2 * α[j] * sum(s_j)

    # swap the HS fields
    swap!(s_i, s_j)

    # calculate the final bosonic action
    Sb′  = U[i] > zero(E) ? zero(E) : 2 * α[i] * sum(s_i)
    Sb′ += U[j] > zero(E) ? zero(E) : 2 * α[j] * sum(s_j)

    # calculate the change in the bosonic action
    ΔSb = Sb′ - Sb

    # update diagonal on-site energy matrix
    @. V_i = V_i - α[i]/Δτ * (s_i - s_j) 
    @. V_j = V_j - α[j]/Δτ * (s_j - s_i) 

    # update propagator matrices
    @fastmath @inbounds for l in eachindex(Bup)
        expmΔτV_l = B[l].expmΔτV::Vector{E}
        expmΔτV_l[site_i] = exp(-Δτ*V_i[l])
        expmΔτV_l[site_j] = exp(-Δτ*V_j[l])
    end

    # calculate new Green's function matrices and determinant of new Green's function matrix
    logdetG′, sgndetG′ = calculate_equaltime_greens!(G′, fermion_greens_calculator_alt, B)

    # calculate acceptance probability P = exp(-ΔS_b)⋅|det(G)/det(G′)|²
    #                                    = exp(-ΔS_b)⋅|det(M′)/det(M)|²
    P_i = exp(-ΔSb + 2*logdetG - 2*logdetG′)

    # accept or reject the update
    if rand(rng) < P_i
        logdetG = logdetG′
        sgndetG = sgndetG′
        copyto!(G, G′)
        copyto!(fermion_greens_calculator, fermion_greens_calculator_alt)
        accepted = true
    else
        # flip HS fields back
        swap!(s_i, s_j)
        # revert diagonal on-site energy matrix
        @. V_i = V_i - α[i]/Δτ * (s_i - s_j) 
        @. V_j = V_j - α[j]/Δτ * (s_j - s_i) 
        # revert propagator matrices
        @fastmath @inbounds for l in eachindex(Bup)
            expmΔτV_l = B[l].expmΔτV::Vector{E}
            expmΔτV_l[site_i] = exp(-Δτ*V_i[l])
            expmΔτV_l[site_j] = exp(-Δτ*V_j[l])
        end
        accepted = false
    end

    return (accepted, logdetG, sgndetG)
end