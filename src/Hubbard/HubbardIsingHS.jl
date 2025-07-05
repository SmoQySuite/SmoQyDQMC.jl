@doc raw"""
    HubbardIsingHSParameters{T<:AbstractFloat, H<:Number}

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
struct HubbardIsingHSParameters{T<:AbstractFloat, H<:Number}

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
    HubbardIsingHSParameters(;
        β::E, Δτ::E,
        hubbard_parameters::HubbardParameters{E},
        rng::AbstractRNG
    ) where {E<:AbstractFloat}

Initialize and return an instance of the [`HubbardIsingHSParameters`](@ref) type.
Note that on-site energies `fpi.V` are modified by ``-U_i/2`` if ``hm.ph_sym_form = false``.
"""
function HubbardIsingHSParameters(;
    β::E, Δτ::E,
    hubbard_parameters::HubbardParameters{E},
    rng::AbstractRNG
) where {E<:AbstractFloat}

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
    initialize!(
        fermion_path_integral_up::FermionPathIntegral,
        fermion_path_integral_dn::FermionPathIntegral,
        hubbard_ising_parameters::HubbardIsingHSParameters
    )

Initialize the contribution from the Hubbard interaction to the [`FermionPathIntegral`](@ref)
instance `fermion_path_integral_up` for spin up and `fermion_path_integral_dn` spin down.
"""
function initialize!(
    fermion_path_integral_up::FermionPathIntegral,
    fermion_path_integral_dn::FermionPathIntegral,
    hubbard_ising_parameters::HubbardIsingHSParameters
)

    @assert fermion_path_integral_up.Sb == fermion_path_integral_dn.Sb "$(fermion_path_integral_up.Sb) ≠ $(fermion_path_integral_dn.Sb)"
    
    (; α, U, Δτ, s, sites) = hubbard_ising_parameters
    Vup = fermion_path_integral_up.V
    Vdn = fermion_path_integral_dn.V

    # add Ising HS field contribution to diagonal on-site energy matrices
    for l in axes(Vup,2)
        for i in eachindex(sites)
            site = sites[i]
            Vup[site,l] = Vup[site,l] - α[i]/Δτ * s[i,l]
            Vdn[site,l] = Vdn[site,l] + sign(U[i]) * α[i]/Δτ * s[i,l]
        end
    end

    # update bosonic action
    Sb_hub = bosonic_action(hubbard_ising_parameters)
    fermion_path_integral_up.Sb += Sb_hub
    fermion_path_integral_dn.Sb += Sb_hub

    return nothing
end


@doc raw"""
    initialize!(
        fermion_path_integral::FermionPathIntegral,
        hubbard_ising_parameters::HubbardIsingHSParameters
    )

Initialize the contribution from an attractive Hubbard interaction to the [`FermionPathIntegral`](@ref)
instance `fermion_path_integral`.
"""
function initialize!(
    fermion_path_integral::FermionPathIntegral,
    hubbard_ising_parameters::HubbardIsingHSParameters
)
    
    (; α, U, Δτ, s, sites) = hubbard_ising_parameters
    V = fermion_path_integral.V

    # make sure its a strictly attractive hubbard interaction
    @assert all(u -> u < 0.0, U)

    # add Ising HS field contribution to diagonal on-site energy matrices
    for l in axes(V,2)
        for i in eachindex(sites)
            site = sites[i]
            V[site,l] = V[site,l] - α[i]/Δτ * s[i,l]
        end
    end

    # update bosonic action
    fermion_path_integral.Sb += bosonic_action(hubbard_ising_parameters)

    return nothing
end


@doc raw"""
    local_updates!(
        # ARGUMENTS
        Gup::Matrix{H}, logdetGup::R, sgndetGup::H,
        Gdn::Matrix{H}, logdetGdn::R, sgndetGdn::H,
        hubbard_ising_parameters::HubbardIsingHSParameters{R};
        # KEYWORD ARGUMENTS
        fermion_path_integral_up::FermionPathIntegral{H},
        fermion_path_integral_dn::FermionPathIntegral{H},
        fermion_greens_calculator_up::FermionGreensCalculator{H},
        fermion_greens_calculator_dn::FermionGreensCalculator{H},
        Bup::Vector{P}, Bdn::Vector{P},
        δG::R, δθ::R,  rng::AbstractRNG,
        δG_max::R = 1e-6,
        update_stabilization_frequency::Bool = true
    ) where {H<:Number, R<:Real, P<:AbstractPropagator}

Sweep through every imaginary time slice and orbital in the lattice, peforming local updates to every
Ising Hubbard-Stratonovich (HS) field.

This method returns a tuple containing `(acceptance_rate, logdetGup, sgndetGup, logdetGdn, sgndetGdn, δG, δθ)`.

# Arguments

- `Gup::Matrix{H}`: Spin-up equal-time Green's function matrix.
- `logdetGup::R`: The log of the absolute value of the determinant of the spin-up equal-time Green's function matrix, ``\log \vert \det G_\uparrow(\tau,\tau) \vert.``
- `sgndetGup::H`: The sign/phase of the determinant of the spin-up equal-time Green's function matrix, ``\det G_\uparrow(\tau,\tau) / \vert \det G_\uparrow(\tau,\tau) \vert.``
- `Gdn::Matrix{H}`: Spin-down equal-time Green's function matrix.
- `logdetGdn::R`: The log of the absolute value of the determinant of the spin-down equal-time Green's function matrix, ``\log \vert \det G_\downarrow(\tau,\tau) \vert.``
- `sgndetGdn::H`: The sign/phase of the determinant of the spin-down equal-time Green's function matrix, ``\det G_\downarrow(\tau,\tau) / \vert \det G_\downarrow(\tau,\tau) \vert.``
- `hubbard_ising_parameters::HubbardIsingHSParameters{R}`: Ising Hubbard-Stratonovich fields and associated parameters to update.

## Keyword Arguments

- `fermion_path_integral_up::FermionPathIntegral{H}`: An instance of the [`FermionPathIntegral`](@ref) type for spin-up electrons.
- `fermion_path_integral_dn::FermionPathIntegral{H}`: An instance of the [`FermionPathIntegral`](@ref) type for spin-down electrons.
- `fermion_greens_calculator_up::FermionGreensCalculator{H}`: An instance of the [`FermionGreensCalculator`](https://smoqysuite.github.io/JDQMCFramework.jl/stable/api/#JDQMCFramework.FermionGreensCalculator) type for the spin-up electrons.
- `fermion_greens_calculator_dn::FermionGreensCalculator{H}`: An instance of the [`FermionGreensCalculator`](https://smoqysuite.github.io/JDQMCFramework.jl/stable/api/#JDQMCFramework.FermionGreensCalculator) type for the spin-down electrons.
- `Bup::Vector{P}`: Spin-up propagators for each imaginary time slice.
- `Bdn::Vector{P}`: Spin-dn propagators for each imaginary time slice.
- `δG_max::R`: Maximum allowed error corrected by numerical stabilization.
- `δG::R`: Previously recorded maximum error in the Green's function corrected by numerical stabilization.
- `δθ::R`: Previously recorded maximum error in the sign/phase of the determinant of the equal-time Green's function matrix corrected by numerical stabilization.
- `rng::AbstractRNG`: Random number generator used in method instead of global random number generator, important for reproducibility.
- `update_stabilization_frequency::Bool = true`: If true, allows the stabilization frequency `n_stab` to be dynamically adjusted.
"""
function local_updates!(
    # ARGUMENTS
    Gup::Matrix{H}, logdetGup::R, sgndetGup::H,
    Gdn::Matrix{H}, logdetGdn::R, sgndetGdn::H,
    hubbard_ising_parameters::HubbardIsingHSParameters{R};
    # KEYWORD ARGUMENTS
    fermion_path_integral_up::FermionPathIntegral{H},
    fermion_path_integral_dn::FermionPathIntegral{H},
    fermion_greens_calculator_up::FermionGreensCalculator{H},
    fermion_greens_calculator_dn::FermionGreensCalculator{H},
    Bup::Vector{P}, Bdn::Vector{P},
    δG::R, δθ::R,  rng::AbstractRNG,
    δG_max::R = 1e-6,
    update_stabilization_frequency::Bool = true
) where {H<:Number, R<:Real, P<:AbstractPropagator}

    # make sure bosonic actions match
    @assert fermion_path_integral_up.Sb == fermion_path_integral_dn.Sb "$(fermion_path_integral_up.Sb) ≠ $(fermion_path_integral_dn.Sb)"

    (; Δτ, U, α, sites, s, update_perm) = hubbard_ising_parameters
    u = @view fermion_path_integral_up.u[:,1]
    v = @view fermion_path_integral_up.v[:,1]

    # get temporary storage matrix
    G′ = fermion_greens_calculator_up.G′

    # get on-site energy matrices for spin up and down electrons for all time slices
    Vup = fermion_path_integral_up.V
    Vdn = fermion_path_integral_dn.V

    # counts of the number of proposed spin flips
    proposed_spin_flips = 0

    # counter for the number of accepted spin flips
    accepted_spin_flips = 0

    # calculate initial bosonic action
    Sb_init = bosonic_action(hubbard_ising_parameters)

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
        partially_wrap_greens_reverse!(Gup, Bup_l, G′)
        partially_wrap_greens_reverse!(Gdn, Bdn_l, G′)

        # shuffle the order in which orbitals/sites will be iterated over
        shuffle!(rng, update_perm)

        # iterate over orbitals in the lattice
        for i in update_perm

            # increment number of proposed spin flips
            proposed_spin_flips += 1

            # calculate the change in the bosonic action, only non-zero for attractive hubbard interactions
            if U[i] < 0.0
                ΔSb_il = -2 * α[i] * s[i,l] # (α⋅[-s]) - (α⋅[s])
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
        partially_wrap_greens_forward!(Gup, Bup_l, G′)
        partially_wrap_greens_forward!(Gdn, Bdn_l, G′)

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
    if update_stabilization_frequency
        (updated, logdetGup, sgndetGup, logdetGdn, sgndetGdn, δG, δθ) = update_stabilization_frequency!(
            Gup, logdetGup, sgndetGup,
            Gdn, logdetGdn, sgndetGdn,
            fermion_greens_calculator_up = fermion_greens_calculator_up,
            fermion_greens_calculator_dn = fermion_greens_calculator_dn,
            Bup = Bup, Bdn = Bdn, δG = δG, δθ = δθ, δG_max = δG_max
        )
    end

    # calculate the acceptance rate
    acceptance_rate = accepted_spin_flips / proposed_spin_flips

    # calculate finale bosonic action
    Sb_final = bosonic_action(hubbard_ising_parameters)

    # update total bosonic action
    ΔSb = Sb_final - Sb_init
    fermion_path_integral_up.Sb += ΔSb
    fermion_path_integral_dn.Sb += ΔSb

    return (acceptance_rate, logdetGup, sgndetGup, logdetGdn, sgndetGdn, δG, δθ)
end


@doc raw"""
    local_updates!(
        # ARGUMENTS
        G::Matrix{H}, logdetG::R, sgndetG::H,
        hubbard_ising_parameters::HubbardIsingHSParameters{R};
        # KEYWORD ARGUMENTS
        fermion_path_integral::FermionPathIntegral{H},
        fermion_greens_calculator::FermionGreensCalculator{H,R},
        B::Vector{P}, δG::R, δθ::R, rng::AbstractRNG,
        δG_max::R = 1e-6,
        update_stabilization_frequency::Bool = true
    ) where {H<:Number, R<:Real, P<:AbstractPropagator}

Sweep through every imaginary time slice and orbital in the lattice, performing local updates to every
Ising Hubbard-Stratonovich (HS) field, assuming strictly attractive Hubbard interactions and perfect spin symmetry.

This method returns the a tuple containing `(acceptance_rate, logdetG, sgndetG, δG, δθ)`.

# Arguments

- `G::Matrix{H}`: Equal-time Green's function matrix.
- `logdetG::R`: The log of the absolute value of the determinant of theequal-time Green's function matrix, ``\log \vert \det G_\uparrow(\tau,\tau) \vert.``
- `sgndetG::H`: The sign/phase of the determinant of the equal-time Green's function matrix, ``\det G_\uparrow(\tau,\tau) / \vert \det G_\uparrow(\tau,\tau) \vert.``
- `hubbard_ising_parameters::HubbardIsingHSParameters{R}`: Ising Hubbard-Stratonovich fields and associated parameters to update.

## Keyword Arguments

- `fermion_path_integral::FermionPathIntegral{H}`: An instance of the [`FermionPathIntegral`](@ref) type.
- `fermion_greens_calculator::FermionGreensCalculator{H,R}`: An instance of the [`FermionGreensCalculator`](https://smoqysuite.github.io/JDQMCFramework.jl/stable/api/#JDQMCFramework.FermionGreensCalculator) type.
- `B::Vector{P}`: Propagators for each imaginary time slice.
- `δG_max::R`: Maximum allowed error corrected by numerical stabilization.
- `δG::R`: Previously recorded maximum error in the Green's function corrected by numerical stabilization.
- `δθ::R`: Previously recorded maximum error in the sign/phase of the determinant of the equal-time Green's function matrix corrected by numerical stabilization.
- `rng::AbstractRNG`: Random number generator used in method instead of global random number generator, important for reproducibility.
- `update_stabilization_frequency::Bool=true`:  If true, allows the stabilization frequency `n_stab` to be dynamically adjusted.
"""
function local_updates!(
    # ARGUMENTS
    G::Matrix{H}, logdetG::R, sgndetG::H,
    hubbard_ising_parameters::HubbardIsingHSParameters{R};
    # KEYWORD ARGUMENTS
    fermion_path_integral::FermionPathIntegral{H},
    fermion_greens_calculator::FermionGreensCalculator{H,R},
    B::Vector{P}, δG::R, δθ::R, rng::AbstractRNG,
    δG_max::R = 1e-6,
    update_stabilization_frequency::Bool = true
) where {H<:Number, R<:Real, P<:AbstractPropagator}

    (; update_perm, U, α, sites, s, Δτ) = hubbard_ising_parameters
    u = @view fermion_path_integral.u[:,1]
    v = @view fermion_path_integral.v[:,1]

    # get temporary storage matrix
    G′ = fermion_greens_calculator.G′

    # get on-site energy matrices for spin up and down electrons for all time slices
    V = fermion_path_integral.V

    # counts of the number of proposed spin flips
    proposed_spin_flips = 0

    # counter for the number of accepted spin flips
    accepted_spin_flips = 0

    # calculate initial bosonic action
    Sb_init = bosonic_action(hubbard_ising_parameters)

    # Iterate over imaginary time τ=Δτ⋅l.
    for l in fermion_greens_calculator

        # Propagate equal-time Green's function matrix to current imaginary time G(τ±Δτ,τ±Δτ) ==> G(τ,τ)
        # depending on whether iterating over imaginary time in the forward or reverse direction
        propagate_equaltime_greens!(G, fermion_greens_calculator, B)

        # apply the transformation G̃(τ,τ) = exp(+Δτ⋅K[l]/2)⋅G(τ,τ)⋅exp(-Δτ⋅K[l]/2)
        # if B[l] = exp(-Δτ⋅K[l]/2)⋅exp(-Δτ⋅V[l])⋅exp(-Δτ⋅K[l]/2),
        # otherwise nothing when B[l] = exp(-Δτ⋅V[l])⋅exp(-Δτ⋅K[l])
        partially_wrap_greens_reverse!(G, B[l], G′)

        # shuffle the order in which orbitals/sites will be iterated over
        shuffle!(rng, update_perm)

        # iterate over orbitals in the lattice
        for i in update_perm

            # increment number of proposed spin flips
            proposed_spin_flips += 1

            # calculate the change in the bosonic action, only non-zero for attractive hubbard interactions
            ΔSb_il = -2 * α[i] * s[i,l] # (α⋅[-s]) - (α⋅[s])

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
        partially_wrap_greens_forward!(G, B[l], G′)

        # Periodically re-calculate the Green's function matrix for numerical stability.
        logdetG, sgndetG, δG′, δθ′ = stabilize_equaltime_greens!(G, logdetG, sgndetG, fermion_greens_calculator, B, update_B̄=true)

        # record the max errors
        δG = maximum((δG, δG′))
        δθ = maximum(abs, (δθ, δθ′))
    end

    # update stabilization frequency if required
    if update_stabilization_frequency
        (updated, logdetG, sgndetG, δG, δθ) = update_stabilization_frequency!(
            G, logdetG, sgndetG,
            fermion_greens_calculator = fermion_greens_calculator,
            B = B, δG = δG, δθ = δθ, δG_max = δG_max
        )
    end

    # calculate the acceptance rate
    acceptance_rate = accepted_spin_flips / proposed_spin_flips

    # calculate finale bosonic action
    Sb_final = bosonic_action(hubbard_ising_parameters)

    # update total bosonic action
    fermion_path_integral.Sb += (Sb_final - Sb_init)

    return (acceptance_rate, logdetG, sgndetG, δG, δθ)
end


@doc raw"""
    reflection_update!(
        # ARGUMENTS
        Gup::Matrix{H}, logdetGup::R, sgndetGup::H,
        Gdn::Matrix{H}, logdetGdn::R, sgndetGdn::H,
        hubbard_ising_parameters::HubbardIsingHSParameters{R};
        # KEYWORD ARGUMENTS
        fermion_path_integral_up::FermionPathIntegral{H},
        fermion_path_integral_dn::FermionPathIntegral{H},
        fermion_greens_calculator_up::FermionGreensCalculator{H,R},
        fermion_greens_calculator_dn::FermionGreensCalculator{H,R},
        fermion_greens_calculator_up_alt::FermionGreensCalculator{H,R},
        fermion_greens_calculator_dn_alt::FermionGreensCalculator{H,R},
        Bup::Vector{P}, Bdn::Vector{P},
        rng::AbstractRNG
    ) where {H<:Number, R<:Real, P<:AbstractPropagator}

Perform a reflection update where the sign of every Ising Hubbard-Stratonovich field on a randomly chosen orbital in the lattice is changed.
This function returns `(accepted, logdetGup, sgndetGup, logdetGdn, sgndetGdn)`.

# Arguments

- `Gup::Matrix{H}`: Spin-up eqaul-time Greens function matrix.
- `logdetGup::R`: Log of the determinant of the spin-up eqaul-time Greens function matrix.
- `sgndetGup::H`: Sign/phase of the determinant of the spin-up eqaul-time Greens function matrix.
- `Gdn::Matrix{H}`: Spin-down eqaul-time Greens function matrix.
- `logdetGdn::R`: Log of the determinant of the spin-down eqaul-time Greens function matrix.
- `sgndetGdn::H`: Sign/phase of the determinant of the spin-down eqaul-time Greens function matrix.
- `hubbard_ising_parameters::HubbardIsingHSParameters{R}`: Ising Hubbard-Stratonovich fields and associated parameters to update.

# Keyword Arguments

- `fermion_path_integral_up::FermionPathIntegral{H}`: An instance of [`FermionPathIntegral`](@ref) type for spin-up electrons.
- `fermion_path_integral_dn::FermionPathIntegral{H}`: An instance of [`FermionPathIntegral`](@ref) type for spin-down electrons.
- `fermion_greens_calculator_up::FermionGreensCalculator{H,R}`: Contains matrix factorization information for current spin-up sector state.
- `fermion_greens_calculator_dn::FermionGreensCalculator{H,R}`: Contains matrix factorization information for current spin-down sector state.
- `fermion_greens_calculator_up_alt::FermionGreensCalculator{H,R}`: Used to calculate matrix factorizations for proposed spin-up sector state.
- `fermion_greens_calculator_dn_alt::FermionGreensCalculator{H,R}`: Used to calculate matrix factorizations for proposed spin-down sector state.
- `Bup::Vector{P}`: Spin-up propagators for each imaginary time slice.
- `Bdn::Vector{P}`: Spin-down propagators for each imaginary time slice.
- `rng::AbstractRNG`: Random number generator used in method instead of global random number generator, important for reproducibility.
"""
function reflection_update!(
    # ARGUMENTS
    Gup::Matrix{H}, logdetGup::R, sgndetGup::H,
    Gdn::Matrix{H}, logdetGdn::R, sgndetGdn::H,
    hubbard_ising_parameters::HubbardIsingHSParameters{R};
    # KEYWORD ARGUMENTS
    fermion_path_integral_up::FermionPathIntegral{H},
    fermion_path_integral_dn::FermionPathIntegral{H},
    fermion_greens_calculator_up::FermionGreensCalculator{H,R},
    fermion_greens_calculator_dn::FermionGreensCalculator{H,R},
    fermion_greens_calculator_up_alt::FermionGreensCalculator{H,R},
    fermion_greens_calculator_dn_alt::FermionGreensCalculator{H,R},
    Bup::Vector{P}, Bdn::Vector{P},
    rng::AbstractRNG
) where {H<:Number, R<:Real, P<:AbstractPropagator}

    @assert fermion_path_integral_up.Sb == fermion_path_integral_dn.Sb "$(fermion_path_integral_up.Sb) ≠ $(fermion_path_integral_dn.Sb)"

    (; N, U, α, sites, s, Δτ) = hubbard_ising_parameters
    Gup′ = fermion_greens_calculator_up_alt.G′
    Gdn′ = fermion_greens_calculator_dn_alt.G′

    # make sure stabilization frequencies match
    if fermion_greens_calculator_up.n_stab != fermion_greens_calculator_up_alt.n_stab
        resize!(fermion_greens_calculator_up_alt, fermion_greens_calculator_up.n_stab)
    end

    # make sure stabilization frequencies match
    if fermion_greens_calculator_dn.n_stab != fermion_greens_calculator_dn_alt.n_stab
        resize!(fermion_greens_calculator_dn_alt, fermion_greens_calculator_dn.n_stab)
    end

    # pick a random site/orbital in lattice with finite Hubbard U to perform reflection update on
    i     = rand(rng, 1:N)
    site  = sites[i]
    s_i   = @view s[i, :]
    Vup_i = @view fermion_path_integral_up.V[site, :]
    Vdn_i = @view fermion_path_integral_dn.V[site, :]

    # reflect all the HS field on site i
    @. s_i = -s_i

    # calculate change in bosonic action, only non-zero for attractive Hubbard interaction
    ΔSb = U[i] > zero(R) ? zero(R) : 2 * α[i] * sum(s_i)

    # update diagonal on-site energy matrix
    @. Vup_i = -2*α[i]/Δτ * s_i + Vup_i
    @. Vdn_i = sign(U[i])*2*α[i]/Δτ * s_i + Vdn_i

    # update propagator matrices
    @fastmath @inbounds for l in eachindex(Bup)
        expmΔτVup_l = Bup[l].expmΔτV
        expmΔτVdn_l = Bdn[l].expmΔτV
        expmΔτVup_l[site] = exp(-Δτ*Vup_i[l])
        expmΔτVdn_l[site] = exp(-Δτ*Vdn_i[l])
    end

    # calculate new Green's function matrices and determinant of new Green's function matrix
    logdetGup′, sgndetGup′ = calculate_equaltime_greens!(Gup′, fermion_greens_calculator_up_alt, Bup)
    logdetGdn′, sgndetGdn′ = calculate_equaltime_greens!(Gdn′, fermion_greens_calculator_dn_alt, Bdn)

    # calculate acceptance probability P = exp(-ΔS_b)⋅|det(Gup)/det(Gup′)|⋅|det(Gdn)/det(Gdn′)|
    #                                    = exp(-ΔS_b)⋅|det(Mup′)/det(Mup)|⋅|det(Mdn′)/det(Mdn)|
    if isfinite(logdetGup′) && isfinite(logdetGdn′)
        P_i = exp(-ΔSb + logdetGup + logdetGdn - logdetGup′ - logdetGdn′)
    else
        P_i = 0.0
    end

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
        fermion_path_integral_up.Sb += ΔSb
        fermion_path_integral_dn.Sb += ΔSb
        accepted = true
    else
        # flip HS field back
        @. s_i = -s_i
        # revert diagonal on-site energy matrix
        @. Vup_i = -2*α[i]/Δτ * s_i + Vup_i
        @. Vdn_i = sign(U[i])*2*α[i]/Δτ * s_i + Vdn_i
        # revert propagator matrices
        @fastmath @inbounds for l in eachindex(Bup)
            expmΔτVup_l = Bup[l].expmΔτV
            expmΔτVdn_l = Bdn[l].expmΔτV
            expmΔτVup_l[site] = exp(-Δτ*Vup_i[l])
            expmΔτVdn_l[site] = exp(-Δτ*Vdn_i[l])
        end
        accepted = false
    end

    return (accepted, logdetGup, sgndetGup, logdetGdn, sgndetGdn)
end

@doc raw"""
    reflection_update!(
        # ARGUMENTS
        G::Matrix{H}, logdetG::R, sgndetG::H,
        hubbard_ising_parameters::HubbardIsingHSParameters{R};
        # KEYWORD ARGUMENTS
        fermion_path_integral::FermionPathIntegral{H},
        fermion_greens_calculator::FermionGreensCalculator{H,R},
        fermion_greens_calculator_alt::FermionGreensCalculator{H,R},
        B::Vector{P},
        rng::AbstractRNG
    ) where {H<:Number, R<:Real, P<:AbstractPropagator}

Perform a reflection update where the sign of every Ising Hubbard-Stratonovich field on a randomly chosen orbital in the lattice is changed.
This function returns `(accepted, logdetG, sgndetG)`. This method assumes strictly attractive Hubbard interactions.

# Arguments

- `G::Matrix{H}`: Eqaul-time Greens function matrix.
- `logdetG::R`: Log of the determinant of the eqaul-time Greens function matrix.
- `sgndetG::H`: Sign/phase of the determinant of the eqaul-time Greens function matrix.
- `hubbard_ising_parameters::HubbardIsingHSParameters{R}`: Ising Hubbard-Stratonovich fields and associated parameters to update.

# Keyword Arguments

- `fermion_path_integral::FermionPathIntegral{H}`: An instance of [`FermionPathIntegral`](@ref) type.
- `fermion_greens_calculator::FermionGreensCalculator{H,R}`: Contains matrix factorization information for current state.
- `fermion_greens_calculator_alt::FermionGreensCalculator{H,R}`: Used to calculate matrix factorizations for proposed state.
- `B::Vector{P}`: Propagators for each imaginary time slice.
- `rng::AbstractRNG`: Random number generator used in method instead of global random number generator, important for reproducibility.
"""
function reflection_update!(
    # ARGUMENTS
    G::Matrix{H}, logdetG::R, sgndetG::H,
    hubbard_ising_parameters::HubbardIsingHSParameters{R};
    # KEYWORD ARGUMENTS
    fermion_path_integral::FermionPathIntegral{H},
    fermion_greens_calculator::FermionGreensCalculator{H,R},
    fermion_greens_calculator_alt::FermionGreensCalculator{H,R},
    B::Vector{P},
    rng::AbstractRNG
) where {H<:Number, R<:Real, P<:AbstractPropagator}

    (; N, U, α, sites, s, Δτ) = hubbard_ising_parameters
    G′ = fermion_greens_calculator_alt.G′

    # make sure stabilization frequencies match
    if fermion_greens_calculator.n_stab != fermion_greens_calculator_alt.n_stab
        resize!(fermion_greens_calculator_alt, fermion_greens_calculator.n_stab)
    end

    # pick a random site/orbital in lattice with finite Hubbard U to perform reflection update on
    i    = rand(rng, 1:N)
    site = sites[i]
    s_i  = @view s[i, :]
    V_i  = @view fermion_path_integral.V[site, :]

    # reflect all the HS field on site i
    @. s_i = -s_i

    # calculate change in bosonic action, only non-zero for attractive Hubbard interaction
    ΔSb = 2 * α[i] * sum(s_i)

    # update diagonal on-site energy matrix
    @. V_i = -2*α[i]/Δτ * s_i + V_i

    # update propagator matrices
    @fastmath @inbounds for l in eachindex(B)
        expmΔτV_l = B[l].expmΔτV
        expmΔτV_l[site] = exp(-Δτ*V_i[l])
    end

    # calculate new Green's function matrices and determinant of new Green's function matrix
    logdetG′, sgndetG′ = calculate_equaltime_greens!(G′, fermion_greens_calculator_alt, B)

    # calculate acceptance probability P = exp(-ΔS_b)⋅|det(G)/det(G′)|²
    #                                    = exp(-ΔS_b)⋅|det(M′)/det(M)|²
    if isfinite(logdetG′)
        P_i = exp(-ΔSb + 2*logdetG - 2*logdetG′)
    else
        P_i = 0.0
    end

    # accept or reject the update
    if rand(rng) < P_i
        logdetG = logdetG′
        sgndetG = sgndetG′
        copyto!(G, G′)
        copyto!(fermion_greens_calculator, fermion_greens_calculator_alt)
        fermion_path_integral.Sb += ΔSb
        accepted = true
    else
        # flip HS field back
        @. s_i = -s_i
        # revert diagonal on-site energy matrix
        @. V_i = -2*α[i]/Δτ * s_i + V_i
        # revert propagator matrices
        @fastmath @inbounds for l in eachindex(B)
            expmΔτV_l = B[l].expmΔτV
            expmΔτV_l[site] = exp(-Δτ*V_i[l])
        end
        accepted = false
    end

    return (accepted, logdetG, sgndetG)
end


@doc raw"""
    swap_update!(
        # ARGUMENTS
        Gup::Matrix{H}, logdetGup::R, sgndetGup::H,
        Gdn::Matrix{H}, logdetGdn::R, sgndetGdn::H,
        hubbard_ising_parameters::HubbardIsingHSParameters{R};
        # KEYWORD ARGUMENTS
        fermion_path_integral_up::FermionPathIntegral{H},
        fermion_path_integral_dn::FermionPathIntegral{H},
        fermion_greens_calculator_up::FermionGreensCalculator{H,R},
        fermion_greens_calculator_dn::FermionGreensCalculator{H,R},
        fermion_greens_calculator_up_alt::FermionGreensCalculator{H,R},
        fermion_greens_calculator_dn_alt::FermionGreensCalculator{H,R},
        Bup::Vector{P}, Bdn::Vector{P},
        rng::AbstractRNG
    ) where {H<:Number, R<:Real, P<:AbstractPropagator}

Perform a swap update where the HS fields associated with two randomly chosen sites in the lattice are exchanged.
This function returns `(accepted, logdetGup, sgndetGup, logdetGdn, sgndetGdn)`.

# Arguments

- `Gup::Matrix{H}`: Spin-up eqaul-time Greens function matrix.
- `logdetGup::R`: Log of the determinant of the spin-up eqaul-time Greens function matrix.
- `sgndetGup::H`: Sign/phase of the determinant of the spin-up eqaul-time Greens function matrix.
- `Gdn::Matrix{H}`: Spin-down eqaul-time Greens function matrix.
- `logdetGdn::R`: Log of the determinant of the spin-down eqaul-time Greens function matrix.
- `sgndetGdn::H`: Sign/phase of the determinant of the spin-down eqaul-time Greens function matrix.
- `hubbard_ising_parameters::HubbardIsingHSParameters{R}`: Ising Hubbard-Stratonovich fields and associated parameters to update.

# Keyword Arguments

- `fermion_path_integral_up::FermionPathIntegral{H}`: An instance of [`FermionPathIntegral`](@ref) type for spin-up electrons.
- `fermion_path_integral_dn::FermionPathIntegral{H}`: An instance of [`FermionPathIntegral`](@ref) type for spin-down electrons.
- `fermion_greens_calculator_up::FermionGreensCalculator{H,R}`: Contains matrix factorization information for current spin-up sector state.
- `fermion_greens_calculator_dn::FermionGreensCalculator{H,R}`: Contains matrix factorization information for current spin-down sector state.
- `fermion_greens_calculator_up_alt::FermionGreensCalculator{H,R}`: Used to calculate matrix factorizations for proposed spin-up sector state.
- `fermion_greens_calculator_dn_alt::FermionGreensCalculator{H,R}`: Used to calculate matrix factorizations for proposed spin-up sector state.
- `Bup::Vector{P}`: Spin-up propagators for each imaginary time slice.
- `Bdn::Vector{P}`: Spin-down propagators for each imaginary time slice.
- `rng::AbstractRNG`: Random number generator used in method instead of global random number generator, important for reproducibility.
"""
function swap_update!(
    # ARGUMENTS
    Gup::Matrix{H}, logdetGup::R, sgndetGup::H,
    Gdn::Matrix{H}, logdetGdn::R, sgndetGdn::H,
    hubbard_ising_parameters::HubbardIsingHSParameters{R};
    # KEYWORD ARGUMENTS
    fermion_path_integral_up::FermionPathIntegral{H},
    fermion_path_integral_dn::FermionPathIntegral{H},
    fermion_greens_calculator_up::FermionGreensCalculator{H,R},
    fermion_greens_calculator_dn::FermionGreensCalculator{H,R},
    fermion_greens_calculator_up_alt::FermionGreensCalculator{H,R},
    fermion_greens_calculator_dn_alt::FermionGreensCalculator{H,R},
    Bup::Vector{P}, Bdn::Vector{P},
    rng::AbstractRNG
) where {H<:Number, R<:Real, P<:AbstractPropagator}

    @assert fermion_path_integral_up.Sb == fermion_path_integral_dn.Sb "$(fermion_path_integral_up.Sb) ≠ $(fermion_path_integral_dn.Sb)"

    (; N, U, α, sites, s, Δτ) = hubbard_ising_parameters
    Gup′ = fermion_greens_calculator_up_alt.G′
    Gdn′ = fermion_greens_calculator_dn_alt.G′

    # make sure stabilization frequencies match
    if fermion_greens_calculator_up.n_stab != fermion_greens_calculator_up_alt.n_stab
        resize!(fermion_greens_calculator_up_alt, fermion_greens_calculator_up.n_stab)
    end

    # make sure stabilization frequencies match
    if fermion_greens_calculator_dn.n_stab != fermion_greens_calculator_dn_alt.n_stab
        resize!(fermion_greens_calculator_dn_alt, fermion_greens_calculator_dn.n_stab)
    end

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
    Sb  = U[i] > zero(R) ? zero(R) : α[i] * sum(s_i)
    Sb += U[j] > zero(R) ? zero(R) : α[j] * sum(s_j)

    # swap the HS fields
    swap!(s_i, s_j)

    # calculate the final bosonic action
    Sb′  = U[i] > zero(R) ? zero(R) : α[i] * sum(s_i)
    Sb′ += U[j] > zero(R) ? zero(R) : α[j] * sum(s_j)

    # calculate the change in the bosonic action
    ΔSb = Sb′ - Sb

    # update diagonal on-site energy matrix
    @. Vup_i = Vup_i - α[i]/Δτ * (s_i - s_j) 
    @. Vdn_i = Vdn_i + sign(U[i])*α[i]/Δτ * (s_i - s_j) 
    @. Vup_j = Vup_j - α[j]/Δτ * (s_j - s_i) 
    @. Vdn_j = Vdn_j + sign(U[i])*α[j]/Δτ * (s_j - s_i)

    # update propagator matrices
    @fastmath @inbounds for l in eachindex(Bup)
        expmΔτVup_l = Bup[l].expmΔτV
        expmΔτVdn_l = Bdn[l].expmΔτV
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
    if isfinite(logdetGup′) && isfinite(logdetGdn′)
        P_i = exp(-ΔSb + logdetGup + logdetGdn - logdetGup′ - logdetGdn′)
    else
        P_i = 0.0
    end

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
        fermion_path_integral_up.Sb += ΔSb
        fermion_path_integral_dn.Sb += ΔSb
        accepted = true
    else
        # flip HS fields back
        swap!(s_i, s_j)
        # revert diagonal on-site energy matrix
        @. Vup_i = Vup_i - α[i]/Δτ * (s_i - s_j) 
        @. Vdn_i = Vdn_i + sign(U[i])*α[i]/Δτ * (s_i - s_j) 
        @. Vup_j = Vup_j - α[j]/Δτ * (s_j - s_i) 
        @. Vdn_j = Vdn_j + sign(U[i])*α[j]/Δτ * (s_j - s_i)
        # revert propagator matrices
        @fastmath @inbounds for l in eachindex(Bup)
            expmΔτVup_l = Bup[l].expmΔτV
            expmΔτVdn_l = Bdn[l].expmΔτV
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
    swap_update!(
        # ARGUMENTS
        G::Matrix{H}, logdetG::R, sgndetG::H,
        hubbard_ising_parameters::HubbardIsingHSParameters{R};
        # KEYWORD ARGUMENTS
        fermion_path_integral::FermionPathIntegral{H},
        fermion_greens_calculator::FermionGreensCalculator{H,R},
        fermion_greens_calculator_alt::FermionGreensCalculator{H,R},
        B::Vector{P}, rng::AbstractRNG
    ) where {H<:Number, R<:Real, P<:AbstractPropagator}

For strictly attractive Hubbard interactions, perform a swap update where the HS fields associated with two randomly chosen
sites in the lattice are exchanged. This function returns `(accepted, logdetG, sgndetG)`.

# Arguments

- `G::Matrix{H}`: Eqaul-time Greens function matrix.
- `logdetG::R`: Log of the determinant of the eqaul-time Greens function matrix.
- `sgndetG::H`: Sign/phase of the determinant of the eqaul-time Greens function matrix.
- `hubbard_ising_parameters::HubbardIsingHSParameters{R}`: Ising Hubbard-Stratonovich fields and associated parameters to update.

# Keyword Arguments

- `fermion_path_integral::FermionPathIntegral{H}`: An instance of [`FermionPathIntegral`](@ref) type.
- `fermion_greens_calculator::FermionGreensCalculator{H,R}`: Contains matrix factorization information for current state.
- `fermion_greens_calculator_alt::FermionGreensCalculator{H,R}`: Used to calculate matrix factorizations for proposed state.
- `B::Vector{P}`: Propagators for each imaginary time slice.
- `rng::AbstractRNG`: Random number generator used in method instead of global random number generator, important for reproducibility.
"""
function swap_update!(
    # ARGUMENTS
    G::Matrix{H}, logdetG::R, sgndetG::H,
    hubbard_ising_parameters::HubbardIsingHSParameters{R};
    # KEYWORD ARGUMENTS
    fermion_path_integral::FermionPathIntegral{H},
    fermion_greens_calculator::FermionGreensCalculator{H,R},
    fermion_greens_calculator_alt::FermionGreensCalculator{H,R},
    B::Vector{P}, rng::AbstractRNG
) where {H<:Number, R<:Real, P<:AbstractPropagator}

    (; N, U, α, sites, s, Δτ) = hubbard_ising_parameters
    G′ = fermion_greens_calculator_alt.G′

    # make sure stabilization frequencies match
    if fermion_greens_calculator.n_stab != fermion_greens_calculator_alt.n_stab
        resize!(fermion_greens_calculator_alt, fermion_greens_calculator.n_stab)
    end

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
    Sb  = U[i] > zero(R) ? zero(R) : α[i] * sum(s_i)
    Sb += U[j] > zero(R) ? zero(R) : α[j] * sum(s_j)

    # swap the HS fields
    swap!(s_i, s_j)

    # calculate the final bosonic action
    Sb′  = U[i] > zero(R) ? zero(R) : α[i] * sum(s_i)
    Sb′ += U[j] > zero(R) ? zero(R) : α[j] * sum(s_j)

    # calculate the change in the bosonic action
    ΔSb = Sb′ - Sb

    # update diagonal on-site energy matrix
    @. V_i = V_i - α[i]/Δτ * (s_i - s_j) 
    @. V_j = V_j - α[j]/Δτ * (s_j - s_i) 

    # update propagator matrices
    @fastmath @inbounds for l in eachindex(B)
        expmΔτV_l = B[l].expmΔτV
        expmΔτV_l[site_i] = exp(-Δτ*V_i[l])
        expmΔτV_l[site_j] = exp(-Δτ*V_j[l])
    end

    # calculate new Green's function matrices and determinant of new Green's function matrix
    logdetG′, sgndetG′ = calculate_equaltime_greens!(G′, fermion_greens_calculator_alt, B)

    # calculate acceptance probability P = exp(-ΔS_b)⋅|det(G)/det(G′)|²
    #                                    = exp(-ΔS_b)⋅|det(M′)/det(M)|²
    if isfinite(logdetG′)
        P_i = exp(-ΔSb + 2*logdetG - 2*logdetG′)
    else
        P_i = 0.0
    end

    # accept or reject the update
    if rand(rng) < P_i
        logdetG = logdetG′
        sgndetG = sgndetG′
        copyto!(G, G′)
        copyto!(fermion_greens_calculator, fermion_greens_calculator_alt)
        fermion_path_integral.Sb += ΔSb
        accepted = true
    else
        # flip HS fields back
        swap!(s_i, s_j)
        # revert diagonal on-site energy matrix
        @. V_i = V_i - α[i]/Δτ * (s_i - s_j) 
        @. V_j = V_j - α[j]/Δτ * (s_j - s_i) 
        # revert propagator matrices
        @fastmath @inbounds for l in eachindex(B)
            expmΔτV_l = B[l].expmΔτV
            expmΔτV_l[site_i] = exp(-Δτ*V_i[l])
            expmΔτV_l[site_j] = exp(-Δτ*V_j[l])
        end
        accepted = false
    end

    return (accepted, logdetG, sgndetG)
end

# caluclate hubbard interaction contribution to total bosonic action
function bosonic_action(
    hubbard_ising_parameters::HubbardIsingHSParameters{E}
) where {E <: AbstractFloat}

    (; s, α, U) = hubbard_ising_parameters

    # initialize bosonic action to zero
    Sb = zero(E)

    # iterate of time slices
    @inbounds for l in axes(s,2)
        # iterate of sites
        for i in axes(s,1)
            # incremement the bosonic action
            Sb += U[i] > 0 ? 0.0 : α[i] * s[i,l]
        end
    end


    return Sb
end