@doc raw"""
    HubbardIsingHSParameters{T<:AbstractFloat, F<:Int} <: AbstractHubbardHS{F}

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
struct HubbardIsingHSParameters{T<:AbstractFloat, F<:Int} <: AbstractHubbardHS{F}

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
    s::Matrix{F}

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
                hubbard_ising_parameters::HubbardIsingHSParameters{E,F}) where {T,E,F}

Initialize the contribution from the Hubbard interaction to the [`FermionPathIntegral`](@ref)
instance `fermion_path_integral_up` for spin up and `fermion_path_integral_dn` spin down.
"""
function initialize!(fermion_path_integral_up::FermionPathIntegral{T,E},
                     fermion_path_integral_dn::FermionPathIntegral{T,E},
                     hubbard_ising_parameters::HubbardIsingHSParameters{E,F}) where {T,E,F}
    
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

    return nothing
end


@doc raw"""
    initialize!(fermion_path_integral::FermionPathIntegral{T,E},
                hubbard_ising_parameters::HubbardIsingHSParameters{E,F}) where {T,E,F}

Initialize the contribution from an attractive Hubbard interaction to the [`FermionPathIntegral`](@ref)
instance `fermion_path_integral`.
"""
function initialize!(fermion_path_integral::FermionPathIntegral{T,E},
                     hubbard_ising_parameters::HubbardIsingHSParameters{E,F}) where {T,E,F}
    
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

    return nothing
end


@doc raw"""
    local_updates!(Gup::Matrix{T}, logdetGup::E, sgndetGup::T,
                   Gdn::Matrix{T}, logdetGdn::E, sgndetGdn::T,
                   hubbard_ising_parameters::HubbardIsingHSParameters{E,F};
                   fermion_path_integral_up::FermionPathIntegral{T,E},
                   fermion_path_integral_dn::FermionPathIntegral{T,E},
                   fermion_greens_calculator_up::FermionGreensCalculator{T,E},
                   fermion_greens_calculator_dn::FermionGreensCalculator{T,E},
                   Bup::Vector{P}, Bdn::Vector{P},
                   δG_max::E, δG::E, δθ::E,  rng::AbstractRNG,
                   update_stabilization_frequency::Bool=true) where {T<:Number, E<:AbstractFloat, P<:AbstractPropagator{T,E}}

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
- `hubbard_ising_parameters::HubbardIsingHSParameters{E,F}`: Ising Hubbard-Stratonovich fields and associated parameters to update.

## Keyword Arguments

- `fermion_path_integral_up::FermionPathIntegral{T,E}`: An instance of the [`FermionPathIntegral`](@ref) type for spin-up electrons.
- `fermion_path_integral_dn::FermionPathIntegral{T,E}`: An instance of the [`FermionPathIntegral`](@ref) type for spin-down electrons.
- `fermion_greens_calculator_up::FermionGreensCalculator{T,E}`: An instance of the [`FermionGreensCalculator`](https://smoqysuite.github.io/JDQMCFramework.jl/stable/api/#JDQMCFramework.FermionGreensCalculator) type for the spin-up electrons.
- `fermion_greens_calculator_dn::FermionGreensCalculator{T,E}`: An instance of the [`FermionGreensCalculator`](https://smoqysuite.github.io/JDQMCFramework.jl/stable/api/#JDQMCFramework.FermionGreensCalculator) type for the spin-down electrons.
- `Bup::Vector{P}`: Spin-up propagators for each imaginary time slice.
- `Bdn::Vector{P}`: Spin-dn propagators for each imaginary time slice.
- `δG_max::E`: Maximum allowed error corrected by numerical stabilization.
- `δG::E`: Previously recorded maximum error in the Green's function corrected by numerical stabilization.
- `δθ::T`: Previously recorded maximum error in the sign/phase of the determinant of the equal-time Green's function matrix corrected by numerical stabilization.
- `rng::AbstractRNG`: Random number generator used in method instead of global random number generator, important for reproducibility.
- `update_stabilization_frequency::Bool=true`: If true, allows the stabilization frequency `n_stab` to be dynamically adjusted.
"""
function local_updates!(Gup::Matrix{T}, logdetGup::E, sgndetGup::T,
                        Gdn::Matrix{T}, logdetGdn::E, sgndetGdn::T,
                        hubbard_ising_parameters::HubbardIsingHSParameters{E,F};
                        fermion_path_integral_up::FermionPathIntegral{T,E},
                        fermion_path_integral_dn::FermionPathIntegral{T,E},
                        fermion_greens_calculator_up::FermionGreensCalculator{T,E},
                        fermion_greens_calculator_dn::FermionGreensCalculator{T,E},
                        Bup::Vector{P}, Bdn::Vector{P},
                        δG_max::E, δG::E, δθ::E,  rng::AbstractRNG,
                        update_stabilization_frequency::Bool=true) where {T<:Number, E<:AbstractFloat, F<:Int, P<:AbstractPropagator{T,E}}

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
                ΔSb_il = -2 * α[i] * s[i,l]
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
    if update_stabilization_frequency
        (updated, logdetGup, sgndetGup, logdetGdn, sgndetGdn, δG, δθ) = update_stabalization_frequency!(
            Gup, logdetGup, sgndetGup,
            Gdn, logdetGdn, sgndetGdn,
            fermion_greens_calculator_up = fermion_greens_calculator_up,
            fermion_greens_calculator_dn = fermion_greens_calculator_dn,
            Bup = Bup, Bdn = Bdn, δG = δG, δθ = δθ, δG_max = δG_max
        )
    end

    # calculate the acceptance rate
    acceptance_rate = accepted_spin_flips / proposed_spin_flips

    return (acceptance_rate, logdetGup, sgndetGup, logdetGdn, sgndetGdn, δG, δθ)
end


@doc raw"""
    local_updates!(G::Matrix{T}, logdetG::E, sgndetG::T,
                   hubbard_ising_parameters::HubbardIsingHSParameters{E,F};
                   fermion_path_integral::FermionPathIntegral{T,E},
                   fermion_greens_calculator::FermionGreensCalculator{T,E},
                   B::Vector{P}, δG_max::E, δG::E, δθ::E, rng::AbstractRNG,
                   update_stabilization_frequency::Bool=true) where {T<:Number, E<:AbstractFloat, F<:Int, P<:AbstractPropagator{T,E}}

Sweep through every imaginary time slice and orbital in the lattice, performing local updates to every
Ising Hubbard-Stratonovich (HS) field, assuming strictly attractive Hubbard interactions and perfect spin symmetry.

This method returns the a tuple containing `(acceptance_rate, logdetG, sgndetG, δG, δθ)`.

# Arguments

- `G::Matrix{T}`: Equal-time Green's function matrix.
- `logdetG::E`: The log of the absolute value of the determinant of theequal-time Green's function matrix, ``\log \vert \det G_\uparrow(\tau,\tau) \vert.``
- `sgndetG::T`: The sign/phase of the determinant of the equal-time Green's function matrix, ``\det G_\uparrow(\tau,\tau) / \vert \det G_\uparrow(\tau,\tau) \vert.``
- `hubbard_ising_parameters::HubbardIsingHSParameters{E,F}`: Ising Hubbard-Stratonovich fields and associated parameters to update.

## Keyword Arguments

- `fermion_path_integral::FermionPathIntegral{T,E}`: An instance of the [`FermionPathIntegral`](@ref) type.
- `fermion_greens_calculator::FermionGreensCalculator{T,E}`: An instance of the [`FermionGreensCalculator`](https://smoqysuite.github.io/JDQMCFramework.jl/stable/api/#JDQMCFramework.FermionGreensCalculator) type.
- `B::Vector{P}`: Propagators for each imaginary time slice.
- `δG_max::E`: Maximum allowed error corrected by numerical stabilization.
- `δG::E`: Previously recorded maximum error in the Green's function corrected by numerical stabilization.
- `δθ::T`: Previously recorded maximum error in the sign/phase of the determinant of the equal-time Green's function matrix corrected by numerical stabilization.
- `rng::AbstractRNG`: Random number generator used in method instead of global random number generator, important for reproducibility.
- `update_stabilization_frequency::Bool=true`:  If true, allows the stabilization frequency `n_stab` to be dynamically adjusted.
"""
function local_updates!(G::Matrix{T}, logdetG::E, sgndetG::T,
                        hubbard_ising_parameters::HubbardIsingHSParameters{E,F};
                        fermion_path_integral::FermionPathIntegral{T,E},
                        fermion_greens_calculator::FermionGreensCalculator{T,E},
                        B::Vector{P}, δG_max::E, δG::E, δθ::E, rng::AbstractRNG,
                        update_stabilization_frequency::Bool=true) where {T<:Number, E<:AbstractFloat, F<:Int, P<:AbstractPropagator{T,E}}

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
        reverse_partially_wrap_greens(G, B[l], G′)

        # Periodically re-calculate the Green's function matrix for numerical stability.
        logdetG, sgndetG, δG′, δθ′ = stabilize_equaltime_greens!(G, logdetG, sgndetG, fermion_greens_calculator, B, update_B̄=true)

        # record the max errors
        δG = maximum((δG, δG′))
        δθ = maximum(abs, (δθ, δθ′))
    end

    # update stabilization frequency if required
    if update_stabilization_frequency
        (updated, logdetG, sgndetG, δG, δθ) = update_stabalization_frequency!(
            G, logdetG, sgndetG,
            fermion_greens_calculator = fermion_greens_calculator,
            B = B, δG = δG, δθ = δθ, δG_max = δG_max
        )
    end

    # calculate the acceptance rate
    acceptance_rate = accepted_spin_flips / proposed_spin_flips

    return (acceptance_rate, logdetG, sgndetG, δG, δθ)
end