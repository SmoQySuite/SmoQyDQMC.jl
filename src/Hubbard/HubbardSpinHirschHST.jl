@doc raw"""
    HubbardSpinHirschHST{T<:Number, E<:AbstractFloat}

This type represent a Hubbard-Stratonovich (HS) transformation introduced by
Hirsch using the spin channel, where the introduced HS fields take on the two
values ``s = \pm 1``. Specifically, the Hubbard interaction is decoupled as
```math
e^{-\Delta\tau U\left(n_{\uparrow}-\tfrac{1}{2}\right)\left(n_{\downarrow}-\tfrac{1}{2}\right)}
 = \gamma\sum_{s=\pm1}e^{-\Delta\tau\alpha(n_{\uparrow}-n_{\downarrow})s},
```
where
```math
\gamma=\frac{1}{2}e^{-\Delta\tau U/4}
```
and
```
\alpha = \frac{1}{\Delta\tau}\cosh^{-1}\left(e^{\Delta\tau U/2}\right)
       = \begin{cases}
            \frac{1}{\Delta\tau}\cosh^{-1}\left(e^{\Delta\tau|U|/2}\right), & U>0\\
            \frac{{\rm i}}{\Delta\tau}\cos^{-1}\left(e^{-\Delta\tau|U|/2}\right), & U<0.
         \end{cases}
```
"""
struct HubbardSpinHirschHST{T<:Number, E<:AbstractFloat}

    # inverse temperature
    β::E

    # discretization in imaginary time
    Δτ::E

    # length of imaginary time axis
    Lτ::Int

    # number of orbitals with finite Hubbard U
    N::Int

    # each finite hubbard interaction
    U::Vector{E}
    
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
    HubbardSpinHirschHST(;
        # KEYWORD ARGUMENTS
        hubbard_parameters::HubbardParameters{E},
        β::E, Δτ::E, rng::AbstractRNG
    ) where {E<:AbstractFloat}

Initialize an instance of the [`HubbardSpinHirschHST`](@ref) type.
"""
function HubbardSpinHirschHST(;
    # KEYWORD ARGUMENTS
    hubbard_parameters::HubbardParameters{E},
    β::E, Δτ::E, rng::AbstractRNG
) where {E<:AbstractFloat}

    (; U, sites) = hubbard_parameters

    # if any attractive Hubbard interactions, then complex field coefficients
    T = any(u -> u < 0, U) ? Complex{E} : E

    # calculate length of imaginary-time axis
    Lτ = round(Int, β / Δτ)

    # calculate HS transformation coefficients
    α = zeros(T, length(U))
    @. α = (U ≥ 0.0) ? acosh(exp(Δτ*abs(U)/2))/Δτ : 1.0im*acos(exp(-Δτ*abs(U)/2))/Δτ

    # number of sites with Hubbard interaction
    N = length(U)

    # initialize HS fields
    s = rand(rng, -1:2:1, N, Lτ)

    # initialize update permuation order
    update_perm = collect(1:N)

    return HubbardSpinHirschHST{T,E}(β, Δτ, Lτ, N, U, α, sites, s, update_perm)
end


@doc raw"""
    initialize!(
        fermion_path_integral_up::FermionPathIntegral{H,T,U,R},
        fermion_path_integral_dn::FermionPathIntegral{H,T,U,R},
        hst_parameters::HubbardSpinHirschHST{U},
    ) where {H<:Number, T<:Number, U<:Number, R<:Real}

Initialize the `fermion_path_integral_up` and `fermion_path_integral_dn`
to reflect the current Hubbard-Stratonovich field configuration stored in the
`hst_parameters` type.
"""
function initialize!(
    fermion_path_integral_up::FermionPathIntegral{H,T,U,R},
    fermion_path_integral_dn::FermionPathIntegral{H,T,U,R},
    hst_parameters::HubbardSpinHirschHST{U},
) where {H<:Number, T<:Number, U<:Number, R<:Real}

    (; sites, α) = hst_parameters
    Vup = fermion_path_integral_up.V
    Vdn = fermion_path_integral_dn.V

    # iterate over sites with Hubbard U interactions
    for i in eachindex(sites)
        site = sites[i]
        @views @. Vup[site,:] += α[i] * s[i,:]
        @views @. Vdn[site,:] -= α[i] * s[i,:]
    end

    return nothing
end

@doc raw"""
    local_updates!(
        # ARGUMENTS
        Gup::Matrix{H}, logdetGup::R, sgndetGup::H,
        Gdn::Matrix{H}, logdetGdn::R, sgndetGdn::H,
        hst_parameters::HubbardSpinHirschHST{T,R};
        # KEYWORD ARGUMENTS
        fermion_path_integral_up::FermionPathIntegral{H},
        fermion_path_integral_dn::FermionPathIntegral{H},
        fermion_greens_calculator_up::FermionGreensCalculator{H},
        fermion_greens_calculator_dn::FermionGreensCalculator{H},
        Bup::Vector{P}, Bdn::Vector{P},
        δG::R, δθ::R,  rng::AbstractRNG,
        δG_max::R = 1e-6,
        update_stabilization_frequency::Bool = true
    ) where {H<:Number, T<:Number, R<:Real, P<:AbstractPropagator}

Attempt a local update to spin-channel Hirsch Hubbard-Stratonovich field.
This method returns a tuple containing `(acceptance_rate, logdetGup, sgndetGup, logdetGdn, sgndetGdn, δG, δθ)`.

# Arguments

- `Gup::Matrix{H}`: Spin-up equal-time Green's function matrix.
- `logdetGup::R`: The log of the absolute value of the determinant of the spin-up equal-time Green's function matrix, ``\log \vert \det G_\uparrow(\tau,\tau) \vert.``
- `sgndetGup::H`: The sign/phase of the determinant of the spin-up equal-time Green's function matrix, ``\det G_\uparrow(\tau,\tau) / \vert \det G_\uparrow(\tau,\tau) \vert.``
- `Gdn::Matrix{H}`: Spin-down equal-time Green's function matrix.
- `logdetGdn::R`: The log of the absolute value of the determinant of the spin-down equal-time Green's function matrix, ``\log \vert \det G_\downarrow(\tau,\tau) \vert.``
- `sgndetGdn::H`: The sign/phase of the determinant of the spin-down equal-time Green's function matrix, ``\det G_\downarrow(\tau,\tau) / \vert \det G_\downarrow(\tau,\tau) \vert.``
- `hst_parameters::HubbardSpinHirschHST{T,R}`: Type representing Hubbard-Stratonovich transformation.

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
    hst_parameters::HubbardSpinHirschHST{T,R};
    # KEYWORD ARGUMENTS
    fermion_path_integral_up::FermionPathIntegral{H},
    fermion_path_integral_dn::FermionPathIntegral{H},
    fermion_greens_calculator_up::FermionGreensCalculator{H},
    fermion_greens_calculator_dn::FermionGreensCalculator{H},
    Bup::Vector{P}, Bdn::Vector{P},
    δG::R, δθ::R,  rng::AbstractRNG,
    δG_max::R = 1e-6,
    update_stabilization_frequency::Bool = true
) where {H<:Number, T<:Number, R<:Real, P<:AbstractPropagator}

    # make sure bosonic actions match
    @assert fermion_path_integral_up.Sb == fermion_path_integral_dn.Sb "$(fermion_path_integral_up.Sb) ≠ $(fermion_path_integral_dn.Sb)"

    (; Δτ, U, α, sites, s, update_perm, N) = hst_parameters
    (; u, v) = fermion_path_integral_dn

    # get temporary storage matrix
    G′ = fermion_greens_calculator_up.G′

    # get on-site energy matrices for spin up and down electrons for all time slices
    Vup = fermion_path_integral_up.V
    Vdn = fermion_path_integral_dn.V

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
        partially_wrap_greens_reverse!(Gup, Bup_l, G′)
        partially_wrap_greens_reverse!(Gdn, Bdn_l, G′)

        # shuffle the order in which orbitals/sites will be iterated over
        shuffle!(rng, update_perm)

        # iterate over orbitals in the lattice
        for i in update_perm

            # increment number of proposed spin flips
            proposed_spin_flips += 1

            # get the site
            site = sites[i]

            # calculate the new value of Vup[i,l] and Vdn[i,l] resulting from the
            # HS field have it's sign flipped from s[i,l] ==> -s[i,l]
            s_il′ = -s[i,l]
            Vup_il′ = +2 * α[i] * s_il′ + Vup[site,l]
            Vdn_il′ = -2 * α[i] * s_il′ + Vdn[site,l]

            # calculate spin-up determinant ratio associated with Ising HS spin flip
            Rup_il, Δup_il = local_update_det_ratio(Gup, Bup_l, Vup_il′, site, Δτ)
            Rdn_il, Δdn_il = local_update_det_ratio(Gdn, Bdn_l, Vdn_il′, site, Δτ)

            # calculate acceptance probability
            P_il = abs(Rup_il * Rdn_il)

            # accept or reject proposed update
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
        (updated, logdetGup, sgndetGup, logdetGdn, sgndetGdn, δG, δθ) = update_stabalization_frequency!(
            Gup, logdetGup, sgndetGup,
            Gdn, logdetGdn, sgndetGdn,
            fermion_greens_calculator_up = fermion_greens_calculator_up,
            fermion_greens_calculator_dn = fermion_greens_calculator_dn,
            Bup = Bup, Bdn = Bdn, δG = δG, δθ = δθ, δG_max = δG_max
        )
    end

    # calculate the acceptance rate
    acceptance_rate = accepted_spin_flips / N

    return (acceptance_rate, logdetGup, sgndetGup, logdetGdn, sgndetGdn, δG, δθ)
end


@doc raw"""
    reflection_update!(
        # ARGUMENTS
        Gup::Matrix{H}, logdetGup::R, sgndetGup::H,
        Gdn::Matrix{H}, logdetGdn::R, sgndetGdn::H,
        hst_parameters::HubbardSpinHirschHST{T,R};
        # KEYWORD ARGUMENTS
        fermion_path_integral_up::FermionPathIntegral{H},
        fermion_path_integral_dn::FermionPathIntegral{H},
        fermion_greens_calculator_up::FermionGreensCalculator{H,R},
        fermion_greens_calculator_dn::FermionGreensCalculator{H,R},
        fermion_greens_calculator_up_alt::FermionGreensCalculator{H,R},
        fermion_greens_calculator_dn_alt::FermionGreensCalculator{H,R},
        Bup::Vector{P}, Bdn::Vector{P},
        rng::AbstractRNG
    ) where {H<:Number, T<:Number, R<:Real, P<:AbstractPropagator}

Perform a reflection update in which the sign of every spin-channel Hirsch Hubbard-Stratonovich field on a randomly chosen orbital in the lattice is changed.
This function returns `(accepted, logdetGup, sgndetGup, logdetGdn, sgndetGdn)`.

# Arguments

- `Gup::Matrix{H}`: Spin-up eqaul-time Greens function matrix.
- `logdetGup::R`: Log of the determinant of the spin-up eqaul-time Greens function matrix.
- `sgndetGup::H`: Sign/phase of the determinant of the spin-up eqaul-time Greens function matrix.
- `Gdn::Matrix{H}`: Spin-down eqaul-time Greens function matrix.
- `logdetGdn::R`: Log of the determinant of the spin-down eqaul-time Greens function matrix.
- `sgndetGdn::H`: Sign/phase of the determinant of the spin-down eqaul-time Greens function matrix.
- `hst_parameters::HubbardSpinHirschHST{T,R}`: Hubbard-Stratonovich fields and associated parameters to update.

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
    hst_parameters::HubbardSpinHirschHST{T,R};
    # KEYWORD ARGUMENTS
    fermion_path_integral_up::FermionPathIntegral{H},
    fermion_path_integral_dn::FermionPathIntegral{H},
    fermion_greens_calculator_up::FermionGreensCalculator{H,R},
    fermion_greens_calculator_dn::FermionGreensCalculator{H,R},
    fermion_greens_calculator_up_alt::FermionGreensCalculator{H,R},
    fermion_greens_calculator_dn_alt::FermionGreensCalculator{H,R},
    Bup::Vector{P}, Bdn::Vector{P},
    rng::AbstractRNG
) where {H<:Number, T<:Number, R<:Real, P<:AbstractPropagator}

    (; Δτ, α, sites, s, N) = hst_parameters
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
    s_i′ = s_i

    # update diagonal on-site energy matrix:
    # ΔV_up = [α⋅(-s)]  - [α⋅s]  = [α⋅s′]  - [α⋅(-s′)]  = +2⋅α⋅s′
    # ΔV_dn = [-α⋅(-s)] - [-α⋅s] = [-α⋅s′] - [-α⋅(-s′)] = -2⋅α⋅s′
    @. Vup_i = +2*α[i] * s_i′ + Vup_i
    @. Vdn_i = -2*α[i] * s_i′ + Vdn_i

    # update propagator matrices
    @inbounds for l in eachindex(Bup)
        expmΔτVup_l = Bup[l].expmΔτV
        expmΔτVdn_l = Bdn[l].expmΔτV
        expmΔτVup_l[site] = exp(-Δτ*Vup_i[l])
        expmΔτVdn_l[site] = exp(-Δτ*Vdn_i[l])
    end

    # calculate new Green's function matrices and determinant of new Green's function matrix
    logdetGup′, sgndetGup′ = calculate_equaltime_greens!(Gup′, fermion_greens_calculator_up_alt, Bup)
    logdetGdn′, sgndetGdn′ = calculate_equaltime_greens!(Gdn′, fermion_greens_calculator_dn_alt, Bdn)

    # calculate acceptance probability P = exp(-ΔSf) = exp(-(Sf′ - Sf))
    #                                    = exp(-(logdetGup′ + logdetGdn′ - logdetGup - logdetGdn))
    #                                    = exp(logdetGup + logdetGdn - logdetGup′ - logdetGdn′)
    if isfinite(logdetGup′) && isfinite(logdetGdn′)
        P_i = exp(logdetGup + logdetGdn - logdetGup′ - logdetGdn′)
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
        accepted = true
    else
        # flip HS field back
        @. s_i = -s_i
        # revert diagonal on-site energy matrix
        @. Vup_i = -2*α[i]/Δτ * s_i + Vup_i
        @. Vdn_i = +2*α[i]/Δτ * s_i + Vdn_i
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
    swap_update!(
        # ARGUMENTS
        Gup::Matrix{H}, logdetGup::R, sgndetGup::H,
        Gdn::Matrix{H}, logdetGdn::R, sgndetGdn::H,
        hst_parameters::HubbardSpinHirschHST{T,R};
        # KEYWORD ARGUMENTS
        fermion_path_integral_up::FermionPathIntegral{H},
        fermion_path_integral_dn::FermionPathIntegral{H},
        fermion_greens_calculator_up::FermionGreensCalculator{H,R},
        fermion_greens_calculator_dn::FermionGreensCalculator{H,R},
        fermion_greens_calculator_up_alt::FermionGreensCalculator{H,R},
        fermion_greens_calculator_dn_alt::FermionGreensCalculator{H,R},
        Bup::Vector{P}, Bdn::Vector{P},
        rng::AbstractRNG
    ) where {H<:Number, T<:Number, R<:Real, P<:AbstractPropagator}

Perform a reflection update in which the sign of every spin-channel Hirsch Hubbard-Stratonovich field on a randomly chosen orbital in the lattice is changed.
This function returns `(accepted, logdetGup, sgndetGup, logdetGdn, sgndetGdn)`.

# Arguments

- `Gup::Matrix{H}`: Spin-up eqaul-time Greens function matrix.
- `logdetGup::R`: Log of the determinant of the spin-up eqaul-time Greens function matrix.
- `sgndetGup::H`: Sign/phase of the determinant of the spin-up eqaul-time Greens function matrix.
- `Gdn::Matrix{H}`: Spin-down eqaul-time Greens function matrix.
- `logdetGdn::R`: Log of the determinant of the spin-down eqaul-time Greens function matrix.
- `sgndetGdn::H`: Sign/phase of the determinant of the spin-down eqaul-time Greens function matrix.
- `hst_parameters::HubbardSpinHirschHST{T,R}`: Hubbard-Stratonovich fields and associated parameters to update.

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
function swap_update!(
    # ARGUMENTS
    Gup::Matrix{H}, logdetGup::R, sgndetGup::H,
    Gdn::Matrix{H}, logdetGdn::R, sgndetGdn::H,
    hst_parameters::HubbardSpinHirschHST{T,R};
    # KEYWORD ARGUMENTS
    fermion_path_integral_up::FermionPathIntegral{H},
    fermion_path_integral_dn::FermionPathIntegral{H},
    fermion_greens_calculator_up::FermionGreensCalculator{H,R},
    fermion_greens_calculator_dn::FermionGreensCalculator{H,R},
    fermion_greens_calculator_up_alt::FermionGreensCalculator{H,R},
    fermion_greens_calculator_dn_alt::FermionGreensCalculator{H,R},
    Bup::Vector{P}, Bdn::Vector{P},
    rng::AbstractRNG
) where {H<:Number, T<:Number, R<:Real, P<:AbstractPropagator}

    (; Δτ, α, sites, s, N) = hst_parameters
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
    i, j = draw2(rng, N)

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

    # swap the HS fields
    swap!(s_i, s_j)

    # update potential energy matrices
    @. Vup_i = Vup_i + α[i] * (s_i - s_j)
    @. Vdn_i = Vdn_i - α[i] * (s_i - s_j)
    @. Vup_j = Vup_j + α[j] * (s_j - s_i)
    @. Vdn_j = Vdn_j - α[j] * (s_j - s_i)

    # update propagator matrices
    @inbounds for l in eachindex(Bup)
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

    # calculate acceptance probability P = exp(-ΔSf) = exp(-(Sf′ - Sf))
    #                                    = exp(-(logdetGup′ + logdetGdn′ - logdetGup - logdetGdn))
    #                                    = exp(logdetGup + logdetGdn - logdetGup′ - logdetGdn′)
    if isfinite(logdetGup′) && isfinite(logdetGdn′)
        P_i = exp(logdetGup + logdetGdn - logdetGup′ - logdetGdn′)
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
        accepted = true
    else
        # flip HS fields back
        swap!(s_i, s_j)
        # revert diagonal on-site energy matrix
        @. Vup_i = Vup_i + α[i] * (s_i - s_j)
        @. Vdn_i = Vdn_i - α[i] * (s_i - s_j)
        @. Vup_j = Vup_j + α[j] * (s_j - s_i)
        @. Vdn_j = Vdn_j - α[j] * (s_j - s_i)
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