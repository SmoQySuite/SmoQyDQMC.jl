
@doc raw"""
    HubbardDensityHirschHST{T<:Number, E<:AbstractFloat}

This type represents a Hubbard-Stratonovich (HS) transformation for decoupling the local Hubbard interaction in the density channel,
where the introduced HS fields take on the four discrete values ``s \in \{ -2, -1, +1, +2 \}.``
Note that the Hubbard interaction can be written in the density channel as
```math
U(\hat{n}_{\uparrow}-\tfrac{1}{2})(\hat{n}_{\downarrow}-\tfrac{1}{2})=-\tfrac{U}{2}(\hat{n}_{\uparrow}+\hat{n}_{\downarrow})^{2}-\tfrac{U}{4},
```
different only by a constant energy offset ``-U/4`` which does not matter.
Therefore, we can perform a Gauss-Hermite Hubbard-Statonovich transformation in the density channel as
```math
e^{-\Delta\tau\left[\frac{U}{2}\right](\hat{n}_{\uparrow}+\hat{n}_{\downarrow}-1)^{2}}
 = \frac{1}{4}\sum_{s=\pm1,\pm2}e^{-S_{\text{GF}}(s)-\Delta\tau\hat{V}(s)}+\mathcal{O}\left((\Delta\tau\lambda)^{4}\right)
```
where ``\hat{V}(s)=\alpha\eta(s)(\hat{n}_{\uparrow}+\hat{n}_{\downarrow}-1)`` and ``\alpha = \sqrt{-U/(2\Delta\tau)}``.
In the above expression,
```math
S_{\text{GH}}(s)=-\log\left(1+\sqrt{6}\left(1-\tfrac{2}{3}|s|\right)\right)
```
and
```math
\eta(s)=\frac{s}{|s|}\sqrt{6(1-\sqrt{6})+4\sqrt{6}|s|}.
```
Note that ``\alpha`` is strictly real when ``U \le 0`` and strictly imaginary when ``U > 0``.
"""
struct HubbardDensityGaussHermiteHST{T<:Number, E<:AbstractFloat}

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
    
    # HST coupling coefficient
    α::Vector{T}

    # site index associated with each Hubbard U
    sites::Vector{Int}

    # Ising Hubbard-Stratonovich fields
    s::Matrix{Int}

    # order in which to iterate over orbitals when updating Hubbard-Stratonovich fields.
    update_perm::Vector{Int}
end

@doc raw"""
    HubbardDensityGaussHermiteHST(;
        # KEYWORD ARGUMENTS
        hubbard_parameters::HubbardParameters{E},
        β::E, Δτ::E, rng::AbstractRNG
    ) where {E<:AbstractFloat}

Initialize an instance of the [`HubbardDensityGaussHermiteHST`](@ref) type.
"""
function HubbardDensityGaussHermiteHST(;
    # KEYWORD ARGUMENTS
    hubbard_parameters::HubbardParameters{E},
    β::E, Δτ::E, rng::AbstractRNG
) where {E<:AbstractFloat}

    (; U, sites) = hubbard_parameters

    # if any attractive Hubbard interactions, then complex field coefficients
    T = any(u -> u > 0, U) ? Complex{E} : E

    # calculate length of imaginary-time axis
    Lτ = round(Int, β / Δτ)

    # calculate HS transformation coefficients
    α = zeros(T, length(U))
    @. α = sqrt(-T(U)/(2*Δτ))

    # number of sites with Hubbard interaction
    N = length(U)

    # initialize HS fields
    s = rand(rng, (-2,-1,+1,+2), (N, Lτ))

    # initialize update permuation order
    update_perm = collect(1:N)

    return HubbardDensityGaussHermiteHST{T,E}(β, Δτ, Lτ, N, U, α, sites, s, update_perm)
end


@doc raw"""
    initialize!(
        fermion_path_integral_up::FermionPathIntegral{H,T,U,R},
        fermion_path_integral_dn::FermionPathIntegral{H,T,U,R},
        hst_parameters::HubbardDensityGaussHermiteHST{U}
    ) where {H<:Number, T<:Number, U<:Number, R<:Real}

    initialize!(
        fermion_path_integral::FermionPathIntegral{H,T,U,R},
        hst_parameters::HubbardDensityGaussHermiteHST{U},
    ) where {H<:Number, T<:Number, U<:Number, R<:Real}

Initialize [`FermionPathIntegral`](@ref) instances to reflect the initial
HS field configuration represented by the [`HubbardDensityGaussHermiteHST`](@ref) type.
"""
function initialize!(
    fermion_path_integral_up::FermionPathIntegral{H,T,U,R},
    fermion_path_integral_dn::FermionPathIntegral{H,T,U,R},
    hst_parameters::HubbardDensityGaussHermiteHST{U}
) where {H<:Number, T<:Number, U<:Number, R<:Real}

    # make sure bosonic actions match
    @assert fermion_path_integral_up.Sb == fermion_path_integral_dn.Sb "$(fermion_path_integral_up.Sb) ≠ $(fermion_path_integral_dn.Sb)"

    initialize!(fermion_path_integral_up, hst_parameters)
    initialize!(fermion_path_integral_dn, hst_parameters)

    return nothing
end

function initialize!(
    fermion_path_integral::FermionPathIntegral{H,T,U,R},
    hst_parameters::HubbardDensityGaussHermiteHST{U},
) where {H<:Number, T<:Number, U<:Number, R<:Real}

    (; sites, α, s, Δτ) = hst_parameters
    V = fermion_path_integral.V

    # iterate over sites with Hubbard U interactions
    for i in eachindex(sites)
        site = sites[i]
        s_i = @view s[i,:]
        V_i = @view V[site,:]
        @. V_i += α[i] * eval_η(s_i)
        fermion_path_integral.Sb += eval_Sgh(s_i) - Δτ*α[i]*sum(eval_η, s_i)
    end

    return nothing
end


@doc raw"""
    local_updates!(
        # ARGUMENTS
        Gup::Matrix{H}, logdetGup::R, sgndetGup::H,
        Gdn::Matrix{H}, logdetGdn::R, sgndetGdn::H,
        hst_parameters::HubbardDensityGaussHermiteHST{T,R};
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

Perform local updates to density-channel Hirsch Hubbard-Stratonovich fields.
This method returns a tuple containing `(acceptance_rate, logdetGup, sgndetGup, logdetGdn, sgndetGdn, δG, δθ)`.

# Arguments

- `Gup::Matrix{H}`: Spin-up equal-time Green's function matrix.
- `logdetGup::R`: The log of the absolute value of the determinant of the spin-up equal-time Green's function matrix, ``\log \vert \det G_\uparrow(\tau,\tau) \vert.``
- `sgndetGup::H`: The sign/phase of the determinant of the spin-up equal-time Green's function matrix, ``\det G_\uparrow(\tau,\tau) / \vert \det G_\uparrow(\tau,\tau) \vert.``
- `Gdn::Matrix{H}`: Spin-down equal-time Green's function matrix.
- `logdetGdn::R`: The log of the absolute value of the determinant of the spin-down equal-time Green's function matrix, ``\log \vert \det G_\downarrow(\tau,\tau) \vert.``
- `sgndetGdn::H`: The sign/phase of the determinant of the spin-down equal-time Green's function matrix, ``\det G_\downarrow(\tau,\tau) / \vert \det G_\downarrow(\tau,\tau) \vert.``
- `hst_parameters::HubbardDensityGaussHermiteHST{T,R}`: Type representing Hubbard-Stratonovich transformation.

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
    hst_parameters::HubbardDensityGaussHermiteHST{T,R};
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

    @assert !( (H<:Real) &&  (T<:Complex)) "Green's function matrices are real while Hubbard-Stratonovich transformation is complex."
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

            # get the site
            site = sites[i]

            # propose a HS field update and calculate the change in the potential
            # energy matrix and bosonic action
            s_il     = s[i,l]
            η_il     = eval_η(s_il)
            Vup_il   = Vup[site, l]
            Vdn_il   = Vdn[site, l]
            Sb_il    = eval_Sgh(s_il) - Δτ*α[i]*η_il
            s_il′    = sample_new_ghhsf(rng, s_il)
            η_il′    = eval_η(s_il′)
            Sb_il′   = eval_Sgh(s_il′) - Δτ*α[i]*η_il′
            ΔV_il    = (+α[i] * η_il′) - (+α[i] * η_il)
            Vup_il′  = ΔV_il + Vup_il
            Vdn_il′  = ΔV_il + Vdn_il

            # calculate spin-up determinant ratio associated with Ising HS spin flip
            Rup_il, Δup_il = local_update_det_ratio(Gup, Bup_l, Vup_il′, site, Δτ)
            Rdn_il, Δdn_il = local_update_det_ratio(Gdn, Bdn_l, Vdn_il′, site, Δτ)

            # calculate the change in bosonic action
            ΔSb = Sb_il′ - Sb_il

            # calculate acceptance probability
            P_il = abs(exp(-ΔSb) * Rup_il * Rdn_il)

            # accept or reject proposed update
            if rand(rng) < P_il

                # increment the cound of accepted spin flips
                accepted_spin_flips += 1

                # flip the spin
                s[i,l] = s_il′

                # update diagonal on-site energy matrix
                Vup[site,l] = Vup_il′
                Vdn[site,l] = Vdn_il′

                # udpate bosonic action
                fermion_path_integral_up.Sb += ΔSb
                fermion_path_integral_dn.Sb += ΔSb

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
    acceptance_rate = accepted_spin_flips / length(s)

    return (acceptance_rate, logdetGup, sgndetGup, logdetGdn, sgndetGdn, δG, δθ)
end

@doc raw"""
    local_updates!(
        # ARGUMENTS
        G::Matrix{H}, logdetG::R, sgndetG::H,
        hst_parameters::HubbardDensityGaussHermiteHST{T,R};
        # KEYWORD ARGUMENTS
        fermion_path_integral::FermionPathIntegral{H},
        fermion_greens_calculator::FermionGreensCalculator{H},
        B::Vector{P},
        δG::R, δθ::R,  rng::AbstractRNG,
        δG_max::R = 1e-6,
        update_stabilization_frequency::Bool = true
    ) where {H<:Number, T<:Number, R<:Real, P<:AbstractPropagator}

Perform local updates to density-channel Hirsch Hubbard-Stratonovich fields.
This method returns a tuple containing `(acceptance_rate, logdetG, sgndetG, δG, δθ)`.

# Arguments

- `G::Matrix{H}`: Equal-time Green's function matrix.
- `logdetG::R`: The log of the absolute value of the determinant of the equal-time Green's function matrix, ``\log \vert \det G(\tau,\tau) \vert.``
- `sgndetG::H`: The sign/phase of the determinant of the equal-time Green's function matrix, ``\det G(\tau,\tau) / \vert \det G(\tau,\tau) \vert.``
- `hst_parameters::HubbardDensityGaussHermiteHST{T,R}`: Type representing Hubbard-Stratonovich transformation.

## Keyword Arguments

- `fermion_path_integral::FermionPathIntegral{H}`: An instance of the [`FermionPathIntegral`](@ref).
- `fermion_greens_calculator::FermionGreensCalculator{H}`: An instance of the [`FermionGreensCalculator`](https://smoqysuite.github.io/JDQMCFramework.jl/stable/api/#JDQMCFramework.FermionGreensCalculator) type.
- `B::Vector{P}`: Propagators for each imaginary time slice.
- `δG_max::R`: Maximum allowed error corrected by numerical stabilization.
- `δG::R`: Previously recorded maximum error in the Green's function corrected by numerical stabilization.
- `δθ::R`: Previously recorded maximum error in the sign/phase of the determinant of the equal-time Green's function matrix corrected by numerical stabilization.
- `rng::AbstractRNG`: Random number generator used in method instead of global random number generator, important for reproducibility.
- `update_stabilization_frequency::Bool = true`: If true, allows the stabilization frequency `n_stab` to be dynamically adjusted.
"""
function local_updates!(
    # ARGUMENTS
    G::Matrix{H}, logdetG::R, sgndetG::H,
    hst_parameters::HubbardDensityGaussHermiteHST{T,R};
    # KEYWORD ARGUMENTS
    fermion_path_integral::FermionPathIntegral{H},
    fermion_greens_calculator::FermionGreensCalculator{H},
    B::Vector{P},
    δG::R, δθ::R,  rng::AbstractRNG,
    δG_max::R = 1e-6,
    update_stabilization_frequency::Bool = true
) where {H<:Number, T<:Number, R<:Real, P<:AbstractPropagator}

    @assert !( (H<:Real) &&  (T<:Complex)) "Green's function matrices are real while Hubbard-Stratonovich transformation is complex."

    (; Δτ, U, α, sites, s, update_perm, N) = hst_parameters
    (; u, v) = fermion_path_integral

    # get temporary storage matrix
    G′ = fermion_greens_calculator.G′

    # get on-site energy matrices for spin up and down electrons for all time slices
    V = fermion_path_integral.V

    # counter for the number of accepted spin flips
    accepted_spin_flips = 0

    # Iterate over imaginary time τ=Δτ⋅l.
    for l in fermion_greens_calculator

        # Propagate equal-time Green's function matrix to current imaginary time G(τ±Δτ,τ±Δτ) ==> G(τ,τ)
        # depending on whether iterating over imaginary time in the forward or reverse direction
        propagate_equaltime_greens!(G, fermion_greens_calculator, B)

        # get propagators for current time slice
        B_l = B[l]

        # apply the transformation G̃(τ,τ) = exp(+Δτ⋅K[l]/2)⋅G(τ,τ)⋅exp(-Δτ⋅K[l]/2)
        # if B[l] = exp(-Δτ⋅K[l]/2)⋅exp(-Δτ⋅V[l])⋅exp(-Δτ⋅K[l]/2),
        # otherwise nothing when B[l] = exp(-Δτ⋅V[l])⋅exp(-Δτ⋅K[l])
        partially_wrap_greens_reverse!(G, B_l, G′)

        # shuffle the order in which orbitals/sites will be iterated over
        shuffle!(rng, update_perm)

        # iterate over orbitals in the lattice
        for i in update_perm

            # get the site
            site = sites[i]

            # propose a HS field update and calculate the change in the potential
            # energy matrix and bosonic action
            s_il   = s[i,l]
            η_il   = eval_η(s_il)
            V_il   = V[site,l]
            Sb_il  = eval_Sgh(s_il) - Δτ*α[i]*η_il
            s_il′  = sample_new_ghhsf(rng, s_il)
            η_il′  = eval_η(s_il′)
            Sb_il′ = eval_Sgh(s_il′) - Δτ*α[i]*η_il′
            ΔV_il  = (+α[i] * η_il′) - (+α[i] * η_il)
            V_il′  = V_il + ΔV_il

            # calculate spin-up determinant ratio associated with Ising HS spin flip
            R_il, Δ_il = local_update_det_ratio(G, B_l, V_il′, site, Δτ)

            # calculate the change in bosonic action
            ΔSb = Sb_il′ - Sb_il

            # calculate acceptance probability
            P_il = abs(exp(-ΔSb) * R_il^2)

            # accept or reject proposed update
            if rand(rng) < P_il

                # increment the cound of accepted spin flips
                accepted_spin_flips += 1

                # flip the spin
                s[i,l] = s_il′

                # update diagonal on-site energy matrix
                V[site,l] = V_il′

                # udpate bosonic action
                fermion_path_integral.Sb += ΔSb

                # update the spin-up and down Green's function
                logdetG, sgndetG = local_update_greens!(G, logdetG, sgndetG, B_l, R_il, Δ_il, site, u, v)
            end
        end

        # apply the transformation G(τ,τ) = exp(-Δτ⋅K[l]/2)⋅G̃(τ,τ)⋅exp(+Δτ⋅K[l]/2)
        # if B[l] = exp(-Δτ⋅K[l]/2)⋅exp(-Δτ⋅V[l])⋅exp(-Δτ⋅K[l]/2),
        # otherwise nothing when B[l] = exp(-Δτ⋅V[l])⋅exp(-Δτ⋅K[l])
        partially_wrap_greens_forward!(G, B_l, G′)

        # Periodically re-calculate the Green's function matrix for numerical stability.
        logdetG, sgndetG, δG, δθ = stabilize_equaltime_greens!(G, logdetG, sgndetG, fermion_greens_calculator, B, update_B̄=true)
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
    acceptance_rate = accepted_spin_flips / length(s)

    return (acceptance_rate, logdetG, sgndetG, δG, δθ)
end


@doc raw"""
    reflection_update!(
        # ARGUMENTS
        Gup::Matrix{H}, logdetGup::R, sgndetGup::H,
        Gdn::Matrix{H}, logdetGdn::R, sgndetGdn::H,
        hst_parameters::HubbardDensityGaussHermiteHST{T,R};
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
- `hst_parameters::HubbardDensityGaussHermiteHST{T,R}`: Hubbard-Stratonovich fields and associated parameters to update.

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
    hst_parameters::HubbardDensityGaussHermiteHST{T,R};
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

    @assert fermion_path_integral_up.Sb == fermion_path_integral_dn.Sb "$(fermion_path_integral_up.Sb) ≠ $(fermion_path_integral_dn.Sb)"

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

    # calculate initial bosonic action
    Sb = -Δτ * α[i] * sum(eval_η, s_i)

    # reflect all the HS field on site i
    s_i′ = s_i
    @. s_i′ = -s_i

    # calculate final bosonic action
    Sb′ = -Δτ * α[i] * sum(eval_η, s_i)

    # calculate change in bosonic action
    ΔSb = Sb′ - Sb

    # update diagonal on-site energy matrix:
    @. Vup_i += (+α[i]*eval_η(s_i′)) - (+α[i]*eval_η(-s_i′))
    @. Vdn_i += (+α[i]*eval_η(s_i′)) - (+α[i]*eval_η(-s_i′))

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

    # calculate acceptance probability P = exp(-ΔS) = exp(-ΔSb - ΔSf) = exp(-ΔSb - (Sf′ - Sf))
    #                                    = exp(-ΔSb - (logdetGup′ + logdetGdn′ - logdetGup - logdetGdn))
    #                                    = exp(-ΔSb + logdetGup + logdetGdn - logdetGup′ - logdetGdn′)
    if isfinite(logdetGup′) && isfinite(logdetGdn′)
        P_i = exp(-real(ΔSb) + logdetGup + logdetGdn - logdetGup′ - logdetGdn′)
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
        @. s_i = -s_i′
        # revert diagonal on-site energy matrix
        @. Vup_i += (+α[i]*eval_η(s_i′)) - (+α[i]*eval_η(-s_i′))
        @. Vdn_i += (+α[i]*eval_η(s_i′)) - (+α[i]*eval_η(-s_i′))
        # revert propagator matrices
        @inbounds for l in eachindex(Bup)
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
        hst_parameters::HubbardDensityGaussHermiteHST{T,R};
        # KEYWORD ARGUMENTS
        fermion_path_integral::FermionPathIntegral{H},
        fermion_greens_calculator::FermionGreensCalculator{H,R},
        fermion_greens_calculator_alt::FermionGreensCalculator{H,R},
        B::Vector{P},
        rng::AbstractRNG
    ) where {H<:Number, T<:Number, R<:Real, P<:AbstractPropagator}

Perform a reflection update in which the sign of every spin-channel Hirsch Hubbard-Stratonovich field on a randomly chosen orbital in the lattice is changed.
This function returns `(accepted, logdetG, sgndetG)`.

# Arguments

- `G::Matrix{H}`: Equal-time Greens function matrix.
- `logdetG::R`: Log of the determinant of the spin-up equal-time Greens function matrix.
- `sgndetG::H`: Sign/phase of the determinant of the spin-up equal-time Greens function matrix.
- `hst_parameters::HubbardDensityGaussHermiteHST{T,R}`: Hubbard-Stratonovich fields and associated parameters to update.

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
    hst_parameters::HubbardDensityGaussHermiteHST{T,R};
    # KEYWORD ARGUMENTS
    fermion_path_integral::FermionPathIntegral{H},
    fermion_greens_calculator::FermionGreensCalculator{H,R},
    fermion_greens_calculator_alt::FermionGreensCalculator{H,R},
    B::Vector{P},
    rng::AbstractRNG
) where {H<:Number, T<:Number, R<:Real, P<:AbstractPropagator}

    (; Δτ, α, sites, s, N) = hst_parameters
    G′ = fermion_greens_calculator_alt.G′

    # make sure stabilization frequencies match
    if fermion_greens_calculator.n_stab != fermion_greens_calculator_alt.n_stab
        resize!(fermion_greens_calculator_alt, fermion_greens_calculator.n_stab)
    end

    # pick a random site/orbital in lattice with finite Hubbard U to perform reflection update on
    i     = rand(rng, 1:N)
    site  = sites[i]
    s_i   = @view s[i, :]
    V_i = @view fermion_path_integral.V[site, :]

    # calculate initial bosonic action
    Sb = -Δτ * α[i] * sum(eval_η, s_i)

    # reflect all the HS field on site i
    s_i′ = s_i
    @. s_i′ = -s_i

    # calculate final bosonic action
    Sb′ = -Δτ * α[i] * sum(eval_η, s_i)

    # calculate change in bosonic action
    ΔSb = Sb′ - Sb

    # update diagonal on-site energy matrix:
    @. V_i += (+α[i]*eval_η(s_i′)) - (+α[i]*eval_η(-s_i′))

    # update propagator matrices
    @inbounds for l in eachindex(B)
        expmΔτV_l = B[l].expmΔτV
        expmΔτV_l[site] = exp(-Δτ*V_i[l])
    end

    # calculate new Green's function matrices and determinant of new Green's function matrix
    logdetG′, sgndetG′ = calculate_equaltime_greens!(G′, fermion_greens_calculator_alt, B)

    # calculate acceptance probability P = exp(-ΔS) = exp(-ΔSb - ΔSf) = exp(-ΔSb - (Sf′ - Sf))
    #                                    = exp(-ΔSb - (logdetGup′ + logdetGdn′ - logdetGup - logdetGdn))
    #                                    = exp(-ΔSb + logdetGup + logdetGdn - logdetGup′ - logdetGdn′)
    if isfinite(logdetG′)
        P_i = exp(-real(ΔSb) + 2*logdetG - 2*logdetG′)
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
        @. s_i = -s_i′
        # revert diagonal on-site energy matrix
        @. V_i += (+α[i]*eval_η(s_i′)) - (+α[i]*eval_η(-s_i′))
        # revert propagator matrices
        @inbounds for l in eachindex(B)
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
        hst_parameters::HubbardDensityGaussHermiteHST{T,R};
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
- `hst_parameters::HubbardDensityGaussHermiteHST{T,R}`: Hubbard-Stratonovich fields and associated parameters to update.

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
    hst_parameters::HubbardDensityGaussHermiteHST{T,R};
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

    @assert fermion_path_integral_up.Sb == fermion_path_integral_dn.Sb "$(fermion_path_integral_up.Sb) ≠ $(fermion_path_integral_dn.Sb)"

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

    # calculate initial bosonic action associated with pair of sites
    Sb = -Δτ * α[i] * sum(eval_η, s_i) - Δτ * α[j] * sum(eval_η, s_j)

    # swap the HS fields
    swap!(s_i, s_j)

    # calculate initial bosonic action associated with pair of sites
    Sb′ = -Δτ * α[i] * sum(eval_η, s_i) - Δτ * α[j] * sum(eval_η, s_j)

    # calculate the change in the bosonic action
    ΔSb = Sb′ - Sb

    # update potential energy matrices
    @. Vup_i += (+α[i]*eval_η(s_i)) - (+α[i]*eval_η(s_j))
    @. Vdn_i += (+α[i]*eval_η(s_i)) - (+α[i]*eval_η(s_j))
    @. Vup_j += (+α[j]*eval_η(s_j)) - (+α[j]*eval_η(s_i))
    @. Vdn_j += (+α[j]*eval_η(s_j)) - (+α[j]*eval_η(s_i))

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

    # calculate acceptance probability
    if isfinite(logdetGup′) && isfinite(logdetGdn′)
        P_i = exp(-real(ΔSb) + logdetGup + logdetGdn - logdetGup′ - logdetGdn′)
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
        @. Vup_i += (+α[i]*eval_η(s_i)) - (+α[i]*eval_η(s_j))
        @. Vdn_i += (+α[i]*eval_η(s_i)) - (+α[i]*eval_η(s_j))
        @. Vup_j += (+α[j]*eval_η(s_j)) - (+α[j]*eval_η(s_i))
        @. Vdn_j += (+α[j]*eval_η(s_j)) - (+α[j]*eval_η(s_i))
        # revert propagator matrices
        @inbounds for l in eachindex(Bup)
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
    hst_parameters::HubbardDensityGaussHermiteHST{T,R};
    # KEYWORD ARGUMENTS
    fermion_path_integral::FermionPathIntegral{H},
    fermion_greens_calculator::FermionGreensCalculator{H,R},
    fermion_greens_calculator_alt::FermionGreensCalculator{H,R},
    Bup::Vector{P},
    rng::AbstractRNG
) where {H<:Number, T<:Number, R<:Real, P<:AbstractPropagator}

Perform a reflection update in which the sign of every spin-channel Hirsch Hubbard-Stratonovich field on a randomly chosen orbital in the lattice is changed.
This function returns `(accepted, logdetG, sgndetG)`.

# Arguments

- `G::Matrix{H}`: Eqaul-time Greens function matrix.
- `logdetG::R`: Log of the determinant of the eqaul-time Greens function matrix.
- `sgndetG::H`: Sign/phase of the determinant of the eqaul-time Greens function matrix.
- `hst_parameters::HubbardDensityGaussHermiteHST{T,R}`: Hubbard-Stratonovich fields and associated parameters to update.

# Keyword Arguments

- `fermion_path_integral::FermionPathIntegral{H}`: An instance of [`FermionPathIntegral`](@ref) type.
- `fermion_greens_calculator::FermionGreensCalculator{H,R}`: Contains matrix factorization information.
- `fermion_greens_calculator_alt::FermionGreensCalculator{H,R}`: Used to calculate matrix factorizations for proposed state.
- `B::Vector{P}`: Propagators for each imaginary time slice.
- `rng::AbstractRNG`: Random number generator used in method instead of global random number generator, important for reproducibility.
"""
function swap_update!(
    # ARGUMENTS
    G::Matrix{H}, logdetG::R, sgndetG::H,
    hst_parameters::HubbardDensityGaussHermiteHST{T,R};
    # KEYWORD ARGUMENTS
    fermion_path_integral::FermionPathIntegral{H},
    fermion_greens_calculator::FermionGreensCalculator{H,R},
    fermion_greens_calculator_alt::FermionGreensCalculator{H,R},
    B::Vector{P},
    rng::AbstractRNG
) where {H<:Number, T<:Number, R<:Real, P<:AbstractPropagator}

    (; Δτ, α, sites, s, N) = hst_parameters
    G′ = fermion_greens_calculator_alt.G′

    # make sure stabilization frequencies match
    if fermion_greens_calculator.n_stab != fermion_greens_calculator_alt.n_stab
        resize!(fermion_greens_calculator_alt, fermion_greens_calculator.n_stab)
    end

    # ranomly pick two sites with Hubbard U interaction on them
    i, j = draw2(rng, N)

    # get the site index associted with each Hubbard U
    site_i = sites[i]
    site_j = sites[j]

    # get the HS fields associated with each site
    s_i = @view s[i,:]
    s_j = @view s[j,:]
    V_i = @view fermion_path_integral.V[site_i, :]
    V_j = @view fermion_path_integral.V[site_j, :]

    # calculate initial bosonic action associated with pair of sites
    Sb = -Δτ * α[i] * sum(eval_η, s_i) - Δτ * α[j] * sum(eval_η, s_j)

    # swap the HS fields
    swap!(s_i, s_j)

    # calculate initial bosonic action associated with pair of sites
    Sb′ = -Δτ * α[i] * sum(eval_η, s_i) - Δτ * α[j] * sum(eval_η, s_j)

    # calculate the change in the bosonic action
    ΔSb = Sb′ - Sb

    # update potential energy matrices
    @. V_i += (+α[i]*eval_η(s_i)) - (+α[i]*eval_η(s_j))
    @. V_j += (+α[j]*eval_η(s_j)) - (+α[j]*eval_η(s_i))

    # update propagator matrices
    @inbounds for l in eachindex(B)
        expmΔτV_l = B[l].expmΔτV
        expmΔτV_l[site_i] = exp(-Δτ*V_i[l])
        expmΔτV_l[site_j] = exp(-Δτ*V_j[l])
    end

    # calculate new Green's function matrices and determinant of new Green's function matrix
    logdetG′, sgndetG′ = calculate_equaltime_greens!(G′, fermion_greens_calculator_alt, B)

    # calculate acceptance probability P = exp(-ΔS)
    if isfinite(logdetG′)
        P_i = exp(-real(ΔSb) + 2*logdetG - 2*logdetG′)
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
        @. V_i += (+α[i]*eval_η(s_i)) - (+α[i]*eval_η(s_j))
        @. V_j += (+α[j]*eval_η(s_j)) - (+α[j]*eval_η(s_i))
        # revert propagator matrices
        @inbounds for l in eachindex(B)
            expmΔτV_l = B[l].expmΔτV
            expmΔτV_l[site_i] = exp(-Δτ*V_i[l])
            expmΔτV_l[site_j] = exp(-Δτ*V_j[l])
        end
        accepted = false
    end

    return (accepted, logdetG, sgndetG)
end