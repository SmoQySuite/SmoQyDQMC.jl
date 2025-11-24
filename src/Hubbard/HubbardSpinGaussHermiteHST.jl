@doc raw"""
    HubbardSpinGaussHermiteHST{T,R} <: AbstractAsymHST{T,R}

This type represents a Hubbard-Stratonovich (HS) transformation for decoupling the local Hubbard interaction in the spin channel,
where the introduced HS fields take on the four discrete values ``s \in \{ -2, -1, +1, +2 \}.``
Note that the Hubbard interaction can be written in the spin channel as
```math
U(\hat{n}_{\uparrow}-\tfrac{1}{2})(\hat{n}_{\downarrow}-\tfrac{1}{2})=-\tfrac{U}{2}(\hat{n}_{\uparrow}-\hat{n}_{\downarrow})^{2}+\tfrac{U}{4},
```
different only by a constant energy offset ``U/4`` which does not matter.
Therefore, we can perform a Gauss-Hermite Hubbard-Statonovich transformation in the spin channel as
```math
e^{-\Delta\tau\left[-\frac{U}{2}\right](\hat{n}_{\uparrow}-\hat{n}_{\downarrow})^{2}}
    = \frac{1}{4}\sum_{s=\pm1,\pm2}e^{-S_{\text{GH}}(s)-\Delta\tau\hat{V}(s)}+\mathcal{O}\left((\Delta\tau U)^{4}\right),
```
where ``\hat{V}(s)=\alpha\eta(s)(\hat{n}_{\uparrow}-\hat{n}_{\downarrow})`` and ``\alpha = \sqrt{U/(2\Delta\tau)}``.
In the above expression,
```math
S_{\text{GH}}(s)=-\log\left(1+\sqrt{6}\left(1-\tfrac{2}{3}|s|\right)\right)
```
and
```math
\eta(s)=\frac{s}{|s|}\sqrt{6(1-\sqrt{6})+4\sqrt{6}|s|}.
```
Note that ``\alpha`` is strictly real when ``U \ge 0`` and strictly imaginary when ``U < 0``.
"""
struct HubbardSpinGaussHermiteHST{T,R} <: AbstractAsymHST{T,R}

    # inverse temperature
    β::R

    # discretization in imaginary time
    Δτ::R

    # length of imaginary time axis
    Lτ::Int

    # number of orbitals with finite Hubbard U
    N::Int

    # each finite hubbard interaction
    U::Vector{R}
    
    # HST coupling coefficient
    α::Vector{T}

    # site index associated with each Hubbard U
    sites::Vector{Int}

    # Hubbard-Stratonovich fields
    s::Matrix{Int}

    # order in which to iterate over orbitals when updating Hubbard-Stratonovich fields.
    update_perm::Vector{Int}
end

@doc raw"""
    HubbardSpinGaussHermiteHST(;
        # KEYWORD ARGUMENTS
        hubbard_parameters::HubbardParameters{E},
        β::E, Δτ::E, rng::AbstractRNG
    ) where {E<:AbstractFloat}

Initialize an instance of the [`HubbardSpinGaussHermiteHST`](@ref) type.
"""
function HubbardSpinGaussHermiteHST(;
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
    @. α = sqrt(T(U)/(2*Δτ))

    # number of sites with Hubbard interaction
    N = length(U)

    # initialize HS fields
    s = rand(rng, (-2,-1,+1,+2), (N, Lτ))

    # initialize update permuation order
    update_perm = collect(1:N)

    return HubbardSpinGaussHermiteHST{T,E}(β, Δτ, Lτ, N, U, α, sites, s, update_perm)
end

# initialize fermion path integral to reflect HS field config
function _initialize!(
    fermion_path_integral_up::FermionPathIntegral{H},
    fermion_path_integral_dn::FermionPathIntegral{H},
    hst_parameters::HubbardSpinGaussHermiteHST{T}
) where {H<:Number, T<:Number}

    @assert !((H<:Real) &&  (T<:Complex)) "Green's function matrices are real while HubbardSpinGaussHermiteHST is complex."
    @assert fermion_path_integral_up.Sb == fermion_path_integral_dn.Sb "$(fermion_path_integral_up.Sb) ≠ $(fermion_path_integral_dn.Sb)"

    (; sites, α, s) = hst_parameters
    Vup = fermion_path_integral_up.V
    Vdn = fermion_path_integral_dn.V

    # iterate over sites with Hubbard U interactions
    for i in eachindex(sites)
        site = sites[i]
        @views @. Vup[site,:] += α[i] * eval_η(s[i,:])
        @views @. Vdn[site,:] -= α[i] * eval_η(s[i,:])
    end

    # calculate the contribution to the bosonic action
    Sb = eval_Sgh(s)
    fermion_path_integral_up.Sb += Sb
    fermion_path_integral_dn.Sb += Sb

    return nothing
end


function _local_updates!(
    Gup::Matrix{H}, logdetGup::R, sgndetGup::H,
    Gdn::Matrix{H}, logdetGdn::R, sgndetGdn::H,
    hst_parameters::HubbardSpinGaussHermiteHST{T,R},
    fermion_path_integral_up::FermionPathIntegral{H},
    fermion_path_integral_dn::FermionPathIntegral{H},
    Bup::P, Bdn::P, l::Int, rng::AbstractRNG
) where {H<:Number, T<:Number, R<:Real, P<:AbstractPropagator}

    # make sure bosonic actions match
    @assert fermion_path_integral_up.Sb == fermion_path_integral_dn.Sb "$(fermion_path_integral_up.Sb) ≠ $(fermion_path_integral_dn.Sb)"

    (; Δτ, U, α, sites, s, update_perm, N) = hst_parameters
    u = @view fermion_path_integral_up.u[:,1]
    v = @view fermion_path_integral_up.v[:,1]

    # get on-site energy matrices for spin up and down electrons for all time slices
    Vup = fermion_path_integral_up.V
    Vdn = fermion_path_integral_dn.V

    # get Hubbard-Stratonovich transformation
    s = @view hst_parameters.s[:,l]

    # counter for the number of accepted spin flips
    accepted_spin_flips = 0

    # shuffle the order in which orbitals/sites will be iterated over
    shuffle!(rng, update_perm)

    # iterate over orbitals in the lattice
    for i in update_perm

        # get the site
        site = sites[i]

        # propose a HS field update and calculate the change in the potential
        # energy matrix and bosonic action
        s_il    = s[i]
        η_il    = eval_η(s_il)
        Sb_il   = eval_Sgh(s_il)
        s_il′   = sample_new_ghhsf(rng, s_il)
        η_il′   = eval_η(s_il′)
        Sb_il′  = eval_Sgh(s_il′)
        ΔVup_il = (+α[i] * η_il′) - (+α[i] * η_il)
        ΔVdn_il = (-α[i] * η_il′) - (-α[i] * η_il)

        # calculate spin-up determinant ratio associated with Ising HS spin flip
        Rup_il, Δup_il = local_update_det_ratio(Gup, ΔVup_il, site, Δτ)
        Rdn_il, Δdn_il = local_update_det_ratio(Gdn, ΔVdn_il, site, Δτ)

        # calculate the change in bosonic action
        ΔSb = Sb_il′ - Sb_il

        # calculate acceptance probability
        P_il = abs(exp(-ΔSb) * Rup_il * Rdn_il)

        # accept or reject proposed update
        if rand(rng) < P_il

            # increment the count of accepted spin flips
            accepted_spin_flips += 1

            # flip the spin
            s[i] = s_il′

            # update diagonal on-site energy matrix
            Vup[site,l] += ΔVup_il
            Vdn[site,l] += ΔVdn_il

            # update bosonic action
            fermion_path_integral_up.Sb += ΔSb
            fermion_path_integral_dn.Sb += ΔSb

            # update the spin-up and down Green's function
            logdetGup, sgndetGup = local_update_greens!(Gup, logdetGup, sgndetGup, Bup, Rup_il, Δup_il, site, u, v)
            logdetGdn, sgndetGdn = local_update_greens!(Gdn, logdetGdn, sgndetGdn, Bdn, Rdn_il, Δdn_il, site, u, v)
        end
    end

    # calculate the acceptance rate
    acceptance_rate = accepted_spin_flips / length(s)

    return acceptance_rate, logdetGup, sgndetGup, logdetGdn, sgndetGdn
end

# perform reflection update
function _reflection_update!(
    # ARGUMENTS
    Gup::Matrix{H}, logdetGup::R, sgndetGup::H,
    Gdn::Matrix{H}, logdetGdn::R, sgndetGdn::H,
    hst_parameters::HubbardSpinGaussHermiteHST{T,R};
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
    
    # reflect all the HS field on site i
    s_i′ = s_i
    @. s_i′ = -s_i

    # update diagonal on-site energy matrix:
    @. Vup_i += (+α[i]*eval_η(s_i′)) - (+α[i]*eval_η(-s_i′))
    @. Vdn_i += (-α[i]*eval_η(s_i′)) - (-α[i]*eval_η(-s_i′))

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

    # calculate acceptance probability
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
        @. s_i = -s_i′
        # revert diagonal on-site energy matrix
        @. Vup_i += (+α[i]*eval_η(s_i)) - (+α[i]*eval_η(-s_i))
        @. Vdn_i += (-α[i]*eval_η(s_i)) - (-α[i]*eval_η(-s_i))
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


# perform swap update
function _swap_update!(
    # ARGUMENTS
    Gup::Matrix{H}, logdetGup::R, sgndetGup::H,
    Gdn::Matrix{H}, logdetGdn::R, sgndetGdn::H,
    hst_parameters::HubbardSpinGaussHermiteHST{T,R};
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

    # swap the HS fields
    swap!(s_i, s_j)

    # update potential energy matrices
    @. Vup_i += (+α[i]*eval_η(s_i)) - (+α[i]*eval_η(s_j))
    @. Vdn_i += (-α[i]*eval_η(s_i)) - (-α[i]*eval_η(s_j))
    @. Vup_j += (+α[j]*eval_η(s_j)) - (+α[j]*eval_η(s_i))
    @. Vdn_j += (-α[j]*eval_η(s_j)) - (-α[j]*eval_η(s_i))

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
        @. Vup_i += (+α[i]*eval_η(s_i)) - (+α[i]*eval_η(s_j))
        @. Vdn_i += (-α[i]*eval_η(s_i)) - (-α[i]*eval_η(s_j))
        @. Vup_j += (+α[j]*eval_η(s_j)) - (+α[j]*eval_η(s_i))
        @. Vdn_j += (-α[j]*eval_η(s_j)) - (-α[j]*eval_η(s_i))
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