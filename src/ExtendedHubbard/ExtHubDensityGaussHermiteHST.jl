@doc raw"""
    ExtHubDensityGaussHermiteHST{T<:Number, E<:AbstractFloat}

This type represents a Hubbard-Stratonovich (HS) transformation for decoupling the extended Hubbard interaction,
where the introduced HS fields take on the four discrete values ``s \in \{ -2, -1, +1, +2 \}.``

Specifically, we perform the Gauss-Hermite Hubbard-Stratonovich transformation
```math
e^{-\Delta\tau\left[\tfrac{V}{2}\right](\hat{n}_{\mathbf{i}}+\hat{n}_{\mathbf{j}}-2)^{2}} =
= \frac{1}{4}\sum_{s=\pm1,\pm2}e^{-S_{\text{GH}}(s)-\Delta\tau\hat{V}(s)}+\mathcal{O}\left(\left[\tfrac{\Delta\tau V}{2}\right]^{4}\right)
```
where ``hat{V}(s) = \alpha\eta(s)(\hat{n}_{\mathbf{i}}+\hat{n}_{\mathbf{j}}-2)`` and ``\alpha=\sqrt{\frac{-V}{2\Delta\tau}}``.
In the above expression,
```math
S_{\text{GH}}(s)=-\log\left(1+\sqrt{6}\left(1-\tfrac{2}{3}|s|\right)\right)
```
and
```math
\eta(s)=\frac{s}{|s|}\sqrt{6(1-\sqrt{6})+4\sqrt{6}|s|}.
```
Note that ``\alpha`` is strictly real when ``V \le 0`` and strictly imaginary when ``V > 0``.
"""
struct ExtHubDensityGaussHermiteHST{T<:Number, E<:AbstractFloat}

    # inverse temperature
    β::E

    # discretization in imaginary time
    Δτ::E

    # length of imaginary time axis
    Lτ::Int

    # number of bonds with finite Hubbard U
    N::Int

    # each finite extended hubbard interaction
    V::Vector{E}
    
    # HST coupling coefficient
    α::Vector{T}

    # site index associated with each Hubbard U
    neighbor_table::Matrix{Int}

    # Hubbard-Stratonovich fields
    s::Matrix{Int}

    # order in which to iterate over orbitals when updating Hubbard-Stratonovich fields.
    update_perm::Vector{Int}

    # record the bond ID types associated with extended hubbard interactions
    bond_ids::Vector{Int}

    # whether particle-hole symmetric form for extended Hubbard interaction was used
    ph_sym_form::Bool
end


@doc raw"""
    ExtHubDensityGaussHermiteHST(;
        # KEYWORD ARGUMENTS
        extended_hubbard_parameters::ExtendedHubbardParameters{E},
        β::E, Δτ::E, rng::AbstractRNG
    ) where {E<:AbstractFloat}

Initialize an instance of the [`ExtHubDensityGaussHermiteHST`](@ref) type.
"""
function ExtHubDensityGaussHermiteHST(;
    # KEYWORD ARGUMENTS
    extended_hubbard_parameters::ExtendedHubbardParameters{E},
    β::E, Δτ::E, rng::AbstractRNG
) where {E<:AbstractFloat}

    (; V, neighbor_table, bond_ids, ph_sym_form) = extended_hubbard_parameters

    # if any attractive Hubbard interactions, then complex field coefficients
    T = any(v -> v > 0, V) ? Complex{E} : E

    # calculate length of imaginary-time axis
    Lτ = round(Int, β / Δτ)

    # calculate HS transformation coefficients
    α = zeros(T, length(V))
    @. α = sqrt(-T(V)/(2*Δτ))

    # number of sites with Hubbard interaction
    N = length(V)

    # initialize HS fields
    s = rand(rng, (-2,-1,+1,+2), (N, Lτ))

    # initialize update permuation order
    update_perm = collect(1:N)

    return ExtHubDensityGaussHermiteHST{T,E}(β, Δτ, Lτ, N, V, α, neighbor_table, s, update_perm, bond_ids, ph_sym_form)
end


@doc raw"""
    init_renormalized_hubbard_parameters(;
        # KEYWORD ARGUMENTS
        hubbard_parameters::HubbardParameters{E},
        hst_parameters::ExtHubDensityGaussHermiteHST{T,E},
        model_geometry::ModelGeometry{D,E}
    ) where {D, T<:Number, E<:AbstractFloat}

Returns a new instance of the [`HubbardParameters`](@ref) type with the Hubbard interactions renormalized
based on the [`ExtHubDensityGaussHermiteHST`](@ref) definition. Refer to the definition of the
[`ExtendedHubbardModel`](@ref) to see where this renormalization of the local Hubbard interaction comes from.

Note that either both the local and extended Hubbard interactions need to be initialized using the particle-hole
symmetric or asymmetric form for the interaction (as specified by `ph_sym_form` keyword argument), and cannot use opposite conventions.
Additionally, the [`HubbardModel`](@ref) defintion used to create the `hubbard_parameters` instance of the [`HubbardParameters`](@ref)
passed to this function must initialize a Hubbard interaction on each type of orbital species/ID appearing
in an extended Hubbard interaction, even if this means initializing the local Hubbard interaction to ``U = 0``.
"""
function init_renormalized_hubbard_parameters(;
    # KEYWORD ARGUMENTS
    hubbard_parameters::HubbardParameters{E},
    hst_parameters::ExtHubDensityGaussHermiteHST{T,E},
    model_geometry::ModelGeometry{D,E}
) where {D, T<:Number, E<:AbstractFloat}

    @assert(
        hubbard_parameters.ph_sym_form == hst_parameters.ph_sym_form,
        "Either both the local and extended Hubbard interactions need to use the particle-hole symmetric or asymmetric form, they cannot use opposite conventions."
    )

    (; U, sites, orbital_ids, ph_sym_form) = hubbard_parameters
    (; V, neighbor_table, bond_ids) = hst_parameters
    (; bonds, unit_cell, lattice) = model_geometry

    
    # iterate over bonds that define extended hubbard interactions
    for bond_id in bond_ids

        # get orbital species associated with bond
        a, b = bonds[bond_id].orbitals

        # check to make sure Hubbard interaction are defined for each type of bond
        @assert (a ∈ orbital_ids) "Hubbard interaction for ORBITAL_ID = $a needs to initialized in HubbardModel definition. Note that initialization to U = 0 is allowed."
        @assert (b ∈ orbital_ids) "Hubbard interaction for ORBITAL_ID = $b needs to initialized in HubbardModel definition. Note that initialization to U = 0 is allowed."
    end

    # number of unit cells in lattice
    N = lattice.N

    # number of types of extended Hubbard interactions
    n_V = length(bond_ids)

    # number of types of Hubbard interactions
    n_U = length(orbital_ids)

    # copy bare hubbard interaction
    Ũ = copy(U)

    # iterate over extended Hubbard interaction neighbors
    for n in axes(neighbor_table, 2)

        # get the pair of orbitals with extended Hubbard interactions
        i = neighbor_table[1,n]
        j = neighbor_table[2,n]

        # get the Hubbard U index associated with each site
        m_i = findfirst(e -> e == i, sites)
        m_j = findfirst(e -> e == j, sites)

        # calculate renormalized hubbard interaction
        Ũ[m_i] -= V[n]
        Ũ[m_j] -= V[n]
    end

    return HubbardParameters(Ũ, sites, orbital_ids, ph_sym_form)
end


@doc raw"""
    initialize!(
        fermion_path_integral_up::FermionPathIntegral{H},
        fermion_path_integral_dn::FermionPathIntegral{H},
        hst_parameters::ExtHubDensityGaussHermiteHST{T}
    ) where {H<:Number, T<:Number}

    initialize!(
        fermion_path_integral::FermionPathIntegral{H},
        hst_parameters::ExtHubDensityGaussHermiteHST{T},
    ) where {H<:Number, T<:Number}

Initialize [`FermionPathIntegral`](@ref) instances to reflect the initial
HS field configuration represented by the [`ExtHubDensityGaussHermiteHST`](@ref) type.
"""
function initialize!(
    fermion_path_integral_up::FermionPathIntegral{H},
    fermion_path_integral_dn::FermionPathIntegral{H},
    hst_parameters::ExtHubDensityGaussHermiteHST{T}
) where {H<:Number, T<:Number}

    # make sure bosonic actions match
    @assert fermion_path_integral_up.Sb == fermion_path_integral_dn.Sb "$(fermion_path_integral_up.Sb) ≠ $(fermion_path_integral_dn.Sb)"

    initialize!(fermion_path_integral_up, hst_parameters)
    initialize!(fermion_path_integral_dn, hst_parameters)

    return nothing
end

function initialize!(
    fermion_path_integral::FermionPathIntegral{H},
    hst_parameters::ExtHubDensityGaussHermiteHST{T},
) where {H<:Number, T<:Number}

    @assert !( (H<:Real) &&  (T<:Complex)) "Green's function matrices are real while Hubbard-Stratonovich transformation is complex."

    (; neighbor_table, α, s, Δτ, ph_sym_form) = hst_parameters
    V′ = hst_parameters.V
    V = fermion_path_integral.V

    # iterate over sites with Hubbard U interactions
    for n in axes(neighbor_table, 2)
        i = neighbor_table[1, n]
        j = neighbor_table[2, n]
        s_n = @view s[n,:]
        V_i = @view V[i,:]
        V_j = @view V[j,:]
        @. V_i += α[n] * eval_η(s_n)
        @. V_j += α[n] * eval_η(s_n)
        fermion_path_integral.Sb += eval_Sgh(s_n) - 2*Δτ*α[n]*sum(eval_η, s_n)
    end

    # resize u and v arrays for performing local updates
    if isone(size(fermion_path_integral.u, 2))
        Nsites = size(fermion_path_integral.u, 1)
        fermion_path_integral.u = zeros(H, (Nsites, 2))
        fermion_path_integral.v = zeros(H, (Nsites, 2))
    end

    # modify on-site energy if asymmetric form for coupling is being used
    # shift on-site energies if necessary
    if !ph_sym_form
        # iterate over imaginary-time slices
        for l in axes(V,2)
            # iterate of extended Hubbard interaction coupling
            for n in eachindex(V′)
                # get the pair of sites connected by extended hubbard interaction
                i = neighbor_table[1,n]
                j = neighbor_table[2,n]
                # shift on-site energies by +V/2
                V[i,l] = V[i,l] + 0.5*V′[n]
                V[j,l] = V[j,l] + 0.5*V′[n]
            end
        end
    end


    return nothing
end


# calculate determinant ratio
function eval_R(
    G::AbstractMatrix{T}, Δ::E, i::Int, j::Int
) where {T<:Number, E<:Number}

    return (1 + Δ*(1-G[i,i])) * (1 + Δ*(1-G[j,j])) - Δ^2*G[i,j]*G[j,i]
end

# PERFORMANCE CRITICAL FUNCTION!!!
# udpate green's function matrix.
function update_G!(
    G::AbstractMatrix{H}, logdetG::E, sgndetG::T,
    B::AbstractPropagator,
    Δ::Number, R::Number, i::Int, j::Int,
    u::AbstractMatrix{H}, v::AbstractMatrix{H}
) where {H<:Number, T<:Number, E<:Real}

    # vector if indice pairs
    ij = SVector(i,j)
    # update spin-up green's function matrix
    D = SMatrix{2,2}( 1+Δ*(1-G[j,j]) , Δ*G[j,i] , Δ*G[i,j] , 1+Δ*(1-G[i,i]) )
    # u = G[:,(i,j)]⋅D where u is a (N×2) matrix, G[:,(i,j)] is an (N×2) matrix, and D is a (2×2) matrix
    G_ij_columns = @view G[:,ij]
    mul!(u, G_ij_columns, D)
    # vᵀ = G[(i,j),:] - I[(i,j),:] where vᵀ is a (2×N) matrix, G[(i,j),:] is a (2×N) matrix, and I[(i,j),:] is a (2×N) matrix
    G_ij_rows = @view G[ij,:]
    vt = adjoint(v)
    copyto!(vt, G_ij_rows)
    vt[1,i] = vt[1,i] - 1
    vt[2,j] = vt[2,j] - 1
    # G = G + Δ/R⋅[u⋅vᵀ] = G + Δ/R ⋅ (G[:,(i,j)]⋅D) ⋅ (G[(i,j),:]-I[(i,j),:])
    BLAS.gemm!('N', 'C', T(Δ/R), u, v, T(1.0), G)
    # update the propagator matrix
    B.expmΔτV[i] = (1 + Δ) * B.expmΔτV[i]
    B.expmΔτV[j] = (1 + Δ) * B.expmΔτV[j]
    # update determinant
    invR = inv(R)
    logdetG′ = logdetG + log(abs(invR))
    sgndetG′ = sign(invR) * sgndetG

    return logdetG′, sgndetG′
end


@doc raw"""
    local_updates!(
        # ARGUMENTS
        Gup::Matrix{H}, logdetGup::R, sgndetGup::H,
        Gdn::Matrix{H}, logdetGdn::R, sgndetGdn::H,
        hst_parameters::ExtHubDensityGaussHermiteHST{T,R};
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

Perform local updates to extended Hubbard interaction Gauss-Hermite Hubbard-Stratonovich fields.
This method returns a tuple containing `(acceptance_rate, logdetGup, sgndetGup, logdetGdn, sgndetGdn, δG, δθ)`.

# Arguments

- `Gup::Matrix{H}`: Spin-up equal-time Green's function matrix.
- `logdetGup::R`: The log of the absolute value of the determinant of the spin-up equal-time Green's function matrix, ``\log \vert \det G_\uparrow(\tau,\tau) \vert.``
- `sgndetGup::H`: The sign/phase of the determinant of the spin-up equal-time Green's function matrix, ``\det G_\uparrow(\tau,\tau) / \vert \det G_\uparrow(\tau,\tau) \vert.``
- `Gdn::Matrix{H}`: Spin-down equal-time Green's function matrix.
- `logdetGdn::R`: The log of the absolute value of the determinant of the spin-down equal-time Green's function matrix, ``\log \vert \det G_\downarrow(\tau,\tau) \vert.``
- `sgndetGdn::H`: The sign/phase of the determinant of the spin-down equal-time Green's function matrix, ``\det G_\downarrow(\tau,\tau) / \vert \det G_\downarrow(\tau,\tau) \vert.``
- `hst_parameters::ExtHubDensityGaussHermiteHST{T,R}`: Type representing Hubbard-Stratonovich transformation.

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
    hst_parameters::ExtHubDensityGaussHermiteHST{T,R};
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

    (; Δτ, α, neighbor_table, s, update_perm, N) = hst_parameters

    if size(fermion_path_integral_up.u,2) > 2
        u = @view fermion_path_integral_up.u[:,1:2]
        v = @view fermion_path_integral_up.v[:,1:2]
    else
        (; u, v) = fermion_path_integral_up
    end

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

        # iterate over bonds in the lattice
        for n in update_perm

            # get the pair of sites coupled associated with extended hubbard interaction
            i = neighbor_table[1,n]
            j = neighbor_table[2,n]
            (i,j) = (i>j) ? (j,i) : (i,j)

            # propose a HS field update and calculate the change in the potential
            # energy matrix and bosonic action
            s_nl    = s[n,l]
            η_nl    = eval_η(s_nl)
            Sb_nl   = eval_Sgh(s_nl) - 2*Δτ*α[n]*η_nl
            s_nl′   = sample_new_ghhsf(rng, s_nl)
            η_nl′   = eval_η(s_nl′)
            Sb_nl′  = eval_Sgh(s_nl′) - 2*Δτ*α[n]*η_nl′
            ΔV_nl   = (+α[n] * η_nl′) - (+α[n] * η_nl)

            # Note that Δ_il = exp(-Δτ[V_il(s′) - V_il(s)]) - 1 and Δ_jl = exp(-Δτ[V_jl(s′) - V_jl(s)]) - 1
            # are equal because we are coupling to a density channel, and there we say that
            # Δ_nl = Δ_il = Δ_jl where n labels the bond connecting sites i and j
            Δ_nl = expm1(-Δτ*ΔV_nl)

            # calculate the determinant ratio for each spin species
            Rup_nl = eval_R(Gup, Δ_nl, i, j)
            Rdn_nl = eval_R(Gdn, Δ_nl, i, j)

            # calculate the change in bosonic action
            ΔSb = Sb_nl′ - Sb_nl

            # calculate acceptance probability
            P_nl = abs(exp(-ΔSb) * Rup_nl * Rdn_nl)

            # accept or reject proposed update
            if rand(rng) < P_nl

                # increment the cound of accepted spin flips
                accepted_spin_flips += 1

                # update HS field
                s[n,l] = s_nl′

                # update diagonal on-site energy matrix
                Vup[i,l] += ΔV_nl
                Vup[j,l] += ΔV_nl
                Vdn[i,l] += ΔV_nl
                Vdn[j,l] += ΔV_nl

                # udpate bosonic action
                fermion_path_integral_up.Sb += ΔSb
                fermion_path_integral_dn.Sb += ΔSb

                # update spin-up green's function matrix
                logdetGup, sgndetGup = update_G!(Gup, logdetGup, sgndetGup, Bup_l, Δ_nl, Rup_nl, i, j, u, v)
                logdetGdn, sgndetGdn = update_G!(Gdn, logdetGdn, sgndetGdn, Bdn_l, Δ_nl, Rdn_nl, i, j, u, v)
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
        hst_parameters::ExtHubDensityGaussHermiteHST{T,R};
        # KEYWORD ARGUMENTS
        fermion_path_integral::FermionPathIntegral{H},
        fermion_greens_calculator::FermionGreensCalculator{H},
        B::Vector{P}, δG::R, δθ::R,  rng::AbstractRNG,
        δG_max::R = 1e-6, update_stabilization_frequency::Bool = true
    ) where {H<:Number, T<:Number, R<:Real, P<:AbstractPropagator}

Perform local updates to extended Hubbard interaction Gauss-Hermite Hubbard-Stratonovich fields.
This method returns a tuple containing `(acceptance_rate, logdetG, sgndetG, δG, δθ)`.

# Arguments

- `G::Matrix{H}`: Equal-time Green's function matrix.
- `logdetG::R`: The log of the absolute value of the determinant of the equal-time Green's function matrix, ``\log \vert \det G(\tau,\tau) \vert.``
- `sgndetG::H`: The sign/phase of the determinant of the equal-time Green's function matrix, ``\det G(\tau,\tau) / \vert \det G(\tau,\tau) \vert.``
- `hst_parameters::ExtHubDensityGaussHermiteHST{T,R}`: Type representing Hubbard-Stratonovich transformation.

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
    hst_parameters::ExtHubDensityGaussHermiteHST{T,R};
    # KEYWORD ARGUMENTS
    fermion_path_integral::FermionPathIntegral{H},
    fermion_greens_calculator::FermionGreensCalculator{H},
    B::Vector{P}, δG::R, δθ::R,  rng::AbstractRNG,
    δG_max::R = 1e-6, update_stabilization_frequency::Bool = true
) where {H<:Number, T<:Number, R<:Real, P<:AbstractPropagator}

    @assert !( (H<:Real) &&  (T<:Complex)) "Green's function matrices are real while Hubbard-Stratonovich transformation is complex."
    (; Δτ, α, neighbor_table, s, update_perm, N) = hst_parameters
    
    if size(fermion_path_integral.u,2) > 2
        u = @view fermion_path_integral.u[:,1:2]
        v = @view fermion_path_integral.v[:,1:2]
    else
        (; u, v) = fermion_path_integral
    end

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
        B_l = B[l]::P
        # apply the transformation G̃(τ,τ) = exp(+Δτ⋅K[l]/2)⋅G(τ,τ)⋅exp(-Δτ⋅K[l]/2)
        # if B[l] = exp(-Δτ⋅K[l]/2)⋅exp(-Δτ⋅V[l])⋅exp(-Δτ⋅K[l]/2),
        # otherwise nothing when B[l] = exp(-Δτ⋅V[l])⋅exp(-Δτ⋅K[l])
        partially_wrap_greens_reverse!(G, B_l, G′)

        # shuffle the order in which orbitals/sites will be iterated over
        shuffle!(rng, update_perm)

        # iterate over bonds in the lattice
        for n in update_perm

            # get the pair of sites coupled associated with extended hubbard interaction
            i = neighbor_table[1,n]
            j = neighbor_table[2,n]
            (i,j) = (i>j) ? (j,i) : (i,j)

            # propose a HS field update and calculate the change in the potential
            # energy matrix and bosonic action
            s_nl    = s[n,l]
            η_nl    = eval_η(s_nl)
            Sb_nl   = eval_Sgh(s_nl) - 2*Δτ*α[n]*η_nl
            s_nl′   = sample_new_ghhsf(rng, s_nl)
            η_nl′   = eval_η(s_nl′)
            Sb_nl′  = eval_Sgh(s_nl′) - 2*Δτ*α[n]*η_nl′
            ΔV_nl   = (+α[n] * η_nl′) - (+α[n] * η_nl)

            # Note that Δ_il = exp(-Δτ[V_il(s′) - V_il(s)]) and Δ_jl = exp(-Δτ[V_jl(s′) - V_jl(s)])
            # are equal because we are coupling to a density channel, and there we say that
            # Δ_nl = Δ_il = Δ_jl where n labels the bond connecting sites i and j
            Δ_nl = exp(-Δτ*ΔV_nl) - 1

            # calculate the determinant ratio for each spin species
            R_nl = eval_R(G, Δ_nl, i, j)

            # calculate the change in bosonic action
            ΔSb = Sb_nl′ - Sb_nl

            # calculate acceptance probability
            P_nl = abs(exp(-ΔSb) * R_nl^2)

            # accept or reject proposed update
            if rand(rng) < P_nl

                # increment the cound of accepted spin flips
                accepted_spin_flips += 1

                # update HS field
                s[n,l] = s_nl′

                # update diagonal on-site energy matrix
                V[i,l] += ΔV_nl
                V[j,l] += ΔV_nl

                # udpate bosonic action
                fermion_path_integral.Sb += ΔSb

                # update spin-up green's function matrix
                logdetG, sgndetG = update_G!(G, logdetG, sgndetG, B_l, Δ_nl, R_nl, i, j, u, v)
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
        hst_parameters::ExtHubDensityGaussHermiteHST{T,R};
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

Perform a reflection update in which the sign of every density-channel Gauss-Hermite Hubbard-Stratonovich field used to decouple the
extended Hubbard interaction on a randomly chosen bond in the lattice is changed.
This function returns `(accepted, logdetG, sgndetG)`.

# Arguments

- `Gup::Matrix{H}`: Spin-up eqaul-time Greens function matrix.
- `logdetGup::R`: Log of the determinant of the spin-up eqaul-time Greens function matrix.
- `sgndetGup::H`: Sign/phase of the determinant of the spin-up eqaul-time Greens function matrix.
- `Gdn::Matrix{H}`: Spin-down eqaul-time Greens function matrix.
- `logdetGdn::R`: Log of the determinant of the spin-down eqaul-time Greens function matrix.
- `sgndetGdn::H`: Sign/phase of the determinant of the spin-down eqaul-time Greens function matrix.
- `hst_parameters::ExtHubDensityGaussHermiteHST{T,R}`: Hubbard-Stratonovich fields and associated parameters to update.

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
    hst_parameters::ExtHubDensityGaussHermiteHST{T,R};
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

    (; Δτ, α, neighbor_table, s, N) = hst_parameters
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

    # pick a random bond in lattice with extend Hubbard U to perform reflection update on
    n     = rand(rng, 1:N)
    i     = neighbor_table[1,n]
    j     = neighbor_table[2,n]
    s_n   = @view s[n, :]
    Vup_i = @view fermion_path_integral_up.V[i, :]
    Vdn_i = @view fermion_path_integral_dn.V[i, :]
    Vup_j = @view fermion_path_integral_up.V[j, :]
    Vdn_j = @view fermion_path_integral_dn.V[j, :]

    # calculate initial bosonic action
    Sb = -2 * Δτ * α[n] * sum(eval_η, s_n)

    # reflect all the HS field on site i
    s_n′ = s_n
    @. s_n′ = -s_n

    # calculate final bosonic action
    Sb′ = -2 * Δτ * α[n] * sum(eval_η, s_n)

    # calculate change in bosonic action
    ΔSb = Sb′ - Sb

    # update diagonal on-site energy matrix:
    @. Vup_i += (+α[n]*eval_η(s_n′)) - (+α[n]*eval_η(-s_n′))
    @. Vdn_i += (+α[n]*eval_η(s_n′)) - (+α[n]*eval_η(-s_n′))
    @. Vup_j += (+α[n]*eval_η(s_n′)) - (+α[n]*eval_η(-s_n′))
    @. Vdn_j += (+α[n]*eval_η(s_n′)) - (+α[n]*eval_η(-s_n′))

    # update propagator matrices
    @inbounds for l in eachindex(Bup)
        expmΔτVup_l = Bup[l].expmΔτV
        expmΔτVdn_l = Bdn[l].expmΔτV
        expmΔτVup_l[i] = exp(-Δτ*Vup_i[l])
        expmΔτVdn_l[i] = exp(-Δτ*Vdn_i[l])
        expmΔτVup_l[j] = exp(-Δτ*Vup_j[l])
        expmΔτVdn_l[j] = exp(-Δτ*Vdn_j[l])
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
        @. s_n = -s_n′
        # revert diagonal on-site energy matrix
        @. Vup_i += (+α[n]*eval_η(s_n)) - (+α[n]*eval_η(-s_n))
        @. Vdn_i += (+α[n]*eval_η(s_n)) - (+α[n]*eval_η(-s_n))
        @. Vup_j += (+α[n]*eval_η(s_n)) - (+α[n]*eval_η(-s_n))
        @. Vdn_j += (+α[n]*eval_η(s_n)) - (+α[n]*eval_η(-s_n))
        # revert propagator matrices
        @inbounds for l in eachindex(Bup)
            expmΔτVup_l = Bup[l].expmΔτV
            expmΔτVdn_l = Bdn[l].expmΔτV
            expmΔτVup_l[i] = exp(-Δτ*Vup_i[l])
            expmΔτVdn_l[i] = exp(-Δτ*Vdn_i[l])
            expmΔτVup_l[j] = exp(-Δτ*Vup_j[l])
            expmΔτVdn_l[j] = exp(-Δτ*Vdn_j[l])
        end
        accepted = false
    end

    return (accepted, logdetGup, sgndetGup, logdetGdn, sgndetGdn)
end

@doc raw"""
    reflection_update!(
        # ARGUMENTS
        G::Matrix{H}, logdetG::R, sgndetG::H,
        hst_parameters::ExtHubDensityGaussHermiteHST{T,R};
        # KEYWORD ARGUMENTS
        fermion_path_integral::FermionPathIntegral{H},
        fermion_greens_calculator::FermionGreensCalculator{H,R},
        fermion_greens_calculator_alt::FermionGreensCalculator{H,R},
        B::Vector{P}, rng::AbstractRNG
    ) where {H<:Number, T<:Number, R<:Real, P<:AbstractPropagator}

Perform a reflection update in which the sign of every density-channel Gauss-Hermite Hubbard-Stratonovich field used to decouple the
extended Hubbard interaction on a randomly chosen bond in the lattice is changed.
This function returns `(accepted, logdetG, sgndetG)`.

# Arguments

- `G::Matrix{H}`: Equal-time Greens function matrix.
- `logdetG::R`: Log of the determinant of the spin-up equal-time Greens function matrix.
- `sgndetG::H`: Sign/phase of the determinant of the spin-up equal-time Greens function matrix.
- `hst_parameters::ExtHubDensityGaussHermiteHST{T,R}`: Hubbard-Stratonovich fields and associated parameters to update.

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
    hst_parameters::ExtHubDensityGaussHermiteHST{T,R};
    # KEYWORD ARGUMENTS
    fermion_path_integral::FermionPathIntegral{H},
    fermion_greens_calculator::FermionGreensCalculator{H,R},
    fermion_greens_calculator_alt::FermionGreensCalculator{H,R},
    B::Vector{P}, rng::AbstractRNG
) where {H<:Number, T<:Number, R<:Real, P<:AbstractPropagator}

    (; Δτ, α, neighbor_table, s, N) = hst_parameters
    G′ = fermion_greens_calculator_alt.G′

    # make sure stabilization frequencies match
    if fermion_greens_calculator.n_stab != fermion_greens_calculator_alt.n_stab
        resize!(fermion_greens_calculator_alt, fermion_greens_calculator.n_stab)
    end

    # pick a random bond in lattice with extend Hubbard U to perform reflection update on
    n     = rand(rng, 1:N)
    i     = neighbor_table[1,n]
    j     = neighbor_table[2,n]
    s_n   = @view s[n, :]
    V_i = @view fermion_path_integral.V[i, :]
    V_j = @view fermion_path_integral.V[j, :]

    # calculate initial bosonic action
    Sb = -2 * Δτ * α[n] * sum(eval_η, s_n)

    # reflect all the HS field on site i
    s_n′ = s_n
    @. s_n′ = -s_n

    # calculate final bosonic action
    Sb′ = -2 * Δτ * α[n] * sum(eval_η, s_n)

    # calculate change in bosonic action
    ΔSb = Sb′ - Sb

    # update diagonal on-site energy matrix:
    @. V_i += (+α[n]*eval_η(s_n′)) - (+α[n]*eval_η(-s_n′))
    @. V_j += (+α[n]*eval_η(s_n′)) - (+α[n]*eval_η(-s_n′))

    # update propagator matrices
    @inbounds for l in eachindex(B)
        expmΔτV_l = B[l].expmΔτV
        expmΔτV_l[i] = exp(-Δτ*V_i[l])
        expmΔτV_l[j] = exp(-Δτ*V_j[l])
    end

    # calculate new Green's function matrices and determinant of new Green's function matrix
    logdetG′, sgndetG′ = calculate_equaltime_greens!(G′, fermion_greens_calculator_alt, B)

    # calculate acceptance probability
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
        @. s_n = -s_n′
        # revert diagonal on-site energy matrix
        @. V_i += (+α[n]*eval_η(s_n)) - (+α[n]*eval_η(-s_n))
        @. V_j += (+α[n]*eval_η(s_n)) - (+α[n]*eval_η(-s_n))
        # revert propagator matrices
        @inbounds for l in eachindex(B)
            expmΔτV_l = B[l].expmΔτV
            expmΔτV_l[i] = exp(-Δτ*V_i[l])
            expmΔτV_l[j] = exp(-Δτ*V_j[l])
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
        hst_parameters::ExtHubDensityGaussHermiteHST{T,R};
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

Perform a swap update in which the density-channel Gauss-Hermite Hubbard-Stratonovich fields used to decouple the extebded Hubbard interaction
on a randomly sampled pair of bonds in the lattice are swapped.
This function returns `(accepted, logdetGup, sgndetGup, logdetGdn, sgndetGdn)`.

# Arguments

- `Gup::Matrix{H}`: Spin-up eqaul-time Greens function matrix.
- `logdetGup::R`: Log of the determinant of the spin-up eqaul-time Greens function matrix.
- `sgndetGup::H`: Sign/phase of the determinant of the spin-up eqaul-time Greens function matrix.
- `Gdn::Matrix{H}`: Spin-down eqaul-time Greens function matrix.
- `logdetGdn::R`: Log of the determinant of the spin-down eqaul-time Greens function matrix.
- `sgndetGdn::H`: Sign/phase of the determinant of the spin-down eqaul-time Greens function matrix.
- `hst_parameters::ExtHubDensityGaussHermiteHST{T,R}`: Hubbard-Stratonovich fields and associated parameters to update.

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
    hst_parameters::ExtHubDensityGaussHermiteHST{T,R};
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

    (; Δτ, α, neighbor_table, s, N) = hst_parameters
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
    n, m = draw2(rng, N)

    # get the site index associted with each Hubbard U
    i_n = neighbor_table[1,n]
    j_n = neighbor_table[2,n]
    i_m = neighbor_table[1,m]
    j_m = neighbor_table[2,m]

    # get the HS fields associated with each site
    s_n = @view s[n,:]
    s_m = @view s[m,:]

    Vup_i_n = @view fermion_path_integral_up.V[i_n, :]
    Vup_j_n = @view fermion_path_integral_up.V[j_n, :]
    Vdn_i_n = @view fermion_path_integral_dn.V[i_n, :]
    Vdn_j_n = @view fermion_path_integral_dn.V[j_n, :]

    Vup_i_m = @view fermion_path_integral_up.V[i_m, :]
    Vup_j_m = @view fermion_path_integral_up.V[j_m, :]
    Vdn_i_m = @view fermion_path_integral_dn.V[i_m, :]
    Vdn_j_m = @view fermion_path_integral_dn.V[j_m, :]

    # calculate initial bosonic action associated with pair of sites
    Sb  = -2*Δτ * α[n] * sum(eval_η, s_n) - 2*Δτ * α[m] * sum(eval_η, s_m)

    # swap the HS fields
    swap!(s_n, s_m)

    # calculate initial bosonic action associated with pair of sites
    Sb′ = -2*Δτ * α[n] * sum(eval_η, s_n) - 2*Δτ * α[m] * sum(eval_η, s_m)

    # calculate the change in the bosonic action
    ΔSb = Sb′ - Sb

    # update potential energy matrices
    @. Vup_i_n += (+α[n]*eval_η(s_n)) - (+α[n]*eval_η(s_m))
    @. Vup_j_n += (+α[n]*eval_η(s_n)) - (+α[n]*eval_η(s_m))
    @. Vdn_i_n += (+α[n]*eval_η(s_n)) - (+α[n]*eval_η(s_m))
    @. Vdn_j_n += (+α[n]*eval_η(s_n)) - (+α[n]*eval_η(s_m))
    @. Vup_i_m += (+α[m]*eval_η(s_m)) - (+α[m]*eval_η(s_n))
    @. Vup_j_m += (+α[m]*eval_η(s_m)) - (+α[m]*eval_η(s_n))
    @. Vdn_i_m += (+α[m]*eval_η(s_m)) - (+α[m]*eval_η(s_n))
    @. Vdn_j_m += (+α[m]*eval_η(s_m)) - (+α[m]*eval_η(s_n))


    # update propagator matrices
    @inbounds for l in eachindex(Bup)
        expmΔτVup_l = Bup[l].expmΔτV
        expmΔτVdn_l = Bdn[l].expmΔτV
        expmΔτVup_l[i_n] = exp(-Δτ*Vup_i_n[l])
        expmΔτVdn_l[i_n] = exp(-Δτ*Vdn_i_n[l])
        expmΔτVup_l[j_n] = exp(-Δτ*Vup_j_n[l])
        expmΔτVdn_l[j_n] = exp(-Δτ*Vdn_j_n[l])
        expmΔτVup_l[i_m] = exp(-Δτ*Vup_i_m[l])
        expmΔτVdn_l[i_m] = exp(-Δτ*Vdn_i_m[l])
        expmΔτVup_l[j_m] = exp(-Δτ*Vup_j_m[l])
        expmΔτVdn_l[j_m] = exp(-Δτ*Vdn_j_m[l])
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
        swap!(s_n, s_m)
        # revert diagonal on-site energy matrix
        @. Vup_i_n += (+α[n]*eval_η(s_n)) - (+α[n]*eval_η(s_m))
        @. Vup_j_n += (+α[n]*eval_η(s_n)) - (+α[n]*eval_η(s_m))
        @. Vdn_i_n += (+α[n]*eval_η(s_n)) - (+α[n]*eval_η(s_m))
        @. Vdn_j_n += (+α[n]*eval_η(s_n)) - (+α[n]*eval_η(s_m))
        @. Vup_i_m += (+α[m]*eval_η(s_m)) - (+α[m]*eval_η(s_n))
        @. Vup_j_m += (+α[m]*eval_η(s_m)) - (+α[m]*eval_η(s_n))
        @. Vdn_i_m += (+α[m]*eval_η(s_m)) - (+α[m]*eval_η(s_n))
        @. Vdn_j_m += (+α[m]*eval_η(s_m)) - (+α[m]*eval_η(s_n))
        # revert propagator matrices
        @inbounds for l in eachindex(Bup)
            expmΔτVup_l = Bup[l].expmΔτV
            expmΔτVdn_l = Bdn[l].expmΔτV
            expmΔτVup_l[i_n] = exp(-Δτ*Vup_i_n[l])
            expmΔτVdn_l[i_n] = exp(-Δτ*Vdn_i_n[l])
            expmΔτVup_l[j_n] = exp(-Δτ*Vup_j_n[l])
            expmΔτVdn_l[j_n] = exp(-Δτ*Vdn_j_n[l])
            expmΔτVup_l[i_m] = exp(-Δτ*Vup_i_m[l])
            expmΔτVdn_l[i_m] = exp(-Δτ*Vdn_i_m[l])
            expmΔτVup_l[j_m] = exp(-Δτ*Vup_j_m[l])
            expmΔτVdn_l[j_m] = exp(-Δτ*Vdn_j_m[l])
        end
        accepted = false
    end

    return (accepted, logdetGup, sgndetGup, logdetGdn, sgndetGdn)
end

@doc raw"""
    swap_update!(
        # ARGUMENTS
        G::Matrix{H}, logdetG::R, sgndetG::H,
        hst_parameters::ExtHubDensityGaussHermiteHST{T,R};
        # KEYWORD ARGUMENTS
        fermion_path_integral::FermionPathIntegral{H},
        fermion_greens_calculator::FermionGreensCalculator{H,R},
        fermion_greens_calculator_alt::FermionGreensCalculator{H,R},
        B::Vector{P},
        rng::AbstractRNG
    ) where {H<:Number, T<:Number, R<:Real, P<:AbstractPropagator}

Perform a swap update in which the density-channel Gauss-Hermite Hubbard-Stratonovich fields used to decouple the extebded Hubbard interaction
on a randomly sampled pair of bonds in the lattice are swapped.
This function returns `(accepted, logdetG, sgndetG)`.

# Arguments

- `G::Matrix{H}`: Equal-time Greens function matrix.
- `logdetG::R`: Log of the determinant of the equal-time Greens function matrix.
- `sgndetG::H`: Sign/phase of the determinant of the equal-time Greens function matrix.
- `hst_parameters::ExtHubDensityGaussHermiteHST{T,R}`: Hubbard-Stratonovich fields and associated parameters to update.

# Keyword Arguments

- `fermion_path_integral::FermionPathIntegral{H}`: An instance of [`FermionPathIntegral`](@ref) type for the electrons.
- `fermion_greens_calculator::FermionGreensCalculator{H,R}`: Contains matrix factorization information for current state.
- `fermion_greens_calculator_alt::FermionGreensCalculator{H,R}`: Used to calculate matrix factorizations for proposed state.
- `B::Vector{P}`: Propagators for each imaginary time slice.
- `rng::AbstractRNG`: Random number generator used in method instead of global random number generator, important for reproducibility.
"""
function swap_update!(
    # ARGUMENTS
    G::Matrix{H}, logdetG::R, sgndetG::H,
    hst_parameters::ExtHubDensityGaussHermiteHST{T,R};
    # KEYWORD ARGUMENTS
    fermion_path_integral::FermionPathIntegral{H},
    fermion_greens_calculator::FermionGreensCalculator{H,R},
    fermion_greens_calculator_alt::FermionGreensCalculator{H,R},
    B::Vector{P},
    rng::AbstractRNG
) where {H<:Number, T<:Number, R<:Real, P<:AbstractPropagator}

    (; Δτ, α, neighbor_table, s, N) = hst_parameters
    G′ = fermion_greens_calculator_alt.G′

    # make sure stabilization frequencies match
    if fermion_greens_calculator.n_stab != fermion_greens_calculator_alt.n_stab
        resize!(fermion_greens_calculator_alt, fermion_greens_calculator.n_stab)
    end

    # ranomly pick two sites with Hubbard U interaction on them
    n, m = draw2(rng, N)

    # get the site index associted with each Hubbard U
    i_n = neighbor_table[1,n]
    j_n = neighbor_table[2,n]
    i_m = neighbor_table[1,m]
    j_m = neighbor_table[2,m]

    # get the HS fields associated with each site
    s_n = @view s[n,:]
    s_m = @view s[m,:]

    V_i_n = @view fermion_path_integral.V[i_n, :]
    V_j_n = @view fermion_path_integral.V[j_n, :]
    V_i_m = @view fermion_path_integral.V[i_m, :]
    V_j_m = @view fermion_path_integral.V[j_m, :]

    # calculate initial bosonic action associated with pair of sites
    Sb  = -2*Δτ * α[n] * sum(eval_η, s_n) - 2*Δτ * α[m] * sum(eval_η, s_m)

    # swap the HS fields
    swap!(s_n, s_m)

    # calculate initial bosonic action associated with pair of sites
    Sb′ = -2*Δτ * α[n] * sum(eval_η, s_n) - 2*Δτ * α[m] * sum(eval_η, s_m)

    # calculate the change in the bosonic action
    ΔSb = Sb′ - Sb

    # update potential energy matrices
    @. V_i_n += (+α[n]*eval_η(s_n)) - (+α[n]*eval_η(s_m))
    @. V_j_n += (+α[n]*eval_η(s_n)) - (+α[n]*eval_η(s_m))
    @. V_i_m += (+α[m]*eval_η(s_m)) - (+α[m]*eval_η(s_n))
    @. V_j_m += (+α[m]*eval_η(s_m)) - (+α[m]*eval_η(s_n))


    # update propagator matrices
    @inbounds for l in eachindex(B)
        expmΔτV_l = B[l].expmΔτV
        expmΔτV_l[i_n] = exp(-Δτ*V_i_n[l])
        expmΔτV_l[j_n] = exp(-Δτ*V_j_n[l])
        expmΔτV_l[i_m] = exp(-Δτ*V_i_m[l])
        expmΔτV_l[j_m] = exp(-Δτ*V_j_m[l])
    end

    # calculate new Green's function matrices and determinant of new Green's function matrix
    logdetG′, sgndetG′ = calculate_equaltime_greens!(G′, fermion_greens_calculator_alt, B)

    # calculate acceptance probability
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
        swap!(s_n, s_m)
        # revert diagonal on-site energy matrix
        @. V_i_n += (+α[n]*eval_η(s_n)) - (+α[n]*eval_η(s_m))
        @. V_j_n += (+α[n]*eval_η(s_n)) - (+α[n]*eval_η(s_m))
        @. V_i_m += (+α[m]*eval_η(s_m)) - (+α[m]*eval_η(s_n))
        @. V_j_m += (+α[m]*eval_η(s_m)) - (+α[m]*eval_η(s_n))
        # revert propagator matrices
        @inbounds for l in eachindex(B)
            expmΔτV_l = B[l].expmΔτV
            expmΔτV_l[i_n] = exp(-Δτ*V_i_n[l])
            expmΔτV_l[j_n] = exp(-Δτ*V_j_n[l])
            expmΔτV_l[i_m] = exp(-Δτ*V_i_m[l])
            expmΔτV_l[j_m] = exp(-Δτ*V_j_m[l])
        end
        accepted = false
    end

    return (accepted, logdetG, sgndetG)
end