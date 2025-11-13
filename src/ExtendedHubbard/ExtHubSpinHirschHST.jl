struct ExtHubSpinHirschHST{T,R} <: AbstractAsymHST{T,R}

    # inverse temperature
    β::R

    # discretization in imaginary time
    Δτ::R

    # length of imaginary time axis
    Lτ::Int

    # number of bonds with finite Hubbard U
    N::Int

    # each finite extended hubbard interaction
    V::Vector{R}
    
    # HST coupling coefficient
    α::Vector{T}

    # site index associated with each Hubbard U
    neighbor_table::Matrix{Int}

    # Hubbard-Stratonovich fields
    s_upup::Array{Int, 2}
    s_dndn::Array{Int, 2}
    s_updn::Array{Int, 2}
    s_dnup::Array{Int, 2}

    # order in which to iterate over orbitals when updating Hubbard-Stratonovich fields.
    update_perm::Vector{Int}

    # record the bond ID types associated with extended hubbard interactions
    bond_ids::Vector{Int}
end

function ExtHubSpinHirschHST(;
    # KEYWORD ARGUMENTS
    extended_hubbard_parameters::ExtendedHubbardParameters{R},
    β::R, Δτ::R, rng::AbstractRNG
) where {R<:AbstractFloat}

    (; V, neighbor_table, bond_ids, ph_sym_form) = extended_hubbard_parameters

    # if any attractive Hubbard interactions, then complex field coefficients
    T = any(v -> v < 0, V) ? Complex{E} : E

    # calculate length of imaginary-time axis
    Lτ = round(Int, β / Δτ)

    # calculate HS transformation coefficients
    α = zeros(T, length(V))
    @. α = acosh(exp(Δτ*T(V)/2))/Δτ

    # number of sites with Hubbard interaction
    N = length(V)

    # initialize HS fields
    s_upup = rand(rng, -1:2:1, (N, Lτ))
    s_dndn = rand(rng, -1:2:1, (N, Lτ))
    s_updn = rand(rng, -1:2:1, (N, Lτ))
    s_dnup = rand(rng, -1:2:1, (N, Lτ))

    # initialize update permutation order
    update_perm = collect(1:N)

    return ExtHubSpinHirschHST{T,R}(β, Δτ, Lτ, N, V, α, neighbor_table, s_upup, s_dndn, s_updn, s_dnup, update_perm, bond_ids)
end


# initialize fermion path integral to reflect HS field config
function _initialize!(
    fermion_path_integral_up::FermionPathIntegral{H},
    fermion_path_integral_dn::FermionPathIntegral{H},
    hst_parameters::ExtHubSpinHirschHST{T}
) where {H<:Number, T<:Number}

    @assert !((H<:Real) &&  (T<:Complex)) "Green's function matrices are real while ExtHubSpinHirschHST is complex."
    @assert fermion_path_integral_up.Sb == fermion_path_integral_dn.Sb "$(fermion_path_integral_up.Sb) ≠ $(fermion_path_integral_dn.Sb)"

    (; neighbor_table, α, s_upup, s_dndn, s_updn, s_dnup) = hst_parameters
    Vup = fermion_path_integral_up.V
    Vdn = fermion_path_integral_dn.V

    # iterate over sites with Hubbard U interactions
    for b in axes(neighbor_table, 2)
        i, j = neighbor_table[1,b], neighbor_table[2, b]
        @views @. Vup[i,:] += -α[i] * s_upup[j,:]
        @views @. Vdn[i,:] += -α[i] * s_dndn[j,:]
        @views @. Vup[i,:] += -α[i] * s_updn[j,:]
        @views @. Vdn[i,:] += -α[i] * s_dnup[j,:]
        @views @. Vup[j,:] += +α[j] * s_upup[i,:]
        @views @. Vdn[j,:] += +α[j] * s_dndn[i,:]
        @views @. Vup[j,:] += +α[j] * s_updn[i,:]
        @views @. Vdn[j,:] += +α[j] * s_dnup[i,:]
    end

    return nothing
end


# perform local updates for specified imaginary-time slice
function _local_updates!(
    Gup::Matrix{H}, logdetGup::R, sgndetGup::H,
    Gdn::Matrix{H}, logdetGdn::R, sgndetGdn::H,
    hst_parameters::ExtHubSpinHirschHST{T,R},
    fermion_path_integral_up::FermionPathIntegral{H},
    fermion_path_integral_dn::FermionPathIntegral{H},
    Bup::P, Bdn::P, l::Int, rng::AbstractRNG
) where {H<:Number, T<:Number, R<:Real, P<:AbstractPropagator}

    (; Δτ, α, neighbor_table, update_perm, N) = hst_parameters
    u′ = @view fermion_path_integral_up.u[:,1]
    v′ = @view fermion_path_integral_up.v[:,1]
    u″ = @view fermion_path_integral_dn.u[:,1:2]
    v″ = @view fermion_path_integral_dn.v[:,1:2]

    # get on-site energy matrices for spin up and down electrons for all time slices
    Vup = fermion_path_integral_up.V
    Vdn = fermion_path_integral_dn.V

    # get the relevant HS fields
    s_upup = @view hst_parameters.s_upup[:,l]
    s_dndn = @view hst_parameters.s_upup[:,l]
    s_updn = @view hst_parameters.s_upup[:,l]
    s_dnup = @view hst_parameters.s_upup[:,l]

    # shuffle the order in which orbitals/sites will be iterated over
    shuffle!(rng, update_perm)

    # iterate over orbitals in the lattice
    for n in update_perm

        # get the pair of sites coupled associated with extended hubbard interaction
        i, j = neighbor_table[1,n], neighbor_table[2,n]

        # perform local updates for each of the four types of HS fields
        _local_update!(
            Gup, logdetGup, sgndetGup, Bup[l], Vup[l], s_upup,
            n, i, j, Δτ, α, rng, u″, v″
        )
        _local_update!(
            Gdn, logdetGdn, sgndetGdn, Bdn[l], Vdn[l], s_dndn,
            n, i, j, Δτ, α, rng, u″, v″
        )
        _local_update!(
            Gup, logdetGup, sgndetGup, Bup[l], Vup[l],
            Gdn, logdetGdn, sgndetGdn, Bdn[l], Vdn[l],
            s_updn, n, i, j, Δτ, α, rng, u′, v′
        )
        _local_update!(
            Gdn, logdetGdn, sgndetGdn, Bdn[l], Vdn[l],
            Gup, logdetGup, sgndetGup, Bup[l], Vup[l],
            s_dnup, n, i, j, Δτ, α, rng, u′, v′
        )
    end

    # calculate the acceptance rate
    acceptance_rate = accepted_spin_flips / length(s)

    return acceptance_rate
end

# perform a local update for a HS field that couples the densities of different spin species
function _local_update!(
    Gi, logdetGi, sgndetGi, Bi, Vi,
    Gj, logdetGj, sgndetGj, Bj, Vj,
    s, n, i, j, Δτ, α, rng, u, v
)

    # calculate the change in the i & j matrix elements of the diagonal on-site energy matrices
    ΔV = -2 * α[n] * s[n]

    # calculate determinant ratio associated with spin flip
    Ri, Δi = local_update_det_ratio(Gi, -ΔV, i, Δτ)
    Rj, Δj = local_update_det_ratio(Gj, +ΔV, j, Δτ)

    # calculate the acceptance probability
    Pij = abs(Ri * Rj)

    # accept
    if rand(rng) < Pij

        # flip the spin
        s[n] = -s[n]

        # update the diagonal potential energy matrices
        Vi[i] += -ΔV
        Vj[j] += +ΔV

        # update the corresponding Green's function matrices
        logdetGi, sgndetGi = local_update_greens!(Gi, logdetGi, sgndetGi, Bi, Ri, Δi, i, u, v)
        logdetGj, sgndetGj = local_update_greens!(Gj, logdetGj, sgndetGj, Bj, Rj, Δj, j, u, v)

        # record the update as being accepted
        accepted = true

    # reject the update
    else

        # record the update as being rejected
        accepted = false
    end

    return (accepted, Gi, logdetGi, sgndetGi, Gj, logdetGj, sgndetGj)
end

# perform a local update for a HS field that couples the densities of the same spin species
function _local_update!(
    G, logdetG, sgndetG, B, V,
    s, n, i, j, Δτ, α, rng, u, v
)

    # calculate the change in the potential energy matrix
    ΔV = -2 * α[n] * s[n]

    # calculate changes in matrix elements
    Δii = exp(+Δτ*ΔV)
    Δjj = exp(-Δτ*ΔV)

    # calculate determinant ratio
    Rij = eval_R(G, Δii, Δjj, i, j)

    # calculate acceptance probability
    Pij = abs(Rij)

    # accept update
    if rand(rng) < Pij

        # record accepted update
        accepted = true

        # flip the spin
        s[n] = -s[n]

        # update diagonal potential energy matrix
        V[i] += -ΔV
        V[j] += +ΔV

        # update green's function matrix
        logdetG, sgndetG = update_G!(G, logdetG, sgndetG, B, Rij, Δii, Δjj, i, j, u, v)

    # reject update
    else

        # record rejected update
        accepted = false
    end

    return (accepted, G, logdetG, sgndetG)
end