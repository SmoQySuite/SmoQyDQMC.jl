@doc raw"""
    update_stabilization_frequency!(
        # ARGUMENTS
        Gup::Matrix{H}, logdetGup::R, sgndetGup::H,
        Gdn::Matrix{H}, logdetGdn::R, sgndetGdn::H;
        # KEYWORD ARGUMENTS
        fermion_greens_calculator_up::FermionGreensCalculator{H,R},
        fermion_greens_calculator_dn::FermionGreensCalculator{H,R},
        Bup::Vector{P}, Bdn::Vector{P},
        δG::R, δθ::R,
        δG_max::R,
        δG_min::R = 0.0,
        active::Bool = true
    ) where {H<:Number, R<:Real, P<:AbstractPropagator}

If the corrected error in the Green's function matrix is too large, `δG > δG_max`, then increase the frequency of
numerical stabilization by decrementing `n_stab` such that it is updated to `n_stab = max(n_stab - 1, 1)`,
and update the equal-time Green's function matrices and all related variables and types.
If `δG < δG_min`, then instead `n_stab` is increased by one instead, `n_stab = n_stab + 1`.
If the frequency of stabilization is updated, then `δG` and `δθ` are reset to zero.
This function also throws a `@warn` if a numerical instability is encountered.
This method returns a tuple of the following variables:
```julia
(logdetGup, sgndetGdn, logdetGup, sgndetGdn, δG, δθ),
```
where `updated = true` if `n_stab` was decremented.
If `active = false`, then this function simply warns when a numerical instability is encountered,
and does not actual update/decrement `n_stab`.
"""
function update_stabilization_frequency!(
    # ARGUMENTS
    Gup::Matrix{H}, logdetGup::R, sgndetGup::H,
    Gdn::Matrix{H}, logdetGdn::R, sgndetGdn::H;
    # KEYWORD ARGUMENTS
    fermion_greens_calculator_up::FermionGreensCalculator{H,R},
    fermion_greens_calculator_dn::FermionGreensCalculator{H,R},
    Bup::Vector{P}, Bdn::Vector{P},
    δG::R, δθ::R,
    δG_max::R,
    δG_min::R = 0.0,
    active::Bool = true
) where {H<:Number, R<:Real, P<:AbstractPropagator}

    # make sure all fermion greens calculators are using the same stabilization period
    n_stab_up = fermion_greens_calculator_up.n_stab
    n_stab_dn = fermion_greens_calculator_dn.n_stab
    @assert n_stab_up == n_stab_dn
    n_stab = n_stab_up

    # initialize updated to false
    updated = false

    # if numerical instability occurred
    if δG > δG_max || (notfinite(δG)) || (notfinite(logdetGup)) || (notfinite(logdetGdn))
        
        # if n_stab can still be reduced
        if active && (fermion_greens_calculator_up.n_stab > 1)

            # warn of numerical instability
            @warn "Numerical instability encountered and stabilization frequency will be decremented." δG_max δG logdetGup logdetGdn n_stab n_stab-1

            # set updated to true
            updated = true

            # decrease the stabilization period by one
            n_stab = n_stab - 1

            # resize spin up fermion greens calculator accordingly
            logdetGup, sgndetGup = resize!(fermion_greens_calculator_up, Gup, logdetGup, sgndetGup, Bup, n_stab)

            # resize spin down fermion greens calculator accordingly
            logdetGdn, sgndetGdn = resize!(fermion_greens_calculator_dn, Gdn, logdetGdn, sgndetGdn, Bdn, n_stab)

            # if failed to evaluate determinant correctly
            if notfinite(logdetGup) || notfinite(logdetGdn)

                # throw error
                error("Error updating stabilization frequency to `n_stab = $(n_stab)`: `logdetGup = $(logdetGup)` and `logdetGdn = $(logdetGdn)`.")
            end

            # initialize errors associated with numerical instability to zero
            δG = zero(R)
            δθ = zero(R)

        else

            # warn of numerical instability
            @warn "Numerical instability encountered." δG_max δG logdetGup logdetGdn n_stab
        end

    # if very numerically stable
    elseif active && (δG < δG_min)

        # notify that stabilization period being increased
        @info "Stabilization frequency is being increased by one." δG_max δG logdetGup logdetGdn n_stab n_stab+1

        # increase the stabilization period by one
        n_stab = n_stab + 1

        # resize spin up fermion greens calculator accordingly
        logdetGup, sgndetGup = resize!(fermion_greens_calculator_up, Gup, logdetGup, sgndetGup, Bup, n_stab)

        # resize spin down fermion greens calculator accordingly
        logdetGdn, sgndetGdn = resize!(fermion_greens_calculator_dn, Gdn, logdetGdn, sgndetGdn, Bdn, n_stab)

        # if failed to evaluate determinant correctly
        if notfinite(logdetGup) || notfinite(logdetGdn)

            # throw error
            error("Error updating stabilization frequency to `n_stab = $(n_stab)`: `logdetGup = $(logdetGup)` and `logdetGdn = $(logdetGdn)`.")
        end

        # initialize errors associated with numerical instability to zero
        δG = zero(R)
        δθ = zero(R)
    end

    return (updated, logdetGup, sgndetGup, logdetGdn, sgndetGdn, δG, δθ)
end

@doc raw"""
    update_stabilization_frequency!(
        # ARGUMENTS
        G::Matrix{H}, logdetG::R, sgndetG::H;
        # KEYWORD ARGUMENTS
        fermion_greens_calculator::FermionGreensCalculator{H,R},
        B::Vector{P},
        δG::R, δθ::R,
        δG_max::R,
        δG_min::R = 0.0,
        active::Bool = true
    ) where {H<:Number, R<:Real, P<:AbstractPropagator}

If the corrected error in the Green's function matrix is too large, `δG > δG_max`, then increase the frequency of
numerical stabilization by decrementing `n_stab` such that it is updated to `n_stab = max(n_stab - 1, 1)`,
and update the equal-time Green's function matrices and all related variables and types.
If `δG < δG_min`, then instead `n_stab` is increased by one instead, `n_stab = n_stab + 1`.
If the frequency of stabilization is updated, then `δG` and `δθ` are reset to zero.
This function also throws a `@warn` if a numerical instability is encountered.
This method returns a tuple of the following variables:
```julia
(updated, logdetG, sgndetG, δG, δθ),
```
where `updated = true` if `n_stab` was decremented.
If `active = false`, then this function simply warns when a numerical instability is encountered,
and does not actual update/decrement `n_stab`.
"""
function update_stabilization_frequency!(
    # ARGUMENTS
    G::Matrix{H}, logdetG::R, sgndetG::H;
    # KEYWORD ARGUMENTS
    fermion_greens_calculator::FermionGreensCalculator{H,R},
    B::Vector{P},
    δG::R, δθ::R,
    δG_max::R,
    δG_min::R = 0.0,
    active::Bool = true
) where {H<:Number, R<:Real, P<:AbstractPropagator}

    # make sure all fermion greens calculators are using the same stabilization period
    n_stab = fermion_greens_calculator.n_stab

    # initialize updated to false
    updated = false

    # if numerical instability occurred
    if δG > δG_max || notfinite(δG) || notfinite(logdetG)

        # if n_stab can still be reduced
        if active && (fermion_greens_calculator.n_stab > 1)

            # warn of numerical instability
            @warn "Numerical instability encountered and stabilization period will be decremented." δG_max δG logdetG n_stab n_stab - 1

            # set updated to true
            updated = true

            # decrease the stabilization period by one
            n_stab = n_stab - 1

            # resize spin up fermion greens calculator accordingly
            logdetG, sgndetG = resize!(fermion_greens_calculator, G, logdetG, sgndetG, B, n_stab)

            # if failed to evaluate determinant correctly
            if notfinite(logdetG)

                # throw error
                error("Error updating stabilization period to `n_stab = $(n_stab)`: `logdetG = $(logdetG)`.")
            end

            # initialize errors associated with numerical instability to zero
            δG = zero(R)
            δθ = zero(R)

        else

            # warn of numerical instability
            @warn "Numerical instability encountered." δG_max δG logdetG n_stab
        end

    # if very numerically stable
    elseif active && (δG < δG_min)

        # notify that stabilization period being increased
        @info "Stabilization frequency is being increased by one." δG_min δG logdetG n_stab n_stab+1

        # increase the stabilization period by one
        n_stab = n_stab + 1

        # resize spin up fermion greens calculator accordingly
        logdetG, sgndetG = resize!(fermion_greens_calculator, G, logdetG, sgndetG, B, n_stab)

        # if failed to evaluate determinant correctly
        if notfinite(logdetG)

            # throw error
            error("Error updating stabilization period to `n_stab = $(n_stab)`: `logdetG = $(logdetG)`.")
        end

        # initialize errors associated with numerical instability to zero
        δG = zero(R)
        δθ = zero(R)
    end

    return (updated, logdetG, sgndetG, δG, δθ)
end