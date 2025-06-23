@doc raw"""
    update_stabilization_frequency!(
        Gup::Matrix{H}, logdetGup::R, sgndetGup::H,
        Gdn::Matrix{H}, logdetGdn::R, sgndetGdn::H;
        fermion_greens_calculator_up::FermionGreensCalculator{H,R},
        fermion_greens_calculator_dn::FermionGreensCalculator{H,R},
        Bup::Vector{P}, Bdn::Vector{P}, δG::R, δθ::R, δG_max::R
    ) where {H<:Number, R<:Real, P<:AbstractPropagator}

If the corrected error in the Green's function matrix is too large, `δG > δG_max`, then increase the frequency of
numerical stablization by decrementing `n_stab` such that it is updated to `n_stab = max(n_stab - 1, 1)`,
and update the equal-time Green's function matrices and all related variables and types.
If the frequency of stabilization is udpated, then `δG` and `δθ` are reset to zero.
This method returns a tuple of the following variables:
```julia
(logdetGup, sgndetGdn, logdetGup, sgndetGdn, δG, δθ),
```
where `updated = true` if `n_stab` was decremented.
"""
function update_stabilization_frequency!(
    Gup::Matrix{H}, logdetGup::R, sgndetGup::H,
    Gdn::Matrix{H}, logdetGdn::R, sgndetGdn::H;
    fermion_greens_calculator_up::FermionGreensCalculator{H,R},
    fermion_greens_calculator_dn::FermionGreensCalculator{H,R},
    Bup::Vector{P}, Bdn::Vector{P}, δG::R, δθ::R, δG_max::R
) where {H<:Number, R<:Real, P<:AbstractPropagator}

    # make sure all fermoin greens calculators are using the same stabilizaiton period
    n_stab = fermion_greens_calculator_up.n_stab::Int
    n_stab_dn = fermion_greens_calculator_dn.n_stab::Int
    @assert n_stab == n_stab_dn

    # initialize updated to false
    updated = false

    # if numerical instability occured
    if δG > δG_max || (!isfinite(δG)) || (!isfinite(logdetGup)) || (!isfinite(logdetGdn))
        
        # if n_stab can still be reduced
        if fermion_greens_calculator_up.n_stab > 1

            # set updated to true
            updated = true

            # decrease the stabilization period by one
            n_stab = n_stab - 1

            # resize spin up fermion greens calculator accordingly
            logdetGup, sgndetGup = resize!(fermion_greens_calculator_up, Gup, logdetGup, sgndetGup, Bup, n_stab)

            # resize spin down fermion greens calculator accordingly
            logdetGdn, sgndetGdn = resize!(fermion_greens_calculator_dn, Gdn, logdetGdn, sgndetGdn, Bdn, n_stab)

            # if failed to evaluate determinant correctly
            if isnan(logdetGup) || isnan(logdetGdn)

                # throw error
                error("Error: `logdetGup = $(logdetGup)` and `logdetGdn = $(logdetGdn)`.")
            end

            # intialize errors associated with numerical instability to zero
            δG = zero(R)
            δθ = zero(R)
        end
    end

    return (updated, logdetGup, sgndetGup, logdetGdn, sgndetGdn, δG, δθ)
end

@doc raw"""
    update_stabilization_frequency!(
        G::Matrix{H}, logdetG::R, sgndetG::H;
        fermion_greens_calculator::FermionGreensCalculator{H,R},
        B::Vector{P}, δG::R, δθ::R, δG_max::R
    ) where {H<:Number, R<:Real, P<:AbstractPropagator}

If the corrected error in the Green's function matrix is too large, `δG > δG_max`, then increase the frequency of
numerical stablization by decrementing `n_stab` such that it is updated to `n_stab = max(n_stab - 1, 1)`,
and update the equal-time Green's function matrices and all related variables and types.
If the frequency of stabilization is udpated, then `δG` and `δθ` are reset to zero.
This method returns a tuple of the following variables:
```julia
(updated, logdetG, sgndetG, δG, δθ),
```
where `updated = true` if `n_stab` was decremented.
"""
function update_stabilization_frequency!(
    G::Matrix{H}, logdetG::R, sgndetG::H;
    fermion_greens_calculator::FermionGreensCalculator{H,R},
    B::Vector{P}, δG::R, δθ::R, δG_max::R
) where {H<:Number, R<:Real, P<:AbstractPropagator}

    # make sure all fermoin greens calculators are using the same stabilizaiton period
    n_stab = fermion_greens_calculator.n_stab::Int

    # initialize updated to false
    updated = false

    # if numerical instability occured
    if δG > δG_max || (!isfinite(δG)) || (!isfinite(logdetG))

        # if n_stab can still be reduced
        if fermion_greens_calculator.n_stab > 1

            # set updated to true
            updated = true

            # decrease the stabilization period by one
            n_stab = n_stab - 1

            # resize spin up fermion greens calculator accordingly
            logdetG, sgndetG = resize!(fermion_greens_calculator, G, logdetG, sgndetG, B, n_stab)

            # if failed to evaluate determinant correctly
            if !isfinite(logdetG)

                # throw error
                error("Error: `logdetG = $(logdetG)`.")
            end

            # intialize errors associated with numerical instability to zero
            δG = zero(R)
            δθ = zero(R)
        end
    end

    return (updated, logdetG, sgndetG, δG, δθ)
end