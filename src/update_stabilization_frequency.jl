@doc raw"""
    update_stabalization_frequency!(Gup::Matrix{T}, logdetGup::E, sgndetGup::T,
                                    Gdn::Matrix{T}, logdetGdn::E, sgndetGdn::T;
                                    fermion_greens_calculator_up::FermionGreensCalculator{T,E},
                                    fermion_greens_calculator_dn::FermionGreensCalculator{T,E},
                                    Bup::Vector{P}, Bdn::Vector{P}, δG::E, δθ::T, n_stab::Int,
                                    δG_max::E) where {E<:AbstractFloat, T<:Number, P<:AbstractPropagator{T,E}}

If the corrected error in the Green's funciton matrix is too large, `δG > δG_max`, then increase the frequency of
numerical stablization by decrementing `n_stab` such that it is updated to `n_stab = max(n_stab - 1, 1)`,
and update the equal-time Green's function matrices and all related variables and types.
If the frequency of stabilization is udpated, then `δG` and `δθ` are reset to zero.
This method returns a tuple of the following variables:
```julia
(logdetGup, sgndetGdn, logdetGup, sgndetGdn, δG, δθ)
```
"""
function update_stabalization_frequency!(Gup::Matrix{T}, logdetGup::E, sgndetGup::T,
                                         Gdn::Matrix{T}, logdetGdn::E, sgndetGdn::T;
                                         fermion_greens_calculator_up::FermionGreensCalculator{T,E},
                                         fermion_greens_calculator_dn::FermionGreensCalculator{T,E},
                                         Bup::Vector{P}, Bdn::Vector{P}, δG::E, δθ::T,
                                         δG_max::E) where {E<:AbstractFloat, T<:Number, P<:AbstractPropagator{T,E}}

    # make sure all fermoin greens calculators are using the same stabilizaiton period
    n_stab = fermion_greens_calculator_up.n_stab::Int
    n_stab_dn = fermion_greens_calculator_dn.n_stab::Int
    @assert n_stab == n_stab_dn

    # if numerical instability occured
    if δG > δG_max || isnan(δG)
        
        # if n_stab = 1 already
        if fermion_greens_calculator_up.n_stab == 1

            # throw error as stabilization frequency can no longer be increased
            error("Error: `δG = $(δG)` and `n_stab = 1` already.")

        # increase stabilization frequency
        else

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
            δG = zero(E)
            δθ = zero(T)
        end
    end

    return (logdetGup, sgndetGup, logdetGdn, sgndetGdn, δG, δθ)
end

@doc raw"""
    update_stabalization_frequency!(G::Matrix{T}, logdetG::E, sgndetG::T;
                                    fermion_greens_calculator::FermionGreensCalculator{T,E},
                                    B::Vector{P}, δG::E, δθ::T, n_stab::Int,
                                    δG_max::E) where {E<:AbstractFloat, T<:Number, P<:AbstractPropagator{T,E}}

If the corrected error in the Green's funciton matrix is too large, `δG > δG_max`, then increase the frequency of
numerical stablization by decrementing `n_stab` such that it is updated to `n_stab = max(n_stab - 1, 1)`,
and update the equal-time Green's function matrices and all related variables and types.
If the frequency of stabilization is udpated, then `δG` and `δθ` are reset to zero.
This method returns a tuple of the following variables:
```julia
(logdetG, sgndetG, δG, δθ)
```
"""
function update_stabalization_frequency!(G::Matrix{T}, logdetG::E, sgndetG::T;
                                         fermion_greens_calculator::FermionGreensCalculator{T,E},
                                         B::Vector{P}, δG::E, δθ::T,
                                         δG_max::E) where {E<:AbstractFloat, T<:Number, P<:AbstractPropagator{T,E}}

    # make sure all fermoin greens calculators are using the same stabilizaiton period
    n_stab = fermion_greens_calculator.n_stab::Int

    # if numerical instability occured
    if δG > δG_max || isnan(δG)
        
        # if n_stab = 1 already
        if fermion_greens_calculator.n_stab == 1

            # throw error as stabilization frequency can no longer be increased
            error("Error: `δG = $(δG)` and `n_stab = 1` already.")

        # increase stabilization frequency
        else

            # decrease the stabilization period by one
            n_stab = n_stab - 1

            # resize spin up fermion greens calculator accordingly
            logdetG, sgndetG = resize!(fermion_greens_calculator, G, logdetG, sgndetG, B, n_stab)

            # if failed to evaluate determinant correctly
            if isnan(logdetG)

                # throw error
                error("Error: `logdetG = NaN`.")
            end

            # intialize errors associated with numerical instability to zero
            δG = zero(E)
            δθ = zero(T)
        end
    end

    return (logdetG, sgndetG, δG, δθ)
end