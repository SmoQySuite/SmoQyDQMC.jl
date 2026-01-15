@doc raw"""
    abstract type AbstractHST{T<:Number, R<:AbstractFloat} end

Abstract type to represent a Hubbard-Stratonovich transformation.
Here `T` is the effective Hubbard-Stratonovich field type, specifying whether the Hubbard-Stratonovich
transformation is real or complex.
"""
abstract type AbstractHST{T<:Number, R<:AbstractFloat} end

@doc raw"""
    abstract type AbstractSymHST{T, R} <: AbstractHST{T, R} end

Abstract type to represent a Hubbard-Stratonovich transformation that couples to each spin species symmetrically.
Here `T` is the effective Hubbard-Stratonovich field type, specifying whether the Hubbard-Stratonovich
transformation is real or complex.
"""
abstract type AbstractSymHST{T, R} <: AbstractHST{T, R} end

@doc raw"""
    abstract type AbstractAsymHST{T, R} <: AbstractHST{T, R} end

Abstract type to represent a Hubbard-Stratonovich transformation that couples to each spin species asymmetrically.
Here `T` is the effective Hubbard-Stratonovich field type, specifying whether the Hubbard-Stratonovich
transformation is real or complex.
"""
abstract type AbstractAsymHST{T, R} <: AbstractHST{T, R} end

@doc raw"""

    initialize!(
        fermion_path_integral_up::FermionPathIntegral{H},
        fermion_path_integral_dn::FermionPathIntegral{H},
        hst_parameters::AbstractHST{T}
    ) where {H<:Number, T<:Number}

    initialize!(
        fermion_path_integral_up::FermionPathIntegral{H},
        fermion_path_integral_dn::FermionPathIntegral{H},
        hst_parameters::Tuple{Vararg{HST} where HST<:AbstractHST{T}}
    ) where {H<:Number, T<:Number}

    initialize!(
        fermion_path_integral::FermionPathIntegral{H},
        hst_parameters::AbstractSymHST{T}
    ) where {H<:Number, T<:Number}

    initialize!(
        fermion_path_integral::FermionPathIntegral{H},
        hst_parameters::Tuple{Vararg{HST} where HST<:AbstractSymHST{T}}
    ) where {H<:Number, T<:Number}

Initialize a `FermionPathIntegral` integral type to reflect the the current Hubbard-Stratonovich
transformation type represented by `hst_parameters`.
"""
function initialize!(
    fermion_path_integral_up::FermionPathIntegral{H},
    fermion_path_integral_dn::FermionPathIntegral{H},
    hst_parameters::AbstractHST{T}
) where {H<:Number, T<:Number}

    @assert !((H<:Real) &&  (T<:Complex)) "Green's function matrices are real while HubbardSpinHirschHST is complex."
    @assert fermion_path_integral_up.Sb == fermion_path_integral_dn.Sb "$(fermion_path_integral_up.Sb) ≠ $(fermion_path_integral_dn.Sb)"
    _initialize!(fermion_path_integral_up, fermion_path_integral_dn, hst_parameters)

    return nothing
end

function initialize!(
    fermion_path_integral_up::FermionPathIntegral{H},
    fermion_path_integral_dn::FermionPathIntegral{H},
    hst_parameters::Tuple{Vararg{HST} where HST<:AbstractHST{T}}
) where {H<:Number, T<:Number}

    for hst_params in hst_parameters
        _initialize!(fermion_path_integral_up, fermion_path_integral_dn, hst_params)
    end

    return nothing
end

function initialize!(
    fermion_path_integral::FermionPathIntegral{H},
    hst_parameters::AbstractSymHST{T}
) where {H<:Number, T<:Number}

    @assert !((H<:Real) &&  (T<:Complex)) "Green's function matrices are real while HubbardSpinHirschHST is complex."
    _initialize!(fermion_path_integral, hst_parameters)

    return nothing
end

function initialize!(
    fermion_path_integral::FermionPathIntegral{H},
    hst_parameters::Tuple{Vararg{HST} where HST<:AbstractSymHST{T}}
) where {H<:Number, T<:Number}

    for hst_params in hst_parameters
        _initialize!(fermion_path_integral, hst_params)
    end

    return nothing
end


@doc raw"""
    local_updates!(
        # ARGUMENTS
        Gup::Matrix{H}, logdetGup::R, sgndetGup::H,
        Gdn::Matrix{H}, logdetGdn::R, sgndetGdn::H,
        hst_parameters::AbstractHST{T,R};
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

Perform local updates to Hubbard-Stratonovich fields stored in `hst_parameters`.
This method returns a tuple containing `(acceptance_rate, logdetGup, sgndetGup, logdetGdn, sgndetGdn, δG, δθ)`.

# Arguments

- `Gup::Matrix{H}`: Spin-up equal-time Green's function matrix.
- `logdetGup::R`: The log of the absolute value of the determinant of the spin-up equal-time Green's function matrix, ``\log \vert \det G_\uparrow(\tau,\tau) \vert.``
- `sgndetGup::H`: The sign/phase of the determinant of the spin-up equal-time Green's function matrix, ``\det G_\uparrow(\tau,\tau) / \vert \det G_\uparrow(\tau,\tau) \vert.``
- `Gdn::Matrix{H}`: Spin-down equal-time Green's function matrix.
- `logdetGdn::R`: The log of the absolute value of the determinant of the spin-down equal-time Green's function matrix, ``\log \vert \det G_\downarrow(\tau,\tau) \vert.``
- `sgndetGdn::H`: The sign/phase of the determinant of the spin-down equal-time Green's function matrix, ``\det G_\downarrow(\tau,\tau) / \vert \det G_\downarrow(\tau,\tau) \vert.``
- `hst_parameters::AbstractHST{T,R}`: Type representing Hubbard-Stratonovich transformation.

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
    hst_parameters::AbstractHST{T,R};
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

    # get temporary storage matrix
    G′ = fermion_greens_calculator_up.G′

    # record acceptance rate
    acceptance_rate = zero(R)

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

        # perform local updates for the current imaginary-time slice
        acceptance_rate_l, logdetGup, sgndetGup, logdetGdn, sgndetGdn = _local_updates!(
            Gup, logdetGup, sgndetGup, Gdn, logdetGdn, sgndetGdn,
            hst_parameters, fermion_path_integral_up, fermion_path_integral_dn,
            Bup_l, Bdn_l, l, rng
        )
        acceptance_rate += acceptance_rate_l

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

    # normalize acceptance rate
    acceptance_rate /= fermion_path_integral_up.Lτ

    return (acceptance_rate, logdetGup, sgndetGup, logdetGdn, sgndetGdn, δG, δθ)
end

@doc raw"""
    local_updates!(
        # ARGUMENTS
        Gup::Matrix{H}, logdet,Gup::R, sgndetGup::H,
        Gdn::Matrix{H}, logdetGdn::R, sgndetGdn::H,
        hst_parameters::Tuple{Vararg{HST,N} where HST<:AbstractHST{T,R}};
        # KEYWORD ARGUMENTS
        fermion_path_integral_up::FermionPathIntegral{H},
        fermion_path_integral_dn::FermionPathIntegral{H},
        fermion_greens_calculator_up::FermionGreensCalculator{H},
        fermion_greens_calculator_dn::FermionGreensCalculator{H},
        Bup::Vector{P}, Bdn::Vector{P},
        δG::R, δθ::R,  rng::AbstractRNG,
        δG_max::R = 1e-6,
        update_stabilization_frequency::Bool = true
    ) where {H<:Number, T<:Number, R<:Real, P<:AbstractPropagator, N}

Perform local updates to Hubbard-Stratonovich fields for `N` different types of Hubbard-Stratonovich transformations.
This method returns a tuple containing `(acceptance_rates, logdetGup, sgndetGup, logdetGdn, sgndetGdn, δG, δθ)`.
Note that `acceptance_rates` is a tuple returning the acceptance rate for local updates of each type of Hubbard-Stratonovich field that was sampled.

# Arguments

- `Gup::Matrix{H}`: Spin-up equal-time Green's function matrix.
- `logdetGup::R`: The log of the absolute value of the determinant of the spin-up equal-time Green's function matrix, ``\log \vert \det G_\uparrow(\tau,\tau) \vert.``
- `sgndetGup::H`: The sign/phase of the determinant of the spin-up equal-time Green's function matrix, ``\det G_\uparrow(\tau,\tau) / \vert \det G_\uparrow(\tau,\tau) \vert.``
- `Gdn::Matrix{H}`: Spin-down equal-time Green's function matrix.
- `logdetGdn::R`: The log of the absolute value of the determinant of the spin-down equal-time Green's function matrix, ``\log \vert \det G_\downarrow(\tau,\tau) \vert.``
- `sgndetGdn::H`: The sign/phase of the determinant of the spin-down equal-time Green's function matrix, ``\det G_\downarrow(\tau,\tau) / \vert \det G_\downarrow(\tau,\tau) \vert.``
- `hst_parameters::Tuple{Vararg{HST,N} where HST<:AbstractHST{T,R}}`: Tuple of parameters for multiple different Hubbard-Stratonovich transformation fields that will be sampled.

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
    hst_parameters::Tuple{Vararg{HST,N} where HST<:AbstractHST{T,R}};
    # KEYWORD ARGUMENTS
    fermion_path_integral_up::FermionPathIntegral{H},
    fermion_path_integral_dn::FermionPathIntegral{H},
    fermion_greens_calculator_up::FermionGreensCalculator{H},
    fermion_greens_calculator_dn::FermionGreensCalculator{H},
    Bup::Vector{P}, Bdn::Vector{P},
    δG::R, δθ::R,  rng::AbstractRNG,
    δG_max::R = 1e-6,
    update_stabilization_frequency::Bool = true
) where {H<:Number, T<:Number, R<:Real, P<:AbstractPropagator, N}

    @assert !( (H<:Real) &&  (T<:Complex)) "Green's function matrices are real while Hubbard-Stratonovich transformation is complex."

    # get temporary storage matrix
    G′ = fermion_greens_calculator_up.G′

    # initialize vector to record acceptance rates
    acceptance_rates = @MVector zeros(R, N)

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

        # iterate over different types of Hubbard-Stratonovich Transformations
        for n in 1:N

            # perform local updates for the current imaginary-time slice
            acceptance_rate_l, logdetGup, sgndetGup, logdetGdn, sgndetGdn, = _local_updates!(
                Gup, logdetGup, sgndetGup, Gdn, logdetGdn, sgndetGdn,
                hst_parameters[n], fermion_path_integral_up, fermion_path_integral_dn,
                Bup_l, Bdn_l, l, rng
            )

            # record the acceptance rate
            acceptance_rates[n] += acceptance_rate_l
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

    # normalize the acceptance rates and convert to tuple
    @. acceptance_rates /= fermion_path_integral_up.Lτ
    acceptance_rates = tuple(acceptance_rates...)

    return (acceptance_rates, logdetGup, sgndetGup, logdetGdn, sgndetGdn, δG, δθ)
end

@doc raw"""
    local_updates!(
        # ARGUMENTS
        G::Matrix{H}, logdetG::R, sgndetG::H,
        hst_parameters::AbstractSymHST{T,R};
        # KEYWORD ARGUMENTS
        fermion_path_integral::FermionPathIntegral{H},
        fermion_greens_calculator::FermionGreensCalculator{H},
        B::Vector{P},
        δG::R, δθ::R,  rng::AbstractRNG,
        δG_max::R = 1e-6,
        update_stabilization_frequency::Bool = true
    ) where {H<:Number, T<:Number, R<:Real, P<:AbstractPropagator}

Perform local updates to Hubbard-Stratonovich fields for a spin-symmetric (density channel) Hubbard-Stratonovich transformation.
This method returns a tuple containing `(acceptance_rate, logdetG, sgndetG, δG, δθ)`.

# Arguments

- `G::Matrix{H}`: Equal-time Green's function matrix.
- `logdetG::R`: The log of the absolute value of the determinant of the equal-time Green's function matrix, ``\log \vert \det G(\tau,\tau) \vert.``
- `sgndetG::H`: The sign/phase of the determinant of the equal-time Green's function matrix, ``\det G(\tau,\tau) / \vert \det G(\tau,\tau) \vert.``
- `hst_parameters::AbstractSymHST{T,R}`: Type representing spin-symmetric Hubbard-Stratonovich transformation.

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
    hst_parameters::AbstractSymHST{T,R};
    # KEYWORD ARGUMENTS
    fermion_path_integral::FermionPathIntegral{H},
    fermion_greens_calculator::FermionGreensCalculator{H},
    B::Vector{P},
    δG::R, δθ::R,  rng::AbstractRNG,
    δG_max::R = 1e-6,
    update_stabilization_frequency::Bool = true
) where {H<:Number, T<:Number, R<:Real, P<:AbstractPropagator}

    @assert !( (H<:Real) &&  (T<:Complex)) "Green's function matrices are real while Hubbard-Stratonovich transformation is complex."

    # get temporary storage matrix
    G′ = fermion_greens_calculator.G′

    # record acceptance rate
    acceptance_rate = zero(R)

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

        # perform local update for current imaginary-time slice
        acceptance_rate_l, logdetG, sgndetG = _local_updates!(
            G, logdetG, sgndetG,
            hst_parameters, fermion_path_integral,
            B_l, l, rng
        )
        acceptance_rate += acceptance_rate_l

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

    # normalize acceptance rate
    acceptance_rate /= fermion_path_integral.Lτ

    return (acceptance_rate, logdetG, sgndetG, δG, δθ)
end

@doc raw"""
    local_updates!(
        # ARGUMENTS
        G::Matrix{H}, logdetG::R, sgndetG::H,
        hst_parameters::Tuple{Vararg{HST,N} where HST<:AbstractSymHST{T,R}};
        # KEYWORD ARGUMENTS
        fermion_path_integral::FermionPathIntegral{H},
        fermion_greens_calculator::FermionGreensCalculator{H},
        B::Vector{P},
        δG::R, δθ::R,  rng::AbstractRNG,
        δG_max::R = 1e-6,
        update_stabilization_frequency::Bool = true
    ) where {H<:Number, T<:Number, R<:Real, P<:AbstractPropagator, N}

Perform local updates to multiple types Hubbard-Stratonovich fields for a spin-symmetric (density channel) Hubbard-Stratonovich transformation.
This method returns a tuple containing `(acceptance_rate, logdetG, sgndetG, δG, δθ)`.

# Arguments

- `G::Matrix{H}`: Equal-time Green's function matrix.
- `logdetG::R`: The log of the absolute value of the determinant of the equal-time Green's function matrix, ``\log \vert \det G(\tau,\tau) \vert.``
- `sgndetG::H`: The sign/phase of the determinant of the equal-time Green's function matrix, ``\det G(\tau,\tau) / \vert \det G(\tau,\tau) \vert.``
- `hst_parameters::Tuple{Vararg{HST,N} where HST<:AbstractSymHST{T,R}}`: Tuple of parameters for multiple different spin-symmetric Hubbard-Stratonovich transformation fields that will be sampled.

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
    hst_parameters::Tuple{Vararg{HST,N} where HST<:AbstractSymHST{T,R}};
    # KEYWORD ARGUMENTS
    fermion_path_integral::FermionPathIntegral{H},
    fermion_greens_calculator::FermionGreensCalculator{H},
    B::Vector{P},
    δG::R, δθ::R,  rng::AbstractRNG,
    δG_max::R = 1e-6,
    update_stabilization_frequency::Bool = true
) where {H<:Number, T<:Number, R<:Real, P<:AbstractPropagator, N}

    @assert !( (H<:Real) &&  (T<:Complex)) "Green's function matrices are real while Hubbard-Stratonovich transformation is complex."

    # get temporary storage matrix
    G′ = fermion_greens_calculator.G′

    # initialize vector to record acceptance rates
    acceptance_rates = @MVector zeros(R, N)

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

        # iterate over different types of Hubbard-Stratonovich Transformations
        for n in 1:N

            # perform local updates for the current imaginary-time slice
            acceptance_rate_l, logdetG, sgndetG = _local_updates!(
                G, logdetG, sgndetG,
                hst_parameters[n], fermion_path_integral,
                B_l, l, rng
            )

            # record the acceptance rate
            acceptance_rates[n] += acceptance_rate_l
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

    # normalize acceptance rate
    @. acceptance_rates /= fermion_path_integral.Lτ

    return (acceptance_rates, logdetG, sgndetG, δG, δθ)
end


@doc raw"""
    reflection_update!(
        # ARGUMENTS
        Gup::Matrix{H}, logdetGup::R, sgndetGup::H,
        Gdn::Matrix{H}, logdetGdn::R, sgndetGdn::H,
        hst_parameters::AbstractHST{T,R};
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

Perform a reflection update for a Hubbard-Stratonovich field on a randomly chosen location in the lattice for all imaginary-time slices.
This function returns `(accepted, logdetGup, sgndetGup, logdetGdn, sgndetGdn)`.

# Arguments

- `Gup::Matrix{H}`: Spin-up equal-time Greens function matrix.
- `logdetGup::R`: Log of the determinant of the spin-up equal-time Greens function matrix.
- `sgndetGup::H`: Sign/phase of the determinant of the spin-up equal-time Greens function matrix.
- `Gdn::Matrix{H}`: Spin-down equal-time Greens function matrix.
- `logdetGdn::R`: Log of the determinant of the spin-down equal-time Greens function matrix.
- `sgndetGdn::H`: Sign/phase of the determinant of the spin-down equal-time Greens function matrix.
- `hst_parameters::AbstractHST{T,R}`: Hubbard-Stratonovich fields and associated parameters to update.

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
    hst_parameters::AbstractHST{T,R};
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
    
    (accepted, logdetGup, sgndetGup, logdetGdn, sgndetGdn) = _reflection_update!(
        Gup, logdetGup, sgndetGup,
        Gdn, logdetGdn, sgndetGdn,
        hst_parameters;
        fermion_path_integral_up, fermion_path_integral_dn,
        fermion_greens_calculator_up, fermion_greens_calculator_dn,
        fermion_greens_calculator_up_alt, fermion_greens_calculator_dn_alt,
        Bup, Bdn, rng
    )

    return (accepted, logdetGup, sgndetGup, logdetGdn, sgndetGdn)
end

@doc raw"""
    reflection_update!(
        # ARGUMENTS
        G::Matrix{H}, logdetG::R, sgndetG::H,
        hst_parameters::AbstractSymHST{T,R};
        # KEYWORD ARGUMENTS
        fermion_path_integral::FermionPathIntegral{H},
        fermion_greens_calculator::FermionGreensCalculator{H,R},
        fermion_greens_calculator_alt::FermionGreensCalculator{H,R},
        B::Vector{P},
        rng::AbstractRNG
    ) where {H<:Number, T<:Number, R<:Real, P<:AbstractPropagator}

Perform a reflection update for a spin-symmetric (density channel) Hubbard-Stratonovich field on a randomly chosen location in the lattice for all imaginary-time slices.
This function returns `(accepted, logdetG, sgndetG)`.

# Arguments

- `G::Matrix{H}`: The current Hubbard-Stratonovich field matrix.
- `logdetG::R`: Log of the determinant of the Hubbard-Stratonovich field matrix.
- `sgndetG::H`: Sign/phase of the determinant of the Hubbard-Stratonovich field matrix.
- `hst_parameters::AbstractSymHST{T,R}`: Hubbard-Stratonovich fields and associated parameters to update.

# Keyword Arguments

- `fermion_path_integral::FermionPathIntegral{H}`: An instance of [`FermionPathIntegral`](@ref).
- `fermion_greens_calculator::FermionGreensCalculator{H,R}`: Contains matrix factorization information for current state.
- `fermion_greens_calculator_alt::FermionGreensCalculator{H,R}`: Used to calculate matrix factorizations for proposed state.
- `B::Vector{P}`: Propagators for each imaginary time slice.
- `rng::AbstractRNG`: Random number generator used in method instead of global random number
"""
function reflection_update!(
    # ARGUMENTS
    G::Matrix{H}, logdetG::R, sgndetG::H,
    hst_parameters::AbstractSymHST{T,R};
    # KEYWORD ARGUMENTS
    fermion_path_integral::FermionPathIntegral{H},
    fermion_greens_calculator::FermionGreensCalculator{H,R},
    fermion_greens_calculator_alt::FermionGreensCalculator{H,R},
    B::Vector{P},
    rng::AbstractRNG
) where {H<:Number, T<:Number, R<:Real, P<:AbstractPropagator}

    (accepted, logdetG, sgndetG) = _reflection_update!(
        G, logdetG, sgndetG,
        hst_parameters;
        fermion_path_integral,
        fermion_greens_calculator,
        fermion_greens_calculator_alt,
        B, rng
    )

    return (accepted, logdetG, sgndetG)
end

# default reflection update method
function _reflection_update!(
    # ARGUMENTS
    Gup::Matrix{H}, logdetGup::R, sgndetGup::H,
    Gdn::Matrix{H}, logdetGdn::R, sgndetGdn::H,
    hst_parameters::AbstractHST{T,R};
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

    return 0.0
end

# default reflection update method
function _reflection_update!(
    # ARGUMENTS
    G::Matrix{H}, logdetG::R, sgndetG::H,
    hst_parameters::AbstractSymHST{T,R};
    # KEYWORD ARGUMENTS
    fermion_path_integral::FermionPathIntegral{H},
    fermion_greens_calculator::FermionGreensCalculator{H,R},
    fermion_greens_calculator_alt::FermionGreensCalculator{H,R},
    B::Vector{P},
    rng::AbstractRNG
) where {H<:Number, T<:Number, R<:Real, P<:AbstractPropagator}

    return 0.0
end


@doc raw"""
    swap_update!(
        # ARGUMENTS
        Gup::Matrix{H}, logdetGup::R, sgndetGup::H,
        Gdn::Matrix{H}, logdetGdn::R, sgndetGdn::H,
        hst_parameters::AbstractHST{T,R};
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

Perform a swap update for a Hubbard-Stratonovich field between a pair of randomly chosen locations in the lattice for all imaginary-time slices.
This function returns `(accepted, logdetGup, sgndetGup, logdetGdn, sgndetGdn)`.

# Arguments

- `Gup::Matrix{H}`: Spin-up equal-time Greens function matrix.
- `logdetGup::R`: Log of the determinant of the spin-up equal-time Greens function matrix.
- `sgndetGup::H`: Sign/phase of the determinant of the spin-up equal-time Greens function matrix.
- `Gdn::Matrix{H}`: Spin-down equal-time Greens function matrix.
- `logdetGdn::R`: Log of the determinant of the spin-down equal-time Greens function matrix.
- `sgndetGdn::H`: Sign/phase of the determinant of the spin-down equal-time Greens function matrix.
- `hst_parameters::AbstractHST{T,R}`: Hubbard-Stratonovich fields and associated parameters to update.

# Keyword Arguments

- `fermion_path_integral_up::FermionPathIntegral{H}`: An instance of [`FermionPathIntegral`](@ref) type for spin-up electrons.
- `fermion_path_integral_dn::FermionPathIntegral{H}`: An instance of [`FermionPathIntegral`](@ref) type for spin-down electrons.
- `fermion_greens_calculator_up::FermionGreensCalculator{H,R}`: Contains matrix factorization information for current spin-up sector state.
- `fermion_greens_calculator_dn::FermionGreensCalculator{H,R}`: Contains matrix factorization information for current spin-down sector state.
- `fermion_greens_calculator_up_alt::FermionGreensCalculator{H,R}`: Used to calculate matrix factorizations for proposed spin-up sector state.
- `fermion_greens_calculator_dn_alt::FermionGreensCalculator{H,R}`: Used to calculate matrix factorizations for proposed spin-down sector state.
- `Bup::Vector{P}`: Spin-up propagators for each imaginary time slice.
- `Bdn::Vector{P}`: Spin-down propagators for each imaginary time slice
- `rng::AbstractRNG`: Random number generator used in method instead of global random number generator, important for reproducibility.
"""
function swap_update!(
    # ARGUMENTS
    Gup::Matrix{H}, logdetGup::R, sgndetGup::H,
    Gdn::Matrix{H}, logdetGdn::R, sgndetGdn::H,
    hst_parameters::AbstractHST{T,R};
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

    # ensure there is more then one interaction in the lattice that was decoupled with a HS transformation
    if hst_parameters.N > 1
        # perform swap update
        (accepted, logdetGup, sgndetGup, logdetGdn, sgndetGdn) =_swap_update!(
            Gup, logdetGup, sgndetGup,
            Gdn, logdetGdn, sgndetGdn,
            hst_parameters;
            fermion_path_integral_up, fermion_path_integral_dn,
            fermion_greens_calculator_up, fermion_greens_calculator_dn,
            fermion_greens_calculator_up_alt, fermion_greens_calculator_dn_alt,
            Bup, Bdn, rng
        )
    else
        accepted = false
    end

    return (accepted, logdetGup, sgndetGup, logdetGdn, sgndetGdn)
end

@doc raw"""
    swap_update!(
        # ARGUMENTS
        G::Matrix{H}, logdetG::R, sgndetG::H,
        hst_parameters::AbstractSymHST{T,R};
        # KEYWORD ARGUMENTS
        fermion_path_integral::FermionPathIntegral{H},
        fermion_greens_calculator::FermionGreensCalculator{H,R},
        fermion_greens_calculator_alt::FermionGreensCalculator{H,R},
        B::Vector{P},
        rng::AbstractRNG
    ) where {H<:Number, T<:Number, R<:Real, P<:AbstractPropagator}

Perform a swap update for a spin-symmetric (density channel) Hubbard-Stratonovich field between a pair of randomly chosen locations in the lattice for all imaginary-time slices.
This function returns `(accepted, logdetG, sgndetG)`.

# Arguments

- `G::Matrix{H}`: The current Hubbard-Stratonovich field matrix.
- `logdetG::R`: Log of the determinant of the Hubbard-Stratonovich field
- `sgndetG::H`: Sign of the determinant of the Hubbard-Stratonovich field matrix.
- `hst_parameters::AbstractSymHST{T,R}`: Hubbard-Stratonovich fields and associated parameters to update.

# Keyword Arguments

- `fermion_path_integral::FermionPathIntegral{H}`: An instance of [`FermionPathIntegral`](@ref).
- `fermion_greens_calculator::FermionGreensCalculator{H,R}`: Contains matrix factorization information for current state.
- `fermion_greens_calculator_alt::FermionGreensCalculator{H,R}`: Used to calculate matrix factorizations for proposed state.
- `B::Vector{P}`: Propagators for each imaginary time slice.
- `rng::AbstractRNG`: Random number generator used in method instead of global random number generator, important for reproducibility.
"""
function swap_update!(
    # ARGUMENTS
    G::Matrix{H}, logdetG::R, sgndetG::H,
    hst_parameters::AbstractSymHST{T,R};
    # KEYWORD ARGUMENTS
    fermion_path_integral::FermionPathIntegral{H},
    fermion_greens_calculator::FermionGreensCalculator{H,R},
    fermion_greens_calculator_alt::FermionGreensCalculator{H,R},
    B::Vector{P},
    rng::AbstractRNG
) where {H<:Number, T<:Number, R<:Real, P<:AbstractPropagator}

    # ensure there is more then one interaction in the lattice that was decoupled with a HS transformation
    if hst_parameters.N > 1
        # perform swap update
        (accepted, logdetG, sgndetG) = _swap_update!(
            G, logdetG, sgndetG,
            hst_parameters;
            fermion_path_integral,
            fermion_greens_calculator,
            fermion_greens_calculator_alt,
            B, rng
        )
    else
        accepted = false
    end

    return (accepted, logdetG, sgndetG)
end

# default swap update method
function _swap_update!(
    # ARGUMENTS
    Gup::Matrix{H}, logdetGup::R, sgndetGup::H,
    Gdn::Matrix{H}, logdetGdn::R, sgndetGdn::H,
    hst_parameters::AbstractHST{T,R};
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

    return 0.0
end

# default swap update method
function _swap_update!(
    # ARGUMENTS
    G::Matrix{H}, logdetG::R, sgndetG::H,
    hst_parameters::AbstractSymHST{T,R};
    # KEYWORD ARGUMENTS
    fermion_path_integral::FermionPathIntegral{H},
    fermion_greens_calculator::FermionGreensCalculator{H,R},
    fermion_greens_calculator_alt::FermionGreensCalculator{H,R},
    B::Vector{P},
    rng::AbstractRNG
) where {H<:Number, T<:Number, R<:Real, P<:AbstractPropagator}

    return 0.0
end