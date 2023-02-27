function _hmc_update!(
    Gup::Matrix{T}, logdetGup::E, sgndetGup::E, Gup′::Matrix{T},
    Gdn::Matrix{T}, logdetGdn::E, sgndetGdn::E, Gdn′::Matrix{T},
    hubbard_hs_parameters::AbstractHubbardHS{E};
    fermion_path_integral_up::FermionPathIntegral{T,E},
    fermion_path_integral_dn::FermionPathIntegral{T,E},
    fermion_greens_calculator_up::FermionGreensCalculator{T,E},
    fermion_greens_calculator_dn::FermionGreensCalculator{T,E},
    fermion_greens_calculator_up_alt::FermionGreensCalculator{T,E},
    fermion_greens_calculator_dn_alt::FermionGreensCalculator{T,E},
    Bup::Vector{P}, Bdn::Vector{P},
    δG_max::E, δG::E, δθ::E, rng::AbstractRNG,
    initialize_force::Bool = true,
    δG_reject::E = 1e-2) where {T<:Number, E<:AbstractFloat, P<:AbstractPropagator{T,E}}

    (; s, s0, v, dSds) = hubbard_hs_parameters

    return (accepted, logdetGup, sgndetGup, logdetGdn, sgndetGdn, δG, δθ)
end