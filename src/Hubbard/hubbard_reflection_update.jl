@doc raw"""
    reflection_update!(Gup::Matrix{T}, logdetGup::E, sgndetGup::T,
                       Gdn::Matrix{T}, logdetGdn::E, sgndetGdn::T,
                       hubbard_ising_parameters::HubbardIsingHSParameters{E,F};
                       fermion_path_integral_up::FermionPathIntegral{T,E},
                       fermion_path_integral_dn::FermionPathIntegral{T,E},
                       fermion_greens_calculator_up::FermionGreensCalculator{T,E},
                       fermion_greens_calculator_dn::FermionGreensCalculator{T,E},
                       fermion_greens_calculator_up_alt::FermionGreensCalculator{T,E},
                       fermion_greens_calculator_dn_alt::FermionGreensCalculator{T,E},
                       Bup::Vector{P}, Bdn::Vector{P},
                       rng::AbstractRNG) where {T<:Number, E<:AbstractFloat, F<:Int, P<:AbstractPropagator{T,E}}

Perform a reflection update where the sign of every Ising Hubbard-Stratonovich field on a randomly chosen orbital in the lattice is changed.
This function returns `(accepted, logdetGup, sgndetGup, logdetGdn, sgndetGdn)`.

# Arguments

- `Gup::Matrix{T}`: Spin-up eqaul-time Greens function matrix.
- `logdetGup::E`: Log of the determinant of the spin-up eqaul-time Greens function matrix.
- `sgndetGup::T`: Sign/phase of the determinant of the spin-up eqaul-time Greens function matrix.
- `Gdn::Matrix{T}`: Spin-down eqaul-time Greens function matrix.
- `logdetGdn::E`: Log of the determinant of the spin-down eqaul-time Greens function matrix.
- `sgndetGdn::T`: Sign/phase of the determinant of the spin-down eqaul-time Greens function matrix.
- `hubbard_ising_parameters::HubbardIsingHSParameters{E,F}`: Ising Hubbard-Stratonovich fields and associated parameters to update.

# Keyword Arguments

- `fermion_path_integral_up::FermionPathIntegral{T,E}`: An instance of [`FermionPathIntegral`](@ref) type for spin-up electrons.
- `fermion_path_integral_dn::FermionPathIntegral{T,E}`: An instance of [`FermionPathIntegral`](@ref) type for spin-down electrons.
- `fermion_greens_calculator_up::FermionGreensCalculator{T,E}`: Contains matrix factorization information for current spin-up sector state.
- `fermion_greens_calculator_dn::FermionGreensCalculator{T,E}`: Contains matrix factorization information for current spin-down sector state.
- `fermion_greens_calculator_up_alt::FermionGreensCalculator{T,E}`: Used to calculate matrix factorizations for proposed spin-up sector state.
- `fermion_greens_calculator_dn_alt::FermionGreensCalculator{T,E}`: Used to calculate matrix factorizations for proposed spin-up sector state.
- `Bup::Vector{P}`: Spin-up propagators for each imaginary time slice.
- `Bdn::Vector{P}`: Spin-down propagators for each imaginary time slice.
- `rng::AbstractRNG`: Random number generator used in method instead of global random number generator, important for reproducibility.
"""
function reflection_update!(Gup::Matrix{T}, logdetGup::E, sgndetGup::T,
                            Gdn::Matrix{T}, logdetGdn::E, sgndetGdn::T,
                            hubbard_ising_parameters::HubbardIsingHSParameters{E,F};
                            fermion_path_integral_up::FermionPathIntegral{T,E},
                            fermion_path_integral_dn::FermionPathIntegral{T,E},
                            fermion_greens_calculator_up::FermionGreensCalculator{T,E},
                            fermion_greens_calculator_dn::FermionGreensCalculator{T,E},
                            fermion_greens_calculator_up_alt::FermionGreensCalculator{T,E},
                            fermion_greens_calculator_dn_alt::FermionGreensCalculator{T,E},
                            Bup::Vector{P}, Bdn::Vector{P},
                            rng::AbstractRNG) where {T<:Number, E<:AbstractFloat, F<:Int, P<:AbstractPropagator{T,E}}

    (; N, U, α, sites, s, Δτ) = hubbard_ising_parameters
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

    # calculate change in bosonic action, only non-zero for attractive Hubbard interaction
    ΔSb = U[i] > zero(E) ? zero(E) : 2 * α[i] * sum(s_i)

    # update diagonal on-site energy matrix
    @. Vup_i = -2*α[i]/Δτ * s_i + Vup_i
    @. Vdn_i = +2*α[i]/Δτ * s_i + Vdn_i

    # update propagator matrices
    @fastmath @inbounds for l in eachindex(Bup)
        expmΔτVup_l = Bup[l].expmΔτV::Vector{E}
        expmΔτVdn_l = Bdn[l].expmΔτV::Vector{E}
        expmΔτVup_l[site] = exp(-Δτ*Vup_i[l])
        expmΔτVdn_l[site] = exp(-Δτ*Vdn_i[l])
    end

    # calculate new Green's function matrices and determinant of new Green's function matrix
    logdetGup′, sgndetGup′ = calculate_equaltime_greens!(Gup′, fermion_greens_calculator_up_alt, Bup)
    logdetGdn′, sgndetGdn′ = calculate_equaltime_greens!(Gdn′, fermion_greens_calculator_dn_alt, Bdn)

    # calculate acceptance probability P = exp(-ΔS_b)⋅|det(Gup)/det(Gup′)|⋅|det(Gdn)/det(Gdn′)|
    #                                    = exp(-ΔS_b)⋅|det(Mup′)/det(Mup)|⋅|det(Mdn′)/det(Mdn)|
    if isfinite(logdetGup′) && isfinite(logdetGdn′)
        P_i = exp(-ΔSb + logdetGup + logdetGdn - logdetGup′ - logdetGdn′)
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
            expmΔτVup_l = Bup[l].expmΔτV::Vector{E}
            expmΔτVdn_l = Bdn[l].expmΔτV::Vector{E}
            expmΔτVup_l[site] = exp(-Δτ*Vup_i[l])
            expmΔτVdn_l[site] = exp(-Δτ*Vdn_i[l])
        end
        accepted = false
    end

    return (accepted, logdetGup, sgndetGup, logdetGdn, sgndetGdn)
end

@doc raw"""
    reflection_update!(G::Matrix{T}, logdetG::E, sgndetG::T,
                       hubbard_ising_parameters::HubbardIsingHSParameters{E,F};
                       fermion_path_integral::FermionPathIntegral{T,E},
                       fermion_greens_calculator::FermionGreensCalculator{T,E},
                       fermion_greens_calculator_alt::FermionGreensCalculator{T,E},
                       B::Vector{P},
                       rng::AbstractRNG) where {T<:Number, E<:AbstractFloat, F<:Int, P<:AbstractPropagator{T,E}}

Perform a reflection update where the sign of every Ising Hubbard-Stratonovich field on a randomly chosen orbital in the lattice is changed.
This function returns `(accepted, logdetG, sgndetG)`. This method assumes strictly attractive Hubbard interactions.

# Arguments

- `G::Matrix{T}`: Eqaul-time Greens function matrix.
- `logdetG::E`: Log of the determinant of the eqaul-time Greens function matrix.
- `sgndetG::T`: Sign/phase of the determinant of the eqaul-time Greens function matrix.
- `hubbard_ising_parameters::HubbardIsingHSParameters{E,F}`: Ising Hubbard-Stratonovich fields and associated parameters to update.

# Keyword Arguments

- `fermion_path_integral::FermionPathIntegral{T,E}`: An instance of [`FermionPathIntegral`](@ref) type.
- `fermion_greens_calculator::FermionGreensCalculator{T,E}`: Contains matrix factorization information for current state.
- `fermion_greens_calculator_alt::FermionGreensCalculator{T,E}`: Used to calculate matrix factorizations for proposed state.
- `B::Vector{P}`: Propagators for each imaginary time slice.
- `rng::AbstractRNG`: Random number generator used in method instead of global random number generator, important for reproducibility.
"""
function reflection_update!(G::Matrix{T}, logdetG::E, sgndetG::T,
                            hubbard_ising_parameters::HubbardIsingHSParameters{E, F};
                            fermion_path_integral::FermionPathIntegral{T,E},
                            fermion_greens_calculator::FermionGreensCalculator{T,E},
                            fermion_greens_calculator_alt::FermionGreensCalculator{T,E},
                            B::Vector{P},
                            rng::AbstractRNG) where {T<:Number, E<:AbstractFloat, F<:Int, P<:AbstractPropagator{T,E}}

    (; N, U, α, sites, s, Δτ) = hubbard_ising_parameters
    G′ = fermion_greens_calculator_alt.G′

    # make sure stabilization frequencies match
    if fermion_greens_calculator.n_stab != fermion_greens_calculator_alt.n_stab
        resize!(fermion_greens_calculator_alt, fermion_greens_calculator.n_stab)
    end

    # pick a random site/orbital in lattice with finite Hubbard U to perform reflection update on
    i    = rand(rng, 1:N)
    site = sites[i]
    s_i  = @view s[i, :]
    V_i  = @view fermion_path_integral.V[site, :]

    # reflect all the HS field on site i
    @. s_i = -s_i

    # calculate change in bosonic action, only non-zero for attractive Hubbard interaction
    ΔSb = 2 * α[i] * sum(s_i)

    # update diagonal on-site energy matrix
    @. V_i = -2*α[i]/Δτ * s_i + V_i

    # update propagator matrices
    @fastmath @inbounds for l in eachindex(B)
        expmΔτV_l = B[l].expmΔτV::Vector{E}
        expmΔτV_l[site] = exp(-Δτ*V_i[l])
    end

    # calculate new Green's function matrices and determinant of new Green's function matrix
    logdetG′, sgndetG′ = calculate_equaltime_greens!(G′, fermion_greens_calculator_alt, B)

    # calculate acceptance probability P = exp(-ΔS_b)⋅|det(G)/det(G′)|²
    #                                    = exp(-ΔS_b)⋅|det(M′)/det(M)|²
    if isfinite(logdetG′)
        P_i = exp(-ΔSb + 2*logdetG - 2*logdetG′)
    else
        P_i = 0.0
    end

    # accept or reject the update
    if rand(rng) < P_i
        logdetG = logdetG′
        sgndetG = sgndetG′
        copyto!(G, G′)
        copyto!(fermion_greens_calculator, fermion_greens_calculator_alt)
        accepted = true
    else
        # flip HS field back
        @. s_i = -s_i
        # revert diagonal on-site energy matrix
        @. V_i = -2*α[i]/Δτ * s_i + V_i
        # revert propagator matrices
        @fastmath @inbounds for l in eachindex(B)
            expmΔτV_l = B[l].expmΔτV::Vector{E}
            expmΔτV_l[site] = exp(-Δτ*V_i[l])
        end
        accepted = false
    end

    return (accepted, logdetG, sgndetG)
end