@doc raw"""
    swap_update!(Gup::Matrix{T}, logdetGup::E, sgndetGup::T,
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

Perform a swap update where the HS fields associated with two randomly chosen sites in the lattice are exchanged.
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
function swap_update!(Gup::Matrix{T}, logdetGup::E, sgndetGup::T,
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

    # ranomly pick two sites with Hubbard U interaction on them
    i = rand(rng, 1:N)
    j = i
    while j==i
        j = rand(rng, 1:N)
    end

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

    # calculate the initial bosonic action
    Sb  = U[i] > zero(E) ? zero(E) : 2 * α[i] * sum(s_i)
    Sb += U[j] > zero(E) ? zero(E) : 2 * α[j] * sum(s_j)

    # swap the HS fields
    swap!(s_i, s_j)

    # calculate the final bosonic action
    Sb′  = U[i] > zero(E) ? zero(E) : 2 * α[i] * sum(s_i)
    Sb′ += U[j] > zero(E) ? zero(E) : 2 * α[j] * sum(s_j)

    # calculate the change in the bosonic action
    ΔSb = Sb′ - Sb

    # update diagonal on-site energy matrix
    @. Vup_i = Vup_i - α[i]/Δτ * (s_i - s_j) 
    @. Vdn_i = Vdn_i + α[i]/Δτ * (s_i - s_j) 
    @. Vup_j = Vup_j - α[j]/Δτ * (s_j - s_i) 
    @. Vdn_j = Vdn_j + α[j]/Δτ * (s_j - s_i)

    # update propagator matrices
    @fastmath @inbounds for l in eachindex(Bup)
        expmΔτVup_l = Bup[l].expmΔτV::Vector{E}
        expmΔτVdn_l = Bdn[l].expmΔτV::Vector{E}
        expmΔτVup_l[site_i] = exp(-Δτ*Vup_i[l])
        expmΔτVdn_l[site_i] = exp(-Δτ*Vdn_i[l])
        expmΔτVup_l[site_j] = exp(-Δτ*Vup_j[l])
        expmΔτVdn_l[site_j] = exp(-Δτ*Vdn_j[l])
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
        # flip HS fields back
        swap!(s_i, s_j)
        # revert diagonal on-site energy matrix
        @. Vup_i = Vup_i - α[i]/Δτ * (s_i - s_j) 
        @. Vdn_i = Vdn_i + α[i]/Δτ * (s_i - s_j) 
        @. Vup_j = Vup_j - α[j]/Δτ * (s_j - s_i) 
        @. Vdn_j = Vdn_j + α[j]/Δτ * (s_j - s_i)
        # revert propagator matrices
        @fastmath @inbounds for l in eachindex(Bup)
            expmΔτVup_l = Bup[l].expmΔτV::Vector{E}
            expmΔτVdn_l = Bdn[l].expmΔτV::Vector{E}
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
    swap_update!(G::Matrix{T}, logdetG::E, sgndetG::T,
                 hubbard_ising_parameters::HubbardIsingHSParameters{E,F};
                 fermion_path_integral::FermionPathIntegral{T,E},
                 fermion_greens_calculator::FermionGreensCalculator{T,E},
                 fermion_greens_calculator_alt::FermionGreensCalculator{T,E},
                 B::Vector{P}, rng::AbstractRNG) where {T<:Number, E<:AbstractFloat, P<:AbstractPropagator{T,E}}

For strictly attractive Hubbard interactions, perform a swap update where the HS fields associated with two randomly chosen
sites in the lattice are exchanged. This function returns `(accepted, logdetG, sgndetG)`.

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
function swap_update!(G::Matrix{T}, logdetG::E, sgndetG::T,
                      hubbard_ising_parameters::HubbardIsingHSParameters{E,F};
                      fermion_path_integral::FermionPathIntegral{T,E},
                      fermion_greens_calculator::FermionGreensCalculator{T,E},
                      fermion_greens_calculator_alt::FermionGreensCalculator{T,E},
                      B::Vector{P}, rng::AbstractRNG) where {T<:Number, E<:AbstractFloat, F<:Int, P<:AbstractPropagator{T,E}}

    (; N, U, α, sites, s, Δτ) = hubbard_ising_parameters
    G′ = fermion_greens_calculator_alt.G′

    # make sure stabilization frequencies match
    if fermion_greens_calculator.n_stab != fermion_greens_calculator_alt.n_stab
        resize!(fermion_greens_calculator_alt, fermion_greens_calculator.n_stab)
    end

    # ranomly pick two sites with Hubbard U interaction on them
    i = rand(rng, 1:N)
    j = i
    while j==i
        j = rand(rng, 1:N)
    end

    # get the site index associted with each Hubbard U
    site_i = sites[i]
    site_j = sites[j]

    # get the HS fields associated with each site
    s_i = @view s[i,:]
    s_j = @view s[j,:]
    V_i = @view fermion_path_integral.V[site_i, :]
    V_j = @view fermion_path_integral.V[site_j, :]

    # calculate the initial bosonic action
    Sb  = U[i] > zero(E) ? zero(E) : 2 * α[i] * sum(s_i)
    Sb += U[j] > zero(E) ? zero(E) : 2 * α[j] * sum(s_j)

    # swap the HS fields
    swap!(s_i, s_j)

    # calculate the final bosonic action
    Sb′  = U[i] > zero(E) ? zero(E) : 2 * α[i] * sum(s_i)
    Sb′ += U[j] > zero(E) ? zero(E) : 2 * α[j] * sum(s_j)

    # calculate the change in the bosonic action
    ΔSb = Sb′ - Sb

    # update diagonal on-site energy matrix
    @. V_i = V_i - α[i]/Δτ * (s_i - s_j) 
    @. V_j = V_j - α[j]/Δτ * (s_j - s_i) 

    # update propagator matrices
    @fastmath @inbounds for l in eachindex(B)
        expmΔτV_l = B[l].expmΔτV::Vector{E}
        expmΔτV_l[site_i] = exp(-Δτ*V_i[l])
        expmΔτV_l[site_j] = exp(-Δτ*V_j[l])
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
        # flip HS fields back
        swap!(s_i, s_j)
        # revert diagonal on-site energy matrix
        @. V_i = V_i - α[i]/Δτ * (s_i - s_j) 
        @. V_j = V_j - α[j]/Δτ * (s_j - s_i) 
        # revert propagator matrices
        @fastmath @inbounds for l in eachindex(B)
            expmΔτV_l = B[l].expmΔτV::Vector{E}
            expmΔτV_l[site_i] = exp(-Δτ*V_i[l])
            expmΔτV_l[site_j] = exp(-Δτ*V_j[l])
        end
        accepted = false
    end

    return (accepted, logdetG, sgndetG)
end




function swap_update!(Gup::Matrix{T}, logdetGup::E, sgndetGup::T,
                      Gdn::Matrix{T}, logdetGdn::E, sgndetGdn::T,
                      hubbard_hs_parameters::AbstractHubbardHS{E};
                      fermion_path_integral_up::FermionPathIntegral{T,E},
                      fermion_path_integral_dn::FermionPathIntegral{T,E},
                      fermion_greens_calculator_up::FermionGreensCalculator{T,E},
                      fermion_greens_calculator_dn::FermionGreensCalculator{T,E},
                      fermion_greens_calculator_up_alt::FermionGreensCalculator{T,E},
                      fermion_greens_calculator_dn_alt::FermionGreensCalculator{T,E},
                      Bup::Vector{P}, Bdn::Vector{P},
                      rng::AbstractRNG) where {T<:Number, E<:AbstractFloat, P<:AbstractPropagator{T,E}}

    (; N, U, sites, s, Δτ, Lτ) = hubbard_hs_parameters
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
    i = rand(rng, 1:N)
    j = i
    while j==i
        j = rand(rng, 1:N)
    end

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

    # calculate the initial bosonic action
    Sb = _bosonic_action(hubbard_hs_parameters)

    # substract off effect of HS field on Vup and Vdn
    _update_V!(fermion_path_integral_up, hubbard_hs_parameters, i, +1, -1)
    _update_V!(fermion_path_integral_dn, hubbard_hs_parameters, i, -1, -1)
    _update_V!(fermion_path_integral_up, hubbard_hs_parameters, j, +1, -1)
    _update_V!(fermion_path_integral_dn, hubbard_hs_parameters, j, -1, -1)

    # swap the HS fields
    swap!(s_i, s_j)

    # add effect of HS field on Vup and Vdn
    _update_V!(fermion_path_integral_up, hubbard_hs_parameters, i, +1, +1)
    _update_V!(fermion_path_integral_dn, hubbard_hs_parameters, i, -1, +1)
    _update_V!(fermion_path_integral_up, hubbard_hs_parameters, j, +1, +1)
    _update_V!(fermion_path_integral_dn, hubbard_hs_parameters, j, -1, +1)

    # calculate the final bosonic action
    Sb′ = _bosonic_action(hubbard_hs_parameters)

    # calculate the change in the bosonic action
    ΔSb = Sb′ - Sb

    # update propagator matrices
    @fastmath @inbounds for l in eachindex(Bup)
        expmΔτVup_l = Bup[l].expmΔτV::Vector{E}
        expmΔτVdn_l = Bdn[l].expmΔτV::Vector{E}
        expmΔτVup_l[site_i] = exp(-Δτ*Vup_i[l])
        expmΔτVdn_l[site_i] = exp(-Δτ*Vdn_i[l])
        expmΔτVup_l[site_j] = exp(-Δτ*Vup_j[l])
        expmΔτVdn_l[site_j] = exp(-Δτ*Vdn_j[l])
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
        # substract off effect of HS field on Vup and Vdn
        _update_V!(fermion_path_integral_up, hubbard_hs_parameters, i, +1, -1)
        _update_V!(fermion_path_integral_dn, hubbard_hs_parameters, i, -1, -1)
        _update_V!(fermion_path_integral_up, hubbard_hs_parameters, j, +1, -1)
        _update_V!(fermion_path_integral_dn, hubbard_hs_parameters, j, -1, -1)
        # flip HS fields back
        swap!(s_i, s_j)
        # add effect of HS field on Vup and Vdn
        _update_V!(fermion_path_integral_up, hubbard_hs_parameters, i, +1, +1)
        _update_V!(fermion_path_integral_dn, hubbard_hs_parameters, i, -1, +1)
        _update_V!(fermion_path_integral_up, hubbard_hs_parameters, j, +1, +1)
        _update_V!(fermion_path_integral_dn, hubbard_hs_parameters, j, -1, +1)
        # revert propagator matrices
        @fastmath @inbounds for l in eachindex(Bup)
            expmΔτVup_l = Bup[l].expmΔτV::Vector{E}
            expmΔτVdn_l = Bdn[l].expmΔτV::Vector{E}
            expmΔτVup_l[site_i] = exp(-Δτ*Vup_i[l])
            expmΔτVdn_l[site_i] = exp(-Δτ*Vdn_i[l])
            expmΔτVup_l[site_j] = exp(-Δτ*Vup_j[l])
            expmΔτVdn_l[site_j] = exp(-Δτ*Vdn_j[l])
        end
        accepted = false
    end

    return (accepted, logdetGup, sgndetGup, logdetGdn, sgndetGdn)
end



function swap_update!(G::Matrix{T}, logdetG::E, sgndetG::T,
                      hubbard_hs_parameters::AbstractHubbardHS{E};
                      fermion_path_integral::FermionPathIntegral{T,E},
                      fermion_greens_calculator::FermionGreensCalculator{T,E},
                      fermion_greens_calculator_alt::FermionGreensCalculator{T,E},
                      B::Vector{P}, rng::AbstractRNG) where {T<:Number, E<:AbstractFloat, P<:AbstractPropagator{T,E}}

    (; N, U, sites, s, Δτ, Lτ) = hubbard_hs_parameters
    G′ = fermion_greens_calculator_alt.G′

    # make sure stabilization frequencies match
    if fermion_greens_calculator.n_stab != fermion_greens_calculator_alt.n_stab
        resize!(fermion_greens_calculator_alt, fermion_greens_calculator.n_stab)
    end

    # ranomly pick two sites with Hubbard U interaction on them
    i = rand(rng, 1:N)
    j = i
    while j==i
        j = rand(rng, 1:N)
    end

    # get the site index associted with each Hubbard U
    site_i = sites[i]
    site_j = sites[j]

    # get the HS fields associated with each site
    s_i = @view s[i,:]
    s_j = @view s[j,:]
    V_i = @view fermion_path_integral.V[site_i, :]
    V_j = @view fermion_path_integral.V[site_j, :]

    # calculate the initial bosonic action
    Sb = _bosonic_action(hubbard_hs_parameters)

    # substract off effect of HS field on Vup and Vdn
    _update_V!(fermion_path_integral, hubbard_hs_parameters, i, +1, -1)
    _update_V!(fermion_path_integral, hubbard_hs_parameters, j, +1, -1)

    # swap the HS fields
    swap!(s_i, s_j)

    # add effect of HS field on Vup and Vdn
    _update_V!(fermion_path_integral, hubbard_hs_parameters, i, +1, +1)
    _update_V!(fermion_path_integral, hubbard_hs_parameters, j, +1, +1)

    # calculate the final bosonic action
    Sb′ = _bosonic_action(hubbard_hs_parameters)

    # calculate the change in the bosonic action
    ΔSb = Sb′ - Sb

    # update propagator matrices
    @fastmath @inbounds for l in eachindex(B)
        expmΔτV_l = B[l].expmΔτV::Vector{E}
        expmΔτV_l[site_i] = exp(-Δτ*V_i[l])
        expmΔτV_l[site_j] = exp(-Δτ*V_j[l])
    end

    # calculate new Green's function matrices and determinant of new Green's function matrix
    logdetG′, sgndetG′ = calculate_equaltime_greens!(G′, fermion_greens_calculator_alt, B)

    # calculate acceptance probability P = exp(-ΔS_b)⋅|det(Gup)/det(Gup′)|⋅|det(Gdn)/det(Gdn′)|
    #                                    = exp(-ΔS_b)⋅|det(Mup′)/det(Mup)|⋅|det(Mdn′)/det(Mdn)|
    if isfinite(logdetGup′) && isfinite(logdetGdn′)
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
        # substract off effect of HS field on V
        _update_V!(fermion_path_integral, hubbard_hs_parameters, i, +1, -1)
        _update_V!(fermion_path_integral, hubbard_hs_parameters, j, +1, -1)
        # flip HS fields back
        swap!(s_i, s_j)
        # add effect of HS field on V
        _update_V!(fermion_path_integral, hubbard_hs_parameters, i, +1, +1)
        _update_V!(fermion_path_integral, hubbard_hs_parameters, j, +1, +1)
        # revert propagator matrices
        @fastmath @inbounds for l in eachindex(B)
            expmΔτV_l = B[l].expmΔτV::Vector{E}
            expmΔτV_l[site_i] = exp(-Δτ*V_i[l])
            expmΔτV_l[site_j] = exp(-Δτ*V_j[l])
        end
        accepted = false
    end

    return (accepted, logdetG, sgndetG)
end


# for updating V matrix for continuous HS fields
function _update_V!(fermion_path_integral::FermionPathIntegral{T,E},
                    hubbard_hs_parameters::AbstractHubbardHS{E},
                    i::Int, σ::Int, sgn::Int) where {T, E<:AbstractFloat}

    (; s, U, sites, Δτ) = hubbard_hs_parameters
    (; V) = fermion_path_integral

    # get the corresponding site/orbital associated with the HS field
    site = sites[i]

    # iterate over imaginary time axis
    for l in axes(s,2)
        # upate fermion path integral
        V[site,l] = V[site,l] + sgn * (-σ*heaviside(U[i]) - heaviside(-U[i]))/Δτ * eval_a(i, l, hubbard_hs_parameters)
    end

    return nothing
end