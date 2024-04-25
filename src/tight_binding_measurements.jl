@doc raw"""
    measure_onsite_energy(
        tight_binding_parameters::TightBindingParameters{T,E},
        G::Matrix{T}, orbital_id::Int
    ) where {T<:Number, E<:AbstractFloat}

Measure and return the on-site energy ``\epsilon_\textrm{on-site} = (\epsilon - \mu)\langle \hat{n}_\sigma \rangle``
for the `orbital_id` in the unit cell.
"""
function measure_onsite_energy(
    tight_binding_parameters::TightBindingParameters{T,E},
    G::Matrix{T}, orbital_id::Int
) where {T<:Number, E<:AbstractFloat}

    (; ϵ, μ, norbital) = tight_binding_parameters

    # initialize energy to zero
    e = zero(E)

    # number of orbitals in lattice
    Norbital = size(G, 1)

    # number of unit cells in lattice
    Nunitcells = Norbital ÷ norbital

    # iterate over unit cells
    for u in 1:Nunitcells
        # get the site in the unit cell associated with specified orbital species
        i = (u-1)*norbital + orbital_id
        # calculate on-site energy
        e += (ϵ[i] - μ) * (1 - G[i,i])
    end
    # normalize measurement
    e /= Nunitcells

    return e
end


@doc raw"""
    measure_bare_hopping_energy(
        tight_binding_parameters::TightBindingParameters{T,E},
        G::Matrix{T}, bond_id::Int
    ) where {T<:Number, E<:AbstractFloat}

Calculate the average bare hopping energy
``\epsilon_{\rm hopping} = -\langle t_{\langle i,j \rangle} \hat{c}^\dagger_{\sigma,i} \hat{c}_{\sigma,j} + {\rm h.c.} \rangle``
for the hopping defined by the bond corresponding to `bond_id`.
"""
function measure_bare_hopping_energy(
    tight_binding_parameters::TightBindingParameters{T,E},
    G::Matrix{T}, bond_id::Int
) where {T<:Number, E<:AbstractFloat}

    (; t, neighbor_table, bond_slices, bond_ids) = tight_binding_parameters

    # initialize hopping energy to zero
    h = zero(T)

    # check if hopping associated with bond ID
    if bond_id in bond_ids

        # get the bond slice index associated with bond id
        bond_id_index = findfirst(i -> i==bond_id, bond_ids)

        # get the neighbor table associated with the bond/hopping in question
        nt = @view neighbor_table[:, bond_slices[bond_id_index]]

        # get the hopping associated with the bond/hopping in question
        t′ = @view t[bond_slices[bond_id_index]]

        # iterate over each bond/hopping
        @fastmath @inbounds for n in axes(nt, 2)
            # get the pair of hoppings
            j = nt[1,n] # annihilate electron on orbital j
            i = nt[2,n] # create electron on orbital i
            # calculate the hopping energy
            hij = -G[j,i] # hopping amplitude from site j to site i
            h += (-t′[n] * hij) + conj(-t′[n] * hij)
        end

        # noramalize the measurement
        h /= length(t′)
    end

    return h
end

@doc raw"""
    measure_hopping_energy(
        tight_binding_parameters::TightBindingParameters{T,E},
        fermion_path_integral::FermionPathIntegral{T,E},
        G::Matrix{T}, bond_id::Int
    ) where {T<:Number, E<:AbstractFloat}

Calculate the average hopping energy
``\epsilon_{\rm hopping} = -\langle t_{l,\langle i,j \rangle} \hat{c}^\dagger_{\sigma,i} \hat{c}_{\sigma,j} + {\rm h.c.} \rangle``
for the hopping defined by the bond corresponding to `bond_id`.
"""
function measure_hopping_energy(
    tight_binding_parameters::TightBindingParameters{T,E},
    fermion_path_integral::FermionPathIntegral{T,E},
    G::Matrix{T}, bond_id::Int
) where {T<:Number, E<:AbstractFloat}

    (; neighbor_table, bond_slices, bond_ids) = tight_binding_parameters
    (; t, Lτ) = fermion_path_integral

    # initialize hopping energy to zero
    h = zero(T)

    # check if hopping associated with bond ID
    if bond_id in bond_ids

        # get the bond slice index associated with bond id
        bond_id_index = findfirst(i -> i==bond_id, bond_ids)

        # get the neighbor table associated with the bond/hopping in question
        nt = @view neighbor_table[:, bond_slices[bond_id_index]]

        # get the hopping associated with the bond/hopping in question
        t′ = @view t[bond_slices[bond_id_index], Lτ]

        # iterate over each bond/hopping
        @fastmath @inbounds for n in axes(nt, 2)
            # get the pair of hoppings
            j = nt[1,n] # annihilate electron on orbital j
            i = nt[2,n] # create electron on orbital i
            # calculate the hopping energy
            hij = -G[j,i] # hopping amplitude from site j to site i
            h += (-t′[n] * hij) + conj(-t′[n] * hij)
        end

        # noramalize the measurement
        h /= length(t′)
    end

    return h
end

@doc raw"""
    measure_hopping_amplitude(
        tight_binding_parameters::TightBindingParameters{T,E},
        fermion_path_integral::FermionPathIntegral{T,E},
        bond_id::Int
    ) where {T<:Number, E<:AbstractFloat}

Calculate the average hopping amplitude for the hopping defined by the bond corresponding to `bond_id`.
"""
function measure_hopping_amplitude(
    tight_binding_parameters::TightBindingParameters{T,E},
    fermion_path_integral::FermionPathIntegral{T,E},
    bond_id::Int
) where {T<:Number, E<:AbstractFloat}

    (; neighbor_table, bond_slices, bond_ids) = tight_binding_parameters
    (; t, Lτ) = fermion_path_integral

    # initialize hopping energy to zero
    t_avg = zero(T)

    # check if hopping associated with bond ID
    if bond_id in bond_ids

        # get the bond slice index associated with bond id
        bond_id_index = findfirst(i -> i==bond_id, bond_ids)

        # get the hopping associated with the bond/hopping in question
        t′ = @view t[bond_slices[bond_id_index], :]

        # noramalize the measurement
        t_avg += mean(t′)
    end

    return t_avg
end

@doc raw"""
    measure_hopping_inversion(
        tight_binding_parameters::TightBindingParameters{T,E},
        fermion_path_integral::FermionPathIntegral{T,E},
        bond_id::Int
    ) where {T<:Number, E<:AbstractFloat}

Measure the fraction of time the sign of the instaneous modulated hopping ampltiude ``t_{l,(\mathbf{i},\nu),(\mathbf{j},\gamma)}``
is inverted relative to the bare hopping amplitude ``t_{(\mathbf{i},\nu),(\mathbf{j},\gamma)}``, where ``l`` is the
imaginary time-slice index.
"""
function measure_hopping_inversion(
    tight_binding_parameters::TightBindingParameters{T,E},
    fermion_path_integral::FermionPathIntegral{T,E},
    bond_id::Int
) where {T<:Number, E<:AbstractFloat}

    (; neighbor_table, bond_slices, bond_ids) = tight_binding_parameters
    (; Lτ) = fermion_path_integral

    # bare hopping amplitudes
    t0 = tight_binding_parameters.t

    # modulated hopping amplitudes
    tτ = fermion_path_integral.t

    # instaneous hopping inversion fraction
    hopping_inversion = zero(E)

    # check if hopping associated with bond ID
    if bond_id in bond_ids

        # get the bond slice index associated with bond id
        bond_id_index = findfirst(i -> i==bond_id, bond_ids)

        # get the bare hopping amplitudes associated with bond
        t0′ = @view t0[bond_slices[bond_id_index]]

        # get modulated hopping amplitudes associated with bond
        tτ′ = @view tτ[bond_slices[bond_id_index], :]

        # iterate over imaginary time slices
        for l in axes(tτ′,2)
            # iterate over each specific hopping/bond
            for h in eachindex(t0′)
                # detect whether the sign of the hopping was inverted
                hopping_inversion += !(sign(real(t0′[h])) == sign(real(tτ′[h,l])))
            end
        end

        # normalize fraction
        hopping_inversion /= length(tτ′)
    end

    return hopping_inversion
end

@doc raw"""
    measure_hopping_inversion_avg(
        tight_binding_parameters::TightBindingParameters{T,E},
        fermion_path_integral::FermionPathIntegral{T,E},
        bond_id::Int
    ) where {T<:Number, E<:AbstractFloat}

Measure the fraction of time the sign of the imaginary-time averaged modulated hopping ampltiude
``\bar{t}_{(\mathbf{i},\nu),(\mathbf{j},\gamma)}`` is inverted relative to the bare hopping amplitude
``t_{(\mathbf{i},\nu),(\mathbf{j},\gamma)}``.
"""
function measure_hopping_inversion_avg(
    tight_binding_parameters::TightBindingParameters{T,E},
    fermion_path_integral::FermionPathIntegral{T,E},
    bond_id::Int
) where {T<:Number, E<:AbstractFloat}

    (; neighbor_table, bond_slices, bond_ids) = tight_binding_parameters
    (; Lτ) = fermion_path_integral

    # bare hopping amplitudes
    t0 = tight_binding_parameters.t

    # modulated hopping amplitudes
    tτ = fermion_path_integral.t

    # instaneous hopping inversion fraction
    hopping_inversion_avg = zero(E)

    # check if hopping associated with bond ID
    if bond_id in bond_ids

        # get the bond slice index associated with bond id
        bond_id_index = findfirst(i -> i==bond_id, bond_ids)

        # get the bare hopping amplitudes associated with bond
        t0′ = @view t0[bond_slices[bond_id_index]]

        # get modulated hopping amplitudes associated with bond
        tτ′ = @view tτ[bond_slices[bond_id_index], :]

        # iterate over each specific hopping/bond
        for h in eachindex(t0′)

            # calculate average modulated hopping amplitude
            thτ′ = @view tτ′[h,:]
            thτ′_avg = mean(thτ′)
            hopping_inversion_avg += !(sign(real(t0′[h])) == sign(real(thτ′_avg)))
        end

        # normalize fraction
        hopping_inversion_avg /= length(t0′)
    end

    return hopping_inversion_avg
end