@doc raw"""
    measure_onsite_energy(
        tight_binding_parameters::TightBindingParameters{T,E},
        G::Matrix{H}, orbital_id::Int
    ) where {H<:Number, T<:Number, E<:AbstractFloat}

Measure and return the on-site energy ``\epsilon_\textrm{on-site} = (\epsilon - \mu)\langle \hat{n}_\sigma \rangle``
for the `orbital_id` in the unit cell.
"""
function measure_onsite_energy(
    tight_binding_parameters::TightBindingParameters{T,E},
    G::Matrix{H}, orbital_id::Int
) where {H<:Number, T<:Number, E<:AbstractFloat}

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
        G::Matrix{H}, hopping_id::Int
    ) where {H<:Number, T<:Number, E<:AbstractFloat}

Calculate the average bare hopping energy
``\epsilon_{\rm hopping} = -\langle t_{\langle i,j \rangle} \hat{c}^\dagger_{\sigma,i} \hat{c}_{\sigma,j} + {\rm h.c.} \rangle``
for the hopping defined by the `hopping_id`.
"""
function measure_bare_hopping_energy(
    tight_binding_parameters::TightBindingParameters{T,E},
    G::Matrix{H}, hopping_id::Int
) where {H<:Number, T<:Number, E<:AbstractFloat}

    (; t, neighbor_table, bond_slices) = tight_binding_parameters

    # initialize hopping energy to zero
    h = zero(T)

    # get the neighbor table associated with the bond/hopping in question
    nt = @view neighbor_table[:, bond_slices[hopping_id]]

    # get the hopping associated with the bond/hopping in question
    t′ = @view t[bond_slices[hopping_id]]

    # iterate over each bond/hopping
    @fastmath @inbounds for n in axes(nt, 2)
        # hopping from site j to site i: -⟨t₀[i,j]⋅cᵀ[i]c[j] + (t₀[i,j])ᵀ⋅cᵀ[i]⋅c[j]⟩
        j = nt[1,n]
        i = nt[2,n]
        # hopping amplitude from site j to site i: ⟨cᵀ[i]c[j]⟩ = -⟨c[j]cᵀ[i]⟩ = -G[j,i]
        hij = -G[j,i]
        # hopping amplitude from site i to site j: ⟨cᵀ[j]c[i]⟩ = -⟨c[i]cᵀ[j]⟩ = -G[i,j]
        hji = -G[i,j]
        # calculate the bare hopping energy
        h += -t′[n] * hij + conj(-t′[n]) * hji
    end

    # normalize the measurement
    h /= length(t′)

    return h
end

@doc raw"""
    measure_hopping_energy(
        tight_binding_parameters::TightBindingParameters{T,E},
        fermion_path_integral::FermionPathIntegral{H},
        G::Matrix{H}, hopping_id::Int
    ) where {H<:Number, T<:Number, E<:AbstractFloat}

Calculate the average hopping energy
``\epsilon_{\rm hopping} = -\langle t_{l,\langle i,j \rangle} \hat{c}^\dagger_{\sigma,i} \hat{c}_{\sigma,j} + {\rm h.c.} \rangle``
for the hopping defined by the the `hopping_id`.
"""
function measure_hopping_energy(
    tight_binding_parameters::TightBindingParameters{T,E},
    fermion_path_integral::FermionPathIntegral{H},
    G::Matrix{H}, hopping_id::Int
) where {H<:Number, T<:Number, E<:AbstractFloat}

    (; neighbor_table, bond_slices, bond_ids) = tight_binding_parameters
    (; t, Lτ) = fermion_path_integral

    # initialize hopping energy to zero
    h = zero(T)

    # get the neighbor table associated with the bond/hopping in question
    nt = @view neighbor_table[:, bond_slices[hopping_id]]

    # get the hopping associated with the bond/hopping in question
    t′ = @view t[bond_slices[hopping_id], Lτ]

    # iterate over each bond/hopping
    @fastmath @inbounds for n in axes(nt, 2)
        # hopping from site j to site i: -⟨t₀[i,j]⋅cᵀ[i]c[j] + (t₀[i,j])ᵀ⋅cᵀ[i]⋅c[j]⟩
        j = nt[1,n]
        i = nt[2,n]
        # hopping amplitude from site j to site i: ⟨cᵀ[i]c[j]⟩ = -⟨c[j]cᵀ[i]⟩ = -G[j,i]
        hij = -G[j,i]
        # hopping amplitude from site i to site j: ⟨cᵀ[j]c[i]⟩ = -⟨c[i]cᵀ[j]⟩ = -G[i,j]
        hji = -G[i,j]
        # calculate the bare hopping energy
        h += -t′[n] * hij + conj(-t′[n]) * hji
    end

    # normalize the measurement
    h /= length(t′)

    return h
end

@doc raw"""
    measure_hopping_amplitude(
        tight_binding_parameters::TightBindingParameters{T,E},
        fermion_path_integral::FermionPathIntegral{H},
        hopping_id::Int
    ) where {H<:Number, T<:Number, E<:AbstractFloat}

Calculate the average hopping amplitude for the hopping defined by the `hopping_id`.
"""
function measure_hopping_amplitude(
    tight_binding_parameters::TightBindingParameters{T,E},
    fermion_path_integral::FermionPathIntegral{H},
    hopping_id::Int
) where {H<:Number, T<:Number, E<:AbstractFloat}

    (; neighbor_table, bond_slices, bond_ids) = tight_binding_parameters
    (; t, Lτ) = fermion_path_integral

    # initialize hopping energy to zero
    t_avg = zero(T)

    # get the hopping associated with the bond/hopping in question
    t′ = @view t[bond_slices[hopping_id], :]

    # normalize the measurement
    t_avg += mean(t′)

    return t_avg
end

@doc raw"""
    measure_hopping_inversion(
        tight_binding_parameters::TightBindingParameters{T,E},
        fermion_path_integral::FermionPathIntegral{H},
        hopping_id::Int
    ) where {H<:Number, T<:Number, E<:AbstractFloat}

Measure the fraction of time the sign of the instantaneous modulated hopping amplitude ``t_{l,(\mathbf{i},\nu),(\mathbf{j},\gamma)}``
is inverted relative to the bare hopping amplitude ``t_{(\mathbf{i},\nu),(\mathbf{j},\gamma)}``, where ``l`` is the
imaginary time-slice index.
"""
function measure_hopping_inversion(
    tight_binding_parameters::TightBindingParameters{T,E},
    fermion_path_integral::FermionPathIntegral{H},
    hopping_id::Int
) where {H<:Number, T<:Number, E<:AbstractFloat}

    (; neighbor_table, bond_slices, bond_ids) = tight_binding_parameters
    (; Lτ) = fermion_path_integral

    # bare hopping amplitudes
    t0 = tight_binding_parameters.t

    # modulated hopping amplitudes
    tτ = fermion_path_integral.t

    # instantaneous hopping inversion fraction
    hopping_inversion = zero(E)

    # get the bare hopping amplitudes associated with bond
    t0′ = @view t0[bond_slices[hopping_id]]

    # get modulated hopping amplitudes associated with bond
    tτ′ = @view tτ[bond_slices[hopping_id], :]

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

    return hopping_inversion
end

@doc raw"""
    measure_hopping_inversion_avg(
        tight_binding_parameters::TightBindingParameters{T,E},
        fermion_path_integral::FermionPathIntegral{H},
        hopping_id::Int
    ) where {H<:Number, T<:Number, E<:AbstractFloat}

Measure the fraction of time the sign of the imaginary-time averaged modulated hopping amplitude
``\bar{t}_{(\mathbf{i},\nu),(\mathbf{j},\gamma)}`` is inverted relative to the bare hopping amplitude
``t_{(\mathbf{i},\nu),(\mathbf{j},\gamma)}``.
"""
function measure_hopping_inversion_avg(
    tight_binding_parameters::TightBindingParameters{T,E},
    fermion_path_integral::FermionPathIntegral{H},
    hopping_id::Int
) where {H<:Number, T<:Number, E<:AbstractFloat}

    (; neighbor_table, bond_slices, bond_ids) = tight_binding_parameters
    (; Lτ) = fermion_path_integral

    # bare hopping amplitudes
    t0 = tight_binding_parameters.t

    # modulated hopping amplitudes
    tτ = fermion_path_integral.t

    # instantaneous hopping inversion fraction
    hopping_inversion_avg = zero(E)

    # get the bare hopping amplitudes associated with bond
    t0′ = @view t0[bond_slices[hopping_id]]

    # get modulated hopping amplitudes associated with bond
    tτ′ = @view tτ[bond_slices[hopping_id], :]

    # iterate over each specific hopping/bond
    for h in eachindex(t0′)

        # calculate average modulated hopping amplitude
        thτ′ = @view tτ′[h,:]
        thτ′_avg = mean(thτ′)
        hopping_inversion_avg += !(sign(real(t0′[h])) == sign(real(thτ′_avg)))
    end

    # normalize fraction
    hopping_inversion_avg /= length(t0′)

    return hopping_inversion_avg
end