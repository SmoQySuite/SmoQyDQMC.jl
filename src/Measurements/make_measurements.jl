#######################################################
## HIGHEST LEVEL/EXPORTED MAKE MEASUREMENTS FUNCTION ##
#######################################################

@doc raw"""
    make_measurements!(
        measurement_container::NamedTuple,
        logdetGup::E, sgndetGup::T, Gup::AbstractMatrix{T},
        Gup_ττ::AbstractMatrix{T}, Gup_τ0::AbstractMatrix{T}, Gup_0τ::AbstractMatrix{T},
        logdetGdn::E, sgndetGdn::T, Gdn::AbstractMatrix{T},
        Gdn_ττ::AbstractMatrix{T}, Gdn_τ0::AbstractMatrix{T}, Gdn_0τ::AbstractMatrix{T};
        # Keyword Arguments Start Here
        fermion_path_integral_up::FermionPathIntegral{T,E},
        fermion_path_integral_dn::FermionPathIntegral{T,E},
        fermion_greens_calculator_up::FermionGreensCalculator{T,E},
        fermion_greens_calculator_dn::FermionGreensCalculator{T,E},
        Bup::Vector{P}, Bdn::Vector{P}, δG_max::E, δG::E, δθ::E,
        model_geometry::ModelGeometry{D,E,N},
        tight_binding_parameters::Union{Nothing, TightBindingParameters{T,E}} = nothing,
        tight_binding_parameters_up::Union{Nothing, TightBindingParameters{T,E}} = nothing,
        tight_binding_parameters_dn::Union{Nothing, TightBindingParameters{T,E}} = nothing,
        coupling_parameters::Tuple
    ) where {T<:Number, E<:AbstractFloat, D, N, P<:AbstractPropagator{T,E}}

Make measurements, including time-displaced correlation and zero Matsubara frequency measurements.
This method also returns `(logdetGup, sgndetGup, logdetGdn, sgndetGdn, δG, δθ)`.
Note that either the keywork `tight_binding_parameters` needs to be specified, or
`tight_binding_parameters_up` and `tight_binding_parameters_dn` both need to be specified.
"""
function make_measurements!(
    measurement_container::NamedTuple,
    logdetGup::E, sgndetGup::T, Gup::AbstractMatrix{T},
    Gup_ττ::AbstractMatrix{T}, Gup_τ0::AbstractMatrix{T}, Gup_0τ::AbstractMatrix{T},
    logdetGdn::E, sgndetGdn::T, Gdn::AbstractMatrix{T},
    Gdn_ττ::AbstractMatrix{T}, Gdn_τ0::AbstractMatrix{T}, Gdn_0τ::AbstractMatrix{T};
    # Keyword Arguments Start Here
    fermion_path_integral_up::FermionPathIntegral{T,E},
    fermion_path_integral_dn::FermionPathIntegral{T,E},
    fermion_greens_calculator_up::FermionGreensCalculator{T,E},
    fermion_greens_calculator_dn::FermionGreensCalculator{T,E},
    Bup::Vector{P}, Bdn::Vector{P}, δG_max::E, δG::E, δθ::E,
    model_geometry::ModelGeometry{D,E,N},
    tight_binding_parameters::Union{Nothing, TightBindingParameters{T,E}} = nothing,
    tight_binding_parameters_up::Union{Nothing, TightBindingParameters{T,E}} = nothing,
    tight_binding_parameters_dn::Union{Nothing, TightBindingParameters{T,E}} = nothing,
    coupling_parameters::Tuple
) where {T<:Number, E<:AbstractFloat, D, N, P<:AbstractPropagator{T,E}}

    # extract temporary storage vectors
    (; time_displaced_correlations, equaltime_correlations, a, a′, a″) = measurement_container

    # assign spin-up and spin-down tight-binding parameters if necessary
    if !isnothing(tight_binding_parameters)
        tight_binding_parameters_up = tight_binding_parameters
        tight_binding_parameters_dn = tight_binding_parameters
    end

    # calculate sign
    sgn = sgndetGup * sgndetGdn
    sgn /= abs(sgn) # normalize just to be cautious

    # make global measurements
    global_measurements = measurement_container.global_measurements
    make_global_measurements!(
        global_measurements,
        tight_binding_parameters_up,
        tight_binding_parameters_dn,
        coupling_parameters,
        Gup, logdetGup, sgndetGup,
        Gdn, logdetGdn, sgndetGdn
    )

    # make local measurements
    local_measurements = measurement_container.local_measurements
    make_local_measurements!(
        local_measurements,
        Gup, Gdn, sgn,
        model_geometry,
        tight_binding_parameters_up, tight_binding_parameters_dn,
        fermion_path_integral_up, fermion_path_integral_dn,
        coupling_parameters
    )

    # initialize green's function matrices G(τ,0), G(0,τ) and G(τ,τ) based on G(0,0)
    initialize_unequaltime_greens!(Gup_τ0, Gup_0τ, Gup_ττ, Gup)
    initialize_unequaltime_greens!(Gdn_τ0, Gdn_0τ, Gdn_ττ, Gdn)

    # make equal-time correlation measurements
    make_equaltime_measurements!(
        equaltime_correlations, sgn,
        Gup, Gup_ττ, Gup_τ0, Gup_0τ,
        Gdn, Gdn_ττ, Gdn_τ0, Gdn_0τ,
        model_geometry, tight_binding_parameters_up, tight_binding_parameters_dn,
        fermion_path_integral_up, fermion_path_integral_dn
    )

    # if there are time-displaced measurements to make
    if length(time_displaced_correlations) > 0

        # make time-displaced measuresurements for τ = l⋅Δτ = 0
        make_time_displaced_measurements!(
            time_displaced_correlations, 0, sgn,
            Gup, Gup_ττ, Gup_τ0, Gup_0τ, Gdn, Gdn_ττ, Gdn_τ0, Gdn_0τ,
            model_geometry, tight_binding_parameters_up, tight_binding_parameters_dn,
            fermion_path_integral_up, fermion_path_integral_dn
        )

        # iterate over imaginary time slice
        for l in fermion_greens_calculator_up

            # Propagate Green's function matrices to current imaginary time slice
            propagate_unequaltime_greens!(Gup_τ0, Gup_0τ, Gup_ττ, fermion_greens_calculator_up, Bup)
            propagate_unequaltime_greens!(Gdn_τ0, Gdn_0τ, Gdn_ττ, fermion_greens_calculator_dn, Bdn)

            # make time-displaced measuresurements for τ = l⋅Δτ
            make_time_displaced_measurements!(
                time_displaced_correlations, l, sgn,
                Gup, Gup_ττ, Gup_τ0, Gup_0τ, Gdn, Gdn_ττ, Gdn_τ0, Gdn_0τ,
                model_geometry, tight_binding_parameters_up, tight_binding_parameters_dn,
                fermion_path_integral_up, fermion_path_integral_dn
            )

            # Periodically re-calculate the Green's function matrix for numerical stability.
            logdetGup, sgndetGup, δGup, δθup = stabilize_unequaltime_greens!(
                Gup_τ0, Gup_0τ, Gup_ττ, logdetGup, sgndetGup,
                fermion_greens_calculator_up, Bup, update_B̄=false
            )
            logdetGdn, sgndetGdn, δGdn, δθdn = stabilize_unequaltime_greens!(
                Gdn_τ0, Gdn_0τ, Gdn_ττ, logdetGdn, sgndetGdn,
                fermion_greens_calculator_dn, Bdn, update_B̄=false
            )

            # record the max errors
            δG = maximum((δG, δGup, δGdn))
            δθ = maximum(abs, (δθ, δθup, δθdn))

            # Keep up and down spin Green's functions synchronized as iterating over imaginary time.
            iterate(fermion_greens_calculator_dn, fermion_greens_calculator_up.forward)
        end
    end

    # measure equal-time phonon greens function
    if haskey(equaltime_correlations, "phonon_greens")

        # get the electron-phonon parameters
        indx = findfirst(i -> typeof(i) <: ElectronPhononParameters, coupling_parameters)

        # measure phonon green's function
        measure_equaltime_phonon_greens!(equaltime_correlations["phonon_greens"], coupling_parameters[indx], model_geometry, sgn, a, a′, a″)
    end

    # measure time-displaced phonon greens function
    if haskey(time_displaced_correlations, "phonon_greens")

        # get the electron-phonon parameters
        indx = findfirst(i -> typeof(i) <: ElectronPhononParameters, coupling_parameters)

        # measure phonon green's function
        measure_time_displaced_phonon_greens!(time_displaced_correlations["phonon_greens"], coupling_parameters[indx], model_geometry, sgn, a, a′, a″)
    end

    return (logdetGup, sgndetGup, logdetGdn, sgndetGdn, δG, δθ)
end

@doc raw"""
    make_measurements!(
        measurement_container::NamedTuple,
        logdetG::E, sgndetG::T, G::AbstractMatrix{T},
        G_ττ::AbstractMatrix{T}, G_τ0::AbstractMatrix{T}, G_0τ::AbstractMatrix{T};
        # Keyword Arguments Start Here
        fermion_path_integral::FermionPathIntegral{T,E},
        fermion_greens_calculator::FermionGreensCalculator{T,E},
        B::Vector{P}, δG_max::E, δG::E, δθ::E,
        model_geometry::ModelGeometry{D,E,N},
        tight_binding_parameters::TightBindingParameters{T,E},
        coupling_parameters::Tuple
    ) where {T<:Number, E<:AbstractFloat, D, N, P<:AbstractPropagator{T,E}}

Make measurements, including time-displaced correlation and zero Matsubara frequency measurements.
This method also returns `(logdetG, sgndetG, δG, δθ)`.
"""
function make_measurements!(
    measurement_container::NamedTuple,
    logdetG::E, sgndetG::T, G::AbstractMatrix{T},
    G_ττ::AbstractMatrix{T}, G_τ0::AbstractMatrix{T}, G_0τ::AbstractMatrix{T};
    # Keyword Arguments Start Here
    fermion_path_integral::FermionPathIntegral{T,E},
    fermion_greens_calculator::FermionGreensCalculator{T,E},
    B::Vector{P}, δG_max::E, δG::E, δθ::E,
    model_geometry::ModelGeometry{D,E,N},
    tight_binding_parameters::TightBindingParameters{T,E},
    coupling_parameters::Tuple
) where {T<:Number, E<:AbstractFloat, D, N, P<:AbstractPropagator{T,E}}

    # extract temporary storage vectors
    (; time_displaced_correlations, equaltime_correlations, a, a′, a″) = measurement_container

    # calculate sign
    sgn = sgndetG^2
    sgn /= abs(sgn) # normalize just to be cautious

    # make global measurements
    global_measurements = measurement_container.global_measurements
    make_global_measurements!(
        global_measurements,
        tight_binding_parameters,
        tight_binding_parameters,
        coupling_parameters,
        G, logdetG, sgndetG,
        G, logdetG, sgndetG
    )

    # make local measurements
    local_measurements = measurement_container.local_measurements
    make_local_measurements!(
        local_measurements,
        G, G, sgn, model_geometry,
        tight_binding_parameters, tight_binding_parameters,
        fermion_path_integral, fermion_path_integral,
        coupling_parameters
    )

    # initialize green's function matrices G(τ,0), G(0,τ) and G(τ,τ) based on G(0,0)
    initialize_unequaltime_greens!(G_τ0, G_0τ, G_ττ, G)

    # make equal-time correlation measurements
    make_equaltime_measurements!(
        equaltime_correlations, sgn,
        G, G_ττ, G_τ0, G_0τ, G, G_ττ, G_τ0, G_0τ,
        model_geometry, tight_binding_parameters, tight_binding_parameters,
        fermion_path_integral, fermion_path_integral
    )

    # if there are time-displaced measurements to make
    if length(time_displaced_correlations) > 0

        # make time-displaced measuresurements of τ = 0
        make_time_displaced_measurements!(
            time_displaced_correlations, 0, sgn,
            G, G_ττ, G_τ0, G_0τ, G, G_ττ, G_τ0, G_0τ,
            model_geometry, tight_binding_parameters, tight_binding_parameters,
            fermion_path_integral, fermion_path_integral
        )

        # iterate over imaginary time slice
        for l in fermion_greens_calculator

            # Propagate Green's function matrices to current imaginary time slice
            propagate_unequaltime_greens!(G_τ0, G_0τ, G_ττ, fermion_greens_calculator, B)

            # make time-displaced measuresurements of τ = l⋅Δτ
            make_time_displaced_measurements!(
                time_displaced_correlations, l, sgn,
                G, G_ττ, G_τ0, G_0τ, G, G_ττ, G_τ0, G_0τ,
                model_geometry, tight_binding_parameters, tight_binding_parameters,
                fermion_path_integral, fermion_path_integral
            )

            # Periodically re-calculate the Green's function matrix for numerical stability.
            logdetG, sgndetG, δG′, δθ = stabilize_unequaltime_greens!(G_τ0, G_0τ, G_ττ, logdetG, sgndetG, fermion_greens_calculator, B, update_B̄=false)

            # record maximum stablization error
            δG = max(δG′, δG)
        end
    end

    # measure equal-time phonon greens function
    if haskey(equaltime_correlations, "phonon_greens")

        # get the electron-phonon parameters
        indx = findfirst(i -> typeof(i) <: ElectronPhononParameters, coupling_parameters)

        # measure phonon green's function
        measure_equaltime_phonon_greens!(equaltime_correlations["phonon_greens"], coupling_parameters[indx], model_geometry, sgn, a, a′, a″)
    end

    # measure time-displaced phonon greens function
    if haskey(time_displaced_correlations, "phonon_greens")

        # get the electron-phonon parameters
        indx = findfirst(i -> typeof(i) <: ElectronPhononParameters, coupling_parameters)

        # measure phonon green's function
        measure_time_displaced_phonon_greens!(time_displaced_correlations["phonon_greens"], coupling_parameters[indx], model_geometry, sgn, a, a′, a″)
    end

    return (logdetG, sgndetG, δG, δθ)
end


##############################
## MAKE GLOBAL MEASUREMENTS ##
##############################

# make global measurements
function make_global_measurements!(
    global_measurements::Dict{String, Complex{E}},
    tight_binding_parameters_up::TightBindingParameters{T,E},
    tight_binding_parameters_dn::TightBindingParameters{T,E},
    coupling_parameters::Tuple,
    Gup::AbstractMatrix{T}, logdetGup::T, sgndetGup::T,
    Gdn::AbstractMatrix{T}, logdetGdn::T, sgndetGdn::T,
) where {T<:Number, E<:AbstractFloat}

    # number of orbitals in lattice
    N = size(Gup, 1)

    # measure the sign
    sgn = sgndetGup * sgndetGdn
    sgn /= abs(sgn) # normalize just to be cautious
    global_measurements["sgn"] += sgn

    # measure the spin resolved sign
    global_measurements["sgndetGup"] += sgndetGup
    global_measurements["sgndetGdn"] += sgndetGdn
    global_measurements["sgndetG"]   += (sgndetGup + sgndetGdn)/2

    # measure log|det(G)|
    global_measurements["logdetGup"] += logdetGup
    global_measurements["logdetGdn"] += logdetGdn

    # measure fermionic action
    Sf = logdetGup + logdetGdn
    global_measurements["action_fermionic"] += Sf

    # measure bosonic action
    Sb = zero(E)
    for i in eachindex(coupling_parameters)
        Sb += bosonic_action(coupling_parameters[i])
    end
    global_measurements["action_bosonic"] += Sb

    # measure total action
    S = Sb + Sf
    global_measurements["action_total"] += S

    # measure average density
    nup = measure_n(Gup)
    ndn = measure_n(Gdn)
    global_measurements["density_up"] += sgn * nup
    global_measurements["density_dn"] += sgn * ndn
    global_measurements["density"] += sgn * (nup + ndn)

    # measure double occupancy
    global_measurements["double_occ"] += sgn * measure_double_occ(Gup, Gdn)

    # measure ⟨N²⟩
    global_measurements["Nsqrd"] += sgn * measure_Nsqrd(Gup, Gdn)

    # measure chemical potential
    global_measurements["chemical_potential"] += tight_binding_parameters_up.μ

    return nothing
end


#############################
## MAKE LOCAL MEASUREMENTS ##
#############################

# make local measurements
function make_local_measurements!(
    local_measurements::Dict{String, Vector{Complex{E}}},
    Gup::AbstractMatrix{T}, Gdn::AbstractMatrix{T}, sgn::T,
    model_geometry::ModelGeometry{D,E,N},
    tight_binding_parameters_up::TightBindingParameters{T,E},
    tight_binding_parameters_dn::TightBindingParameters{T,E},
    fermion_path_integral_up::FermionPathIntegral{T,E},
    fermion_path_integral_dn::FermionPathIntegral{T,E},
    coupling_parameters::Tuple
) where {T<:Number, E<:AbstractFloat, D, N}

    # number of orbitals per unit cell
    unit_cell = model_geometry.unit_cell::UnitCell{D,E,N}
    norbital = unit_cell.n

    # iterate over orbital species
    for n in 1:norbital
        # measure density
        nup = measure_n(Gup, n, unit_cell)
        ndn = measure_n(Gdn, n, unit_cell)
        local_measurements["density_up"][n] += sgn * nup
        local_measurements["density_dn"][n] += sgn * ndn
        local_measurements["density"][n] += sgn * (nup + ndn)
        # measure double occupancy
        local_measurements["double_occ"][n] += sgn * measure_double_occ(Gup, Gdn, n, unit_cell)
    end

    # make tight-binding measurements
    make_local_measurements!(local_measurements, Gup, Gdn, sgn, model_geometry,
                             tight_binding_parameters_up, tight_binding_parameters_dn,
                             fermion_path_integral_up, fermion_path_integral_dn)

    # make local measurements associated with couplings
    for coupling_parameter in coupling_parameters
        make_local_measurements!(local_measurements, Gup, Gdn, sgn, model_geometry,
                                 coupling_parameter, tight_binding_parameters_up, tight_binding_parameters_dn)
    end
    
    return nothing
end

# make local measurements associated with tight-binding model
function make_local_measurements!(
    local_measurements::Dict{String, Vector{Complex{E}}},
    Gup::AbstractMatrix{T}, Gdn::AbstractMatrix{T}, sgn::T,
    model_geometry::ModelGeometry{D,E,N},
    tight_binding_parameters_up::TightBindingParameters{T,E},
    tight_binding_parameters_dn::TightBindingParameters{T,E},
    fermion_path_integral_up::FermionPathIntegral{T,E},
    fermion_path_integral_dn::FermionPathIntegral{T,E}
) where {T<:Number, E<:AbstractFloat, D, N}

    # number of orbitals per unit cell
    norbital = tight_binding_parameters_up.norbital

    # number of types of hopping
    bond_ids = tight_binding_parameters_up.bond_ids
    nhopping = length(tight_binding_parameters_up.bond_ids)

    # measure on-site energy
    for n in 1:norbital
        eup = sgn * measure_onsite_energy(tight_binding_parameters_up, Gup, n)
        edn = sgn * measure_onsite_energy(tight_binding_parameters_dn, Gdn, n)
        e = eup + edn
        local_measurements["onsite_energy_up"][n] += eup
        local_measurements["onsite_energy_dn"][n] += edn
        local_measurements["onsite_energy"][n] += e
    end

    # measure hopping energy
    if nhopping > 0
        for n in 1:nhopping

            # get the bond ID corresponding to the hopping
            bond_id = bond_ids[n]

            # measure bare hopping energy
            hup = sgn * measure_bare_hopping_energy(tight_binding_parameters_up, Gup, bond_id)
            hdn = sgn * measure_bare_hopping_energy(tight_binding_parameters_dn, Gdn, bond_id)
            h = hup + hdn
            local_measurements["bare_hopping_energy_up"][n] += hup
            local_measurements["bare_hopping_energy_dn"][n] += hdn
            local_measurements["bare_hopping_energy"][n] += h

            # measure hopping amplitude
            hup = sgn * measure_hopping_energy(tight_binding_parameters_up, fermion_path_integral_up, Gup, bond_id)
            hdn = sgn * measure_hopping_energy(tight_binding_parameters_dn, fermion_path_integral_up, Gdn, bond_id)
            h = hup + hdn
            local_measurements["hopping_energy_up"][n] += hup
            local_measurements["hopping_energy_dn"][n] += hdn
            local_measurements["hopping_energy"][n] += h

            # measure hopping amplitude
            tup = sgn * measure_hopping_amplitude(tight_binding_parameters_up, fermion_path_integral_up, bond_id)
            tdn = sgn * measure_hopping_amplitude(tight_binding_parameters_dn, fermion_path_integral_up, bond_id)
            tn = (tup + tup)/2
            local_measurements["hopping_amplitude_up"][n] += tup
            local_measurements["hopping_amplitude_dn"][n] += tdn
            local_measurements["hopping_amplitude"][n] += tn

            # measure hopping inversion
            tup = sgn * measure_hopping_inversion(tight_binding_parameters_up, fermion_path_integral_up, bond_id)
            tdn = sgn * measure_hopping_inversion(tight_binding_parameters_dn, fermion_path_integral_up, bond_id)
            tn = (tup + tup)/2
            local_measurements["hopping_inversion_up"][n] += tup
            local_measurements["hopping_inversion_dn"][n] += tdn
            local_measurements["hopping_inversion"][n] += tn

            # measure hopping inversion
            tup = sgn * measure_hopping_inversion_avg(tight_binding_parameters_up, fermion_path_integral_up, bond_id)
            tdn = sgn * measure_hopping_inversion_avg(tight_binding_parameters_dn, fermion_path_integral_up, bond_id)
            tn = (tup + tup)/2
            local_measurements["hopping_inversion_avg_up"][n] += tup
            local_measurements["hopping_inversion_avg_dn"][n] += tdn
            local_measurements["hopping_inversion_avg"][n] += tn
        end
    end

    return nothing
end

# make local measurements associated with hubbard model
function make_local_measurements!(
    local_measurements::Dict{String, Vector{Complex{E}}},
    Gup::AbstractMatrix{T}, Gdn::AbstractMatrix{T}, sgn::T,
    model_geometry::ModelGeometry{D,E,N},
    hubbard_parameters::HubbardParameters{E},
    tight_binding_parameters_up::TightBindingParameters{T,E},
    tight_binding_parameters_dn::TightBindingParameters{T,E}
) where {T<:Number, E<:AbstractFloat, D, N}

    # measure hubbard energy for each orbital in unit cell
    hubbard_energies = local_measurements["hubbard_energy"]
    for orbital in eachindex(hubbard_energies)
        hubbard_energies[orbital] += sgn * measure_hubbard_energy(hubbard_parameters, Gup, Gdn, orbital)
    end

    return nothing
end

# make local measurements associated with electron-phonon model
function make_local_measurements!(
    local_measurements::Dict{String, Vector{Complex{E}}},
    Gup::AbstractMatrix{T}, Gdn::AbstractMatrix{T}, sgn::T,
    model_geometry::ModelGeometry{D,E,N},
    electron_phonon_parameters::ElectronPhononParameters{T,E},
    tight_binding_parameters_up::TightBindingParameters{T,E},
    tight_binding_parameters_dn::TightBindingParameters{T,E}
) where {T<:Number, E<:AbstractFloat, D, N}

    x = electron_phonon_parameters.x
    nphonon = electron_phonon_parameters.phonon_parameters.nphonon::Int # number of phonon modes per unit cell
    nholstein = electron_phonon_parameters.holstein_parameters_up.nholstein::Int # number of types of holstein couplings
    nssh = electron_phonon_parameters.ssh_parameters_up.nssh::Int # number of types of ssh coupling
    ndispersion = electron_phonon_parameters.dispersion_parameters.ndispersion::Int # number of types of dispersive phonon couplings

    # make phonon mode related measurements
    for n in 1:nphonon
        local_measurements["phonon_kin_energy"][n]   += sgn * measure_phonon_kinetic_energy(electron_phonon_parameters, n)
        local_measurements["phonon_pot_energy"][n] += sgn * measure_phonon_potential_energy(electron_phonon_parameters, n)
        local_measurements["X"][n]  += sgn * measure_phonon_position_moment(electron_phonon_parameters, n, 1)
        local_measurements["X2"][n] += sgn * measure_phonon_position_moment(electron_phonon_parameters, n, 2)
        local_measurements["X3"][n] += sgn * measure_phonon_position_moment(electron_phonon_parameters, n, 3)
        local_measurements["X4"][n] += sgn * measure_phonon_position_moment(electron_phonon_parameters, n, 4)
    end

    # check if finite number of holstein couplings
    if nholstein > 0
        holstein_parameters_up = electron_phonon_parameters.holstein_parameters_up
        holstein_parameters_dn = electron_phonon_parameters.holstein_parameters_dn
        # make holstein coupling related measurements
        for n in 1:nholstein
            ϵ_hol_up = sgn * measure_holstein_energy(holstein_parameters_up, Gup, x, n)
            ϵ_hol_dn = sgn * measure_holstein_energy(holstein_parameters_dn, Gdn, x, n)
            ϵ_hol = ϵ_hol_up + ϵ_hol_dn
            local_measurements["holstein_energy_up"][n] += ϵ_hol_up
            local_measurements["holstein_energy_dn"][n] += ϵ_hol_dn
            local_measurements["holstein_energy"][n]    += ϵ_hol
        end
    end

    # check if finite number of ssh couplings
    if nssh > 0
        # make ssh coupling related measurements
        for n in 1:nssh
            ssh_parameters_up = electron_phonon_parameters.ssh_parameters_up
            ssh_parameters_dn = electron_phonon_parameters.ssh_parameters_dn
            ϵ_ssh_up = sgn * measure_ssh_energy(ssh_parameters_up, Gup, x, n)
            ϵ_ssh_dn = sgn * measure_ssh_energy(ssh_parameters_dn, Gdn, x, n)
            ϵ_ssh = ϵ_ssh_up + ϵ_ssh_dn
            local_measurements["ssh_energy_up"][n] += ϵ_ssh_up
            local_measurements["ssh_energy_dn"][n] += ϵ_ssh_dn
            local_measurements["ssh_energy"][n] += ϵ_ssh
        end
    end

    # check if finite number of dispersive phonon coupling
    if ndispersion > 0
        # make ssh coupling related measurements
        for n in 1:ndispersion
            local_measurements["dispersion_energy"][n] += sgn * measure_dispersion_energy(electron_phonon_parameters, n)
        end
    end

    return nothing
end

# null local measurements to undefined parameters types
function make_local_measurements!(
    local_measurements::Dict{String, Vector{Complex{E}}},
    Gup::AbstractMatrix{T}, Gdn::AbstractMatrix{T}, sgn::T,
    model_geometry::ModelGeometry{D,E,N},
    some_model_parameters,
    tight_binding_parameters_up::TightBindingParameters{T,E},
    tight_binding_parameters_dn::TightBindingParameters{T,E}
) where {T<:Number, E<:AbstractFloat, D, N}

    return nothing
end

############################################
## MAKE CORRELATION FUNCTION MEASUREMENTS ##
############################################

# make purely electronic equal-time correlation measurements
function make_equaltime_measurements!(
    equaltime_correlations::Dict{String, CorrelationContainer{D,E}}, sgn::T,
    Gup::AbstractMatrix{T}, Gup_ττ::AbstractMatrix{T}, Gup_τ0::AbstractMatrix{T}, Gup_0τ::AbstractMatrix{T},
    Gdn::AbstractMatrix{T}, Gdn_ττ::AbstractMatrix{T}, Gdn_τ0::AbstractMatrix{T}, Gdn_0τ::AbstractMatrix{T},
    model_geometry::ModelGeometry{D,E,N},
    tight_binding_parameters_up::TightBindingParameters{T,E},
    tight_binding_parameters_dn::TightBindingParameters{T,E},
    fermion_path_integral_up::FermionPathIntegral{T,E},
    fermion_path_integral_dn::FermionPathIntegral{T,E}
) where {T<:Number, E<:AbstractFloat, D, N}

    Lτ = fermion_path_integral_up.Lτ::Int
    unit_cell = model_geometry.unit_cell::UnitCell{D,E,N}
    lattice = model_geometry.lattice::Lattice{D}
    bonds = model_geometry.bonds::Vector{Bond{D}}

    # note that for equal-time measurements:
    # Gup_τ0 = Gup and Gdn_τ0 = Gdn
    # Gup_ττ = Gup and Gdn_ττ = Gdn
    # Gup_0τ = Gup-I and Gdn_0τ = Gdn-I

    # iterate over equal-time correlation function getting measured
    for correlation in keys(equaltime_correlations)
        
        correlation_container = equaltime_correlations[correlation]::CorrelationContainer{D,E}
        id_pairs = correlation_container.id_pairs::Vector{NTuple{2,Int}}
        bond_id_pairs = correlation_container.bond_id_pairs::Vector{NTuple{2,Int}}
        correlations = correlation_container.correlations::Vector{Array{Complex{E}, D}}

        if correlation == "greens"

            for i in eachindex(id_pairs)
                pair = id_pairs[i]
                correlation = correlations[i]
                greens!(correlation, pair[2], pair[1], unit_cell, lattice, Gup_τ0, sgn/2)
                greens!(correlation, pair[2], pair[1], unit_cell, lattice, Gdn_τ0, sgn/2)
            end

        elseif correlation == "greens_up"

            for i in eachindex(id_pairs)
                pair = id_pairs[i]
                correlation = correlations[i]
                greens!(correlation, pair[2], pair[1], unit_cell, lattice, Gup_τ0, sgn)
            end

        elseif correlation == "greens_dn"

            for i in eachindex(id_pairs)
                pair = id_pairs[i]
                correlation = correlations[i]
                greens!(correlation, pair[2], pair[1], unit_cell, lattice, Gdn_τ0, sgn)
            end

        elseif correlation == "greens_tautau"

            for i in eachindex(id_pairs)
                pair = id_pairs[i]
                correlation = correlations[i]
                greens!(correlation, pair[2], pair[1], unit_cell, lattice, Gup_τ0, sgn/2)
                greens!(correlation, pair[2], pair[1], unit_cell, lattice, Gdn_τ0, sgn/2)
            end

        elseif correlation == "greens_tautau_up"

            for i in eachindex(id_pairs)
                pair = id_pairs[i]
                correlation = correlations[i]
                greens!(correlation, pair[2], pair[1], unit_cell, lattice, Gup_τ0, sgn)
            end

        elseif correlation == "greens_tautau_dn"

            for i in eachindex(id_pairs)
                pair = id_pairs[i]
                correlation = correlations[i]
                greens!(correlation, pair[2], pair[1], unit_cell, lattice, Gdn_τ0, sgn)
            end    

        elseif correlation == "density_upup"

            for i in eachindex(id_pairs)
                pair = id_pairs[i]
                correlation = correlations[i]
                density_correlation!(correlation, pair[2], pair[1], unit_cell, lattice,
                                     Gup_τ0, Gup_0τ, Gup_ττ, Gup, +1, +1, sgn)
            end

        elseif correlation == "density_dndn"

            for i in eachindex(id_pairs)
                pair = id_pairs[i]
                correlation = correlations[i]
                density_correlation!(correlation, pair[2], pair[1], unit_cell, lattice,
                                     Gdn_τ0, Gdn_0τ, Gdn_ττ, Gdn, -1, -1, sgn)
            end

        elseif correlation == "density_updn"

            for i in eachindex(id_pairs)
                pair = id_pairs[i]
                correlation = correlations[i]
                density_correlation!(correlation, pair[2], pair[1], unit_cell, lattice,
                                     Gup_τ0, Gup_0τ, Gup_ττ, Gdn, +1, -1, sgn)
            end

        elseif correlation == "density_dnup"

            for i in eachindex(id_pairs)
                pair = id_pairs[i]
                correlation = correlations[i]
                density_correlation!(correlation, pair[2], pair[1], unit_cell, lattice,
                                     Gdn_τ0, Gdn_0τ, Gdn_ττ, Gup, -1, +1, sgn)
            end

        elseif correlation == "density"

            for i in eachindex(id_pairs)
                pair = id_pairs[i]
                correlation = correlations[i]
                density_correlation!(correlation, pair[2], pair[1], unit_cell, lattice,
                                     Gup_τ0, Gup_0τ, Gup_ττ, Gup, Gdn_τ0, Gdn_0τ, Gdn_ττ, Gdn, sgn)
            end

        elseif correlation == "pair"

            for i in eachindex(id_pairs)
                pair = id_pairs[i]
                correlation = correlations[i]
                pair_correlation!(correlation, bonds[pair[2]], bonds[pair[1]], unit_cell, lattice,
                                  Gup_τ0, Gdn_τ0, sgn)
            end

        elseif correlation == "spin_x"

            for i in eachindex(id_pairs)
                pair = id_pairs[i]
                correlation = correlations[i]
                spin_x_correlation!(correlation, pair[2], pair[1], unit_cell, lattice,
                                    Gup_τ0, Gup_0τ, Gdn_τ0, Gdn_0τ, sgn)
            end

        elseif correlation == "spin_z"

            for i in eachindex(id_pairs)
                pair = id_pairs[i]
                correlation = correlations[i]
                spin_z_correlation!(correlation, pair[2], pair[1], unit_cell, lattice,
                                    Gup_τ0, Gup_0τ, Gup_ττ, Gup, Gdn_τ0, Gdn_0τ, Gdn_ττ, Gdn, sgn)
            end

        elseif correlation == "bond_upup"

            for i in eachindex(id_pairs)
                pair = id_pairs[i]
                correlation = correlations[i]
                bond_correlation!(correlation, bonds[pair[2]], bonds[pair[1]], unit_cell, lattice,
                                  Gup_τ0, Gup_0τ, Gup_ττ, Gup, +1, +1, sgn)
            end

        elseif correlation == "bond_dndn"

            for i in eachindex(id_pairs)
                pair = id_pairs[i]
                correlation = correlations[i]
                bond_correlation!(correlation, bonds[pair[2]], bonds[pair[1]], unit_cell, lattice,
                                  Gdn_τ0, Gdn_0τ, Gdn_ττ, Gdn, -1, -1, sgn)
            end

        elseif correlation == "bond_updn"

            for i in eachindex(id_pairs)
                pair = id_pairs[i]
                correlation = correlations[i]
                bond_correlation!(correlation, bonds[pair[2]], bonds[pair[1]], unit_cell, lattice,
                                  Gup_τ0, Gup_0τ, Gup_ττ, Gdn, +1, -1, sgn)
            end

        elseif correlation == "bond_dnup"

            for i in eachindex(id_pairs)
                pair = id_pairs[i]
                correlation = correlations[i]
                bond_correlation!(correlation, bonds[pair[2]], bonds[pair[1]], unit_cell, lattice,
                                  Gdn_τ0, Gdn_0τ, Gdn_ττ, Gup, -1, +1, sgn)
            end

        elseif correlation == "bond"

            for i in eachindex(id_pairs)
                pair = id_pairs[i]
                correlation = correlations[i]
                bond_correlation!(correlation, bonds[pair[2]], bonds[pair[1]], unit_cell, lattice,
                                  Gup_τ0, Gup_0τ, Gup_ττ, Gup, Gdn_τ0, Gdn_0τ, Gdn_ττ, Gdn, sgn)
            end

        elseif correlation == "current_upup"

            (; bond_ids, bond_slices) = tight_binding_parameters_up
            tup = fermion_path_integral_up.t

            for i in eachindex(id_pairs)
                # get the hopping IDs associated with current operators
                id_pair = id_pairs[1]
                hopping_id_0 = id_pair[1]
                hopping_id_1 = id_pair[2]
                # get the bond IDs associated with the hopping IDs
                bond_id_0 = bond_ids[hopping_id_0]
                bond_id_1 = bond_ids[hopping_id_1]
                # record bond id pair for current correlation measurement if not already recorded
                if (bond_id_pairs[i][1] == 0) && (bond_id_pairs[i][2] == 0)
                    bond_id_pairs[i] = (bond_id_0, bond_id_1)
                end
                # get the bond definitions
                bond_0 = bonds[bond_id_0]
                bond_1 = bonds[bond_id_1]
                # get the effective hopping amptlitudes for each of the two hopping ID's in question
                tup0 = @view tup[bond_slices[hopping_id_0], Lτ]
                tup0′ = reshape(tup0, lattice.L...)
                tup1 = @view tup[bond_slices[hopping_id_1], Lτ]
                tup1′ = reshape(tup1, lattice.L...)
                # measure the current-current correlation
                correlation = correlations[i]
                current_correlation!(correlation, bond_1, bond_0, tup1′, tup0′, unit_cell, lattice,
                                     Gup_τ0, Gup_0τ, Gup_ττ, Gup, +1, +1, sgn)
            end

        elseif correlation == "current_dndn"

            (; bond_ids, bond_slices) = tight_binding_parameters_dn
            tdn = fermion_path_integral_dn.t

            for i in eachindex(id_pairs)
                # get the hopping IDs associated with current operators
                id_pair = id_pairs[1]
                hopping_id_0 = id_pair[1]
                hopping_id_1 = id_pair[2]
                # get the bond IDs associated with the hopping IDs
                bond_id_0 = bond_ids[hopping_id_0]
                bond_id_1 = bond_ids[hopping_id_1]
                # record bond id pair for current correlation measurement if not already recorded
                if (bond_id_pairs[i][1] == 0) && (bond_id_pairs[i][2] == 0)
                    bond_id_pairs[i] = (bond_id_0, bond_id_1)
                end
                # get the bond definitions
                bond_0 = bonds[bond_id_0]
                bond_1 = bonds[bond_id_1]
                # get the effective hopping amptlitudes for each of the two hopping ID's in question
                tdn0 = @view tdn[bond_slices[hopping_id_0], Lτ]
                tdn0′ = reshape(tdn0, lattice.L...)
                tdn1 = @view tdn[bond_slices[hopping_id_1], Lτ]
                tdn1′ = reshape(tdn1, lattice.L...)
                # measure the current-current correlation
                correlation = correlations[i]
                current_correlation!(correlation, bond_1, bond_0, tdn1′, tdn0′, unit_cell, lattice,
                                     Gdn_τ0, Gdn_0τ, Gdn_ττ, Gdn, -1, -1, sgn)
            end

        elseif correlation == "current_updn"

            (; bond_ids, bond_slices) = tight_binding_parameters_up
            tup = fermion_path_integral_up.t
            tdn = fermion_path_integral_dn.t

            for i in eachindex(id_pairs)
                # get the hopping IDs associated with current operators
                id_pair = id_pairs[1]
                hopping_id_0 = id_pair[1]
                hopping_id_1 = id_pair[2]
                # get the bond IDs associated with the hopping IDs
                bond_id_0 = bond_ids[hopping_id_0]
                bond_id_1 = bond_ids[hopping_id_1]
                # record bond id pair for current correlation measurement if not already recorded
                if (bond_id_pairs[i][1] == 0) && (bond_id_pairs[i][2] == 0)
                    bond_id_pairs[i] = (bond_id_0, bond_id_1)
                end
                # get the bond definitions
                bond_0 = bonds[bond_id_0]
                bond_1 = bonds[bond_id_1]
                # get the effective hopping amptlitudes for each of the two hopping ID's in question
                tup1 = @view tup[bond_slices[hopping_id_1], Lτ]
                tup1′ = reshape(tup1, lattice.L...)
                tdn0 = @view tdn[bond_slices[hopping_id_0], Lτ]
                tdn0′ = reshape(tdn0, lattice.L...)
                # measure the current-current correlation
                correlation = correlations[i]
                current_correlation!(correlation, bond_1, bond_0, tup1′, tdn0′, unit_cell, lattice,
                                    Gup_τ0, Gup_0τ, Gup_ττ, Gdn, +1, -1, sgn)
            end

        elseif correlation == "current_dnup"

            (; bond_ids, bond_slices) = tight_binding_parameters_up
            tup = fermion_path_integral_up.t
            tdn = fermion_path_integral_dn.t

            for i in eachindex(id_pairs)
                # get the hopping IDs associated with current operators
                id_pair = id_pairs[1]
                hopping_id_0 = id_pair[1]
                hopping_id_1 = id_pair[2]
                # get the bond IDs associated with the hopping IDs
                bond_id_0 = bond_ids[hopping_id_0]
                bond_id_1 = bond_ids[hopping_id_1]
                # record bond id pair for current correlation measurement if not already recorded
                if (bond_id_pairs[i][1] == 0) && (bond_id_pairs[i][2] == 0)
                    bond_id_pairs[i] = (bond_id_0, bond_id_1)
                end
                # get the bond definitions
                bond_0 = bonds[bond_id_0]
                bond_1 = bonds[bond_id_1]
                # get the effective hopping amptlitudes for each of the two hopping ID's in question
                tup0 = @view tup[bond_slices[hopping_id_0], Lτ]
                tup0′ = reshape(tup0, lattice.L...)
                tdn1 = @view tdn[bond_slices[hopping_id_1], Lτ]
                tdn1′ = reshape(tdn1, lattice.L...)
                # measure the current-current correlation
                correlation = correlations[i]
                current_correlation!(correlation, bond_1, bond_0, tdn1′, tup0′, unit_cell, lattice,
                                     Gdn_τ0, Gdn_0τ, Gdn_ττ, Gup, -1, +1, sgn)
            end

        elseif correlation == "current"

            (; bond_ids, bond_slices) = tight_binding_parameters_up
            tup = fermion_path_integral_up.t
            tdn = fermion_path_integral_dn.t

            for i in eachindex(id_pairs)
                # get the hopping IDs associated with current operators
                id_pair = id_pairs[1]
                hopping_id_0 = id_pair[1]
                hopping_id_1 = id_pair[2]
                # get the bond IDs associated with the hopping IDs
                bond_id_0 = bond_ids[hopping_id_0]
                bond_id_1 = bond_ids[hopping_id_1]
                # record bond id pair for current correlation measurement if not already recorded
                if (bond_id_pairs[i][1] == 0) && (bond_id_pairs[i][2] == 0)
                    bond_id_pairs[i] = (bond_id_0, bond_id_1)
                end
                # get the bond definitions
                bond_0 = bonds[bond_id_0]
                bond_1 = bonds[bond_id_1]
                # get the effective hopping amptlitudes for each of the two hopping ID's in question
                tup0 = @view tup[bond_slices[hopping_id_0], Lτ]
                tup0′ = reshape(tup0, lattice.L...)
                tup1 = @view tup[bond_slices[hopping_id_1], Lτ]
                tup1′ = reshape(tup1, lattice.L...)
                tdn0 = @view tdn[bond_slices[hopping_id_0], Lτ]
                tdn0′ = reshape(tdn0, lattice.L...)
                tdn1 = @view tdn[bond_slices[hopping_id_1], Lτ]
                tdn1′ = reshape(tdn1, lattice.L...)
                # measure the current-current correlation
                correlation = correlations[i]
                current_correlation!(correlation, bond_1, bond_0, tup1′, tup0′, tdn1′, tdn0′, unit_cell, lattice,
                                     Gup_τ0, Gup_0τ, Gup_ττ, Gup, Gdn_τ0, Gdn_0τ, Gdn_ττ, Gdn, sgn)
            end
        end
    end

    return nothing
end


###########################################################
## MAKE TIME-DISPLACED CORRELATION FUNCTION MEASUREMENTS ##
###########################################################

# make purely electronic time-displaced correlation measurements
function make_time_displaced_measurements!(time_displaced_correlations::Dict{String, CorrelationContainer{P,E}}, l::Int, sgn::T,
                                           Gup::AbstractMatrix{T}, Gup_ττ::AbstractMatrix{T}, Gup_τ0::AbstractMatrix{T}, Gup_0τ::AbstractMatrix{T},
                                           Gdn::AbstractMatrix{T}, Gdn_ττ::AbstractMatrix{T}, Gdn_τ0::AbstractMatrix{T}, Gdn_0τ::AbstractMatrix{T},
                                           model_geometry::ModelGeometry{D,E,N},
                                           tight_binding_parameters_up::TightBindingParameters{T,E},
                                           tight_binding_parameters_dn::TightBindingParameters{T,E},
                                           fermion_path_integral_up::FermionPathIntegral{T,E},
                                           fermion_path_integral_dn::FermionPathIntegral{T,E}) where {T<:Number, E<:AbstractFloat, P, D, N}

    Lτ = fermion_path_integral_up.Lτ::Int
    unit_cell = model_geometry.unit_cell::UnitCell{D,E,N}
    lattice = model_geometry.lattice::Lattice{D}
    bonds = model_geometry.bonds::Vector{Bond{D}}

    # iterate over time-displaced correlation function getting measured
    for correlation in keys(time_displaced_correlations)
        
        correlation_container = time_displaced_correlations[correlation]::CorrelationContainer{P,E}
        id_pairs = correlation_container.id_pairs::Vector{NTuple{2,Int}}
        bond_id_pairs = correlation_container.bond_id_pairs::Vector{NTuple{2,Int}}
        correlations = correlation_container.correlations::Vector{Array{Complex{E}, P}}

        if correlation == "greens"

            for i in eachindex(id_pairs)
                pair = id_pairs[i]
                correlation = selectdim(correlations[i], D+1, l+1)
                greens!(correlation, pair[2], pair[1], unit_cell, lattice, Gup_τ0, sgn/2)
                greens!(correlation, pair[2], pair[1], unit_cell, lattice, Gdn_τ0, sgn/2)
            end

        elseif correlation == "greens_up"

            for i in eachindex(id_pairs)
                pair = id_pairs[i]
                correlation = selectdim(correlations[i], D+1, l+1)
                greens!(correlation, pair[2], pair[1], unit_cell, lattice, Gup_τ0, sgn)
            end

        elseif correlation == "greens_dn"

            for i in eachindex(id_pairs)
                pair = id_pairs[i]
                correlation = selectdim(correlations[i], D+1, l+1)
                greens!(correlation, pair[2], pair[1], unit_cell, lattice, Gdn_τ0, sgn)
            end

        elseif correlation == "greens_tautau"

            for i in eachindex(id_pairs)
                pair = id_pairs[i]
                correlation = selectdim(correlations[i], D+1, l+1)
                greens!(correlation, pair[2], pair[1], unit_cell, lattice, Gup_ττ, sgn/2)
                greens!(correlation, pair[2], pair[1], unit_cell, lattice, Gdn_ττ, sgn/2)
            end

        elseif correlation == "greens_tautau_up"

            for i in eachindex(id_pairs)
                pair = id_pairs[i]
                correlation = selectdim(correlations[i], D+1, l+1)
                greens!(correlation, pair[2], pair[1], unit_cell, lattice, Gup_ττ, sgn)
            end

        elseif correlation == "greens_tautau_dn"

            for i in eachindex(id_pairs)
                pair = id_pairs[i]
                correlation = selectdim(correlations[i], D+1, l+1)
                greens!(correlation, pair[2], pair[1], unit_cell, lattice, Gdn_ττ, sgn)
            end

        elseif correlation == "density_upup"

            for i in eachindex(id_pairs)
                pair = id_pairs[i]
                correlation = selectdim(correlations[i], D+1, l+1)
                density_correlation!(correlation, pair[2], pair[1], unit_cell, lattice,
                                     Gup_τ0, Gup_0τ, Gup_ττ, Gup, +1, +1, sgn)
            end

        elseif correlation == "density_dndn"

            for i in eachindex(id_pairs)
                pair = id_pairs[i]
                correlation = selectdim(correlations[i], D+1, l+1)
                density_correlation!(correlation, pair[2], pair[1], unit_cell, lattice,
                                     Gdn_τ0, Gdn_0τ, Gdn_ττ, Gdn, -1, -1, sgn)
            end

        elseif correlation == "density_updn"

            for i in eachindex(id_pairs)
                pair = id_pairs[i]
                correlation = selectdim(correlations[i], D+1, l+1)
                density_correlation!(correlation, pair[2], pair[1], unit_cell, lattice,
                                     Gup_τ0, Gup_0τ, Gup_ττ, Gdn, +1, -1, sgn)
            end

        elseif correlation == "density_dnup"

            for i in eachindex(id_pairs)
                pair = id_pairs[i]
                correlation = selectdim(correlations[i], D+1, l+1)
                density_correlation!(correlation, pair[2], pair[1], unit_cell, lattice,
                                     Gdn_τ0, Gdn_0τ, Gdn_ττ, Gup, -1, +1, sgn)
            end

        elseif correlation == "density"

            for i in eachindex(id_pairs)
                pair = id_pairs[i]
                correlation = selectdim(correlations[i], D+1, l+1)
                density_correlation!(correlation, pair[2], pair[1], unit_cell, lattice,
                                     Gup_τ0, Gup_0τ, Gup_ττ, Gup, Gdn_τ0, Gdn_0τ, Gdn_ττ, Gdn, sgn)
            end

        elseif correlation == "pair"

            for i in eachindex(id_pairs)
                pair = id_pairs[i]
                correlation = selectdim(correlations[i], D+1, l+1)
                pair_correlation!(correlation, bonds[pair[2]], bonds[pair[1]], unit_cell, lattice,
                                  Gup_τ0, Gdn_τ0, sgn)
            end

        elseif correlation == "spin_x"

            for i in eachindex(id_pairs)
                pair = id_pairs[i]
                correlation = selectdim(correlations[i], D+1, l+1)
                spin_x_correlation!(correlation, pair[2], pair[1], unit_cell, lattice,
                                    Gup_τ0, Gup_0τ, Gdn_τ0, Gdn_0τ, sgn)
            end

        elseif correlation == "spin_z"

            for i in eachindex(id_pairs)
                pair = id_pairs[i]
                correlation = selectdim(correlations[i], D+1, l+1)
                spin_z_correlation!(correlation, pair[2], pair[1], unit_cell, lattice,
                                    Gup_τ0, Gup_0τ, Gup_ττ, Gup, Gdn_τ0, Gdn_0τ, Gdn_ττ, Gdn, sgn)
            end

        elseif correlation == "bond_upup"

            for i in eachindex(id_pairs)
                pair = id_pairs[i]
                correlation = selectdim(correlations[i], D+1, l+1)
                bond_correlation!(correlation, bonds[pair[2]], bonds[pair[1]], unit_cell, lattice,
                                  Gup_τ0, Gup_0τ, Gup_ττ, Gup, +1, +1, sgn)
            end

        elseif correlation == "bond_dndn"

            for i in eachindex(id_pairs)
                pair = id_pairs[i]
                correlation = selectdim(correlations[i], D+1, l+1)
                bond_correlation!(correlation, bonds[pair[2]], bonds[pair[1]], unit_cell, lattice,
                                  Gdn_τ0, Gdn_0τ, Gdn_ττ, Gdn, -1, -1, sgn)
            end

        elseif correlation == "bond_updn"

            for i in eachindex(id_pairs)
                pair = id_pairs[i]
                correlation = selectdim(correlations[i], D+1, l+1)
                bond_correlation!(correlation, bonds[pair[2]], bonds[pair[1]], unit_cell, lattice,
                                  Gup_τ0, Gup_0τ, Gup_ττ, Gdn, +1, -1, sgn)
            end

        elseif correlation == "bond_dnup"

            for i in eachindex(id_pairs)
                pair = id_pairs[i]
                correlation = selectdim(correlations[i], D+1, l+1)
                bond_correlation!(correlation, bonds[pair[2]], bonds[pair[1]], unit_cell, lattice,
                                  Gdn_τ0, Gdn_0τ, Gdn_ττ, Gup, -1, +1, sgn)
            end

        elseif correlation == "bond"

            for i in eachindex(id_pairs)
                pair = id_pairs[i]
                correlation = selectdim(correlations[i], D+1, l+1)
                bond_correlation!(correlation, bonds[pair[2]], bonds[pair[1]], unit_cell, lattice,
                                  Gup_τ0, Gup_0τ, Gup_ττ, Gup, Gdn_τ0, Gdn_0τ, Gdn_ττ, Gdn, sgn)
            end
        
        elseif correlation == "current_upup"

            (; bond_ids, bond_slices) = tight_binding_parameters_up
            tup = fermion_path_integral_up.t

            for i in eachindex(id_pairs)
                # get the hopping IDs associated with current operators
                id_pair = id_pairs[i]
                hopping_id_0 = id_pair[1]
                hopping_id_1 = id_pair[2]
                # get the bond IDs associated with the hopping IDs
                bond_id_0 = bond_ids[hopping_id_0]
                bond_id_1 = bond_ids[hopping_id_1]
                # record bond id pair for current correlation measurement if not already recorded
                if (bond_id_pairs[i][1] == 0) && (bond_id_pairs[i][2] == 0)
                    bond_id_pairs[i] = (bond_id_0, bond_id_1)
                end
                # get the bond definitions
                bond_0 = bonds[bond_id_0]
                bond_1 = bonds[bond_id_1]
                # get the effective hopping amptlitudes for each of the two hopping ID's in question
                tup0 = @view tup[bond_slices[hopping_id_0], Lτ]
                tup0′ = reshape(tup0, lattice.L...)
                tup1 = @view tup[bond_slices[hopping_id_1], mod1(l,Lτ)]
                tup1′ = reshape(tup1, lattice.L...)
                # measure the current-current correlation
                correlation = selectdim(correlations[i], D+1, l+1)
                current_correlation!(correlation, bond_1, bond_0, tup1′, tup0′, unit_cell, lattice,
                                     Gup_τ0, Gup_0τ, Gup_ττ, Gup, +1, +1, sgn)
            end

        elseif correlation == "current_dndn"

            (; bond_ids, bond_slices) = tight_binding_parameters_dn
            tdn = fermion_path_integral_dn.t

            for i in eachindex(id_pairs)
                # get the hopping IDs associated with current operators
                id_pair = id_pairs[i]
                hopping_id_0 = id_pair[1]
                hopping_id_1 = id_pair[2]
                # get the bond IDs associated with the hopping IDs
                bond_id_0 = bond_ids[hopping_id_0]
                bond_id_1 = bond_ids[hopping_id_1]
                # record bond id pair for current correlation measurement if not already recorded
                if (bond_id_pairs[i][1] == 0) && (bond_id_pairs[i][2] == 0)
                    bond_id_pairs[i] = (bond_id_0, bond_id_1)
                end
                # get the bond definitions
                bond_0 = bonds[bond_id_0]
                bond_1 = bonds[bond_id_1]
                # get the effective hopping amptlitudes for each of the two hopping ID's in question
                tdn0 = @view tdn[bond_slices[hopping_id_0], Lτ]
                tdn0′ = reshape(tdn0, lattice.L...)
                tdn1 = @view tdn[bond_slices[hopping_id_1], mod1(l,Lτ)]
                tdn1′ = reshape(tdn1, lattice.L...)
                # measure the current-current correlation
                correlation = selectdim(correlations[i], D+1, l+1)
                current_correlation!(correlation, bond_1, bond_0, tdn1′, tdn0′, unit_cell, lattice,
                                     Gdn_τ0, Gdn_0τ, Gdn_ττ, Gdn, -1, -1, sgn)
            end

        elseif correlation == "current_updn"

            (; bond_ids, bond_slices) = tight_binding_parameters_up
            tup = fermion_path_integral_up.t
            tdn = fermion_path_integral_dn.t

            for i in eachindex(id_pairs)
                # get the hopping IDs associated with current operators
                id_pair = id_pairs[i]
                hopping_id_0 = id_pair[1]
                hopping_id_1 = id_pair[2]
                # get the bond IDs associated with the hopping IDs
                bond_id_0 = bond_ids[hopping_id_0]
                bond_id_1 = bond_ids[hopping_id_1]
                # record bond id pair for current correlation measurement if not already recorded
                if (bond_id_pairs[i][1] == 0) && (bond_id_pairs[i][2] == 0)
                    bond_id_pairs[i] = (bond_id_0, bond_id_1)
                end
                # get the bond definitions
                bond_0 = bonds[bond_id_0]
                bond_1 = bonds[bond_id_1]
                # get the effective hopping amptlitudes for each of the two hopping ID's in question
                tup0 = @view tup[bond_slices[hopping_id_0], Lτ]
                tup0′ = reshape(tup0, lattice.L...)
                tdn1 = @view tdn[bond_slices[hopping_id_1], mod1(l,Lτ)]
                tdn1′ = reshape(tdn1, lattice.L...)
                # measure the current-current correlation
                correlation = selectdim(correlations[i], D+1, l+1)
                current_correlation!(correlation, bond_1, bond_0, tup1′, tdn0′, unit_cell, lattice,
                                     Gup_τ0, Gup_0τ, Gup_ττ, Gdn, +1, -1, sgn)
            end

        elseif correlation == "current_dnup"

            (; bond_ids, bond_slices) = tight_binding_parameters_up
            tup = fermion_path_integral_up.t
            tdn = fermion_path_integral_dn.t

            for i in eachindex(id_pairs)
                # get the hopping IDs associated with current operators
                id_pair = id_pairs[i]
                hopping_id_0 = id_pair[1]
                hopping_id_1 = id_pair[2]
                # get the bond IDs associated with the hopping IDs
                bond_id_0 = bond_ids[hopping_id_0]
                bond_id_1 = bond_ids[hopping_id_1]
                # record bond id pair for current correlation measurement if not already recorded
                if (bond_id_pairs[i][1] == 0) && (bond_id_pairs[i][2] == 0)
                    bond_id_pairs[i] = (bond_id_0, bond_id_1)
                end
                # get the bond definitions
                bond_0 = bonds[bond_id_0]
                bond_1 = bonds[bond_id_1]
                # get the effective hopping amptlitudes for each of the two hopping ID's in question
                tdn0 = @view tdn[bond_slices[hopping_id_0], Lτ]
                tdn0′ = reshape(tdn0, lattice.L...)
                tup1 = @view tup[bond_slices[hopping_id_1], mod1(l,Lτ)]
                tup1′ = reshape(tup1, lattice.L...)
                # measure the current-current correlation
                correlation = selectdim(correlations[i], D+1, l+1)
                current_correlation!(correlation, bond_1, bond_0, tdn1′, tup0′, unit_cell, lattice,
                                     Gdn_τ0, Gdn_0τ, Gdn_ττ, Gup, -1, +1, sgn)
            end

        elseif correlation == "current"

            (; bond_ids, bond_slices) = tight_binding_parameters_up
            tup = fermion_path_integral_up.t
            tdn = fermion_path_integral_dn.t

            for i in eachindex(id_pairs)
                # get the hopping IDs associated with current operators
                id_pair = id_pairs[i]
                hopping_id_0 = id_pair[1]
                hopping_id_1 = id_pair[2]
                # get the bond IDs associated with the hopping IDs
                bond_id_0 = bond_ids[hopping_id_0]
                bond_id_1 = bond_ids[hopping_id_1]
                # record bond id pair for current correlation measurement if not already recorded
                if (bond_id_pairs[i][1] == 0) && (bond_id_pairs[i][2] == 0)
                    bond_id_pairs[i] = (bond_id_0, bond_id_1)
                end
                # get the bond definitions
                bond_0 = bonds[bond_id_0]
                bond_1 = bonds[bond_id_1]
                # get the effective hopping amptlitudes for each of the two hopping ID's in question
                tup0 = @view tup[bond_slices[hopping_id_0], Lτ]
                tup0′ = reshape(tup0, lattice.L...)
                tup1 = @view tup[bond_slices[hopping_id_1], mod1(l,Lτ)]
                tup1′ = reshape(tup1, lattice.L...)
                tdn0 = @view tdn[bond_slices[hopping_id_0], Lτ]
                tdn0′ = reshape(tdn0, lattice.L...)
                tdn1 = @view tdn[bond_slices[hopping_id_1], mod1(l,Lτ)]
                tdn1′ = reshape(tdn1, lattice.L...)
                # measure the current-current correlation
                correlation = selectdim(correlations[i], D+1, l+1)
                current_correlation!(correlation, bond_1, bond_0, tup1′, tup0′, tdn1′, tdn0′, unit_cell, lattice,
                                     Gup_τ0, Gup_0τ, Gup_ττ, Gup, Gdn_τ0, Gdn_0τ, Gdn_ττ, Gdn, sgn)
            end
        end
    end

    return nothing
end


####################################
## MEASURE PHONON GREENS FUNCTION ##
####################################

# measure equal-time phonon greens function
function measure_equaltime_phonon_greens!(phonon_greens::CorrelationContainer{D,E},
                                          electron_phonon_parameters::ElectronPhononParameters{T,E},
                                          model_geometry::ModelGeometry{D,E,N},
                                          sgn::T,
                                          XrX0::AbstractArray{Complex{E},P},
                                          Xr::AbstractArray{Complex{E},P},
                                          X0::AbstractArray{Complex{E},P}) where {T<:Number, E<:AbstractFloat, D, P, N}

    id_pairs = phonon_greens.id_pairs::Vector{NTuple{2,Int}}
    bond_id_pairs = phonon_greens.bond_id_pairs::Vector{NTuple{2,Int}}
    correlations = phonon_greens.correlations::Vector{Array{Complex{E}, D}}
    unit_cell = model_geometry.unit_cell::UnitCell{D,E,N}
    lattice = model_geometry.lattice::Lattice{D}
    bonds = model_geometry.bonds::Vector{Bond{D}}
    phonon_parameters = electron_phonon_parameters.phonon_parameters::PhononParameters{E}

    # get phonon field
    x = electron_phonon_parameters.x::Matrix{E}

    # length of imaginary time axis
    Lτ = size(x,2)

    # size of system in unit cells
    L = lattice.L

    # number of phonons per unit cell
    phonon_parameters = electron_phonon_parameters.phonon_parameters::PhononParameters{E}
    nphonon = phonon_parameters.nphonon::Int

    # reshape phonon field matrix into multi-dimensional array
    x′ = reshape(x, (L..., nphonon, Lτ))

    # get the site associated with each phonon field
    phonon_to_site = reshape(phonon_parameters.phonon_to_site, (lattice.N, nphonon))

    # iterate over all pairs of phonon modes
    for i in eachindex(id_pairs)
        # get the phonon fields associated with the appropriate pair of phonon modes in the unit cell
        id_pair = id_pairs[i]
        correlation = correlations[i]
        x0 = selectdim(x′, D+1, id_pair[1])
        xr = selectdim(x′, D+1, id_pair[2])
        copyto!(X0, x0)
        copyto!(Xr, xr)
        # calculate phonon greens function
        translational_avg!(XrX0, Xr, X0, restore = false)
        # record the equal-time phonon green's function
        XrX0_0 = selectdim(XrX0, D+1, 1)
        @. correlation += sgn * XrX0_0
        # record the bond id pair if not already recorded
        if (bond_id_pairs[i][1] == 0) && (bond_id_pairs[i][2] == 0)
            # get the site ID each phonon mode lives on
            site_1 = phonon_to_site[1, id_pair[1]]
            site_2 = phonon_to_site[1, id_pair[2]]
            # get orbital id associated with site
            orbital_1 = site_to_orbital(site_1, unit_cell)
            orbital_2 = site_to_orbital(site_2, unit_cell)
            # record orbital/bond id pair
            bond_id_pairs[i] = (orbital_1, orbital_2)
        end
    end

    return nothing
end

# measure time-displaced phonon greens function
function measure_time_displaced_phonon_greens!(phonon_greens::CorrelationContainer{P,E}, # time-displaced because P != D
                                               electron_phonon_parameters::ElectronPhononParameters{T,E},
                                               model_geometry::ModelGeometry{D,E,N},
                                               sgn::T,
                                               XrX0::AbstractArray{Complex{E},P},
                                               Xr::AbstractArray{Complex{E},P},
                                               X0::AbstractArray{Complex{E},P}) where {T<:Number, E<:AbstractFloat, D, P, N}

    id_pairs = phonon_greens.id_pairs::Vector{NTuple{2,Int}}
    bond_id_pairs = phonon_greens.bond_id_pairs::Vector{NTuple{2,Int}}
    correlations = phonon_greens.correlations::Vector{Array{Complex{E}, P}}
    unit_cell = model_geometry.unit_cell::UnitCell{D,E,N}
    lattice = model_geometry.lattice::Lattice{D}
    bonds = model_geometry.bonds::Vector{Bond{D}}
    phonon_parameters = electron_phonon_parameters.phonon_parameters::PhononParameters{E}

    # get phonon field
    x = electron_phonon_parameters.x::Matrix{E}

    # length of imaginary time axis
    Lτ = size(x,2)

    # size of system in unit cells
    L = lattice.L

    # number of phonons per unit cell
    phonon_parameters = electron_phonon_parameters.phonon_parameters::PhononParameters{E}
    nphonon = phonon_parameters.nphonon::Int

    # reshape phonon field matrix into multi-dimensional array
    x′ = reshape(x, (L..., nphonon, Lτ))

    # get the site associated with each phonon field
    phonon_to_site = reshape(phonon_parameters.phonon_to_site, (lattice.N, nphonon))

    # iterate over all pairs of phonon modes
    for i in eachindex(id_pairs)
        # get the phonon fields associated with the appropriate pair of phonon modes in the unit cell
        id_pair = id_pairs[i]
        correlation = correlations[i]
        x0 = selectdim(x′, D+1, id_pair[1])
        xr = selectdim(x′, D+1, id_pair[2])
        copyto!(X0, x0)
        copyto!(Xr, xr)
        # calculate phonon greens function
        translational_avg!(XrX0, Xr, X0, restore = false)
        correlation′ = selectdim(correlation, D+1, 1:Lτ)
        @. correlation′ += sgn * XrX0
        correlation_0  = selectdim(correlation, D+1, 1)
        correlation_Lτ = selectdim(correlation, D+1, Lτ+1)
        copyto!(correlation_Lτ, correlation_0)
        # record the bond id pair if not already recorded
        if (bond_id_pairs[i][1] == 0) && (bond_id_pairs[i][2] == 0)
            # get the site ID each phonon mode lives on
            site_1 = phonon_to_site[1, id_pair[1]]
            site_2 = phonon_to_site[1, id_pair[2]]
            # get orbital id associated with site
            orbital_1 = site_to_orbital(site_1, unit_cell)
            orbital_2 = site_to_orbital(site_2, unit_cell)
            # record orbital/bond id pair
            bond_id_pairs[i] = (orbital_1, orbital_2)
        end
    end

    return nothing
end