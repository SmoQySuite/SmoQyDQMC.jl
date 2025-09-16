#######################################################
## HIGHEST LEVEL/EXPORTED MAKE MEASUREMENTS FUNCTION ##
#######################################################

@doc raw"""
    make_measurements!(
        # ARGUMENTS
        measurement_container::NamedTuple,
        logdetGup::E, sgndetGup::T, Gup::AbstractMatrix{T},
        Gup_ττ::AbstractMatrix{T}, Gup_τ0::AbstractMatrix{T}, Gup_0τ::AbstractMatrix{T},
        logdetGdn::E, sgndetGdn::T, Gdn::AbstractMatrix{T},
        Gdn_ττ::AbstractMatrix{T}, Gdn_τ0::AbstractMatrix{T}, Gdn_0τ::AbstractMatrix{T};
        # KEYWORD ARGUMENTS
        fermion_path_integral_up::FermionPathIntegral{T,E},
        fermion_path_integral_dn::FermionPathIntegral{T,E},
        fermion_greens_calculator_up::FermionGreensCalculator{T,E},
        fermion_greens_calculator_dn::FermionGreensCalculator{T,E},
        Bup::Vector{P}, Bdn::Vector{P},
        model_geometry::ModelGeometry{D,E,N},
        tight_binding_parameters::Union{Nothing, TightBindingParameters} = nothing,
        tight_binding_parameters_up::Union{Nothing, TightBindingParameters} = nothing,
        tight_binding_parameters_dn::Union{Nothing, TightBindingParameters} = nothing,
        coupling_parameters::Tuple,
        δG::E, δθ::E, δG_max::E = 1e-6
    ) where {T<:Number, E<:AbstractFloat, D, N, P<:AbstractPropagator}

Make measurements, including time-displaced correlation and zero Matsubara frequency measurements.
This method also returns `(logdetGup, sgndetGup, logdetGdn, sgndetGdn, δG, δθ)`.
Note that either the keywork `tight_binding_parameters` needs to be specified, or
`tight_binding_parameters_up` and `tight_binding_parameters_dn` both need to be specified.
"""
function make_measurements!(
    # ARGUMENTS
    measurement_container::NamedTuple,
    logdetGup::E, sgndetGup::T, Gup::AbstractMatrix{T},
    Gup_ττ::AbstractMatrix{T}, Gup_τ0::AbstractMatrix{T}, Gup_0τ::AbstractMatrix{T},
    logdetGdn::E, sgndetGdn::T, Gdn::AbstractMatrix{T},
    Gdn_ττ::AbstractMatrix{T}, Gdn_τ0::AbstractMatrix{T}, Gdn_0τ::AbstractMatrix{T};
    # KEYWORD ARGUMENTS
    fermion_path_integral_up::FermionPathIntegral{T,E},
    fermion_path_integral_dn::FermionPathIntegral{T,E},
    fermion_greens_calculator_up::FermionGreensCalculator{T,E},
    fermion_greens_calculator_dn::FermionGreensCalculator{T,E},
    Bup::Vector{P}, Bdn::Vector{P},
    model_geometry::ModelGeometry{D,E,N},
    tight_binding_parameters::Union{Nothing, TightBindingParameters} = nothing,
    tight_binding_parameters_up::Union{Nothing, TightBindingParameters} = nothing,
    tight_binding_parameters_dn::Union{Nothing, TightBindingParameters} = nothing,
    coupling_parameters::Tuple,
    δG::E, δθ::E, δG_max::E = 1e-6
) where {T<:Number, E<:AbstractFloat, D, N, P<:AbstractPropagator}

    @assert fermion_path_integral_up.Sb == fermion_path_integral_dn.Sb "$(fermion_path_integral_up.Sb) ≠ $(fermion_path_integral_dn.Sb)"

    # extract temporary storage vectors
    (; 
        time_displaced_correlations,
        equaltime_correlations,
        equaltime_composite_correlations,
        time_displaced_composite_correlations,
        a, a′, a″
    ) = measurement_container
    tmp = selectdim(a, ndims(a), 1)

    # assign spin-up and spin-down tight-binding parameters if necessary
    if !isnothing(tight_binding_parameters)
        tight_binding_parameters_up = tight_binding_parameters
        tight_binding_parameters_dn = tight_binding_parameters
    end

    # calculate sign
    Sb = fermion_path_integral_up.Sb
    sgn = isreal(Sb) ? sign(sgndetGup * sgndetGdn) : sign(exp(-1im*imag(Sb)) * sgndetGup * sgndetGdn)

    # make global measurements
    global_measurements = measurement_container.global_measurements
    make_global_measurements!(
        global_measurements,
        tight_binding_parameters_up,
        tight_binding_parameters_dn,
        coupling_parameters,
        Gup, logdetGup, sgndetGup,
        Gdn, logdetGdn, sgndetGdn,
        sgn, Sb
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

    # make equal-time composite correlation measurements
    make_equaltime_composite_measurements!(
        equaltime_composite_correlations, sgn,
        Gup, Gup_ττ, Gup_τ0, Gup_0τ,
        Gdn, Gdn_ττ, Gdn_τ0, Gdn_0τ,
        model_geometry, tight_binding_parameters_up, tight_binding_parameters_dn,
        fermion_path_integral_up, fermion_path_integral_dn,
        tmp
    )

    # if there are time-displaced measurements to make
    if length(time_displaced_correlations) > 0 || length(time_displaced_composite_correlations) > 0

        # make time-displaced correlation measuresurements for τ = l⋅Δτ = 0
        make_time_displaced_measurements!(
            time_displaced_correlations, 0, sgn,
            Gup, Gup_ττ, Gup_τ0, Gup_0τ, Gdn, Gdn_ττ, Gdn_τ0, Gdn_0τ,
            model_geometry, tight_binding_parameters_up, tight_binding_parameters_dn,
            fermion_path_integral_up, fermion_path_integral_dn
        )

        # make time-displaced composite correlation measuresurements for τ = l⋅Δτ = 0
        make_time_displaced_composite_measurements!(
            time_displaced_composite_correlations, 0, sgn,
            Gup, Gup_ττ, Gup_τ0, Gup_0τ, Gdn, Gdn_ττ, Gdn_τ0, Gdn_0τ,
            model_geometry, tight_binding_parameters_up, tight_binding_parameters_dn,
            fermion_path_integral_up, fermion_path_integral_dn,
            tmp
        )

        # iterate over imaginary time slice
        for l in fermion_greens_calculator_up

            # Propagate Green's function matrices to current imaginary time slice
            propagate_unequaltime_greens!(Gup_τ0, Gup_0τ, Gup_ττ, fermion_greens_calculator_up, Bup)
            propagate_unequaltime_greens!(Gdn_τ0, Gdn_0τ, Gdn_ττ, fermion_greens_calculator_dn, Bdn)

            # make time-displaced correlation measuresurements for τ = l⋅Δτ
            make_time_displaced_measurements!(
                time_displaced_correlations, l, sgn,
                Gup, Gup_ττ, Gup_τ0, Gup_0τ, Gdn, Gdn_ττ, Gdn_τ0, Gdn_0τ,
                model_geometry, tight_binding_parameters_up, tight_binding_parameters_dn,
                fermion_path_integral_up, fermion_path_integral_dn
            )

            # make time-displaced correlation measuresurements for τ = l⋅Δτ
            make_time_displaced_composite_measurements!(
                time_displaced_composite_correlations, l, sgn,
                Gup, Gup_ττ, Gup_τ0, Gup_0τ, Gdn, Gdn_ττ, Gdn_τ0, Gdn_0τ,
                model_geometry, tight_binding_parameters_up, tight_binding_parameters_dn,
                fermion_path_integral_up, fermion_path_integral_dn,
                tmp
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
        # ARGUMENTS
        measurement_container::NamedTuple,
        logdetG::E, sgndetG::T, G::AbstractMatrix{T},
        G_ττ::AbstractMatrix{T}, G_τ0::AbstractMatrix{T}, G_0τ::AbstractMatrix{T};
        # KEYWORD ARGUMENTS
        fermion_path_integral::FermionPathIntegral{T},
        fermion_greens_calculator::FermionGreensCalculator{T,E},
        B::Vector{P},
        model_geometry::ModelGeometry{D,E,N},
        tight_binding_parameters::TightBindingParameters,
        coupling_parameters::Tuple,
        δG::E, δθ::E, δG_max::E = 1e-6
    ) where {T<:Number, E<:AbstractFloat, D, N, P<:AbstractPropagator}

Make measurements, including time-displaced correlation and zero Matsubara frequency measurements.
This method also returns `(logdetG, sgndetG, δG, δθ)`.
"""
function make_measurements!(
    # ARGUMENTS
    measurement_container::NamedTuple,
    logdetG::E, sgndetG::T, G::AbstractMatrix{T},
    G_ττ::AbstractMatrix{T}, G_τ0::AbstractMatrix{T}, G_0τ::AbstractMatrix{T};
    # KEYWORD ARGUMENTS
    fermion_path_integral::FermionPathIntegral{T},
    fermion_greens_calculator::FermionGreensCalculator{T,E},
    B::Vector{P},
    model_geometry::ModelGeometry{D,E,N},
    tight_binding_parameters::TightBindingParameters,
    coupling_parameters::Tuple,
    δG::E, δθ::E, δG_max::E = 1e-6
) where {T<:Number, E<:AbstractFloat, D, N, P<:AbstractPropagator}

    # extract temporary storage vectors
    (;
        time_displaced_correlations,
        equaltime_correlations,
        time_displaced_composite_correlations,
        equaltime_composite_correlations,
        a, a′, a″
    ) = measurement_container
    tmp = selectdim(a, ndims(a), 1)

    # calculate sign
    Sb = fermion_path_integral.Sb
    sgn = isreal(Sb) ? sign(sgndetG^2) : sign(exp(-1im*imag(Sb)) * sgndetG^2)

    # make global measurements
    global_measurements = measurement_container.global_measurements
    make_global_measurements!(
        global_measurements,
        tight_binding_parameters,
        tight_binding_parameters,
        coupling_parameters,
        G, logdetG, sgndetG,
        G, logdetG, sgndetG,
        sgn, Sb
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

    # make equal-time composite correlation measurements
    make_equaltime_composite_measurements!(
        equaltime_composite_correlations, sgn,
        G, G_ττ, G_τ0, G_0τ, G, G_ττ, G_τ0, G_0τ,
        model_geometry, tight_binding_parameters, tight_binding_parameters,
        fermion_path_integral, fermion_path_integral,
        tmp
    )

    # if there are time-displaced measurements to make
    if length(time_displaced_correlations) > 0 || length(time_displaced_composite_correlations) > 0

        # make time-displaced correlation measuresurements of τ = 0
        make_time_displaced_measurements!(
            time_displaced_correlations, 0, sgn,
            G, G_ττ, G_τ0, G_0τ, G, G_ττ, G_τ0, G_0τ,
            model_geometry, tight_binding_parameters, tight_binding_parameters,
            fermion_path_integral, fermion_path_integral,
        )

        # make time-displaced composite correlation measuresurements of τ = 0
        make_time_displaced_composite_measurements!(
            time_displaced_composite_correlations, 0, sgn,
            G, G_ττ, G_τ0, G_0τ, G, G_ττ, G_τ0, G_0τ,
            model_geometry, tight_binding_parameters, tight_binding_parameters,
            fermion_path_integral, fermion_path_integral,
            tmp
        )

        # iterate over imaginary time slice
        for l in fermion_greens_calculator

            # Propagate Green's function matrices to current imaginary time slice
            propagate_unequaltime_greens!(G_τ0, G_0τ, G_ττ, fermion_greens_calculator, B)

            # make time-displaced correlation measuresurements of τ = l⋅Δτ
            make_time_displaced_measurements!(
                time_displaced_correlations, l, sgn,
                G, G_ττ, G_τ0, G_0τ, G, G_ττ, G_τ0, G_0τ,
                model_geometry, tight_binding_parameters, tight_binding_parameters,
                fermion_path_integral, fermion_path_integral
            )

            # make time-displaced composite correlation measuresurements of τ = l⋅Δτ
            make_time_displaced_composite_measurements!(
                time_displaced_composite_correlations, l, sgn,
                G, G_ττ, G_τ0, G_0τ, G, G_ττ, G_τ0, G_0τ,
                model_geometry, tight_binding_parameters, tight_binding_parameters,
                fermion_path_integral, fermion_path_integral,
                tmp
            )

            # Periodically re-calculate the Green's function matrix for numerical stability.
            logdetG, sgndetG, δG′, δθ = stabilize_unequaltime_greens!(G_τ0, G_0τ, G_ττ, logdetG, sgndetG, fermion_greens_calculator, B, update_B̄=false)

            # record maximum stablization error
            δG = max(δG′, δG)
        end
    end

    # determine if electron-phonon parameters were passed
    indx = findfirst(i -> typeof(i) <: ElectronPhononParameters, coupling_parameters)

    # if electron-phonon parameters were passed
    if !isnothing(indx)

        # get electron-phonon coupling parameters
        elph_params = coupling_parameters[indx]

        # measure equal-time phonon greens function
        if haskey(equaltime_correlations, "phonon_greens")
            # measure phonon green's function
            measure_equaltime_phonon_greens!(
                equaltime_correlations["phonon_greens"], elph_params, model_geometry, sgn, a, a′, a″
            )
        end

        # measure time-displaced phonon greens function
        if haskey(time_displaced_correlations, "phonon_greens")
            # measure phonon green's function
            measure_time_displaced_phonon_greens!(
                time_displaced_correlations["phonon_greens"], elph_params, model_geometry, sgn, a, a′, a″
            )
        end

        # iterate over composite equal-time correlations
        for name in keys(equaltime_composite_correlations)
            # check if composite phonon green's function measurement
            if equaltime_composite_correlations[name].correlation == "phonon_greens"
                # measure equal-time composite phonon green's function
                measure_equaltime_composite_phonon_greens!(
                    equaltime_composite_correlations[name], elph_params, model_geometry, sgn, a, a′, a″
                )
            end
        end

        # iterate over composite equal-time correlations
        for name in keys(time_displaced_composite_correlations)
            # check if composite phonon green's function measurement
            if time_displaced_composite_correlations[name].correlation == "phonon_greens"
                # measure equal-time composite phonon green's function
                measure_time_displaced_composite_phonon_greens!(
                    time_displaced_composite_correlations[name], elph_params, model_geometry, sgn, a, a′, a″
                )
            end
        end
    end

    return (logdetG, sgndetG, δG, δθ)
end


##############################
## MAKE GLOBAL MEASUREMENTS ##
##############################

# make global measurements
function make_global_measurements!(
    global_measurements::Dict{String, Complex{E}},
    tight_binding_parameters_up::TightBindingParameters,
    tight_binding_parameters_dn::TightBindingParameters,
    coupling_parameters::Tuple,
    Gup::AbstractMatrix{T}, logdetGup::E, sgndetGup::T,
    Gdn::AbstractMatrix{T}, logdetGdn::E, sgndetGdn::T,
    sgn::T, Sb::T
) where {T<:Number, E<:AbstractFloat}

    # number of orbitals in lattice
    N = size(Gup, 1)

    # measure the sign
    global_measurements["sgn"] += sgn

    # measure the spin resolved sign
    global_measurements["sgndetGup"] += sgndetGup
    global_measurements["sgndetGdn"] += sgndetGdn

    # measure log|det(G)|
    global_measurements["logdetGup"] += logdetGup
    global_measurements["logdetGdn"] += logdetGdn

    # measure fermionic action
    Sf = logdetGup + logdetGdn
    global_measurements["action_fermionic"] += Sf

    # measure bosonic action
    global_measurements["action_bosonic"] += real(Sb)

    # measure total action
    S = real(Sb) + Sf
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
    tight_binding_parameters_up::TightBindingParameters,
    tight_binding_parameters_dn::TightBindingParameters,
    fermion_path_integral_up::FermionPathIntegral{T},
    fermion_path_integral_dn::FermionPathIntegral{T},
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
    make_local_measurements!(
        local_measurements, Gup, Gdn, sgn, model_geometry,
        tight_binding_parameters_up, tight_binding_parameters_dn,
        fermion_path_integral_up, fermion_path_integral_dn
    )

    # make local measurements associated with couplings
    for coupling_parameter in coupling_parameters
        make_local_measurements!(
            local_measurements, Gup, Gdn, sgn, model_geometry,
            coupling_parameter, tight_binding_parameters_up, tight_binding_parameters_dn
        )
    end
    
    return nothing
end

# make local measurements associated with tight-binding model
function make_local_measurements!(
    local_measurements::Dict{String, Vector{Complex{E}}},
    Gup::AbstractMatrix{T}, Gdn::AbstractMatrix{T}, sgn::T,
    model_geometry::ModelGeometry{D,E,N},
    tight_binding_parameters_up::TightBindingParameters,
    tight_binding_parameters_dn::TightBindingParameters,
    fermion_path_integral_up::FermionPathIntegral{T},
    fermion_path_integral_dn::FermionPathIntegral{T}
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
        for hopping_id in 1:nhopping

            # measure bare hopping energy
            hup = sgn * measure_bare_hopping_energy(tight_binding_parameters_up, Gup, hopping_id)
            hdn = sgn * measure_bare_hopping_energy(tight_binding_parameters_dn, Gdn, hopping_id)
            h = hup + hdn
            local_measurements["bare_hopping_energy_up"][hopping_id] += hup
            local_measurements["bare_hopping_energy_dn"][hopping_id] += hdn
            local_measurements["bare_hopping_energy"][hopping_id] += h

            # measure hopping energy
            hup = sgn * measure_hopping_energy(tight_binding_parameters_up, fermion_path_integral_up, Gup, hopping_id)
            hdn = sgn * measure_hopping_energy(tight_binding_parameters_dn, fermion_path_integral_up, Gdn, hopping_id)
            h = hup + hdn
            local_measurements["hopping_energy_up"][hopping_id] += hup
            local_measurements["hopping_energy_dn"][hopping_id] += hdn
            local_measurements["hopping_energy"][hopping_id] += h

            # measure hopping amplitude
            tup = sgn * measure_hopping_amplitude(tight_binding_parameters_up, fermion_path_integral_up, hopping_id)
            tdn = sgn * measure_hopping_amplitude(tight_binding_parameters_dn, fermion_path_integral_up, hopping_id)
            tn = (tup + tup)/2
            local_measurements["hopping_amplitude_up"][hopping_id] += tup
            local_measurements["hopping_amplitude_dn"][hopping_id] += tdn
            local_measurements["hopping_amplitude"][hopping_id] += tn

            # measure hopping inversion
            tup = sgn * measure_hopping_inversion(tight_binding_parameters_up, fermion_path_integral_up, hopping_id)
            tdn = sgn * measure_hopping_inversion(tight_binding_parameters_dn, fermion_path_integral_up, hopping_id)
            tn = (tup + tup)/2
            local_measurements["hopping_inversion_up"][hopping_id] += tup
            local_measurements["hopping_inversion_dn"][hopping_id] += tdn
            local_measurements["hopping_inversion"][hopping_id] += tn

            # measure hopping inversion
            tup = sgn * measure_hopping_inversion_avg(tight_binding_parameters_up, fermion_path_integral_up, hopping_id)
            tdn = sgn * measure_hopping_inversion_avg(tight_binding_parameters_dn, fermion_path_integral_up, hopping_id)
            tn = (tup + tup)/2
            local_measurements["hopping_inversion_avg_up"][hopping_id] += tup
            local_measurements["hopping_inversion_avg_dn"][hopping_id] += tdn
            local_measurements["hopping_inversion_avg"][hopping_id] += tn
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
    tight_binding_parameters_up::TightBindingParameters,
    tight_binding_parameters_dn::TightBindingParameters
) where {T<:Number, E<:AbstractFloat, D, N}

    # measure hubbard energy for each orbital in unit cell
    hubbard_energies = local_measurements["hubbard_energy"]
    for hubbard_id in eachindex(hubbard_energies)
        hubbard_energies[hubbard_id] += sgn * measure_hubbard_energy(hubbard_parameters, Gup, Gdn, hubbard_id)
    end

    return nothing
end


# make local measurements associated with extended hubbard model
function make_local_measurements!(
    local_measurements::Dict{String, Vector{Complex{E}}},
    Gup::AbstractMatrix{T}, Gdn::AbstractMatrix{T}, sgn::T,
    model_geometry::ModelGeometry{D,E,N},
    extended_hubbard_parameters::ExtendedHubbardParameters{E},
    tight_binding_parameters_up::TightBindingParameters,
    tight_binding_parameters_dn::TightBindingParameters
) where {T<:Number, E<:AbstractFloat, D, N}

    # measure hubbard energy for each orbital in unit cell
    ext_hub_energies = local_measurements["ext_hub_energy"]
    for ext_hub_id in eachindex(ext_hub_energies)
        ext_hub_energies[ext_hub_id] += sgn * measure_ext_hub_energy(extended_hubbard_parameters, Gup, Gdn, ext_hub_id)
    end

    return nothing
end


# make local measurements associated with electron-phonon model
function make_local_measurements!(
    local_measurements::Dict{String, Vector{Complex{E}}},
    Gup::AbstractMatrix{T}, Gdn::AbstractMatrix{T}, sgn::T,
    model_geometry::ModelGeometry{D,E,N},
    electron_phonon_parameters::ElectronPhononParameters,
    tight_binding_parameters_up::TightBindingParameters,
    tight_binding_parameters_dn::TightBindingParameters
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
    tight_binding_parameters_up::TightBindingParameters,
    tight_binding_parameters_dn::TightBindingParameters
) where {T<:Number, E<:AbstractFloat, D, N}

    return nothing
end

#######################################################
## MAKE EQUAL-TIME CORRELATION FUNCTION MEASUREMENTS ##
#######################################################

# make purely electronic equal-time correlation measurements
function make_equaltime_measurements!(
    equaltime_correlations::Dict{String, CorrelationContainer{D,E}}, sgn::T,
    Gup::AbstractMatrix{T}, Gup_ττ::AbstractMatrix{T}, Gup_τ0::AbstractMatrix{T}, Gup_0τ::AbstractMatrix{T},
    Gdn::AbstractMatrix{T}, Gdn_ττ::AbstractMatrix{T}, Gdn_τ0::AbstractMatrix{T}, Gdn_0τ::AbstractMatrix{T},
    model_geometry::ModelGeometry{D,E,N},
    tight_binding_parameters_up::TightBindingParameters,
    tight_binding_parameters_dn::TightBindingParameters,
    fermion_path_integral_up::FermionPathIntegral{T},
    fermion_path_integral_dn::FermionPathIntegral{T}
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
        correlations = correlation_container.correlations::Vector{Array{Complex{E}, D}}

        if correlation == "greens"

            for i in eachindex(id_pairs)
                pair = id_pairs[i]
                greens!(correlations[i], pair[2], pair[1], unit_cell, lattice, Gup_τ0, sgn/2)
                greens!(correlations[i], pair[2], pair[1], unit_cell, lattice, Gdn_τ0, sgn/2)
            end

        elseif correlation == "greens_up"

            for i in eachindex(id_pairs)
                pair = id_pairs[i]
                greens!(correlations[i], pair[2], pair[1], unit_cell, lattice, Gup_τ0, sgn)
            end

        elseif correlation == "greens_dn"

            for i in eachindex(id_pairs)
                pair = id_pairs[i]
                greens!(correlations[i], pair[2], pair[1], unit_cell, lattice, Gdn_τ0, sgn)
            end

        elseif correlation == "greens_tautau"

            for i in eachindex(id_pairs)
                pair = id_pairs[i]
                greens!(correlations[i], pair[2], pair[1], unit_cell, lattice, Gup_τ0, sgn/2)
                greens!(correlations[i], pair[2], pair[1], unit_cell, lattice, Gdn_τ0, sgn/2)
            end

        elseif correlation == "greens_tautau_up"

            for i in eachindex(id_pairs)
                pair = id_pairs[i]
                greens!(correlations[i], pair[2], pair[1], unit_cell, lattice, Gup_τ0, sgn)
            end

        elseif correlation == "greens_tautau_dn"

            for i in eachindex(id_pairs)
                pair = id_pairs[i]
                greens!(correlations[i], pair[2], pair[1], unit_cell, lattice, Gdn_τ0, sgn)
            end    

        elseif correlation == "density_upup"

            for i in eachindex(id_pairs)
                pair = id_pairs[i]
                density_correlation!(correlations[i], pair[2], pair[1], unit_cell, lattice,
                                     Gup_τ0, Gup_0τ, Gup_ττ, Gup, +1, +1, sgn)
            end

        elseif correlation == "density_dndn"

            for i in eachindex(id_pairs)
                pair = id_pairs[i]
                density_correlation!(correlations[i], pair[2], pair[1], unit_cell, lattice,
                                     Gdn_τ0, Gdn_0τ, Gdn_ττ, Gdn, -1, -1, sgn)
            end

        elseif correlation == "density_updn"

            for i in eachindex(id_pairs)
                pair = id_pairs[i]
                density_correlation!(correlations[i], pair[2], pair[1], unit_cell, lattice,
                                     Gup_τ0, Gup_0τ, Gup_ττ, Gdn, +1, -1, sgn)
            end

        elseif correlation == "density_dnup"

            for i in eachindex(id_pairs)
                pair = id_pairs[i]
                density_correlation!(correlations[i], pair[2], pair[1], unit_cell, lattice,
                                     Gdn_τ0, Gdn_0τ, Gdn_ττ, Gup, -1, +1, sgn)
            end

        elseif correlation == "density"

            for i in eachindex(id_pairs)
                pair = id_pairs[i]
                density_correlation!(correlations[i], pair[2], pair[1], unit_cell, lattice,
                                     Gup_τ0, Gup_0τ, Gup_ττ, Gup, Gdn_τ0, Gdn_0τ, Gdn_ττ, Gdn, sgn)
            end

        elseif correlation == "pair"

            for i in eachindex(id_pairs)
                pair = id_pairs[i]
                pair_correlation!(correlations[i], bonds[pair[2]], bonds[pair[1]], unit_cell, lattice,
                                  Gup_τ0, Gdn_τ0, sgn)
            end

        elseif correlation == "spin_x"

            for i in eachindex(id_pairs)
                pair = id_pairs[i]
                spin_x_correlation!(correlations[i], pair[2], pair[1], unit_cell, lattice,
                                    Gup_τ0, Gup_0τ, Gdn_τ0, Gdn_0τ, sgn)
            end

        elseif correlation == "spin_z"

            for i in eachindex(id_pairs)
                pair = id_pairs[i]
                spin_z_correlation!(correlations[i], pair[2], pair[1], unit_cell, lattice,
                                    Gup_τ0, Gup_0τ, Gup_ττ, Gup, Gdn_τ0, Gdn_0τ, Gdn_ττ, Gdn, sgn)
            end

        elseif correlation == "bond_upup"

            for i in eachindex(id_pairs)
                pair = id_pairs[i]
                bond_correlation!(correlations[i], bonds[pair[2]], bonds[pair[1]], unit_cell, lattice,
                                  Gup_τ0, Gup_0τ, Gup_ττ, Gup, +1, +1, sgn)
            end

        elseif correlation == "bond_dndn"

            for i in eachindex(id_pairs)
                pair = id_pairs[i]
                bond_correlation!(correlations[i], bonds[pair[2]], bonds[pair[1]], unit_cell, lattice,
                                  Gdn_τ0, Gdn_0τ, Gdn_ττ, Gdn, -1, -1, sgn)
            end

        elseif correlation == "bond_updn"

            for i in eachindex(id_pairs)
                pair = id_pairs[i]
                bond_correlation!(correlations[i], bonds[pair[2]], bonds[pair[1]], unit_cell, lattice,
                                  Gup_τ0, Gup_0τ, Gup_ττ, Gdn, +1, -1, sgn)
            end

        elseif correlation == "bond_dnup"

            for i in eachindex(id_pairs)
                pair = id_pairs[i]
                bond_correlation!(correlations[i], bonds[pair[2]], bonds[pair[1]], unit_cell, lattice,
                                  Gdn_τ0, Gdn_0τ, Gdn_ττ, Gup, -1, +1, sgn)
            end

        elseif correlation == "bond"

            for i in eachindex(id_pairs)
                pair = id_pairs[i]
                bond_correlation!(correlations[i], bonds[pair[2]], bonds[pair[1]], unit_cell, lattice,
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
                # get the bond definitions
                bond_0 = bonds[bond_id_0]
                bond_1 = bonds[bond_id_1]
                # get the effective hopping amptlitudes for each of the two hopping ID's in question
                tup0 = @view tup[bond_slices[hopping_id_0], Lτ]
                tup0′ = reshape(tup0, lattice.L...)
                tup1 = @view tup[bond_slices[hopping_id_1], Lτ]
                tup1′ = reshape(tup1, lattice.L...)
                # measure the current-current correlation
                current_correlation!(correlations[i], bond_1, bond_0, tup1′, tup0′, unit_cell, lattice,
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
                # get the bond definitions
                bond_0 = bonds[bond_id_0]
                bond_1 = bonds[bond_id_1]
                # get the effective hopping amptlitudes for each of the two hopping ID's in question
                tdn0 = @view tdn[bond_slices[hopping_id_0], Lτ]
                tdn0′ = reshape(tdn0, lattice.L...)
                tdn1 = @view tdn[bond_slices[hopping_id_1], Lτ]
                tdn1′ = reshape(tdn1, lattice.L...)
                # measure the current-current correlation
                current_correlation!(correlations[i], bond_1, bond_0, tdn1′, tdn0′, unit_cell, lattice,
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
                # get the bond definitions
                bond_0 = bonds[bond_id_0]
                bond_1 = bonds[bond_id_1]
                # get the effective hopping amptlitudes for each of the two hopping ID's in question
                tup1 = @view tup[bond_slices[hopping_id_1], Lτ]
                tup1′ = reshape(tup1, lattice.L...)
                tdn0 = @view tdn[bond_slices[hopping_id_0], Lτ]
                tdn0′ = reshape(tdn0, lattice.L...)
                # measure the current-current correlation
                current_correlation!(correlations[i], bond_1, bond_0, tup1′, tdn0′, unit_cell, lattice,
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
                # get the bond definitions
                bond_0 = bonds[bond_id_0]
                bond_1 = bonds[bond_id_1]
                # get the effective hopping amptlitudes for each of the two hopping ID's in question
                tup0 = @view tup[bond_slices[hopping_id_0], Lτ]
                tup0′ = reshape(tup0, lattice.L...)
                tdn1 = @view tdn[bond_slices[hopping_id_1], Lτ]
                tdn1′ = reshape(tdn1, lattice.L...)
                # measure the current-current correlation
                current_correlation!(correlations[i], bond_1, bond_0, tdn1′, tup0′, unit_cell, lattice,
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
                current_correlation!(correlations[i], bond_1, bond_0, tup1′, tup0′, tdn1′, tdn0′, unit_cell, lattice,
                                     Gup_τ0, Gup_0τ, Gup_ττ, Gup, Gdn_τ0, Gdn_0τ, Gdn_ττ, Gdn, sgn)
            end
        end
    end

    return nothing
end


#################################################################
## MAKE EQUAL-TIME COMPOSITE CORRELATION FUNCTION MEASUREMENTS ##
#################################################################

# make purely electronic equal-time composite correlation measurements
function make_equaltime_composite_measurements!(
    equaltime_composite_correlations::Dict{String, CompositeCorrelationContainer{D,D,E}}, sgn::T,
    Gup::AbstractMatrix{T}, Gup_ττ::AbstractMatrix{T}, Gup_τ0::AbstractMatrix{T}, Gup_0τ::AbstractMatrix{T},
    Gdn::AbstractMatrix{T}, Gdn_ττ::AbstractMatrix{T}, Gdn_τ0::AbstractMatrix{T}, Gdn_0τ::AbstractMatrix{T},
    model_geometry::ModelGeometry{D,E,N},
    tight_binding_parameters_up::TightBindingParameters,
    tight_binding_parameters_dn::TightBindingParameters,
    fermion_path_integral_up::FermionPathIntegral{T},
    fermion_path_integral_dn::FermionPathIntegral{T},
    tmp::AbstractArray{Complex{E}, D}
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
    for name in keys(equaltime_composite_correlations)
        
        correlation_container = equaltime_composite_correlations[name]::CompositeCorrelationContainer{D,D,E}
        correlation = correlation_container.correlation
        id_pairs = correlation_container.id_pairs::Vector{NTuple{2,Int}}
        coefficients = correlation_container.coefficients::Vector{Complex{E}}
        correlations = correlation_container.correlations::Array{Complex{E}, D}
        structure_factors = correlation_container.structure_factors::Array{Complex{E}, D}
        displacement_vecs = correlation_container.displacement_vecs::Vector{SVector{D,E}}

        if correlation == "greens"

            for i in eachindex(id_pairs)
                fill!(tmp, 0)
                coef = coefficients[i]
                id_pair = id_pairs[i]
                greens!(tmp, id_pair[2], id_pair[1], unit_cell, lattice, Gup_τ0, coef*sgn/2)
                greens!(tmp, id_pair[2], id_pair[1], unit_cell, lattice, Gdn_τ0, coef*sgn/2)
                @. correlations += tmp
                fourier_transform!(tmp, displacement_vecs[i], unit_cell, lattice)
                @. structure_factors += tmp
            end

        elseif correlation == "greens_up"

            for i in eachindex(id_pairs)
                fill!(tmp, 0)
                coef = coefficients[i]
                id_pair = id_pairs[i]
                greens!(tmp, id_pair[2], id_pair[1], unit_cell, lattice, Gup_τ0, coef*sgn)
                @. correlations += tmp
                fourier_transform!(tmp, displacement_vecs[i], unit_cell, lattice)
                @. structure_factors += tmp
            end

        elseif correlation == "greens_dn"

            for i in eachindex(id_pairs)
                fill!(tmp, 0)
                coef = coefficients[i]
                id_pair = id_pairs[i]
                greens!(tmp, id_pair[2], id_pair[1], unit_cell, lattice, Gdn_τ0, coef*sgn)
                @. correlations += tmp
                fourier_transform!(tmp, displacement_vecs[i], unit_cell, lattice)
                @. structure_factors += tmp
            end

        elseif correlation == "greens_tautau"

            for i in eachindex(id_pairs)
                fill!(tmp, 0)
                coef = coefficients[i]
                id_pair = id_pairs[i]
                greens!(tmp, id_pair[2], id_pair[1], unit_cell, lattice, Gup_τ0, coef*sgn/2)
                greens!(tmp, id_pair[2], id_pair[1], unit_cell, lattice, Gdn_τ0, coef*sgn/2)
                @. correlations += tmp
                fourier_transform!(tmp, displacement_vecs[i], unit_cell, lattice)
                @. structure_factors += tmp
            end

        elseif correlation == "greens_tautau_up"

            for i in eachindex(id_pairs)
                fill!(tmp, 0)
                coef = coefficients[i]
                id_pair = id_pairs[i]
                greens!(tmp, id_pair[2], id_pair[1], unit_cell, lattice, Gup_τ0, coef*sgn)
                @. correlations += tmp
                fourier_transform!(tmp, displacement_vecs[i], unit_cell, lattice)
                @. structure_factors += tmp
            end

        elseif correlation == "greens_tautau_dn"

            for i in eachindex(id_pairs)
                fill!(tmp, 0)
                coef = coefficients[i]
                id_pair = id_pairs[i]
                greens!(tmp, id_pair[2], id_pair[1], unit_cell, lattice, Gdn_τ0, coef*sgn)
                @. correlations += tmp
                fourier_transform!(tmp, displacement_vecs[i], unit_cell, lattice)
                @. structure_factors += tmp
            end    

        elseif correlation == "density_upup"

            for i in eachindex(id_pairs)
                fill!(tmp, 0)
                coef = coefficients[i]
                id_pair = id_pairs[i]
                density_correlation!(
                    tmp, id_pair[2], id_pair[1], unit_cell, lattice,
                    Gup_τ0, Gup_0τ, Gup_ττ, Gup, +1, +1, coef*sgn
                )
                @. correlations += tmp
                fourier_transform!(tmp, displacement_vecs[i], unit_cell, lattice)
                @. structure_factors += tmp
            end

        elseif correlation == "density_dndn"

            for i in eachindex(id_pairs)
                fill!(tmp, 0)
                coef = coefficients[i]
                id_pair = id_pairs[i]
                density_correlation!(
                    tmp, id_pair[2], id_pair[1], unit_cell, lattice,
                    Gdn_τ0, Gdn_0τ, Gdn_ττ, Gdn, -1, -1, coef*sgn
                )
                @. correlations += tmp
                fourier_transform!(tmp, displacement_vecs[i], unit_cell, lattice)
                @. structure_factors += tmp
            end

        elseif correlation == "density_updn"

            for i in eachindex(id_pairs)
                fill!(tmp, 0)
                coef = coefficients[i]
                id_pair = id_pairs[i]
                density_correlation!(
                    tmp, id_pair[2], id_pair[1], unit_cell, lattice,
                    Gup_τ0, Gup_0τ, Gup_ττ, Gdn, +1, -1, coef*sgn
                )
                @. correlations += tmp
                fourier_transform!(tmp, displacement_vecs[i], unit_cell, lattice)
                @. structure_factors += tmp
            end

        elseif correlation == "density_dnup"

            for i in eachindex(id_pairs)
                fill!(tmp, 0)
                coef = coefficients[i]
                id_pair = id_pairs[i]
                density_correlation!(
                    tmp, id_pair[2], id_pair[1], unit_cell, lattice,
                    Gdn_τ0, Gdn_0τ, Gdn_ττ, Gup, -1, +1, coef*sgn
                )
                @. correlations += tmp
                fourier_transform!(tmp, displacement_vecs[i], unit_cell, lattice)
                @. structure_factors += tmp
            end

        elseif correlation == "density"

            for i in eachindex(id_pairs)
                fill!(tmp, 0)
                coef = coefficients[i]
                id_pair = id_pairs[i]
                density_correlation!(
                    tmp, id_pair[2], id_pair[1], unit_cell, lattice,
                    Gup_τ0, Gup_0τ, Gup_ττ, Gup, Gdn_τ0, Gdn_0τ, Gdn_ττ, Gdn, coef*sgn
                )
                @. correlations += tmp
                fourier_transform!(tmp, displacement_vecs[i], unit_cell, lattice)
                @. structure_factors += tmp
            end

        elseif correlation == "pair"

            for i in eachindex(id_pairs)
                fill!(tmp, 0)
                coef = coefficients[i]
                id_pair = id_pairs[i]
                pair_correlation!(
                    tmp, bonds[id_pair[2]], bonds[id_pair[1]], unit_cell, lattice,
                    Gup_τ0, Gdn_τ0, coef*sgn
                )
                @. correlations += tmp
                fourier_transform!(tmp, displacement_vecs[i], unit_cell, lattice)
                @. structure_factors += tmp
            end

        elseif correlation == "spin_x"

            for i in eachindex(id_pairs)
                fill!(tmp, 0)
                coef = coefficients[i]
                id_pair = id_pairs[i]
                spin_x_correlation!(
                    tmp, id_pair[2], id_pair[1], unit_cell, lattice,
                    Gup_τ0, Gup_0τ, Gdn_τ0, Gdn_0τ, coef*sgn
                )
                @. correlations += tmp
                fourier_transform!(tmp, displacement_vecs[i], unit_cell, lattice)
                @. structure_factors += tmp
            end

        elseif correlation == "spin_z"

            for i in eachindex(id_pairs)
                fill!(tmp, 0)
                coef = coefficients[i]
                id_pair = id_pairs[i]
                spin_z_correlation!(
                    tmp, id_pair[2], id_pair[1], unit_cell, lattice,
                    Gup_τ0, Gup_0τ, Gup_ττ, Gup, Gdn_τ0, Gdn_0τ, Gdn_ττ, Gdn, coef*sgn
                )
                @. correlations += tmp
                fourier_transform!(tmp, displacement_vecs[i], unit_cell, lattice)
                @. structure_factors += tmp
            end

        elseif correlation == "bond_upup"

            for i in eachindex(id_pairs)
                fill!(tmp, 0)
                coef = coefficients[i]
                id_pair = id_pairs[i]
                bond_correlation!(
                    tmp, bonds[id_pair[2]], bonds[id_pair[1]], unit_cell, lattice,
                    Gup_τ0, Gup_0τ, Gup_ττ, Gup, +1, +1, coef*sgn
                )
                @. correlations += tmp
                fourier_transform!(tmp, displacement_vecs[i], unit_cell, lattice)
                @. structure_factors += tmp
            end

        elseif correlation == "bond_dndn"

            for i in eachindex(id_pairs)
                fill!(tmp, 0)
                coef = coefficients[i]
                id_pair = id_pairs[i]
                bond_correlation!(
                    tmp, bonds[id_pair[2]], bonds[id_pair[1]], unit_cell, lattice,
                    Gdn_τ0, Gdn_0τ, Gdn_ττ, Gdn, -1, -1, coef*sgn
                )
                @. correlations += tmp
                fourier_transform!(tmp, displacement_vecs[i], unit_cell, lattice)
                @. structure_factors += tmp
            end

        elseif correlation == "bond_updn"

            for i in eachindex(id_pairs)
                fill!(tmp, 0)
                coef = coefficients[i]
                id_pair = id_pairs[i]
                bond_correlation!(
                    tmp, bonds[id_pair[2]], bonds[id_pair[1]], unit_cell, lattice,
                    Gup_τ0, Gup_0τ, Gup_ττ, Gdn, +1, -1, coef*sgn
                )
                @. correlations += tmp
                fourier_transform!(tmp, displacement_vecs[i], unit_cell, lattice)
                @. structure_factors += tmp
            end

        elseif correlation == "bond_dnup"

            for i in eachindex(id_pairs)
                fill!(tmp, 0)
                coef = coefficients[i]
                id_pair = id_pairs[i]
                bond_correlation!(
                    tmp, bonds[id_pair[2]], bonds[id_pair[1]], unit_cell, lattice,
                    Gdn_τ0, Gdn_0τ, Gdn_ττ, Gup, -1, +1, coef*sgn
                )
                @. correlations += tmp
                fourier_transform!(tmp, displacement_vecs[i], unit_cell, lattice)
                @. structure_factors += tmp
            end

        elseif correlation == "bond"

            for i in eachindex(id_pairs)
                fill!(tmp, 0)
                coef = coefficients[i]
                id_pair = id_pairs[i]
                bond_correlation!(
                    tmp, bonds[id_pair[2]], bonds[id_pair[1]], unit_cell, lattice,
                    Gup_τ0, Gup_0τ, Gup_ττ, Gup, Gdn_τ0, Gdn_0τ, Gdn_ττ, Gdn, coef*sgn
                )
                @. correlations += tmp
                fourier_transform!(tmp, displacement_vecs[i], unit_cell, lattice)
                @. structure_factors += tmp
            end

        elseif correlation == "current_upup"

            (; bond_ids, bond_slices) = tight_binding_parameters_up
            tup = fermion_path_integral_up.t

            for i in eachindex(ids)
                fill!(tmp, 0)
                coef = coefficients[i]
                id_pair = id_pairs[i]
                # get the hopping IDs associated with current operators
                hopping_id_0 = id_pair[1]
                hopping_id_1 = id_pair[2]
                # get the bond IDs associated with the hopping IDs
                bond_id_0 = bond_ids[hopping_id_0]
                bond_id_1 = bond_ids[hopping_id_1]
                # get the bond definitions
                bond_0 = bonds[bond_id_0]
                bond_1 = bonds[bond_id_1]
                # get the effective hopping amptlitudes for each of the two hopping ID's in question
                tup0 = @view tup[bond_slices[hopping_id_0], Lτ]
                tup0′ = reshape(tup0, lattice.L...)
                tup1 = @view tup[bond_slices[hopping_id_1], Lτ]
                tup1′ = reshape(tup1, lattice.L...)
                # measure the current-current correlation
                current_correlation!(
                    tmp, bond_1, bond_0, tup1′, tup0′, unit_cell, lattice,
                    Gup_τ0, Gup_0τ, Gup_ττ, Gup, +1, +1, coef*sgn
                )
                @. correlations += tmp
                fourier_transform!(tmp, displacement_vecs[i], unit_cell, lattice)
                @. structure_factors += tmp
            end

        elseif correlation == "current_dndn"

            (; bond_ids, bond_slices) = tight_binding_parameters_dn
            tdn = fermion_path_integral_dn.t

            for i in eachindex(ids)
                fill!(tmp, 0)
                coef = coefficients[i]
                id_pair = id_pairs[i]
                # get the hopping IDs associated with current operators
                hopping_id_0 = id_pair[1]
                hopping_id_1 = id_pair[2]
                # get the bond IDs associated with the hopping IDs
                bond_id_0 = bond_ids[hopping_id_0]
                bond_id_1 = bond_ids[hopping_id_1]
                # get the bond definitions
                bond_0 = bonds[bond_id_0]
                bond_1 = bonds[bond_id_1]
                # get the effective hopping amptlitudes for each of the two hopping ID's in question
                tdn0 = @view tdn[bond_slices[hopping_id_0], Lτ]
                tdn0′ = reshape(tdn0, lattice.L...)
                tdn1 = @view tdn[bond_slices[hopping_id_1], Lτ]
                tdn1′ = reshape(tdn1, lattice.L...)
                # measure the current-current correlation
                current_correlation!(
                    tmp, bond_1, bond_0, tdn1′, tdn0′, unit_cell, lattice,
                    Gdn_τ0, Gdn_0τ, Gdn_ττ, Gdn, -1, -1, coef*sgn
                )
                @. correlations += tmp
                fourier_transform!(tmp, displacement_vecs[i], unit_cell, lattice)
                @. structure_factors += tmp
            end

        elseif correlation == "current_updn"

            (; bond_ids, bond_slices) = tight_binding_parameters_up
            tup = fermion_path_integral_up.t
            tdn = fermion_path_integral_dn.t

            for i in eachindex(ids)
                fill!(tmp, 0)
                coef = coefficients[i]
                id_pair = id_pairs[i]
                # get the hopping IDs associated with current operators
                hopping_id_0 = id_pair[1]
                hopping_id_1 = id_pair[2]
                # get the bond IDs associated with the hopping IDs
                bond_id_0 = bond_ids[hopping_id_0]
                bond_id_1 = bond_ids[hopping_id_1]
                # get the bond definitions
                bond_0 = bonds[bond_id_0]
                bond_1 = bonds[bond_id_1]
                # get the effective hopping amptlitudes for each of the two hopping ID's in question
                tdn0 = @view tdn[bond_slices[hopping_id_0], Lτ]
                tdn0′ = reshape(tdn0, lattice.L...)
                tup1 = @view tup[bond_slices[hopping_id_1], Lτ]
                tup1′ = reshape(tup1, lattice.L...)
                # measure the current-current correlation
                current_correlation!(
                    tmp, bond_1, bond_0, tup1′, tdn0′, unit_cell, lattice,
                    Gup_τ0, Gup_0τ, Gup_ττ, Gdn, +1, -1, coef*sgn
                )
                @. correlations += tmp
                fourier_transform!(tmp, displacement_vecs[i], unit_cell, lattice)
                @. structure_factors += tmp
            end

        elseif correlation == "current_dnup"

            (; bond_ids, bond_slices) = tight_binding_parameters_up
            tup = fermion_path_integral_up.t
            tdn = fermion_path_integral_dn.t

            for i in eachindex(id_pairs)
                fill!(tmp, 0)
                coef = coefficients[i]
                id_pair = id_pairs[i]
                # get the hopping IDs associated with current operators
                hopping_id_0 = id_pair[1]
                hopping_id_1 = id_pair[2]
                # get the bond IDs associated with the hopping IDs
                bond_id_0 = bond_ids[hopping_id_0]
                bond_id_1 = bond_ids[hopping_id_1]
                # get the bond definitions
                bond_0 = bonds[bond_id_0]
                bond_1 = bonds[bond_id_1]
                # get the effective hopping amptlitudes for each of the two hopping ID's in question
                tup0 = @view tup[bond_slices[hopping_id_0], Lτ]
                tup0′ = reshape(tup0, lattice.L...)
                tdn1 = @view tdn[bond_slices[hopping_id_1], Lτ]
                tdn1′ = reshape(tdn1, lattice.L...)
                # measure the current-current correlation
                current_correlation!(
                    tmp, bond_1, bond_0, tdn1′, tup0′, unit_cell, lattice,
                    Gdn_τ0, Gdn_0τ, Gdn_ττ, Gup, -1, +1, coef*sgn
                )
                @. correlations += tmp
                fourier_transform!(tmp, displacement_vecs[i], unit_cell, lattice)
                @. structure_factors += tmp
            end

        elseif correlation == "current"

            (; bond_ids, bond_slices) = tight_binding_parameters_up
            tup = fermion_path_integral_up.t
            tdn = fermion_path_integral_dn.t

            for i in eachindex(id_pairs)
                fill!(tmp, 0)
                coef = coefficients[i]
                id_pair = id_pairs[i]
                # get the hopping IDs associated with current operators
                hopping_id_0 = id_pair[1]
                hopping_id_1 = id_pair[2]
                # get the bond IDs associated with the hopping IDs
                bond_id_0 = bond_ids[hopping_id_0]
                bond_id_1 = bond_ids[hopping_id_1]
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
                current_correlation!(
                    tmp, bond_1, bond_0, tup1′, tup0′, tdn1′, tdn0′, unit_cell, lattice,
                    Gup_τ0, Gup_0τ, Gup_ττ, Gup, Gdn_τ0, Gdn_0τ, Gdn_ττ, Gdn, coef*sgn
                )
                @. correlations += tmp
                fourier_transform!(tmp, displacement_vecs[i], unit_cell, lattice)
                @. structure_factors += tmp
            end
        end
    end

    return nothing
end


###########################################################
## MAKE TIME-DISPLACED CORRELATION FUNCTION MEASUREMENTS ##
###########################################################

# make purely electronic time-displaced correlation measurements
function make_time_displaced_measurements!(
    time_displaced_correlations::Dict{String, CorrelationContainer{P,E}}, l::Int, sgn::T,
    Gup::AbstractMatrix{T}, Gup_ττ::AbstractMatrix{T}, Gup_τ0::AbstractMatrix{T}, Gup_0τ::AbstractMatrix{T},
    Gdn::AbstractMatrix{T}, Gdn_ττ::AbstractMatrix{T}, Gdn_τ0::AbstractMatrix{T}, Gdn_0τ::AbstractMatrix{T},
    model_geometry::ModelGeometry{D,E,N},
    tight_binding_parameters_up::TightBindingParameters,
    tight_binding_parameters_dn::TightBindingParameters,
    fermion_path_integral_up::FermionPathIntegral{T},
    fermion_path_integral_dn::FermionPathIntegral{T}
) where {T<:Number, E<:AbstractFloat, P, D, N}

    Lτ = fermion_path_integral_up.Lτ::Int
    unit_cell = model_geometry.unit_cell::UnitCell{D,E,N}
    lattice = model_geometry.lattice::Lattice{D}
    bonds = model_geometry.bonds::Vector{Bond{D}}

    # iterate over time-displaced correlation function getting measured
    for correlation in keys(time_displaced_correlations)
        
        correlation_container = time_displaced_correlations[correlation]::CorrelationContainer{P,E}
        id_pairs = correlation_container.id_pairs::Vector{NTuple{2,Int}}
        correlations = correlation_container.correlations::Vector{Array{Complex{E}, P}}

        if correlation == "greens"

            for i in eachindex(id_pairs)
                pair = id_pairs[i]
                correlation_array = selectdim(correlations[i], D+1, l+1)
                greens!(correlation_array, pair[2], pair[1], unit_cell, lattice, Gup_τ0, sgn/2)
                greens!(correlation_array, pair[2], pair[1], unit_cell, lattice, Gdn_τ0, sgn/2)
            end

        elseif correlation == "greens_up"

            for i in eachindex(id_pairs)
                pair = id_pairs[i]
                correlation_array = selectdim(correlations[i], D+1, l+1)
                greens!(correlation_array, pair[2], pair[1], unit_cell, lattice, Gup_τ0, sgn)
            end

        elseif correlation == "greens_dn"

            for i in eachindex(id_pairs)
                pair = id_pairs[i]
                correlation_array = selectdim(correlations[i], D+1, l+1)
                greens!(correlation_array, pair[2], pair[1], unit_cell, lattice, Gdn_τ0, sgn)
            end

        elseif correlation == "greens_tautau"

            for i in eachindex(id_pairs)
                pair = id_pairs[i]
                correlation_array = selectdim(correlations[i], D+1, l+1)
                greens!(correlation_array, pair[2], pair[1], unit_cell, lattice, Gup_ττ, sgn/2)
                greens!(correlation_array, pair[2], pair[1], unit_cell, lattice, Gdn_ττ, sgn/2)
            end

        elseif correlation == "greens_tautau_up"

            for i in eachindex(id_pairs)
                pair = id_pairs[i]
                correlation_array = selectdim(correlations[i], D+1, l+1)
                greens!(correlation_array, pair[2], pair[1], unit_cell, lattice, Gup_ττ, sgn)
            end

        elseif correlation == "greens_tautau_dn"

            for i in eachindex(id_pairs)
                pair = id_pairs[i]
                correlation_array = selectdim(correlations[i], D+1, l+1)
                greens!(correlation_array, pair[2], pair[1], unit_cell, lattice, Gdn_ττ, sgn)
            end

        elseif correlation == "density_upup"

            for i in eachindex(id_pairs)
                pair = id_pairs[i]
                correlation_array = selectdim(correlations[i], D+1, l+1)
                density_correlation!(correlation_array, pair[2], pair[1], unit_cell, lattice,
                                     Gup_τ0, Gup_0τ, Gup_ττ, Gup, +1, +1, sgn)
            end

        elseif correlation == "density_dndn"

            for i in eachindex(id_pairs)
                pair = id_pairs[i]
                correlation_array = selectdim(correlations[i], D+1, l+1)
                density_correlation!(correlation_array, pair[2], pair[1], unit_cell, lattice,
                                     Gdn_τ0, Gdn_0τ, Gdn_ττ, Gdn, -1, -1, sgn)
            end

        elseif correlation == "density_updn"

            for i in eachindex(id_pairs)
                pair = id_pairs[i]
                correlation_array = selectdim(correlations[i], D+1, l+1)
                density_correlation!(correlation_array, pair[2], pair[1], unit_cell, lattice,
                                     Gup_τ0, Gup_0τ, Gup_ττ, Gdn, +1, -1, sgn)
            end

        elseif correlation == "density_dnup"

            for i in eachindex(id_pairs)
                pair = id_pairs[i]
                correlation_array = selectdim(correlations[i], D+1, l+1)
                density_correlation!(correlation_array, pair[2], pair[1], unit_cell, lattice,
                                     Gdn_τ0, Gdn_0τ, Gdn_ττ, Gup, -1, +1, sgn)
            end

        elseif correlation == "density"

            for i in eachindex(id_pairs)
                pair = id_pairs[i]
                correlation_array = selectdim(correlations[i], D+1, l+1)
                density_correlation!(correlation_array, pair[2], pair[1], unit_cell, lattice,
                                     Gup_τ0, Gup_0τ, Gup_ττ, Gup, Gdn_τ0, Gdn_0τ, Gdn_ττ, Gdn, sgn)
            end

        elseif correlation == "pair"

            for i in eachindex(id_pairs)
                pair = id_pairs[i]
                correlation_array = selectdim(correlations[i], D+1, l+1)
                pair_correlation!(correlation_array, bonds[pair[2]], bonds[pair[1]], unit_cell, lattice,
                                  Gup_τ0, Gdn_τ0, sgn)
            end

        elseif correlation == "spin_x"

            for i in eachindex(id_pairs)
                pair = id_pairs[i]
                correlation_array = selectdim(correlations[i], D+1, l+1)
                spin_x_correlation!(correlation_array, pair[2], pair[1], unit_cell, lattice,
                                    Gup_τ0, Gup_0τ, Gdn_τ0, Gdn_0τ, sgn)
            end

        elseif correlation == "spin_z"

            for i in eachindex(id_pairs)
                pair = id_pairs[i]
                correlation_array = selectdim(correlations[i], D+1, l+1)
                spin_z_correlation!(correlation_array, pair[2], pair[1], unit_cell, lattice,
                                    Gup_τ0, Gup_0τ, Gup_ττ, Gup, Gdn_τ0, Gdn_0τ, Gdn_ττ, Gdn, sgn)
            end

        elseif correlation == "bond_upup"

            for i in eachindex(id_pairs)
                pair = id_pairs[i]
                correlation_array = selectdim(correlations[i], D+1, l+1)
                bond_correlation!(correlation_array, bonds[pair[2]], bonds[pair[1]], unit_cell, lattice,
                                  Gup_τ0, Gup_0τ, Gup_ττ, Gup, +1, +1, sgn)
            end

        elseif correlation == "bond_dndn"

            for i in eachindex(id_pairs)
                pair = id_pairs[i]
                correlation_array = selectdim(correlations[i], D+1, l+1)
                bond_correlation!(correlation_array, bonds[pair[2]], bonds[pair[1]], unit_cell, lattice,
                                  Gdn_τ0, Gdn_0τ, Gdn_ττ, Gdn, -1, -1, sgn)
            end

        elseif correlation == "bond_updn"

            for i in eachindex(id_pairs)
                pair = id_pairs[i]
                correlation_array = selectdim(correlations[i], D+1, l+1)
                bond_correlation!(correlation_array, bonds[pair[2]], bonds[pair[1]], unit_cell, lattice,
                                  Gup_τ0, Gup_0τ, Gup_ττ, Gdn, +1, -1, sgn)
            end

        elseif correlation == "bond_dnup"

            for i in eachindex(id_pairs)
                pair = id_pairs[i]
                correlation_array = selectdim(correlations[i], D+1, l+1)
                bond_correlation!(correlation_array, bonds[pair[2]], bonds[pair[1]], unit_cell, lattice,
                                  Gdn_τ0, Gdn_0τ, Gdn_ττ, Gup, -1, +1, sgn)
            end

        elseif correlation == "bond"

            for i in eachindex(id_pairs)
                pair = id_pairs[i]
                correlation_array = selectdim(correlations[i], D+1, l+1)
                bond_correlation!(correlation_array, bonds[pair[2]], bonds[pair[1]], unit_cell, lattice,
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
                # get the bond definitions
                bond_0 = bonds[bond_id_0]
                bond_1 = bonds[bond_id_1]
                # get the effective hopping amptlitudes for each of the two hopping ID's in question
                tup0 = @view tup[bond_slices[hopping_id_0], Lτ]
                tup0′ = reshape(tup0, lattice.L...)
                tup1 = @view tup[bond_slices[hopping_id_1], mod1(l,Lτ)]
                tup1′ = reshape(tup1, lattice.L...)
                # measure the current-current correlation
                correlation_array = selectdim(correlations[i], D+1, l+1)
                current_correlation!(correlation_array, bond_1, bond_0, tup1′, tup0′, unit_cell, lattice,
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
                # get the bond definitions
                bond_0 = bonds[bond_id_0]
                bond_1 = bonds[bond_id_1]
                # get the effective hopping amptlitudes for each of the two hopping ID's in question
                tdn0 = @view tdn[bond_slices[hopping_id_0], Lτ]
                tdn0′ = reshape(tdn0, lattice.L...)
                tdn1 = @view tdn[bond_slices[hopping_id_1], mod1(l,Lτ)]
                tdn1′ = reshape(tdn1, lattice.L...)
                # measure the current-current correlation
                correlation_array = selectdim(correlations[i], D+1, l+1)
                current_correlation!(correlation_array, bond_1, bond_0, tdn1′, tdn0′, unit_cell, lattice,
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
                # get the bond definitions
                bond_0 = bonds[bond_id_0]
                bond_1 = bonds[bond_id_1]
                # get the effective hopping amptlitudes for each of the two hopping ID's in question
                tdn0 = @view tup[bond_slices[hopping_id_0], Lτ]
                tdn0′ = reshape(tdn0, lattice.L...)
                tup1 = @view tdn[bond_slices[hopping_id_1], mod1(l,Lτ)]
                tup1′ = reshape(tup1, lattice.L...)
                # measure the current-current correlation
                correlation_array = selectdim(correlations[i], D+1, l+1)
                current_correlation!(correlation_array, bond_1, bond_0, tup1′, tdn0′, unit_cell, lattice,
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
                # get the bond definitions
                bond_0 = bonds[bond_id_0]
                bond_1 = bonds[bond_id_1]
                # get the effective hopping amptlitudes for each of the two hopping ID's in question
                tup0 = @view tdn[bond_slices[hopping_id_0], Lτ]
                tup0′ = reshape(tup0, lattice.L...)
                tdn1 = @view tdn[bond_slices[hopping_id_1], mod1(l,Lτ)]
                tdn1′ = reshape(tdn1, lattice.L...)
                # measure the current-current correlation
                correlation_array = selectdim(correlations[i], D+1, l+1)
                current_correlation!(correlation_array, bond_1, bond_0, tdn1′, tup0′, unit_cell, lattice,
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
                correlation_array = selectdim(correlations[i], D+1, l+1)
                current_correlation!(correlation_array, bond_1, bond_0, tup1′, tup0′, tdn1′, tdn0′, unit_cell, lattice,
                                     Gup_τ0, Gup_0τ, Gup_ττ, Gup, Gdn_τ0, Gdn_0τ, Gdn_ττ, Gdn, sgn)
            end
        end
    end

    return nothing
end


#####################################################################
## MAKE TIME-DISPLACED COMPOSITE CORRELATION FUNCTION MEASUREMENTS ##
#####################################################################

# make purely electronic time-displaced correlation measurements
function make_time_displaced_composite_measurements!(
    time_displaced_composite_correlations::Dict{String, CompositeCorrelationContainer{D,P,E}}, l::Int, sgn::T,
    Gup::AbstractMatrix{T}, Gup_ττ::AbstractMatrix{T}, Gup_τ0::AbstractMatrix{T}, Gup_0τ::AbstractMatrix{T},
    Gdn::AbstractMatrix{T}, Gdn_ττ::AbstractMatrix{T}, Gdn_τ0::AbstractMatrix{T}, Gdn_0τ::AbstractMatrix{T},
    model_geometry::ModelGeometry{D,E,N},
    tight_binding_parameters_up::TightBindingParameters,
    tight_binding_parameters_dn::TightBindingParameters,
    fermion_path_integral_up::FermionPathIntegral{T},
    fermion_path_integral_dn::FermionPathIntegral{T},
    tmp::AbstractArray{Complex{E}, D}
) where {T<:Number, E<:AbstractFloat, P, D, N}

    Lτ = fermion_path_integral_up.Lτ::Int
    unit_cell = model_geometry.unit_cell::UnitCell{D,E,N}
    lattice = model_geometry.lattice::Lattice{D}
    bonds = model_geometry.bonds::Vector{Bond{D}}

    # iterate over time-displaced correlation function getting measured
    for name in keys(time_displaced_composite_correlations)
        
        correlation_container = time_displaced_composite_correlations[name]::CompositeCorrelationContainer{D,P,E}
        correlation = correlation_container.correlation
        id_pairs = correlation_container.id_pairs::Vector{NTuple{2,Int}}
        coefficients = correlation_container.coefficients::Vector{Complex{E}}
        correlations = correlation_container.correlations::Array{Complex{E}, P}
        correlation_array = selectdim(correlations, D+1, l+1)
        structure_factors = correlation_container.structure_factors::Array{Complex{E}, P}
        structure_factor_array = selectdim(structure_factors, D+1, l+1)
        displacement_vecs = correlation_container.displacement_vecs::Vector{SVector{D,E}}

        if correlation == "greens"

            for i in eachindex(id_pairs)
                fill!(tmp, 0)
                coef = coefficients[i]
                id_pair = id_pairs[i]
                greens!(tmp, id_pair[2], id_pair[1], unit_cell, lattice, Gup_τ0, coef*sgn/2)
                greens!(tmp, id_pair[2], id_pair[1], unit_cell, lattice, Gdn_τ0, coef*sgn/2)
                @. correlation_array += tmp
                fourier_transform!(tmp, displacement_vecs[i], unit_cell, lattice)
                @. structure_factor_array += tmp
            end

        elseif correlation == "greens_up"

            for i in eachindex(id_pairs)
                fill!(tmp, 0)
                coef = coefficients[i]
                id_pair = id_pairs[i]
                greens!(tmp, id_pair[2], id_pair[1], unit_cell, lattice, Gup_τ0, coef*sgn)
                @. correlation_array += tmp
                fourier_transform!(tmp, displacement_vecs[i], unit_cell, lattice)
                @. structure_factor_array += tmp
            end

        elseif correlation == "greens_dn"

            for i in eachindex(id_pairs)
                fill!(tmp, 0)
                coef = coefficients[i]
                id_pair = id_pairs[i]
                greens!(tmp, id_pair[2], id_pair[1], unit_cell, lattice, Gdn_τ0, coef*sgn)
                @. correlation_array += tmp
                fourier_transform!(tmp, displacement_vecs[i], unit_cell, lattice)
                @. structure_factor_array += tmp
            end

        elseif correlation == "greens_tautau"

            for i in eachindex(id_pairs)
                fill!(tmp, 0)
                coef = coefficients[i]
                greens!(tmp, id_pair[2], id_pair[1], unit_cell, lattice, Gup_ττ, coef*sgn/2)
                greens!(tmp, id_pair[2], id_pair[1], unit_cell, lattice, Gdn_ττ, coef*sgn/2)
                @. correlation_array += tmp
                fourier_transform!(tmp, displacement_vecs[i], unit_cell, lattice)
                @. structure_factor_array += tmp
            end

        elseif correlation == "greens_tautau_up"

            for i in eachindex(id_pairs)
                fill!(tmp, 0)
                coef = coefficients[i]
                id_pair = id_pairs[i]
                greens!(tmp, id_pair[2], id_pair[1], unit_cell, lattice, Gup_ττ, coef*sgn)
                @. correlation_array += tmp
                fourier_transform!(tmp, displacement_vecs[i], unit_cell, lattice)
                @. structure_factor_array += tmp
            end

        elseif correlation == "greens_tautau_dn"

            for i in eachindex(id_pairs)
                fill!(tmp, 0)
                coef = coefficients[i]
                greens!(tmp, id_pair[2], id_pair[1], unit_cell, lattice, Gdn_ττ, coef*sgn)
                @. correlation_array += tmp
                fourier_transform!(tmp, displacement_vecs[i], unit_cell, lattice)
                @. structure_factor_array += tmp
            end

        elseif correlation == "density_upup"

            for i in eachindex(id_pairs)
                fill!(tmp, 0)
                coef = coefficients[i]
                id_pair = id_pairs[i]
                density_correlation!(
                    tmp, id_pair[2], id_pair[1], unit_cell, lattice,
                    Gup_τ0, Gup_0τ, Gup_ττ, Gup, +1, +1, coef*sgn
                )
                @. correlation_array += tmp
                fourier_transform!(tmp, displacement_vecs[i], unit_cell, lattice)
                @. structure_factor_array += tmp
            end

        elseif correlation == "density_dndn"

            for i in eachindex(ids)
                fill!(tmp, 0)
                coef = coefficients[i]
                id_pair = id_pairs[i]
                density_correlation!(
                    tmp, id_pair[2], id_pair[1], unit_cell, lattice,
                    Gdn_τ0, Gdn_0τ, Gdn_ττ, Gdn, +1, +1, coef*sgn
                )
                @. correlation_array += tmp
                fourier_transform!(tmp, displacement_vecs[i], unit_cell, lattice)
                @. structure_factor_array += tmp
            end

        elseif correlation == "density_updn"

            for i in eachindex(id_pairs)
                fill!(tmp, 0)
                coef = coefficients[i]
                id_pair = id_pairs[i]
                density_correlation!(
                    tmp, id_pair[2], id_pair[1], unit_cell, lattice,
                    Gup_τ0, Gup_0τ, Gup_ττ, Gdn, +1, -1, coef*sgn
                )
                @. correlation_array += tmp
                fourier_transform!(tmp, displacement_vecs[i], unit_cell, lattice)
                @. structure_factor_array += tmp
            end

        elseif correlation == "density_dnup"

            for i in eachindex(id_pairs)
                fill!(tmp, 0)
                coef = coefficients[i]
                id_pair = id_pairs[i]
                density_correlation!(
                    tmp, id_pair[2], id_pair[1], unit_cell, lattice,
                    Gdn_τ0, Gdn_0τ, Gdn_ττ, Gup, -1, +1, coef*sgn
                )
                @. correlation_array += tmp
                fourier_transform!(tmp, displacement_vecs[i], unit_cell, lattice)
                @. structure_factor_array += tmp
            end

        elseif correlation == "density"

            for i in eachindex(id_pairs)
                fill!(tmp, 0)
                coef = coefficients[i]
                id_pair = id_pairs[i]
                density_correlation!(
                    tmp, id_pair[2], id_pair[1], unit_cell, lattice,
                    Gup_τ0, Gup_0τ, Gup_ττ, Gup, Gdn_τ0, Gdn_0τ, Gdn_ττ, Gdn, coef*sgn
                )
                @. correlation_array += tmp
                fourier_transform!(tmp, displacement_vecs[i], unit_cell, lattice)
                @. structure_factor_array += tmp
            end

        elseif correlation == "pair"

            for i in eachindex(id_pairs)
                fill!(tmp, 0)
                coef = coefficients[i]
                id_pair = id_pairs[i]
                pair_correlation!(
                    tmp, bonds[id_pair[2]], bonds[id_pair[1]], unit_cell, lattice,
                    Gup_τ0, Gdn_τ0, coef*sgn
                )
                @. correlation_array += tmp
                fourier_transform!(tmp, displacement_vecs[i], unit_cell, lattice)
                @. structure_factor_array += tmp
            end

        elseif correlation == "spin_x"

            for i in eachindex(id_pairs)
                fill!(tmp, 0)
                coef = coefficients[i]
                id_pair = id_pairs[i]
                spin_x_correlation!(
                    tmp, id_pair[2], id_pair[1], unit_cell, lattice,
                    Gup_τ0, Gup_0τ, Gdn_τ0, Gdn_0τ, coef*sgn
                )
                @. correlation_array += tmp
                fourier_transform!(tmp, displacement_vecs[i], unit_cell, lattice)
                @. structure_factor_array += tmp
            end

        elseif correlation == "spin_z"

            for i in eachindex(id_pairs)
                fill!(tmp, 0)
                coef = coefficients[i]
                id_pair = id_pairs[i]
                spin_z_correlation!(
                    tmp, id_pair[2], id_pair[1], unit_cell, lattice,
                    Gup_τ0, Gup_0τ, Gup_ττ, Gup, Gdn_τ0, Gdn_0τ, Gdn_ττ, Gdn, coef*sgn
                )
                @. correlation_array += tmp
                fourier_transform!(tmp, displacement_vecs[i], unit_cell, lattice)
                @. structure_factor_array += tmp
            end

        elseif correlation == "bond_upup"

            for i in eachindex(id_pairs)
                fill!(tmp, 0)
                coef = coefficients[i]
                id_pair = id_pairs[i]
                bond_correlation!(
                    tmp, bonds[id_pair[2]], bonds[id_pair[1]], unit_cell, lattice,
                    Gup_τ0, Gup_0τ, Gup_ττ, Gup, +1, +1, coef*sgn
                )
                @. correlation_array += tmp
                fourier_transform!(tmp, displacement_vecs[i], unit_cell, lattice)
                @. structure_factor_array += tmp
            end

        elseif correlation == "bond_dndn"

            for i in eachindex(id_pairs)
                fill!(tmp, 0)
                coef = coefficients[i]
                id_pair = id_pairs[i]
                bond_correlation!(
                    tmp, bonds[id_pair[2]], bonds[id_pair[1]], unit_cell, lattice,
                    Gdn_τ0, Gdn_0τ, Gdn_ττ, Gdn, -1, -1, coef*sgn
                )
                @. correlation_array += tmp
                fourier_transform!(tmp, displacement_vecs[i], unit_cell, lattice)
                @. structure_factor_array += tmp
            end

        elseif correlation == "bond_updn"

            for i in eachindex(id_pairs)
                fill!(tmp, 0)
                coef = coefficients[i]
                id_pair = id_pairs[i]
                bond_correlation!(
                    tmp, bonds[id_pair[2]], bonds[id_pair[1]], unit_cell, lattice,
                    Gup_τ0, Gup_0τ, Gup_ττ, Gdn, +1, -1, coef*sgn
                )
                @. correlation_array += tmp
                fourier_transform!(tmp, displacement_vecs[i], unit_cell, lattice)
                @. structure_factor_array += tmp
            end

        elseif correlation == "bond_dnup"

            for i in eachindex(id_pairs)
                fill!(tmp, 0)
                coef = coefficients[i]
                id_pair = id_pairs[i]
                bond_correlation!(
                    tmp, bonds[id_pair[2]], bonds[id_pair[1]], unit_cell, lattice,
                    Gdn_τ0, Gdn_0τ, Gdn_ττ, Gup, -1, +1, coef*sgn
                )
                @. correlation_array += tmp
                fourier_transform!(tmp, displacement_vecs[i], unit_cell, lattice)
                @. structure_factor_array += tmp
            end

        elseif correlation == "bond"

            for i in eachindex(id_pairs)
                fill!(tmp, 0)
                coef = coefficients[i]
                id_pair = id_pairs[i]
                bond_correlation!(
                    tmp, bonds[id_pair[2]], bonds[id_pair[1]], unit_cell, lattice,
                    Gup_τ0, Gup_0τ, Gup_ττ, Gup, Gdn_τ0, Gdn_0τ, Gdn_ττ, Gdn, coef*sgn
                )
                @. correlation_array += tmp
                fourier_transform!(tmp, displacement_vecs[i], unit_cell, lattice)
                @. structure_factor_array += tmp
            end
        
        elseif correlation == "current_upup"

            (; bond_ids, bond_slices) = tight_binding_parameters_up
            tup = fermion_path_integral_up.t

            for i in eachindex(id_pairs)
                fill!(tmp, 0)
                coef = coefficients[i]
                id_pair = id_pairs[i]
                # get the hopping IDs associated with current operators
                hopping_id_0 = id_pair[1]
                hopping_id_1 = id_pair[2]
                # get the bond IDs associated with the hopping IDs
                bond_id_0 = bond_ids[hopping_id_0]
                bond_id_1 = bond_ids[hopping_id_1]
                # get the bond definitions
                bond_0 = bonds[bond_id_0]
                bond_1 = bonds[bond_id_1]
                # get the effective hopping amptlitudes for each of the two hopping ID's in question
                tup0 = @view tup[bond_slices[hopping_id_0], Lτ]
                tup0′ = reshape(tup0, lattice.L...)
                tup1 = @view tup[bond_slices[hopping_id_1], mod1(l,Lτ)]
                tup1′ = reshape(tup1, lattice.L...)
                # measure the current-current correlation
                current_correlation!(
                    tmp, bond_1, bond_0, tup1′, tup0′, unit_cell, lattice,
                    Gup_τ0, Gup_0τ, Gup_ττ, Gup, +1, +1, coef*sgn
                )
                @. correlation_array += tmp
                fourier_transform!(tmp, displacement_vecs[i], unit_cell, lattice)
                @. structure_factor_array += tmp
            end

        elseif correlation == "current_dndn"

            (; bond_ids, bond_slices) = tight_binding_parameters_dn
            tdn = fermion_path_integral_dn.t

            for i in eachindex(id_pairs)
                fill!(tmp, 0)
                coef = coefficients[i]
                id_pair = id_pairs[i]
                # get the hopping IDs associated with current operators
                hopping_id_0 = id_pair[1]
                hopping_id_1 = id_pair[2]
                # get the bond IDs associated with the hopping IDs
                bond_id_0 = bond_ids[hopping_id_0]
                bond_id_1 = bond_ids[hopping_id_1]
                # get the bond definitions
                bond_0 = bonds[bond_id_0]
                bond_1 = bonds[bond_id_1]
                # get the effective hopping amptlitudes for each of the two hopping ID's in question
                tdn0 = @view tdn[bond_slices[hopping_id_0], Lτ]
                tdn0′ = reshape(tdn0, lattice.L...)
                tdn1 = @view tdn[bond_slices[hopping_id_1], mod1(l,Lτ)]
                tdn1′ = reshape(tdn1, lattice.L...)
                # measure the current-current correlation
                current_correlation!(
                    tmp, bond_1, bond_0, tdn1′, tdn0′, unit_cell, lattice,
                    Gdn_τ0, Gdn_0τ, Gdn_ττ, Gdn, -1, -1, coef*sgn
                )
                @. correlation_array += tmp
                fourier_transform!(tmp, displacement_vecs[i], unit_cell, lattice)
                @. structure_factor_array += tmp
            end

        elseif correlation == "current_updn"

            (; bond_ids, bond_slices) = tight_binding_parameters_up
            tup = fermion_path_integral_up.t
            tdn = fermion_path_integral_dn.t

            for i in eachindex(id_pairs)
                fill!(tmp, 0)
                coef = coefficients[i]
                id_pair = id_pairs[i]
                # get the hopping IDs associated with current operators
                hopping_id_0 = id_pair[1]
                hopping_id_1 = id_pair[2]
                # get the bond IDs associated with the hopping IDs
                bond_id_0 = bond_ids[hopping_id_0]
                bond_id_1 = bond_ids[hopping_id_1]
                # get the bond definitions
                bond_0 = bonds[bond_id_0]
                bond_1 = bonds[bond_id_1]
                # get the effective hopping amptlitudes for each of the two hopping ID's in question
                tdn0 = @view tdn[bond_slices[hopping_id_0], Lτ]
                tdn0′ = reshape(tdn0, lattice.L...)
                tup1 = @view tup[bond_slices[hopping_id_1], mod1(l,Lτ)]
                tup1′ = reshape(tup1, lattice.L...)
                # measure the current-current correlation
                current_correlation!(
                    tmp, bond_1, bond_0, tup1′, tdn0′, unit_cell, lattice,
                    Gup_τ0, Gup_0τ, Gup_ττ, Gdn, +1, -1, coef*sgn
                )
                @. correlation_array += tmp
                fourier_transform!(tmp, displacement_vecs[i], unit_cell, lattice)
                @. structure_factor_array += tmp
            end

        elseif correlation == "current_dnup"

            (; bond_ids, bond_slices) = tight_binding_parameters_up
            tup = fermion_path_integral_up.t
            tdn = fermion_path_integral_dn.t

            for i in eachindex(id_pairs)
                fill!(tmp, 0)
                coef = coefficients[i]
                id_pair = id_pairs[i]
                # get the hopping IDs associated with current operators
                hopping_id_0 = id_pair[1]
                hopping_id_1 = id_pair[2]
                # get the bond IDs associated with the hopping IDs
                bond_id_0 = bond_ids[hopping_id_0]
                bond_id_1 = bond_ids[hopping_id_1]
                # get the bond definitions
                bond_0 = bonds[bond_id_0]
                bond_1 = bonds[bond_id_1]
                # get the effective hopping amptlitudes for each of the two hopping ID's in question
                tup0 = @view tdn[bond_slices[hopping_id_0], Lτ]
                tup0′ = reshape(tup0, lattice.L...)
                tdn1 = @view tdn[bond_slices[hopping_id_1], mod1(l,Lτ)]
                tdn1′ = reshape(tdn1, lattice.L...)
                # measure the current-current correlation
                current_correlation!(
                    tmp, bond_1, bond_0, tdn1′, tup0′, unit_cell, lattice,
                    Gdn_τ0, Gdn_0τ, Gdn_ττ, Gup, -1, +1, coef*sgn
                )
                @. correlation_array += tmp
                fourier_transform!(tmp, displacement_vecs[i], unit_cell, lattice)
                @. structure_factor_array += tmp
            end

        elseif correlation == "current"

            (; bond_ids, bond_slices) = tight_binding_parameters_up
            tup = fermion_path_integral_up.t
            tdn = fermion_path_integral_dn.t

            for i in eachindex(id_pairs)
                fill!(tmp, 0)
                coef = coefficients[i]
                id_pair = id_pairs[i]
                # get the hopping IDs associated with current operators
                hopping_id_0 = id_pair[1]
                hopping_id_1 = id_pair[2]
                # get the bond IDs associated with the hopping IDs
                bond_id_0 = bond_ids[hopping_id_0]
                bond_id_1 = bond_ids[hopping_id_1]
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
                current_correlation!(
                    tmp, bond_1, bond_0, tup1′, tup0′, tdn1′, tdn0′, unit_cell, lattice,
                    Gup_τ0, Gup_0τ, Gup_ττ, Gup, Gdn_τ0, Gdn_0τ, Gdn_ττ, Gdn, coef*sgn
                )
                @. correlation_array += tmp
                fourier_transform!(tmp, displacement_vecs[i], unit_cell, lattice)
                @. structure_factor_array += tmp
            end
        end
    end

    return nothing
end


####################################
## MEASURE PHONON GREENS FUNCTION ##
####################################

# measure equal-time phonon greens function
function measure_equaltime_phonon_greens!(
    phonon_greens::CorrelationContainer{D,E},
    electron_phonon_parameters::ElectronPhononParameters{T,E},
    model_geometry::ModelGeometry{D,E,N},
    sgn::T,
    XrX0::AbstractArray{Complex{E},P},
    Xr::AbstractArray{Complex{E},P},
    X0::AbstractArray{Complex{E},P}
) where {T<:Number, E<:AbstractFloat, D, P, N}

    id_pairs = phonon_greens.id_pairs::Vector{NTuple{2,Int}}
    correlations = phonon_greens.correlations::Vector{Array{Complex{E}, D}}
    lattice = model_geometry.lattice::Lattice{D}
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
    end

    return nothing
end

# measure equal-time composite phonon greens function
function measure_equaltime_composite_phonon_greens!(
    phonon_greens::CompositeCorrelationContainer{D,D,E},
    electron_phonon_parameters::ElectronPhononParameters{T,E},
    model_geometry::ModelGeometry{D,E,N},
    sgn::T,
    XrX0::AbstractArray{Complex{E},P},
    Xr::AbstractArray{Complex{E},P},
    X0::AbstractArray{Complex{E},P}
) where {T<:Number, E<:AbstractFloat, D, P, N}

    @assert phonon_greens.correlation == "phonon_greens"
    id_pairs = phonon_greens.id_pairs::Vector{NTuple{2,Int}}
    coefficients = phonon_greens.coefficients::Vector{Complex{E}}
    correlations = phonon_greens.correlations::Array{Complex{E}, D}
    structure_factors = phonon_greens.structure_factors::Array{Complex{E}, D}
    lattice = model_geometry.lattice::Lattice{D}
    unit_cell = model_geometry.unit_cell::UnitCell{D,E}
    phonon_parameters = electron_phonon_parameters.phonon_parameters::PhononParameters{E}
    phonon_basis_vecs = phonon_parameters.basis_vecs::Vector{Int}

    # get phonon field
    x = electron_phonon_parameters.x::Matrix{E}

    # length of imaginary time axis
    Lτ = size(x,2)

    # size of system in unit cells
    L = lattice.L

    # number of unit cells
    N_unitcells = prod(L)

    # number of phonons per unit cell
    phonon_parameters = electron_phonon_parameters.phonon_parameters::PhononParameters{E}
    nphonon = phonon_parameters.nphonon::Int

    # reshape phonon field matrix into multi-dimensional array
    x′ = reshape(x, (L..., nphonon, Lτ))

    # initialize vector to represent difference between phonon basis vectors
    r = zeros(T, D)

    # iterate over all pairs of phonon modes
    for i in eachindex(id_pairs)
        # get phonon ids
        phonon_1_id = id_pairs[i][1]
        phonon_2_id = id_pairs[i][2]
        # get the phonon fields associated with the appropriate pair of phonon modes in the unit cell
        x0 = selectdim(x′, D+1, phonon_1_id)
        xr = selectdim(x′, D+1, phonon_2_id)
        copyto!(X0, x0)
        copyto!(Xr, xr)
        # calculate phonon greens function
        translational_avg!(XrX0, Xr, X0, restore = false)
        # record the equal-time phonon green's function in position space
        XrX0_0 = selectdim(XrX0, D+1, 1)
        coef = coefficients[i]
        @. correlations += sgn * coef * XrX0_0
        # record the equal-time phonon green's function in momentum space
        fourier_transform!(XrX0_0, displacement_vecs[i], unit_cell, lattice)
        @. structure_factors += sgn * coef * XrX0_0
    end

    return nothing
end

# measure time-displaced phonon greens function
function measure_time_displaced_phonon_greens!(
    phonon_greens::CorrelationContainer{P,E}, # time-displaced because P != D
    electron_phonon_parameters::ElectronPhononParameters{T,E},
    model_geometry::ModelGeometry{D,E,N},
    sgn::T,
    XrX0::AbstractArray{Complex{E},P},
    Xr::AbstractArray{Complex{E},P},
    X0::AbstractArray{Complex{E},P}
) where {T<:Number, E<:AbstractFloat, D, P, N}

    id_pairs = phonon_greens.id_pairs::Vector{NTuple{2,Int}}
    correlations = phonon_greens.correlations::Vector{Array{Complex{E}, P}}
    lattice = model_geometry.lattice::Lattice{D}
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
    end

    return nothing
end

# measure time-displaced composite phonon greens function
function measure_time_displaced_composite_phonon_greens!(
    phonon_greens::CompositeCorrelationContainer{D,P,E}, # time-displaced because P != D
    electron_phonon_parameters::ElectronPhononParameters{T,E},
    model_geometry::ModelGeometry{D,E,N},
    sgn::T,
    XrX0::AbstractArray{Complex{E},P},
    Xr::AbstractArray{Complex{E},P},
    X0::AbstractArray{Complex{E},P}
) where {T<:Number, E<:AbstractFloat, D, P, N}

    @assert phonon_greens.correlation == "phonon_greens"
    id_pairs = phonon_greens.id_pairs::Vector{NTuple{2,Int}}
    coefficients = phonon_greens.coefficients::Vector{Complex{E}}
    correlations = phonon_greens.correlations::Array{Complex{E}, D}
    structure_factors = phonon_greens.structure_factors::Array{Complex{E}, D}
    lattice = model_geometry.lattice::Lattice{D}
    unit_cell = model_geometry.unit_cell::UnitCell{D,E,N}
    phonon_parameters = electron_phonon_parameters.phonon_parameters::PhononParameters{E}
    phonon_basis_vecs = phonon_parameters.basis_vecs::Vector{Int}

    # get phonon field
    x = electron_phonon_parameters.x::Matrix{E}

    # length of imaginary time axis
    Lτ = size(x,2)

    # size of system in unit cells
    L = lattice.L

    # number of unit cells
    N_unitcells = prod(L)

    # number of phonons per unit cell
    phonon_parameters = electron_phonon_parameters.phonon_parameters::PhononParameters{E}
    nphonon = phonon_parameters.nphonon::Int

    # reshape phonon field matrix into multi-dimensional array
    x′ = reshape(x, (L..., nphonon, Lτ))

    # initialize vector to represent difference between phonon basis vectors
    r = zeros(T, D)

    # iterate over all pairs of phonon modes
    for i in eachindex(id_pairs)
        # get phonon ids
        phonon_1_id = id_pairs[i][1]
        phonon_2_id = id_pairs[i][2]
        # get the phonon fields associated with the appropriate pair of phonon modes in the unit cell
        x0 = selectdim(x′, D+1, phonon_id_1)
        xr = selectdim(x′, D+1, phonon_id_2)
        copyto!(X0, x0)
        copyto!(Xr, xr)
        # calculate phonon greens function in position space
        translational_avg!(XrX0, Xr, X0, restore = false)
        correlation′ = selectdim(correlations, D+1, 1:Lτ)
        coef = coefficients[i]
        @. correlation′ += coef * sgn * XrX0
        correlation_0  = selectdim(correlations, D+1, 1)
        correlation_Lτ = selectdim(correlations, D+1, Lτ+1)
        copyto!(correlation_Lτ, correlation_0)
        # calculate phonon greens function in position space in momentum space
        fourier_transform!(XrX0, displacement_vecs[i], D+1, unit_cell, lattice)
        structure_factors′ = selectdim(structure_factors, D+1, 1:Lτ)
        @. structure_factors′ += coef * sgn * XrX0
        structure_factors_0  = selectdim(structure_factors, D+1, 1)
        structure_factors_Lτ = selectdim(structure_factors, D+1, Lτ+1)
        copyto!(structure_factors_Lτ, structure_factors_0)
    end

    return nothing
end