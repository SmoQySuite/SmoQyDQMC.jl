#######################################################
## HIGHEST LEVEL/EXPORTED MAKE MEASUREMENTS FUNCTION ##
#######################################################

@doc raw"""
    make_measurements!(measurement_container::NamedTuple,
                       logdetGup::E, sgndetGup::T, Gup::AbstractMatrix{T}, Gup_ττ::AbstractMatrix{T}, Gup_τ0::AbstractMatrix{T}, Gup_0τ::AbstractMatrix{T},
                       logdetGdn::E, sgndetGdn::T, Gdn::AbstractMatrix{T}, Gdn_ττ::AbstractMatrix{T}, Gdn_τ0::AbstractMatrix{T}, Gdn_0τ::AbstractMatrix{T};
                       fermion_path_integral_up::FermionPathIntegral{T,E},
                       fermion_path_integral_dn::FermionPathIntegral{T,E},
                       fermion_greens_calculator_up::FermionGreensCalculator{T,E},
                       fermion_greens_calculator_dn::FermionGreensCalculator{T,E},
                       Bup::Vector{P}, Bdn::Vector{P}, δG_max::E, δG::E, δθ::E,
                       model_geometry::ModelGeometry{D,E,N},
                       tight_binding_parameters::TightBindingParameters{T,E},
                       coupling_parameters::Tuple) where {T<:Number, E<:AbstractFloat, D, N, P<:AbstractPropagator{T,E}}

Make measurements, including time-displaced correlation and zero Matsubara frequency measurements.
This method also returns `(logdetGup, sgndetGup, logdetGdn, sgndetGdn, δG, δθ)`.
"""
function make_measurements!(measurement_container::NamedTuple,
                            logdetGup::E, sgndetGup::T, Gup::AbstractMatrix{T}, Gup_ττ::AbstractMatrix{T}, Gup_τ0::AbstractMatrix{T}, Gup_0τ::AbstractMatrix{T},
                            logdetGdn::E, sgndetGdn::T, Gdn::AbstractMatrix{T}, Gdn_ττ::AbstractMatrix{T}, Gdn_τ0::AbstractMatrix{T}, Gdn_0τ::AbstractMatrix{T};
                            fermion_path_integral_up::FermionPathIntegral{T,E},
                            fermion_path_integral_dn::FermionPathIntegral{T,E},
                            fermion_greens_calculator_up::FermionGreensCalculator{T,E},
                            fermion_greens_calculator_dn::FermionGreensCalculator{T,E},
                            Bup::Vector{P}, Bdn::Vector{P}, δG_max::E, δG::E, δθ::E,
                            model_geometry::ModelGeometry{D,E,N},
                            tight_binding_parameters::TightBindingParameters{T,E},
                            coupling_parameters::Tuple) where {T<:Number, E<:AbstractFloat, D, N, P<:AbstractPropagator{T,E}}

    # extract temporary storage vectors
    (; time_displaced_correlations, equaltime_correlations, a, a′, a″) = measurement_container

    # calculate sign
    sgn = sgndetGup * sgndetGdn
    sgn /= abs(sgn) # normalize just to be cautious

    # make global measurements
    global_measurements = measurement_container.global_measurements
    make_global_measurements!(global_measurements, tight_binding_parameters, sgndetGup, sgndetGdn, Gup, Gdn)

    # make local measurements
    local_measurements = measurement_container.local_measurements
    make_local_measurements!(local_measurements, Gup, Gdn, sgn, model_geometry, tight_binding_parameters, coupling_parameters)

    # initialize green's function matrices G(τ,0), G(0,τ) and G(τ,τ) based on G(0,0)
    initialize_unequaltime_greens!(Gup_τ0, Gup_0τ, Gup_ττ, Gup)
    initialize_unequaltime_greens!(Gdn_τ0, Gdn_0τ, Gdn_ττ, Gdn)

    # make equal-time correlation measurements
    make_equaltime_measurements!(equaltime_correlations, sgn,
                                 Gup, Gup_ττ, Gup_τ0, Gup_0τ,
                                 Gdn, Gdn_ττ, Gdn_τ0, Gdn_0τ,
                                 model_geometry, tight_binding_parameters,
                                 fermion_path_integral_up, fermion_path_integral_dn)

    # if there are time-displaced measurements to make
    if length(time_displaced_correlations) > 0

        # make time-displaced measuresurements for τ = l⋅Δτ = 0
        make_time_displaced_measurements!(time_displaced_correlations, 0, sgn,
                                          Gup, Gup_ττ, Gup_τ0, Gup_0τ, Gdn, Gdn_ττ, Gdn_τ0, Gdn_0τ,
                                          model_geometry, tight_binding_parameters,
                                          fermion_path_integral_up, fermion_path_integral_dn)

        # iterate over imaginary time slice
        for l in fermion_greens_calculator_up

            # Propagate Green's function matrices to current imaginary time slice
            propagate_unequaltime_greens!(Gup_τ0, Gup_0τ, Gup_ττ, fermion_greens_calculator_up, Bup)
            propagate_unequaltime_greens!(Gdn_τ0, Gdn_0τ, Gdn_ττ, fermion_greens_calculator_dn, Bdn)

            # make time-displaced measuresurements for τ = l⋅Δτ
            make_time_displaced_measurements!(time_displaced_correlations, l, sgn,
                                              Gup, Gup_ττ, Gup_τ0, Gup_0τ, Gdn, Gdn_ττ, Gdn_τ0, Gdn_0τ,
                                              model_geometry, tight_binding_parameters,
                                              fermion_path_integral_up, fermion_path_integral_dn)

            # Periodically re-calculate the Green's function matrix for numerical stability.
            logdetGup, sgndetGup, δGup, δθup = stabilize_unequaltime_greens!(Gup_τ0, Gup_0τ, Gup_ττ, logdetGup, sgndetGup, fermion_greens_calculator_up, Bup, update_B̄=false)
            logdetGdn, sgndetGdn, δGdn, δθdn = stabilize_unequaltime_greens!(Gdn_τ0, Gdn_0τ, Gdn_ττ, logdetGdn, sgndetGdn, fermion_greens_calculator_dn, Bdn, update_B̄=false)

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
    make_measurements!(measurement_container::NamedTuple,
                       logdetG::E, sgndetG::T, G::AbstractMatrix{T},
                       G_ττ::AbstractMatrix{T}, G_τ0::AbstractMatrix{T}, G_0τ::AbstractMatrix{T};
                       fermion_path_integral::FermionPathIntegral{T,E},
                       fermion_greens_calculator::FermionGreensCalculator{T,E},
                       B::Vector{P}, δG_max::E, δG::E, δθ::E,
                       model_geometry::ModelGeometry{D,E,N},
                       tight_binding_parameters::TightBindingParameters{T,E},
                       coupling_parameters::Tuple) where {T<:Number, E<:AbstractFloat, D, N, P<:AbstractPropagator{T,E}}

Make measurements, including time-displaced correlation and zero Matsubara frequency measurements.
This method also returns `(logdetG, sgndetG, δG, δθ)`.
"""
function make_measurements!(measurement_container::NamedTuple,
                            logdetG::E, sgndetG::T, G::AbstractMatrix{T},
                            G_ττ::AbstractMatrix{T}, G_τ0::AbstractMatrix{T}, G_0τ::AbstractMatrix{T};
                            fermion_path_integral::FermionPathIntegral{T,E},
                            fermion_greens_calculator::FermionGreensCalculator{T,E},
                            B::Vector{P}, δG_max::E, δG::E, δθ::E,
                            model_geometry::ModelGeometry{D,E,N},
                            tight_binding_parameters::TightBindingParameters{T,E},
                            coupling_parameters::Tuple) where {T<:Number, E<:AbstractFloat, D, N, P<:AbstractPropagator{T,E}}

    # extract temporary storage vectors
    (; time_displaced_correlations, equaltime_correlations, a, a′, a″) = measurement_container

    # calculate sign
    sgn = sgndetG^2
    sgn /= abs(sgn) # normalize just to be cautious

    # make global measurements
    global_measurements = measurement_container.global_measurements
    make_global_measurements!(global_measurements, tight_binding_parameters, sgndetG, sgndetG, G, G)

    # make local measurements
    local_measurements = measurement_container.local_measurements
    make_local_measurements!(local_measurements, G, G, sgn, model_geometry, tight_binding_parameters, coupling_parameters)

    # initialize green's function matrices G(τ,0), G(0,τ) and G(τ,τ) based on G(0,0)
    initialize_unequaltime_greens!(G_τ0, G_0τ, G_ττ, G)

    # make equal-time correlation measurements
    make_equaltime_measurements!(equaltime_correlations, sgn,
                                 G, G_ττ, G_τ0, G_0τ, G, G_ττ, G_τ0, G_0τ,
                                 model_geometry, tight_binding_parameters,
                                 fermion_path_integral, fermion_path_integral)

    # if there are time-displaced measurements to make
    if length(time_displaced_correlations) > 0

        # make time-displaced measuresurements of τ = 0
        make_time_displaced_measurements!(time_displaced_correlations, 0, sgn,
                                          G, G_ττ, G_τ0, G_0τ, G, G_ττ, G_τ0, G_0τ,
                                          model_geometry, tight_binding_parameters,
                                          fermion_path_integral, fermion_path_integral)

        # iterate over imaginary time slice
        for l in fermion_greens_calculator

            # Propagate Green's function matrices to current imaginary time slice
            propagate_unequaltime_greens!(G_τ0, G_0τ, G_ττ, fermion_greens_calculator, B)

            # make time-displaced measuresurements of τ = l⋅Δτ
            make_time_displaced_measurements!(time_displaced_correlations, l, sgn,
                                              G, G_ττ, G_τ0, G_0τ, G, G_ττ, G_τ0, G_0τ,
                                              model_geometry, tight_binding_parameters,
                                              fermion_path_integral, fermion_path_integral)

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
        measure_eqaultime_phonon_greens!(equaltime_correlations["phonon_greens"], coupling_parameters[indx], model_geometry, sgn, a, a′, a″)
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
function make_global_measurements!(global_measurements::Dict{String, Complex{E}},
                                   tight_binding_parameters::TightBindingParameters{T,E},
                                   sgndetGup::T, sgndetGdn::T,
                                   Gup::AbstractMatrix{T}, Gdn::AbstractMatrix{T}) where {T<:Number, E<:AbstractFloat}

    # number of orbitals in lattice
    N = size(Gup, 1)

    # measure the sign
    sgn = sgndetGup * sgndetGdn
    sgn /= abs(sgn) # normalize just to be cautious
    global_measurements["sgn"] += sgn

    # record the spin resolved sign
    global_measurements["sgndetGup"] += sgndetGup
    global_measurements["sgndetGdn"] += sgndetGdn

    # measure average density
    global_measurements["density"] += sgn * (measure_n(Gup) + measure_n(Gdn))

    # measure double occupancy
    global_measurements["double_occ"] += sgn * measure_double_occ(Gup, Gdn)

    # measure ⟨N²⟩
    global_measurements["Nsqrd"] += sgn * measure_Nsqrd(Gup, Gdn)

    # measure chemical potential
    global_measurements["chemical_potential"] += tight_binding_parameters.μ

    return nothing
end


#############################
## MAKE LOCAL MEASUREMENTS ##
#############################

# make local measurements
function make_local_measurements!(local_measurements::Dict{String, Vector{Complex{E}}},
                                  Gup::AbstractMatrix{T}, Gdn::AbstractMatrix{T}, sgn::T,
                                  model_geometry::ModelGeometry{D,E,N},
                                  tight_binding_parameters::TightBindingParameters{T,E},
                                  coupling_parameters::Tuple) where {T<:Number, E<:AbstractFloat, D, N}

    # number of orbitals per unit cell
    unit_cell = model_geometry.unit_cell::UnitCell{D,E,N}
    norbital = unit_cell.n

    # iterate over orbital species
    for n in 1:norbital
        # measure density
        local_measurements["density"][n] += sgn * (measure_n(Gup, n, unit_cell) + measure_n(Gdn, n, unit_cell))
        # measure double occupancy
        local_measurements["double_occ"][n] += sgn * measure_double_occ(Gup, Gdn, n, unit_cell)
    end

    # make tight-binding measurements
    make_local_measurements!(local_measurements, Gup, Gdn, sgn, model_geometry, tight_binding_parameters)

    # make local measurements associated with couplings
    for coupling_parameter in coupling_parameters
        make_local_measurements!(local_measurements, Gup, Gdn, sgn, model_geometry, coupling_parameter, tight_binding_parameters)
    end
    
    return nothing
end

# make local measurements associated with tight-binding model
function make_local_measurements!(local_measurements::Dict{String, Vector{Complex{E}}},
                                  Gup::AbstractMatrix{T}, Gdn::AbstractMatrix{T}, sgn::T,
                                  model_geometry::ModelGeometry{D,E,N},
                                  tight_binding_parameters::TightBindingParameters{T,E}) where {T<:Number, E<:AbstractFloat, D, N}

    # number of orbitals per unit cell
    norbital = tight_binding_parameters.norbital

    # number of types of hopping
    bond_ids = tight_binding_parameters.bond_ids
    nhopping = length(tight_binding_parameters.bond_ids)

    # measure on-site energy
    for n in 1:norbital
        local_measurements["onsite_energy"][n] += sgn * measure_onsite_energy(tight_binding_parameters, Gup, Gdn, n)
    end

    # measure hopping energy
    if nhopping > 0
        for h in 1:nhopping
            local_measurements["hopping_energy"][h] += sgn * measure_hopping_energy(tight_binding_parameters, Gup, Gdn, bond_ids[h])
        end
    end

    return nothing
end

# make local measurements associated with hubbard model
function make_local_measurements!(local_measurements::Dict{String, Vector{Complex{E}}},
                                  Gup::AbstractMatrix{T}, Gdn::AbstractMatrix{T}, sgn::T,
                                  model_geometry::ModelGeometry{D,E,N},
                                  hubbard_parameters::HubbardParameters{T},
                                  tight_binding_parameters::TightBindingParameters{T,E}) where {T<:Number, E<:AbstractFloat, D, N}

    # measure hubbard energy for each orbital in unit cell
    hubbard_energies = local_measurements["hubbard_energy"]
    for orbital in eachindex(hubbard_energies)
        hubbard_energies[orbital] += sgn * measure_hubbard_energy(hubbard_parameters, Gup, Gdn, orbital)
    end

    return nothing
end

# make local measurements associated with electron-phonon model
function make_local_measurements!(local_measurements::Dict{String, Vector{Complex{E}}},
                                  Gup::AbstractMatrix{T}, Gdn::AbstractMatrix{T}, sgn::T,
                                  model_geometry::ModelGeometry{D,E,N},
                                  electron_phonon_parameters::ElectronPhononParameters{T,E},
                                  tight_binding_parameters::TightBindingParameters{T,E}) where {T<:Number, E<:AbstractFloat, D, N}

    nphonon = electron_phonon_parameters.phonon_parameters.nphonon::Int # number of phonon modes per unit cell
    nholstein = electron_phonon_parameters.holstein_parameters.nholstein::Int # number of types of holstein couplings
    nssh = electron_phonon_parameters.ssh_parameters.nssh::Int # number of types of ssh coupling
    ndispersion = electron_phonon_parameters.dispersion_parameters.ndispersion::Int # number of types of dispersive phonon couplings

    # make phonon mode related measurements
    for n in 1:nphonon
        local_measurements["phonon_kinetic_energy"][n]   += sgn * measure_phonon_kinetic_energy(electron_phonon_parameters, n)
        local_measurements["phonon_potential_energy"][n] += sgn * measure_phonon_potential_energy(electron_phonon_parameters, n)
        local_measurements["X"][n]  += sgn * measure_phonon_position_moment(electron_phonon_parameters, n, 1)
        local_measurements["X2"][n] += sgn * measure_phonon_position_moment(electron_phonon_parameters, n, 2)
        local_measurements["X3"][n] += sgn * measure_phonon_position_moment(electron_phonon_parameters, n, 3)
        local_measurements["X4"][n] += sgn * measure_phonon_position_moment(electron_phonon_parameters, n, 4)
    end

    # check if finite number of holstein couplings
    if nholstein > 0
        # make holstein coupling related measurements
        for n in 1:nholstein
            local_measurements["holstein_energy"][n] += sgn * measure_holstein_energy(electron_phonon_parameters, Gup, Gdn, n)
        end
    end

    # check if finite number of ssh couplings
    if nssh > 0
        # make ssh coupling related measurements
        for n in 1:nssh
            local_measurements["ssh_energy"][n] += sgn * measure_ssh_energy(electron_phonon_parameters, Gup, Gdn, n)
            local_measurements["ssh_sgn_switch"][n] += sgn * measure_ssh_sgn_switch(electron_phonon_parameters, tight_binding_parameters, n)
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


############################################
## MAKE CORRELATION FUNCTION MEASUREMENTS ##
############################################

# make purely electronic equal-time correlation measurements
function make_equaltime_measurements!(equaltime_correlations::Dict{String, CorrelationContainer{D,E}}, sgn::T,
                                      Gup::AbstractMatrix{T}, Gup_ττ::AbstractMatrix{T}, Gup_τ0::AbstractMatrix{T}, Gup_0τ::AbstractMatrix{T},
                                      Gdn::AbstractMatrix{T}, Gdn_ττ::AbstractMatrix{T}, Gdn_τ0::AbstractMatrix{T}, Gdn_0τ::AbstractMatrix{T},
                                      model_geometry::ModelGeometry{D,E,N},
                                      tight_binding_parameters::TightBindingParameters{T,E},
                                      fermion_path_integral_up::FermionPathIntegral{T,E},
                                      fermion_path_integral_dn::FermionPathIntegral{T,E},) where {T<:Number, E<:AbstractFloat, D, N}

    unit_cell = model_geometry.unit_cell::UnitCell{D,E,N}
    lattice = model_geometry.lattice::Lattice{D}
    bonds = model_geometry.bonds::Vector{Bond{D}}

    # iterate over equal-time correlation function getting measured
    for correlation in keys(equaltime_correlations)
        
        correlation_container = equaltime_correlations[correlation]::CorrelationContainer{D,E}
        pairs = correlation_container.pairs::Vector{NTuple{2,Int}}
        correlations = correlation_container.correlations::Vector{Array{Complex{T}, D}}

        if correlation == "greens"

            for i in eachindex(pairs)
                pair = pairs[i]
                correlation = correlations[i]
                greens!(correlation, pair[2], pair[1], unit_cell, lattice, Gup_τ0, sgn/2)
                greens!(correlation, pair[2], pair[1], unit_cell, lattice, Gdn_τ0, sgn/2)
            end

        elseif correlation == "greens_up"

            for i in eachindex(pairs)
                pair = pairs[i]
                correlation = correlations[i]
                greens!(correlation, pair[2], pair[1], unit_cell, lattice, Gup_τ0, sgn)
            end

        elseif correlation == "greens_dn"

            for i in eachindex(pairs)
                pair = pairs[i]
                correlation = correlations[i]
                greens!(correlation, pair[2], pair[1], unit_cell, lattice, Gdn_τ0, sgn)
            end

        elseif correlation == "density"

            for i in eachindex(pairs)
                pair = pairs[i]
                correlation = correlations[i]
                density_correlation!(correlation, pair[2], pair[1], unit_cell, lattice,
                                     Gup_τ0, Gup_0τ, Gup_ττ, Gup, Gdn_τ0, Gdn_0τ, Gdn_ττ, Gdn, sgn)
            end

        elseif correlation == "pair"

            for i in eachindex(pairs)
                pair = pairs[i]
                correlation = correlations[i]
                pair_correlation!(correlation, bonds[pair[2]], bonds[pair[1]], unit_cell, lattice,
                                  Gup_τ0, Gdn_τ0, sgn)
            end

        elseif correlation == "spin_x"

            for i in eachindex(pairs)
                pair = pairs[i]
                correlation = correlations[i]
                spin_x_correlation!(correlation, pair[2], pair[1], unit_cell, lattice,
                                    Gup_τ0, Gup_0τ, Gdn_τ0, Gdn_0τ, sgn)
            end

        elseif correlation == "spin_z"

            for i in eachindex(pairs)
                pair = pairs[i]
                correlation = correlations[i]
                spin_z_correlation!(correlation, pair[2], pair[1], unit_cell, lattice,
                                    Gup_τ0, Gup_0τ, Gup_ττ, Gup, Gdn_τ0, Gdn_0τ, Gdn_ττ, Gdn, sgn)
            end

        elseif correlation == "bond"

            for i in eachindex(pairs)
                pair = pairs[i]
                correlation = correlations[i]
                bond_correlation!(correlation, bonds[pair[2]], bonds[pair[1]], unit_cell, lattice,
                                  Gup_τ0, Gup_0τ, Gup_ττ, Gup, Gdn_τ0, Gdn_0τ, Gdn_ττ, Gdn, sgn)
            end

        elseif correlation == "current"

            (; bond_ids, bond_slices) = tight_binding_parameters
            (; t, Lτ) = fermion_path_integral_up

            for i in eachindex(pairs)
                # get the hopping IDs associated with current operators
                pair = pairs[1]
                hopping_id_0 = pair[1]
                hopping_id_1 = pair[2]
                # get the bond IDs associated with the hopping IDs
                bond_id_0 = bond_ids[hopping_id_0]
                bond_id_1 = bond_ids[hopping_id_1]
                # get the bond definitions
                bond_0 = bonds[bond_id_0]
                bond_1 = bonds[bond_id_1]
                # get the effective hopping amptlitudes for each of the two hopping ID's in question
                t0 = @view t[bond_slices[hopping_id_0], Lτ]
                t0′ = reshape(t0, lattice.L...)
                t1 = @view t[bond_slices[hopping_id_1], Lτ]
                t1′ = reshape(t1, lattice.L...)
                # measure the current-current correlation
                correlation = correlations[i]
                current_correlation!(correlation, bond_1, bond_0, t1′, t0′, unit_cell, lattice,
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
function make_time_displaced_measurements!(time_displaced_correlations::Dict{String, CorrelationContainer{P,T}}, l::Int, sgn::T,
                                           Gup::AbstractMatrix{T}, Gup_ττ::AbstractMatrix{T}, Gup_τ0::AbstractMatrix{T}, Gup_0τ::AbstractMatrix{T},
                                           Gdn::AbstractMatrix{T}, Gdn_ττ::AbstractMatrix{T}, Gdn_τ0::AbstractMatrix{T}, Gdn_0τ::AbstractMatrix{T},
                                           model_geometry::ModelGeometry{D,E,N},
                                           tight_binding_parameters::TightBindingParameters{T,E},
                                           fermion_path_integral_up::FermionPathIntegral{T,E},
                                           fermion_path_integral_dn::FermionPathIntegral{T,E},) where {T<:Number, E<:AbstractFloat, P, D, N}

    unit_cell = model_geometry.unit_cell::UnitCell{D,E,N}
    lattice = model_geometry.lattice::Lattice{D}
    bonds = model_geometry.bonds::Vector{Bond{D}}

    # iterate over time-displaced correlation function getting measured
    for correlation in keys(time_displaced_correlations)
        
        correlation_container = time_displaced_correlations[correlation]::CorrelationContainer{P,E}
        pairs = correlation_container.pairs::Vector{NTuple{2,Int}}
        correlations = correlation_container.correlations::Vector{Array{Complex{T}, P}}

        if correlation == "greens"

            for i in eachindex(pairs)
                pair = pairs[i]
                correlation = selectdim(correlations[i], D+1, l+1)
                greens!(correlation, pair[2], pair[1], unit_cell, lattice, Gup_τ0, sgn/2)
                greens!(correlation, pair[2], pair[1], unit_cell, lattice, Gdn_τ0, sgn/2)
            end

        elseif correlation == "greens_up"

            for i in eachindex(pairs)
                pair = pairs[i]
                correlation = selectdim(correlations[i], D+1, l+1)
                greens!(correlation, pair[2], pair[1], unit_cell, lattice, Gup_τ0, sgn)
            end

        elseif correlation == "greens_dn"

            for i in eachindex(pairs)
                pair = pairs[i]
                correlation = selectdim(correlations[i], D+1, l+1)
                greens!(correlation, pair[2], pair[1], unit_cell, lattice, Gdn_τ0, sgn)
            end

        elseif correlation == "density"

            for i in eachindex(pairs)
                pair = pairs[i]
                correlation = selectdim(correlations[i], D+1, l+1)
                density_correlation!(correlation, pair[2], pair[1], unit_cell, lattice,
                                     Gup_τ0, Gup_0τ, Gup_ττ, Gup, Gdn_τ0, Gdn_0τ, Gdn_ττ, Gdn, sgn)
            end

        elseif correlation == "pair"

            for i in eachindex(pairs)
                pair = pairs[i]
                correlation = selectdim(correlations[i], D+1, l+1)
                pair_correlation!(correlation, bonds[pair[2]], bonds[pair[1]], unit_cell, lattice,
                                  Gup_τ0, Gdn_τ0, sgn)
            end

        elseif correlation == "spin_x"

            for i in eachindex(pairs)
                pair = pairs[i]
                correlation = selectdim(correlations[i], D+1, l+1)
                spin_x_correlation!(correlation, pair[2], pair[1], unit_cell, lattice,
                                    Gup_τ0, Gup_0τ, Gdn_τ0, Gdn_0τ, sgn)
            end

        elseif correlation == "spin_z"

            for i in eachindex(pairs)
                pair = pairs[i]
                correlation = selectdim(correlations[i], D+1, l+1)
                spin_z_correlation!(correlation, pair[2], pair[1], unit_cell, lattice,
                                    Gup_τ0, Gup_0τ, Gup_ττ, Gup, Gdn_τ0, Gdn_0τ, Gdn_ττ, Gdn, sgn)
            end

        elseif correlation == "bond"

            for i in eachindex(pairs)
                pair = pairs[i]
                correlation = selectdim(correlations[i], D+1, l+1)
                bond_correlation!(correlation, bonds[pair[2]], bonds[pair[1]], unit_cell, lattice,
                                  Gup_τ0, Gup_0τ, Gup_ττ, Gup, Gdn_τ0, Gdn_0τ, Gdn_ττ, Gdn, sgn)
            end
        
        elseif correlation == "current"

            (; bond_ids, bond_slices) = tight_binding_parameters
            (; t, Lτ) = fermion_path_integral_up

            for i in eachindex(pairs)
                # get the hopping IDs associated with current operators
                pair = pairs[i]
                hopping_id_0 = pair[1]
                hopping_id_1 = pair[2]
                # get the bond IDs associated with the hopping IDs
                bond_id_0 = bond_ids[hopping_id_0]
                bond_id_1 = bond_ids[hopping_id_1]
                # get the bond definitions
                bond_0 = bonds[bond_id_0]
                bond_1 = bonds[bond_id_1]
                # get the effective hopping amptlitudes for each of the two hopping ID's in question
                t0 = @view t[bond_slices[hopping_id_0], Lτ]
                t0′ = reshape(t0, lattice.L...)
                t1 = @view t[bond_slices[hopping_id_1], mod1(l,Lτ)]
                t1′ = reshape(t1, lattice.L...)
                # measure the current-current correlation
                correlation = selectdim(correlations[i], D+1, l+1)
                current_correlation!(correlation, bond_1, bond_0, t1′, t0′, unit_cell, lattice,
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
function measure_eqaultime_phonon_greens!(phonon_greens::CorrelationContainer{D,E},
                                          electron_phonon_parameters::ElectronPhononParameters{T,E},
                                          model_geometry::ModelGeometry{D,E,N},
                                          sgn::T,
                                          XrX0::AbstractArray{Complex{E},P},
                                          Xr::AbstractArray{Complex{E},P},
                                          X0::AbstractArray{Complex{E},P}) where {T<:Number, E<:AbstractFloat, D, P, N}

    pairs = phonon_greens.pairs::Vector{NTuple{2,Int}}
    correlations = phonon_greens.correlations::Vector{Array{Complex{E}, D}}
    unit_cell = model_geometry.unit_cell::UnitCell{D,E,N}
    lattice = model_geometry.lattice::Lattice{D}
    bonds = model_geometry.bonds::Vector{Bond{D}}

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
    for i in eachindex(pairs)
        # get the phonon fields associated with the appropriate pair of phonon modes in the unit cell
        pair = pairs[i]
        correlation = correlations[i]
        x0 = selectdim(x′, D+1, pair[1])
        xr = selectdim(x′, D+1, pair[2])
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

# measure time-displaced phonon greens function
function measure_time_displaced_phonon_greens!(phonon_greens::CorrelationContainer{P,E}, # time-displaced because P != D
                                               electron_phonon_parameters::ElectronPhononParameters{T,E},
                                               model_geometry::ModelGeometry{D,E,N},
                                               sgn::T,
                                               XrX0::AbstractArray{Complex{E},P},
                                               Xr::AbstractArray{Complex{E},P},
                                               X0::AbstractArray{Complex{E},P}) where {T<:Number, E<:AbstractFloat, D, P, N}

    pairs = phonon_greens.pairs::Vector{NTuple{2,Int}}
    correlations = phonon_greens.correlations::Vector{Array{Complex{E}, P}}
    unit_cell = model_geometry.unit_cell::UnitCell{D,E,N}
    lattice = model_geometry.lattice::Lattice{D}
    bonds = model_geometry.bonds::Vector{Bond{D}}

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
    for i in eachindex(pairs)
        # get the phonon fields associated with the appropriate pair of phonon modes in the unit cell
        pair = pairs[i]
        correlation = correlations[i]
        x0 = selectdim(x′, D+1, pair[1])
        xr = selectdim(x′, D+1, pair[2])
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