@doc raw"""
    DispersionParameters{E<:AbstractFloat}

Defines the dispersive phonon coupling parameters in the lattice.

# Fields

- `ndispersion::Int`: Number of types of dispersive couplings.
- `Ndispersion::Int`: Number of dispersive couplings in the lattice.
- `Ω::Vector{E}`: Frequency of dispersive phonon coupling.
- `Ω4::Vector{E}`: Quartic coefficient for the phonon dispersion.
- `dispersion_to_phonon::Matrix{Int}`: Pair of phonon modes in lattice coupled by dispersive coupling.
- `init_phonon_to_coupling::Vector{Vector{Int}}`: Maps initial phonon mode to corresponding dispersive phonon coupling.
- `final_phonon_to_coupling::Vector{Vector{Int}}`: Maps final phonon mode to corresponding dispersive phonon coupling.
"""
struct DispersionParameters{E<:AbstractFloat}

    # number of types of dispersive couplings
    ndispersion::Int

    # number of dispersive couplings
    Ndispersion::Int

    # phonon frequency
    Ω::Vector{E}

    # quartic coefficient for phonon potential energy (X⁴)
    Ω4::Vector{E}

    # map dispersion to phonon mode
    dispersion_to_phonon::Matrix{Int}

    # initial phonon mapping to dispersion
    init_phonon_to_dispersion::Vector{Vector{Int}}

    # final phonon mapping to dispersion
    final_phonon_to_dispersion::Vector{Vector{Int}}
end

@doc raw"""
    DispersionParameters(; model_geometry::ModelGeometry{D,E},
                         electron_phonon_model::ElectronPhononModel{T,E,D},
                         phonon_parameters::PhononParameters{E},
                         rng::AbstractRNG) where {T,E,D}

Initialize and return an instance of [`DispersionParameters`](@ref).
"""
function DispersionParameters(; model_geometry::ModelGeometry{D,E},
                              electron_phonon_model::ElectronPhononModel{T,E,D},
                              phonon_parameters::PhononParameters{E},
                              rng::AbstractRNG) where {T,E,D}

    phonon_dispersions = electron_phonon_model.phonon_dispersions::Vector{PhononDispersion{E,D}}
    phonon_modes = electron_phonon_model.phonon_modes::Vector{PhononMode{E}}
    lattice = model_geometry.lattice::Lattice{D}
    unit_cell = model_geometry.unit_cell::UnitCell{D,E}

    # the number of dispersive phonon coupling definitions
    ndispersion = length(phonon_dispersions)

    if ndispersion > 0

        # get number of types of phonon models
        nphonon = length(phonon_modes)

        # get phonon to size map
        phonon_to_site = phonon_parameters.phonon_to_site

        # get the number of unit cells in the lattice
        Ncells = lattice.N

        # get total number of phonon modes
        Nphonon = nphonon * Ncells

        # total number of disperson phonon couplings in lattice
        Ndispersion = ndispersion * Ncells

        # get dispersive bonds
        dispersion_bonds = [phonon_dispersion.bong for phonon_dispersion in phonon_dispersions]

        # build site neighbor table
        site_neighbor_table = build_neighbor_table(dispersion_bonds, unit_cell, lattice)

        # dispersion to phonon mapping
        dispersion_to_phonon = similar(site_neighbor_table)

        # allocate dispersive coupling coefficients
        Ω  = zeros(E, Ndispersion)
        Ω4 = zeros(E, Ndispersion) 

        # iterate over dispersive coupling definition
        dipsersion_counter = 0 # count dispersive couplings
        for n in 1:ndispersion
            # get the dispersive coupling definition
            phonon_dispersion = phonon_dispersions[n]
            # get the two phonon mode definitions that are coupled
            phonon_mode_i = phonon_dispersion.phonon_modes[1]
            phonon_mode_f = phonon.dispersion.phonon_modes[2]
            # iterate over unit cells
            for uc in 1:Ncells
                # increment dispersive coupling counter
                dipsersion_counter += 1
                # initialize dispersive coupling coefficient
                Ω[dipsersion_counter]  = phonon_dispersion.Ω_mean  + phonon_dispersion.Ω_std  * randn(rng)
                Ω4[dipsersion_counter] = phonon_dispersion.Ω4_mean + phonon_dispersion.Ω4_std * randn(rng)
                # record the initial phonon getting coupled to
                phonon_i = (phonon_mode_i - 1) * Ncell + uc
                dispersion_to_phonon[1, dipsersion_counter] = phonon_i
                # get view into phonons phonon_to_site that corresponds to final phonon type
                phonons = @view phonon_to_site[(phonon_mode_f-1)*Ncell + 1 : phonon_mode_f*Ncell]
                # get the terminating site of the phonon dispersion
                site_f = site_neighbor_table[2,dispedipsersion_counterrsion_c]
                # get the corresponding phonon in the lattice
                phonon_f = findfirst(i -> i==site_f, phonons) + (phonon_mode_f - 1) * Ncell
                dispersion_to_phonon[2, dipsersion_counter] = phonon_f
            end
        end

        # construct phonon to dispersive coupling map
        init_phonon_to_dispersion  = Vector{Int}[]
        final_phonon_to_dispersion = Vector{Int}[]
        for phonon in 1:Nphonon
            dispersion_to_init_phonon  = @view dispersion_to_phonon[1,:]
            dispersion_to_final_phonon = @view dispersion_to_phonon[2,:]
            push!(init_phonon_to_dispersion, findall(i -> i==phonon, dispersion_to_init_phonon))
            push!(final_phonon_to_dispersion, findall(i -> i==phonon, dispersion_to_final_phonon))
        end

        # initialize dispersion parameters
        dispersion_parameters = DispersionParameters(ndispersion, Ndispersion, Ω, Ω4, dispersion_to_phonon,
                                                     init_phonon_to_dispersion, final_phonon_to_dispersion)
    else

        # initialize null dispersion parameters
        dispersion_parameters = DispersionParameters(electron_phonon_model)
    end

    return dispersion_parameters
end

@doc raw"""
    DispersionParameters(electron_phonon_model::ElectronPhononModel{T,E,D}) where {T,E,D}

Initialize and return null (empty) instance of [`DispersionParameters`](@ref).
"""
function DispersionParameters(electron_phonon_model::ElectronPhononModel{T,E,D}) where {T,E,D}

    return DispersionParameters(0, 0, T[], T[], Matrix{Int}(undef,2,0), Vector{Int}[], Vector{Int}[])
end