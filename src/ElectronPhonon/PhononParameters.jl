@doc raw"""
    PhononParameters{E<:AbstractFloat}

Defines the parameters for each phonon in the lattice, includes the phonon field configuration.

# Fields

- `nphonon::Int`: Number of type of phonon modes.
- `Nphonon::Int`: Total number of phonon modes in finite lattice.
- `M::Int`: Mass of each phonon mode.
- `Ω::Int`: Frequency of each phonon mode.
- `Ω4::Int`: Quartic phonon coefficient for each phonon mode.
- `phonon_to_site::Vector{Int}`: Map each phonon to the site it lives on in the lattice.
- `site_to_phonons::Vector{Vector{Int}}`: Maps the site to the phonon modes on it, allowing for multiple modes to reside on a single site.
"""
struct PhononParameters{E<:AbstractFloat}

    # number of types of phonon modes
    nphonon::Int

    # number of phonon modes
    Nphonon::Int

    # phonon masses
    M::Vector{E}

    # phonon frequency
    Ω::Vector{E}

    # quartic coefficient for phonon potential energy (X⁴)
    Ω4::Vector{E}

    # map phonon field to site in lattice
    phonon_to_site::Vector{Int}

    # map sites to phonon fields (note that multiple fields may live on a single site)
    site_to_phonons::Vector{Vector{Int}}
end

@doc raw"""
    PhononParameters(; model_geometry::ModelGeometry{D,E},
                     electron_phonon_model::ElectronPhononModel{T,E,D},
                     rng::AbstractRNG) where {T,E,D}

Initialize and return an instance of [`PhononParameters`](@ref).
"""
function PhononParameters(; model_geometry::ModelGeometry{D,E},
                          electron_phonon_model::ElectronPhononModel{T,E,D},
                          rng::AbstractRNG) where {T,E,D}

    lattice = model_geometry.lattice::Lattice{D}
    unit_cell = model_geometry.unit_cell::UnitCell{D,E}

    # get totals number of sites/orbitals in lattice
    Nsites = nsites(unit_cell, lattice)

    # get number of unit cells
    Ncells = lattice.N

    # get the phonon mode defintions
    phonon_modes = electron_phonon_model.phonon_modes::Vector{PhononMode{E}}

    # get the number of phonon mode definitions
    nphonon = length(phonon_modes)

    # get the total number of phonon modes in the lattice
    Nphonon = nphonon * Ncells

    # allocate array of masses for each phonon mode
    M = zeros(E,Nphonon)

    # allocate array of phonon frequncies for each phonon mode
    Ω = zeros(E,Nphonon)

    # allocate array of quartic coefficient for each phonon mode
    Ω4 = zeros(E,Nphonon)

    # allocate phonon_to_site
    phonon_to_site = zeros(Int, Nphonon)

    # allocate site_to_phonons
    site_to_phonons = [Int[] for i in 1:Nsites]

    # iterate over phonon modes
    phonon = 0 # phonon counter
    for nph in 1:nphonon
        # get the phonon mode
        phonon_mode = phonon_modes[nph]::PhononMode{E}
        # get the orbital species associated with phonon mode
        orbital = phonon_mode.orbital
        # iterate over unit cells in lattice
        for uc in 1:Ncells
            # increment phonon counter
            phonon += 1
            # get site associated with phonon mode
            site = loc_to_site(uc, orbital, unit_cell)
            # record phonon ==> site
            phonon_to_site[phonon] = site
            # record site ==> phonon
            push!(site_to_phonons[site], phonon)
            # assign phonon mass
            M[phonon] = phonon_mode.M
            # assign phonon freuqency
            Ω[phonon] = phonon_mode.Ω_mean + phonon_mode.Ω_std * randn(rng)
            # assign quartic phonon coefficient
            Ω4[phonon] = phonon_mode.Ω4_mean + phonon_mode.Ω4_std * randn(rng)
        end
    end

    return PhononParameters(nphonon, Nphonon, M, Ω, Ω4, phonon_to_site, site_to_phonons)
end