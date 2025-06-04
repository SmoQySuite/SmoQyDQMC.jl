@doc raw"""
    PhononParameters{E<:AbstractFloat}

Defines the parameters for each phonon in the lattice, includes the phonon field configuration.

# Fields

- `nphonon::Int`: Number of type of phonon modes.
- `Nphonon::Int`: Total number of phonon modes in finite lattice.
- `M::Int`: Mass of each phonon mode.
- `Ω::Int`: Frequency of each phonon mode.
- `Ω4::Int`: Quartic phonon coefficient for each phonon mode.
- `basis_vecs::Vector{Vector{E}}`: Basis vector for each of the `nphonon` types of phonon mode.`
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

    # basis vector for each type of phonon mode
    basis_vecs::Vector{Vector{E}}
end

@doc raw"""
    PhononParameters(;
        # KEYWORD ARGUMENTS
        model_geometry::ModelGeometry{D,E},
        electron_phonon_model::ElectronPhononModel{T,E,D},
        rng::AbstractRNG
    ) where {T,E,D}

Initialize and return an instance of [`PhononParameters`](@ref).
"""
function PhononParameters(;
    # KEYWORD ARGUMENTS
    model_geometry::ModelGeometry{D,E},
    electron_phonon_model::ElectronPhononModel{T,E,D},
    rng::AbstractRNG
) where {T,E,D}

    lattice = model_geometry.lattice::Lattice{D}

    # get number of unit cells
    Ncells = lattice.N

    # get the phonon mode defintions
    phonon_modes = electron_phonon_model.phonon_modes

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

    # get basis vectors
    basis_vecs = [[phonon_modes.basis_vec...] for phonon_modes in phonon_modes]

    # iterate over phonon modes
    phonon = 0 # phonon counter
    for nph in 1:nphonon
        # get the phonon mode
        phonon_mode = phonon_modes[nph]::PhononMode{E}
        # iterate over unit cells in lattice
        for uc in 1:Ncells
            # increment phonon counter
            phonon += 1
            # assign phonon mass
            M[phonon] = phonon_mode.M
            # assign phonon freuqency
            Ω[phonon] = phonon_mode.Ω_mean + phonon_mode.Ω_std * randn(rng)
            # assign quartic phonon coefficient
            Ω4[phonon] = phonon_mode.Ω4_mean + phonon_mode.Ω4_std * randn(rng)
        end
    end

    return PhononParameters(nphonon, Nphonon, M, Ω, Ω4, basis_vecs)
end