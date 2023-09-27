##############################################
## DEFINE TYPES OF CORRELATION MEASUREMENTS ##
##############################################

@doc raw"""
    const CORRELATION_FUNCTIONS = Dict(
        "greens"           => "ORBITAL_ID",
        "greens_up"        => "ORBITAL_ID",
        "greens_dn"        => "ORBITAL_ID",
        "greens_tautau"    => "ORBITAL_ID",
        "greens_tautau_up" => "ORBITAL_ID",
        "greens_tautau_dn" => "ORBITAL_ID",
        "density"          => "ORBITAL_ID",
        "spin_x"           => "ORBITAL_ID",
        "spin_z"           => "ORBITAL_ID",
        "pair"             => "BOND_ID",
        "bond"             => "BOND_ID",
        "phonon_greens"    => "PHONON_ID",
        "current"          => "HOPPING_ID"
    )

List of all the correlation functions that can be measured, along with the corresponding
type of ID the correlation measurement is reported in terms of.
Correlation functions are well defined in both position and momentum space.
"""
const CORRELATION_FUNCTIONS = Dict(
    "greens"           => "ORBITAL_ID",
    "greens_up"        => "ORBITAL_ID",
    "greens_dn"        => "ORBITAL_ID",
    "greens_tautau"    => "ORBITAL_ID",
    "greens_tautau_up" => "ORBITAL_ID",
    "greens_tautau_dn" => "ORBITAL_ID",
    "density"          => "ORBITAL_ID",
    "spin_x"           => "ORBITAL_ID",
    "spin_z"           => "ORBITAL_ID",
    "pair"             => "BOND_ID",
    "bond"             => "BOND_ID",
    "phonon_greens"    => "PHONON_ID",
    "current"          => "HOPPING_ID"
)


@doc raw"""
    const LOCAL_MEASUREMENTS = Dict(
        "density"           => "ORBITAL_ID",
        "double_occ"        => "ORBITAL_ID",
        "onsite_energy"     => "ORBITAL_ID",
        "hopping_energy"    => "HOPPING_ID",
        "hubbard_energy"    => "ORBITAL_ID",
        "phonon_kin_energy" => "PHONON_ID",
        "phonon_pot_energy" => "PHONON_ID",
        "X"                 => "PHONON_ID",
        "X2"                => "PHONON_ID",
        "X3"                => "PHONON_ID",
        "X4"                => "PHONON_ID",
        "holstein_energy"   => "HOLSTEIN_ID",
        "ssh_energy"        => "SSH_ID",
        "ssh_sgn_switch"    => "SSH_ID",
        "dispersion_energy" => "DISPERSION_ID"
    )

List of all the local measurements than can be made, with a mapping to the
corresponding type of ID each measurement is reported in terms of.
"""
const LOCAL_MEASUREMENTS = Dict(
    "density"           => "ORBITAL_ID",
    "double_occ"        => "ORBITAL_ID",
    "onsite_energy"     => "ORBITAL_ID",
    "hopping_energy"    => "HOPPING_ID",
    "hubbard_energy"    => "ORBITAL_ID",
    "phonon_kin_energy" => "PHONON_ID",
    "phonon_pot_energy" => "PHONON_ID",
    "X"                 => "PHONON_ID",
    "X2"                => "PHONON_ID",
    "X3"                => "PHONON_ID",
    "X4"                => "PHONON_ID",
    "holstein_energy"   => "HOLSTEIN_ID",
    "ssh_energy"        => "SSH_ID",
    "ssh_sgn_switch"    => "SSH_ID",
    "dispersion_energy" => "DISPERSION_ID"
)
