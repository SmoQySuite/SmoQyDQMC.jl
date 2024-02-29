##############################################
## DEFINE TYPES OF CORRELATION MEASUREMENTS ##
##############################################

@doc raw"""
    CORRELATION_FUNCTIONS = Base.ImmutableDict(
        "greens"           => "ORBITAL_ID",
        "greens_up"        => "ORBITAL_ID",
        "greens_dn"        => "ORBITAL_ID",
        "greens_tautau"    => "ORBITAL_ID",
        "greens_tautau_up" => "ORBITAL_ID",
        "greens_tautau_dn" => "ORBITAL_ID",
        "density"          => "ORBITAL_ID",
        "density_upup"     => "ORBITAL_ID",
        "density_dndn"     => "ORBITAL_ID",
        "density_updn"     => "ORBITAL_ID",
        "density_dnup"     => "ORBITAL_ID",
        "spin_x"           => "ORBITAL_ID",
        "spin_z"           => "ORBITAL_ID",
        "pair"             => "BOND_ID",
        "bond"             => "BOND_ID",
        "bond_upup"        => "BOND_ID",
        "bond_dndn"        => "BOND_ID",
        "bond_updn"        => "BOND_ID",
        "bond_dnup"        => "BOND_ID",
        "current"          => "HOPPING_ID",
        "current_upup"     => "HOPPING_ID",
        "current_dndn"     => "HOPPING_ID",
        "current_updn"     => "HOPPING_ID",
        "current_dnup"     => "HOPPING_ID",
        "phonon_greens"    => "PHONON_ID"
    )

List of all the correlation functions that can be measured, along with the corresponding
type of ID the correlation measurement is reported in terms of.
Correlation functions are well defined in both position and momentum space.
"""
const CORRELATION_FUNCTIONS = Base.ImmutableDict(
    "greens"           => "ORBITAL_ID",
    "greens_up"        => "ORBITAL_ID",
    "greens_dn"        => "ORBITAL_ID",
    "greens_tautau"    => "ORBITAL_ID",
    "greens_tautau_up" => "ORBITAL_ID",
    "greens_tautau_dn" => "ORBITAL_ID",
    "density"          => "ORBITAL_ID",
    "density_upup"     => "ORBITAL_ID",
    "density_dndn"     => "ORBITAL_ID",
    "density_updn"     => "ORBITAL_ID",
    "density_dnup"     => "ORBITAL_ID",
    "spin_x"           => "ORBITAL_ID",
    "spin_z"           => "ORBITAL_ID",
    "pair"             => "BOND_ID",
    "bond"             => "BOND_ID",
    "bond_upup"        => "BOND_ID",
    "bond_dndn"        => "BOND_ID",
    "bond_updn"        => "BOND_ID",
    "bond_dnup"        => "BOND_ID",
    "current"          => "HOPPING_ID",
    "current_upup"     => "HOPPING_ID",
    "current_dndn"     => "HOPPING_ID",
    "current_updn"     => "HOPPING_ID",
    "current_dnup"     => "HOPPING_ID",
    "phonon_greens"    => "PHONON_ID"
)


@doc raw"""
    const LOCAL_MEASUREMENTS = Base.ImmutableDict(
        "density"                  => "ORBITAL_ID",
        "density_up"               => "ORBITAL_ID",
        "density_dn"               => "ORBITAL_ID",
        "double_occ"               => "ORBITAL_ID",
        "onsite_energy"            => "ORBITAL_ID",
        "onsite_energy_up"         => "ORBITAL_ID",
        "onsite_energy_dn"         => "ORBITAL_ID",
        "bare_hopping_energy"      => "HOPPING_ID",
        "bare_hopping_energy_up"   => "HOPPING_ID",
        "bare_hopping_energy_dn"   => "HOPPING_ID",
        "hopping_energy"           => "HOPPING_ID",
        "hopping_energy_up"        => "HOPPING_ID",
        "hopping_energy_dn"        => "HOPPING_ID",
        "hopping_amplitude_up"     => "HOPPING_ID",
        "hopping_amplitude_dn"     => "HOPPING_ID",
        "hopping_amplitude"        => "HOPPING_ID",
        "hopping_inversion_avg_up" => "HOPPING_ID",
        "hopping_inversion_avg_dn" => "HOPPING_ID",
        "hopping_inversion_avg"    => "HOPPING_ID",
        "hopping_inversion_up"     => "HOPPING_ID",
        "hopping_inversion_dn"     => "HOPPING_ID",
        "hopping_inversion"        => "HOPPING_ID",
        "hubbard_energy"           => "ORBITAL_ID",
        "phonon_kin_energy"        => "PHONON_ID",
        "phonon_pot_energy"        => "PHONON_ID",
        "X"                        => "PHONON_ID",
        "X2"                       => "PHONON_ID",
        "X3"                       => "PHONON_ID",
        "X4"                       => "PHONON_ID",
        "holstein_energy"          => "HOLSTEIN_ID",
        "holstein_energy_up"       => "HOLSTEIN_ID",
        "holstein_energy_dn"       => "HOLSTEIN_ID",
        "ssh_energy"               => "SSH_ID",
        "ssh_energy_up"            => "SSH_ID",
        "ssh_energy_dn"            => "SSH_ID",
        "dispersion_energy"        => "DISPERSION_ID"
    )

List of all the local measurements than can be made, with a mapping to the
corresponding type of ID each measurement is reported in terms of.
"""
const LOCAL_MEASUREMENTS = Base.ImmutableDict(
    "density"                  => "ORBITAL_ID",
    "density_up"               => "ORBITAL_ID",
    "density_dn"               => "ORBITAL_ID",
    "double_occ"               => "ORBITAL_ID",
    "onsite_energy"            => "ORBITAL_ID",
    "onsite_energy_up"         => "ORBITAL_ID",
    "onsite_energy_dn"         => "ORBITAL_ID",
    "bare_hopping_energy"      => "HOPPING_ID",
    "bare_hopping_energy_up"   => "HOPPING_ID",
    "bare_hopping_energy_dn"   => "HOPPING_ID",
    "hopping_energy"           => "HOPPING_ID",
    "hopping_energy_up"        => "HOPPING_ID",
    "hopping_energy_dn"        => "HOPPING_ID",
    "hopping_amplitude_up"     => "HOPPING_ID",
    "hopping_amplitude_dn"     => "HOPPING_ID",
    "hopping_amplitude"        => "HOPPING_ID",
    "hopping_inversion_avg_up" => "HOPPING_ID",
    "hopping_inversion_avg_dn" => "HOPPING_ID",
    "hopping_inversion_avg"    => "HOPPING_ID",
    "hopping_inversion_up"     => "HOPPING_ID",
    "hopping_inversion_dn"     => "HOPPING_ID",
    "hopping_inversion"        => "HOPPING_ID",
    "hubbard_energy"           => "ORBITAL_ID",
    "phonon_kin_energy"        => "PHONON_ID",
    "phonon_pot_energy"        => "PHONON_ID",
    "X"                        => "PHONON_ID",
    "X2"                       => "PHONON_ID",
    "X3"                       => "PHONON_ID",
    "X4"                       => "PHONON_ID",
    "holstein_energy"          => "HOLSTEIN_ID",
    "holstein_energy_up"       => "HOLSTEIN_ID",
    "holstein_energy_dn"       => "HOLSTEIN_ID",
    "ssh_energy"               => "SSH_ID",
    "ssh_energy_up"            => "SSH_ID",
    "ssh_energy_dn"            => "SSH_ID",
    "dispersion_energy"        => "DISPERSION_ID"
)

@doc raw"""
    const GLOBAL_MEASUREMENTS = (
        "logdetGup",
        "logdetGdn",
        "sgndetGup",
        "sgndetGdn",
        "sgn",
        "action_total",
        "action_bosonic",
        "action_fermionic",
        "density",
        "density_up",
        "density_dn",
        "double_occ",
        "Nsqrd"
    )

List of all the global measurements that are made.
"""
const GLOBAL_MEASUREMENTS = (
    "logdetGup",
    "logdetGdn",
    "sgndetGup",
    "sgndetGdn",
    "sgn",
    "action_total",
    "action_bosonic",
    "action_fermionic",
    "density",
    "density_up",
    "density_dn",
    "double_occ",
    "Nsqrd"
)