module SmoQyDQMC

# import external dependencies
using LinearAlgebra
using FFTW
using Random
using Statistics
using Printf
using StaticArrays
using OffsetArrays
using JLD2
using FastLapackInterface
using BinningAnalysis
using Reexport
using PkgVersion
using TOML
using Glob
using MPI

# import "our" packages
using MuTuner
using Checkerboard
using LatticeUtilities
using JDQMCFramework
using JDQMCMeasurements
using StableLinearAlgebra

# import methods from to overload
import LinearAlgebra: mul!, lmul!, rmul!

# importing routines for multiplying a dense matrix by a diagonal matrix that
# is represented by a vector
import StableLinearAlgebra: mul_D!, lmul_D!, rmul_D!, div_D!, ldiv_D!, rdiv_D!

# get and set package version number as global constant
const SMOQYDQMC_VERSION = PkgVersion.@Version 0

# re-export external package/modules as sub-modules
@reexport import LatticeUtilities
@reexport import MuTuner
@reexport import Checkerboard
@reexport import JDQMCFramework
@reexport import JDQMCMeasurements

###########################################
## GENERAL DQMC SIMULATION INFASTRUCTURE ##
###########################################

# useful functions to called elsewhere in the code
include("utilities.jl")

# function to update the frequency of numerical stabilization
include("update_stabilization_frequency.jl")
export update_stabalization_frequency!

# define SimulationInfo struct for tracking things like simulation ID, process ID,
# and data folder location
include("SimulationInfo.jl")
export SimulationInfo, save_simulation_info, initialize_datafolder

# defines all aspects of model geometry appearing in model, including the UnitCell,
# Lattice, and a list of Bond defintions as defined in the LatticeUtilities package
include("ModelGeometry.jl")
export ModelGeometry, add_bond!, get_bond_id

# Deals with defining the non-interacting tight-binding Hamiltonian that appears in the model
include("TightBinding.jl")
export TightBindingModel, TightBindingParameters

# Exports the FermionPathIntegral type that describes the hopping matrices K and on-site energy
# matrices V for each imaginary time slice l
include("FermionPathIntegral.jl")
export FermionPathIntegral, initialize_propagators, calculate_propagators!, calculate_propagator!

# method for updating chemical potential using MuTuner
include("update_chemical_potential.jl")
export update_chemical_potential!, save_density_tuning_profile

###################
## HUBBARD MODEL ##
###################

# Define HubbardModel
include("Hubbard/HubbardModel.jl")
export HubbardModel, HubbardParameters, initialize!

# Implement Ising Hubbard-Statonovich (HS) decoupling of Hubbard interaction, and various methods for update the IS HS fields
include("Hubbard/HubbardIsingHS.jl")
export HubbardIsingHSParameters, local_updates!

###########################
## ELECTRON-PHONON MODEL ##
###########################

# Define electron-phonon model agnostic to lattice size
include("ElectronPhonon/ElectronPhononModel.jl")
export ElectronPhononModel, PhononMode, HolsteinCoupling, SSHCoupling, PhononDispersion
export add_phonon_mode!, add_holstein_coupling!, add_ssh_coupling!, add_phonon_dispersion!

# Define various electron-phonon parameter i.e. given a electron-phonon model,
# define all the parameters in the model given a specific finite lattice size
include("ElectronPhonon/PhononParameters.jl")
include("ElectronPhonon/HolsteinParameters.jl")
include("ElectronPhonon/SSHParameters.jl")
include("ElectronPhonon/DispersionParameters.jl")
include("ElectronPhonon/ElectronPhononParameters.jl")
export ElectronPhononParameters, update!

# methods for evaluating the bosonic action Sb and its derivative with respect to phonon fields ∂Sb/∂x
include("ElectronPhonon/bosonic_action.jl")

# methods for evaluating the derivative of the fermionic action with respect to phonon fields ∂Sf/∂x
include("ElectronPhonon/fermionic_action_derivative.jl")

# implements fourier mass matrix to use in HMC/Langevin updates, which gives us fourier acceleration
include("ElectronPhonon/FourierMassMatrix.jl")

# low-level (private) hybrid/hamiltonian monte carlo (HMC) update method
include("ElectronPhonon/hmc_update.jl")

# defines HMC udpater struct and public API for perform HMC updates to phonon fields
include("ElectronPhonon/HMCUpdater.jl")
export HMCUpdater, hmc_update!

# defines Exact Fourier Acceleration HMC update method
include("ElectronPhonon/EFAHMCUpdater.jl")
export EFAHMCUpdater

# impelment reflection and swap updates for phonon fields
include("ElectronPhonon/reflection_update.jl")
include("ElectronPhonon/swap_update.jl")
export reflection_update!, swap_update!

##########################################
## MEASUREMENTS, DATA ANALYSIS & OUTPUT ##
##########################################

# method to write model summary/definition to TOML file
include("model_summary.jl")
export model_summary

# implement tight-bding Hamiltonian measurements
include("tight_binding_measurements.jl")
export measure_onsite_energy, measure_hopping_energy, measure_bare_hopping_energy
export measure_hopping_amplitude, measure_hopping_inversion, measure_hopping_inversion_avg

# relevant hubbard specific measurements
include("Hubbard/hubbard_model_measurements.jl")
export measure_hubbard_energy

# measurements associated with bare phonon modes
include("ElectronPhonon/phonon_measurements.jl")
export measure_phonon_kinetic_energy, measure_phonon_potential_energy, measure_phonon_position_moment

# measurements for holstein interaction
include("ElectronPhonon/holstein_measurements.jl")
export measure_holstein_energy

# measurements for ssh interaction
include("ElectronPhonon/ssh_measurements.jl")
export measure_ssh_energy, measure_ssh_sgn_switch

# measurements for phonon dispersion
include("ElectronPhonon/dispersion_measurements.jl")
export measure_dispersion_energy

# defines dictionaries as global variables that contain the names of all
# local measurements and correlation measurements that can be made, and the
# type ID type they are reported in terms of
include("Measurements/measurement_name_dicts.jl")
export LOCAL_MEASUREMENTS
export CORRELATION_FUNCTIONS

# Define CorrelationContainer struct to store correlation measurements in.
include("Measurements/CorrelationContainer.jl")

# initialize measurement container
include("Measurements/initialize_measurements.jl")
export initialize_measurement_container
export initialize_measurements!
export initialize_correlation_measurement!, initialize_correlation_measurements!
export initialize_measurement_directories

# make measurements
include("Measurements/make_measurements.jl")
export make_measurements!

# write measurements to file.
# additionally, the two following things are done here:
# 1. fourier transform position space correlation to momentum space
# 2. perform integration over imaginary time of correlation function to calculate susceptibilies
include("Measurements/write_measurements.jl")
export write_measurements!

# process measurements at end of the simulation to get final averages and error bars for all measurements
include("Measurements/process_measurements_utils.jl")
include("Measurements/process_global_measurements.jl")
export process_global_measurements
include("Measurements/process_local_measurements.jl")
export process_local_measurements
include("Measurements/process_correlation_measurements.jl")
include("Measurements/process_correlation_measurements_mpi.jl")
export process_correlation_measurement, process_correlation_measurements
include("Measurements/process_measurements.jl")
export process_measurements

# process composite correlation measurements i.e. calculate functions of correlation functions
include("Measurements/process_composite_correlation.jl")
export composite_correlation_stat

# tools for converted binned data, that is saved as *.jld2 binary files, to single csv file
include("Measurements/binned_data_to_csv.jl")
export global_measurement_bins_to_csv, local_measurement_bins_to_csv, correlation_bins_to_csv

############################
## PACKAGE INITIALIZATION ##
############################

# set number of threads for FFTW and BLAS to 1.
# we assume for now the default OpenBLAS that ships with Julia is used.
# this behavior will in general need to be changed if we want to use other BLAS/LAPACK libraries.
function __init__()

    BLAS.set_num_threads(1)
    FFTW.set_num_threads(1)
    return nothing
end

end
