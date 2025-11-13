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
using CodecZlib
using FastLapackInterface
using Reexport
using PkgVersion
using TOML
using Glob
using MPI
using Format
using HDF5

# import "our" packages
using MuTuner
using Checkerboard
using LatticeUtilities
using JDQMCFramework
using JDQMCMeasurements
using StableLinearAlgebra

# import methods from to overload
import LinearAlgebra: mul!, lmul!, rmul!

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
export update_stabilization_frequency!

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
export FermionPathIntegral
export initialize_propagators, calculate_propagators!, calculate_propagator!

# method for updating chemical potential using MuTuner
include("update_chemical_potential.jl")
export update_chemical_potential!, save_density_tuning_profile

# utility functions for implementing a Gauss-Hermite Hubbard-Stratonovich Transformation
include("GaussHermiteHSTUtilities.jl")

# define abstract Hubbard-Stratonovich type
include("AbstractHST.jl")
export AbstractHST, AbstractSymHST, AbstractAsymHST
export local_updates!, reflection_updates!, swap_updates!

###################
## HUBBARD MODEL ##
###################

# Define HubbardModel
include("Hubbard/HubbardModel.jl")
export HubbardModel, HubbardParameters, initialize!

# Spin-Channel Hirsch Hubbard-Stratonovich Transformation
include("Hubbard/HubbardSpinHirschHST.jl")
export HubbardSpinHirschHST

# Charge-Channel Hirsch Hubbard-Stratonovich Transformation
include("Hubbard/HubbardDensityHirschHST.jl")
export HubbardDensityHirschHST

# Spin-Channel Gauss-Hermite Hubbard-Stratonovich Transformation
include("Hubbard/HubbardSpinGaussHermiteHST.jl")
export HubbardSpinGaussHermiteHST

# Density-Channel Gauss-Hermite Hubbard-Stratonovich Transformation
include("Hubbard/HubbardDensityGaussHermiteHST.jl")
export HubbardDensityGaussHermiteHST

############################
## EXTENDED HUBBARD MODEL ##
############################

# Define ExtendedHubbardModel
include("ExtendedHubbard/ExtendedHubbardModel.jl")
export ExtendedHubbardModel

# Define ExtendedHubbardParameters
include("ExtendedHubbard/ExtendedHubbardParameters.jl")
export ExtendedHubbardParameters

# Define Extended Hubbard model local energy measurement
include("ExtendedHubbard/ext_hub_model_measurements.jl")
export measure_ext_hub_energy

# Define Extended Hubbard Density Channel Gauss-Hermite Hubbard-Stratonovich Transformation
include("ExtendedHubbard/ExtHubDensityGaussHermiteHST.jl")
export ExtHubDensityGaussHermiteHST, init_renormalized_hubbard_parameters

# Define Extended Hubbard Spin Channel Hirsch Hubbard-Stratonovich Transformation
include("ExtendedHubbard/ExtHubSpinHirschHST.jl")
export ExtHubSpinHirschHST

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

# implement exact fourier acceleration integration of equation of motion
include("ElectronPhonon/ExactFourierAccelerator.jl")

# defines Exact Fourier Acceleration HMC update method
include("ElectronPhonon/EFAHMCUpdater.jl")
export EFAHMCUpdater, hmc_update!

# impelment reflection, swap and radial updates for phonon fields
include("ElectronPhonon/reflection_update.jl")
include("ElectronPhonon/swap_update.jl")
include("ElectronPhonon/radial_update.jl")
export reflection_update!, swap_update!, radial_update!

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
export measure_ssh_energy

# measurements for phonon dispersion
include("ElectronPhonon/dispersion_measurements.jl")
export measure_dispersion_energy

# defines dictionaries as global variables that contain the names of all
# local measurements and correlation measurements that can be made, and the
# type ID type they are reported in terms of
include("Measurements/measurement_names.jl")
export GLOBAL_MEASUREMENTS
export LOCAL_MEASUREMENTS
export CORRELATION_FUNCTIONS

# Define CorrelationContainer struct to store correlation measurements in.
include("Measurements/CorrelationContainer.jl")

# Define CompositeCorrelationContainer struct to store composite correlation measurements
include("Measurements/CompositeCorrelationContainer.jl")

# initialize measurement container
include("Measurements/initialize_measurements.jl")
export initialize_measurement_container
export initialize_measurements!
export initialize_correlation_measurement!, initialize_correlation_measurements!
export initialize_composite_correlation_measurement!
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

# implements function to merge HDF5 bin files for a given pID (process ID)
# into a single HDF5 file. Also includes functions to delete all binned data.
include("Measurements/merge_bins.jl")
export merge_bins, rm_bins

# implementes utility function for converting numbers to string
include("Measurements/num_to_string_formatter.jl")

# functions for exporting binned global measurements to file
include("Measurements/export_global_bins.jl")
export export_global_bins_to_h5, export_global_bins_to_csv

# functions for exporting binned local measurements to file
include("Measurements/export_local_bins.jl")
export export_local_bins_to_h5, export_local_bins_to_csv

# function for exporting binned correlation data to HDF5 file
include("Measurements/export_correlation_bins_to_h5.jl")
export export_correlation_bins_to_h5

# function for exporting binned correlation data to CSV file
include("Measurements/export_correlation_bins_to_csv.jl")
export export_correlation_bins_to_csv

# function for exporting global measurement stats to csv file
include("Measurements/export_global_stats_to_csv.jl")
export export_global_stats_to_csv

# function for exporting global measurement stats to csv file
include("Measurements/export_local_stats_to_csv.jl")
export export_local_stats_to_csv

# function for exporting correlation measurement stats to csv file
include("Measurements/export_correlation_stats_to_csv.jl")
export export_correlation_stats_to_csv

# internal functions for processing the binned data to calculate final
# statistics using a single process
include("Measurements/process_measurements_internals.jl")

# internal functions for processing the binned data to calculate final
# statistics using MPI parallelization to accelerate the computation
include("Measurements/process_measurements_internals_mpi.jl")

# public api functions for processing measurements
include("Measurements/process_measurements.jl")
export process_measurements

# export functions for computing correlation ratios
include("Measurements/compute_correlation_ratio.jl")
export compute_correlation_ratio, compute_composite_correlation_ratio

# export function to compute function of correlation measurements
include("Measurements/compute_function_of_correlations.jl")
export compute_function_of_correlations

# utilties for checkpoint simulations
include("Measurements/checkpointing_utilities.jl")
export write_jld2_checkpoint, read_jld2_checkpoint
export rm_jld2_checkpoints, rename_complete_simulation

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
