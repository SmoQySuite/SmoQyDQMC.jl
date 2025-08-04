```@meta
CollapsedDocStrings = true
```

# API

## Simulation Information Type and Methods

```@docs
SimulationInfo
SimulationInfo(;)
initialize_datafolder
model_summary
```

## Model Geometry Type and Methods

```@docs
ModelGeometry
ModelGeometry(::UnitCell{D}, ::Lattice{D}) where {D}
add_bond!
get_bond_id
```

## Fermion Path Integral Type and Methods

```@docs
FermionPathIntegral
FermionPathIntegral(;)
initialize_propagators
calculate_propagators!
calculate_propagator!
```

## Update Numerical Stabilization Frequency

```@docs
update_stabilization_frequency!
```

## Tight-Binding Model

```@docs
TightBindingModel
TightBindingModel(;)
TightBindingParameters
TightBindingParameters(;)
measure_onsite_energy
measure_hopping_energy
measure_bare_hopping_energy
measure_hopping_amplitude
measure_hopping_inversion
measure_hopping_inversion_avg
```

## Hubbard Model

- [Hubbard Model Measurements](@ref)
- [Hubbard Interaction Hubbard-Stratonovich Transformations](@ref)
    - [Spin Channel Hirsch Hubbard-Stratonovich Transformation](@ref)
    - [Spin Channel Gauss-Hermite Hubbard-Stratonovich Transformation](@ref)
    - [Density Channel Hirsch Hubbard-Stratonovich Transformation](@ref)
    - [Density Channel Gauss-Hermite Hubbard-Stratonovich Transformation](@ref)
    - [(LEGACY) Ising Hubbard-Stratonovich Transformation](@ref)

```@docs
HubbardModel
HubbardModel(;)
HubbardParameters
HubbardParameters(;)
initialize!(::FermionPathIntegral, ::FermionPathIntegral, ::HubbardParameters)
```

### Hubbard Model Measurements

```@docs
measure_hubbard_energy
```

### Hubbard Interaction Hubbard-Stratonovich Transformations

Below the different types of Hubbard-Stratonovich transformations (HSTs) that can be used
to decouple a local Hubbard interaction are listed.

#### Spin Channel Hirsch Hubbard-Stratonovich Transformation

```@docs
HubbardSpinHirschHST
HubbardSpinHirschHST(;)
initialize!(::FermionPathIntegral{H}, ::FermionPathIntegral{H}, ::HubbardSpinHirschHST{T}) where {H<:Number, T<:Number}
local_updates!(::Matrix{H}, ::R, ::H, ::Matrix{H}, ::R, ::H, ::HubbardSpinHirschHST{T,R}) where {H<:Number, T<:Number, R<:Real, P<:AbstractPropagator}
reflection_update!(::Matrix{H}, ::R, ::H, ::Matrix{H}, ::R, ::H, ::HubbardSpinHirschHST{T,R};) where {H<:Number, T<:Number, R<:Real, P<:AbstractPropagator}
swap_update!(::Matrix{H}, ::R, ::H, ::Matrix{H}, ::R, ::H, ::HubbardSpinHirschHST{T,R}) where {H<:Number, T<:Number, R<:Real, P<:AbstractPropagator}
```

#### Spin Channel Gauss-Hermite Hubbard-Stratonovich Transformation
```@docs
HubbardSpinGaussHermiteHST
HubbardSpinGaussHermiteHST(;)
initialize!(::FermionPathIntegral{H}, ::FermionPathIntegral{H}, ::HubbardSpinGaussHermiteHST{T}) where {H<:Number, T<:Number}
local_updates!(::Matrix{H}, ::R, ::H, ::Matrix{H}, ::R, ::H, ::HubbardSpinGaussHermiteHST{T,R}) where {H<:Number, T<:Number, R<:Real, P<:AbstractPropagator}
reflection_update!(::Matrix{H}, ::R, ::H, ::Matrix{H}, ::R, ::H, ::HubbardSpinGaussHermiteHST{T,R}) where {H<:Number, T<:Number, R<:Real, P<:AbstractPropagator}
swap_update!(::Matrix{H}, ::R, ::H, ::Matrix{H}, ::R, ::H, ::HubbardSpinGaussHermiteHST{T,R}) where {H<:Number, T<:Number, R<:Real, P<:AbstractPropagator}
```

#### Density Channel Hirsch Hubbard-Stratonovich Transformation

```@docs
HubbardDensityHirschHST
HubbardDensityHirschHST(;)
initialize!(::FermionPathIntegral{H}, ::FermionPathIntegral{H}, ::HubbardDensityHirschHST{T}) where {H<:Number, T<:Number}
local_updates!(::Matrix{H}, ::R, ::H, ::Matrix{H}, ::R, ::H, ::HubbardDensityHirschHST{T,R}) where {H<:Number, T<:Number, R<:Real, P<:AbstractPropagator}
local_updates!(::Matrix{H}, ::R, ::H, ::HubbardDensityHirschHST{T,R}) where {H<:Number, T<:Number, R<:Real, P<:AbstractPropagator}
reflection_update!(::Matrix{H}, ::R, ::H, ::Matrix{H}, ::R, ::H, ::HubbardDensityHirschHST{T,R}) where {H<:Number, T<:Number, R<:Real, P<:AbstractPropagator}
reflection_update!(::Matrix{H}, ::R, ::H, ::HubbardDensityHirschHST{T,R}) where {H<:Number, T<:Number, R<:Real, P<:AbstractPropagator}
swap_update!(::Matrix{H}, ::R, ::H, ::Matrix{H}, ::R, ::H, ::HubbardDensityHirschHST{T,R}) where {H<:Number, T<:Number, R<:Real, P<:AbstractPropagator}
swap_update!(::Matrix{H}, ::R, ::H, ::HubbardDensityHirschHST{T,R}) where {H<:Number, T<:Number, R<:Real, P<:AbstractPropagator}
```

#### Density Channel Gauss-Hermite Hubbard-Stratonovich Transformation

```@docs
HubbardDensityGaussHermiteHST
HubbardDensityGaussHermiteHST(;)
initialize!(::FermionPathIntegral{H}, ::FermionPathIntegral{H}, ::HubbardDensityGaussHermiteHST{T}) where {H<:Number, T<:Number}
local_updates!(::Matrix{H}, ::R, ::H, ::Matrix{H}, ::R, ::H, ::HubbardDensityGaussHermiteHST{T,R}) where {H<:Number, T<:Number, R<:Real, P<:AbstractPropagator}
local_updates!(::Matrix{H}, ::R, ::H, ::HubbardDensityGaussHermiteHST{T,R}) where {H<:Number, T<:Number, R<:Real, P<:AbstractPropagator}
reflection_update!(::Matrix{H}, ::R, ::H, ::Matrix{H}, ::R, ::H, ::HubbardDensityGaussHermiteHST{T,R}) where {H<:Number, T<:Number, R<:Real, P<:AbstractPropagator}
reflection_update!(::Matrix{H}, ::R, ::H, ::HubbardDensityGaussHermiteHST{T,R}) where {H<:Number, T<:Number, R<:Real, P<:AbstractPropagator}
swap_update!(::Matrix{H}, ::R, ::H, ::Matrix{H}, ::R, ::H, ::HubbardDensityGaussHermiteHST{T,R}) where {H<:Number, T<:Number, R<:Real, P<:AbstractPropagator}
swap_update!(::Matrix{H}, ::R, ::H, ::HubbardDensityGaussHermiteHST{T,R}) where {H<:Number, T<:Number, R<:Real, P<:AbstractPropagator}
```

#### (LEGACY) Ising Hubbard-Stratonovich Transformation

This transformation is equivalent to the [Spin Channel Hirsch Hubbard-Stratonovich Transformation](@ref) when the Hubbard interaction is attractive,
and the [Density Channel Hirsch Hubbard-Stratonovich Transformation](@ref) when the Hubbard interaction is attractive.
This ensure that the Hubbard-Stratonovich fields and coefficients remain strictly real regardless of whether the Hubbard interaction is repulsive or attractive.

```@docs
HubbardIsingHSParameters
HubbardIsingHSParameters(;)
initialize!(::FermionPathIntegral, ::FermionPathIntegral, ::HubbardIsingHSParameters)
initialize!(::FermionPathIntegral, ::HubbardIsingHSParameters)
local_updates!(::Matrix{H}, ::R, ::H, ::Matrix{H}, ::R, ::H, ::HubbardIsingHSParameters{R}) where {H<:Number, R<:Real, P<:AbstractPropagator}
local_updates!(::Matrix{H}, ::R, ::H, ::HubbardIsingHSParameters{R}) where {H<:Number, R<:Real, P<:AbstractPropagator}
reflection_update!(::Matrix{H}, ::R, ::H, ::Matrix{H}, ::R, ::H, ::HubbardIsingHSParameters{R}) where {H<:Number, R<:Real, P<:AbstractPropagator}
reflection_update!(::Matrix{H}, ::R, ::H, ::HubbardIsingHSParameters{R}) where {H<:Number, R<:Real, P<:AbstractPropagator}
swap_update!(::Matrix{H}, ::R, ::H, ::Matrix{H}, ::R, ::H, ::HubbardIsingHSParameters{R}) where {H<:Number, R<:Real, P<:AbstractPropagator}
swap_update!(::Matrix{H}, ::R, ::H, ::HubbardIsingHSParameters{R}) where {H<:Number, R<:Real, P<:AbstractPropagator}
```

## Extended Hubbard Model

- [Extended Hubbard Model Measurements](@ref)
- [Extended Hubbard Gauss-Hermite Hubbard-Stratonovich Transformation](@ref)

```@docs
ExtendedHubbardModel
ExtendedHubbardModel(;)
ExtendedHubbardParameters
ExtendedHubbardParameters(;)
initialize!(::FermionPathIntegral, ::FermionPathIntegral, ::ExtendedHubbardParameters)
```

### Extended Hubbard Model Measurements

```@docs
measure_ext_hub_energy
```

### Extended Hubbard Gauss-Hermite Hubbard-Stratonovich Transformation

```@docs
ExtHubDensityGaussHermiteHST
ExtHubDensityGaussHermiteHST(;)
init_renormalized_hubbard_parameters
initialize!(::FermionPathIntegral{H}, ::FermionPathIntegral{H}, ::ExtHubDensityGaussHermiteHST{T}) where {H<:Number, T<:Number}
local_updates!(::Matrix{H}, ::R, ::H, ::Matrix{H}, ::R, ::H, ::ExtHubDensityGaussHermiteHST{T,R}) where {H<:Number, T<:Number, R<:Real, P<:AbstractPropagator}
local_updates!(::Matrix{H}, ::R, ::H, ::ExtHubDensityGaussHermiteHST{T,R};) where {H<:Number, T<:Number, R<:Real, P<:AbstractPropagator}
reflection_update!(::Matrix{H}, ::R, ::H, ::Matrix{H}, ::R, ::H, ::ExtHubDensityGaussHermiteHST{T,R}) where {H<:Number, T<:Number, R<:Real, P<:AbstractPropagator}
reflection_update!(::Matrix{H}, ::R, ::H,::ExtHubDensityGaussHermiteHST{T,R}) where {H<:Number, T<:Number, R<:Real, P<:AbstractPropagator}
swap_update!(::Matrix{H}, ::R, ::H,::Matrix{H}, ::R, ::H, ::ExtHubDensityGaussHermiteHST{T,R}) where {H<:Number, T<:Number, R<:Real, P<:AbstractPropagator}
swap_update!(::Matrix{H}, ::R, ::H, ::ExtHubDensityGaussHermiteHST{T,R}) where {H<:Number, T<:Number, R<:Real, P<:AbstractPropagator}
```

## Electron-Phonon Model

 - [Electron-Phonon Model Types and Method](@ref)
 - [Electron-Phonon Parameter Types and Methods](@ref)
 - [Electron-Phonon Measurements](@ref)
 - [Electron-Phonon Updates](@ref)

### Electron-Phonon Model Types and Method

```@docs
ElectronPhononModel
ElectronPhononModel(;)
PhononMode
PhononMode(;)
HolsteinCoupling
HolsteinCoupling(;)
SSHCoupling
SSHCoupling(;)
PhononDispersion
PhononDispersion(;)
add_phonon_mode!
add_holstein_coupling!
add_ssh_coupling!
add_phonon_dispersion!
```

### Electron-Phonon Parameter Types and Methods

```@docs
ElectronPhononParameters
ElectronPhononParameters(;)
SmoQyDQMC.PhononParameters
SmoQyDQMC.HolsteinParameters
SmoQyDQMC.SSHParameters
SmoQyDQMC.DispersionParameters
initialize!(::FermionPathIntegral{T,E}, ::FermionPathIntegral{T,E}, ::ElectronPhononParameters{T,E}) where {T,E}
initialize!(::FermionPathIntegral{T,E}, ::ElectronPhononParameters{T,E}) where {T,E}
update!(::FermionPathIntegral{T,E}, ::FermionPathIntegral{T,E}, ::ElectronPhononParameters{T,E}, ::Matrix{E}, ::Matrix{E}) where {T,E}
update!(::FermionPathIntegral{T,E}, ::ElectronPhononParameters{T,E}, ::Matrix{E}, ::Matrix{E}) where {T,E}
update!(::FermionPathIntegral{T,E}, ::ElectronPhononParameters{T,E}, ::Matrix{E}, ::Int) where {T,E}
```

### Electron-Phonon Measurements

```@docs
measure_phonon_kinetic_energy
measure_phonon_potential_energy
measure_phonon_position_moment
measure_holstein_energy
measure_ssh_energy
measure_dispersion_energy
```

### Electron-Phonon Updates

```@docs
EFAHMCUpdater
EFAHMCUpdater(;)
hmc_update!(::Matrix{H}, ::R, ::H, ::Matrix{H}, ::R, ::H, ::ElectronPhononParameters{T,R}, ::EFAHMCUpdater{T,R}) where {H<:Number, T<:Number, R<:Real, P<:AbstractPropagator{T}}
hmc_update!(::Matrix{H}, ::R, ::H, ::ElectronPhononParameters{T,R}, ::EFAHMCUpdater{T,R}) where {H<:Number, T<:Number, R<:Real, P<:AbstractPropagator{T}}
reflection_update!(::Matrix{H}, ::R, ::H, ::Matrix{H}, ::R, ::H, ::ElectronPhononParameters{T,R}) where {H<:Number, T<:Number, R<:Real, P<:AbstractPropagator{T}}
reflection_update!(::Matrix{H}, ::R, ::H, ::ElectronPhononParameters{T,R}) where {H<:Number, T<:Number, R<:Real, P<:AbstractPropagator{T}}
swap_update!(::Matrix{H}, ::R, ::H, ::Matrix{H}, ::R, ::H, ::ElectronPhononParameters{T,R}) where {H<:Number, T<:Number, R<:Real, P<:AbstractPropagator{T}}
swap_update!(::Matrix{H}, ::R, ::H, ::ElectronPhononParameters{T,R};) where {H<:Number, T<:Number, R<:Real, P<:AbstractPropagator{T}}
radial_update!(::Matrix{H}, ::R, ::H, ::Matrix{H}, ::R, ::H, ::ElectronPhononParameters{T,R}) where {H<:Number, T<:Number, R<:Real, P<:AbstractPropagator{T}}
radial_update!(::Matrix{H}, ::R, ::H, ::ElectronPhononParameters{T,R}) where {H<:Number, T<:Number, R<:Real, P<:AbstractPropagator{T}}
```

## Density and Chemical Potential Tuning

```@docs
update_chemical_potential!
save_density_tuning_profile
```

## Measurement Methods

- [Measurement Names](@ref)
- [Initialize Measurements](@ref)
- [Make Measurements](@ref)
- [Write Measurements](@ref)
- [Checkpointing Utilities](@ref)
- [Process Measurements](@ref)
- [Export Measurements](@ref)

### Measurement Names

```@docs
GLOBAL_MEASUREMENTS
LOCAL_MEASUREMENTS
CORRELATION_FUNCTIONS
```

### Initialize Measurements

```@docs
initialize_measurement_container
initialize_measurements!
initialize_correlation_measurements!
initialize_composite_correlation_measurement!
```

### Make Measurements

```@docs
make_measurements!
```

### Write Measurements

```@docs
write_measurements!
merge_bins
rm_bins
```

## Checkpointing Utilities

```@docs
write_jld2_checkpoint
read_jld2_checkpoint
rm_jld2_checkpoints
rename_complete_simulation
```

### Process Measurements

```@docs
save_simulation_info
process_measurements
compute_correlation_ratio
compute_composite_correlation_ratio
compute_function_of_correlations
```

### Export Measurements

```@docs
export_global_stats_to_csv
export_global_bins_to_h5
export_global_bins_to_csv
export_local_stats_to_csv
export_local_bins_to_csv
export_local_bins_to_h5
export_correlation_stats_to_csv
export_correlation_bins_to_csv
export_correlation_bins_to_h5
```