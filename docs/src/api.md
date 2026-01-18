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
```

## Hubbard Model

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

## Extended Hubbard Model

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

## Hubbard-Stratonovich Transformations

Below are the abstract types used to represent generic Hubbard-Stratonovich transformations.

```@docs
AbstractHST
AbstractSymHST
AbstractAsymHST
```

Below is the shared API for the [`AbstractHST`](@ref) type.

```@docs
initialize!(::FermionPathIntegral{H}, ::FermionPathIntegral{H}, ::AbstractHST{T}) where {H<:Number, T<:Number}
local_updates!(::Matrix{H}, ::R, ::H, ::Matrix{H}, ::R, ::H, ::AbstractHST{T,R}) where {H<:Number, T<:Number, R<:Real, P<:AbstractPropagator}
local_updates!(::Matrix{H}, ::R, ::H, ::Matrix{H}, ::R, ::H, ::Tuple) where {H<:Number, R<:Real, P<:AbstractPropagator}
local_updates!(::Matrix{H}, ::R, ::H, ::AbstractSymHST{T,R}) where {H<:Number, T<:Number, R<:Real, P<:AbstractPropagator}
local_updates!(::Matrix{H}, ::R, ::H, ::Tuple) where {H<:Number, R<:Real, P<:AbstractPropagator}
reflection_update!(::Matrix{H}, ::R, ::H, ::Matrix{H}, ::R, ::H, ::AbstractHST{T,R}) where {H<:Number, T<:Number, R<:Real, P<:AbstractPropagator}
reflection_update!(::Matrix{H}, ::R, ::H, hst_parameters::AbstractSymHST{T,R}) where {H<:Number, T<:Number, R<:Real, P<:AbstractPropagator}
swap_update!(::Matrix{H}, ::R, ::H, ::Matrix{H}, ::R, ::H, ::AbstractHST{T,R}) where {H<:Number, T<:Number, R<:Real, P<:AbstractPropagator}
swap_update!(::Matrix{H}, ::R, ::H, hst_parameters::AbstractSymHST{T,R}) where {H<:Number, T<:Number, R<:Real, P<:AbstractPropagator}
```

### Hubbard Model Hubbard-Stratonovich Transformations

```@docs
HubbardSpinHirschHST
HubbardSpinHirschHST()
HubbardSpinGaussHermiteHST
HubbardSpinGaussHermiteHST()
HubbardDensityHirschHST
HubbardDensityHirschHST()
HubbardDensityGaussHermiteHST
HubbardDensityGaussHermiteHST()
```

### Extended Hubbard Model Hubbard-Stratonovich Transformations

```@docs
ExtHubSpinHirschHST
ExtHubSpinHirschHST()
ExtHubDensityGaussHermiteHST
ExtHubDensityGaussHermiteHST()
init_renormalized_hubbard_parameters()
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
initialize!(::FermionPathIntegral{H,T}, ::FermionPathIntegral{H,T}, ::ElectronPhononParameters{T,R}) where {H<:Number, T<:Number, R<:AbstractFloat}
initialize!(::FermionPathIntegral{H,T}, ::ElectronPhononParameters{T,R}) where {H<:Number, T<:Number, R<:AbstractFloat}
update!(::FermionPathIntegral{H,T}, ::FermionPathIntegral{H,T}, ::ElectronPhononParameters{T,R}, ::Matrix{R}, ::Matrix{R}) where {H<:Number, T<:Number, R<:AbstractFloat}
update!(::FermionPathIntegral{H,T}, ::ElectronPhononParameters{T,R}, ::Matrix{R}, ::Matrix{R}) where {H<:Number, T<:Number, R<:AbstractFloat}
update!(::FermionPathIntegral{H,T}, ::ElectronPhononParameters{T,R}, ::Matrix{R}, ::Int) where {H<:Number, T<:Number, R<:AbstractFloat}
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