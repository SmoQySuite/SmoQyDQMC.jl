# API

## Simulation Information Type and Methods

- [`SimulationInfo`](@ref)
- [`initialize_datafolder`](@ref)
- [`model_summary`](@ref)

```@docs
SimulationInfo
SimulationInfo(;)
initialize_datafolder
model_summary
```

## Model Geometry Type and Methods

- [`ModelGeometry`](@ref)
- [`add_bond!`](@ref)
- [`get_bond_id`](@ref)

```@docs
ModelGeometry
ModelGeometry(::UnitCell{D}, ::Lattice{D}) where {D}
add_bond!
get_bond_id
```

## Fermion Path Integral Type and Methods

- [`FermionPathIntegral`](@ref)
- [`initialize_propagators`](@ref)
- [`calculate_propagators!`](@ref)
- [`calculate_propagator!`](@ref)

```@docs
FermionPathIntegral
FermionPathIntegral(;)
initialize_propagators
calculate_propagators!
calculate_propagator!
```

## Update Numerical Stabilization Frequency

- [`update_stabalization_frequency!`](@ref)

```@docs
update_stabalization_frequency!
```

## Tight-Binding Model

- [`TightBindingModel`](@ref)
- [`TightBindingParameters`](@ref)
- [`measure_onsite_energy`](@ref)
- [`measure_hopping_energy`](@ref)
- [`measure_bare_hopping_energy`](@ref)
- [`measure_hopping_amplitude`](@ref)
- [`measure_hopping_inversion`](@ref)
- [`measure_hopping_inversion_avg`](@ref)

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

- [`HubbardModel`](@ref)
- [`HubbardParameters`](@ref)
- [`initialize!`](@ref)

**Hubbard Model Measurements**

- [`measure_hubbard_energy`](@ref)

**Hubbard Ising Hubbard-Stratonovich Transformation Types and Methods**

- [`HubbardIsingHSParameters`](@ref)
- [`initialize!`](@ref)
- [`local_updates!`](@ref)
- [`reflection_update!`](@ref)
- [`swap_update!`](@ref)

```@docs
HubbardModel
HubbardModel(;)
HubbardParameters
HubbardParameters(;)
initialize!(::FermionPathIntegral{T,E}, ::FermionPathIntegral{T,E}, ::HubbardParameters{E}) where {T,E}
```

### Hubbard Model Measurements

```@docs
measure_hubbard_energy
```

### Hubbard Ising Hubbard-Stratonovich Transformation Types and Methods

```@docs
HubbardIsingHSParameters
HubbardIsingHSParameters(;)
initialize!(::FermionPathIntegral{T,E}, ::FermionPathIntegral{T,E}, ::HubbardIsingHSParameters{E}) where {T,E}
initialize!(::FermionPathIntegral{T,E}, ::HubbardIsingHSParameters{E}) where {T,E}
local_updates!(::Matrix{T}, ::E, ::T, ::Matrix{T}, ::E, ::T, ::HubbardIsingHSParameters{E})  where {T<:Number, E<:AbstractFloat, P<:AbstractPropagator{T,E}}
local_updates!(::Matrix{T}, ::E, ::T, ::HubbardIsingHSParameters{E})  where {T<:Number, E<:AbstractFloat, P<:AbstractPropagator{T,E}}
reflection_update!(::Matrix{T}, ::E, ::T, ::Matrix{T}, ::E, ::T, ::HubbardIsingHSParameters{E}) where {T<:Number, E<:AbstractFloat, P<:AbstractPropagator{T,E}}
reflection_update!(::Matrix{T}, ::E, ::T, ::HubbardIsingHSParameters{E}) where {T<:Number, E<:AbstractFloat, P<:AbstractPropagator{T,E}}
swap_update!(::Matrix{T}, ::E, ::T, ::Matrix{T}, ::E, ::T, ::HubbardIsingHSParameters{E}) where {T<:Number, E<:AbstractFloat, P<:AbstractPropagator{T,E}}
swap_update!(::Matrix{T}, ::E, ::T, ::HubbardIsingHSParameters{E}) where {T<:Number, E<:AbstractFloat, P<:AbstractPropagator{T,E}}
```

## Electron-Phonon Model

**Electron-Phonon Model Types and Method**

- [`ElectronPhononModel`](@ref)
- [`PhononMode`](@ref)
- [`HolsteinCoupling`](@ref)
- [`SSHCoupling`](@ref)
- [`PhononDispersion`](@ref)
- [`add_phonon_mode!`](@ref)
- [`add_holstein_coupling!`](@ref)
- [`add_ssh_coupling!`](@ref)
- [`add_phonon_dispersion!`](@ref)

**Electron-Phonon Parameter Types and Methods**

- [`ElectronPhononParameters`](@ref)
- [`SmoQyDQMC.PhononParameters`](@ref)
- [`SmoQyDQMC.HolsteinParameters`](@ref)
- [`SmoQyDQMC.SSHParameters`](@ref)
- [`SmoQyDQMC.DispersionParameters`](@ref)
- [`initialize!`](@ref)
- [`update!`](@ref)

**Electron-Phonon Measurements**

- [`measure_phonon_kinetic_energy`](@ref)
- [`measure_phonon_potential_energy`](@ref)
- [`measure_phonon_position_moment`](@ref)
- [`measure_holstein_energy`](@ref)
- [`measure_ssh_energy`](@ref)
- [`measure_dispersion_energy`](@ref)

**Electron-Phonon Updates**

- [`HMCUpdater`](@ref)
- [`EFAHMCUpdater`](@ref)
- [`hmc_update!`](@ref)
- [`SmoQyDQMC.FourierMassMatrix`](@ref)
- [`reflection_update!`](@ref)
- [`swap_update!`](@ref)
- [`radial_update!`](@ref)

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
HMCUpdater
HMCUpdater(;)
EFAHMCUpdater
EFAHMCUpdater(;)
hmc_update!(::Matrix{T}, ::E, ::T, ::Matrix{T}, ::E, ::T, ::ElectronPhononParameters{T,E}, ::HMCUpdater{T,E}) where {T<:Number, E<:AbstractFloat, P<:AbstractPropagator{T,E}}
hmc_update!(::Matrix{T}, ::E, ::T, ::ElectronPhononParameters{T,E}, ::HMCUpdater{T,E}) where {T<:Number, E<:AbstractFloat, P<:AbstractPropagator{T,E}}
hmc_update!(::Matrix{T}, ::E, ::T, ::Matrix{T}, ::E, ::T, ::ElectronPhononParameters{T,E}, ::EFAHMCUpdater{T,E}) where {T, E, P<:AbstractPropagator{T,E}}
hmc_update!(::Matrix{T}, ::E, ::T, ::ElectronPhononParameters{T,E}, ::EFAHMCUpdater{T,E}) where {T, E, P<:AbstractPropagator{T,E}}
SmoQyDQMC.FourierMassMatrix
reflection_update!(::Matrix{T}, ::E, ::T, ::Matrix{T}, ::E, ::T, ::ElectronPhononParameters{T,E}) where {T<:Number, E<:AbstractFloat, P<:AbstractPropagator{T,E}}
reflection_update!(::Matrix{T}, ::E, ::T, ::ElectronPhononParameters{T,E}) where {T<:Number, E<:AbstractFloat, P<:AbstractPropagator{T,E}}
swap_update!(::Matrix{T}, ::E, ::T, ::Matrix{T}, ::E, ::T, ::ElectronPhononParameters{T,E}) where {T<:Number, E<:AbstractFloat, P<:AbstractPropagator{T,E}}
swap_update!(::Matrix{T}, ::E, ::T, ::ElectronPhononParameters{T,E}) where {T<:Number, E<:AbstractFloat, P<:AbstractPropagator{T,E}}
radial_update!(::Matrix{T}, ::E, ::T, ::Matrix{T}, ::E, ::T, ::ElectronPhononParameters{T,E}) where {T<:Number, E<:AbstractFloat, P<:AbstractPropagator{T,E}}
radial_update!(::Matrix{T}, ::E, ::T, ::ElectronPhononParameters{T,E}) where {T<:Number, E<:AbstractFloat, P<:AbstractPropagator{T,E}}
```

## Density and Chemical Potential Tuning

- [`update_chemical_potential!`](@ref)
- [`save_density_tuning_profile`](@ref)

```@docs
update_chemical_potential!
save_density_tuning_profile
```

## Measurement Methods

- [`GLOBAL_MEASUREMENTS`](@ref)
- [`LOCAL_MEASUREMENTS`](@ref)
- [`CORRELATION_FUNCTIONS`](@ref)

**Initialize Measurements**

- [`initialize_measurement_container`](@ref)
- [`initialize_measurements!`](@ref)
- [`initialize_correlation_measurements!`](@ref)
- [`initialize_composite_correlation_measurement!`](@ref)

**Make Measurements**

- [`make_measurements!`](@ref)

**Write Measurements**

- [`write_measurements!`](@ref)
- [`merge_bins`](@ref)
- [`rm_bins`](@ref)

**Checkpointing Utilities**

- [`write_jld2_checkpoint`](@ref)
- [`read_jld2_checkpoint`](@ref)
- [`rm_jld2_checkpoints`](@ref)
- [`rename_complete_simulation`](@ref)

**Process Measurements**

- [`save_simulation_info`](@ref)
- [`process_measurements`](@ref)
- [`compute_correlation_ratio`](@ref)
- [`compute_composite_correlation_ratio`](@ref)

**Export Measurements**

- [`export_global_stats_to_csv`](@ref)
- [`export_global_bins_to_h5`](@ref)
- [`export_global_bins_to_csv`](@ref)

- [`export_local_stats_to_csv`](@ref)
- [`export_local_bins_to_csv`](@ref)
- [`export_local_bins_to_h5`](@ref)

- [`export_correlation_stats_to_csv`](@ref)
- [`export_correlation_bins_to_csv`](@ref)
- [`export_correlation_bins_to_h5`](@ref)

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

### Make Measreuments

```@docs
make_measurements!
```

### Write Measreuments

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