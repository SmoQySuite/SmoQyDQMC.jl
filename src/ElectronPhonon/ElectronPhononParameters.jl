@doc raw"""
    ElectronPhononParameters{T<:Number, E<:AbstractFloat}

Describes all parameters in the electron-phonon model.

# Fields

- `β::E`: Inverse temperature.
- `Δτ::E`: Discretization in imaginary time.
- `Lτ::Int`: Length of imaginary time axis.
- `x::Matrix{E}`: Phonon fields, where each column represents the phonon fields for a given imaginary time slice.
- `phonon_parameters::PhononParameters{E}`: Refer to [`PhononParameters`](@ref).
- `holstein_parameters_up::HolsteinParameters{E}`: Spin up [`HolsteinParameters`](@ref).
- `holstein_parameters_dn::HolsteinParameters{E}`: Spin down [`HolsteinParameters`](@ref).
- `ssh_parameters_up::SSHParameters{T}`: Spin up [`SSHParameters`](@ref).
- `ssh_parameters_dn::SSHParameters{T}`: Spin down [`SSHParameters`](@ref).
- `dispersion_parameters::DispersionParameters{E}`: Refer to [`DispersionParameters`](@ref).
"""
struct ElectronPhononParameters{T<:Number, E<:AbstractFloat}

    # inverse temperature
    β::E

    # discretization in imaginary time
    Δτ::E

    # length of imaginary time axis
    Lτ::Int

    # phonon fields
    x::Matrix{E}

    # all the phonon parameters
    phonon_parameters::PhononParameters{E}

    # all the phonon dispersion parameters
    dispersion_parameters::DispersionParameters{E}
    
    # all the spin-up holstein coupling parameters
    holstein_parameters_up::HolsteinParameters{E}

    # all the spin-down holstein coupling parameters
    holstein_parameters_dn::HolsteinParameters{E}

    # all the spin-up ssh coupling parameters
    ssh_parameters_up::SSHParameters{T}

    # all the spin-down ssh coupling parameters
    ssh_parameters_dn::SSHParameters{T}
end

@doc raw"""
    ElectronPhononParameters(;
        β::E, Δτ::E,
        model_geometry::ModelGeometry{D,E},
        tight_binding_parameters::Union{TightBindingParameters{T,E}, Nothing} = nothing,
        tight_binding_parameters_up::Union{TightBindingParameters{T,E}, Nothing} = nothing,
        tight_binding_parameters_dn::Union{TightBindingParameters{T,E}, Nothing} = nothing,
        electron_phonon_model::ElectronPhononModel{T,E,D},
        rng::AbstractRNG
    ) where {T,E,D}

Initialize and return an instance of [`ElectronPhononParameters`](@ref).
"""
function ElectronPhononParameters(;
    β::E, Δτ::E,
    model_geometry::ModelGeometry{D,E},
    tight_binding_parameters::Union{TightBindingParameters{T,E}, Nothing} = nothing,
    tight_binding_parameters_up::Union{TightBindingParameters{T,E}, Nothing} = nothing,
    tight_binding_parameters_dn::Union{TightBindingParameters{T,E}, Nothing} = nothing,
    electron_phonon_model::ElectronPhononModel{T,E,D},
    rng::AbstractRNG
) where {T,E,D}

    # specify spin-up and spin-down tight binding parameters if need
    if !isnothing(tight_binding_parameters)

        tight_binding_parameters_up = tight_binding_parameters
        tight_binding_parameters_dn = tight_binding_parameters
    end

    # initialize phonon parameters
    phonon_parameters = PhononParameters(model_geometry = model_geometry,
                                         electron_phonon_model = electron_phonon_model,
                                         rng = rng)

    # initialize phonon dispersion parameters
    dispersion_parameters = DispersionParameters(
        model_geometry = model_geometry,
        electron_phonon_model = electron_phonon_model,
        phonon_parameters = phonon_parameters,
        rng = rng
    )

    # initialize spin-down holstein parameters
    holstein_parameters_up, holstein_parameters_dn = HolsteinParameters(
        model_geometry = model_geometry,
        electron_phonon_model = electron_phonon_model,
        rng = rng
    )

    # initialize spin-up ssh parameters
    ssh_parameters_up, ssh_parameters_dn = SSHParameters(
        model_geometry = model_geometry,
        electron_phonon_model = electron_phonon_model,
        tight_binding_parameters_up = tight_binding_parameters_up,
        tight_binding_parameters_dn = tight_binding_parameters_dn,
        rng = rng
    )

    # evaluate length of imaginary time axis
    Lτ = eval_length_imaginary_axis(β, Δτ)

    # get relevant phonon parameters
    (; Nphonon, M, Ω) = phonon_parameters

    # allocate phonon fields
    x = zeros(E, Nphonon, Lτ)

    # iterate over phonons
    for phonon in 1:Nphonon
        # if finite phonon mass
        if isfinite(M[phonon])
            # get the phonon fields
            x_p = @view x[phonon,:]
            # initialize phonon field
            if iszero(Ω[phonon])
                # uncertainty in phonon position
                Δx = std_x_qho(β, 1.0, M[phonon])
                # assign initial phonon position
                x0 = Δx * randn(rng)
                @. x_p = x0
            else
                # uncertainty in phonon position
                Δx = std_x_qho(β, Ω[phonon], M[phonon])
                # assign initial phonon position
                x0 = Δx * randn(rng)
                @. x_p = x0
            end
        end
    end

    # initialize electron-phonon parameters
    electron_phonon_parameters = ElectronPhononParameters(
        β, Δτ, Lτ, x,
        phonon_parameters,
        dispersion_parameters,
        holstein_parameters_up, holstein_parameters_dn,
        ssh_parameters_up, ssh_parameters_dn
    )
    
    return electron_phonon_parameters
end


@doc raw"""
    initialize!(
        fermion_path_integral_up::FermionPathIntegral{T,E},
        fermion_path_integral_dn::FermionPathIntegral{T,E},
        electron_phonon_parameters::ElectronPhononParameters{T,E}
    ) where {T,E}

Initialize the contribution of an [`ElectronPhononParameters`](@ref) to a [`FermionPathIntegral`](@ref).
"""
function initialize!(
    fermion_path_integral_up::FermionPathIntegral{T,E},
    fermion_path_integral_dn::FermionPathIntegral{T,E},
    electron_phonon_parameters::ElectronPhononParameters{T,E}
) where {T,E}

    # initialize spin up fermion path integral
    initialize!(fermion_path_integral_up, electron_phonon_parameters, spin = +1)

    # initialize spin down fermion path integral
    initialize!(fermion_path_integral_dn, electron_phonon_parameters, spin = -1)

    return nothing
end

@doc raw"""
    initialize!(
        fermion_path_integral::FermionPathIntegral{T,E},
        electron_phonon_parameters::ElectronPhononParameters{T,E};
        spin::Int = +1,
    ) where {T,E}

Initialize the contribution of an [`ElectronPhononParameters`](@ref) to a [`FermionPathIntegral`](@ref).
"""
function initialize!(
    fermion_path_integral::FermionPathIntegral{T,E},
    electron_phonon_parameters::ElectronPhononParameters{T,E};
    spin::Int = +1
) where {T,E}

    x = electron_phonon_parameters.x
    if isone(spin)
        holstein_parameters = electron_phonon_parameters.holstein_parameters_up
        ssh_parameters = electron_phonon_parameters.ssh_parameters_up
    else
        holstein_parameters = electron_phonon_parameters.holstein_parameters_dn
        ssh_parameters = electron_phonon_parameters.ssh_parameters_dn
    end

    # update fermion path integral based on holstein interaction
    update!(fermion_path_integral, holstein_parameters, x, 1)

    # update fermion path integral based on ssh interaction
    update!(fermion_path_integral, ssh_parameters, x, 1)

    return nothing
end


@doc raw"""
    update!(
        fermion_path_integral_up::FermionPathIntegral{T,E},
        fermion_path_integral_dn::FermionPathIntegral{T,E},
        electron_phonon_parameters::ElectronPhononParameters{T,E},
        x′::Matrix{E},
        x::Matrix{E}
    ) where {T,E}

Update a [`FermionPathIntegral`](@ref) to reflect a change in the phonon configuration from `x` to `x′`.
"""
function update!(
    fermion_path_integral_up::FermionPathIntegral{T,E},
    fermion_path_integral_dn::FermionPathIntegral{T,E},
    electron_phonon_parameters::ElectronPhononParameters{T,E},
    x′::Matrix{E},
    x::Matrix{E}
) where {T,E}

    # update spin up fermion path integral
    update!(fermion_path_integral_up, electron_phonon_parameters, x′, x, spin = +1)

    # update spin down fermion path integral
    update!(fermion_path_integral_dn, electron_phonon_parameters, x′, x, spin = -1)

    return nothing
end

@doc raw"""
    update!(
        fermion_path_integral::FermionPathIntegral{T,E},
        electron_phonon_parameters::ElectronPhononParameters{T,E},
        x′::Matrix{E},
        x::Matrix{E};
        spin::Int = +1
    ) where {T,E}

Update a [`FermionPathIntegral`](@ref) to reflect a change in the phonon configuration from `x` to `x′`.
"""
function update!(
    fermion_path_integral::FermionPathIntegral{T,E},
    electron_phonon_parameters::ElectronPhononParameters{T,E},
    x′::Matrix{E},
    x::Matrix{E};
    spin::Int = +1
) where {T,E}

    if isone(spin)
        holstein_parameters = electron_phonon_parameters.holstein_parameters_up
        ssh_parameters = electron_phonon_parameters.ssh_parameters_up
    else
        holstein_parameters = electron_phonon_parameters.holstein_parameters_dn
        ssh_parameters = electron_phonon_parameters.ssh_parameters_dn
    end

    # update fermion path integral based on holstein interaction and new phonon configration
    update!(fermion_path_integral, holstein_parameters, x, -1)
    update!(fermion_path_integral, holstein_parameters, x′, +1)

    # update fermion path integral based on ssh interaction and new phonon configration
    update!(fermion_path_integral, ssh_parameters, x, -1)
    update!(fermion_path_integral, ssh_parameters, x′, +1)

    return nothing
end

@doc raw"""
    update!(fermion_path_integral::FermionPathIntegral{T,E},
        electron_phonon_parameters::ElectronPhononParameters{T,E},
        x::Matrix{E},
        sgn::Int;
        spin::Int = +1
    ) where {T,E}

Update a [`FermionPathIntegral`](@ref) according to `sgn * x`.
"""
function update!(fermion_path_integral::FermionPathIntegral{T,E},
    electron_phonon_parameters::ElectronPhononParameters{T,E},
    x::Matrix{E},
    sgn::Int;
    spin::Int = +1
) where {T,E}

    if isone(spin)
        holstein_parameters = electron_phonon_parameters.holstein_parameters_up
        ssh_parameters = electron_phonon_parameters.ssh_parameters_up
    else
        holstein_parameters = electron_phonon_parameters.holstein_parameters_dn
        ssh_parameters = electron_phonon_parameters.ssh_parameters_dn
    end

    # update fermion path integral based on holstein interaction and new phonon configration
    update!(fermion_path_integral, holstein_parameters, x, sgn)

    # update fermion path integral based on ssh interaction and new phonon configration
    update!(fermion_path_integral, ssh_parameters, x, sgn)

    return nothing
end


# Given a quantum harmonic oscillator with frequency Ω and mass M at an
# inverse temperature of β, return the standard deviation of the equilibrium
# distribution for the phonon position.
function std_x_qho(β::T, Ω::T, M::T) where {T<:AbstractFloat}

    ΔX = inv(sqrt(2 * M * Ω * tanh(β*Ω/2)))
    return ΔX
end


# Calculate the reduced mass given the mass of two phonons `M` and `M′`.
function reduced_mass(M::T, M′::T) where {T<:AbstractFloat}

    if !isfinite(M)
        M″ = M′
    elseif !isfinite(M′)
        M″ = M
    else
        M″ = (M*M′)/(M+M′)
    end

    return M″
end