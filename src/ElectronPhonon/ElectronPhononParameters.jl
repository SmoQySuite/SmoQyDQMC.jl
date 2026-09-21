@doc raw"""
    ElectronPhononParameters{T<:Number, E<:AbstractFloat}

Describes all parameters in the electron-phonon model.

# Fields

- `β::E`: Inverse temperature.
- `Δτ::E`: Discretization in imaginary time.
- `Lτ::Int`: Length of imaginary time axis.
- `x::Matrix{E}`: Phonon fields, where each column represents the phonon fields for a given imaginary time slice.
- `phonon_parameters::PhononParameters{E,D}`: Refer to [`PhononParameters`](@ref).
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

    # initialize electron-phonon parameters
    electron_phonon_parameters = ElectronPhononParameters(
        β, Δτ, Lτ, x,
        phonon_parameters,
        dispersion_parameters,
        holstein_parameters_up, holstein_parameters_dn,
        ssh_parameters_up, ssh_parameters_dn
    )

    # initialize phonon fields
    initialize_phonon_fields!(
        electron_phonon_parameters, rng
    )
    
    return electron_phonon_parameters
end


@doc raw"""
    initialize!(
        fermion_path_integral_up::FermionPathIntegral{H,T},
        fermion_path_integral_dn::FermionPathIntegral{H,T},
        electron_phonon_parameters::ElectronPhononParameters{T,R}
    ) where {H<:Number, T<:Number, R<:AbstractFloat}

Initialize the contribution of an [`ElectronPhononParameters`](@ref) to a [`FermionPathIntegral`](@ref).
"""
function initialize!(
    fermion_path_integral_up::FermionPathIntegral{H,T},
    fermion_path_integral_dn::FermionPathIntegral{H,T},
    electron_phonon_parameters::ElectronPhononParameters{T,R}
) where {H<:Number, T<:Number, R<:AbstractFloat}

    # initialize spin up fermion path integral
    initialize!(fermion_path_integral_up, electron_phonon_parameters, spin = +1)

    # initialize spin down fermion path integral
    initialize!(fermion_path_integral_dn, electron_phonon_parameters, spin = -1)

    return nothing
end

@doc raw"""
    initialize!(
        # ARGUMENTS
        fermion_path_integral::FermionPathIntegral{H,T},
        electron_phonon_parameters::ElectronPhononParameters{T,R};
        # KEYWORD ARGUMENTS
        spin::Int = +1
    ) where {H<:Number, T<:Number, R<:AbstractFloat}

Initialize the contribution of an [`ElectronPhononParameters`](@ref) to a [`FermionPathIntegral`](@ref).
"""
function initialize!(
    # ARGUMENTS
    fermion_path_integral::FermionPathIntegral{H,T},
    electron_phonon_parameters::ElectronPhononParameters{T,R};
    # KEYWORD ARGUMENTS
    spin::Int = +1
) where {H<:Number, T<:Number, R<:AbstractFloat}

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

    # calculate bosonic action
    fermion_path_integral.Sb += bosonic_action(electron_phonon_parameters)

    return nothing
end


@doc raw"""
    update!(
        fermion_path_integral_up::FermionPathIntegral{H,T},
        fermion_path_integral_dn::FermionPathIntegral{H,T},
        electron_phonon_parameters::ElectronPhononParameters{T,R},
        x′::Matrix{R},
        x::Matrix{R}
    ) where {H<:Number, T<:Number, R<:AbstractFloat}

Update a [`FermionPathIntegral`](@ref) to reflect a change in the phonon configuration from `x` to `x′`.
"""
function update!(
    fermion_path_integral_up::FermionPathIntegral{H,T},
    fermion_path_integral_dn::FermionPathIntegral{H,T},
    electron_phonon_parameters::ElectronPhononParameters{T,R},
    x′::Matrix{R},
    x::Matrix{R}
) where {H<:Number, T<:Number, R<:AbstractFloat}

    # update spin up fermion path integral
    update!(fermion_path_integral_up, electron_phonon_parameters, x′, x, spin = +1)

    # update spin down fermion path integral
    update!(fermion_path_integral_dn, electron_phonon_parameters, x′, x, spin = -1)

    return nothing
end

@doc raw"""
    update!(
        # ARGUMENTS
        fermion_path_integral::FermionPathIntegral{H,T},
        electron_phonon_parameters::ElectronPhononParameters{T,R},
        x′::Matrix{R},
        x::Matrix{R};
        # KEYWORD ARGUMENTS
        spin::Int = +1
    ) where {H<:Number, T<:Number, R<:AbstractFloat}

Update a [`FermionPathIntegral`](@ref) to reflect a change in the phonon configuration from `x` to `x′`.
"""
function update!(
    # ARGUMENTS
    fermion_path_integral::FermionPathIntegral{H,T},
    electron_phonon_parameters::ElectronPhononParameters{T,R},
    x′::Matrix{R},
    x::Matrix{R};
    # KEYWORD ARGUMENTS
    spin::Int = +1
) where {H<:Number, T<:Number, R<:AbstractFloat}

    if isone(spin)
        holstein_parameters = electron_phonon_parameters.holstein_parameters_up
        ssh_parameters = electron_phonon_parameters.ssh_parameters_up
    else
        holstein_parameters = electron_phonon_parameters.holstein_parameters_dn
        ssh_parameters = electron_phonon_parameters.ssh_parameters_dn
    end

    # update fermion path integral based on holstein interaction and new phonon configuration
    update!(fermion_path_integral, holstein_parameters, x, -1)
    update!(fermion_path_integral, holstein_parameters, x′, +1)

    # update fermion path integral based on ssh interaction and new phonon configuration
    update!(fermion_path_integral, ssh_parameters, x, -1)
    update!(fermion_path_integral, ssh_parameters, x′, +1)

    return nothing
end

@doc raw"""
    update!(
        # ARGUMENTS
        fermion_path_integral::FermionPathIntegral{H,T},
        electron_phonon_parameters::ElectronPhononParameters{T,R},
        x::Matrix{R},
        sgn::Int;
        # KEYWORD ARGUMENTS
        spin::Int = +1
    ) where {H<:Number, T<:Number, R<:AbstractFloat}

Update a [`FermionPathIntegral`](@ref) according to `sgn * x`.
"""
function update!(
    # ARGUMENTS
    fermion_path_integral::FermionPathIntegral{H,T},
    electron_phonon_parameters::ElectronPhononParameters{T,R},
    x::Matrix{R},
    sgn::Int;
    # KEYWORD ARGUMENTS
    spin::Int = +1
) where {H<:Number, T<:Number, R<:AbstractFloat}

    if isone(spin)
        holstein_parameters = electron_phonon_parameters.holstein_parameters_up
        ssh_parameters = electron_phonon_parameters.ssh_parameters_up
    else
        holstein_parameters = electron_phonon_parameters.holstein_parameters_dn
        ssh_parameters = electron_phonon_parameters.ssh_parameters_dn
    end

    # update fermion path integral based on holstein interaction and new phonon configuration
    update!(fermion_path_integral, holstein_parameters, x, sgn)

    # update fermion path integral based on ssh interaction and new phonon configuration
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

    if notfinite(M)
        M″ = M′
    elseif notfinite(M′)
        M″ = M
    else
        M″ = (M*M′)/(M+M′)
    end

    return M″
end

# initialize phonon fields
function initialize_phonon_fields!(
    electron_phonon_parameters::ElectronPhononParameters{T,R},
    rng::AbstractRNG
) where {T<:Number, R<:AbstractFloat}

    (; x, β, phonon_parameters) = electron_phonon_parameters
    (; Ω, M) = phonon_parameters

    # iterate over phonon modes in lattice
    for n in eachindex(Ω)
        # get the phonon fields associated with the phonon mode
        x_n = @view x[n,:]
        # get frequency associated with phonon mode.
        # if zero frequency then it defaults to unity.
        Ω_n = iszero(Ω[n]) ? one(R) : Ω[n]
        # get mass associated with phonon mode
        M_n = M[n]
        # if finite phonon mass
        if isfinite(M_n)
            # directly sample equilibrium quantum harmonic oscillator path
            sample_qho_path!(x_n, Ω_n, M_n, β, rng)
        # if a "frozen" phonon mode
        else
            # set frozen phonon mode fields to zero
            fill!(x_n, zero(R))
        end

    end

    return nothing
end

# sample equilibrium paths of quantum harmonic oscillator
function sample_qho_path!(
    x::AbstractVector{R},
    Ω::R, M::R, β::R,
    rng::AbstractRNG
) where {R<:AbstractFloat}

    # calculate length of imaginary-time axis
    Lτ = length(x)

    # calculate imaginary-time discretization
    Δτ = β/Lτ

    # sample the starting point from the exact diagonal density matrix
    # ρ(x,x;β) ∝ exp[-M Ω tanh(βΩ/2) x²]  ⇒  σ² = 1 / (2 M Ω tanh(βΩ/2))
    σ_0 = one(R) / sqrt(2 * M * Ω * tanh(β * Ω / 2))
    x[1] = σ_0 * randn(rng, R)
    x_end = x[1]  # periodic boundary condition: path must return to x[1]

    # precompute quantities for the single-step propagator
    coth_Δτ = one(R) / tanh(Ω * Δτ)
    csch_Δτ = one(R) / sinh(Ω * Δτ)

    # Levy construction: sample x[k] conditioned on x[k-1] and the endpoint x_end,
    # which is reached after the remaining imaginary time τ′ = (Lτ - k + 1)Δτ
    for k in 2:Lτ
        τ′ = (Lτ - k + 1) * Δτ
        γ_1 = coth_Δτ + one(R) / tanh(Ω * τ′)
        γ_2 = x[k-1] * csch_Δτ + x_end / sinh(Ω * τ′)
        μ = γ_2 / γ_1
        σ = one(R) / sqrt(M * Ω * γ_1)
        x[k] = μ + σ * randn(rng, R)
    end

    return nothing
end