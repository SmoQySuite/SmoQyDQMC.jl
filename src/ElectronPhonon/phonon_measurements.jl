#######################################
## PHONON KINETIC ENERGY MEASUREMENT ##
#######################################

@doc raw"""
    measure_phonon_kinetic_energy(electron_phonon_parameters::ElectronPhononParameters{T,E},
                                  n::Int) where {T<:Number, E<:AbstractFloat}

Evaluate the average phonon kinetic energy for phonon mode `n`.
The measurement is made using the expression
```math
\langle K \rangle = \frac{1}{2\Delta\tau} - \frac{M}{2}\bigg\langle\frac{(x_{l+1}-x_{l})^{2}}{\Delta\tau^{2}}\bigg\rangle. 
```
"""
function measure_phonon_kinetic_energy(electron_phonon_parameters::ElectronPhononParameters{T,E},
                                       n::Int) where {T<:Number, E<:AbstractFloat}

    (; x, Δτ) = electron_phonon_parameters
    phonon_parameters = electron_phonon_parameters.phonon_parameters::PhononParameters{E}

    # calculate phonon kinetic energy
    K = measure_phonon_kinetic_energy(phonon_parameters, x, Δτ, n)

    return K
end

function measure_phonon_kinetic_energy(phonon_parameters::PhononParameters{T},
                                       x::Matrix{T}, Δτ::T, n::Int) where {T<:AbstractFloat}

    (; M, nphonon, Nphonon) = phonon_parameters

    # initialize phonon kinetic energy to zero
    K = zero(T)

    # length of imaginary time axis
    Lτ = size(x,2)

    # number of unit cells in lattice
    Nunitcells = Nphonon ÷ nphonon

    # reshape phonon field
    x′ = reshape(x, (Nunitcells, nphonon, Lτ))
    M′ = reshape(M, (Nunitcells, nphonon))

    # only non-zero kinetic energy if finite mass
    if isfinite(M′[1,n])
        # calculate (M/2)⋅⟨(x[l+1]-x[l])²/Δτ²⟩
        # iterate over imaginary time slice
        @fastmath @inbounds for l in axes(x′, 3)
            # iterate over unit cells
            for u in axes(x′, 1)
                # calculate K = 1/(2Δτ) - (M/2)⋅⟨(x[l+1]-x[l])²/Δτ²⟩
                K += 1/(2*Δτ) - M′[u,n]/2 * (x′[u,n,mod1(l+1,Lτ)] - x′[u,n,l])^2 / Δτ^2
            end
        end

        # normalize the kinetic energy
        K = K / (Nunitcells * Lτ)
    end

    return K
end

#########################################
## PHONON POTENTIAL ENERGY MEASUREMENT ##
#########################################

@doc raw"""
    measure_phonon_potential_energy(electron_phonon_parameters::ElectronPhononParameters{T,E},
                                    n::Int) where {T<:Number, E<:AbstractFloat}

Calculate the average phonon potential energy, given by
```math
U = \frac{1}{2} M \Omega^2 \langle \hat{X}^2 \rangle + \frac{1}{24} M \Omega_4^2 \langle \hat{X}^4 \rangle,
```
for phonon mode `n` in the unit cell.
"""
function measure_phonon_potential_energy(electron_phonon_parameters::ElectronPhononParameters{T,E},
                                         n::Int) where {T<:Number, E<:AbstractFloat}

    phonon_parameters = electron_phonon_parameters.phonon_parameters::PhononParameters{E}
    (; x) = electron_phonon_parameters
    (; Ω, Ω4, M, nphonon) = phonon_parameters

    U = measure_phonon_potential_energy(x, M, Ω, Ω4, nphonon, n)

    return U
end

function measure_phonon_potential_energy(x::Matrix{T}, M::Vector{T}, Ω::Vector{T}, Ω4::Vector{T},
                                         nphonon::Int, n::Int) where {T<:AbstractFloat}

    # length of imaginary time axis
    Lτ = size(x, 2)

    # total number of phonon modes in lattice
    Nphonon = size(x, 1)

    # number of unit cells in lattice
    Nunitcell = Nphonon ÷ nphonon

    # initialize phonon potential energy to zero
    U = zero(T)

    # reshape arrays
    x′  = reshape(x,  (Nunitcell, nphonon, Lτ))
    M′  = reshape(M,  (Nunitcell, nphonon))
    Ω′  = reshape(Ω,  (Nunitcell, nphonon))
    Ω4′ = reshape(Ω4, (Nunitcell, nphonon))

    # make sure phonon mass is finite
    if isfinite(M′[1,n])
        # iterate over imaginary time
        @fastmath @inbounds for l in axes(x′,3)
            # iterate over unit cells
            for u in axes(x′,1)
                # calcualte phonon potential energy
                U += M′[u,n]*Ω′[u,n]^2*x′[u,n,l]^2/2 + M′[u,n]*Ω4′[u,n]^2*x′[u,n,l]^4/24
            end
        end
        # normalize phonon potential energy measurement
        U /= (Nunitcell * Lτ)
    end

    return U
end

########################################
## MEASURE MOMENTS OF PHONON POSITION ##
########################################

@doc raw"""
    measure_phonon_position_moment(electron_phonon_parameters::ElectronPhononParameters{T,E},
                                   n::Int, m::Int) where {T<:Number, E<:AbstractFloat}

Measure ``\langle X^m \rangle`` for phonon mode `n` in the unit cell.
"""
function measure_phonon_position_moment(electron_phonon_parameters::ElectronPhononParameters{T,E},
                                        n::Int, m::Int) where {T<:Number, E<:AbstractFloat}

    phonon_parameters = electron_phonon_parameters.phonon_parameters::PhononParameters{E}
    nphonon = phonon_parameters.nphonon::Int
    x = electron_phonon_parameters.x::Matrix{E}
    xm = measure_phonon_position_moment(x, nphonon, n, m)

    return xm
end

function measure_phonon_position_moment(x::Matrix{T}, nphonon::Int, n::Int, m::Int) where {T<:AbstractFloat}

    # length of imaginary time axis
    Lτ = size(x,2)

    # total number of phonon modes in lattice
    Nphonon = size(x,1)

    # number of unit cells in lattice
    Nunitcell = Nphonon ÷ nphonon

    # reshape phonon fields
    x′ = reshape(x, (Nunitcell, nphonon, Lτ))

    # phonon moment
    xm = zero(T)

    # iterate over imaginary time slices
    for l in axes(x′,3)
        # iterate over unit cells
        for u in axes(x′,1)
            xm += x′[u,n,l]^m
        end
    end

    # normalize measurement
    xm /= (Nunitcell * Lτ)

    return xm
end