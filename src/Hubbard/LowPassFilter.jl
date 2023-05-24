@doc raw"""
    LowPassFilter{T<:AbstractFloat, PFFT, PIFFT}

Type implementing low-pass filter mass matrix that sets all signals above cutoff
frequency `ω_cutoff` to zero. In the case that `ω_cutoff = ω_max`, then
the low-pass filter mass matrix reduces to the identity matrix. If `Lτ` is the length
of the imaginary time axis then `ω_max = Lτ ÷ 2` and `0 <= ω_cutoff <= ω_max`.

# Fields

- `ω_max::Int`: Maximum frequency.
- `ω_cutoff::Int`: Cutoff frequency.
- `v::Vector{Complex{T}}`: Vector containing data to be filtered.
- `ṽ::Vector{Complex{T}}`: Fourier transform of `v`.
- `pfft::PFFT`: Forward FFT plan to perform transformation from imaginary time to frequency space without allocations.
- `pifft::PIFFT`: Inverse FFT plan to perform transformation from frequency to imaginary time space without allocations.
"""
struct LowPassFilter{T<:AbstractFloat, PFFT, PIFFT}

    # maximum frequency
    ω_max::Int

    # cutoff frequency i.e. all higher frequency contributions set to zero
    ω_cutoff::Int

    # array to contain data to be filtered
    v::Vector{Complex{T}}

    # fourier transform of data to be filtered
    ṽ::Vector{Complex{T}}

    # forward FFT plan
    pfft::PFFT

    # reverse FFT plan
    pifft::PIFFT
end

@doc raw"""
    LowPassFilter(Lτ::Int, ω_cutoff::Int, T::DataType=Float64)

Initialize and return an instance of [`LowPassFilter`](@ref).
Note that `ω_max = Lτ ÷ 2`, and `0 < ω_cutoff <= ω_max`.
"""
function LowPassFilter(Lτ::Int, ω_cutoff::Int, T::DataType=Float64)

    ω_max = Lτ ÷ 2
    @assert 0 <= ω_cutoff <= ω_max
    v = zeros(Complex{T}, Lτ)
    ṽ = zeros(Complex{T}, Lτ)
    pfft = plan_fft(v, flags=FFTW.PATIENT)
    pifft = plan_ifft(v, flags=FFTW.PATIENT)

    return LowPassFilter(ω_max, ω_cutoff, v, ṽ, pfft, pifft)
end


function apply_filter!(u::AbstractArray, I::UniformScaling, ignore...)

    return nothing
end

function apply_filter!(u::AbstractVector{T}, lpf::LowPassFilter{T}) where {T<:AbstractFloat}

    @assert length(u) == length(lpf.v) "length(u) = $(length(u)), length(lpf.v) = $(length(lpf.v))"
    copyto!(lpf.v, u)
    _filter!(lpf)
    @. u = real(lpf.v)

    return nothing
end

function apply_filter!(u::AbstractVector{Complex{T}}, lpf::LowPassFilter{T}) where {T<:AbstractFloat}

    @assert length(u) == lpf.ω_max "length(u) = $(size(u)), lpf.ω_max = $(lpf.ω_max)"
    copyto!(lpf.v, u)
    _filter!(lpf)
    copyto!(u, lpf.v)

    return nothing
end

function apply_filter!(U::AbstractMatrix, lpf::LowPassFilter; dim::Int = 1)

    for u in eachslice(U, dims = dim)
        apply_filter!(u, lpf)
    end

    return nothing
end

# apply low pass filter to v
function _filter!(lpf::LowPassFilter{T}) where {T<:AbstractFloat}

    (; ω_cutoff, ω_max, v, ṽ, pfft, pifft) = lpf
    mul!(ṽ, pfft, v)
    Lτ = length(v)
    @inbounds for ω in (ω_cutoff+1):ω_max
        ω′     = Lτ - ω
        ṽ[ω+1] = zero(T)
        ṽ[ω′+1]= zero(T)
    end
    mul!(v, pifft, ṽ)

    return nothing
end