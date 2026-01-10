@doc raw"""
    model_summary(;
        simulation_info::SimulationInfo,
        β::T, Δτ::T, model_geometry::ModelGeometry,
        tight_binding_model::Union{TightBindingModel,Nothing} = nothing,
        tight_binding_model_up::Union{TightBindingModel,Nothing} = nothing,
        tight_binding_model_dn::Union{TightBindingModel,Nothing} = nothing,
        interactions::Union{Tuple,Nothing} = nothing
    ) where {T<:AbstractFloat}

Write model to summary to file. Note that either `tight_binding_model` or
`tight_binding_model_up` and `tight_binding_model_dn` need to be specified.
"""
function model_summary(;
    simulation_info::SimulationInfo,
    β::T, Δτ::T, model_geometry::ModelGeometry,
    tight_binding_model::Union{TightBindingModel,Nothing} = nothing,
    tight_binding_model_up::Union{TightBindingModel,Nothing} = nothing,
    tight_binding_model_dn::Union{TightBindingModel,Nothing} = nothing,
    interactions::Union{Tuple,Nothing} = nothing
) where {T<:AbstractFloat}

    # if process ID is 1
    if iszero(simulation_info.pID)

        # construct full filename, including filepath
        fn = joinpath(simulation_info.datafolder, "model_summary.toml")

        # get the length of the imaginary time axis
        Lτ = eval_length_imaginary_axis(β, Δτ)

        # open file to write to
        open(fn, "w") do fout
            # write β
            @printf fout "beta = %.6f\n\n" β
            # write Δτ
            @printf fout "dtau = %.6f\n\n" Δτ
            # write Lτ
            @printf fout "L_tau = %d\n\n" Lτ
            # write model geometry out to file
            show(fout, "text/plain", model_geometry)
            # write tight binding models to file
            if isnothing(tight_binding_model)
                @assert tight_binding_model_up.μ == tight_binding_model_dn.μ
                # write spin up tight-binding model to file
                show(fout, MIME("text/plain"), tight_binding_model_up, spin = +1)
                # write spin-down tight-binding model to file
                show(fout, MIME("text/plain"), tight_binding_model_dn, spin = -1)
            else
                # write tight-binding model to file assuming spin symmetry
                show(fout, MIME("text/plain"), tight_binding_model)
            end
            # write various interactions to file
            if !isnothing(interactions)
                for interaction in interactions
                    show(fout, "text/plain", interaction)
                end
            end
        end
    end

    return nothing
end