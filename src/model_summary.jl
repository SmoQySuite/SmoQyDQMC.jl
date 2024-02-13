@doc raw"""
    model_summary(;
        simulation_info::SimulationInfo,
        β::T, Δτ::T, model_geometry::ModelGeometry,
        tight_binding_model::TightBindingModel,
        tight_binding_model_dn::TightBindingModel = tight_binding_model,
        interactions::Tuple
    ) where {T<:AbstractFloat}

Write model to summary to file.
"""
function model_summary(;
    simulation_info::SimulationInfo,
    β::T, Δτ::T, model_geometry::ModelGeometry,
    tight_binding_model::TightBindingModel,
    tight_binding_model_dn::TightBindingModel = tight_binding_model,
    interactions::Tuple
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
            # write tight binding model to file
            show(fout, "text/plain", tight_binding_model)
            if tight_binding_model.spin != tight_binding_model_dn.spin
                show(fout, "text/plain", tight_binding_model_dn)
            end
            # write various interactions to file
            for interaction in interactions
                show(fout, "text/plain", interaction)
            end
        end
    end

    return nothing
end