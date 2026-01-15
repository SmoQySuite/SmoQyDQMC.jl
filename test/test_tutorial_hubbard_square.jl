@testitem "Square Hubbard Tutorial" begin

    include("../tutorials/hubbard_square.jl")
    @test isnothing(
        run_simulation(;
            sID = abs(rand(Int)),
            U = 4.0,
            t′ = 0.0,
            μ = 0.0,
            L = 4,
            β = 1.0,
            N_therm = 1,
            N_measurements = 2,
            N_updates = 1,
            N_bins = 2,
            filepath = tempdir()
        )
    )
end