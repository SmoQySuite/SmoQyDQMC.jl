@testitem "Optical SSH Square Example" begin

    include("../examples/ossh_square.jl")
    MPI.Init()
    @test isnothing(
        run_simulation(
            MPI.COMM_WORLD,
            sID = abs(rand(Int)),
            Ω = 1.0,
            α = 1.0,
            μ = 0.0,
            L = 4,
            β = 1.0,
            N_therm = 1,
            N_measurements = 2,
            N_bins = 2,
            checkpoint_freq = 1.0,
            filepath = tempdir()
        )
    )
end