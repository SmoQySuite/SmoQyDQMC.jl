# integration test of square hubbard model tutorial
@testitem "Honeycomb Holstein Tutorial" begin

    include("../tutorials/holstein_honeycomb.jl")
    @test isnothing(
        run_simulation(
            sID       = abs(rand(Int)),
            Ω         = 1.0,
            α         = 1.0,
            μ         = 0.0,
            L         = 3,
            β         = 3.0,
            N_therm   = 2,
            N_updates = 2,
            N_bins    = 2,
            filepath  = tempdir()
        )
    )
end