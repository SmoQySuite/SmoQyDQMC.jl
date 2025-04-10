# integration test of square hubbard model tutorial
@testitem "Square Hubbard Tutorial" begin

    include("../tutorials/hubbard_square.jl")
    @test isnothing(
        run_simulation(;
            sID       = 1,
            U         = 4.0,
            t′        = 0.0,
            μ         = 0.0,
            L         = 4,
            β         = 3.0,
            N_therm   = 10,
            N_updates = 10,
            N_bins    = 10,
            filepath  = tempdir()
        )
    )
end