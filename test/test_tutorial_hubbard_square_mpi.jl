# integration test of square hubbard model tutorial
@testitem "Square Hubbard with MPI Tutorial" begin

    include("../tutorials/hubbard_square_mpi.jl")

    MPI.Init()
    @test isnothing(
        run_simulation(
            MPI.COMM_WORLD;
            sID       = rand(Int),
            U         = 4.0,
            t′        = 0.0,
            μ         = 0.0,
            L         = 4,
            β         = 1.0,
            N_therm   = 2,
            N_updates = 2,
            N_bins    = 2,
            filepath  = tempdir()
        )
    )
    MPI.Finalize()
end