@testitem "Honeycomb Holstein with MPI Tutorial" begin

    include("../tutorials/holstein_honeycomb_mpi.jl")
    MPI.Init()
    @test isnothing(
        run_simulation(
            MPI.COMM_WORLD,
            sID       = rand(Int),
            Ω         = 1.0,
            α         = 1.0,
            μ         = 0.0,
            L         = 3,
            β         = 1.0,
            N_therm   = 2,
            N_updates = 2,
            N_bins    = 2,
            filepath  = tempdir()
        )
    )
end