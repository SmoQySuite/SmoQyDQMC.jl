# integration test that runs a DQMC simulation of a Hubbard-Holstein model on a square lattice.
@testitem "Hubbard-Holstein Square Example" begin
    
    include("../examples/hubbard_holstein_square.jl")
    sID = abs(rand(Int))
    U = 4.0
    Ω = 1.0
    α = 1.0
    μ = 0.0
    β = 2.0
    L = 4
    N_burnin = 2
    N_updates = 2
    N_bins = 2
    @test isnothing(run_hubbard_holstein_square_simulation(sID, U, Ω, α, μ, β, L, N_burnin, N_updates, N_bins, filepath = tempdir()))
end