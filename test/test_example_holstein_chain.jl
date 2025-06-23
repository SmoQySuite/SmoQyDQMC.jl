# integration test that runs a DQMC simulation of a Hubbard Chain at half-filling
@testitem "Holstein Chain Example" begin
    
    include("../examples/holstein_chain.jl")
    sID = abs(rand(Int))
    Ω = 1.0
    α = 1.0
    n = 1.0
    μ = 0.0
    β = 2.0
    L = 4
    N_burnin = 2
    N_updates = 2
    N_bins = 2
    @test isnothing(run_holstein_chain_simulation(sID, Ω, α, n, μ, β, L, N_burnin, N_updates, N_bins, filepath = tempdir()))
end