# integration test that runs a DQMC simulation of a Hubbard Chain at half-filling
@testitem "Optical SSH Chain" begin
    
    include("../literate_scripts/ossh_chain.jl")
    sID = 1
    Ω = 1.0
    α = 1.0
    μ = 0.0
    β = 2.0
    L = 4
    N_burnin = 10
    N_updates = 10
    N_bins = 10
    @test isnothing(run_ossh_chain_simulation(sID, Ω, α, μ, β, L, N_burnin, N_updates, N_bins, filepath = tempdir()))
end