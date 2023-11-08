using SmoQyDQMC
using Documenter
using DocumenterCitations
using Literate
using LatticeUtilities
using JDQMCFramework

bib = CitationBibliography(
    joinpath(@__DIR__, "src", "references.bib");
    style=:numeric
)

example_names = ["hubbard_chain", "hubbard_chain_mpi", "hubbard_chain_checkpoint", "holstein_chain",
                 "ossh_chain", "bssh_chain", "hubbard_holstein_square", "hubbard_threeband",
                 "holstein_kagome", "hubbard_honeycomb"]
example_literate_sources = [joinpath(@__DIR__, "..", "literate_scripts", name*".jl") for name in example_names]
example_script_destinations = [joinpath(@__DIR__, "..", "example_scripts") for name in example_names]
example_documentation_destination = joinpath(@__DIR__, "src", "examples")
example_documentation_paths = ["examples/$name.md" for name in example_names]

DocMeta.setdocmeta!(SmoQyDQMC, :DocTestSetup, :(using SmoQyDQMC); recursive=true)

for i in eachindex(example_names)
    Literate.markdown(example_literate_sources[i], example_documentation_destination; 
                      execute = false,
                      documenter = false)
    Literate.script(example_literate_sources[i], example_script_destinations[i])
end

makedocs(;
    plugins=[bib],
    modules=[SmoQyDQMC],
    authors="Benjamin Cohen-Stead <benwcs@gmail.com>",
    repo="https://github.com/SmoQySuite/SmoQyDQMC.jl/blob/{commit}{path}#{line}",
    sitename="SmoQyDQMC.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://SmoQySuite.github.io/SmoQyDQMC.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Supported Hamiltonians" => "hamiltonian.md",
        "Simulation Output Overview" => "simulation_output.md",
        "API" => "api.md",
        "Examples" => example_documentation_paths,
    ],
    draft = false
)

deploydocs(;
    repo="github.com/SmoQySuite/SmoQyDQMC.jl.git",
    devbranch="main",
)
