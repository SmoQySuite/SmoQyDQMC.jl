using SmoQyDQMC
using Documenter
using Literate
using LatticeUtilities
using JDQMCFramework

example_names = ["hubbard_chain", "hubbard_chain_mpi"]
example_literate_sources = [joinpath(@__DIR__, "..", "examples", name*".jl") for name in example_names]
example_script_destinations = [joinpath(@__DIR__, "..", "scripts") for name in example_names]
example_documentation_destination = joinpath(@__DIR__, "src", "examples")
example_documentation_paths = ["examples/$name.md" for name in example_names]

DocMeta.setdocmeta!(SmoQyDQMC, :DocTestSetup, :(using SmoQyDQMC); recursive=true)

for i in eachindex(example_names)
    Literate.markdown(example_literate_sources[i], example_documentation_destination; 
                      execute = false,
                      documenter = true)
    Literate.script(example_literate_sources[i], example_script_destinations[i])
end

makedocs(;
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
        "Examples" => example_documentation_paths,
        "API" => "api.md",
    ],
    draft = false
)

deploydocs(;
    repo="github.com/SmoQySuite/SmoQyDQMC.jl.git",
    devbranch="main",
)
