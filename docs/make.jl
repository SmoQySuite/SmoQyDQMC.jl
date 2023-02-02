using SmoQyDQMC
using Documenter
using LatticeUtilities
using JDQMCFramework

DocMeta.setdocmeta!(SmoQyDQMC, :DocTestSetup, :(using SmoQyDQMC); recursive=true)

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
        "Examples" => "examples.md",
        "API" => "api.md",
    ],
)

deploydocs(;
    repo="github.com/SmoQySuite/SmoQyDQMC.jl.git",
    devbranch="main",
)
