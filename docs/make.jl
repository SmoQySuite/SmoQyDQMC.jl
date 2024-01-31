using SmoQyDQMC
using Documenter
using DocumenterCitations
using Literate
using LatticeUtilities
using JDQMCFramework

# generates script and notebook versions of tutorials based on literate example
function build_examples(example_sources, destdir)
    assetsdir = joinpath(fill("..", length(splitpath(destdir)))..., "assets")

    destpath = joinpath(@__DIR__, "src", destdir)
    isdir(destpath) && rm(destpath; recursive=true)

    # Transform each Literate source file to Markdown for subsequent processing by
    # Documenter.
    for source in example_sources
        # Extract "example" from "path/example.jl"
        name = splitext(basename(source))[1]
        
        # Preprocess each example by adding a notebook download link at the top. The
        # relative path is hardcoded according to the layout of `gh-pages` branch,
        # which is set up by `Documenter.deploydocs`.
        function preprocess(str)
            """
            # Download this example as a [Julia script]($assetsdir/scripts/$name.jl).

            """ * str
        end
        # Write to `src/$destpath/$name.md`
        Literate.markdown(source, destpath; preprocess, credit=false)
    end

    # Create Jupyter notebooks and Julia script for each Literate example. These
    # will be stored in the `assets/` directory of the hosted docs.
    for source in example_sources

        # Build julia scripts
        Literate.script(source, scripts_path; credit=false)
    end

    # Return paths `$destpath/$name.md` for each new Markdown file (relative to
    # `src/`)
    return map(example_sources) do source
        name = splitext(basename(source))[1]
        joinpath(destdir, "$name.md")
    end
end

# Remove existing Documenter `build` directory
build_path = joinpath(@__DIR__, "build")
isdir(build_path) && rm(build_path; recursive=true)
# Create `build/assets` directories
scripts_path = joinpath(build_path, "assets", "scripts")
mkpath.([scripts_path,])

# initialize bibliography
bib = CitationBibliography(
    joinpath(@__DIR__, "src", "references.bib");
    style=:numeric
)

DocMeta.setdocmeta!(SmoQyDQMC, :DocTestSetup, :(using SmoQyDQMC); recursive=true)

examples = ["hubbard_chain", "hubbard_chain_mpi", "hubbard_chain_checkpoint", "holstein_chain",
            "ossh_chain", "bssh_chain", "hubbard_holstein_square", "hubbard_threeband",
            "holstein_kagome", "hubbard_honeycomb"]
example_sources = [joinpath(pkgdir(SmoQyDQMC, "examples"), example*".jl") for example in examples]
example_mds = build_examples(example_sources, "examples")

makedocs(;
    clean = false,
    plugins=[bib],
    modules=[SmoQyDQMC],
    authors="Benjamin Cohen-Stead <benwcs@gmail.com>",
    repo="https://github.com/SmoQySuite/SmoQyDQMC.jl/blob/{commit}{path}#{line}",
    sitename="SmoQyDQMC.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://smoqysuite.github.io/SmoQyDQMC.jl/stable/",
        edit_link="main",
        assets=String[]
    ),
    pages=[
        "Home" => "index.md",
        "Supported Hamiltonians" => "hamiltonian.md",
        "Simulation Output Overview" => "simulation_output.md",
        "API" => "api.md",
        "Examples" => example_mds,
    ],
    draft = true
)

deploydocs(;
    repo="github.com/SmoQySuite/SmoQyDQMC.jl.git",
    devbranch="main",
)
