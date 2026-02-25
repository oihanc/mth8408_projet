using Pkg

Pkg.activate("env")

required_packages = [
    "LinearAlgebra",
    "LinearOperators",
    "Printf",
    "NLPModels",
    "ADNLPModels",
    "OptimizationProblems",
    "SolverTools",
    "SolverCore",
    "SolverBenchmark",
    "JLD2",
    "Plots",
    "ArgParse",
]

function ensure_pkg(pkg::String)
    if !haskey(Pkg.project().dependencies, pkg)
        Pkg.add(pkg)
    end
end

for pkg in required_packages
    ensure_pkg(pkg)
end

# # install matplotlib and PyPlot
# using Conda
# Conda.add("matplotlib")
# Pkg.add("PyPlot")

# install Krylov local fork
Pkg.develop(path="../Krylov.jl")           # adjust path if required
Pkg.develop(path="../JSOSolvers.jl")       # adjust path if required