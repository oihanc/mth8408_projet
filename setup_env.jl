using Pkg

Pkg.activate("test_diom_tr")

packages = [
    # "SuiteSparseMatrixCollection",
    # "MatrixMarket",
    "SparseArrays",
    "LinearAlgebra",
    "CSV",
    "DataFrames",
    "Printf",
    # "Conda",
    "LinearOperators",
    "OptimizationProblems",
    "SolverTools",
    "SolverCore",
    "SolverBenchmark",
    "NLPModels",
    "ADNLPModels",
    "NLPModelsIpopt",
    "JLD2",
    "Plots",
]

for pkg in packages
    println("Adding: ", pkg)
    Pkg.add(pkg)
end

# # install matplotlib and PyPlot
# # using Conda
# # Conda.add("matplotlib")
# # Pkg.add("PyPlot")

# # install Krylov local fork
# Pkg.develop(path="/home/oicor/Krylov.jl")           # adjust path if required
# Pkg.develop(path="/home/oicor/JSOSolvers.jl")       # adjust path if required