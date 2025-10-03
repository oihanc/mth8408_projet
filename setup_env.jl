using Pkg

Pkg.activate("test_diom_tr")

packages = [
    "SuiteSparseMatrixCollection",
    "MatrixMarket",
    "SparseArrays",
    "LinearAlgebra",
    "CSV",
    "DataFrames",
    "Printf",
    "Conda"
]

for pkg in packages
    try
        Pkg.add(pkg)
    catch err
        @warn "Failed to add $pkg" exception=(err, catch_backtrace())
    end
end

# install matplotlib and PyPlot
using Conda
Conda.add("matplotlib")
Pkg.add("PyPlot")

# install Krylov local fork
Pkg.develop(path="/Krylov.jl")      # adjust path if required