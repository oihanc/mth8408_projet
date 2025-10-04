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
    "Conda",
    "PyCall"
]

for pkg in packages
    try
        Pkg.add(pkg)
    catch err
        @warn "Failed to add $pkg" exception=(err, catch_backtrace())
    end
end


using Conda
Conda.add("python=3.11")
Conda.add("matplotlib=3.10.0")

ENV["PYTHON"] = ""
Pkg.build("PyCall")

Pkg.add("PyPlot")

# install Krylov local fork
Pkg.develop(path="Krylov.jl")      # adjust path if required