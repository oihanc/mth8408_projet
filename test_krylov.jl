using Pkg
Pkg.activate("test_diom_tr")


using Krylov, LinearAlgebra, SparseArrays, Printf, Random, Test

import Krylov: solution, statistics, results, elapsed_time,
               solution_count, iteration_count, Aprod_count, Atprod_count,
               issolved, issolved_primal, issolved_dual

include("Krylov.jl/test/test_utils.jl")

include("Krylov.jl/test/test_cg.jl")
include("Krylov.jl/test/test_diom.jl")

# Pkg.test("Krylov"; coverage=true)
