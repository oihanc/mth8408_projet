using Pkg
Pkg.activate("test_diom_tr")
Pkg.develop(path="../Krylov.jl")   # change path to local Krylov fork

# Pkg.add("LinearOperators")
# Pkg.add("JLD2")
# Pkg.add("Plots")


# Pkg.add("NLPModels")
# Pkg.add("ADNLPModels")
# Pkg.add("OptimizationProblems")
# Pkg.add("JSOSolvers")
# Pkg.add("SolverCore")


using Krylov, LinearAlgebra, SparseArrays, Printf, Random, Test

using Printf, LinearOperators
using JLD2, Plots

using Profile

using NLPModels, ADNLPModels
using OptimizationProblems, OptimizationProblems.ADNLPProblems
using JSOSolvers
using SolverCore

meta = OptimizationProblems.meta
problem_list = ["fletchcr", "nondquar", "woods", "broydn7d", "sparsine"]
nvars = 100

problems = (OptimizationProblems.ADNLPProblems.eval(Meta.parse(problem))(n=nvars) for problem ∈ problem_list)
subsolvers = [:cg, :diom]

for nlp in problems
    println("------- ", nlp.meta.name, " -------")
    
    obj_values = []
    
    for sub in subsolvers
        println("--- subsolver = ", sub, " ---")
        solver = TrunkSolver(nlp, subsolver=sub)
        stats = solve!(solver, nlp)
        
        push!(obj_values, stats.objective)

        println("stats = \n", stats)
    end

    println("objective values = ", obj_values)
end