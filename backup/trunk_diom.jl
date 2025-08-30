using Pkg
Pkg.activate("projet_env")

using LinearAlgebra, NLPModels , ADNLPModels, Printf, LinearOperators, Krylov
using OptimizationProblems, OptimizationProblems.ADNLPProblems, JSOSolvers, SolverTools, SolverCore, SolverBenchmark, NLPModelsIpopt

# include("diom_tr.jl")

# Base.include(Krylov, "diom_tr.jl")

include("TrunkSolverLBFGS.jl")

f(x) = x[1]^2 * x[2]^2
x0 = ones(2)
nlp = ADNLPModel(f, x0)

A = hess_op(nlp, x0)
b = grad(nlp, x0)

solver = TrunkSolverLBFGS.TrunkSolver(nlp, subsolver=:cg)
stats = solve!(solver, nlp, verbose=0)
println(stats)


solver = TrunkSolverLBFGS.TrunkSolver(nlp, subsolver=:lbfgs, mem=10)
stats = solve!(solver, nlp, verbose=0)
println(stats)

solver = TrunkSolverLBFGS.TrunkSolver(nlp, subsolver=:diom, mem=10)
stats = solve!(solver, nlp, verbose=0)
println(stats)






