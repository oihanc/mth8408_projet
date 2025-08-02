

using Pkg
Pkg.activate("projet_env_phase2_3")
# Pkg.add("ADNLPModels")
# Pkg.add("NLPModels")
# Pkg.add("Krylov")
# Pkg.add("LinearOperators")
# Pkg.add("JSOSolvers")
# # Pkg.add("Plots")
# Pkg.add("SolverTools")
# Pkg.add("SolverCore")
# Pkg.add("OptimizationProblems") # collection + outils pour sélectionner les problèmes


# # include("subsolvers.jl")

# # Pkg.add("SuiteSparseMatrixCollection")
# # Pkg.add("MatrixMarket")
# Pkg.add("SolverBenchmark")
# Pkg.add("NLPModelsIpopt")
Pkg.add("JLD2")
Pkg.add("Plots")

# TODO: add CUTest

using LinearAlgebra, NLPModels , ADNLPModels, Printf, LinearOperators, Krylov
using OptimizationProblems, OptimizationProblems.ADNLPProblems, JSOSolvers, SolverTools, SolverCore, SolverBenchmark, Ipopt, NLPModelsIpopt
using JLD2, Plots

include("TRSolver.jl")

DEBUG = false
EXE_PROBLEMS = false

meta = OptimizationProblems.meta
problem_list = meta[(meta.ncon.==0).&.!meta.has_bounds.&(meta.nvar.==100), :name]
problems = nothing

problem_to_exe = ["fletchcr", "nondquar", "woods", "broydn7d", "sparsine"]

if DEBUG
    problems = (OptimizationProblems.ADNLPProblems.eval(Meta.parse(problem))() for problem ∈ problem_to_exe)
else
    problems = (OptimizationProblems.ADNLPProblems.eval(Meta.parse(problem))() for problem ∈ problem_list)
end

solvers = Dict(
    :ipopt => nlp -> ipopt(nlp, print_level=0),
    :trunk => nlp -> trunk(nlp, verbose=0),
    :trsolver_cg => nlp -> trsolver(nlp, subsolver=:cg, max_time=10.0),
    :trsolver_lbfgs_1 => nlp -> trsolver(nlp, subsolver=:lbfgs, max_time=10.0, mem=1),
    :trsolver_lbfgs_5 => nlp -> trsolver(nlp, subsolver=:lbfgs, max_time=10.0, mem=5),
    :trsolver_lbfgs_10 => nlp -> trsolver(nlp, subsolver=:lbfgs, max_time=10.0, mem=10),
    :trsolver_lbfgs_10_no_scaling => nlp -> trsolver(nlp, subsolver=:lbfgs, max_time=10.0, mem=5, scaling=false),
)

if EXE_PROBLEMS
    stats = bmark_solvers(solvers, problems)
    @save "stats_opt_problems.jld2" stats
else
    @load "stats_opt_problems.jld2" stats
end

plt_iter = performance_profile(stats, df -> df.iter, tol = 1e-5, xlabel="Iterations ratio")
savefig(plt_iter, "performance_profile_iter.png")

plt_iter = performance_profile(stats, df -> df.neval_obj, tol = 1e-5, xlabel="Objective evaluations ratio")
savefig(plt_iter, "performance_profile_neval.png")

plt_time = performance_profile(stats, df -> df.elapsed_time, tol = 1e-5, xlabel="Elapsed time ratio")
savefig(plt_time, "performance_profile_time.png")
