

using Pkg
Pkg.activate("projet_env")
# Pkg.add("ADNLPModels")
# Pkg.add("NLPModels")
# Pkg.add("Krylov")
# Pkg.add("LinearOperators")
# Pkg.add("JSOSolvers")
# Pkg.add("SolverTools")
# Pkg.add("SolverCore")
# Pkg.add("OptimizationProblems")
# Pkg.add("SolverBenchmark")
# Pkg.add("NLPModelsIpopt")
# Pkg.add("JLD2")
# Pkg.add("Plots")

# TODO: add CUTest

using LinearAlgebra, NLPModels , ADNLPModels, Printf, LinearOperators, Krylov
using OptimizationProblems, OptimizationProblems.ADNLPProblems, JSOSolvers, SolverTools, SolverCore, SolverBenchmark, NLPModelsIpopt
using JLD2, Plots

include("TRSolver.jl")

DEBUG = true            # if DEBUG is set to true, the benchmark is performed on 5 selected functions.
EXE_PROBLEMS = true     # if EXE_PROBLEMS

nvars = 1000

meta = OptimizationProblems.meta

if DEBUG
    problem_list = ["fletchcr", "nondquar", "woods", "broydn7d", "sparsine"]
else
    problem_list = meta[(meta.variable_nvar.==true).&(meta.ncon.==0).&.!meta.has_bounds.&(meta.minimize.==true), :name]
end

# problem_list = problem_list[1:10]

problems = (OptimizationProblems.ADNLPProblems.eval(Meta.parse(problem))(n=nvars) for problem ∈ problem_list)

solvers = Dict(
    # :ipopt => nlp -> ipopt(nlp, print_level=0),
    :trunk => nlp -> trunk(nlp, verbose=0),
    :trsolver_lbfgs_1 => nlp -> trsolver(nlp, subsolver=:lbfgs, max_time=30.0, mem=1, fixed_sub_rtol=true),
    :trsolver_lbfgs_5 => nlp -> trsolver(nlp, subsolver=:lbfgs, max_time=30.0, mem=5, fixed_sub_rtol=true),
    :trsolver_lbfgs_10 => nlp -> trsolver(nlp, subsolver=:lbfgs, max_time=30.0, mem=10, fixed_sub_rtol=true),
    # :trsolver_lbfgs_2 => nlp -> trsolver(nlp, subsolver=:lbfgs, max_time=30.0, mem=2, fixed_sub_rtol=true),
    # :trsolver_lbfgs_5 => nlp -> trsolver(nlp, subsolver=:lbfgs, max_time=30.0, mem=5, fixed_sub_rtol=true),
    :trsolver_cg => nlp -> trsolver(nlp, subsolver=:cg, max_time=30.0, fixed_sub_rtol=true),
)



# solvers = Dict(
#     :ipopt => nlp -> ipopt(nlp, print_level=0),
#     :trunk => nlp -> trunk(nlp, verbose=0),
#     :trsolver_cg => nlp -> trsolver(nlp, subsolver=:cg, max_time=30.0),
#     :trsolver_lbfgs_1 => nlp -> trsolver(nlp, subsolver=:lbfgs, max_time=30.0, mem=1),
#     :trsolver_lbfgs_5 => nlp -> trsolver(nlp, subsolver=:lbfgs, max_time=30.0, mem=5),
#     :trsolver_lbfgs_10 => nlp -> trsolver(nlp, subsolver=:lbfgs, max_time=30.0, mem=10),
#     :trsolver_lbfgs_10 => nlp -> trsolver(nlp, subsolver=:lbfgs, max_time=30.0, mem=20),
#     :trsolver_lbfgs_10_no_scaling => nlp -> trsolver(nlp, subsolver=:lbfgs, max_time=30.0, mem=20, scaling=false),
# )

if EXE_PROBLEMS
    stats = bmark_solvers(solvers, problems)
    @save "data/stats_opt_problems.jld2" stats
else
    @load "data/stats_opt_problems.jld2" stats
end

# set default plot dpi
default(dpi=300)

plt_iter = performance_profile(stats, df -> df.iter, tol = 1e-5, xlabel="Iterations ratio")
savefig(plt_iter, "figures_large/performance_profile_iter.png")

# TODO: add gradient evaluations?
plt_iter = performance_profile(stats, df -> df.neval_obj, tol = 1e-5, xlabel="Objective evaluations ratio")
savefig(plt_iter, "figures_large/performance_profile_neval.png")

plt_iter = performance_profile(stats, df -> df.neval_hprod, tol = 1e-5, xlabel="h_prod ratio")
savefig(plt_iter, "figures_large/performance_profile_hprod.png")

plt_time = performance_profile(stats, df -> df.elapsed_time, tol = 1e-5, xlabel="Elapsed time ratio")
savefig(plt_time, "figures_large/performance_profile_time.png")

# lbfgs_stats = Dict(k => v for (k, v) in stats if occursin("lbfgs", String(k)))
# plt_iter = performance_profile(lbfgs_stats, df -> df.elapsed_time, tol = 1e-5, xlabel="Elapsed time ratio")
# savefig(plt_iter, "figures/performance_profile_time_lbfgs.png")
