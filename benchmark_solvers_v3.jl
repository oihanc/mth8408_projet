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
# Pkg.add("ProgressBars")

# TODO: add CUTest

using LinearAlgebra, NLPModels, ADNLPModels, Printf, LinearOperators, Krylov
using OptimizationProblems, OptimizationProblems.ADNLPProblems, JSOSolvers, SolverTools, SolverCore, SolverBenchmark, NLPModelsIpopt
using JLD2, Plots
using ProgressBars

include("TrunkSolverUpdate.jl")


DEBUG = true            # if DEBUG is set to true, the benchmark is performed on 5 selected functions.
EXE_PROBLEMS = true     # if EXE_PROBLEMS

nvars = 1_000
n_problems = 20

meta = OptimizationProblems.meta

if DEBUG
    problem_list = ["fletchcr", "nondquar", "woods", "broydn7d", "sparsine"]
else
    problem_list = meta[(meta.variable_nvar.==true).&(meta.ncon.==0).&.!meta.has_bounds.&(meta.minimize.==true), :name]
end

if n_problems > length(problem_list)
    n_problems = length(problem_list)
end

problem_list = problem_list[1:n_problems]

problems = (OptimizationProblems.ADNLPProblems.eval(Meta.parse(problem))(n=nvars) for problem ∈ problem_list)


solvers = Dict(
    :trunk_cg => nlp -> trunk(nlp, max_time=10.0, verbose=0),
    # :trunk_lbfgs_5 => nlp -> TrunkSolverLBFGS.trunk(nlp, subsolver=:lbfgs, mem=5, scaling=true, verbose=0),
    # :trunk_lbfgs_2 => nlp -> TrunkSolverLBFGS.trunk(nlp, subsolver=:lbfgs, mem=2, scaling=true, verbose=0),
    # :trunk_lbfgs_1 => nlp -> TrunkSolverLBFGS.trunk(nlp, subsolver=:lbfgs, mem=1, scaling=true, verbose=0),
    :trunk_lbfgs_20 => nlp -> TrunkSolverLBFGS.trunk(nlp, max_time=10.0, subsolver=:lbfgs, mem=100, scaling=true, verbose=0),
    :trunk_diom_10 => nlp -> TrunkSolverLBFGS.trunk(nlp, subsolver=:diom, mem=10, max_time=120.0, verbose=0),
    :trunk_diom_20 => nlp -> TrunkSolverLBFGS.trunk(nlp, subsolver=:diom, mem=20, max_time=120.0, verbose=0),
)


if EXE_PROBLEMS
    stats = bmark_solvers(solvers, problems)
    @save "data/stats_opt_problems.jld2" stats
else
    @load "data/stats_opt_problems.jld2" stats
end

# set default plot dpi
default(dpi=300)

plt_iter = performance_profile(stats, df -> df.iter, tol = 1e-5, xlabel="Iterations ratio")
savefig(plt_iter, "figures_new/performance_profile_iter.png")

# TODO: add gradient evaluations?
plt_iter = performance_profile(stats, df -> df.neval_obj, tol = 1e-5, xlabel="Objective evaluations ratio")
savefig(plt_iter, "figures_new/performance_profile_neval.png")

plt_iter = performance_profile(stats, df -> df.neval_hprod, tol = 1e-5, xlabel="h_prod ratio")
savefig(plt_iter, "figures_new/performance_profile_hprod.png")

plt_time = performance_profile(stats, df -> df.elapsed_time, tol = 1e-5, xlabel="Elapsed time ratio")
savefig(plt_time, "figures_new/performance_profile_time.png")

# lbfgs_stats = Dict(k => v for (k, v) in stats if occursin("lbfgs", String(k)))
# plt_iter = performance_profile(lbfgs_stats, df -> df.elapsed_time, tol = 1e-5, xlabel="Elapsed time ratio")
# savefig(plt_iter, "figures/performance_profile_time_lbfgs.png")
