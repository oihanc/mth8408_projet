

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

problems = (OptimizationProblems.ADNLPProblems.eval(Meta.parse(problem))(n=nvars) for problem ∈ problem_list)

max_i = 5
plt = plot(layout = (max_i, 2), size=(800, 1200))

for (i, nlp) in enumerate(problems)
    if i > max_i
        break
    end

    println("------- ",  nlp.meta.name, " -------")

    elapsed_time = Float64[]
    gradients = Float64[]

    function save_gradients(nlp, solver, stats)
        push!(elapsed_time, stats.elapsed_time)
        push!(gradients, stats.dual_feas)
    end

    stats = trsolver(nlp, subsolver=:lbfgs, max_time=30.0, mem=10, callback=save_gradients, fixed_sub_rtol=true, scaling=true)
    @printf "status= %s  |  obj= %.3e  |  time= %.3e  |  \n" stats.status stats.objective stats.elapsed_time

    plot!(plt[i, 1], gradients, label="LBFGS", yscale=:log10)
    plot!(plt[i, 2], elapsed_time, gradients, label="LBFGS", yscale=:log10)

    elapsed_time = Float64[]
    gradients = Float64[]
    stats = trsolver(nlp, subsolver=:cg, max_time=30.0, callback=save_gradients, fixed_sub_rtol=true)
    plot!(plt[i, 1], gradients, label="CG", yscale=:log10)
    plot!(plt[i, 2], elapsed_time, gradients, label="CG", yscale=:log10)
    @printf "status= %s  |  obj= %.3e  |  time= %.3e  |  \n" stats.status stats.objective stats.elapsed_time

end

display(plt)













# solvers = Dict(
#     # :ipopt => nlp -> ipopt(nlp, print_level=0),
#     :trunk => nlp -> trunk(nlp, verbose=0),
#     :trsolver_lbfgs_1 => nlp -> trsolver(nlp, subsolver=:lbfgs, max_time=10.0, mem=1),
#     # :trsolver_lbfgs_2 => nlp -> trsolver(nlp, subsolver=:lbfgs, max_time=10.0, mem=2),
#     # :trsolver_lbfgs_5 => nlp -> trsolver(nlp, subsolver=:lbfgs, max_time=10.0, mem=5),
#     :trsolver_lbfgs_10 => nlp -> trsolver(nlp, subsolver=:lbfgs, max_time=10.0, mem=10),
#     :trsolver_cg => nlp -> trsolver(nlp, subsolver=:cg, max_time=10.0),
# )



# # solvers = Dict(
# #     # :ipopt => nlp -> ipopt(nlp, print_level=0),
# #     # :trunk => nlp -> trunk(nlp, verbose=0),
# #     :trsolver_cg => nlp -> trsolver(nlp, subsolver=:cg, max_time=10.0),
# #     :trsolver_lbfgs_1 => nlp -> trsolver(nlp, subsolver=:lbfgs, max_time=10.0, mem=1),
# #     :trsolver_lbfgs_5 => nlp -> trsolver(nlp, subsolver=:lbfgs, max_time=10.0, mem=5),
# #     :trsolver_lbfgs_10 => nlp -> trsolver(nlp, subsolver=:lbfgs, max_time=10.0, mem=10),
# #     # :trsolver_lbfgs_10_no_scaling => nlp -> trsolver(nlp, subsolver=:lbfgs, max_time=10.0, mem=5, scaling=false),
# # )

# if EXE_PROBLEMS
#     stats = bmark_solvers(solvers, problems)
#     @save "data/stats_opt_problems.jld2" stats
# else
#     @load "data/stats_opt_problems.jld2" stats
# end

# # set default plot dpi
# default(dpi=300)

# plt_iter = performance_profile(stats, df -> df.iter, tol = 1e-5, xlabel="Iterations ratio")
# savefig(plt_iter, "figures_large/performance_profile_iter.png")

# plt_iter = performance_profile(stats, df -> df.neval_obj, tol = 1e-5, xlabel="Objective evaluations ratio")
# savefig(plt_iter, "figures_large/performance_profile_neval.png")

# plt_time = performance_profile(stats, df -> df.elapsed_time, tol = 1e-5, xlabel="Elapsed time ratio")
# savefig(plt_time, "figures_large/performance_profile_time.png")

# # lbfgs_stats = Dict(k => v for (k, v) in stats if occursin("lbfgs", String(k)))
# # plt_iter = performance_profile(lbfgs_stats, df -> df.elapsed_time, tol = 1e-5, xlabel="Elapsed time ratio")
# # savefig(plt_iter, "figures/performance_profile_time_lbfgs.png")
