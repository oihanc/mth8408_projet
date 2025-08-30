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


@load "data/stats_opt_problems.jld2" stats


typeof(stats)

solvers = keys(stats)
dfs = (stats[s] for s in solvers)



plt = plot(layout = (1, 2), size=(1000, 600))


for (solver, df) in stats

    println(propertynames(df))


    # Filter only solved runs
    solved_df = filter(:status => ==(:first_order), df)

    # Now you have only the rows where the solver converged properly
    iterations = solved_df[:, :iter]
    elapsed_time = solved_df[:, :elapsed_time]

    # Sort for empirical CDF
    sorted_iters = sort(iterations)
    sorted_time = sort(elapsed_time)

    portion_solved = (1:length(sorted_iters)) ./ length(df[:, :status])

    println("portion solved= ", portion_solved)

    # Plot (number iterations vs portion solved)
    plot!(plt[1, 1], sorted_iters, portion_solved, linetype=:steppost, label=String(solver))
    plot!(plt[1, 2], sorted_time, portion_solved, linetype=:steppost, label=String(solver))
end

    display(plt)




# problems = keys(df)
# println(problems)
# df = (df[nlp] for nlp in problems)


# println(first(df))

# for stats in df
#     println(stats)
# end


