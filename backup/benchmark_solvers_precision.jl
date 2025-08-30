using Pkg
Pkg.activate("projet_env")

# Pkg.add("DataFrames")
# Pkg.add("CSV")

using Random

using DataFrames, CSV

using OptimizationProblems, OptimizationProblems.ADNLPProblems
using NLPModels, ADNLPModels

using Quadmath
using JSOSolvers

include("TrunkSolverUpdate.jl")


DEBUG = true

nvars = 1000
n_problems = 20

result_file = "benchmark.csv"

meta = OptimizationProblems.meta

if DEBUG
    problem_list = ["fletchcr", "nondquar", "woods", "broydn7d", "sparsine"]
else
    problem_list = meta[(meta.variable_nvar.==true).&(meta.ncon.==0).&.!meta.has_bounds.&(meta.minimize.==true), :name]
end

if !isfile(result_file)
    CSV.write(result_file, DataFrame(
        problem=String[],
        nvar=Int16[],
        precision=Float64[],
        solver=String[],
        status=String[],
        objective=Float64[],
        dual_feas=Float64[],
        iter=Int16[],
        elapsed_time=Float64[]
    ))
end

row = DataFrame((
        problem=String[],
        nvar=Int16[],
        precision=Type[],
        solver=String[],
        status=String[],
        objective=Float64[],
        dual_feas=Float64[],
        iter=Int16[],
        elapsed_time=Float64[]
))


# precisions = [Float32, Float64, Float128]
precisions = [Float32, Float64]

solvers = Dict(
    :trunk_cg => (nlp; kwargs...) -> trunk(nlp; kwargs...),
    :trunk_lbfgs_1 => (nlp; kwargs...) -> TrunkSolverUpdate.trunk(nlp; subsolver=:lbfgs, mem=1, kwargs...),
    :trunk_lbfgs_2 => (nlp; kwargs...) -> TrunkSolverUpdate.trunk(nlp; subsolver=:lbfgs, mem=2, kwargs...),
    :trunk_lbfgs_20 => (nlp; kwargs...) -> TrunkSolverUpdate.trunk(nlp; subsolver=:lbfgs, mem=20, kwargs...),
    :trunk_lbfgs_100 => (nlp; kwargs...) -> TrunkSolverUpdate.trunk(nlp; subsolver=:lbfgs, mem=100, kwargs...),
)


max_time = 120.0


problems = (OptimizationProblems.ADNLPProblems.eval(Meta.parse(problem))(n=nvars, type=Float32) for problem ∈ problem_list)

for float_type in precisions
    println("======= Float type= ", float_type, " =======")

    problems = (OptimizationProblems.ADNLPProblems.eval(Meta.parse(problem))(n=nvars, type=float_type) for problem ∈ problem_list)

    for nlp in problems

        shuffled_solvers = shuffle(collect(solvers))

        for (sol_name, sol) in shuffled_solvers
            println("solver= ", sol_name)
            stats = sol(nlp; max_time=max_time, verbose=0)

            push!(row, (
                problem = nlp.meta.name,
                nvar = nlp.meta.nvar,
                precision = float_type,
                solver = String(sol_name),
                status = string(stats.status),
                objective = stats.objective,
                dual_feas = stats.dual_feas,
                iter = stats.iter,
                elapsed_time = stats.elapsed_time,
            ))

            CSV.write(result_file, row; append=true)
        end
    end
end


        # println("elapsed_time= ", stats.elapsed_time)
        # println("dual_feas= ", stats.dual_feas)
        # println("iter= ", stats.iter)