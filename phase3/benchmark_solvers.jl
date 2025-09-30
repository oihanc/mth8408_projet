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

using LinearAlgebra

include("TrunkSolverUpdate.jl")
include("TRSolverModule.jl")


DEBUG = false   # if set to true -> only 5 test problems

nvars = 500

result_file = "benchmark_n500_float32.csv"

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
        elapsed_time=Float64[],
        neval_obj=Int16[],
        neval_grad=Int16[],
        neval_hprod=Int16[],
    ))
end


problem_list = problem_list[1:50]

precisions = [Float32, Float64, Float128]

max_time = 30.0

solvers = Dict(
    :trunk_cg => (nlp; kwargs...) -> trunk(nlp; kwargs...),
    
    :trunk_lbfgs_2 => (nlp; kwargs...) -> TrunkSolverUpdate.trunk(nlp; subsolver=:lbfgs, mem=2, kwargs...),
    :trunk_lbfgs_5 => (nlp; kwargs...) -> TrunkSolverUpdate.trunk(nlp; subsolver=:lbfgs, mem=5, kwargs...),
    :trunk_lbfgs_10 => (nlp; kwargs...) -> TrunkSolverUpdate.trunk(nlp; subsolver=:lbfgs, mem=10, kwargs...),
    :trunk_lbfgs_20 => (nlp; kwargs...) -> TrunkSolverUpdate.trunk(nlp; subsolver=:lbfgs, mem=20, kwargs...),
    :trunk_lbfgs_50 => (nlp; kwargs...) -> TrunkSolverUpdate.trunk(nlp; subsolver=:lbfgs, mem=50, kwargs...),
    :trunk_lbfgs_100 => (nlp; kwargs...) -> TrunkSolverUpdate.trunk(nlp; subsolver=:lbfgs, mem=100, kwargs...),

    :trunk_diom_2 => (nlp; kwargs...) -> TrunkSolverUpdate.trunk(nlp; subsolver=:diom, mem=2, kwargs...),
    :trunk_diom_5 => (nlp; kwargs...) -> TrunkSolverUpdate.trunk(nlp; subsolver=:diom, mem=5, kwargs...),
    :trunk_diom_10 => (nlp; kwargs...) -> TrunkSolverUpdate.trunk(nlp; subsolver=:diom, mem=10, kwargs...),
    :trunk_diom_20 => (nlp; kwargs...) -> TrunkSolverUpdate.trunk(nlp; subsolver=:diom, mem=20, kwargs...),
    :trunk_diom_50 => (nlp; kwargs...) -> TrunkSolverUpdate.trunk(nlp; subsolver=:diom, mem=50, kwargs...),
    :trunk_diom_100 => (nlp; kwargs...) -> TrunkSolverUpdate.trunk(nlp; subsolver=:diom, mem=100, kwargs...),

    # :trsolver_lbfgs_2 => (nlp; fixed_sub_rtol=true, kwargs...) -> TRSolverModule.trsolver(nlp; subsolver=:lbfgs, mem=2, kwargs...),
    # :trsolver_lbfgs_50 => (nlp; fixed_sub_rtol=true, kwargs...) -> TRSolverModule.trsolver(nlp; subsolver=:lbfgs, mem=50, kwargs...),
    # :trsolver_lbfgs_100 => (nlp; fixed_sub_rtol=true, kwargs...) -> TRSolverModule.trsolver(nlp; subsolver=:lbfgs, mem=100, kwargs...),
    # :trsolver_lbfgs_500 => (nlp; fixed_sub_rtol=true, kwargs...) -> TRSolverModule.trsolver(nlp; subsolver=:lbfgs, mem=500, kwargs...),

    # :trsolver_diom_2 => (nlp; fixed_sub_rtol=true, kwargs...) -> TRSolverModule.trsolver(nlp; subsolver=:diom, mem=2, kwargs...),
    # :trsolver_diom_5 => (nlp; fixed_sub_rtol=true, kwargs...) -> TRSolverModule.trsolver(nlp; subsolver=:diom, mem=5, kwargs...),
    # :trsolver_diom_50 => (nlp; fixed_sub_rtol=true, kwargs...) -> TRSolverModule.trsolver(nlp; subsolver=:diom, mem=50, kwargs...),
    # :trsolver_diom_100 => (nlp; fixed_sub_rtol=true, kwargs...) -> TRSolverModule.trsolver(nlp; subsolver=:diom, mem=100, kwargs...),
    # :trsolver_diom_500 => (nlp; fixed_sub_rtol=true, kwargs...) -> TRSolverModule.trsolver(nlp; subsolver=:diom, mem=500, kwargs...),
)

# df = CSV.read("benchmark_n100_float128.csv", DataFrame)
# problem_list = unique(df.problem)

# df = CSV.read(result_file, DataFrame)
# problems_in_file = unique(df.problem)

# println(length(problem_list))
# problem_list = filter(p -> !(p in problems_in_file), problem_list)
# println(length(problem_list))

# problems = (OptimizationProblems.ADNLPProblems.eval(Meta.parse(problem))(n=nvars, type=Float32) for problem ∈ problem_list)

for float_type in precisions
    println("======= Float type= ", float_type, " =======")

    problems = (OptimizationProblems.ADNLPProblems.eval(Meta.parse(problem))(n=nvars, type=float_type) for problem ∈ problem_list)

    for nlp in problems

        x0 = copy(nlp.meta.x0)

        shuffled_solvers = shuffle(collect(solvers))

        for (sol_name, sol) in shuffled_solvers

            if norm(x0 .- nlp.meta.x0) > √(eps(float_type))
                throw("NLPModel.meta.x0 was changed. Review solver implementation!")
            end

            reset!(nlp)

            println("solver= ", sol_name)

            try
                stats = sol(nlp; max_time=max_time, verbose=0)

                row = DataFrame((
                problem = [nlp.meta.name],
                nvar = [nlp.meta.nvar],
                precision = [float_type],
                solver = [String(sol_name)],
                status = [string(stats.status)],
                objective = [stats.objective],
                dual_feas = [stats.dual_feas],
                iter = [stats.iter],
                elapsed_time = [stats.elapsed_time],
                neval_obj = [neval_obj(nlp)],
                neval_grad = [neval_grad(nlp)],
                neval_hprod = [neval_hprod(nlp)],
            ))

            CSV.write(result_file, row; append=true)
            
            catch e
                println("For problem= ", nlp.meta.name)

                if isa(e, MethodError)
                    println("caught a MethodError: $e")
                elseif isa(e, ArgumentError)
                    println("caught an ArgumentError: $e")
                else
                    println("caught an unexpected error: $e")
                end
            end

        end
    end
end