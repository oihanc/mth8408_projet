# --- activating environment ---
using Pkg
Pkg.activate("../env")

using ArgParse

using LinearAlgebra, NLPModels, ADNLPModels, Printf, LinearOperators, Krylov
using OptimizationProblems, OptimizationProblems.ADNLPProblems, JSOSolvers, SolverTools, SolverCore, SolverBenchmark #, NLPModelsIpopt
using JLD2, Plots


N = 10


function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--opt1"
            help = "an option with an argument"
        "dim"
            help = "a positional argument"
            arg_type = Int
            required = true
    end

    return parse_args(s)
end

function main()

    parsed_args = parse_commandline()

    N = parsed_args["dim"]

    meta = OptimizationProblems.meta

    if N != 0
        problem_names = meta[(meta.ncon .== 0) .& .!meta.has_bounds .& (meta.variable_nvar .== true) .& (meta.minimize.==true), :name]
        problems = (eval(Meta.parse(problem))(n=N) for problem ∈ problem_names)
    else
        problem_names = meta[(meta.ncon .== 0) .& .!meta.has_bounds .& (50 .<= meta.nvar) .& (meta.minimize.==true), :name]
        problems = (eval(Meta.parse(problem))() for problem ∈ problem_names)
    end
    

    println("--> Problem found:    $(length(problem_names))")

    solvers = Dict(
        # :ipopt => nlp -> ipopt(nlp, print_level=0),
        :trunk_cg => nlp -> trunk(nlp, verbose=0),
        # :trunk_diom_2 => nlp -> trunk(nlp, verbose=0, subsolver=:diom, subsolver_kwargs=(memory=2,)),
        :trunk_diom_20 => nlp -> trunk(nlp, verbose=0, subsolver=:diom, subsolver_kwargs=(memory=20,)),
        :trunk_diom_50 => nlp -> trunk(nlp, verbose=0, subsolver=:diom, subsolver_kwargs=(memory=50,)),
        #:trunk_diom_100 => nlp -> trunk(nlp, verbose=0, subsolver=:diom, subsolver_kwargs=(memory=100,)),
        # :trunk_diom_200 => nlp -> trunk(nlp, verbose=0, subsolver=:diom, subsolver_kwargs=(memory=200,)),
    )

    stats = bmark_solvers(solvers, problems)

    @save "stats_dim_$(N).jld2" stats

    performance_profile(stats, df -> df.neval_hprod)

end


main()