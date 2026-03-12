# --- activating environment ---
using Pkg
Pkg.activate("../env")

using ArgParse

using LinearAlgebra, NLPModels, ADNLPModels, Printf, LinearOperators, Krylov
using OptimizationProblems, OptimizationProblems.ADNLPProblems, JSOSolvers, SolverTools, SolverCore, SolverBenchmark #, NLPModelsIpopt
using JLD2, Plots

N = 5
to_skip = ["variational"]
max_time = 240.

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--nvar"
            help = "a positional argument"
            arg_type = Int
            default = 100
        "--precision"
            arg_type = Int
            help = "10^-(precision)"
            default = 8
        "--subsolver"
        arg_type = String
            help = "cg, diom or lbfgs"
            default = "cg"
        "--memory"
            arg_type = Int
            help = "memory"
            default = 20
    end

    return parse_args(s)
end

function main()

    parsed_args = parse_commandline()

    nvar = parsed_args["nvar"]
    prec = parsed_args["precision"]
    subsolver = parsed_args["subsolver"]
    memory = parsed_args["memory"]
    
    precision = 10.0^(-prec)

    directory = "nvar_$(nvar)_prec_$(prec)"
    mkpath(directory)


    meta = OptimizationProblems.meta

    if nvar != 0
        problem_names = meta[(meta.ncon .== 0) .& .!meta.has_bounds .& (meta.variable_nvar .== true) .& (meta.minimize.==true), :name]
        problems = (eval(Meta.parse(problem))(n=nvar) for problem ∈ problem_names)
    else
        problem_names = meta[(meta.ncon .== 0) .& .!meta.has_bounds .& (50 .<= meta.nvar) .& (meta.minimize.==true), :name]
        problems = (eval(Meta.parse(problem))() for problem ∈ problem_names)
    end
    

    println("--> Problem found:    $(length(problem_names))")

    subsolver_symb = Symbol(subsolver)

    if subsolver == "cg"
        solver_symbol = solver_symbol = Symbol("trunk_", subsolver)
        solvers = Dict(
            solver_symbol => nlp -> trunk(nlp, verbose=0, max_time=max_time, atol=precision, rtol=precision),
        )
        res_path = joinpath(directory, "stats_$(subsolver).jld2")
    elseif subsolver in ["diom", "lbfgs"]
        solver_symbol = Symbol("trunk_", subsolver, "_", memory)
        
        solvers = Dict(
            solver_symbol => nlp -> trunk(nlp, verbose=0, max_time=max_time, atol=precision, rtol=precision, subsolver=subsolver_symb, subsolver_kwargs=(memory=memory,))
        )
        res_path = joinpath(directory, "stats_$(subsolver)_mem$(memory).jld2")
    end
    

    stats = bmark_solvers(solvers, problems, skipif = prob -> prob.meta.name ∈ to_skip)

    @save res_path stats

end


main()
