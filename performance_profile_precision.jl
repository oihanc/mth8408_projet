using Pkg
Pkg.activate("projet_env")

using DataFrames, CSV, Plots

"""
Create performance profiles for a given CSV results file.

# Arguments
- `filename::String`: path to CSV file with benchmark results
- `nvar::Int`: filter problems by number of variables
- `precision::String`: filter by precision string (e.g. "Float64", "Float128")
"""
function performance_profiles(filename::String; 
                              nvar::Union{Nothing,Int}=nothing, 
                              precision::Union{Nothing,String}=nothing,
                              solvers::Union{Nothing, Vector{String}}=nothing)

    df = CSV.read(filename, DataFrame)

    if nvar !== nothing && :nvar in Symbol.(names(df))
        df = subset(df, :nvar => ByRow(==(nvar)))
    end
    
    if precision !== nothing && :precision in Symbol.(names(df))
        df = subset(df, :precision => ByRow(==(precision)))
    end

    if solvers !== nothing && :solver in Symbol.(names(df))
        df = subset(df, :solver => ByRow(in(solvers)))
    end

    # Collect solvers and problems
    solvers  = unique(df.solver)
    problems = unique(df.problem)

    metrics = [:iter, :elapsed_time, :neval_hprod]
    titles  = ["Iterations", "Elapsed time (s)", "Neval hprod"]

    plt = plot(layout=(1,3), size=(1200, 700), legend=:bottomright)

    for (j, metric) in enumerate(metrics)

        println("metric= ", metric)

        # ratios per solver
        ratios = Dict(s => Float64[] for s in solvers)

        for prob in problems
            sub = df[df.problem .== prob, :]

            # select only converged runs
            solved = subset(sub, :status => ByRow(==("first_order")))
            
            # if all solvers failed to solve a problem
            if nrow(solved) == 0 || !(metric in Symbol.(names(solved)))
                for s in solvers
                    push!(ratios[s], Inf)
                end
                continue
            end

            best = minimum(solved[:, metric])

            for s in solvers
                row = filter(:solver => ==(s), solved)
                if nrow(row) == 0
                    push!(ratios[s], Inf)
                else
                    push!(ratios[s], row[1, metric] / best)
                end
            end
        end

        max_ratio = maximum(x -> isfinite(x) ? x : -Inf, vcat(values(ratios)...))


        for (solver, rvals) in ratios
            finite_rvals = sort(rvals)
            portion = (1:length(finite_rvals)) ./ length(problems)


            plot!(plt[j], finite_rvals, portion, 
                  linetype=:steppost, label=solver,
                  xlabel=titles[j], ylabel="Portion solved")
        end
    end

    display(plt)
    return plt
end


solvers = [
    "trunk_cg", 
    "trunk_lbfgs_2", 
    "trunk_lbfgs_50",
    "trunk_lbfgs_100",
    "trunk_lbfgs_500",
]

# solvers = [
#     "trunk_cg", 
#     "trunk_diom_2", 
#     "trunk_diom_5", 
#     "trunk_diom_50",
#     "trunk_diom_100",
#     "trunk_diom_500",
# ]

# Example usage
performance_profiles("benchmark_all_float128.csv"; nvar=100, precision="Float128", solvers=solvers)