using Pkg
Pkg.activate("projet_env")

using DataFrames, CSV, Plots

"""
    performance_profiles(filename::String; 
                              nvar::Union{Nothing,Int}=nothing, 
                              precision::Union{Nothing,String}=nothing,
                              solvers::Union{Nothing, Vector{String}}=nothing)
                              
create performance profiles for a given CSV results file.

# arguments
- `filename::String`: path to CSV file with benchmark results
- `nvar::Int`: filter problems by the number of variables
- `precision::String`: filter by the precision string (e.g. "Float64", "Float128")
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

    plt = plot(layout=(1,3), size=(1500, 500), legend=:bottomright, framestyle=:box, margin=10Plots.mm)

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

        # extend each line to the maximum ratio
        allvals = vcat(values(ratios)...)
        finite_vals = filter(isfinite, allvals)
        ratio_max = 1.05 * maximum(finite_vals) 

        for (solver, rvals) in ratios
            finite_rvals = sort(filter(isfinite, rvals))
            portion = (1:length(finite_rvals)) ./ length(problems)

            # extend the step horizontally to ratio_max
            if !isempty(finite_rvals)
                xvals = vcat(finite_rvals, ratio_max)
                yvals = vcat(portion, portion[end])
            else
                # all failures, plot a horizontal line at 0
                xvals = [1.0, ratio_max]
                yvals = [0.0, 0.0]
            end

            plot!(plt[j], xvals, yvals,
                  linetype=:steppost, label=solver,
                  xlabel=titles[j], xaxis=:log, ylabel="Portion solved")
        end
    end

    display(plt)
    return plt
end


# solvers = [
#     "trunk_cg", 
#     "trunk_lbfgs_2", 
#     "trunk_lbfgs_50",
#     "trunk_lbfgs_100",
# ]

# solvers = [
#     "trunk_cg", 
#     "trunk_diom_2", 
#     "trunk_diom_5", 
#     "trunk_diom_50",
#     "trunk_diom_100",
# ]

solvers = [
    "trunk_cg", 
    "trunk_lbfgs_100",
    "trunk_diom_100",
]


performance_profiles("benchmark_n500_float64.csv"; nvar=500, precision="Float64", solvers=solvers)