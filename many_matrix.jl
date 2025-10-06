using Pkg
Pkg.activate("test_diom_tr")

using SuiteSparseMatrixCollection
using MatrixMarket
using SparseArrays
using LinearAlgebra

using CSV
using DataFrames

using Krylov

using PyCall
using PyPlot
using Printf


function test_on_collection(num_convex, solvers)
    ssmc = ssmc_db()
    # filter real, symmetric, positive definite
    sel = ssmc[(ssmc.numerical_symmetry .== 1) .&
               (ssmc.positive_definite .== true) .&
               (ssmc.real .== true), :]

    println("Found $(size(sel,1)) candidate matrices")
    
    sel = first(sel, num_convex)

    # fetch/download them (if not already)
    # download all of them at once
    # paths = fetch_ssmc(sel, format="MM")

    results = Dict{String, Any}()

    # define and create saving directory
    directory = "benchmark_convex/"
    mkpath(directory)

    for (i, matmeta) in enumerate(eachrow(sel))

        # matmeta has fields “group” and “name”
        # pname = string(matmeta.group, "_", matmeta.name)
        # path = paths[i]  # directory containing .mtx
        # file = joinpath(path, matmeta.name * ".mtx")

        # download matrix only when it's its time to benchmark it
        pname = string(matmeta.group, "_", matmeta.name)
        path = fetch_ssmc(matmeta.group, matmeta.name; format="MM")
        file = joinpath(path, matmeta.name * ".mtx")

        println("------- (", i, "/", num_convex, ")  name = ", pname, " -------")

        # read matrix
        A = MatrixMarket.mmread(file)
        A = Symmetric(A)

        n, n = size(A)
        b = ones(n) ./ √n

        convex_results = Dict()

        for sol in solvers
            
            solver = sol[1]
            mem = sol[2]
            solver_name = sol[3]

            println("solver = ", solver, "  |  mem = ", mem)

            try
                # perform blank run for the first iteration for accurate elapsed time measurement
                if i == 1
                    _ = benchmark_krylov(A, b, solver=solver, mem=mem)
                end

                df = benchmark_krylov(A, b, solver=solver, mem=mem)

                # save data to .csv file
                result_file = joinpath(directory, string(pname, "_", solver_name, ".csv"))
                CSV.write(result_file, df)

                convex_results[solver_name] = df

            catch err
                println("error ", err)
            end
        end

        plot_path = joinpath(directory, "$pname.pdf")
        plot_name = pname
        convex_dim = n
        convex_cond = cond(Array(A), 2)
        draw_plot(plot_path, plot_name, convex_dim, convex_cond, convex_results)

    end
end


function benchmark_krylov(A, b; solver = :cg, mem=nothing)

    n, n = size(A)

    objectives = Float64[0.0]
    elapsed_time = Float64[0.0]

    # compute quadratic objective
    # function compute_obj(x, A, g)
    #     return dot(g, x) + 0.5 * dot(x, A*x)
    # end

    # function objective_callback(workspace)
    #     push!(objectives, compute_obj(workspace.x, A, b))

    #     timing_callback(workspace)
    #     return false
    # end

    
    function timing_callback(workspace)
        push!(elapsed_time, time() - start_time)
        return false
    end

    if solver == :cg
        krylov_solver = krylov_workspace(Val(solver), A, -b)
    elseif solver == :diom
        mem = round(Int, mem*n)
        krylov_solver = krylov_workspace(Val(solver), A, -b, memory=mem)
    end

    start_time = time()

    # perform problem's first trial. Save objective values
    krylov_solve!(krylov_solver, A, -b; itmax=10*n, history=true, callback=timing_callback)
    stats = krylov_solver.stats

    first_res = copy(stats.residuals)
    objectives = copy(stats.quadras)
    
    if solver == :cg
        krylov_solver = krylov_workspace(Val(solver), A, -b)
    elseif solver == :diom
        krylov_solver = krylov_workspace(Val(solver), A, -b, memory=mem)
    end

    
    elapsed_time = Float64[0.0]
    start_time = time()

    # perform peroblem' second trial. Measure elapsed time. Do not save objective values.
    krylov_solve!(krylov_solver, A, -b; itmax=10*n, history=true, callback=timing_callback)
    stats = krylov_solver.stats

    # check residuals
    if length(first_res) != length(stats.residuals)
        throw(string("First and second run residuals do not have the same dimensions: ", length(first_res), " vs ", length(stats.residuals), "."))
    end
    error = norm(first_res .- stats.residuals)
    if error > sqrt(eps())
        println(string("Residual mismatch between first and second run. Error = ", error))
    end

    # check objective values
    if length(objectives) != length(stats.quadras)
        throw(string("First and second run objectives do not have the same dimensions: ", length(first_res), " vs ", length(stats.residuals), "."))
    end
    error = norm(objectives .- stats.quadras)
    if error > sqrt(eps())
        println(string("Objective mismatch between first and second run. Error = ", error))
    end

    df = DataFrame(
        residuals = stats.residuals,
        objectives = stats.quadras,
        elapsed_time = elapsed_time
    )

    return df
end


function draw_plot(plot_path, plot_name, convex_dim, convex_cond, convex_data)

    # change default linewidth -> does not work
    # plt.rcParams["lines.linewidth"] = 0.25
    # println("New default linewidth: ", plt.rcParams["lines.linewidth"])

    sorted_keys = sort(collect(keys(convex_data)))

    fig, ax = plt.subplots(2, 2, figsize=(12, 8))

    convex_cond = @sprintf("%.3E", convex_cond)

    suptitle = string(plot_name, "\n dim = ", convex_dim, "; cond = ", convex_cond)

    fig.suptitle(suptitle)

    for key in sorted_keys
        dict = convex_data[key]
        residuals = dict.residuals
        objectives = dict.objectives
        elapsed_time = dict.elapsed_time

        ax[1, 1].plot(residuals, linewidth=0.5, label=key)
        ax[1, 2].plot(objectives, linewidth=0.5)
        ax[2, 1].plot(elapsed_time*1e3, residuals, linewidth=0.5)
        ax[2, 2].plot(elapsed_time*1e3, objectives, linewidth=0.5)

    end

    ax[1, 1].set_xlabel("Iteration")
    ax[1, 1].set_ylabel("Residuals")
    ax[1, 1].set_yscale("log")

    ax[1, 2].set_xlabel("Iteration")
    ax[1, 2].set_ylabel("Objective value")

    ax[2, 1].set_xlabel("Elapsed time (ms)")
    ax[2, 1].set_ylabel("Residuals")
    ax[2, 1].set_yscale("log")

    ax[2, 2].set_xlabel("Elapsed time (ms)")
    ax[2, 2].set_ylabel("Objective value")

    ax[1, 1].legend()

    plt.tight_layout()
    plt.savefig(plot_path, dpi=600)
    close(fig)

end


solvers = [
    (:cg, nothing, "cg"),
    (:diom, 0.25, "diom_0.25"),
    (:diom, 0.50, "diom_0.50"),
    (:diom, 0.75, "diom_0.75"),
    (:diom, 1.00, "diom_1.00"),
]


test_on_collection(100, solvers)