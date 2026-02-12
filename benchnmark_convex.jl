using Pkg
Pkg.activate("test_diom_tr")

using SuiteSparseMatrixCollection
using MatrixMarket
using SparseArrays
using LinearAlgebra

using CSV
using DataFrames

using Krylov

using PyPlot
using Printf

using LinearOperators


mutable struct LBFGSStats
    niter::Int8
    residuals::Vector{Float64}
    elapsed_time::Vector{Float64}
end


mutable struct LBFGSWorkspace{T, FC <: T, S <: AbstractVector{T}} <: KrylovWorkspace{T, FC, S}
    n::Int      # dimension
    m::Int
    mem::Int    # memory
    Hk  # Inverse LBFGS Operator
    x0::AbstractVector{T}   # starting point
    pk::AbstractVector{T}   # current point
    gk::AbstractVector{T}   # gradient
    dk::AbstractVector{T}   # search direction
    bk::AbstractVector{T}
    sk::AbstractVector{T}   # step update
    yk::AbstractVector{T}   # gradient update
    x::AbstractVector{T}    # solution vector
    stats::Union{Nothing, LBFGSStats}

    function LBFGSWorkspace{T,FC,S}(n::Int, mem::Int; scaling::Bool=false) where {T, FC<:T, S<:AbstractVector{T}}
        Hk = InverseLBFGSOperator(T, n, mem=mem; scaling=scaling)
        x0 = zeros(T, n)
        pk = zeros(T, n)
        gk = zeros(T, n)
        dk = zeros(T, n)
        bk = zeros(T, n)
        sk = zeros(T, n)
        yk = zeros(T, n)
        x = zeros(T, n)
        stats = LBFGSStats(0, ones(Float64, 1), zeros(Float64, 1))     # Review
        return new{T,FC,S}(n, n, mem, Hk, x0, pk, gk, dk, bk, sk, yk, x, stats)
    end
end

function lbfgs_tr end

function lbfgs_tr! end


function lbfgs_tr(
    A::Union{AbstractLinearOperator, Any},
    b::AbstractVector{T}; 
    mem::Int=2,
    radius::T=zero(T),
    atol::T=√eps(T),
    rtol::T=√eps(T),
    itmax::Int=0,
    verbose::Int=0,
    callback=workspace -> false,
    ) where {T}
    
    workspace = LBFGSWorkspace{eltype(b), eltype(b), AbstractVector{eltype(b)}}(length(b), mem)
    return lbfgs_tr!(workspace, A, b; radius, atol, rtol, itmax, verbose, callback)
end


function lbfgs_tr!(
    ws :: LBFGSWorkspace{T, FC, S}, 
    A, 
    b::AbstractVector{FC}; 
    radius::T=zero(T), 
    atol::T=√eps(T), 
    rtol::T=√eps(T), 
    itmax::Int=0, 
    verbose::Int=0, 
    callback=workspace -> false) where {T<:AbstractFloat, FC<:T, S<:AbstractVector{FC}}
    
    reset!(ws.Hk)

    copyto!(ws.gk, b)
    gnorm0 = gnormk = norm(ws.gk)

    tolerance = atol + rtol*gnorm0

    if itmax == 0
        itmax = 2*ws.n
    end

    fill!(ws.pk, zero(T))
    alphak = zero(T)

    callback(ws)

    k = 1
    done = false

    while !done

        push!(ws.stats.residuals, gnormk)

        mul!(ws.dk, -ws.Hk, ws.gk)  # compute search direction
        mul!(ws.bk, A, ws.dk)       # compute curvature

        dkbk = dot(ws.dk, ws.bk)

        # handle negative curvature
        if radius > 0.0 && dkbk <= 0
            alphak = -sign(dot(ws.gk, ws.dk))*2*radius/norm(ws.dk)
        else
            alphak = -dot(ws.gk, ws.dk)/dkbk    # compute search step
        end
        
        ws.sk .= alphak .* ws.dk    # scale search direction
        ws.pk .+= ws.sk             # update current point

        if radius > 0 && norm(ws.pk) >= radius
            ws.pk .-= ws.sk
            pksk = dot(ws.pk, ws.sk)
            sksk = dot(ws.sk, ws.sk)

            tau = (-pksk + sqrt(pksk^2 + sksk*(radius^2 - dot(ws.pk, ws.pk))))/sksk

            return ws.pk .+ tau .* ws.sk, nothing # TODO: add SimpleStats
        end
        
        ws.yk .= alphak .* ws.bk    # compute gradient update
        ws.gk .+= ws.yk             # update gradient
        
        gnormk = norm(ws.gk)
        k += 1

        if gnormk <= tolerance
            done = true
        elseif k >= itmax
            done = true
        end

        # update the inverse Hessian approximation
        if !done
            push!(ws.Hk, ws.sk, ws.yk)
        end

        ws.x .= ws.pk
        callback(ws)
    end

    push!(ws.stats.residuals, gnormk)

    return ws.pk, ws.stats # TODO: add SimpleStats
end


import Krylov: krylov_solve!

function krylov_solve!(
    ws, # ::LBFGSWorkspace{T},
    A, #::LinearOperator{T},
    b::AbstractVector{T};
    atol::T = √eps(T),
    rtol::T = √eps(T),
    radius::T = zero(T),
    itmax::Int = 0,
    timemax::Float64 = Inf,
    verbose::Int = 0,
    callback = ws -> false,
    kwargs...
) where {T}

    x, stats = lbfgs_tr!(ws, A, -1 .* b, radius=radius, atol=atol, rtol=rtol, itmax=itmax, verbose=verbose, callback=callback)

    ws.x = x
    ws.stats = stats
end

function lbfgs_workspace(::Val{:lbfgs}, n::Int, V::Type{<:AbstractVector}; mem=5, scaling=false)
    return LBFGSWorkspace{eltype(V), eltype(V), V}(n, mem; scaling=scaling)
end

# println(methods(krylov_solve!))


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
    directory = "benchmark_convex_2026/"
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

            # try
                # perform blank run for the first iteration for accurate elapsed time measurement
                if i == 1
                    _ = benchmark_krylov(A, b, solver=solver, mem=mem)
                end

                df = benchmark_krylov(A, b, solver=solver, mem=mem)

                # save data to .csv file
                result_file = joinpath(directory, string(pname, "_", solver_name, ".csv"))
                CSV.write(result_file, df)

                convex_results[solver_name] = df

            # catch err
            #     # println("Caught error thrown at $(err.file):$(err.line)")
            #     println("error ", err)
            # end
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
    function compute_obj(x, A, g)
        return dot(g, x) + 0.5 * dot(x, A*x)
    end

    function objective_callback(workspace)
        push!(objectives, compute_obj(workspace.x, A, b))
        timing_callback(workspace)
        return false
    end

    
    function timing_callback(workspace)
        push!(elapsed_time, time() - start_time)
        return false
    end

    if solver == :cg
        krylov_solver = krylov_workspace(Val(solver), A, -b)
    elseif solver == :diom
        krylov_solver = krylov_workspace(Val(solver), A, -b, memory=mem)
    elseif solver == :lbfgs
        n, n = size(A)
        krylov_solver = lbfgs_workspace(Val(solver), n, typeof(b), mem=mem, scaling=false)
    end

    start_time = time()

    # perform problem's first trial. Save objective values
    krylov_solve!(krylov_solver, A, -b; itmax=n, history=true, callback=objective_callback) # itmax=10*n

    println("krylov solve done")

    stats = krylov_solver.stats

    first_res = copy(stats.residuals)
    
    if solver == :cg
        krylov_solver = krylov_workspace(Val(solver), A, -b)
    elseif solver == :diom
        krylov_solver = krylov_workspace(Val(solver), A, -b, memory=mem)
    elseif solver == :lbfgs
        n, n = size(A)
        krylov_solver = lbfgs_workspace(Val(solver), n, typeof(b), mem=mem)
    end

    elapsed_time = Float64[0.0]
    start_time = time()

    # perform peroblem' second trial. Measure elapsed time. Do not save objective values.
    krylov_solve!(krylov_solver, A, -b; itmax=n, history=true, callback=timing_callback)
    stats = krylov_solver.stats

    if length(first_res) != length(stats.residuals)
        throw(string("First and second run residuals do not have the same dimensions: ", length(first_res), " vs ", length(stats.residuals), "."))
    end
    error = norm(first_res .- stats.residuals)
    if error > sqrt(eps())
        throw("Residual mismatch between first and second run.")
    end

    df = DataFrame(
        residuals = stats.residuals,
        objectives = objectives,
        elapsed_time = elapsed_time
    )

    return df
end


function draw_plot(plot_path, plot_name, convex_dim, convex_cond, convex_data)

    # change default linewidth -> does not work
    # plt.rcParams["lines.linewidth"] = 0.25
    # println("New default linewidth: ", plt.rcParams["lines.linewidth"])

    sorted_keys = sort(collect(keys(convex_data)))

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))

    convex_cond = @sprintf("%.3E", convex_cond)

    suptitle = string(plot_name, "\n dim = ", convex_dim, "; cond = ", convex_cond)

    fig.suptitle(suptitle)

    for key in sorted_keys
        dict = convex_data[key]
        residuals = dict.residuals
        objectives = dict.objectives
        elapsed_time = dict.elapsed_time

        ax[1].plot(residuals, linewidth=0.5, label=key)
        ax[2].plot(objectives, linewidth=0.5)
        # ax[2, 1].plot(elapsed_time*1e3, residuals, linewidth=0.5)
        # ax[2, 2].plot(elapsed_time*1e3, objectives, linewidth=0.5)

    end

    ax[1].set_xlabel("Iteration")
    ax[1].set_ylabel("Residuals")
    ax[1].set_yscale("log")

    ax[2].set_xlabel("Iteration")
    ax[2].set_ylabel("Objective value")

    # ax[2, 1].set_xlabel("Elapsed time (ms)")
    # ax[2, 1].set_ylabel("Residuals")
    # ax[2, 1].set_yscale("log")

    # ax[2, 2].set_xlabel("Elapsed time (ms)")
    # ax[2, 2].set_ylabel("Objective value")

    ax[1].legend()

    plt.tight_layout()
    plt.savefig(plot_path, dpi=600)
    close(fig)

end


solvers = [
    (:cg, nothing, "cg"),
    (:diom, 100, "diom_100"),
    (:diom, 200, "diom_200"),
    (:lbfgs, 50, "lbfgs_50"),
    (:lbfgs, 100, "lbfgs_100"),
    (:lbfgs, 200, "lbfgs_200"),
]

test_on_collection(3, solvers)