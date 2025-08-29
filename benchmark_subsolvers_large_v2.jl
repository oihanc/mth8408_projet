


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

# Pkg.add("SuiteSparseMatrixCollection")
# Pkg.add("MatrixMarket")

# Pkg.add("Profile")
# Pkg.add("BenchmarkTools")

using LinearAlgebra, NLPModels , ADNLPModels, Printf, LinearOperators, Krylov
using OptimizationProblems, OptimizationProblems.ADNLPProblems, JSOSolvers, SolverTools, SolverCore, SolverBenchmark, NLPModelsIpopt
using JLD2, Plots

using  SuiteSparseMatrixCollection, MatrixMarket

using Profile, BenchmarkTools

# include("subsolvers.jl")
# include("TRSolver.jl")
include("TrunkSolverUpdate.jl")

DEBUG = false
EXE_PROBLEMS = true

nvars = 1000
nstart = 5
nprob = 1

meta = OptimizationProblems.meta

problem_list = meta[(meta.variable_nvar.==true).&(meta.ncon.==0).&.!meta.has_bounds.&(meta.minimize.==true), :name]

# println("problem_list= ", problem_list)
# println("length= ", length(problem_list))

# problems = (
#   eval(meta.parse("ADNLPProblems.$(pb[:name])()")) for pb in problem_list
# )

problems = (OptimizationProblems.ADNLPProblems.eval(Meta.parse(problem))(n=nvars) for problem ∈ problem_list[nstart:nstart+nprob])

# no minimize -> 79 with minimize -> 79 (too!)


# problems = nothing

# problem_to_exe = ["fletchcr", "nondquar", "woods", "broydn7d", "sparsine"]

# if DEBUG
#     problems = (OptimizationProblems.ADNLPProblems.eval(Meta.parse(problem))() for problem ∈ problem_to_exe)
# else
#     problems = (OptimizationProblems.ADNLPProblems.eval(Meta.parse(problem))() for problem ∈ problem_list)
# end


"""
      get_mm()
Charge une matrice depuis la SuiteSparse Matrix Collection
"""
function get_mm(matrix_name)
  ssmc = ssmc_db()
  pb = ssmc_matrices(ssmc, "", matrix_name)
  fetch_ssmc(pb, format="MM")
  pb_path = fetch_ssmc(pb, format="MM")
  path_mtx = pb_path[1]
  A = MatrixMarket.mmread(joinpath(path_mtx, matrix_name * ".mtx"))
  #b = MatrixMarket.mmread(joinpath(path_mtx, matrix_name * "_b.mtx"))
  return A
end

"""
      memory(n, p)
Génère des indices équidistants pour mémoire limitée
"""
function memory(n, p)
    @assert 1 ≤ p ≤ n "p doit être entre 1 et n"
    indices = [floor(Int, i*n/p) for i in 1:p]
    indices = unique(sort(indices))  
    return indices
end



function compute_obj(x, A, g)
  return dot(g, x) + 0.5 * dot(x, A*x)
end

function elapsed_time_comparison(A, b, name, listmem, atol, rtol; compute_obj_flag=false, itmax=10*length(b))
  n = length(b)
  x0 = zeros(n)
  println("INIT obj= ", dot(b, x0) + 0.5*dot(x0, A*x0))

  plt = plot(layout = (4, 2), size=(1200, 1200))

  elapsed_time = Float64[0.0]
  cg_obj = Vector{Float64}()
  start_time = time()

  obj = Float64[compute_obj(x0, A, b)]

  function timing(workspace)
    push!(elapsed_time, time() - start_time)
    if compute_obj_flag
      push!(obj, compute_obj(workspace.x, A, b))
    end

    return false
  end

  (xcg, statscg) = cg(A, -b; atol=atol, rtol=rtol, callback=timing, history=true, itmax=itmax)
  println("elapsed time: ", time() - start_time)
  println("CG obj= ", dot(b, xcg) + 0.5*dot(xcg, A*xcg))

  gr()
  # plot for LBFGS comparison
  plot!(plt[1, 1], statscg.residuals, label="cg", title="CG vs L-BFGS", lw=1, yaxis=:log, linestyle = :dot, xlabel="Iterations", ylabel="‖r‖", legend = :bottomleft)
  plot!(plt[3, 1], elapsed_time*1.0e3, statscg.residuals, label="‖r‖ cg ", lw=1, yaxis=:log, linestyle = :dot, xlabel="Elapsed time (ms)", ylabel="‖r‖", legend = :bottomleft)

  # plot for DIOM comparison
  plot!(plt[1, 2], statscg.residuals, label="cg", title="CG vs DIOM", lw=1, xlabel="Iterations", ylabel="‖r‖", yaxis=:log, linestyle = :dot, legend = :bottomleft)
  plot!(plt[3, 2], elapsed_time*1.0e3, statscg.residuals, label="‖r‖ cg", lw=1, xlabel="Elapsed time (ms)", ylabel="‖r‖", yaxis=:log, linestyle = :dot, legend = :bottomleft)

  if compute_obj_flag
    # objective plotted against iterations
    plot!(plt[2, 1], obj, label="cg", lw=1, xlabel="Iterations", ylabel="Objective Value", legend = :topright)
    plot!(plt[2, 2], obj, label="cg", lw=1, xlabel="Iterations", ylabel="Objective Value", legend = :topright)

    # objective plotted against elapsed time
    plot!(plt[4, 1], elapsed_time*1.0e3, obj, label="cg", lw=1, xlabel="Elapsed time (ms)", ylabel="Objective Value", legend = :topright)
    plot!(plt[4, 2], elapsed_time*1.0e3, obj, label="cg", lw=1, xlabel="Elapsed time (ms)", ylabel="Objective Value", legend = :topright)
  end

  # ------- L-BFGS -------
  for mem in listmem

    mem_percent = round(mem/length(b)*100)

    lbfgs_elapsed_time = Vector{Float64}()
    lbfgs_residual = Vector{Float64}()
    lbfgs_obj = Vector{Float64}()
    start_time = time()

    function lbfgs_logger(ws)
      push!(lbfgs_elapsed_time, time() - start_time)
      push!(lbfgs_residual, norm(ws.gk))
      push!(lbfgs_obj, compute_obj(ws.pk, A, b))
    end

      Hk = InverseLBFGSOperator(length(b), mem = mem, scaling = true)

      start_time = time()

      (xlbfgs, statslbfgs) = TrunkSolverUpdate.lbfgs_tr(A, b; mem=mem, atol=atol, rtol=rtol, callback=lbfgs_logger, itmax=itmax) 
      # println("elapsed time: ", time() - start_time)
      # println("LBFGS obj= ", dot(b, xlbfgs) + 0.5*dot(xlbfgs, A*xlbfgs))

      plot!(plt[1, 1], lbfgs_residual, label="lbfgs $(m = mem_percent)%", lw=1, linestyle = :dash)
      plot!(plt[3, 1], lbfgs_elapsed_time*1.0e3, lbfgs_residual, label="lbfgs $(m = mem_percent)%", lw=1, linestyle = :dash)

      if compute_obj_flag
        plot!(plt[2, 1], lbfgs_obj, label="lbfgs $(m = mem_percent)%", lw=1, legend = :topright)
        plot!(plt[4, 1], lbfgs_elapsed_time*1.0e3, lbfgs_obj, label="lbfgs $(m = mem_percent)%", lw=1, legend = :topright)
      end
  end 

  # ------- DIOM -------
  for mem in listmem

    mem_percent = round(mem/length(b)*100)

    obj = Float64[compute_obj(x0, A, b)]

    elapsed_time = Float64[0.0]

    start_time = time()

    (xdiom, statsdiom) = diom(A, -b; memory = mem, atol=atol, rtol=rtol, callback=timing, history=true, itmax=itmax)

    plot!(plt[1, 2], statsdiom.residuals, label="diom $(m = mem_percent)%", lw=1, linestyle = :dash)
    plot!(plt[3, 2], elapsed_time*1.0e3, statsdiom.residuals, label="diom $(m = mem_percent)", lw=1, linestyle = :dash)

    if compute_obj_flag
      plot!(plt[2, 2], obj, label="diom $(m = mem_percent)%", lw=1, legend = :topright)
      plot!(plt[4, 2], elapsed_time*1.0e3, obj, label="diom $(m = mem_percent)%", lw=1, legend = :topright)
    end

  end
    
  savefig("figures_large/cg_lbfgs_diom_comparison_$(name).png")
end


# for nlp in problems
#     A = hess_op(nlp, nlp.meta.x0)
#     b = grad(nlp, nlp.meta.x0)
#     n = length(b)

#     # A = get_mm("494_bus")
#     # n,n = size(A)
#     # b = randn(eltype(A), n)
#     atol = 1e-9
#     rtol = 1e-9
#     p=4
#     listmem = [1, 2 ,5]
    
#     elapsed_time_comparison(A, b, nlp.meta.name, listmem, atol, rtol)
# end

T = Float64

A = get_mm("494_bus")
# A = get_mm("1138_bus")
# A = get_mm("bcsstk16")
n,n = size(A)
b = randn(eltype(A), n)
atol = √eps(T)
rtol = √eps(T)
p=4
listmem = memory(n, p)
# println("listmem= ", listmem)
listmem = [2, 20, 50, 100, 200]
elapsed_time_comparison(A, b, "494_bus", listmem, atol, rtol, compute_obj_flag=true, itmax=5*n)

# 