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


# Pkg.add("SparseArrays")

using LinearAlgebra, NLPModels , ADNLPModels, Printf, LinearOperators, Krylov
using OptimizationProblems, OptimizationProblems.ADNLPProblems, JSOSolvers, SolverTools, SolverCore, SolverBenchmark, NLPModelsIpopt
using JLD2, Plots

using  SuiteSparseMatrixCollection, MatrixMarket

using Profile

using SparseArrays

include("TrunkSolverUpdate.jl")

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

# compute quadratic objective
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
  plot!(plt[1, 1], statscg.residuals, label="cg", title="CG vs L-BFGS", lw=1, yaxis=:log, linestyle = :dot, xlabel="Iterations", ylabel="‖r‖", legend = :topright)
  plot!(plt[3, 1], elapsed_time*1.0e3, statscg.residuals, label="‖r‖ cg ", lw=1, yaxis=:log, linestyle = :dot, xlabel="Elapsed time (ms)", ylabel="‖r‖", legend = :topright)

  # plot for DIOM comparison
  plot!(plt[1, 2], statscg.residuals, label="cg", title="CG vs DIOM", lw=1, xlabel="Iterations", ylabel="‖r‖", yaxis=:log, linestyle = :dot, legend = :topright)
  plot!(plt[3, 2], elapsed_time*1.0e3, statscg.residuals, label="‖r‖ cg", lw=1, xlabel="Elapsed time (ms)", ylabel="‖r‖", yaxis=:log, linestyle = :dot, legend = :topright)

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

    # mem_percent = round(mem/length(b)*100, digits=3)

    lbfgs_elapsed_time = Vector{Float64}()
    lbfgs_residual = Vector{Float64}()
    lbfgs_obj = Vector{Float64}()
    start_time = time()

    function lbfgs_logger(ws)
      push!(lbfgs_elapsed_time, time() - start_time)
      push!(lbfgs_residual, norm(ws.gk))
      push!(lbfgs_obj, compute_obj(ws.pk, A, b))
    end

      start_time = time()

      (xlbfgs, statslbfgs) = TrunkSolverUpdate.lbfgs_tr(A, b; mem=mem, atol=atol, rtol=rtol, callback=lbfgs_logger, itmax=itmax) 
      # println("elapsed time: ", time() - start_time)
      # println("LBFGS obj= ", dot(b, xlbfgs) + 0.5*dot(xlbfgs, A*xlbfgs))

      plot!(plt[1, 1], lbfgs_residual, label="lbfgs $(m = mem)", lw=1, linestyle = :dash)
      plot!(plt[3, 1], lbfgs_elapsed_time*1.0e3, lbfgs_residual, label="lbfgs $(m = mem)", lw=1, linestyle = :dash)

      if compute_obj_flag
        plot!(plt[2, 1], lbfgs_obj, label="lbfgs $(m = mem)", lw=1, legend = :topright)
        plot!(plt[4, 1], lbfgs_elapsed_time*1.0e3, lbfgs_obj, label="lbfgs $(m = mem)", lw=1, legend = :topright)
      end
  end 

  # ------- DIOM -------
  for mem in listmem

    # mem_percent = round(mem/length(b)*100)

    obj = Float64[compute_obj(x0, A, b)]

    elapsed_time = Float64[0.0]

    start_time = time()

    (xdiom, statsdiom) = diom(A, -b; memory = mem, atol=atol, rtol=rtol, callback=timing, history=true, itmax=itmax)

    plot!(plt[1, 2], statsdiom.residuals, label="diom $(m = mem)", lw=1, linestyle = :dash)
    plot!(plt[3, 2], elapsed_time*1.0e3, statsdiom.residuals, label="diom $(m = mem)", lw=1, linestyle = :dash)

    if compute_obj_flag
      plot!(plt[2, 2], obj, label="diom $(m = mem)", lw=1, legend = :topright)
      plot!(plt[4, 2], elapsed_time*1.0e3, obj, label="diom $(m = mem)", lw=1, legend = :topright)
    end

  end
    
  savefig("subsolver_comparison/cg_lbfgs_diom_comparison_$(name).png")
end

T = Float64

matrices = ["494_bus", "1138_bus", "bcsstk16"]

for matrix in matrices
  # display matrix
  A = get_mm(matrix)
  display(spy(A))

  # benchmark matrix on CG, L-BFGS and DIOM
  n,n = size(A)
  b = randn(eltype(A), n)
  atol = √eps(T)
  rtol = √eps(T)
  p=4
  listmem = memory(n, p)
  # println("listmem= ", listmem)
  listmem = [2, 20, 50, 100, 200]
  elapsed_time_comparison(A, b, matrix, listmem, atol, rtol, compute_obj_flag=true, itmax=5*n)
end