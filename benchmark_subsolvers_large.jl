


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

include("subsolvers.jl")
# include("TRSolver.jl")

DEBUG = false
EXE_PROBLEMS = true

nvars = 1000
nstart = 5
nprob = 1

meta = OptimizationProblems.meta

problem_list = meta[(meta.variable_nvar.==true).&(meta.ncon.==0).&.!meta.has_bounds.&(meta.minimize.==true), :name]

println("problem_list= ", problem_list)
println("length= ", length(problem_list))

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


function elapsed_time_comparison(A, b, name, listmem, atol, rtol)
  n = length(b)
  x0 = zeros(n)
  println("INIT obj= ", dot(b, x0) + 0.5*dot(x0, A*x0))

  plt = plot(layout = (3, 1), size=(800, 800))

  cg_elapsed_time = Float64[0.0]
  cg_obj = Vector{Float64}()
  start_time = time()

  function cg_timing(workspace)
    push!(cg_elapsed_time, time() - start_time)
    return false
  end

  

  (xcg, statscg) = cg(A, -b; atol=atol, rtol=rtol, callback=cg_timing, history=true)
  println("elapsed time: ", time() - start_time)
  println("CG obj= ", dot(b, xcg) + 0.5*dot(xcg, A*xcg))

  gr()  
  plot!(plt[1], statscg.residuals, label="‖r‖ cg ", lw=1, yaxis=:log, linestyle = :dot, xlabel="Iterations", legend = :bottomleft)
  plot!(plt[2], cg_elapsed_time*1.0e3, statscg.residuals, label="‖r‖ cg ", lw=1, yaxis=:log, linestyle = :dot, xlabel="Elapsed time", legend = :bottomleft)




  for mem in listmem

  lbfgs_elapsed_time = Vector{Float64}()
  lbfgs_residual = Vector{Float64}()
  start_time = time()

  function lbfgs_logger(ws)
    push!(lbfgs_elapsed_time, time() - start_time)
    push!(lbfgs_residual, ws.gnormk)
  end

    Hk = InverseLBFGSOperator(length(b), mem = mem, scaling = true)
    # start_time = time()
    (xlbfgs,statslbfgs) = lbfgs_tr(A, b; Hk, atol=atol, rtol=rtol, callback=lbfgs_logger) 
    println("elapsed time: ", time() - start_time)
    println("LBFGS obj= ", dot(b, xlbfgs) + 0.5*dot(xlbfgs, A*xlbfgs))

    # println(lbfgs_residual)

    # plot!(plt[1], statslbfgs.residuals, label="‖r‖ lbfgs $(m = 1)", lw=1, linestyle = :dash)
    plot!(plt[1], lbfgs_residual, label="‖r‖ lbfgs $(m = mem)", lw=1, linestyle = :dash)
    plot!(plt[2], lbfgs_elapsed_time*1.0e3, lbfgs_residual, label="‖r‖ lbfgs $(m = mem)", lw=1, linestyle = :dash)
  # plot!(plt[3], statslbfgs.elapsed_time*1.0e3, statslbfgs.objectives, label="‖r‖ lbfgs $(m = 1)", lw=1, linestyle = :dash)
  end 
  # for mem in listmem
  #   println("------- mem = ", mem, " -------")
  #   reset!(Hk)
  #   Hk = InverseLBFGSOperator(length(b), mem = mem, scaling = false)

  #   (xlbfgs,statslbfgs) = lbfgs_tr(A, b; Hk, atol=atol, rtol=rtol)  
  #   # p = round(Int64, 100 * mem / n)
  #   p = round(Int64, mem)

  #   plot!(plt[1], statslbfgs.residuals, label="‖r‖ lbfgs $mem", lw=1,linestyle = :dash)
  #   plot!(plt[2], statslbfgs.elapsed_time*1.0e3, statslbfgs.residuals, label="‖r‖ lbfgs $mem", lw=1,linestyle = :dash)
  #   # plot!(plt[3], statslbfgs.elapsed_time*1.0e3, statslbfgs.objectives, label="‖r‖ lbfgs $mem", lw=1,linestyle = :dash)

  #   # if p > 0 && p < 100
  #   #   plot!(statslbfgs.elapsed_time*1.0e3, statslbfgs.residuals, label="‖r‖ lbfgs $(m = p)%", lw=1,linestyle = :dash)
  #   # else
  #   #   plot!(statslbfgs.elapsed_time*1.0e3, statslbfgs.residuals, label="‖r‖ lbfgs $(m = p)%", lw=1,linestyle = :dot)
  #   # end
  # end
    
  savefig("figures_large/CG_versus_lbfgs_$(name)_time.png")
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

A = get_mm("494_bus")
# A = get_mm("1138_bus")
# A = get_mm("bcsstk16")
n,n = size(A)
b = randn(eltype(A), n)
atol = 1e-9
rtol = 1e-9
p=4
listmem = memory(n, p)
println("listmem= ", listmem)
# listmem = [1, 2, 5, 10, 20]
elapsed_time_comparison(A, b, "494_bus", listmem, atol, rtol)