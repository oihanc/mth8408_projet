


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

DEBUG = true
EXE_PROBLEMS = true

meta = OptimizationProblems.meta
problem_list = meta[(meta.ncon.==0).&.!meta.has_bounds.&(meta.nvar.==100), :name]
problems = nothing

problem_to_exe = ["fletchcr", "nondquar", "woods", "broydn7d", "sparsine"]

if DEBUG
    problems = (OptimizationProblems.ADNLPProblems.eval(Meta.parse(problem))() for problem ∈ problem_to_exe)
else
    problems = (OptimizationProblems.ADNLPProblems.eval(Meta.parse(problem))() for problem ∈ problem_list)
end


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
  
  plt = plot(layout = (2, 1), size=(800, 800))


  cg_elapsed_time = Float64[0.0]
  start_time = time()

  function cg_timing(workspace)
    push!(cg_elapsed_time, time() - start_time)
    return false
  end
  

  (xcg, statscg) = cg(A, b; atol=atol, rtol=rtol, callback=cg_timing, history=true)
  println("elapsed time: ", time() - start_time)

  gr()  
  plot!(plt[1], statscg.residuals, label="‖r‖ cg ", lw=1, yaxis=:log, linestyle = :dot, xlabel="Iterations", legend = :bottomleft)
  plot!(plt[2], cg_elapsed_time*1.0e3, statscg.residuals, label="‖r‖ cg ", lw=1, yaxis=:log, linestyle = :dot, xlabel="Elapsed time", legend = :bottomleft)

  Hk = InverseLBFGSOperator(length(b), mem = 1, scaling = true)
  start_time = time()
  (xlbfgs,statslbfgs) = lbfgs_tr(A, b; Hk, atol=atol, rtol=rtol) 
  println("elapsed time: ", time() - start_time)

  plot!(plt[1], statslbfgs.residuals, label="‖r‖ lbfgs $(m = 1)", lw=1, linestyle = :dash)
  plot!(plt[2], statslbfgs.elapsed_time*1.0e3, statslbfgs.residuals, label="‖r‖ lbfgs $(m = 1)", lw=1, linestyle = :dash)

  for mem in listmem
    reset!(Hk)
    Hk = InverseLBFGSOperator(length(b), mem = mem, scaling = true)

    (xlbfgs,statslbfgs) = lbfgs_tr(A, b; Hk, atol=atol, rtol=rtol)  
    p = round(Int64, 100 * mem / n)

    plot!(plt[1], statslbfgs.residuals, label="‖r‖ lbfgs $mem", lw=1,linestyle = :dash)
    plot!(plt[2], statslbfgs.elapsed_time*1.0e3, statslbfgs.residuals, label="‖r‖ lbfgs $mem", lw=1,linestyle = :dash)

    # if p > 0 && p < 100
    #   plot!(statslbfgs.elapsed_time*1.0e3, statslbfgs.residuals, label="‖r‖ lbfgs $(m = p)%", lw=1,linestyle = :dash)
    # else
    #   plot!(statslbfgs.elapsed_time*1.0e3, statslbfgs.residuals, label="‖r‖ lbfgs $(m = p)%", lw=1,linestyle = :dot)
    # end
  end
    
  savefig("figures/CG_versus_lbfgs_$(name)_time.pdf")
end


for nlp in problems
    A = hess_op(nlp, nlp.meta.x0)
    b = grad(nlp, nlp.meta.x0)
    n = length(b)

    # A = get_mm("494_bus")
    # n,n = size(A)
    # b = randn(eltype(A), n)
    atol = 1e-9
    rtol = 1e-9
    p=4
    listmem = [1, 2, 5, 10, 20]
    
    elapsed_time_comparison(A, b, nlp.meta.name, listmem, atol, rtol)
end

A = get_mm("494_bus")
n,n = size(A)
b = randn(eltype(A), n)
atol = 1e-9
rtol = 1e-9
p=4
# listmem = memory(n, p)
listmem = memory(n, p)
elapsed_time_comparison(A, b, "494_bus", listmem, atol, rtol)