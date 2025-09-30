using Pkg
Pkg.activate("test_diom_tr")
# Pkg.develop(path="/home/corde/Krylov.jl")   # change path to local Krylov fork

# Pkg.add("LinearOperators")
# Pkg.add("JLD2")
# Pkg.add("Plots")

# Pkg.add("SuiteSparseMatrixCollection")
# Pkg.add("MatrixMarket")


using Krylov, LinearAlgebra, SparseArrays, Printf, Random, Test

using Printf, LinearOperators, Krylov
using JLD2, Plots

using  SuiteSparseMatrixCollection, MatrixMarket

using Profile

using SparseArrays

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

plt = plot(layout = (4, 3), size=(1600, 1200))


function elapsed_time_comparison(A, b, name, listmem, atol, rtol; compute_obj_flag=false, itmax=10*length(b), radius=0.0, fig_col=0)
  n = length(b)
  x0 = zeros(n)
  println("INIT obj= ", dot(b, x0) + 0.5*dot(x0, A*x0))

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

  start_time = time()

  if radius == 0.0
    (xcg, statscg) = cg(A, -b; atol=atol, rtol=rtol, callback=timing, history=true, itmax=itmax)
    println("elapsed time: ", time() - start_time)
    println("CG obj= ", dot(b, xcg) + 0.5*dot(xcg, A*xcg))
  else
    (xcg, statscg) = cg(A, -b; atol=atol, rtol=rtol, radius=radius, callback=timing, history=true, itmax=itmax)
    println("elapsed time: ", time() - start_time)
    println("CG obj= ", dot(b, xcg) + 0.5*dot(xcg, A*xcg))
  end

  gr()

  # plot for DIOM comparison
  plot!(plt[1, fig_col], statscg.residuals, label="cg", title="CG vs DIOM", lw=1, xlabel="Iterations", ylabel="‖r‖", yaxis=:log, linestyle = :dot, legend = :topright)
  plot!(plt[3, fig_col], elapsed_time*1.0e3, statscg.residuals, label="‖r‖ cg", lw=1, xlabel="Elapsed time (ms)", ylabel="‖r‖", yaxis=:log, linestyle = :dot, legend = :topright)

  if compute_obj_flag
    # objective plotted against iterations
    # plot!(plt[2, 1], obj, label="cg", lw=1, xlabel="Iterations", ylabel="Objective Value", legend = :topright)
    plot!(plt[2, fig_col], obj, label="cg", lw=1, xlabel="Iterations", ylabel="Objective Value", legend = :topright)

    # objective plotted against elapsed time
    # plot!(plt[4, 1], elapsed_time*1.0e3, obj, label="cg", lw=1, xlabel="Elapsed time (ms)", ylabel="Objective Value", legend = :topright)
    plot!(plt[4, fig_col], elapsed_time*1.0e3, obj, label="cg", lw=1, xlabel="Elapsed time (ms)", ylabel="Objective Value", legend = :topright)
  end

  # ------- DIOM -------
  for mem in listmem

    # mem_percent = round(mem/length(b)*100)

    obj = Float64[compute_obj(x0, A, b)]

    elapsed_time = Float64[0.0]

    start_time = time()

    if radius == 0.0
        (xdiom, statsdiom) = diom(A, -b; memory = mem, atol=atol, rtol=rtol, callback=timing, history=true, itmax=itmax)
    else
        (xdiom, statsdiom) = diom(A, -b; memory = mem, atol=atol, rtol=rtol, radius=radius, callback=timing, history=true, itmax=itmax)
    end

    plot!(plt[1, fig_col], statsdiom.residuals, label="diom $(m = mem)", lw=1, linestyle = :dash)
    plot!(plt[3, fig_col], elapsed_time*1.0e3, statsdiom.residuals, label="diom $(m = mem)", lw=1, linestyle = :dash)

    if compute_obj_flag
      plot!(plt[2, fig_col], obj, label="diom $(m = mem)", lw=1, legend = :topright)
      plot!(plt[4, fig_col], elapsed_time*1.0e3, obj, label="diom $(m = mem)", lw=1, legend = :topright)
    end

  end

  # if radius == 0
  #   savefig("subsolver_comparison/norad_cg_lbfgs_diom_comparison_$(name).png")
  # else
  #   savefig("subsolver_comparison/cg_lbfgs_diom_comparison_$(name).png")
  # end
  
end

T = Float64

radius = 20.0

matrices = ["494_bus", "1138_bus", "bcsstk16"]

fig_col = 0

for matrix in matrices
  # display matrix
  A = get_mm(matrix)
  # display(spy(A))

  global fig_col
  fig_col += 1

  # benchmark matrix on CG, L-BFGS and DIOM
  n,n = size(A)
  b = randn(eltype(A), n)
  atol = √eps(T)
  rtol = √eps(T)
  p=4
  listmem = memory(n, p)
  # println("listmem= ", listmem)
  listmem = [2, 20, 50, 100, 200]
  elapsed_time_comparison(A, b, matrix, listmem, atol, rtol, compute_obj_flag=true, itmax=5*n, radius=radius, fig_col=fig_col)
end

if radius == 0
  plt2 = plot(plt, layout=(4, 3), plot_title="No trust region constraint")
  savefig(plt2, "subsolver_comparison/cg_diom_comparison_no_tr_new.pdf")
else
  plt2 = plot(plt, layout=(4, 3), plot_title="With trust region constraint r = 20")
  savefig(plt2, "subsolver_comparison/cg_diom_comparison_with_tr_new.pdf")
end

