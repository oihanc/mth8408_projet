using Pkg
Pkg.activate("test_diom_tr")

using SuiteSparseMatrixCollection
using MatrixMarket
using SparseArrays
using LinearAlgebra

using CSV
using DataFrames

using Krylov

using Arpack
using PyPlot
using Printf

using LinearOperators


mutable struct LBFGSStats{T}
    niter::Int
    residuals::Vector{T}
    quadras::Vector{T}
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
    stats::Union{Nothing, LBFGSStats{T}}

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
        stats = LBFGSStats{T}(0, T[], T[])     
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
    ws::LBFGSWorkspace{T,FC,S},
    A,
    b::AbstractVector{FC};
    radius::T = zero(T),
    atol::T = √eps(T),
    rtol::T = √eps(T),
    itmax::Int = 0,
    verbose::Int = 0,
    callback = ws -> false
) where {T<:AbstractFloat,FC<:T,S<:AbstractVector{FC}}

    reset!(ws.Hk)

    copyto!(ws.gk, -b)

    gnorm0_sq = dot(ws.gk, ws.gk)
    tol_sq = (atol + rtol*sqrt(gnorm0_sq))^2

    if itmax == 0
        itmax = 2ws.n
    end

    fill!(ws.pk, zero(T))

    #resize!(ws.stats.residuals, 0)
    callback(ws)
    push!(ws.stats.residuals, sqrt(gnorm0_sq))
    quadra = 0
    
    k = 0
    α = zero(T)
    gkd = zero(T)
    while true

        # d = -H g
        mul!(ws.dk, ws.Hk, ws.gk)
        @. ws.dk = -ws.dk

        # A d
        mul!(ws.bk, A, ws.dk)

        quadra += 0.5*α*gkd
        push!(ws.stats.quadras, quadra)

        dkbk = dot(ws.dk, ws.bk)
        gkd  = dot(ws.gk, ws.dk)

        if radius > zero(T) && dkbk <= zero(T)
            nd = norm(ws.dk)
            α = -sign(gkd) * 2radius / nd
        else
            α = -gkd / dkbk
        end
        
        # s = α d   (sans allocation)
        ws.sk .= α * ws.dk

        # p += s
        ws.pk .+= ws.sk

        if radius > zero(T)
            pk2 = dot(ws.pk, ws.pk)
            if pk2 >= radius^2
                @. ws.pk -= ws.sk
                pksk = dot(ws.pk, ws.sk)
                sk2  = dot(ws.sk, ws.sk)
                τ = (-pksk + sqrt(pksk^2 + sk2*(radius^2 - dot(ws.pk,ws.pk)))) / sk2
                return ws.pk .+ τ .* ws.sk, ws.stats
            end
        end

        # y = α A d
        ws.yk .= α * ws.bk

        # g += y
        ws.gk .+= ws.yk

        gnorm_sq = dot(ws.gk, ws.gk)
        k += 1

        if gnorm_sq <= tol_sq || k >= itmax
            break
        end

        push!(ws.stats.residuals, sqrt(gnorm_sq))
        push!(ws.Hk, ws.sk, ws.yk)

        copyto!(ws.x, ws.pk)

        callback(ws)
    end

    return ws.pk, ws.stats
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

    x, stats = lbfgs_tr!(ws, A,  b, radius=radius, atol=atol, rtol=rtol, itmax=itmax, verbose=verbose, callback=callback)

    ws.x = x
    ws.stats = stats
end

function lbfgs_workspace(::Val{:lbfgs}, n::Int, V::Type{<:AbstractVector}; mem=5, scaling=false)
    return LBFGSWorkspace{eltype(V), eltype(V), V}(n, mem; scaling=scaling)
end


function test_on_matrix(group, name, solvers, T;
                        precision_bits::Int = 0)
    ssmc = ssmc_db()

    ssmc = ssmc_db()
    ssmc_matrices(ssmc, group, name)

    # define and create saving directory
    if T == BigFloat
        if precision_bits == 0
            error("You must provide precision_bits for BigFloat.")
        end
        setprecision(BigFloat, precision_bits)
        precision_tag = "big$(precision(BigFloat))"
    elseif T == Float32
        precision_tag = "fp32"
    elseif T == Float64
        precision_tag = "fp64"
    else
        precision_tag = string(T)
    end

    directory = joinpath("benchmark_convex_2026", precision_tag)
    mkpath(directory)


    # download matrix only when it's its time to benchmark it
    
    # to chose 
    pname= string(group, "_", name)
    path = fetch_ssmc(group, name; format="MM")
    file = joinpath(path, name * ".mtx")



    # read matrix
    A = MatrixMarket.mmread(file)
    # A = MatrixMarket.mmread("matrices/mesh2em5/mesh2em5.mtx")
    A = Symmetric(sparse(T.(A)))   

    n, n = size(A)
    b = T.(ones(n)) #./ sqrt(T(n))
    convex_results = Dict()

    for sol in solvers
        
        solver = sol[1]
        mem_spec = sol[2]
        label_spec = sol[3]

        mem = mem_spec isa Function ? mem_spec(n) : mem_spec
        solver_label = label_spec isa Function ? label_spec(n) : label_spec

        println("solver = ", solver, " | mem = ", mem)

        df = benchmark_krylov(A, b, solver=solver, T=T, mem=mem)

        result_file = joinpath(directory,
            string(pname, "_", solver_label, ".csv"))
        CSV.write(result_file, df)

        convex_results[string(solver_label)] = df
    end
    prec_number =
    T == Float64  ? 64  :
    T == Float32  ? 32  :
    T == BigFloat ? precision(BigFloat) :
    0

    plot_path = joinpath(directory, "$(pname)_$(prec_number).pdf")

    plot_name = pname
    convex_dim = n
    convex_cond = cond(Array(Float64.(A)), 2)
    draw_plot(plot_path, plot_name, convex_dim, convex_cond, convex_results, T)

end



function collect_spd_matrices_all(;
        nmin::Int=100,
        nmax::Int=2000,
        kappa_max::Float64=1e10,
        max_matrices::Int=50,
        eig_tol::Float64=1e-6)

    println("Loading SuiteSparse metadata...")
    db = ssmc_db()

    # --- Filtrage metadata initial ---
    sel = db[
        (db.real .== true) .&
        (db.numerical_symmetry .== 1) .&
        (db.positive_definite .== true) .&
        (db.nrows .>= nmin) .&
        (db.nrows .<= nmax),
        :
    ]

    println("Metadata SPD candidates found: ", size(sel,1))

    results = []

    for row in eachrow(sel)

        try
            println("\nTesting $(row.group)/$(row.name)")

            # téléchargement
            path = fetch_ssmc(row.group, row.name; format="MM")
            file = joinpath(path, row.name * ".mtx")

            # lecture
            A = MatrixMarket.mmread(file)
            A = Symmetric(sparse(Float64.(A)))
            n = size(A,1)

            # estimation λmax
            λmax = eigs(A, nev=1, which=:LM, tol=eig_tol)[1][1]

            # estimation λmin
            λmin = eigs(A, nev=1, which=:SM, tol=eig_tol)[1][1]

            if λmin <= 0
                println("  -> rejected (not numerically SPD)")
                continue
            end

            κ = abs(λmax / λmin)

            @printf("  dim = %d | κ ≈ %.3e\n", n, κ)

            if κ < kappa_max
                push!(results, (
                    group = row.group,
                    name = row.name,
                    n = n,
                    kappa = κ
                ))
            end

            if length(results) >= max_matrices
                break
            end

        catch err
            println("  -> error: ", err)
        end
    end

    println("\nAccepted matrices: ", length(results))
    return results
end


function benchmark_krylov(A, b; solver = :cg, T, mem=nothing)

    n, n = size(A)
    nrmb = norm(b)

    if solver == :cg
        krylov_solver = krylov_workspace(Val(solver), A, b)
    elseif solver == :diom
        krylov_solver = krylov_workspace(Val(solver), A, b, memory=mem)
    elseif solver == :lbfgs
        n, n = size(A)
        krylov_solver = lbfgs_workspace(Val(solver), n, typeof(b), mem=mem, scaling=false)
    end


    # perform problem's first trial. Save objective values
    
    krylov_solve!(krylov_solver, A, b; itmax=2*n, history=true, atol=eltype(b)(1e-6), rtol=eltype(b)(1e-6)) # itmax=10*n

    println("krylov solve done")

    stats = krylov_solver.stats

    df = DataFrame(
        residuals = stats.residuals ./ nrmb,
        objectives = stats.quadras,
    )

    return df
end


function draw_plot(plot_path, plot_name, convex_dim, convex_cond, convex_data, T)

    # change default linewidth -> does not work
    # plt.rcParams["lines.linewidth"] = 0.25
    # println("New default linewidth: ", plt.rcParams["lines.linewidth"])
    if T == Float64
        prec_str = "Float64"
    elseif T == BigFloat
        prec_str = "BigFloat $(precision(BigFloat)) bits"
    else
        prec_str = string(T)
    end

    
    sorted_keys = sort(collect(keys(convex_data)))

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))

    convex_cond = @sprintf("%.3E", convex_cond)

    suptitle = string(
    plot_name,
    "\n \$n = ", convex_dim,
    "\$, \$\\kappa_2(A) = ", convex_cond,
    "\$, precision = ", prec_str
    )

    fig.suptitle(suptitle)
    for key in sorted_keys
        dict = convex_data[key]
        println("for", key,"type of residuals:", typeof(dict.residuals))
        println("for", key,"type of objectives: ", typeof(dict.objectives))

        residuals = Float64.(dict.residuals)
        objectives = Float64.(dict.objectives)

        ax[1].plot(residuals, linewidth=0.5, label=key)
        ax[2].plot(objectives, linewidth=0.5)
    end

    ax[1].set_xlabel(L"Iteration $k$")
    ax[1].set_ylabel(L"$\|g_k\|_2 / \|g_0\|_2$")
    ax[1].set_yscale("log")

    ax[2].set_xlabel(L"Iteration $k$")
    ax[2].set_ylabel(L"$f(x_k)$")

    ax[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=600)
    close(fig)

end


solvers = [
    (:cg, nothing, L"\mathrm{CG}"),
    (:diom, 50, L"\mathrm{DIOM}(m=50)"),
    (:diom, n -> Int(n), L"\mathrm{DIOM}(m=n)"),
    (:lbfgs, 25,  L"\mathrm{LBFGS}(m=25)"),
    (:lbfgs, 50, L"\mathrm{LBFGS}(m=50)"),
    (:lbfgs, n -> Int(n), L"\mathrm{LBFGS}(m=n)"),
]


# Test on matrcies with different precisions


#mats = collect_spd_matrices_all(
#    nmin = 100,
#    nmax = 3000,
#    kappa_max = 1e15,
#    max_matrices = 200
#)

#println("Running in precision ", Float64)
#for M in mats
#    test_on_matrix(M.group, M.name, solvers, Float64)
#end
# --------------------
#setprecision(BigFloat, 128)
#println("Running in precision ", BigFloat)
#for M in mats
#    test_on_matrix(M.group, M.name, solvers, BigFloat; precision_bits=128)
#end
# --------------------
#setprecision(BigFloat, 256)

#println("Running in precision ", BigFloat)
#for M in mats
#end


test_on_matrix("HB", "494_bus", solvers, Float64)
test_on_matrix("HB", "494_bus", solvers, BigFloat; precision_bits=128)
test_on_matrix("HB", "494_bus", solvers, BigFloat; precision_bits=256)

test_on_matrix("HB", "bcsstk05", solvers, Float64)
test_on_matrix("HB", "bcsstk05", solvers, BigFloat; precision_bits=128)
test_on_matrix("HB", "bcsstk05", solvers, BigFloat; precision_bits=256)

test_on_matrix("HB", "bcsstm09", solvers, Float64)
test_on_matrix("HB", "bcsstm09", solvers, BigFloat; precision_bits=128)
test_on_matrix("HB", "bcsstm09", solvers, BigFloat; precision_bits=256)

test_on_matrix("HB", "gr_30_30", solvers, Float64)
test_on_matrix("HB", "gr_30_30", solvers, BigFloat; precision_bits=128)
test_on_matrix("HB", "gr_30_30", solvers, BigFloat; precision_bits=256)