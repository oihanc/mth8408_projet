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

using Logging

import Krylov: krylov_workspace

include("subsolvers.jl")


# function LBFGSWorkspace(n::Int, m::Int=10)
#   println("did something")

# end

mutable struct LBFGSWorkspace2{T, FC <: T, S <: AbstractVector{T}} <: KrylovWorkspace{T, FC, S}
    n::Int
    m::Int
    x::AbstractVector{T}
    stats::Union{Nothing, LBFGSStats}

    function LBFGSWorkspace2{T,FC,S}(n::Int, m::Int) where {T,FC<:T,S<:AbstractVector{T}}
        x = zeros(T, n)                # auto-initialize vector
        stats = nothing          # assume you have a default constructor
        return new{T,FC,S}(n, m, x, stats)
    end
end

function krylov_workspace(::Val{:lbfgs}, n::Int, m::Int, V::Type{<:AbstractVector})
    println("this function was executed")
    return LBFGSWorkspace2{eltype(V), eltype(V), V}(n, m) #LBFGSWorkspace(n)   # LBFGSWorkspace{eltype(V), V}(n; m=10)
end

function Krylov.krylov_solve!(
    ws::LBFGSWorkspace2{T},   # your custom workspace type
    A::LinearOperator{T},     # the linear operator
    b::AbstractVector{T};     # rhs vector
    atol::T = sqrt(eps(T)),
    rtol::T = sqrt(eps(T)),
    radius::T = Inf,
    itmax::Int = 0,
    timemax::Float64 = Inf,
    verbose::Int = 0,
    M = I,
    kwargs...
) where {T}

    # Call your own solver (lbfgs_tr) inside
    x, stats = lbfgs_tr(
        A, -1 .* b;                # adapt as needed
        delta = radius,
        atol = atol,
        rtol = rtol,
        itmax = itmax,
        callback = (args...) -> nothing, # or pass through
        kwargs...
    )

    ws.x = x
    ws.stats = stats
    # return x, stats
end



f(x) = x[1]^2 * x[2]^2
x0 = ones(2)
nlp = ADNLPModel(f, x0)

println("--- L-BFGS ---")
solver = TrunkSolver(nlp, subsolver=:lbfgs)
stats = solve!(solver, nlp)
println(stats)
# trunk(nlp, subsolver=:cg)

println("--- CG ---")
solver = TrunkSolver(nlp, subsolver=:cg)
stats = solve!(solver, nlp)
println(stats)

println("--- L-BFGS ---")
solver = TrunkSolver(nlp, subsolver=:lbfgs)
stats = solve!(solver, nlp)
println(stats)
# trunk(nlp, subsolver=:cg)

