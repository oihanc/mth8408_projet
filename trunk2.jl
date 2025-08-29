using Pkg
Pkg.activate("projet_env")

# Pkg.add("ADNLPModels")
# Pkg.add("NLPModels")
# Pkg.add("Krylov")
# Pkg.add("LinearOperators")
Pkg.add("JSOSolvers")
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


abstract type AbstractSubsolverWorkspace{T, V<:AbstractVector{T}} end


# mutable struct TrunkSolver{
#   T,
#   V <: AbstractVector{T},
#   Sub <: AbstractSubsolverWorkspace{T,V},          # <— generalized
#   Op  <: AbstractLinearOperator{T},
# } <: AbstractOptimizationSolver
#   x::V; xt::V
#   gx::V; gt::V
#   gn::V; Hs::V
#   subsolver::Sub
#   H::Op
#   tr::TrustRegion{T,V}
#   params::TRUNKParameterSet
# end

struct MyTrunkSolver{
  T,
  V <: AbstractVector{T},
  Sub <: AbstractSubsolverWorkspace{T,V},          # <— generalized
  Op  <: AbstractLinearOperator{T},
} <: JSOSolvers.TrunkSolver
  x::V; xt::V
  gx::V; gt::V
  gn::V; Hs::V
  subsolver::Sub
  H::Op
  tr::JSOSolvers.TrustRegion{T,V}
  params::JSOSolvers.TRUNKParameterSet
end

mutable struct TrunkSolver{
  T,
  V <: AbstractVector{T},
  Sub <: AbstractSubsolverWorkspace{T,V},          # <— generalized
  Op  <: AbstractLinearOperator{T},
} <: AbstractOptimizationSolver
  x::V; xt::V
  gx::V; gt::V
  gn::V; Hs::V
  subsolver::Sub
  H::Op
  tr::TrustRegion{T,V}
  params::TRUNKParameterSet
end



# # init_subsolver!(::Val, n; kwargs...) = nothing





# function solve_subproblem!(::AbstractSubsolverWorkspace; kwargs...)
#     error("solve_subproblem! not implemented for $(typeof(state))")
# end


# # struct LBFGSState{T,V<:AbstractVector{T}} <: AbstractSubsolverState{T,V}
# #     m::Int                               # memory size
# #     s::Vector{V}                         # last m step vectors
# #     y::Vector{V}                         # last m gradient diffs
# #     ρ::Vector{T}                         # 1/(y⋅s)
# #     α::Vector{T}                         # scratch
# #     q::V                                 # scratch
# #     r::V                                 # scratch
# # end

# # function LBFGSState{T,V}(n::Int; m::Int=10) where {T,V<:AbstractVector{T}}
# #     s = V[]; y = V[]
# #     ρ = T[]; α = zeros(T, m)
# #     q = zeros(T, n); r = similar(q)
# #     return LBFGSState{T,V}(m, s, y, ρ, α, q, r)
# # end


# mutable struct LBFGSWorkspace{T, V <: AbstractVector{T}} <: AbstractSubsolverWorkspace{T, V}
#     gk::Vector{T}
#     pk::Vector{T}
#     dk::Vector{T}
#     bk::Vector{T}
#     sk::Vector{T}
#     yk::Vector{T}
#     residuals::Vector{T}
#     gnormk::T
# end

# function LBFGSWorkspace{T, V}(dim::Int, ::Type{T}=Float64) where {T, V <: AbstractVector{T}}
#     LBFGSWorkspace(
#         zeros(T, dim),
#         zeros(T, dim),
#         zeros(T, dim),
#         zeros(T, dim),
#         zeros(T, dim),
#         zeros(T, dim),
#         Float64[],           # make it more generic
#         0.0
#     )
# end


# # function TrunkSolver(x0::V, H::Op, tr::TrustRegion{T,V}, params::TRUNKParameterSet;
# #                      inner = KrylovWorkspace{T,V}(...)) where {T,V<:AbstractVector{T},Op<:AbstractLinearOperator{T}}
# #     # allocate working vectors xt,gx,gt,gn,Hs, etc.
# #     sub = inner isa AbstractSubsolverState ? inner : error("inner must be a subsolver state")
# #     return TrunkSolver{T,V,typeof(sub),Op}(x, xt, gx, gt, gn, Hs, sub, H, tr, params)
# # end

function init_subsolver(::Val{:cg}, nvar::Int, V::Type{<:AbstractVector})
    return krylov_workspace(Val(:cg), nvar, nvar, V)
end

# function init_subsolver(::Val{:lbfgs}, nvar::Int, V::Type{<:AbstractVector})
#     return LBFGSWorkspace(nvar)
# end



JSOSolver.TrunkSolver(nlp::AbstractNLPModel{T, V};
  bk_max::Int = get(TRUNK_bk_max, nlp),
  monotone::Bool = get(TRUNK_monotone, nlp),
  nm_itmax::Int = get(TRUNK_nm_itmax, nlp),
  subsolver::Symbol = :cg,
) where {T, V <: AbstractVector{T}} = begin
params = TRUNKParameterSet(nlp; bk_max = bk_max, monotone = monotone, nm_itmax = nm_itmax)
  nvar = nlp.meta.nvar
  x = V(undef, nvar)
  xt = V(undef, nvar)
  gx = V(undef, nvar)
  gt = V(undef, nvar)
  gn = isa(nlp, QuasiNewtonModel) ? V(undef, nvar) : V(undef, 0)
  Hs = V(undef, nvar)
  krylov_subsolver = init_subsolver(Val(subsolver), nvar, V)
  Sub = typeof(krylov_subsolver)
  H = hess_op!(nlp, x, Hs)
  Op = typeof(H)
  tr = TrustRegion(gt, one(T))
  return TrunkSolver{T, V, Sub, Op}(x, xt, gx, gt, gn, Hs, krylov_subsolver, H, tr, params)
end


f(x) = x[1]^2 * x[2]^2
x0 = ones(2)
nlp = ADNLPModel(f, x0)

solver = TrunkSolver(nlp, subsolver=:cg)

stats = solve!(solver, nlp)


# trunk(nlp, subsolver=:cg)

