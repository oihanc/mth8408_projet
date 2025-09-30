using Pkg
Pkg.activate("projet_env")

"""
The following script is meant to leverage the trust region implementation `trunk` in JSOSolver and 
make it compatible with a L-BFGS and DIOM trust region quadratic solver
"""

# TODO: add documentation (doc-string)


module TrunkSolverUpdate

using LinearAlgebra
using LinearOperators
using Krylov
using JSOSolvers
using SolverTools
using NLPModels
using ADNLPModels

Base.include(Krylov, "diom_tr.jl")


mutable struct LBFGSStats
    niter::Int
    residuals::Vector{Float64}
    elapsed_time::Vector{Float64}
end


mutable struct LBFGSWorkspace{T, FC <: T, S <: AbstractVector{T}} <: KrylovWorkspace{T, FC, S}
    n::Int      # dimension
    m::Int
    mem::Int    # memory
    Hk::AbstractLinearOperator  # Inverse LBFGS Operator
    x0::AbstractVector{T}   # starting point
    pk::AbstractVector{T}   # current point
    gk::AbstractVector{T}   # gradient
    dk::AbstractVector{T}   # search direction
    bk::AbstractVector{T}
    sk::AbstractVector{T}   # step update
    yk::AbstractVector{T}   # gradient update
    x::AbstractVector{T}    # solution vector

    stats::Union{Nothing, LBFGSStats}

    function LBFGSWorkspace{T,FC,S}(n::Int, mem::Int; scaling::Bool=true) where {T, FC<:T, S<:AbstractVector{T}}
        Hk = InverseLBFGSOperator(T, n, mem=mem; scaling=scaling)
        x0 = zeros(T, n)
        pk = zeros(T, n)
        gk = zeros(T, n)
        dk = zeros(T, n)
        bk = zeros(T, n)
        sk = zeros(T, n)
        yk = zeros(T, n)
        x = zeros(T, n)
        stats = nothing     # Review
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
    callback=workspace -> nothing,
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

        callback(ws)
    end

    return ws.pk, nothing # TODO: add SimpleStats
end


function Krylov.krylov_solve!(
    ws::LBFGSWorkspace{T},
    A::LinearOperator{T},
    b::AbstractVector{T};
    atol::T = √eps(T),
    rtol::T = √eps(T),
    radius::T = zero(T),
    itmax::Int = 0,
    timemax::Float64 = Inf,
    verbose::Int = 0,
    kwargs...
) where {T}

    x, stats = lbfgs_tr!(ws, A, -1 .* b, radius=radius, atol=atol, rtol=rtol, itmax=itmax, verbose=verbose)

    ws.x = x
    ws.stats = stats
end

function lbfgs_workspace(::Val{:lbfgs}, n::Int, V::Type{<:AbstractVector}; mem=5, scaling=true)
    return LBFGSWorkspace{eltype(V), eltype(V), V}(n, mem; scaling=scaling)
end


function Krylov.diom(A::LinearOperator, b::AbstractVector{T}; memory::Int = 20, radius::T=zero(T), atol::T=√eps(T), rtol::T=√eps(T), itmax::Int=0, verbose::Int=0, callback=workspace -> false) where {T <: AbstractFloat}
    workspace = DiomTRWorkspace(A, b; memory=memory)
    diom!(workspace, A, b, radius=radius, atol=atol, rtol=rtol, itmax=itmax, verbose=verbose)
    return workspace.x, workspace
end

function Krylov.krylov_solve!(
    ws::DiomTRWorkspace{T},
    A::LinearOperator{T},
    b::AbstractVector{T};
    atol::T = √eps(T),
    rtol::T = √eps(T),
    radius::T = zero(T),
    itmax::Int = 0,
    timemax::Float64 = Inf,   # TODO
    verbose::Int = 0,
    kwargs...
) where {T}

    diom!(ws, A, b, radius=radius, atol=atol, rtol=rtol, itmax=itmax, verbose=verbose)
end


function TrunkSolver(
  nlp::AbstractNLPModel{T, V};
  bk_max::Int = get(JSOSolvers.TRUNK_bk_max, nlp),
  monotone::Bool = get(JSOSolvers.TRUNK_monotone, nlp),
  nm_itmax::Int = get(JSOSolvers.TRUNK_nm_itmax, nlp),
  subsolver::Symbol = :cg,
  mem::Int=5,
  scaling::Bool=true,
) where {T, V <: AbstractVector{T}}
  params = TRUNKParameterSet(nlp; bk_max = bk_max, monotone = monotone, nm_itmax = nm_itmax)
  nvar = nlp.meta.nvar
  x = V(undef, nvar)
  xt = V(undef, nvar)
  gx = V(undef, nvar)
  gt = V(undef, nvar)
  gn = isa(nlp, JSOSolvers.QuasiNewtonModel) ? V(undef, nvar) : V(undef, 0)
  Hs = V(undef, nvar)

  krylov_subsolver = 
    if subsolver == :cg
      krylov_workspace(Val(subsolver), nvar, nvar, V)
    elseif subsolver == :lbfgs
      lbfgs_workspace(Val(subsolver), nvar, V; mem=mem, scaling=scaling)
    elseif subsolver == :diom
      DiomTRWorkspace(nlp.meta.x0; memory=mem)
    else
      throw("Not a valid subsolver choice.")
    end
  
  Sub = typeof(krylov_subsolver)
  H = hess_op!(nlp, x, Hs)
  Op = typeof(H)
  tr = TrustRegion(gt, one(T))
  return JSOSolvers.TrunkSolver{T, V, Sub, Op}(x, xt, gx, gt, gn, Hs, krylov_subsolver, H, tr, params)
end

function trunk(
  nlp::AbstractNLPModel;
  x::V = nlp.meta.x0,
  bk_max::Int = get(JSOSolvers.TRUNK_bk_max, nlp),
  monotone::Bool = get(JSOSolvers.TRUNK_monotone, nlp),
  nm_itmax::Int = get(JSOSolvers.TRUNK_nm_itmax, nlp),
  subsolver::Symbol = :cg,
  mem::Int=5,
  scaling::Bool=true,
  kwargs...,
) where {V}
  solver = TrunkSolver(
    nlp;
    bk_max = bk_max,
    monotone = monotone,
    nm_itmax = nm_itmax,
    subsolver = subsolver,
    mem = mem,
    scaling = scaling,
  )
  return solve!(solver, nlp; x = x, kwargs...)
end





end # module


# f(x) = x[1]^2 * x[2]^2
# x0 = ones(2)
# nlp = ADNLPModel(f, x0)

# A = hess_op(nlp, x0)
# b = grad(nlp, x0)

# # # lbfgs_tr(A, b)

# solver = TrunkSolver(nlp, subsolver=:lbfgs, mem=2, scaling=false)
# stats = solve!(solver, nlp)
# print(stats)

# # solver = TrunkSolver(nlp, subsolver=:lbfgs, mem=10, scaling=true)
# # stats = solve!(solver, nlp)
# # print(stats)