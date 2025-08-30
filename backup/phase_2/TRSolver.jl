
# TODO: add missing documentation for the functions in this file

using LinearAlgebra, Logging, Printf
using Krylov, LinearOperators, NLPModels, ADNLPModels, SolverTools, SolverCore

include("subsolvers.jl")

import SolverCore.solve!
export solve!

mutable struct TRSolver{
    T, 
    V <: AbstractVector{T},
    Op <: AbstractLinearOperator{T},
} <: AbstractOptimizationSolver
    x::V
    gx::V
    Hs::V
    H::Op
    subsolver::Symbol
end

function TRSolver(
    nlp::AbstractNLPModel{T, V};
    subsolver::Symbol = :cg 
    ) where {T, V <: AbstractVector{T}}
    nvar = nlp.meta.nvar
    x = V(undef, nvar)
    gx = V(undef, nvar)
    Hs = V(undef, nvar)
    H = hess_op!(nlp, x, Hs)
    Op = typeof(H)
    subsolver = subsolver
     
    return TRSolver{T, V, Op}(x, gx, Hs, H, subsolver)
end


function SolverCore.reset!(Solver::TRSolver)
    solver
end


function SolverCore.reset!(Solver::TRSolver, nlp::AbstractNLPModel)
    solver
end


function trsolver(
    nlp::AbstractNLPModel;
    x::V = nlp.meta.x0,
    subsolver::Symbol = :cg,
    kwargs...,
) where {V} 
    solver = TRSolver(nlp; subsolver)
    return solve!(solver, nlp; x = x, kwargs...)
end


function solve!(
    solver::TRSolver{T, V, Op},
    nlp::AbstractNLPModel{T, V},
    stats::GenericExecutionStats{T, V};
    callback = (args...) -> nothing,
    x::V = nlp.meta.x0,
    atol::T = sqrt(eps(T)),
    rtol::T = sqrt(eps(T)),
    sub_rtol = sqrt(eps(T)),
    verbose::Int = 0,
    mu = 0.25,
    eta = 0.01,
    delta_max = Inf,
    max_iter = 3*nlp.meta.nvar,
    max_time = 30.0,
    mem = 1,
    scaling = true,
) where {T, V <: AbstractVector{T}, Op <: AbstractLinearOperator{T}}
if !(nlp.meta.minimize)
    error("TR Solver only works for minimization problem")
end
if !unconstrained(nlp)
    error("TR Solver should only be called for unconstrained problems.")
end

    # start of algorithm

    SolverCore.reset!(stats)
    start_time = time()
    elapsed_time = 0.0
    set_time!(stats, 0.0)

    n = nlp.meta.nvar

    # solver.x .= x
    # x = solver.x
    # grad = solver.gx
    # H = solver.H
    subsolver = solver.subsolver

    x = nlp.meta.x0
    f = obj(nlp, x)
    g = grad(nlp, x)

    H = hess_op(nlp, x)
    
    delta = 1.0

    p = similar(x)
    b = similar(x)

    grad_norm = norm(g)
    tolerance = atol + grad_norm * rtol

    sub_rtol = max(rtol, min(sqrt(grad_norm), T(0.1)))

    iter = 0
    set_iter!(stats, iter)
    set_objective!(stats, f)
    set_dual_residual!(stats, grad_norm)

    status = :unknown
    set_status!(stats, status)

    done = false

    while !done

        iter += 1

        # solve quadratic model subject to a trust region constraint
        if subsolver == :cg
            p, sub_stats = Krylov.cg(H, -g, radius=delta, atol=atol, rtol=sub_rtol)
        elseif subsolver == :lbfgs
            p, sub_stats = lbfgs(H, g, delta=delta, 
                                    atol=atol, 
                                    rtol=sub_rtol, 
                                    mem = mem, 
                                    itmax = 2*n, 
                                    scaling = scaling)
        end
        
        b .= H * p     # using hess_op

        f_new = obj(nlp, x + p)
        rho = (f_new - f)/(dot(p, g) + 0.5*(dot(p, b)))
        
        # bad approximation
        if rho <= mu
            delta *= 0.25

        # very good approximation
        elseif rho > 1 - mu
            delta = min(2.0*delta, delta_max)
        end

        # good approximation -> update current point
        if rho >= eta
            x .+= p
            f = f_new
            g .= grad(nlp, x)
            grad_norm = norm(g)

            H = hess_op(nlp, x)

            sub_rtol = max(rtol, min(sqrt(grad_norm), T(0.1)))
        end

        if grad_norm <= tolerance
            done = true
            status = :first_order
        elseif iter >= max_iter
            done = true
            status = :max_iter
        elseif elapsed_time >= max_time
            done = true
            status = :max_time
        end

        elapsed_time = time() - start_time

        set_iter!(stats, iter)
        set_objective!(stats, f)
        set_dual_residual!(stats, grad_norm)
        set_status!(stats, status)
        set_time!(stats, elapsed_time)
        
        callback(nlp, solver, stats)
    end

    set_solution!(stats, x)
    stats
end
