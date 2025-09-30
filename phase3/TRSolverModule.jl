using Pkg
Pkg.activate("projet_env")

# TODO: add missing documentation for the functions in this file

module TRSolverModule

using LinearAlgebra, Logging, Printf
using Krylov, LinearOperators, NLPModels, ADNLPModels, SolverTools, SolverCore

# include("subsolvers.jl")
include("TrunkSolverUpdate.jl")

import SolverCore.solve!
export solve!

mutable struct TRSolver{
    T, 
    V <: AbstractVector{T},
    Sub <: KrylovWorkspace{T, T, V},
    Op <: AbstractLinearOperator{T},
} <: AbstractOptimizationSolver
    x::V
    g::V
    H::Op
    subsolver_ws::Sub
end

function TRSolver(
    nlp::AbstractNLPModel{T, V};
    subsolver::Symbol = :cg,
    mem::Int=2,
    scaling::Bool=true,
    ) where {T, V <: AbstractVector{T}}
    nvar = nlp.meta.nvar
    x = copy(nlp.meta.x0)     # V(undef, nvar)
    g = grad(nlp, x)    # V(undef, nvar)
    H = hess_op(nlp, x) #, Hs)
    Op = typeof(H)

    subsolver_ws = 
        if subsolver == :cg
            krylov_workspace(Val(subsolver), nvar, nvar, V)
        elseif subsolver == :lbfgs
            TrunkSolverUpdate.lbfgs_workspace(Val(subsolver), nvar, V; mem=mem, scaling=scaling)
        elseif subsolver == :diom
            TrunkSolverUpdate.DiomTRWorkspace(nlp.meta.x0; memory=mem)
        else
            throw("Invalid subsolver.")
        end
    
        Sub = typeof(subsolver_ws)
     
    return TRSolver{T, V, Sub, Op}(x, g, H, subsolver_ws)
end


function SolverCore.reset!(Solver::TRSolver)
    solver
end


function SolverCore.reset!(Solver::TRSolver, nlp::AbstractNLPModel)
    solver
end


"""
    trsolver(nlp; kwargs...)

    A trust-region solver for unconstrained optimization leveraging L-BFGS and DIOM to solve the quadratic model sub-problems.

"""
function trsolver(
    nlp::AbstractNLPModel;
    x::V = nlp.meta.x0,
    subsolver::Symbol = :cg,
    mem::Int = 2,                   # memory parameter
    scaling::Bool = true,           # scaling parameter
    kwargs...,
) where {V} 
    solver = TRSolver(nlp; subsolver, mem=mem, scaling=scaling)
    stats = GenericExecutionStats(nlp)
    return solve!(solver, nlp, stats; x = x, kwargs...)
end


function solve!(
    solver::TRSolver{T, V},
    nlp::AbstractNLPModel{T, V},
    stats::GenericExecutionStats{T, V};
    callback = (args...) -> nothing,
    x::V = nlp.meta.x0,
    atol::T = sqrt(eps(T)),
    rtol::T = sqrt(eps(T)),
    sub_rtol = sqrt(eps(T)),
    fixed_sub_rtol = false,     # use fixed relative tolerance for the quadratic solver
    verbose::Int = 0,   # TODO: to be implemented
    mu = 0.25,
    eta = 0.01,
    delta_max = zero(T),
    max_iter = 10*nlp.meta.nvar,
    max_time = 30.0,
) where {T, V <: AbstractVector{T}}
if !(nlp.meta.minimize)
    error("TR Solver only works for minimization problem")
end
if !unconstrained(nlp)
    error("TR Solver should only be called for unconstrained problems.")
end

    # start of the algorithm

    SolverCore.reset!(stats)
    
    start_time = time()
    elapsed_time = 0.0
    set_time!(stats, 0.0)

    f = obj(nlp, solver.x)

    solver.H = hess_op(nlp, solver.x)

    delta = one(T)

    p = similar(x)
    b = similar(x)

    grad_norm = norm(solver.g)
    tolerance = atol + grad_norm * rtol

    if !fixed_sub_rtol
        sub_rtol = max(rtol, min(sqrt(grad_norm), T(0.1)))
    end 

    iter = 0
    set_iter!(stats, iter)
    set_objective!(stats, f)
    set_dual_residual!(stats, grad_norm)

    status = :unknown
    set_status!(stats, status)

    done = false

    while !done

        iter += 1

        krylov_solve!(solver.subsolver_ws, solver.H, -1 .* solver.g, atol=atol, rtol=sub_rtol, radius=delta, verbose=verbose)
        p .= solver.subsolver_ws.x
        
        b .= solver.H * p     # using hess_op

        f_new = obj(nlp, solver.x .+ p)
        rho = (f_new - f)/(dot(p, solver.g) + (dot(p, b)))/2
        
        # bad approximation
        if rho <= mu
            delta /= 4

        # very good approximation
        elseif rho > 1 - mu && delta_max > 0
            delta = min(delta/2, delta_max)
        end

        # good approximation -> update current point
        if rho >= eta
            solver.x .+= p
            f = f_new
            grad!(nlp, solver.x, solver.g)
            grad_norm = norm(solver.g)
            
            solver.H = hess_op(nlp, solver.x)

            if !fixed_sub_rtol
                sub_rtol = max(rtol, min(sqrt(grad_norm), T(0.1)))
            end
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

    set_solution!(stats, solver.x)
    stats
end

end     # module


# f(x) = x[1]^2 * x[2]^2
# x0 = ones(2)
# nlp = ADNLPModel(f, x0)

# A = hess_op(nlp, x0)
# b = grad(nlp, x0)

# # # lbfgs_tr(A, b)

# # solver = TRSolver(nlp, subsolver=:lbfgs, mem=2, scaling=false)
# # stats = solve!(solver, nlp)
# # print(stats)

# stats = trsolver(nlp, subsolver=:lbfgs)
# println(stats)

# # solver = TrunkSolver(nlp, subsolver=:lbfgs, mem=10, scaling=true)
# # stats = solve!(solver, nlp)
# # print(stats)