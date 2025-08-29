

using Pkg
Pkg.activate("projet_env")

using LinearAlgebra, Random, Printf, LinearOperators


using Krylov
include("subsolvers.jl")


function quadratic_obj(x, g, B)
    return dot(g, x) + 0.5.*dot(x, B * x)
end

rand_orthonormal(n; rng=Random.default_rng()) = Matrix(qr!(randn(rng, n, n)).Q)

function make_B(eigs::AbstractVector{<:Real}; rng=Random.default_rng())
    n = length(eigs)
    Q = rand_orthonormal(n; rng)
    return Symmetric(Q * Diagonal(eigs) * Q')
end

function compare_subsolvers(g, B, delta; print_stats=false)
    x_cg, stats_cg = Krylov.cg(B, -g, radius=delta)
    f_cg = quadratic_obj(x_cg, g, B)

    if print_stats
        println("--- CG Stats ---")
        println(stats_cg)
    end

    x_lbfgs, stats_lbfgs = lbfgs_tr(B, g, 10, delta=delta)
    f_lbfgs = quadratic_obj(x_lbfgs, g, B)

    @printf "|  CG: %.3e  |  LBFGS: %.3e  |\n" f_cg f_lbfgs
end

n = 50

# interior solution
println("------- Interior Solution -------")

B = make_B([range(1.0, 10.0; length=n)...])
g = randn(n)
delta = 1.0e6

compare_subsolvers(g, B, delta)


# boundary solution
println("------- Boundary Solution -------")

B = make_B([range(1.0, 10.0; length=n)...])
delta = 1.0
compare_subsolvers(g, B, delta)


# negative curvature
println("------- Negative Curvature -------")

eigs = vcat(-1.0, range(0.5, 10.0; length=n-1))       # one negative eigenvalue
B = make_B(eigs)
delta = 1.0
compare_subsolvers(g, B, delta, print_stats=false)

# ill-conditioned
println("------- Ill-Conditioned -------")
eigs = [1e-8; range(1.0, 10.0; length=n-1)...]
B = make_B(eigs)
g = randn(n)
delta = 1.0
compare_subsolvers(g, B, delta, print_stats=false)



# large scale