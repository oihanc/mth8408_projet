
# using LinearOperators

struct LBFGSStats
    niter::Int
    residuals::Vector{Float64}
end

"""
    lbfgs(Bj, gk, delta; atol=1e-5, rtol=1e-5, mem = 5, max_iter = 10)

Find the minima of a quadratic model with respect to a trust region. This
implementation is based on: https://www.gerad.ca/fr/papers/G-2019-64
"""
function lbfgs(Bj, gk; delta = 0.0, atol=1e-5, rtol=1e-5, mem = 2, itmax = 0, scaling = true)

    dim = length(gk)

    gnormk = gnorm0 = norm(gk)

    k = 1
    pk = zeros(dim)
    residuals = Float64[gnorm0]
    if itmax == 0
        itmax = 2*dim
    end
    Hk = InverseLBFGSOperator(dim, mem = mem, scaling = scaling)

    # TODO: review the use of norm(gk) here
    while gnormk > atol + rtol * gnorm0 && k <= itmax

        dk = -Hk*gk

        bk =  Bj*dk

        if dot(dk, bk) <= 0
            alphak = -sign(dot(gk, dk))*2*delta/norm(dk)
        else
            alphak = -dot(gk, dk)/dot(dk, bk)
        end

        sk = alphak*dk
        pk = pk + sk

        if delta > 0.0 && norm(pk) >= delta
            pk = pk - sk
            # TODO: optimize implementation
            # compute eq (87) such that norm(pk + tau*sk) = delta (see reference)
            tau = (-dot(pk, sk) + sqrt(dot(pk, sk)^2 + dot(sk, sk)*(delta^2 - dot(pk, pk))))/dot(sk, sk)

            @assert norm(pk + tau .* sk) <= delta+1.0e-8 "Error in trust region limit projection."

            return pk + tau .* sk, LBFGSStats(k, residuals)
        end

        yk = alphak*bk
        gk += yk
        gnormk = norm(gk)
        k += 1

        # update the inverse Hessian approximation
        push!(Hk, sk, yk)
        
        push!(residuals, gnormk)
    end
    return pk,  LBFGSStats(k, residuals)
end

"""
    lbfgs_tr(B, g, delta; atol=1e-5, rtol=1e-5, mem = 5, max_iter = 10)

Find the minima of a quadratic model with respect to a trust region. This
implementation is based on: https://www.gerad.ca/fr/papers/G-2019-64
"""
function lbfgs_tr(B, g; delta = 0.0, atol=sqrt(eps(T)), rtol=sqrt(eps(T)), mem = 2, itmax = 0, scaling = true)

    dim = length(g)

    gnormk = gnorm0 = norm(g)

    iter = 0
    
    pk = zeros(dim)

    dk = similar(pk)
    sk = similar(pk)
    yk = similar(pk)

    residuals = Float64[gnorm0]

    tolerance = atol + rtol * gnorm0


    if itmax == 0
        itmax = 2*dim
    end

    Hk = InverseLBFGSOperator(dim, mem = mem, scaling = scaling)

    done = false
    status = :unknown

    # TODO: review the use of norm(g) here
    while done
        iter += 1

        dk .= -Hk*g

        bk = B*dk

        if dot(dk, bk) <= 0
            alphak = -sign(dot(g, dk))*2*delta/norm(dk)
        else
            alphak = -dot(g, dk)/dot(dk, bk)
        end

        sk .= alphak*dk
        pk .+= sk

        if delta > 0.0 && norm(pk) >= delta
            pk .-= sk
            # TODO: optimize implementation
            # compute eq (87) such that norm(pk + tau*sk) = delta (see reference)
            tau = (-dot(pk, sk) + sqrt(dot(pk, sk)^2 + dot(sk, sk)*(delta^2 - dot(pk, pk))))/dot(sk, sk)

            done = true

            pk .+= tau .* sk
            # return pk + tau .* sk, LBFGSStats(k, residuals)

        end

        yk .= alphak*bk
        g .+= yk
        grad_norm = norm(g)

        if grad_norm <= tolerance
            done = true
            status = :first_order
        elseif iter >= itmax
            done = true
            status = :max_iter
        end
        
        # update the inverse Hessian approximation
        push!(Hk, sk, yk)
        
        # push!(residuals, gnormk)
    end

    return pk #,  LBFGSStats(k, residuals)
end