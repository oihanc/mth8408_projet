struct LBFGSStats
    niter::Int
    residuals::Vector{Float64}
    elapsed_time::Vector{Float64}
    PAP::Matrix{Float64}
end

"""
    lbfgs(Bj, gk, delta; atol=1e-5, rtol=1e-5, mem = 5, max_iter = 10)

Find the minima of a quadratic model with respect to a trust region. This
implementation is based on: https://www.gerad.ca/fr/papers/G-2019-64
"""
function lbfgs(Bj, b; delta = 0.0, atol=1e-26, rtol=1e-26, mem = 2, itmax = 0, scaling = false)

    gk = copy(b)
    dim = length(gk)
    gnormk = gnorm0 = norm(gk)
    
    k = 1
    pk = zeros(dim)
    residuals = Float64[gnorm0]
    elapsed_time = Float64[0.0]

    save_basis = false
    AP_list = Vector{Vector{Float64}}()
    P_list = Vector{Vector{Float64}}()
    
    if itmax == 0
        itmax = 2*dim
    end
    Hk = InverseLBFGSOperator(dim, mem = mem, scaling = scaling)

    while gnormk > atol + rtol * gnorm0 && k <= itmax

        dk = -Hk*gk
        bk =  Bj*dk
        push!(AP_list, bk ./ norm(dk))  
        push!(P_list, dk ./ norm(dk))  

        # handle negative curvature
        if dot(dk, bk) <= 0
            alphak = -sign(dot(gk, dk))*2*delta/norm(dk)
        else
            alphak = -dot(gk, dk)/dot(dk, bk)
        end

        sk = alphak .* dk
        pk = pk + sk

        if delta > 0.0 && norm(pk) >= delta
            pk -= sk
            # TODO: optimize implementation
            # compute eq (87) such that norm(pk + tau*sk) = delta (see reference)
            tau = (-dot(pk, sk) + sqrt(dot(pk, sk)^2 + dot(sk, sk)*(delta^2 - dot(pk, pk))))/dot(sk, sk)

            return pk + tau .* sk
        end

        yk = alphak .* bk
        gk += yk
        
        gnormk = norm(gk)
        k += 1
        
        # update the inverse Hessian approximation
        push!(Hk, sk, yk)
        push!(residuals, gnormk)
        push!(elapsed_time, time() - elapsed_time[1])
    end

    AP  = hcat(AP_list...)   # construit la matrice P de taille (dim, k)
    P   = hcat(P_list...)     
    PAP = P' * AP
    
    return pk,  LBFGSStats(k, residuals, elapsed_time, PAP)
end


function lbfgs_tr(Bj, b, mem=1; Hk = InverseLBFGSOperator(length(b), mem = mem, scaling = true), delta = 0.0, atol=1e-26, rtol=1e-26, itmax = 0, scaling = false)

    gk = copy(b)
    dim = length(gk)
    gnormk = gnorm0 = norm(gk)
    
    k = 1
    pk = zeros(dim)
    residuals = Float64[gnorm0]


    AP_list = Vector{Vector{Float64}}()
    P_list = Vector{Vector{Float64}}()
    
    if itmax == 0
        itmax = 2*dim
    end
    # Hk = InverseLBFGSOperator(dim, mem = mem, scaling = scaling)

    dk = similar(gk)
    bk = similar(gk)
    sk = similar(gk)
    yk = similar(gk)

    save_basis = false
    # AP  = hcat(AP_list...)   # construit la matrice P de taille (dim, k)
    # P   = hcat(P_list...)     
    # PAP = P' * AP
    # PAP = gk' * gk
    PAP = Matrix{Float64}(undef, 0, 0)

    start_time = time()
    elapsed_time = Float64[0.0]

    tolerance = atol + rtol * gnorm0

    while gnormk > tolerance && k <= itmax

        # dk .= -Hk*gk
        mul!(dk, -Hk, gk)
        # bk .=  Bj*dk
        mul!(bk, Bj, dk)
        
        if save_basis
            push!(AP_list, bk ./ norm(dk))  
            push!(P_list, dk ./ norm(dk)) 
        end

        # handle negative curvature
        # if delta > 0.0 && dot(dk, bk) <= 0
        #     alphak = -sign(dot(gk, dk))*2*delta/norm(dk)
        # else
        #     alphak = -dot(gk, dk)/dot(dk, bk)
        # end

        alphak = -dot(gk, dk)/dot(dk, bk)

        sk .= alphak .* dk
        pk .= pk .+ sk

        # if delta > 0.0 && norm(pk) >= delta
        #     pk .-= sk
        #     # TODO: optimize implementation
        #     # compute eq (87) such that norm(pk + tau*sk) = delta (see reference)
        #     tau = (-dot(pk, sk) + sqrt(dot(pk, sk)^2 + dot(sk, sk)*(delta^2 - dot(pk, pk))))/dot(sk, sk)

        #     return pk .+ tau .* sk
        # end

        yk .= alphak .* bk
        gk .+= yk
        
        gnormk = norm(gk)
        k += 1
        
        # update the inverse Hessian approximation
        push!(Hk, sk, yk)

        push!(residuals, gnormk)

        push!(elapsed_time, time() - start_time)
    end

    if save_basis
        AP  = hcat(AP_list...)   # construit la matrice P de taille (dim, k)
        P   = hcat(P_list...)     
        PAP = P' * AP
    end

    return pk, LBFGSStats(k, residuals, elapsed_time, PAP)
end