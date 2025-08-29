



function fletchcr(n::Int)
    @assert n >= 2 "N must be at least for the fletchcr problem."

    f(x) = 100 * sum((x[i+1] -x[i] + 1 - x[i]^2)^2 for i in 1:n-1)
    x0 = [(-1.0)^i for i in 1:n]
    
    return ADNLPModel(f, x0, name="fletchcr")
end

function nondquar(n::Int)
    @assert n ≥ 3 "n must be at least 3 for the nondquar problem."

    f(x) = (x[1] * x[2])^2 + (x[n-1] * x[n])^2 +
           sum((x[i] + x[i+1] + x[n])^4 for i in 1:n-2)
    x0 = randn(n)

    return ADNLPModel(f, x0, name="nondquar")
end

function woods(n::Int)
    @assert n % 4 == 0 "n must be a multiple of 4 for the Woods function."
    m = div(n, 4)

    f(x) = sum(
        100 * (x[4i-3]^2 - x[4i-2])^2 +
        (x[4i-3] - 1)^2 +
        90 * (x[4i]^2 - x[4i-1])^2 +
        (x[4i-1] - 1)^2 +
        10 * (x[4i-2] + x[4i])^2 +
        0.1 * (x[4i-2] - x[4i])^2
        for i in 1:m
    )

    x0 = repeat([-3.0, -1.0, -3.0, -1.0], n÷4) .+ 0.1*randn(n)

    return ADNLPModel(f, x0, name="woods")
end

function broydn7d(n::Int)
    @assert n % 2 == 0 "n must be even for the broydn7d function."
    half = div(n, 2)
    scale = (1 / 7)^3

    f(x) = begin
        s = (2x[2] + (3 - x[1]^2) * x[1])^2
        for i in 2:n-1
            s += (x[i-1] + 2x[i] + (3 - x[i]^2) * x[i])^2
        end
        s += (x[n-1] + (3 - x[n]^2) * x[n])^2
        for i in 1:half
            s += (x[i] + x[i+half])^2
        end
        return scale * s
    end

    x0 = 2.0 * rand(n) .- 1.0

    return ADNLPModel(f, x0, name="broydn7d")
end

function l_sparsine(n::Int)
    f(x) = begin
        s = 0.0
        for i in 1:n
            idx(j) = mod(j * i - 1, n) + 1
            s += sin(x[i]) +
                 sin(x[idx(2)]) +
                 sin(x[idx(3)]) +
                 sin(x[idx(5)]) +
                 sin(x[idx(7)]) +
                 sin(x[idx(11)])^2
        end
        return 0.5 * s
    end

    x0 = 0.5*pi * rand(n)

    return ADNLPModel(f, x0, name="sparsine")
end

