using Base

mutable struct ValueIterationSolver
    eps::Float64
    cost_minimizing::Bool
    π::Dict{Integer, Integer}
    V::Vector{Float64}
    Qs::Vector{Float64}
end
# function ValueIterationSolver(eps, cost_minimizing)
#     return ValueIterationSolver(eps, cost_minimizing)
# end

function lookahead(𝒱::ValueIterationSolver, M, s::Integer, a::Integer)
    S, T, R, V = M.S, M.T, M.R, 𝒱.V
    return R[s][a] + .95*sum(T[s][a][s′] * V[s′] for s′=1:length(S))
end

function backup(𝒱::ValueIterationSolver, M, s::Integer)
    for a = 1:length(M.A)
        𝒱.Qs[a] = lookahead(𝒱, M, s, a)
    end
    a = Base.argmax(𝒱.Qs)
    q = 𝒱.Qs[a]
    return a, q
end

function solve(𝒱::ValueIterationSolver, M)
    𝒱.V = zeros(length(M.S))
    𝒱.Qs = zeros(length(M.A))
    while true
        residual = 0.
        for s = 1:length(M.S)
            a, q = backup(𝒱, M, s)
            residual = max(residual, abs(𝒱.V[s] - q))
            # println(residual)
            𝒱.V[s] = q
            𝒱.π[s] = a
        end
        # println(residual)
        @assert !isnan(residual)
        if residual < 𝒱.eps
            break
        end
    end
end
