using Base

mutable struct LAOStarSolver
    max_iter::Integer
    dead_end_cost::Float64
    ω::Float64
    ϵ::Float64
    π::Dict{Integer,Integer}
    V::Vector{Float64}
    G::Vector{Float64}
    H::Vector{Float64}
    Qs::Vector{Float64}
end

function weighted_lookahead(ℒ::LAOStarSolver, 𝒱::ValueIterationSolver, M, s::Integer, a::Integer)
    S, A, T, C, H = M.S, M.A, M.T, M.C, M.H
    V = 𝒱.V
    G = ℒ.G
    T = T(M,S[s],A[a])
    h = sum(T .* H(M, V, S[s], A[a]))
    g = C(M, S[s], A[a]) + sum(T .* G)
    return g, h
end

function lookahead(ℒ::LAOStarSolver, M, s::Integer, a::Integer)
    S, A, T, C, V = M.S, M.A, M.T, M.C, ℒ.V
    T = T(M,S[s],A[a])
    return C(M,S[s],A[a]) + sum(T .* V)
end

function weighted_backup(ℒ::LAOStarSolver, 𝒱::ValueIterationSolver, M, s::Integer)
    best_q, best_a = ℒ.dead_end_cost, 1
    best_g, best_h = 0., 0.
    for a = 1:length(M.A)
        g, h = weighted_lookahead(ℒ, 𝒱, M, s, a)
        # q = ℒ.ω*g + (1 -ℒ.ω)*h
        q = g + h
        if q < best_q
            best_g = g
            best_h = h
            best_a = a
            best_q = q
        end
    end
    return best_a, best_q, best_g, best_h
end

function backup(ℒ::LAOStarSolver, M, s::Integer)
    for a = 1:length(M.A)
        ℒ.Qs[a] = lookahead(ℒ, M, s, a)
    end
    a = Base.argmin(ℒ.Qs)
    return a, ℒ.Qs[a]
end

function weighted_bellman_update(ℒ::LAOStarSolver, 𝒱::ValueIterationSolver, M, s::Integer)
    a, q, g, h = weighted_backup(ℒ, 𝒱, M, s)
    residual = abs(ℒ.V[s] - q)
    ℒ.V[s] = q
    ℒ.G[s] = g
    ℒ.H[s] = h
    ℒ.π[s] = a
    return residual
end

function bellman_update(ℒ::LAOStarSolver, 𝒱::ValueIterationSolver, M, s::Integer)
    if ℒ.ω ≠ 1.
        return weighted_bellman_update(ℒ, 𝒱, M, s)
    end
    a, q = backup(ℒ, M, s)
    residual = abs(ℒ.V[s] - q)
    ℒ.V[s] = q
    ℒ.π[s] = a
    return residual
end

function expand(ℒ::LAOStarSolver, 𝒱::ValueIterationSolver, M,
                s::Integer, visited::Set{Integer})
    stack = Vector{Integer}()
    push!(stack, s)
    count = 0

    while !isempty(stack)
        v = pop!(stack)
        if v in visited
            continue
        end
        push!(visited, v)
        if v in M.G
            continue
        else
            residual = bellman_update(ℒ, 𝒱, M, v)
            count += 1
            a = ℒ.π[v]
            transitions = M.T(M, M.S[v], M.A[a])
            for s′ = 1:length(M.S)
                if transitions[s′] > .0001
                    push!(stack, s′)
                end
            end
        end
    end
    return count
end

function test_convergence(ℒ::LAOStarSolver, 𝒱::ValueIterationSolver, M,
                          s::Integer, visited::Set{Integer})
    error = 0.0
    stack = Vector{Integer}()
    push!(stack, s)
    while !isempty(stack)
        v = pop!(stack)
        if v in visited
            continue
        end
        push!(visited, v)
        if v in M.G
            continue
        elseif v ∉ keys(ℒ.π)
            return ℒ.dead_end_cost + 1
        else
            error = max(error, bellman_update(ℒ, 𝒱, M, v))
            a = ℒ.π[v]
            transitions = M.T(M, M.S[v], M.A[a])
            for s′ = 1:length(M.S)
                if transitions[s′] != 0.0
                    push!(stack, s′)
                end
            end
        end
    end
    return error
end

function solve(ℒ::LAOStarSolver, 𝒱::ValueIterationSolver, M, s::Integer)
    expanded = 0
    visited = Set{Integer}()

    iter = 0
    while iter < ℒ.max_iter
        empty!(visited)
        expanded += expand(ℒ, 𝒱, M, s, visited)
        empty!(visited)
        res = test_convergence(ℒ, 𝒱, M, s, visited)
        println("Iteration: $iter        Residual: $res")
        if res < ℒ.ϵ
            return ℒ.π[s]
        end
        iter += 1
    end
    return ℒ.π[s]
end
