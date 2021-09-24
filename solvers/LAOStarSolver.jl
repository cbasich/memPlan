using Base
include("VIMDPSolver.jl")

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

# function weighted_lookahead(ℒ::LAOStarSolver, 𝒱::ValueIterationSolver, M, s::Integer, a::Integer)
#     S, A, T, C, H = M.S, M.A, M.T, M.C, M.H
#     V = 𝒱.V
#     G = ℒ.G
#     T = T(M,S[s],A[a])
#     h = sum(T .* H(M, V, S[s], A[a]))*.99
#     g = C(M, S[s], A[a]) + sum(T .* G)*.99
#     return g, h
# end

# function weighted_backup(ℒ::LAOStarSolver, 𝒱::ValueIterationSolver, M, s::Integer)
#     best_q, best_a = ℒ.dead_end_cost, 1
#     best_g, best_h = 0., 0.
#     for a = 1:length(M.A)
#         g, h = weighted_lookahead(ℒ, 𝒱, M, s, a)
#         # q = ℒ.ω*g + (1 -ℒ.ω)*h
#         q = g + h
#         if q < best_q
#             best_g = g
#             best_h = h
#             best_a = a
#             best_q = q
#         end
#     end
#     return best_a, best_q, best_g, best_h
# end

# function weighted_bellman_update(ℒ::LAOStarSolver, 𝒱::ValueIterationSolver, M, s::Integer)
#     a, q, g, h = weighted_backup(ℒ, 𝒱, M, s)
#     residual = abs(ℒ.V[s] - q)
#     ℒ.V[s] = q
#     ℒ.G[s] = g
#     ℒ.H[s] = h
#     ℒ.π[s] = a
#     return residual
# end

function lookahead(ℒ::LAOStarSolver,
                   𝒱::ValueIterationSolver,
                   M,
                   s::Integer,
                   a::Integer)
    S, A, T, R, H, V = M.S, M.A, M.T[s][a], M.R, M.H, ℒ.V

    q = 0.
    for (s′, p) in T
        if haskey(ℒ.π, s′)
            q += p * V[s′]
        else
            q += p * H(M, 𝒱.V, s, a)
        end
    end
    return q + R(M,s,a)
end

function backup(ℒ::LAOStarSolver,
                𝒱::ValueIterationSolver,
                M,
                s::Integer)
    for a = 1:length(M.A)
        ℒ.Qs[a] = lookahead(ℒ, 𝒱, M, s, a)
    end
    a = Base.argmax(ℒ.Qs)
    return a, ℒ.Qs[a]
end

function bellman_update(ℒ::LAOStarSolver,
                        𝒱::ValueIterationSolver,
                        M,
                        s::Integer)
    # if ℒ.ω ≠ 1.
    #     return weighted_bellman_update(ℒ, 𝒱, M, s)
    # end
    a, q = backup(ℒ, 𝒱,  M, s)
    residual = abs(ℒ.V[s] - q)
    ℒ.V[s] = q
    ℒ.π[s] = a
    return residual
end

function expand(ℒ::LAOStarSolver, 𝒱::ValueIterationSolver, M,
                s::Integer, visited::Set{Integer})
    if s ∈ visited
        return 0
    end
    push!(visited, s)
    if terminal(ℳ, M.S[s])
        return 0
    end

    count = 0
    if s ∉ keys(ℒ.π)
        bellman_update(ℒ, 𝒱, M, s)
        return 1
    else
        a = ℒ.π[s]
        for (s′, p) in M.T[s][a]
            count += expand(ℒ, 𝒱, M, s′, visited)
        end
    end
    return count
end

# function expand(ℒ::LAOStarSolver, 𝒱::ValueIterationSolver, M,
#                 s::Integer, visited::Set{Integer})
#     stack = Vector{Integer}()
#     push!(stack, s)
#     count = 0
#
#     while !isempty(stack)
#         v = pop!(stack)
#         # Check if index(v, M.S) is in key(L.pi)
#         if v in visited
#             continue
#         end
#         push!(visited, v)
#         if v in M.G
#             continue
#         else
#             if index(v, M.S) ∉ keys(ℒ.π)
#                 residual = bellman_update(ℒ, 𝒱, M, v)
#                 count += 1
#             else
#                 a = ℒ.π[v]
#                 transitions = M.T(M, M.S[v], M.A[a])
#                 for s′ = 1:length(M.S)
#                     if transitions[s′] > .0001
#                         push!(stack, s′)
#                     end
#                 end
#             end
#
#         end
#     end
#     return count
# end

function test_convergence(ℒ::LAOStarSolver,
                          𝒱::ValueIterationSolver,
                          M,
                          s::Integer,
                          visited::Set{Integer})
    error = 0.0
    if terminal(M.S[s])
        return 0.0
    end

    if s ∈ visited
        return 0.0
    end
    push!(visited, s)
    a = -1
    if s ∈ keys(ℒ.π)
        a = ℒ.π[s]
        for (s′, p ) in M.T[s][a]
            error = max(error, test_convergence(ℒ, 𝒱, M, s′, visited))
        end
    else
        return ℒ.dead_end_cost + 1
    end

    error = max(error, bellman_update(ℒ, 𝒱, M, s))
    if (a == -1 && !haskey(ℒ.π, s)) || (haskey(ℒ.π, s) && a == ℒ.π[s])
        return error
    end

    # println("Action for state $s change from $a to $(ℒ.π[s])")
    return ℒ.dead_end_cost + 1
end

# function test_convergence(ℒ::LAOStarSolver, 𝒱::ValueIterationSolver, M,
#                           s::Integer, visited::Set{Integer})
#     error = 0.0
#     stack = Vector{Integer}()
#     push!(stack, s)
#     while !isempty(stack)
#         v = pop!(stack)
#         if v in visited
#             continue
#         end
#         push!(visited, v)
#         if v in M.G
#             continue
#         elseif v ∉ keys(ℒ.π)
#             return ℒ.dead_end_cost + 1
#         else
#             error = max(error, bellman_update(ℒ, 𝒱, M, v))
#             a = ℒ.π[v]
#             transitions = M.T(M, M.S[v], M.A[a])
#             for s′ = 1:length(M.S)
#                 if transitions[s′] != 0.0
#                     push!(stack, s′)
#                 end
#             end
#         end
#     end
#     return error
# end

function solve(ℒ::LAOStarSolver,
               𝒱::ValueIterationSolver,
               M,
               s::Integer)
    expanded = 0
    visited = Set{Integer}()
    iter = 0
    total_expanded = 0

    error = ℒ.dead_end_cost
    #while iter < ℒ.max_iter
    while true
        while true
            empty!(visited)
            num_expanded = expand(ℒ, 𝒱, M, s, visited)
            total_expanded += num_expanded
            # println(num_expanded, "               ", total_expanded)
            if num_expanded == 0
                break
            end
        end
        # println("\nSTART")
        while true
            empty!(visited)
            error = test_convergence(ℒ, 𝒱, M, s, visited)
            # println(error)
            if error > ℒ.dead_end_cost
                break
            end
            if error < ℒ.ϵ
                return ℒ.π[s], total_expanded
            end
            # println(iter, "            ", error)
        end
        # println("END\n")
        iter += 1
        # println(iter, "            ", error)
    end
    println("Total iterations taken: $iter")
    println("Total nodes expanded: $total_expanded")
    return ℒ.π[s], total_expanded
end
