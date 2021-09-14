using Base
include("VIMDPSolver.jl")

mutable struct FLARESSolver
    max_trials::Integer
    horizon::Integer
    optimal::Bool
    use_probs_for_depth::Bool
    dead_end_cost::Float64
    ϵ::Float64
    π::Dict{Integer,Integer}
    V::Vector{Float64}
    H::Vector{Float64}
    solved::Set{Integer}
    dsolved::Set{Integer}
    Qs::Vector{Float64}#todo: I have no idea what this is supposed to be used for, but it's needed for update, etc cribbed from LAO*
end

function trial(ℱ, 𝒱, M, s::Integer)
    current_state = s
    visited = []
    accumulated_cost = 0.0

    while !(labeled_solved(ℱ, current_state))
        if terminal(current_state)
            break
        end
        push!(visited, current_state)
        bellman_update(ℱ, 𝒱, M, current_state)

        if (accumulated_cost >= ℱ.dead_end_cost)
            break
        end

        greedy_action = ℱ.π[current_state]
        accumulated_cost += M.R(M.S[current_state], M.A[greedy_action])
        current_state = index(generate_successor(M, M.S[currentState], M.A[greedy_action]), M.S)
    end
    while(!isempty(visited))
        current_state = pop!(visited)
        solved = check_solved(ℱ, current_state)
        if (!solved)
            break
        end
    end
end

function labeled_solved(ℱ, s)
    return s ∈ ℱ.solved || s ∈ ℱ.dsolved
end

function residual(ℱ, 𝒱, M, s)
    if !haskey(ℱ.π, s)
        return 0.0
    end
    a = π[s]
    res = lookahead(ℱ, 𝒱, M, s, a) - ℱ.V[s]
    return abs(res)
end

function check_solved(ℱ, 𝒱, M, s::Integer)
    open = []
    closed = []

    current_state = s
    if !labeled_solved(ℱ, s)
        push!(open, s => 0.0)
    else
        return true
    end

    rv = true
    subgraph_within_search_horizon = ℱ.optimal && true
    while !(isempty(open))
        pp = pop!(open)
        current_state = pp.first
        depth = pp.second

        if (ℱ.useProbsForDepth && (depth < 2 * log(horizon)) || (!ℱ.useProbsForDepth && depth > 2*horizon))
            subgraph_within_search_horizon = false
            continue
        end

        if (terminal(M.S[current_state]))
            continue
        end

        push!(closed, pp)
        a = ℱ.π[current_state]
        if (residual(ℱ, 𝒱, M, current_state) > ℱ.ϵ)
            rv = false
        end
        for sp ∈ M.S
            prob = T(M, M.S[current_state], M.A[a])[sp]
            if prob > 0
                if !labeled_solved(ℱ, sp) && sp ∉ closed
                    new_depth = compute_new_depth(ℱ, prob, depth)
                    push!(open, sp => new_depth)
                elseif sp ∈ ℱ.dsolved && sp ∉ ℱ.solved
                    subgraph_within_search_horizon = false
                end
            end
        end
    end

    if rv
        for pp ∈ closed
            delete!(closed, pp)
            if subgraph_within_search_horizon
                push!(ℱ.solved, pp.first)
                push!(ℱ.dsolved, pp.first)
            else
                if (ℱ.use_probs_for_depth && pp.second > log(ℱ.horizon)) || (!ℱ.use_probs_for_depth && pp.second <= ℱ.horizon)

                    push!(ℱ.dsolved, pp.first)
                end
            end
        end
    else
        while !isempty(closed)
            pp = pop!(closed)
            bellman_update(ℱ, 𝒱, M, pp.first)
        end
    end
    return rv
end

function compute_new_depth(ℱ, prob, depth)
    if ℱ.use_probs_for_depth
        return depth + log(prob)
    else
        return depth + 1
    end
end

function lookahead(ℱ::FLARESSolver,
                   𝒱::ValueIterationSolver,
                   M,
                   s::Integer,
                   a::Integer)
    S, A, T, R, H, V = M.S, M.A, M.T, M.R, M.H, ℒ.V
    T = T(M,S[s],A[a])

    q = 0.
    for i=1:length(S)
        if T[i] == 0
            continue
        end
        if haskey(ℱ.π, i)
            q += T[i] * V[i]
        else
            # continue
            q += T[i] * H(M, 𝒱.V, S[s], A[a])
        end
    end
    return q + R(M,S[s],A[a])
end

function backup(ℱ::FLARESSolver,
                𝒱::ValueIterationSolver,
                M,
                s::Integer)
    for a = 1:length(M.A)
        ℱ.Qs[a] = lookahead(ℱ, 𝒱, M, s, a)
    end
    a = Base.argmax(ℱ.Qs)
    return a, ℱ.Qs[a]
end

function bellman_update(ℱ::FLARESSolver,
                        𝒱::ValueIterationSolver,
                        M,
                        s::Integer)
    a, q = backup(ℱ, 𝒱,  M, s)
    residual = abs(ℱ.V[s] - q)
    ℱ.V[s] = q
    ℱ.π[s] = a
    return residual
end

function solve(ℱ::FLARESSolver,
               𝒱::ValueIterationSolver,
               M,
               s::Integer)

    ℱ.horizon = 0
    while true
        trials = 0
        while (!labeled_solved(s) && trials < ℱ.max_trials)
            trial(s)
        end
    end
    return ℱ.π[s], size(ℱ.dsolved)
end
