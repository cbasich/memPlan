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
        if terminal(M.S[current_state])
            break
        end
        push!(visited, current_state)
        bellman_update(ℱ, 𝒱, M, current_state)

        if (accumulated_cost <= ℱ.dead_end_cost)
            break
        end

        greedy_action = get_greedy_action(ℱ, 𝒱, M, current_state)
        accumulated_cost += M.R(M, current_state, greedy_action)
        current_state::Integer = generate_successor(M, current_state, greedy_action)
    end
    while (!isempty(visited))
        current_state = pop!(visited)
        solved = check_solved(ℱ, 𝒱, M, current_state)
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
    a = ℱ.π[s]
    res = lookahead(ℱ, 𝒱, M, s, a) - ℱ.V[s]
    return abs(res)
end

function get_greedy_action(ℱ, 𝒱, M, s::Integer)
    if haskey(ℱ.π, s)
        return ℱ.π[s]
    else
        besta = -1
        bestq = ℱ.dead_end_cost #needs to be negative!!!
        for a ∈ length(M.A)
            tmp = lookahead(ℱ, 𝒱, M, s, a)
            if tmp > bestq
                bestq = tmp
                besta = a
            end
        end
        return besta
    end
end

function check_solved(ℱ, 𝒱, M, s::Integer)
    open = []
    closed = []
    closed_states = []

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

        if (ℱ.use_probs_for_depth && (depth < 2 * log(ℱ.horizon)) || (!ℱ.use_probs_for_depth && depth > 2 * ℱ.horizon))
            subgraph_within_search_horizon = false
            continue
        end

        if (terminal(M.S[current_state]))
            continue
        end

        push!(closed, pp)
        push!(closed_states, pp.first)
        a = get_greedy_action(ℱ, 𝒱, M, current_state)
        if (residual(ℱ, 𝒱, M, current_state) > ℱ.ϵ)
            rv = false
        end
        successor_probs = M.T[s][a]
        for (sp, prob) in successor_probs
            if !labeled_solved(ℱ, sp) && sp ∉ closed_states
                new_depth = compute_new_depth(ℱ, prob, depth)
                push!(open, sp => new_depth)
            elseif sp ∈ ℱ.dsolved && sp ∉ ℱ.solved
                subgraph_within_search_horizon = false
            end
        end
    end

    if rv
        while !isempty(closed)
            pp = pop!(closed)
            _ = pop!(closed_states)
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
            _ = pop!(closed_states)
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
    S, A, T, R, H, V = M.S, M.A, M.T[s][a], M.R, M.H, ℱ.V

    q = 0.
    for (s′, p) in T
       if haskey(ℱ.π, s′)
           q += p * V[s′]
       else
           q += p * H(M, 𝒱.V, s, a)
       end
    end
    return q + R(M,s,a)
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

    trials = 0
    while (!labeled_solved(ℱ,s) && trials < ℱ.max_trials)
        # println("trial: ", trials)
        trial(ℱ, 𝒱, M, s)
        trials += 1
    end
    for (s, state) ∈ enumerate(M.S)
        ℱ.π[s] = get_greedy_action(ℱ, 𝒱, M, s)
    end
    return ℱ.π[s], length(ℱ.dsolved)
end
