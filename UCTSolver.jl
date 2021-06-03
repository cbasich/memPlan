using Base

mutable struct UCTNode
    state
    state_id::Int
    depth::Int
    count::Int
    action_count::Vector{Int}
    Q::Vector{Float64}
end
function UCTNode(ℳ, 𝒱, state, depth)
    Q = initialize_q(ℳ, 𝒱, state)
    return UCTNode(state, index(state, ℳ.S), depth, 0, zeros(length(ℳ.A)), Q)
end

function initialize_q(ℳ, 𝒱, state)
    Q = Vector{Float64}()
    for action ∈ ℳ.A
        push!(Q, ℳ.H(ℳ, 𝒱.V, state, action))
    end
    return Q
end

mutable struct UCTSolver
    V::Vector{Float64}
    visited_::Set{Any}
    max_rollouts_::Int
    cutoff_::Int
    start_depth_::Int
end

# Action Selection
function greedy_action_selection(ℳ, node)
    return ℳ.A[argmax(node.Q)]
end

function ucb1_action_selection(ℳ, 𝒰, node)
    best_value = Inf
    best_action = C_NULL
    for (a, action) ∈ enumerate(ℳ.A)
        # if !ℳ.applicable(node.state, ℳ.A[a])
        #     continue
        # end
        u = ucb1(ℳ, 𝒰, node, a)
        if u < best_value
            best_value = u
            best_action = a
        end
    end
    return best_action
end
function ucb1(ℳ, 𝒰, node, a)
    U = 𝒰.V[node.state_id] * sqrt(log(node.count) / node.action_count[a])
    return U + node.Q[a]
end

# Outcome Selection
function select_outcome(ℳ, 𝒰, node, a)
    s = sample_from_distr(ℳ.T(ℳ, node.state, ℳ.A[a]))
    return ℳ.S[s], s
end
function sample_from_distr(distr::Vector{Float64})
    r = rand()
    mass = 0.
    for (i, p) ∈ enumerate(distr)
        mass += p
        if r ≤ mass
            return i
        end
    end
end

function lookahead(𝒰, ℳ, s, a)
    S, A, T, C, H, V = ℳ.S, ℳ.A, ℳ.T, ℳ.C, ℳ.H, 𝒰.V
    T = T(ℳ,S[s],A[a])

    q = 0.
    for i=1:length(S)
        if T[i] == 0.
            continue
        end
        if
        q += T[i] * 𝒰.V[s]
    end
    return q + C(ℳ, S[s], A[a])
end

function solve(𝒰::UCTSolver, 𝒱, ℳ)
    S, A, T, C, s₀ = ℳ.S, ℳ.A, ℳ.T, ℳ.C, ℳ.s₀

    root = UCTNode(ℳ, 𝒱, s₀, 𝒰.start_depth_)
    for r = 1:𝒰.max_rollouts_
        node = root
        cum_cost = Vector{Float32}()
        push!(cum_cost, 0.)
        nodes_in_rollout = Vector{UCTNode}()
        actions_in_rollout = Vector{}()
        maxSteps = 0
        for i = 1:𝒰.cutoff_
            if node ∉ 𝒰.visited_
                push!(𝒰.visited_, node)
                for a = 1:length(A)
                    # if !applicable(ℳ, node.state, a)
                    #     continue
                    # end
                    node.action_count[a] += 1
                    node.count += 1
                    node.Q[a] = lookahead(𝒰, ℳ, node, a)
                end
            end
            if isgoal(ℳ, node.state)
                break
            end
            maxSteps = i
            a = ucb1_action_selection(ℳ, 𝒰, node)
            state′, s′ = select_outcome(ℳ, 𝒰, node, a)
            cost = cum_cost[i] + ℳ.C(ℳ, node.state, ℳ.A[a])
            push!(cum_cost, cost)
            push!(nodes_in_rollout, node)
            push!(actions_in_rollout, a)
            node = UCTNode(ℳ, 𝒱, state′, node.depth + 1)
        end

        for i = 1:maxSteps
            node = nodes_in_rollout[i]
            a = actions_in_rollout[i]
            node.count += 1
            node.action_count[a] += 1

            cum_cost_node = last(cum_cost) - cum_cost[i]
            δ_target = ((cum_cost_node - node.Q[a])
                       / node.action_count[a])
            node.Q[a] += δ_target
        end
    end

    return greedy_action_selection(ℳ, root)
end
