using Combinatorics
using Statistics
import Base.==

include("MDP.jl")
include(joinpath(@__DIR__, "..", "..", "..", "solvers", "VIMDPSolver.jl"))
include(joinpath(@__DIR__, "..", "..", "..", "solvers", "LAOStarSolver.jl"))

function index(element, collection)
    for i=1:length(collection)
        if collection[i] == element
            return i
        end
    end
    return -1
end

struct MemoryState
    state::DomainState
    action_list::Vector{DomainAction}
end

function ==(s₁::MemoryState, s₂::MemoryState)
    return (s₁.state == s₂.state && s₁.action_list == s₂.action_list)
end

struct MemoryAction
    value::Union{String,Char}
end

struct SOMDP
    M::MDP
    S::Vector{MemoryState}
    A::Vector{MemoryAction}
    T::Function
    R::Function
   s₀::MemoryState
    τ::Dict{Int, Dict{Int, Dict{Int, Float64}}}
    δ::Integer
    H::Function
end

function generate_states(M::MDP, δ::Integer)
    A = M.A

    S = Vector{MemoryState}()
    G = Vector{MemoryState}()
    s₀ = -1
    for depth in 0:δ
        for (i, state) in enumerate(M.S)
            if depth == 0
                s = MemoryState(state, Vector{DomainAction}())
                push!(S, s)
                if state in M.G
                    push!(G, s)
                end
                if state == M.s₀
                    s₀ = length(S)
                end
            else
                for action_list ∈ collect(Base.product(ntuple(i->A, depth)...))
                    s = MemoryState(state, [a for a ∈ action_list])
                    push!(S, s)
                end
            end
        end
    end
    return S, S[s₀], G
end

function generate_actions(M::MDP)
    A = [MemoryAction(a.value) for a in M.A]
    push!(A, MemoryAction("QUERY"))
    return A
end

function eta(state::MemoryState,
            action::MemoryAction)
    return 0.3 * state.ℒ
end

function recurse_transition(ℳ::SOMDP,
                         state::MemoryState,
                        action::MemoryAction,
                        state′::MemoryState)::Float64
    s, a, s′ = index(state, ℳ.S), index(action, ℳ.A), index(state′, ℳ.S)
    if isempty(state.action_list)
        return ℳ.M.T[s][a][s′]
    end

    if s ∈ keys(ℳ.τ)
        if a ∈ keys(ℳ.τ[s])
            if s′ ∈ keys(ℳ.τ[s][a])
                return ℳ.τ[s][a][s′]
            end
        else
            ℳ.τ[s][a] = Dict{Int, Float64}()
        end
    else
        ℳ.τ[s] = Dict(a => Dict{Int, Float64}())
    end

    actionₚ = MemoryAction(last(state.action_list).value)
    stateₚ = MemoryState(state.state,
                         state.action_list[1:length(state.action_list)-1])
    p = 0.

    for bs=1:length(ℳ.M.S)
        q = ℳ.M.T[bs][a][s′]
        if q ≠ 0.
            p += q * recurse_transition(ℳ, stateₚ, actionₚ, ℳ.S[bs])
        end
    end

    ℳ.τ[s][a][s′] = p
    return p
end

function generate_transitions(ℳ::SOMDP,
                           state::MemoryState,
                          action::MemoryAction)
    M, S, A = ℳ.M, ℳ.S, ℳ.A
    T = zeros(length(S))
    if isempty(state.action_list)
        s, a = index(state, S), index(action, A)
        if action.value == "QUERY"
            T[s] = 1.
            return T
        elseif maximum(M.T[s][a]) == 1.
            T[argmax(M.T[s][a])] = 1.
            return T
        else
            ms′ = length(M.S) + length(M.A) * (s-1) + a
            T[ms′] = eta(state, action)
            for (s′, state′) in enumerate(M.S)
                T[s′] = M.T[s][a][s′] * (1 - T[ms′])
            end
        end
    elseif action.value == "QUERY"
        actionₚ = MemoryAction(last(state.action_list).value)
        stateₚ = MemoryState(state.state,
                             state.action_list[1:length(state.action_list)-1])
        for s′ = 1:length(M.S)
            T[s′] = recurse_transition(ℳ, stateₚ, actionₚ, S[s′])
        end
    elseif length(state.action_list) == ℳ.δ
        T[length(M.S)] = 1.
    else
        action_list′ = copy(state.action_list)
        push!(action_list′, DomainAction(action.value))
        mstate′ = MemoryState(state.state, action_list′)
        T[index(mstate′, S)] = .75
        for s′ = 1:length(M.S)
            T[s′] = 0.25recurse_transition(ℳ, state, action, S[s′])
        end
    end
    return T
end

function generate_reward(ℳ::SOMDP,
                      state::MemoryState,
                     action::MemoryAction)
    M, S, A = ℳ.M, ℳ.S, ℳ.A
    if action.value == "QUERY"
        return 3.  ## TODO: Adjust this cost somehow??
    elseif length(state.action_list) == 0
        return M.C[index(state, S)][index(action, A)]
    else
        a = index(action, A)
        actionₚ = MemoryAction(last(state.action_list).value)
        stateₚ = MemoryState(state.state,
                             state.action_list[1:length(state.action_list)-1])
        return (sum(M.C[bs][a] * recurse_transition(ℳ, stateₚ, actionₚ, S[bs])
                                                      for bs = 1:length(M.S)))
    end
end

function generate_heuristic(ℳ::SOMDP,
                             V::Vector{Float64},
                         state::MemoryState,
                        action::MemoryAction)
    M, S, A = ℳ.M, ℳ.S, ℳ.A
    if length(state.action_list) == 0
        return V[index(state, S)]
    else
        actionₚ = MemoryAction(last(state.action_list).value)
        stateₚ = MemoryState(state.state,
                            state.action_list[1:length(state.action_list)-1])
        h = 0.0
        for bs = 1:length(M.S)
            v = V[bs]
            if v ≠ 0.0
                h += v * recurse_transition(ℳ, stateₚ, actionₚ, S[bs])
            end
        end
        return h
        # return (sum(V[bs] * recurse_transition(ℳ, stateₚ, actionₚ, S[bs])
        #                                          for bs = 1:length(M.S)))
    end
    return 0.
end

function generate_successor(ℳ::SOMDP,
                         state::MemoryState,
                        action::MemoryAction)
    thresh = rand()
    p = 0.
    T = ℳ.T(ℳ, s, a)
    for (s′, state′) ∈ enumerate(ℳ.S)
        p += T[s′]
        if p >= thresh
            return state′
        end
    end
end

function simulate(ℳ::SOMDP,
                   𝒱::ValueIterationSolver)
    M, S, A, R, state = ℳ.M, ℳ.S, ℳ.A, ℳ.R, ℳ.s₀
    true_state, G = M.s₀, M.G
    rewards = Vector{Float64}()
    for i = 1:100
        episode_reward = 0.0
        while true_state ∉ G
            if length(state.action_list) > 0
                cum_cost += 3
                state = MemoryState(true_state, Vector{CampusAction}())
            else
                s = index(state, S)
                true_s = index(true_state, M.S)
                a = 𝒱.π[true_s]
                action = M.A[a]
                memory_action = MemoryAction(action.value)
                cum_cost += M.C[true_s][a]
                state = generate_successor(ℳ, state, memory_action)
                if length(state.action_list) == 0
                    true_state = state.state
                else
                    true_state = generate_successor(M, true_s, a)
                end
            end
        end
    end
    println("Average cost to goal: $cum_cost")
end

# function simulate(ℳ::MemorySSP, ℒ::LAOStarSolver, 𝒱::ValueIterationSolver)
#     M, S, A, C = ℳ.M, ℳ.S, ℳ.A, ℳ.C
#     costs = Vector{Float64}()
#     # println("Expected cost to goal: $(ℒ.V[index(state, S)])")
#     for i=1:100
#         state, true_state, G = ℳ.s₀, M.s₀, M.G
#         episode_cost = 0.0
#         while true_state ∉ G
#             s = index(state, S)
#             a, _ = solve(ℒ, 𝒱, ℳ, s)
#             action = A[a]
#             # println("Taking action $action in memory state $state in true state $true_state.")
#             if action.value == "query"
#                 state = MemoryState(true_state, Vector{CampusAction}())
#                 episode_cost += 3
#             else
#                 true_s = index(true_state, M.S)
#                 episode_cost += M.C[true_s][a]
#                 state = generate_successor(ℳ, state, A[a])
#                 if length(state.action_list) == 0
#                     true_state = state.state
#                 else
#                     true_state = generate_successor(M, true_s, a)
#                 end
#             end
#         end
#         push!(costs, episode_cost)
#         println("Episode $i           Total cumulative cost: $(mean(costs)) ⨦ $(std(costs))")
#     end
#     # println("Reached the goal.")
#     println("Total cumulative cost: $(mean(costs)) ⨦ $(std(costs))")
# end
#
# function simulate(ℳ::MemorySSP, 𝒱::ValueIterationSolver, π::MCTSSolver)
#     M, S, A, C = ℳ.M, ℳ.S, ℳ.A, ℳ.C
#     costs = Vector{Float64}()
#     # println("Expected cost to goal: $(ℒ.V[index(state, S)])")
#     for i=1:1
#         state, true_state, G = ℳ.s₀, M.s₀, M.G
#         episode_cost = 0.0
#         while true_state ∉ G
#             # s = index(state, S)
#             # a, _ = solve(ℒ, 𝒱, ℳ, s)
#             action = solve(π, state)
#             # action = A[a]
#             println("Taking action $action in memory state $state in true state $true_state.")
#             if action.value == "query"
#                 state = MemoryState(true_state, Vector{CampusAction}())
#                 episode_cost += 3
#             else
#                 true_s = index(true_state, M.S)
#                 a = index(action, A)
#                 episode_cost += M.C[true_s][a]
#                 state = generate_successor(ℳ, state, action)
#                 if length(state.action_list) == 0
#                     true_state = state.state
#                 else
#                     true_state = generate_successor(M, true_s, a)
#                 end
#             end
#         end
#         push!(costs, episode_cost)
#         println("Episode $i           Total cumulative cost: $(mean(costs)) ⨦ $(std(costs))")
#     end
#     # println("Reached the goal.")
#     println("Total cumulative cost: $(mean(costs)) ⨦ $(std(costs))")
# end

function build_model(M::MDP)
    δ = 1
    S, s₀ = generate_states(M, δ)
    A = generate_actions(M)
    τ = Dict{Int, Dict{Int, Dict{Int, Float64}}}()
    ℳ = MemorySSP(M, S, A, generate_transitions, generate_reward, s₀,
                   τ, δ, generate_heuristic)
    return ℳ, 𝒱
end

function solve_model(ℳ, 𝒱)
    # ℒ = LAOStarSolver(100000, 1000., 1.0, .001, Dict{Integer, Integer}(),
    #      zeros(length(ℳ.S)), zeros(length(ℳ.S)),
    #      zeros(length(ℳ.S)), zeros(length(ℳ.A)))
    # 𝒰 = UCTSolver(zeros(length(ℳ.S)), Set(), 1000, 100, 0)
    U(state) = minimum(heuristic(ℳ, 𝒱.V, state, action) for action in ℳ.A)
    U(state, action) = heuristic(ℳ, 𝒱.V, state, action)

    π = MCTSSolver(ℳ, Dict(), Dict(), U, 10, 10000, 100.0)
    S, s = ℳ.S, ℳ.s₀
    # a, total_expanded = @time solve(ℒ, 𝒱, ℳ, index(s, S))
    # a = @time solve(𝒰, 𝒱, ℳ, )
    a = @time solve(π, s)
    return π, a
    # println("LAO* expanded $total_expanded nodes.")
    # println("Expected cost to goal: $(ℒ.V[index(s,S)])")
    # return ℒ, ℒ.V[index(s, S)]
    # return 𝒰, 𝒰.V[index(s, S)]
end

function main()
    domain_map_file = joinpath(@__DIR__, "..", "maps", "collapse_1.txt")

    println("Starting...")
    M = build_model(domain_map_file)
    V = solve_model(M)

    ℳ, 𝒱 = @time build_model(M)
    # simulate(ℳ, 𝒱)
    println("Solving...")
    println(length(ℳ.S))
    # ℒ, expected_cost = solve_model(ℳ, 𝒱)
    # 𝒰, expected_cost = solve_model(ℳ, 𝒱)
    π, a = solve_model(ℳ, 𝒱)
    expected_cost = π.Q[(ℳ.s₀, a)]
    println("Expected cost from initial state: $expected_cost")
    println("Simulating...")
    # simulate(ℳ, ℒ, 𝒱)
    simulate(ℳ, 𝒱, π)
end

main()
