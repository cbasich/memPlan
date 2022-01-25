using Combinatorics
using Statistics
using Random
using TimerOutputs
# using ProfileView

import Base.==

include("MDP.jl")
include(joinpath(@__DIR__, "..", "..", "..", "solvers", "VIMDPSolver.jl"))
include(joinpath(@__DIR__, "..", "..", "..", "solvers", "LAOStarSolver.jl"))
include(joinpath(@__DIR__, "..", "..", "..", "solvers", "UCTSolverMDP.jl"))
include(joinpath(@__DIR__, "..", "..", "..", "solvers", "MCTSSolver.jl"))
include(joinpath(@__DIR__, "..", "..", "..", "solvers", "FLARESSolver.jl"))


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

function ==(a::MemoryState, b::MemoryState)
    return (a.state == b.state && a.action_list == b.action_list)
end

function Base.hash(a::MemoryState, h::UInt)
    h = hash(a.state, h)
    for act ∈ a.action_list
        h = hash(act, h)
    end
    return h
end

struct MemoryAction
    value::Union{String,Char}
end

function Base.hash(a::MemoryAction, h::UInt)
    return hash(a.value, h)
end

function ==(a::MemoryAction, b::MemoryAction)
    return isequal(a.value, b.value)
end

struct SOMDP
    M::MDP
    S::Vector{MemoryState}
    A::Vector{MemoryAction}
    T::Dict{Int, Dict{Int, Vector{Tuple{Int, Float64}}}}
    R::Function
   s₀::MemoryState
    δ::Integer
    H::Function
    Sindex::Dict{MemoryState, Integer}
    Aindex::Dict{MemoryAction, Integer}
end
function SOMDP(M::MDP,
               S::Vector{MemoryState},
               A::Vector{MemoryAction},
               T::Dict{Int, Dict{Int, Vector{Tuple{Int, Float64}}}},
               R::Function,
               s₀::MemoryState,
               δ::Integer,
               H::Function)

    Aindex, Sindex = generate_index_dicts(A, S)
    ℳ = SOMDP(M, S, A, T, R, s₀, δ, H, Sindex, Aindex)
end

function generate_index_dicts(A::Vector{MemoryAction}, S::Vector{MemoryState})
    Aindex = Dict{MemoryAction, Integer}()
    for (a, action) ∈ enumerate(A)
        Aindex[action] = a
    end
    Sindex = Dict{MemoryState, Int64}()
    for (s, state) ∈ enumerate(S)
        Sindex[state] = s
    end
    return Aindex, Sindex
end

function generate_states(M::MDP, δ::Integer)
    A = M.A

    S = Vector{MemoryState}()
    s₀ = -1
    for depth in 0:δ
        for (i, state) in enumerate(M.S)
            if depth == 0
                s = MemoryState(state, Vector{DomainAction}())
                push!(S, s)
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
    # push!(S, MemoryState(DomainState(-1, -1, '∅', '∅'), DomainAction[]))
    return S, S[s₀]
end

function terminal(ℳ::SOMDP, state::MemoryState)
    return terminal(state.state, ℳ.M.g)
end

function generate_actions(M::MDP)
    A = [MemoryAction(a.value) for a in M.A]
    push!(A, MemoryAction("QUERY"))
    return A
end

function eta(state::DomainState)
    # return 1.0
    ## two_buildings_map
    x, y = state.x, state.y
    if 9 >= x >= 4 && 14 >= y >= 12
        return 0.1
    elseif x == 6 && y == 18
        return 0.1
    elseif 8 >= x >= 4 && 23 >= y >= 22
        return 0.1
    else
        return 0.9
    end
end

function eta(state::MemoryState)
    return eta(state.state)
end

function eta(action::MemoryAction,
              state::MemoryState)
    return eta(state)
end

function generate_transitions(ℳ::SOMDP, incremental::Bool=false)
    M, S, A, T, δ = ℳ.M, ℳ.S, ℳ.A, ℳ.T, ℳ.δ
    for (s, state) in enumerate(S)
        if incremental && length(state.action_list) < δ - 1
            continue
        end
        T[s] = Dict{Int, Vector{Pair{Int, Float64}}}()
        for (a, action) in Iterators.reverse(enumerate(A))
            T[s][a] = generate_transitions(ℳ, s, a)
        end
    end
end

function check_transition_validity(ℳ::SOMDP)
    M, S, A, T = ℳ.M, ℳ.S, ℳ.A, ℳ.T
    for (s, state) in enumerate(S)
        for (a, action) in enumerate(A)
            mass = 0.0
            for (s′, p) in T[s][a]
                mass += p
            end
            if round(mass; digits=4) != 1.0
                println("Transition error at state $state and action $action.")
                println("State index: $s      Action index: $a")
                println("Total probability mass of $mass.")
                println("Transition vector is the following: \n $(T[s][a])")
                @assert false
            end
        end
    end
end

function generate_transitions(ℳ::SOMDP, s::Int, a::Int)
    M, S, A = ℳ.M, ℳ.S, ℳ.A
    state, action = S[s], A[a]
    if state.state.x == -1
        return [(s, 1.0)]
    end

    T = Vector{Tuple{Int, Float64}}()
    # Inside a domain state
    if isempty(state.action_list)
        if action.value == "QUERY" # Do nothing
            return [(s, 1.0)]
        else
            i = argmax(M.T[s][a])
            if M.T[i] == 1.0
                return [(i, 1.0)]
            else
                mass = 0.0
                for (s′, state′) in enumerate(M.S)
                    p = M.T[s][a][s′]
                    if p == 0.0
                        continue
                    end
                    p *= eta(state′)
                    mass += p
                    push!(T, (s′, round(p; digits=5)))
                end
                ms′ = length(M.S) + length(M.A) * (s-1) + a
                mem_p = 1.0 - round(mass; digits=5)
                if mem_p != 0.0
                    push!(T, (ms′, mem_p))
                end
            end
        end
    elseif action.value == "QUERY"  # Here and below is in memory state
        prev_action = MemoryAction(last(state.action_list).value)
        p_a = ℳ.Aindex[prev_action]
        prev_state = MemoryState(state.state,
                      state.action_list[1:length(state.action_list) - 1])
        p_s = ℳ.Sindex[prev_state]

        len = length(ℳ.M.S)
        tmp = Dict{Int, Float64}()
        for (bs, b) in ℳ.T[p_s][a]
            for (bs′, b′) in ℳ.T[bs][p_a]
                if bs′ > len
                    continue
                end
                if !haskey(tmp, bs′)
                    tmp[bs′] = 0.0
                end
                tmp[bs′] += b * ℳ.M.T[bs][p_a][bs′]
            end
        end
        for k in keys(tmp)
            push!(T, (k, round(tmp[k]; digits=5)))
        end
    elseif length(state.action_list) == ℳ.δ
        return [(length(ℳ.S), 1.0)]
    else # Taking non-query action in memory state before depth δ is reached
        action_list′ = [action for action in state.action_list]
        push!(action_list′, DomainAction(action.value))
        mstate′ = MemoryState(state.state, action_list′)
        ms′ = ℳ.Sindex[mstate′]

        tmp = Dict{Int, Float64}()
        len = length(ℳ.M.S)
        mass = 0.0
        for (bs, b) in ℳ.T[s][length(A)]
            for (bs′, b′) in ℳ.T[bs][a]
                if bs′ > len
                    continue
                end
                if !haskey(tmp, bs′)
                    tmp[bs′] = 0.0
                end
                tmp[bs′] += b * M.T[bs][a][bs′] * eta(S[bs′])
            end
        end
        for k in keys(tmp)
            mass += tmp[k]
            push!(T, (k, round(tmp[k]; digits=5)))
        end

        mem_p = 1.0-round(mass; digits=5)
        if mem_p != 0.0
            push!(T, (ms′, mem_p))
        end
    end
    return T
end

function generate_reward(ℳ::SOMDP, s::Int, a::Int)
    M, S, A = ℳ.M, ℳ.S, ℳ.A
    state, action = S[s], A[a]
    if state.state.x == -1
        return -10
    elseif action.value == "QUERY"
        # if length(state.action_list) == ℳ.δ
        #     return 0.0
        # end
        return -3.0
    elseif length(state.action_list) == 0
        return M.R[s][a]
    else
        r = 0.0
        for (bs, b) in ℳ.T[s][length(A)]
            r += b * ℳ.M.R[bs][a]
        end
        return r
    end
end

function generate_heuristic(ℳ::SOMDP, V::Vector{Float64}, s::Int, a::Int)
    M, S, A = ℳ.M, ℳ.S, ℳ.A
    state, action = S[s], A[a]
    if state.state.x == -1
        return 0.
    end
    if length(state.action_list) == 0
        return V[s]
    else
        h = 0.0
        for (bs, b) in ℳ.T[s][length(A)]
            h += b * V[bs]
        end
        return h
    end
    return 0.
end

function generate_successor(ℳ::SOMDP,
                         state::MemoryState,
                        action::MemoryAction)::MemoryState
    thresh = rand()
    p = 0.
    T = ℳ.T[ℳ.Sindex[state]][ℳ.Aindex[action]]
    for (s′, prob) ∈ T
        p += prob
        if p >= thresh
            return ℳ.S[s′]
        end
    end
end

function generate_successor(ℳ::SOMDP,
                             s::Integer,
                             a::Integer)::Integer
    thresh = rand()
    p = 0.
    T = ℳ.T[s][a]
    for (s′, prob) ∈ T
        p += prob
        if p >= thresh
            return s′
        end
    end
end

# function simulate(ℳ::SOMDP,
#                    𝒱::ValueIterationSolver)
#     M, S, A, R, state = ℳ.M, ℳ.S, ℳ.A, ℳ.R, ℳ.s₀
#     true_state, G = M.s₀, M.G
#     rewards = Vector{Float64}()
#     for i = 1:10
#         episode_reward = 0.0
#         while true_state ∉ G
#             if length(state.action_list) > 0
#                 cum_cost += 3
#                 state = MemoryState(true_state, Vector{CampusAction}())
#             else
#                 s = ℳ.Sindex[state]
#                 true_s = index(true_state, M.S)
#                 a = 𝒱.π[true_s]
#                 action = M.A[a]
#                 memory_action = MemoryAction(action.value)
#                 cum_cost += M.C[true_s][a]
#                 state = generate_successor(ℳ, state, memory_action)
#                 if length(state.action_list) == 0
#                     true_state = state.state
#                 else
#                     true_state = generate_successor(M, true_s, a)
#                 end
#             end
#         end
#     end
#     println("Average cost to goal: $cum_cost")
# end

function simulate(ℳ::SOMDP,
                   𝒱::ValueIterationSolver,
                   𝒮::Union{LAOStarSolver,FLARESSolver},
                   m::Int,
                   v::Bool)
    M, S, A, R = ℳ.M, ℳ.S, ℳ.A, ℳ.R
    r = Vector{Float64}()
    # println("Expected cost to goal: $(ℒ.V[index(state, S)])")
    for i ∈ 1:m
        state, true_state = ℳ.s₀, M.s₀
        episode_reward = 0.0
        while true
            s = ℳ.Sindex[state]
            a = 𝒮.π[s]
            action = A[a]
            if v
                println("Taking action $action in memory state $state in true state $true_state.")
            end
            if action.value == "QUERY"
                # if length(state.action_list) < ℳ.δ
                #     episode_reward -= 3.0
                # end
                state = MemoryState(true_state, Vector{DomainAction}())
                episode_reward -= 3
            else
                true_s = index(true_state, M.S)
                episode_reward += M.R[true_s][a]
                state = generate_successor(ℳ, state, A[a])
                if length(state.action_list) == 0
                    true_state = state.state
                else
                    true_state = generate_successor(M, true_s, a)
                end
            end
            # println("Episode reward so far: $episode_reward")

            if terminal(ℳ, state) || terminal(true_state, ℳ.M.g)
                if v
                    println("Terminating in state $state and true state $true_state.")
                end
                break
            end
        end
        push!(r, episode_reward)
        # println("Episode $i || Total cumulative reward:
        #              $(mean(episode_reward)) ⨦ $(std(episode_reward))")
    end
    # println("Reached the goal.")
    println("Total cumulative reward: $(mean(r)) ⨦ $(std(r))")
end
#
# function simulate(ℳ::SOMDP,
#                    𝒱::ValueIterationSolver,
#                    π::MCTSSolver)
#     M, S, A, R = ℳ.M, ℳ.S, ℳ.A, ℳ.R
#     rewards = Vector{Float64}()
#     # println("Expected cost to goal: $(ℒ.V[index(state, S)])")
#     for i=1:1
#         state, true_state = ℳ.s₀, M.s₀
#         r = 0.0
#         while true
#             # s = index(state, S)
#             # a, _ = solve(ℒ, 𝒱, ℳ, s)
#             action = @time solve(π, state)
#             # action = A[a]
#             println("Taking action $action in memory state $state in true state $true_state.")
#             if action.value == "QUERY"
#                 if length(state.action_list) < ℳ.δ
#                     r  -= 3
#                 end
#                 state = MemoryState(true_state, Vector{DomainAction}())
#                 # r -= 3
#             else
#                 true_s = index(true_state, M.S)
#                 a = index(action, A)
#                 r += M.R[true_s][a]
#                 state = generate_successor(ℳ, state, action)
#                 if length(state.action_list) == 0
#                     true_state = state.state
#                 else
#                     true_state = generate_successor(M, true_s, a)
#                 end
#                 print(r)
#             end
#             if terminal(ℳ, state) || terminal(true_state, ℳ.M.g)
#                 println("Terminating in state $state and true state $true_state.")
#                 break
#             end
#         end
#         push!(rewards, r)
#         # println("Episode $i  Total cumulative cost: $(mean(costs)) ⨦ $(std(costs))")
#     end
#     # println("Reached the goal.")
#     println("Average reward: $(mean(costs)) ⨦ $(std(costs))")
# end

function build_model(M::MDP,
                     δ::Int)
    S, s₀ = generate_states(M, δ)
    A = generate_actions(M)
    T = Dict{Int, Dict{Int, Vector{Tuple{Int, Float64}}}}()
    ℳ = SOMDP(M, S, A, T, generate_reward, s₀, δ, generate_heuristic)
    generate_transitions(ℳ)
    println("Checking transition validity")
    check_transition_validity(ℳ)
    return ℳ
end

function build_models(M::MDP,
                 DEPTHS::Vector{Int})
    MODELS = Vector{SOMDP}()
    A = generate_actions(M)
    T = Dict{Int, Dict{Int, Vector{Tuple{Int, Float64}}}}()
    S, s₀ = generate_states(M, 1)
    println(">>>> Building SOMDP for depth δ = 1 <<<<")
    ℳ = SOMDP(M, S, A, T, generate_reward, s₀, 1, generate_heuristic)
    println(">>>> Total States: $(length(S)) <<<<")
    generate_transitions(ℳ, false)
    push!(MODELS, ℳ)
    tmp_ℳ = ℳ
    for δ in DEPTHS
        println(">>>> Building SOMDP for depth δ = $δ <<<<")
        S, s₀ = generate_states(M, δ)
        println(">>>> Total states: $(length(S)) <<<<")
        ℳ = SOMDP(M, S, A, copy(tmp_ℳ.T), generate_reward, s₀, δ, generate_heuristic)
        @time generate_transitions(ℳ, true)
        push!(MODELS, ℳ)
        tmp_ℳ = ℳ
    end
    return MODELS
end

function solve_model(ℳ, 𝒱, solver)
    S, s = ℳ.S, ℳ.s₀
    println("Solving...")

    if solver == "laostar"
        ℒ = LAOStarSolver(100000, 1000., 1.0, .001, Dict{Integer, Integer}(),
            zeros(length(ℳ.S)), zeros(length(ℳ.S)),
            zeros(length(ℳ.S)), zeros(length(ℳ.A)))
        a, total_expanded = @time solve(ℒ, 𝒱, ℳ, index(s, S))
        println("LAO* expanded $total_expanded nodes.")
        println("Expected reward: $(ℒ.V[index(s,S)])")
        return ℒ
    elseif solver == "uct"
        𝒰 = UCTSolver(zeros(length(ℳ.S)), Set(), 1000, 100, 0)
        a = @time solve(𝒰, 𝒱, ℳ)
        println("Expected reward: $(𝒰.V[index(s, S)])")
        return 𝒰
    elseif solver == "mcts"
        U(state) = maximum(generate_heuristic(ℳ, 𝒱.V, state, action)
                                                  for action in ℳ.A)
        U(state, action) = generate_heuristic(ℳ, 𝒱.V, state, action)
        π = MCTSSolver(ℳ, Dict(), Dict(), U, 20, 100, 100.0)
        a = @time solve(π, s)
        println("Expected reard: $(π.Q[(s, a)])")
        return π, a
    elseif solver == "flares"
        ℱ = FLARESSolver(100000, 2, false, false, -1000, 0.001,
                         Dict{Integer, Integer}(),
                         zeros(length(ℳ.S)),
                         zeros(length(ℳ.S)),
                         Set{Integer}(),
                         Set{Integer}(),
                         zeros(length(ℳ.A)))
        a, num = @time solve(ℱ, 𝒱, ℳ, index(s, S))
        println("Expected reward: $(ℱ.V[index(s, S)])")
        return ℱ
    end
end

function solve(ℳ, 𝒱, solver::String)
    if solver == "laostar"
        return solve_model(ℳ, 𝒱, solver)
    elseif solver == "uct"
        return solve_model(ℳ, 𝒱, solver)
    elseif solver == "mcts"
        return solve_model(ℳ, 𝒱, solver)
    elseif solver == "flares"
        return solve_model(ℳ, 𝒱, solver)
    else
        println("Error.")
    end
end

# This is for Connor's benefit running in IDE

function run_somdp()
    ## PARAMS
    MAP_PATH = joinpath(@__DIR__, "..", "maps", "two_buildings.txt")
    SOLVER = "laostar"
    SIM = false
    SIM_COUNT = 100
    VERBOSE = false
    DEPTH = 4
    INIT = 's'
    GOAL = 'g'


    ## MAIN SCRIPT
    begin
        println("Building MDP...")
        M = build_model(MAP_PATH, INIT, GOAL)
        println(">>>> Number of states: $(length(M.S)) <<<<")
        println("Solving MDP...")
        𝒱 = solve_model(M)
        println("Building SOMDP...")
        ℳ = @time build_model(M, DEPTH)
        println("Solving SOMDP...")
        solver = solve(ℳ, 𝒱, SOLVER)
    end

    if SIM
        println("Simulating...")
        simulate(ℳ, 𝒱, solver, SIM_COUNT, VERBOSE)
    end
end

function reachability(ℳ::SOMDP, δ::Int, 𝒮::LAOStarSolver)
    S, state₀, A, T = ℳ.S, ℳ.s₀, ℳ.A, ℳ.T
    s = index(state₀, S)
    π = 𝒮.π

    denominator = 0
    state_stack = Vector{Int}()
    visited_states = Set{Int}()
    push!(state_stack, s)
    while !isempty(state_stack)
        s = pop!(state_stack)
        push!(visited_states, s)
        if length(S[s].action_list) == ℳ.δ
            denominator += 1
        end
        for (a, action) in enumerate(A)
            for (sp, p) in T[s][a]
                if sp ∉ visited_states
                    push!(state_stack, sp)
                end
            end
        end
    end

    reachable = Set{Int}()
    reachable_states = Set{MemoryState}()
    reachable_max_depth = 0
    reachable_max_depths = zeros(δ)
    visited = Vector{Int}()
    push!(visited, s)
    while !isempty(visited)
        s = pop!(visited)
        if s ∈ reachable
            continue
        end
        push!(reachable, s)
        push!(reachable_states, S[s])
        if length(S[s].action_list) > 0
            reachable_max_depths[length(S[s].action_list)] += 1
            # push!(reachable_max_depths[δ], s)
            if length(S[s].action_list) == δ
                reachable_max_depth += 1
            end
        end
        if terminal(ℳ, S[s])
            continue
        end
        if !haskey(π, s)
            continue
        end
        a = π[s]
        for (s′, p) in T[s][a]
            push!(visited, s′)
        end
    end
    # count = 0
    # for (s, state) in enumerate(S)
    #     if length(state.action_list) == δ
    #         count += 1
    #     end
    # end

    println("Reachable max depth states under optimal policy: $reachable_max_depths")
    # println("Percent of total max depth states reachable under optimal policy: $(length(reachable_max_depth)/(length(S) * (length(A)^δ)))")
    println("Percent of total max depth states reachable under optimal policy: $(100.0*length(reachable_max_depth)/denominator)")
    return reachable_states
end

function action_change_experiment(ℳ₁, ℳ₂, 𝒮₁, 𝒮₂)
    S1, S2 = ℳ₁.S, ℳ₂.S

    non_max_term = Set{MemoryState}()
    for (s, state) in enumerate(S1)
        if length(state.action_list) != ℳ₁.δ-1 || !haskey(𝒮₁.π, s)
            continue
        end
        a = 𝒮₁.π[s]
        terminal = true
        for (sp, p) in ℳ₁.T[s][a]
            if length(S1[sp].action_list) == ℳ₁.δ
                terminal = false
            end
        end
        if terminal == true
            push!(non_max_term, state)
        end
    end

    bad_no_good_counterexamples = Set{MemoryState}()
    for state in non_max_term
        s = index(state, S2)
        a = 𝒮₂.π[s]
        for (sp, p) in ℳ₁.T[s][a]
            if length(S2[sp].action_list) == ℳ₁.δ
                push!(bad_no_good_counterexamples, state)
                break
            end
        end
    end

    println("Number of counterexamples is: $(length(bad_no_good_counterexamples))")
end

function run_experiment_script()
    ## PARAMS
    MAP_PATH = joinpath(@__DIR__, "..", "maps", "two_buildings.txt")
    SOLVERS = ["laostar"]
    HEURISTICS = ["vstar", "null"]
    SIM_COUNT = 100
    VERBOSE = false
    ## delta = 1 is always done by default so don't add here.
    DEPTHS = [2,3,4]
    INIT = 's'
    GOAL = 'g'


    println("Building MDP...")
    M = build_model(MAP_PATH, INIT, GOAL)
    println("String buffer")
    println("Number of states: $(length(M.S))")
    println("Solving MDP...")
    𝒱 = solve_model(M)
    println(index(M.s₀, M.S))
    println("Expected reward in base domain: $(𝒱.V[index(M.s₀, M.S)])")
    println("Building SOMDPs...")
    MODELS = build_models(M, DEPTHS)
    println("Solving and Evaluating SOMDPS...")
    to = TimerOutput()
    solvers = Vector{Union{FLARESSolver, LAOStarSolver}}()
    for solver in SOLVERS
        for i=1:length(MODELS)
            model = MODELS[i]
            println("\n", ">>>> Solving SOMDP with depth δ = $(model.δ) <<<<")
            ## TODO: Line below needs to be adjust eventually when we add in
            #        iterating over the different heuristics.
            label = solver * " | " * string(model.δ)
            𝒮 = @timeit to label solve(model, 𝒱, solver)
            # if i > 1
            #     action_change_experiment(MODELS[i-1], model, last(solvers), 𝒮)
            # end
            push!(solvers, 𝒮)

            # println("\n", ">>>> Evaluating with depth = $(model.δ) and solver = $solver <<<<")
            # simulate(model, 𝒱, 𝒮, SIM_COUNT, VERBOSE)

            reachability(model, model.δ, 𝒮)

        end
    end

    show(to, allocations = false)
end

run_experiment_script()
#
run_somdp()
