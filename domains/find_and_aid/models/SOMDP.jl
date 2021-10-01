using Combinatorics
using Statistics
using Random
using TimerOutputs
using ProfileView

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

function ==(a::MemoryAction, b::DomainAction)
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
    for (i, a) ∈ enumerate(A)
        Aindex[a] = i
    end
    Sindex = Dict{MemoryState, Integer}()
    for (i, a) ∈ enumerate(S)
        Sindex[a] = i
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
    # push!(S, MemoryState(DomainState(-1, -1, '↑', -1, Integer[-1,-1,-1]),
    #                      DomainAction[DomainAction("aid")]))
    return S, S[s₀]
end

function terminal(ℳ::SOMDP, state::MemoryState)
    return terminal(state)
end

function terminal(state::MemoryState)
    return terminal(state.state)
end

function generate_actions(M::MDP)
    A = [MemoryAction(a.value) for a in M.A]
    push!(A, MemoryAction("QUERY"))
    return A
end

function eta(state::MemoryState)
    return 1 - (0.3 * state.state.𝓁)
end

function eta(state::DomainState)
    return 1 - (0.3 * state.𝓁)
end

function eta(action::MemoryAction,
             state′::MemoryState)
    return 1 - (0.3 * state′.state.𝓁)
end

function generate_transitions(ℳ::SOMDP)
    M, S, A, T, δ = ℳ.M, ℳ.S, ℳ.A, ℳ.T, ℳ.δ
    for (s, state) in enumerate(S)
        if length(state.action_list) < δ - 1
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
            if round(mass; digits=5) != 1.0
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
                push!(T, (ms′, 1.0 - round(mass; digits=5)))
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
        push!(T, (ms′, round(1.0-mass; digits=5)))
    end
    return T
end

function generate_reward(ℳ::SOMDP, s::Int, a::Int)
    M, S, A = ℳ.M, ℳ.S, ℳ.A
    state, action = S[s], A[a]
    if action.value == "QUERY"
        return (-.2 * sum(state.state.𝒫))
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

function simulate(ℳ::SOMDP,
                   𝒱::ValueIterationSolver)
    M, S, A, R, state = ℳ.M, ℳ.S, ℳ.A, ℳ.R, ℳ.s₀
    true_state, G = M.s₀, M.G
    rewards = Vector{Float64}()
    for i = 1:10
        episode_reward = 0.0
        while true_state ∉ G
            if length(state.action_list) > 0
                cum_cost += 3
                state = MemoryState(true_state, Vector{CampusAction}())
            else
                s = ℳ.Sindex[state]
                true_s = ℳ.Sindex[true_state]index(true_state, M.S)
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
            s = index(state, S)
            a = 𝒮.π[s]
            action = A[a]
            if v
                println("Taking action $action in memory state $state
                                               in true state $true_state.")
            end
            if action.value == "QUERY"
                episode_reward += R(ℳ,s,a)
                state = MemoryState(true_state, Vector{DomainAction}())
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

            if terminal(state) || terminal(true_state)
                if v
                    println("Terminating in state $state and
                                       true state $true_state.")
                end
                break
            end
        end
        push!(r, episode_reward)
        # println("Episode $i || Total cumulative reward:
        #              $(mean(episode_reward)) ⨦ $(std(episode_reward))")
    end
    # println("Reached the goal.")
    println("Total cumulative reward: $(round(mean(r);digits=4)) ⨦ $(std(r))")
end
#
function simulate(ℳ::SOMDP,
                   𝒱::ValueIterationSolver,
                   π::MCTSSolver)
    M, S, A, R = ℳ.M, ℳ.S, ℳ.A, ℳ.R
    rewards = Vector{Float64}()
    # println("Expected cost to goal: $(ℒ.V[index(state, S)])")
    for i=1:1
        state, true_state = ℳ.s₀, M.s₀
        r = 0.0
        while true
            # s = index(state, S)
            # a, _ = solve(ℒ, 𝒱, ℳ, s)
            action = solve(π, state)
            # action = A[a]
            println("Taking action $action in memory state $state
                                           in true state $true_state.")
            if action.value == "QUERY"
                state = MemoryState(true_state, Vector{DomainAction}())
                r -= 3
            else
                true_s = index(true_state, M.S)
                a = index(action, A)
                r += M.R[true_s][a]
                state = generate_successor(ℳ, state, action)
                if length(state.action_list) == 0
                    true_state = state.state
                else
                    true_state = generate_successor(M, true_s, a)
                end
            end
            if terminal(state) || terminal(true_state)
                println("Terminating in state $state and
                                   true state $true_state.")
                break
            end
        end
        push!(rewards, r)
        # println("Episode $i  Total cumulative cost: $(mean(costs)) ⨦ $(std(costs))")
    end
    # println("Reached the goal.")
    println("Average reward: $(mean(costs)) ⨦ $(std(costs))")
end

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

## TODO: Right now this will only build models for one heuristic.
#        To change -- we would need to make the heuristic "settable" afte
#        building the SOMDP to avoid recomputing the transition dictionary
function build_models(M::MDP,
                 DEPTHS::Vector{Int})
    MODELS = Vector{SOMDP}()
    A = generate_actions(M)
    T = Dict{Int, Dict{Int, Vector{Tuple{Int, Float64}}}}()
    S, s₀ = generate_states(M, 1)
    println(">>>> Building SOMDP for depth δ = 1 <<<<")
    ℳ = SOMDP(M, S, A, T, generate_reward, s₀, 1, generate_heuristic)
    generate_transitions(ℳ)
    push!(MODELS, ℳ)
    tmp_ℳ = ℳ
    for δ in DEPTHS
        println(">>>> Building SOMDP for depth δ = $δ <<<<")
        S, s₀ = generate_states(M, δ)
        println(">>>> Number of states: $(length(S)) <<<<")
        ℳ = SOMDP(M, S, A, copy(tmp_ℳ.T), generate_reward, s₀, δ, generate_heuristic)
        @time generate_transitions(ℳ)
        push!(MODELS, ℳ)
        tmp_ℳ = ℳ
    end
    return MODELS
end

function solve_model(ℳ, 𝒱, solver)
    S, s = ℳ.S, ℳ.s₀

    if solver == "laostar"
        ℒ = LAOStarSolver(100000, 1000., 1.0, .001, Dict{Integer, Integer}(),
            zeros(length(ℳ.S)), zeros(length(ℳ.S)),
            zeros(length(ℳ.S)), zeros(length(ℳ.A)))
        a, total_expanded = solve(ℒ, 𝒱, ℳ, index(s, S))
        println("LAO* expanded $total_expanded nodes.")
        println("Expected reward: $(ℒ.V[index(s,S)])")
        return ℒ
    elseif solver == "uct"
        𝒰 = UCTSolver(zeros(length(ℳ.S)), Set(), 1000, 100, 0)
        a = solve(𝒰, 𝒱, ℳ)
        println("Expected reward: $(𝒰.V[index(s, S)])")
        return 𝒰
    elseif solver == "mcts"
        U(state) = maximum(generate_heuristic(ℳ, 𝒱.V, state, action)
                                                  for action in ℳ.A)
        U(state, action) = generate_heuristic(ℳ, 𝒱.V, state, action)
        π = MCTSSolver(ℳ, Dict(), Dict(), U, 20, 100, 100.0)
        a = solve(π, s)
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
        a, num = solve(ℱ, 𝒱, ℳ, index(s, S))
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

function reachability(ℳ::SOMDP, δ::Int, 𝒮::LAOStarSolver)
    S, state₀, A, T = ℳ.S, ℳ.s₀, ℳ.A, ℳ.T
    s = index(state₀, S)
    π = 𝒮.π

    reachable = Set{Int}()
    reachable_max_depth = Set{Int}()
    visited = Vector{Int}()
    push!(visited, s)
    while !isempty(visited)
        s = pop!(visited)
        if s ∈ reachable
            continue
        end
        push!(reachable, s)
        if length(S[s].action_list) == δ
            push!(reachable_max_depth, s)
        end
        if terminal(S[s])
            continue
        end
        a = π[s]
        for (s′, p) in T[s][a]
            push!(visited, s′)
        end
    end
    count = 0
    for (s, state) in enumerate(S)
        if length(state.action_list) == δ
            count += 1
        end
    end

    println("Reachable max depth states under optimal policy: $(length(reachable_max_depth))")
    # println("Percent of total max depth states reachable under optimal policy: $(length(reachable_max_depth)/(length(S) * (length(A)^δ)))")
    println("Percent of total max depth states reachable under optimal policy: $(length(reachable_max_depth)/count)")
end

function run_experiment_script()
    ## PARAMS
    MAP_PATH = joinpath(@__DIR__, "..", "maps", "collapse_2.txt")
    SOLVERS = ["laostar"]
    HEURISTICS = ["vstar", "null"]
    SIM_COUNT = 100
    VERBOSE = false
    ## delta = 1 is always done by default so don't add here.
    DEPTHS = [2,3]

    # PEOPLE_LOCATIONS = [(2,2), (4,7), (3,8)] # COLLAPSE 1
    PEOPLE_LOCATIONS = [(7, 19), (10, 12), (6, 2)] # COLLAPSE 2

    println("Building MDP...")
    M = build_model(MAP_PATH, PEOPLE_LOCATIONS)
    println("Solving MDP...")
    𝒱 = solve_model(M)
    println("Building SOMDPs...")
    MODELS = build_models(M, DEPTHS)
    println("Solving and Evaluating SOMDPS...")
    logger =
    to = TimerOutput()
    solvers = Vector{Union{FLARESSolver, LAOStarSolver}}
    for solver in SOLVERS
        for model in MODELS
            println("\n", ">>>> Solving SOMDP with depth δ = $(model.δ) <<<<")
            ## TODO: Line below needs to be adjust eventually when we add in
            #        iterating over the different heuristics.
            label = solver * " | " * string(model.δ)
            # println(length(model.S))
            𝒮 = @timeit to label solve(model, 𝒱, solver)

            println("\n", ">>>> Evaluating with depth = $(model.δ) and solver = $solver <<<<")
            simulate(model, 𝒱, 𝒮, SIM_COUNT, VERBOSE)

        end
    end

    show(to, allocations = false)
end

function run_somdp()
    ## PARAMS
    # MAP_PATH = joinpath(@__DIR__, "..", "maps", "collapse_2.txt")
    SOLVER = "laostar"
    SIM = true
    SIM_COUNT = 100
    VERBOSE = false
    DEPTH = 1

    ## EXPERIMENT FlAGS
    REACHABILITY = false
    DELTA_COMPARISON = false
    SOLUTION_COMPARISON = false

    # PEOPLE_LOCATIONS = [(2,2), (4,7), (3,8)] # COLLAPSE 1
    PEOPLE_LOCATIONS = [(7, 19), (10, 12), (6, 2)] # COLLAPSE 2

    ## MAIN SCRIPT
    println("Building MDP...")
    M = build_model(MAP_PATH, PEOPLE_LOCATIONS)
    println("Solving MDP...")
    𝒱 = solve_model(M)
    println("Building SOMDP...")
    ℳ = @time build_model(M, DEPTH)
    println("Total state: $(length(ℳ.S))")
    println("Solving SOMDP...")
    solver = @time solve(ℳ, 𝒱, SOLVER)

    if SIM
        println("Simulating...")
        simulate(ℳ, 𝒱, solver, SIM_COUNT, VERBOSE)
    end

    ## Experiment Below Here ##

    if REACHABILITY
        reachability(ℳ, DEPTH, solver)
    end

end

@profview run_experiment_script()

# run_somdp()
