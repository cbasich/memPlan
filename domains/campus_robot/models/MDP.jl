using Combinatorics
using Statistics

import Base.==

include(joinpath(@__DIR__, "..", "..", "..", "solvers", "VIMDPSolver.jl"))

function index(element, collection)
    for i=1:length(collection)
        if collection[i] == element
            return i
        end
    end
    return -1
end

struct DomainState
    x::Integer
    y::Integer
    θ::Char
    o::Char
end

function ==(a::DomainState, b::DomainState)
    return a.x == b.x && a.y == b.y && a.θ == b.θ && a.𝓁 == b.𝓁 && a.𝒫 == b.𝒫
end

struct DomainAction
    value::Union{String,Char}
end

struct MDP
    S::Vector{DomainState}
    A::Vector{DomainAction}
    T
    R
    s₀
    g
end

function generate_grid(filename::String)
    grid = Vector{Vector{Any}}()
    open(filename) do file
        for l in eachline(file)
            row = collect(l)
            deleteat!(row, row .== ' ')
            push!(grid, row)
        end
    end
    return grid
end

function generate_states(grid::Vector{Vector{Any}},
                         init::Char,
                         goal::Char)
    S = Vector{DomainState}()
    s₀, g = DomainState(), DomainState()

    for (i, row) in enumerate(grid)
        for (j, loc) in enumerate(row)
            if loc == 'X'
                continue
            end
            if loc == init
                s₀ = DomainState(i, j, '↑', '∅')
            end
            if loc == goal
                g = DomainState(i, j, '↑', '∅')
            end
            for θ in ['↑', '↓', '→', '←']
                if loc == '.'
                    push!(S, DomainState(i, j, θ, '∅'))
                elseif loc == 'C'
                    for o in ['E', 'L', 'B']
                        push!(S, DomainState(i, j, θ, o))
                    end
                elseif loc == 'D'
                    for o in ['C', 'O']
                        push!(S, DomainState(i, j, θ, o))
                    end
                else
                    push!(S, DomainState(i, j, θ, '∅'))
                end
            end
        end
    end

    dead_end = CampusState(-1, -1, '∅', '∅')
    push!(S, dead_end)
    return S, s₀, g
end

function terminal(ℳ::MDP, state::DomainState)
    return state == ℳ.g
end

function generate_actions()
    A = Vector{DomainAction}()
    for v in ['↑', '↓', '→', '←', "open", "cross", "wait"]
        push!(A, DomainAction(v))
    end
    return A
end

function pos_shift(dir::Char)
    xp, yp = 0, 0
    if dir == '↑'
        xp -= 1
    elseif dir == '→'
        yp += 1
    elseif dir == '↓'
        xp += 1
    else
        yp -= 1
    end
    return xp, yp
end

function slip_right(dir::Char)
    if dir == '↑'
        return '→'
    elseif dir == '→'
        return '↓'
    elseif dir == '↓'
        return '←'
    else
        return '↑'
    end
end

function slip_left(dir::Char)
    if dir == '↑'
        return '←'
    elseif dir == '→'
        return '↑'
    elseif dir == '↓'
        return '→'
    else
        return '↓'
    end
end

function move_distribution(s::Int,
                      action::DomainAction,
                           S::Vector{DomainState},
                        grid::Vector{Vector{Any}})
    state = S[s]
    xp, yp = pos_shift(a.value)
    xp, yp = state.x + xp, state.y + yp
    # xpr, ypr = pos_shift(slip_right(a.value))
    # xpl, ypl = pos_shift(slip_left(a.value))
    # xp, xpr, xpl = xp + s.x, xpr + s.x, xpl + s.x
    # yp, ypr, ypl = yp + s.y, ypr + s.y, ypl + s.y
    distr = zeros(length(S))

    if grid[xp][yp] == 'X'
        distr[s] = 1.0
        return distr
    end

    for (s′, state′) in enumerate(S)
        if state′.θ ≠ action.value
            continue
        end
        p = 0.0
        if state.o in ['C', 'E', 'L', 'B']
            if action.value == s.θ
                continue
            elseif (state′.x, state′.y) == (xp, yp)
                p = .8
            end
        elseif s.o == 'O' && (state′.x, state′.y) == (xp, yp)
            p = 0.8
        else
            if state′.o == 'C' && (state′.x, state′.y) == (xp, yp)
                p = 0.8
            elseif state′.o == '∅' && (state′.x, state′.y) == (xp, yp)
                p = 0.8
            else
                p = ((xp == s′.x && yp == s′.y) ? 1/3 : 0.)
            end
        end
        distr[s′] = p
    end
    distr[s] += 1.0 - sum(distr)
    return distr
end

function wait_distribution(s::Int,
                           S::Vector{DomainState})
    s = S[s]
    distr = zeros(length(S))
    if state.o ∉ ['E', 'L', 'B']
        distr[s] = 1.0
        return distr
    end

    if state.o == 'E'
        distr[s] = 0.75
        distr[s+1] = 0.25
    elseif state.o == 'L'
        distr[s-1] = 0.25
        distr[s] = 0.5
        distr[s+1] = 0.25
    else
        distr[s] = 0.75
        distr[s-1] = 0.25
    end
    return distr
end

function cross_distribution(s::Int,
                            S::Vector{DomainState})
    state = S[s]
    distr = zeros(length(S))
    if state.o ∉ ['E', 'L', 'B']
        distr[s] = 1.0
        return distr
    end

    state′ = (xp, yp, state.θ, '∅')
    if state.o == 'E'
        distr[index(state′, S)] = 1.0
    elseif state.o == 'L'
        distr[index(state′, S)] = 0.5
        distr[s] = 0.5
    else
        distr[index(state′, S)] = 0.1
        distr[length(S)] = 0.1
        distr[s] = 0.8
    end
    return distr
end

function open_distribution(s::Int,
                           S::Vector{DomainState})
    state = S[s]
    distr = zeros(length(S))
    if state.o ≠ 'C'
        distr[s] = 1.0
        return distr
    else
        distr[s+1] = 1.0
        return distr
    end
    return distr
end

function generate_transitions(S::Vector{DomainState},
                              A::Vector{DomainAction},
                           grid::Vector{Vector{Any}},
                             s₀::DomainState,
                              g::DomainState)
    T = [[[0.0 for (i, _) in enumerate(S)]
               for (j, _) in enumerate(A)]
               for (k, _) in enumerate(S)]

    for (s, state) in enumerate(S)
        # First check if in dead_end
        if s == length(S) || state == g
            for a=1:length(A)
                T[s][a][s] = 1.0
            end
            continue
        end

        for (a, action) in enumerate(A)
            if action.value == "cross"
                T[s][a] = cross_distribution(s, S)
            elseif action.value == "open"
                T[s][a] = open_distribution(s, S)
            elseif action.value == "wait"
                T[s][a] = wait_distribution(s, S)
            else
                T[s][a] = move_distribution(s, action, S, grid)
            end
            # if sum(T[s][a]) ≠ 1.0
            #     print(sum)
            #     T[s][a][s] += (1.0 - sum(T[s][a]))
            # end
        end
    end
    return T
end

function generate_rewards(S::Vector{DomainState},
                          A::Vector{DomainAction},
                          g::DomainState
                       grid::Vector{Vector{Any}})
    R = [[-1.0 for (i, _) in enumerate(A)]
               for (j, _) in enumerate(S)]

    for (s, state) in enumerate(S)
        if state == g
            R[s] .= 0.0
            continue
        elseif s == length(S)
            continue
        else
            for (a, action) in enumerate(A)
                if state.o == 'C' && action.value == state.θ
                    R[s][a] *= 5.0
                elseif typeof(action.value) == Char
                    xp, yp = pos_shift(action.value)
                    xp, yp = state.x + xp, state.y + yp
                    if grid[xp][yp] == 'X'
                        R[s][a] *= 5.0
                    end
                end
            end
        end
    return R
end

function check_transition_validity(T, S, A)
    n, m = length(S), length(A)
    for i in 1:n
        for j in 1:m
            if minimum(T[i][j]) < 0.
                println("Transition error at state index $i and action index $j")
                println("with a minimum probability of $(minimum(T[i][j])).")
                println("State: $(S[i])")
                println("Action: $(A[j])")
                for k in 1:n
                    if T[i][j][k] ≠ 0.0
                        println("Succ State $k: $(S[k])     with probability: $(T[i][j][k])")
                    end
                end
                return 0.
            end
            if roun(sum(T[i][j]); digits = 5) ≥ 1.
                continue
            else
                println("Transition error at state index $i and action index $j")
                println("with a total probability mass of $(sum(T[i][j])).")
                println("State: $(S[i])")
                println("Action: $(A[j])")
                println(T[i][j][i])
                return 0.
            end
        end
    end
end

function build_model(filepath::String, init, goal)
    grid = generate_grid(filepath)
    S, s₀, g = generate_states(grid, init, goal)
    A = generate_actions()
    T = generate_transitions(S, A, grid, init, g)
    check_transition_validity(T, S, A)
    R = generate_rewards(S, A, g, grid)
    ℳ = MDP(S, A, T, R, s₀, g)
    return ℳ
end

function solve_model(ℳ::MDP)
    𝒱 = ValueIterationSolver(.001,
                             true,
                             Dict{Integer,Integer}(),
                             zeros(length(ℳ.S)),
                             Vector{Float64}(undef, length(ℳ.A)))
    solve(𝒱, ℳ)
    return 𝒱
end

function generate_successor(ℳ::MDP, s::Int, a::Int)
    thresh = rand()
    p = 0.
    for (s′, state′) ∈ enumerate(ℳ.S)
        p += ℳ.T[s][a][s′]
        if p >= thresh
            return state′
        end
    end
    println("Getting here?    $p     $(sum(ℳ.T[s][a]))")
    println("state $s and action $a")
end

function simulate(ℳ::MDP, 𝒱::ValueIterationSolver)
    S, A, R = ℳ.S, ℳ.A, ℳ.R
    r = 0.
    for i=1:1
        state = ℳ.s₀
        println("Expected reward: $(𝒱.V[index(state, S)])")
        while true
            s = index(state, S)
            a = 𝒱.π[s]
            r += R[s][a]
            println("Taking action $(A[a]) in state $state.")
            state = generate_successor(ℳ, s, a)
            if terminal(state)
                break
            end
        end
        # println("Reached the goal with total cost $cost.")
    end
    println("Average reward: $(r / 1.0)")
end

## This is here for Connor
# function run_MDP()
#     domain_map_file = joinpath(@__DIR__, "..", "maps", "collapse_2.txt")
#     println("Building Model...")
#     people_locations = [(7, 19), (10, 12), (6, 2)]
#     ℳ = build_model(domain_map_file, people_locations)
#     println("Solving Model...")
#     𝒱 = @time solve_model(ℳ)
#     simulate(ℳ, 𝒱)
# end
#
# run_MDP()
