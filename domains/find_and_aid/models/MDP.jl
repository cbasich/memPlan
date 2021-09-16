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
    𝓁::Integer
    𝒫::Vector{Integer}
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
end

function generate_people_smoke_level_vector(grid::Vector{Vector{Any}})
    𝒫 = Vector{Int}()
    for loc in people_locations
        push!(𝒫, parse(Int, grid[loc[1]][loc[2]]))
    end
    return 𝒫
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

## TODO: There is a bug for when people *start* out in '0' smoke level locations
##       We can probably ignore it becuase if they have 0 smoke level we
##       don't do anything with them anyways. Still be careful...
function generate_states(grid::Vector{Vector{Any}})
    S = Vector{DomainState}()
    s₀ = PRESERVE_NONE
    𝒫 = generate_people_smoke_level_vector(grid)
    num_people = length(people_locations)

    for (i, row) in enumerate(grid)
        for (j, loc) in enumerate(row)
            if loc == 'X'
                continue
            end
            if loc == 'S'
                s₀ = DomainState(i, j, '↑', 0, 𝒫)
                loc = '0'
            end
            for θ in ['↑', '↓', '→', '←']
                push!(S, DomainState(i, j, θ, parse(Int, loc), 𝒫))
                for mask in collect(combinations(1:num_people))
                    P = copy(𝒫)
                    P[mask] .= 0
                    push!(S, DomainState(i, j, θ, parse(Int, loc), P))
                end
            end
        end
    end

    for mask in collect(combinations(1:num_people))
        P = copy(𝒫)
        P[mask] .= 0
        push!(S, DomainState(0, 0, '↑', 0, P))
    end
    push!(S, DomainState(0, 0, '↑', 0, 𝒫))
    return S, s₀
end

function terminal(state::DomainState)
    return sum(state.𝒫) == 0
end

function generate_actions()
    A = Vector{DomainAction}()
    for v in ['↑', '↓', '→', '←', "aid"]
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

function move_distribution(state::DomainState,
                          action::DomainAction,
                            grid,
                               S::Vector{DomainState})
    xp, yp = pos_shift(action.value)
    xp, yp = state.x + xp, state.y + yp
    distr = zeros(length(S))

    if xp > length(grid) || yp > length(grid[1])
        distr[index(state, S)] = 1.0
        return distr
    end

    if grid[xp][yp] == 'X'
        outside = DomainState(0, 0, '↑', 0, state.𝒫)
        distr[index(outside, S)] = 0.8
        distr[index(state, S)] = 0.2
        return distr
    end
    # xpr, ypr = pos_shift(slip_right(action.value))
    # xpl, ypl = pos_shift(slip_left(action.value))
    # xp, xpr, xpl = xp + state.x, xpr + state.x, xpl + state.x
    # yp, ypr, ypl = yp + state.y, ypr + state.y, ypl + state.y

    for (s′, state′) in enumerate(S)
        if state′.θ ≠ action.value || state′.𝒫 ≠ state.𝒫
            continue
        end

        if state == state′
            distr[s′] = 0.2
        # elseif state′.x == xpr && state′.y == ypr
        #     distr[s′] = 0.05
        # elseif state′.x == xpl && state′.y == ypl
        #     distr[s′] = 0.05
        elseif state′.x == xp && state′.y == yp
            distr[s′] = 0.8
        end
    end

    distr[index(state, S)] += 1.0 - sum(distr)

    return distr
end

function aid_distribution(state::DomainState,
                          S::Vector{DomainState})
    distr = zeros(length(S))
    loc = (state.x, state.y)

    if sum(state.𝒫) == 0 || loc ∉ people_locations
        distr[index(state, S)] = 1.0
        return distr
    end

    𝒫′ = copy(state.𝒫)
    𝒫′[index(loc, people_locations)] = 0
    s′ = DomainState(state.x, state.y, state.θ, state.𝓁, 𝒫′)
    distr[index(s′, S)] = 1.0
    return distr
end

function generate_transitions(S::Vector{DomainState},
                              A::Vector{DomainAction},
                           grid,
                             s₀::DomainState)
    T = [[[0.0 for (i, _) in enumerate(S)]
               for (j, _) in enumerate(A)]
               for (k, _) in enumerate(S)]

    for (s, state) in enumerate(S)
        # First check if "outside"
        if state.x == 0
            state′ = DomainState(s₀.x, s₀.y, s₀.θ, s₀.𝓁, state.𝒫)
            s′ = index(state′, S)
            for a=1:length(A)
                T[s][a][s′] = 1.0
            end
            continue
        end

        for (a, action) in enumerate(A)
            if action.value == "aid"
                T[s][a] = aid_distribution(state, S)
            else
                T[s][a] = move_distribution(state, action, grid, S)
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
                          A::Vector{DomainAction})
    R = [[-1.0 for (i, _) in enumerate(A)]
               for (j, _) in enumerate(S)]

    for (s, state) in enumerate(S)
        R[s] *= sum(state.𝒫)
        if state.x == 0
            R[s] .-= 2.0 * ceil(sqrt(length(S)))
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
            if sum(T[i][j]) + 0.00001 ≥ 1.
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

function build_model(filepath::String)
    grid = generate_grid(filepath)
    S, s₀ = generate_states(grid)
    A = generate_actions()
    T = generate_transitions(S, A, grid, s₀)
    check_transition_validity(T, S, A)
    R = generate_rewards(S, A)
    ℳ = MDP(S, A, T, R, s₀)
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

# Not sure if there is a better way than manually setting this?
# people_locations = [(15,9), (4,7), (11,18)]
people_locations = [(2,2), (4,7), (3,8)]

function run_MDP()
    domain_map_file = joinpath(@__DIR__, "..", "maps", "collapse_1.txt")
    ℳ = build_model(domain_map_file)
    𝒱 = @time solve_model(ℳ)
    simulate(ℳ, 𝒱)
end

run_MDP()


function generate_map(h::Int, w::Int)
    seed = abs(rand(Int))
    MT = MersenneTwister(seed)
    save_path = joinpath(@__DIR__, "..", "maps", "collapse_$seed.txt")
    io = open(save_path, "w")
    for i = 1:h
        for j = 1:w
            if i == 1 || i == h
                write(io, 'X')
            elseif j == 1 || j == w
                write(io, 'X')
            else
                p = rand(MT)
                if p < 0.5
                    write(io, 'X')
                elseif p < 0.6
                    write(io, '0')
                elseif p < 0.8
                    write(io, '1')
                elseif p < 0.9
                    write(io, '2')
                else
                    write(io, '3')
                end
            end
            write(io, ' ')
        end
        write(io, '\n')
    end
    close(io)
end

# generate_map(20, 20)
