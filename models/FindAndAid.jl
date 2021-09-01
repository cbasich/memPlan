using Combinatorics
using Statistics

import Base.==

include(joinpath(@__DIR__, "..", "solvers", "ValueIterationSolver.jl"))

function index(element, collection)
    for i=1:length(collection)
        if collection[i] == element
            return i
        end
    end
    return -1
end

function ==(a::DomainState, b::DomainState)
    return a.x == b.x && a.y == b.y && a.θ == b.θ && a.ℒ == b.ℒ && a.𝒫 == b.𝒫
end

struct DomainState
    x::Integer
    y::Integer
    θ::Char
    ℒ::Integer
    𝒫::Vector{Integer}
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

# Not sure if there is a better way than manually setting this?
people_locations = [(2,2), (4,5)]

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
            end
            for θ in ['↑', '↓', '→', '←']
                for mask in collect(combinations(1:num_people))
                    P = copy(𝒫)
                    P[mask] .= 0
                    if loc == 'S'
                        loc = '0'
                    end
                    push!(S, DomainState(i, j, θ, parse(Int, loc), P))
                end
            end
        end
    end
    return S, s₀
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
                           S::Vector{DomainState})
    xp, yp = pos_shift(action.value)
    xpr, ypr = pos_shift(slip_right(action.value))
    xpl, ypl = pos_shift(slip_left(action.value))
    xp, xpr, xpl = xp + state.x, xpr + state.x, xpl + state.x
    yp, ypr, ypl = yp + state.y, ypr + state.y, ypl + state.y

    distr = zeros(length(S))
    for (s′, state′) in enumerate(S)
        if state′.θ ≠ action.value || state′.𝒫 ≠ state.𝒫
            continue
        end

        if state == state′
            distr[s′] = 0.1
        elseif state′.x == xpr && state′.y == ypr
            distr[s′] = 0.05
        elseif state′.x == xpl && state′.y == ypl
            distr[s′] = 0.05
        elseif state′.x == xp && state′.y == yp
            distr[s′] = 0.8
        end
    end

    distr[index(state, S)] = 1.0 - sum(distr)

    return distr
end

function aid_distribution(state::DomainState,
                          S::Vector{DomainState})
    distr = zeros(length(S))
    if sum(state.𝒫) == 0
        distr[index(state, S)] = 1.0
        return distr
    end

    loc = (state.x, state.y)
    𝒫′ = copy(state.𝒫)
    for i = 1:length(people_locations)
        if people_locations[i] == loc
            𝒫′[i] = 0
            break
        end
        distr[index(state, S)] = 1.0
        return distr
    end

    s′ = DomainState(state.x, state.y, state.θ, state.ℒ, 𝒫′)
    distr[index(s′, S)] = 1.0
    return distr
end

function generate_transitions(S::Vector{DomainState},
                              A::Vector{DomainAction})
    T = [[[0.0 for (i, _) in enumerate(S)]
               for (j, _) in enumerate(A)]
               for (k, _) in enumerate(S)]

    for (s, state) in enumerate(S)
        # if goal_condition(state)
        #     T[s, :, s] .= 1.0
        #     continue
        # end
        for (a, action) in enumerate(A)
            if action.value == "aid"
                T[s][a] = aid_distribution(state, S)
            else
                T[s][a] = move_distribution(state, action, S)
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
                    println("Succ State: $(S[k])     with probability: $(T[i][j][k])")
                end
                return 0.
            end
            if round(sum(T[i][j])) == 1.
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
    # println(grid)
    # println(grid[2][4])
    S, s₀ = generate_states(grid)
    A = generate_actions()
    T = generate_transitions(S, A)
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

function main()
    domain_map_file = joinpath(@__DIR__, "..", "maps", "collapse_1.txt")
    ℳ = build_model(domain_map_file)
    𝒱 = solve_model(ℳ)
end

main()
