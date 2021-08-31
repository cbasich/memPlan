using Combinatorics
using Statistics

include(joinpath(@__DIR__, "..", "solvers", "ValueIterationSolver.jl"))

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
people_locations = [(1,1), (2,2), (3,3)]

function generate_people_smoke_level_vector(grid::Vector{Vector{Any}})
    𝒫 = Vector{Int}()
    for loc in people_locations
        push!(grid[loc[1]][loc[2]], 𝒫)
    end
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
    S = Vector{CampusState}()
    𝒫 = generate_people_smoke_level_vector(grid)
    num_people = length(people_smoke_level_vector)

    for (i, row) in enumerate(grid)
        for (j, loc) in enumerate(row)
            if loc == 'X'
                continue
            end
            if loc == 'S'
                s₀ = CampusState(i, j, '→', 0, 𝒫)
                push!(S, s₀)
            end
            for θ in ['↑', '↓', '→', '←']
                for mask in collect(combinations(1:num_people))
                    P = copy(𝒫)
                    B[mask] .= 0
                    push!(S, CampusState(i, j, θ, loc, P))
                end
            end
        end
    end
    return S
end

function generate_actions()
    A = Vector{CampusAction}()
    for v in ['↑', '↓', '→', '←', "aid"]
        push!(A, CampusAction(v))
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

function move_distribution(s::DomainState,
                           a::DomainAction,
                           S::Vector{DomainState})
    xp, yp = pos_shift(a.value)
    xpr, ypr = pos_shift(slip_right(a.value))
    xpl, ypl = pos_shift(slip_left(a.value))
    xp, xpr, xpl = xp + s.x, xpr + s.x, xpl + s.x
    yp, ypr, ypl = yp + s.y, ypr + s.y, ypl + s.y

    distr = zeros(length(S))
    for (s′, state′) in enumerate(S)
        if s′.θ ≠ a.value
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

    return distr
end

function aid_distribution(s::DomainState,
                          S::Vector{DomainState})
    distr = zeros(length(S))

    loc = (state.x, state.y)
    for i = 1:length(people_locations)
        if people_locations[i] == loc
            𝒫′ = copy(s.𝒫)
            𝒫′[i] = 0
            break
        end
        distr[index(s, S)] = 1.0
        return distr
    end

    s′ = (s.x, s.y, s.θ, s.ℒ, 𝒫′)
    distr[index(s′, S)] = 1.0
    return distr
end

function generate_transitions(S::Vector{DomainState},
                              A::Vector{DomainAction})
    T = [[[0.0 for (i, _) in enumerate(S)]
               for (j, _) in enumerate(A)]
               for (k, _) in enumerate(S)]

    for (s, state) in enumerate(S)
        if goal_condition(state)
            T[s, :, s] .= 1.0
            continue
        end
        for (a, action) in enumerate(A)
            if goal_condition(state)
                T[s][a][s] = 1.0
                continue
            elseif action == "aid"
                T[s][a] = aid_distribution(state, S)
            else
                T[s][a] = move_distribution(state, action, S)
            end
            if sum(T[s][a]) ≠ 1.0
                T[s][a][s] += (1.0 - sum(T[s][a]))
            end
        end
    end
    return T
end

function generate_rewards(S::Vector{DomainState},
                          A::Vector{DomainAction})
    R = [[-1.0 for (i, _) in enumerate(A)]
               for (j, _) in enumerate(S)]

    for (s, state) in enumerate(S)
        if goal_condition(state)
            R[s] *= 0.0
            continue
        end
        R[s] *= sum(state.𝒫)
    end
end

function check_transition_validity(T, S, A)
    n, m = length(S), length(A)
    for i in 1:n
        for j in 1:m
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
    domain_map_file = ""
    ℳ = build_model(domain_map_file)
end
