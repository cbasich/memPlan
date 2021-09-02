include(joinpath(@__DIR__, "..", "solvers", "ValueIterationSolver.jl"))

struct CampusState
    x::Integer
    y::Integer
    θ::Char
    o::Char
end

struct CampusAction
    value::Union{String,Char}
end

struct CampusSSP
    S::Vector{CampusState}
    A::Vector{CampusAction}
    T
    C
    s₀
    G
end

function generate_grid(filename::String)
    grid = Vector{Vector{Char}}()
    open(filename) do file
        for l in eachline(file)
            row = collect(l)
            deleteat!(row, row .== ' ')
            push!(grid, row)
        end
    end
    return grid
end

function generate_states(grid::Vector{Vector{Char}},
                         init::Char,
                         goals::Vector{Char})
    S = Vector{CampusState}()
    G = Vector{CampusState}()
    s₀ = CampusState(-1, -1, '↑', '∅')
    for (i, row) in enumerate(grid)
        for (j, loc) in enumerate(row)
            for θ in ['↑', '↓', '→', '←']
                if loc == 'X'
                    continue
                elseif loc == '.'
                    state = CampusState(i, j, θ, '∅')
                    push!(S, state)
                elseif loc == 'C'
                    for o in ['E', 'L', 'B']
                        state = CampusState(i, j, θ, o)
                        push!(S, state)
                    end
                elseif loc == 'D'
                    for o in ['C', 'O']
                        state = CampusState(i, j, θ, o)
                        push!(S, state)
                    end
                else
                    state = CampusState(i, j, θ, '∅')
                    push!(S, state)
                    if loc in goals
                        push!(G, state)
                    elseif loc == init
                        s₀ = state
                    end
                end
            end
        end
    end
    dead_end = CampusState(-1, -1, '∅', '∅')
    push!(S, dead_end)
    return S, s₀, G
end

function generate_actions()
    A = Vector{CampusAction}()
    for v in ['↑', '→', '↓', '←', "cross", "open", "wait"]
        a = CampusAction(v)
        push!(A, a)
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

function move_probability(s::CampusState,
                          a::CampusAction,
                         s′::CampusState)
    xp, yp = pos_shift(a.value)
    xpr, ypr = pos_shift(slip_right(a.value))
    xpl, ypl = pos_shift(slip_left(a.value))
    xp, xpr, xpl = xp + s.x, xpr + s.x, xpl + s.x
    yp, ypr, ypl = yp + s.y, ypr + s.y, ypl + s.y

    if s′.θ ≠ a.value
        return 0.
    end

    p = 0.
    if s.o == 'C' || s.o == 'E' || s.o == 'L' || s.o == 'B'
        if a.value == s.θ
            p = (s′ == s ? 1. : 0.)
        else
            if s′.x == xp && s′.y == yp
                p = .9
            elseif s′.x == s.x && s′.y == s.y
                p = .1
            end
        end
    elseif s.o == 'O'
        if s′.x == xp && s′.y == yp
            p = .9
        elseif s′.x == s.x && s′.y == s.y
            p = .1
        end
        p = ((xp == s′.x && yp == s′.y) ? 1. : 0.)
    else
        if s′.o == 'O'
            p = 0.0
        elseif s′.o == 'C'
            p = ((xp == s′.x && yp == s′.y) ? 1. : 0.)
        elseif s′.o == '∅'
            if s′.x == xp && s′.y == yp
                p = .8
            # elseif s′.x == xpr && s′.y == ypr
            #     p = .05
            # elseif s′.x == xpl && s′.y == ypl
            #     p = .05
            elseif s′.x == s.x && s′.y == s.y
                p = .2
            end
        else
            p = ((xp == s′.x && yp == s′.y) ? 1/3 : 0.)
        end
    end
    return p
end

function wait_probability(s::CampusState,
                         s′::CampusState)
    if s′.x ≠ s.x || s′.y ≠ s.y || s′.θ ≠ s.θ
        return 0.
    end
    p = 0.
    if s.o == 'E'
        if s′.o == 'E'
            p = .75
        elseif s′.o == 'L'
            p = .25
        end
    elseif s.o == 'L'
        if s′.o == 'E' || s′.o == 'B'
            p = .25
        elseif s′.o == 'L'
            p = .5
        end
    elseif s.o == 'B'
        if s′.o == 'L'
            p = .25
        elseif s′.o == 'B'
            p = .75
        end
    end
    return p
end

function cross_probability(S::Vector{CampusState},
                           s::CampusState,
                          s′::CampusState)
    if s.o ∉ ['E', 'L', 'B']
        return (s == s′ ? 1. : 0.)
    end

    xp, yp = pos_shift(s.θ)
    xp += s.x
    yp += s.y

    cross_state = CampusState(xp, yp, s.θ, '∅')
    if cross_state ∉ S
        return (s == s′ ? 1. : 0.)
    end

    p = 0.
    if s.o == 'E'
        if s′.x == xp && s′.y == yp && s′.θ == s.θ && s′.o == '∅'
            p = 1.
        end
    elseif s.o == 'L'
        if s′.x == xp && s′.y == yp && s′.θ == s.θ && s′.o == '∅'
            p = .5
        else
            p = .5wait_probability(s, s′)
        end
    elseif s.o == 'B'
        if s′.x == xp && s′.y == yp && s′.θ == s.θ && s′.o == '∅'
            p = .1
        elseif s′.x == -1
            p = .1
        else
            p = .8wait_probability(s, s′)
        end
    else
        p = 0.0
    end
    return p
end

function open_probability(s::CampusState,
                         s′::CampusState)
    p = 0.
    if s.o ≠ 'C'
        p = (s′ == s ? 1. : 0.)
    end
    if s.x == s′.x && s.y == s′.y && s.θ == s′.θ && s′.o == 'O'
        p = 1.
    end
    return p
end

function generate_transition_matrix(S::Vector{CampusState},
                                    A::Vector{CampusAction},
                                    G::Vector{CampusState})
    T = [[[0.0 for (i, _) in enumerate(S)]
               for (j, _) in enumerate(A)]
               for (k, _) in enumerate(S)]
    for (s, state) in enumerate(S)
        for (a, action) in enumerate(A)
            if state in G
                T[s][a][s] = 1.0
                continue
            elseif state.x == -1
                T[s][a][index(G[1], S)] = 1.0
            end
            for (s′, state′) in enumerate(S)
                if action.value == "open"
                    T[s][a][s′] = open_probability(state, state′)
                elseif action.value == "wait"
                    T[s][a][s′] = wait_probability(state, state′)
                elseif action.value == "cross"
                    T[s][a][s′] = cross_probability(S, state, state′)
                else
                    T[s][a][s′] = move_probability(state, action, state′)
                end
            end
            if sum(T[s][a]) ≠ 1.
                T[s][a][s] += (1. - sum(T[s][a]))
            end
        end
    end
    return T
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

function generate_cost_matrix(S::Vector{CampusState},
                              A::Vector{CampusAction},
                              G::Vector{CampusState},
                           grid::Vector{Vector{Char}})
    C = [[1.0 for (i, _) in enumerate(A)]
              for (j, _) in enumerate(S)]

    for (s, state) in enumerate(S)
        if state in G
            C[s] *= 0.0
            continue
        elseif state.x == -1
            C[s] *= 100.0
        end
        for (a, action) in enumerate(A)
            if state.o == 'C' && state.θ == action.value
                C[s][a] = 5.0
            end
            if typeof(action.value) == Char
                xp, yp = pos_shift(action.value)
                x, y = state.x + xp, state.y + yp
                if x < 1 || x > length(grid) || y < 1 || y > length(grid[1]) || grid[x][y] == 'X'
                    C[s][a] = 5.0
                end
            end
        end
    end
    return C
end

function build_model(filepath::String)
    grid = generate_grid(filepath)
    S, s₀, G = generate_states(grid, 's', ['g'])
    A = generate_actions()
    C = generate_cost_matrix(S, A, G, grid)
    T = generate_transition_matrix(S, A, G)
    check_transition_validity(T, S, A)
    ℳ = CampusSSP(S, A, T, C, s₀, G)
    return ℳ
end

function solve_model(ℳ::CampusSSP)
    𝒱 = ValueIterationSolver(.001,
                             true,
          Dict{Integer,Integer}(),
                  zeros(length(ℳ.S)),
        Vector{Float64}(undef, length(ℳ.A)))
    solve(𝒱, ℳ)
    return 𝒱
end

function generate_successor(ℳ::CampusSSP, s::Int, a::Int)
    thresh = rand()
    p = 0.
    for (s′, state′) ∈ enumerate(ℳ.S)
        p += ℳ.T[s][a][s′]
        if p >= thresh
            return state′
        end
    end
end

function index(element, collection)
    for i=1:length(collection)
        if collection[i] == element
            return i
        end
    end
    return -1
end

function simulate(ℳ::CampusSSP, 𝒱::ValueIterationSolver)
    S, A, C, G = ℳ.S, ℳ.A, ℳ.C, ℳ.G
    cost = 0.
    # println(𝒱.V[index(ℳ.s₀, S)])
    for i=1:100
        state = ℳ.s₀
        # println("Expected cost to goal: $(𝒱.V[index(state, S)])")
        while state ∉ G
            s = index(state, S)
            a = 𝒱.π[s]
            cost += C[s][a]
            # println("Taking action $(A[a]) in state $state.")
            state = generate_successor(ℳ, s, a)
        end
        # println("Reached the goal with total cost $cost.")
    end
    println("Average cost to goal: $(cost / 100.0)")
end

# function run_CampusSSP()
#     ℳ = build_model("tiny.txt")
#     𝒱 = @time solve_model(ℳ)
#     simulate(ℳ, 𝒱)
# end

# run_CampusSSP()
