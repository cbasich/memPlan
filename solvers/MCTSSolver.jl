# Base Function Overwrites

function Base.findmax(f::Function, X)
    f_max = -Inf
    x_max = first(X)
    for x in X
        v = f(x)
        if v > f_max
            f_max, x_max = v, x
        end
    end
    return f_max, x_max
end

function Base.findmin(f::Function, X)
    f_min = Inf
    x_min = first(X)
    for x in X
        v = f(x)
        if v < f_min
            f_min, x_min = v, x
        end
    end
    return f_min, x_min
end

Base.argmax(f::Function, X) = findmax(f, X)[2]
Base.argmin(f::Function, X) = findmin(f, X)[2]



#############################

## Note: In this file, "s" and "a" refer to states and actions not indices.
#        specificially for the benefit of readibility.
##

mutable struct MCTSSolver
    ℳ # Problem Model
    N  # Node visit counter
    Q  # Action value estimates
    U  # Value function estimate
    d  # Rollout maximal depth
    m  # Max number of rollouts
    c # Exploration constant
end

function solve(π::MCTSSolver, s)
    num_visits = 0
    for a in π.ℳ.A
        if haskey(π.N, (s, a))
            num_visits += π.N[(s,a)]
        end
    end
    if num_visits >= π.m
        return argmax(a->π.Q[(s,a)], π.ℳ.A)
    end

    for k = 1:π.m
        rollout!(π, s)
    end
    return argmax(a->π.Q[(s,a)], π.ℳ.A)
end

function rollout!(π::MCTSSolver, s, d=π.d)
    if d ≤ 0
        return π.U(s)
    end

    ℳ, N, Q, c = π.ℳ, π.N, π.Q, π.c
    A, R = ℳ.A, ℳ.R
    if !haskey(N, (s, first(A)))
        for a in A
            N[(s,a)] = 0
            Q[(s,a)] = π.U(s, a)
        end
        return π.U(s)
    end

    a = select_action(π, s)
    s′ = generate_successor(ℳ, s, a)
    r = R(ℳ, s, a)
    q = r + rollout!(π, s′, d-1)

    N[(s,a)] += 1
    Q[(s,a)] += (q - Q[(s,a)])/N[(s,a)]
    return q
end

bonus(Nsa, Ns) = Nsa == 0 ? Inf : sqrt(log(Ns)/Nsa)

function select_action(π::MCTSSolver, s)
    A, N, Q, c = π.ℳ.A, π.N, π.Q, π.c
    Ns = sum(N[(s,a)] for a in A)
    return argmax(a->Q[(s,a)] + c * bonus(N[(s,a)], Ns), A)
end
