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

mutable struct MCTSSolver
    ℳ# Problem Model
    N # Node visit counter
    Q # Action value estimates
    U # Value function estimate
    d # Rollout maximal depth
    m # Max number of rollouts
    c # Exploration constant
end

function solve(π::MCTSSolver, s)
    for k = 1:π.m
        rollout!(π, s)
    end
    println(minimum(a->π.Q[(s,a)], π.ℳ.A))
    return argmin(a->π.Q[(s,a)], π.ℳ.A)
end

function rollout!(π::MCTSSolver, s, d=π.d)
    if d ≤ 0
        return π.U(s)
    end

    ℳ, N, Q, c = π.ℳ, π.N, π.Q, π.c
    A = ℳ.A
    if !haskey(N, (s, first(A)))
        for a in A
            N[(s,a)] = 0
            Q[(s,a)] = π.U(s, a)
        end
        # println(π.U(s))
        return π.U(s)
    end

    a = select_action(π, s)
    # println(s, a)
    s′ = generate_successor(ℳ, s, a)
    r = ℳ.C(ℳ, s, a)                # This is actually a positive cost.
    q = r + rollout!(π, s′, d-1)

    N[(s,a)] += 1
    Q[(s,a)] += (q - Q[(s,a)])/N[(s,a)]
    # println(Q[(s,a)])
    # print(q)
    return q
end

bonus(Nsa, Ns) = Nsa == 0 ? Inf : sqrt(log(Ns)/Nsa)

function select_action(π::MCTSSolver, s)
    A, N, Q, c = π.ℳ.A, π.N, π.Q, π.c
    Ns = sum(N[(s,a)] for a in A)
    return argmin(a->Q[(s,a)] + c * bonus(N[(s,a)], Ns), A)
end
