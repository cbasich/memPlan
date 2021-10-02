include(joinpath(@__DIR__, "models", "SOMDP.jl"))

function reachability(ℳ::SOMDP, δ::Int, 𝒮::LAOStarSolver)
    S, state₀, T = ℳ.S, ℳ.s₀, M.T
    s = index(state₀, S)
    π = 𝒮.π

    reachable = Set{MemoryState}()
    reachable_max_depth = Set{MemoryState}()
    visited = Vector{MemoryState}()
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
        a = π[s]
        for (s′, p) in T[s][a]
            push!(visited, s′)
        end
    end

    println("Reachable max depth states under optimal policy:
                               $(length(reachable_max_depth))")
end
