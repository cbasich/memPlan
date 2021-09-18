using POMDPs, POMDPModelTools, QMDP, SARSOP, PointBasedValueIteration, POMDPSimulators

include("SOMDP.jl")


#constants
δ = 1


struct FindPOMDP <: POMDP{DomainState, DomainAction, DomainState}
    ℳ::SOMDP
    ∅::DomainState #null observation
    η::Float64
end

function FindPOMDP(ℳ::SOMDP)
    return FindPOMDP(ℳ, DomainState(-1, -1, '∅', -1, Vector{Integer}()), 0.5)
end

function generate_pomdp(ℳ::SOMDP)

    ∅ = DomainState(-1, -1, '∅', -1, Vector{Integer}())
    Ω = copy(ℳ.M.S)
    push!(Ω, ∅)
    A = copy(M.A)
    push!(A, "QUERY")

    function POMDPs.transition(m::FindPOMDP, state::DomainState, action::DomainAction)
        s,a = m.ℳ.M.Sindex[state], m.ℳ.M.Aindex[action]
        return SparseCat(m.ℳ.M.S, m.ℳ.M.T[s][a])
    end

    function POMDPs.reward(m::FindPOMDP, state::DomainState, action::DomainAction)
        if action.value == "QUERY"
            return -5
        else
            s,a = m.ℳ.M.Sindex[state], m.M.Aindex[action]
            return m.ℳ.M.R[s][a]
        end
    end

    function POMDPs.observation(m::FindPOMDP, state::DomainState, action::DomainAction, successor::DomainState)
        s,a,s′= m.M.Sindex[state], m.ℳ.M.Aindex[action], m.ℳ.M.Sindex[successor]
        return SparseCat([successor, m.∅], [m.η, 1-m.η])
    end

    function POMDPs.observation(m::FindPOMDP, action::DomainAction, successor::DomainState)
        a,s′= m.ℳ.M.Aindex[action], m.ℳ.M.Sindex[successor]
        return SparseCat([successor, m.∅], [m.η, 1-m.η])
    end

    POMDPs.states(pomdp::FindPOMDP) = ℳ.M.S
    POMDPs.actions(pomdp::FindPOMDP) = A
    POMDPs.observations(pomdp::FindPOMDP) = Ω 
    POMDPs.isterminal(pomdp::FindPOMDP, s::DomainState) = terminal(s)
    POMDPs.discount(pomdp::FindPOMDP) = 0.99
    POMDPs.initialstate(pomdp::FindPOMDP) = Deterministic(M.s₀)
    POMDPs.stateindex(pomdp::FindPOMDP, state::DomainState) = pomdp.M.Sindex[state]
    POMDPs.actionindex(pomdp::FindPOMDP, action::DomainAction) = pomdp.M.Aindex[action]

    function POMDPs.obsindex(pomdp::FindPOMDP, obs::DomainState)
        if obs != pomdp.∅
            return pomdp.M.Sindex[obs]
        else
            return length(pomdp.M.S) + 1
        end
    end

    m = FindPOMDP(ℳ)
    return m
end

function build_and_solve()
    domain_map_file = joinpath(@__DIR__, "..", "maps", "collapse_1.txt")
    M = build_model(domain_map_file)
    ℳ = build_model(M, δ)

    m = generate_pomdp(ℳ)

    #solver = QMDPSolver() #x_iterations=20, belres=1e-3, verbose=true)
    solver = SARSOPSolver()
    #solver = PBVISolver(verbose=true)
    policy = @time PointBasedValueIteration.solve(solver, m)

    rsum = 0.0
    for (s,b,a,o,r) in stepthrough(m, policy, "s,b,a,o,r", max_steps=100)
    #    println("s: $s, b: $([pdf(b,s) for s in states(m)]), a: $a, o: $o")
        global rsum += r
    end
    println("Undiscounted reward was $rsum.")
end
