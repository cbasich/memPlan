using POMDPs, POMDPModelTools, QMDP, SARSOP, PointBasedValueIteration, POMDPSimulators

include("MDP.jl")

struct FindPOMDP <: POMDP{DomainState, DomainAction, DomainState}
    M::MDP
    η::Float64
    ∅::DomainState
end

function FindPOMDP(M::MDP, η)
    return FindPOMDP(M, η, DomainState(-1, -1, '∅', -1, Vector{Integer}()))
end

#function POMDPs.gen(m::FindPOMDP, s, a, rng)
#    s′ = generate_successor(m.M, s, a)
#    thresh = rand()
#    if rand > m.η
#        o = s′
#    else
#        o = ∅
#    end
#    r = m.M.R[index(s, m.M.S), index(a, m.M.A)]
#    return (sp = s′, o=o, r=r)
#end

function POMDPs.transition(m::FindPOMDP, state, action)
    s,a = m.M.Sindex[state], m.M.Aindex[action]
    return SparseCat(m.M.S, M.T[s][a])
end

function POMDPs.reward(m::FindPOMDP, state, action)
    s,a = m.M.Sindex[state], m.M.Aindex[action]
    return m.M.R[s][a]
end

function POMDPs.observation(m::FindPOMDP, state::DomainState, action::DomainAction, successor::DomainState)
    s,a,s′= m.M.Sindex[state], m.M.Aindex[action], m.M.Sindex[successor]
    return SparseCat([successor, m.∅], [m.η, 1-m.η])
end

function POMDPs.observation(m::FindPOMDP, action::DomainAction, successor::DomainState)
    a,s′= m.M.Aindex[action], m.M.Sindex[successor]
    return SparseCat([successor, m.∅], [m.η, 1-m.η])
end

#solver = QMDPSolver() #x_iterations=20, belres=1e-3, verbose=true)
#solver = SARSOPSolver()
solver = PBVISolver(verbose=true)

domain_map_file = joinpath(@__DIR__, "..", "maps", "collapse_1.txt")
M = build_model(domain_map_file)

∅ = DomainState(-1, -1, '∅', -1, Vector{Integer}())
Ω = copy(M.S)
push!(Ω, ∅)
POMDPs.states(pomdp::FindPOMDP) = M.S
POMDPs.actions(pomdp::FindPOMDP) = M.A
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

m = FindPOMDP(M, 0.5)
policy = @time PointBasedValueIteration.solve(solver, m)

rsum = 0.0
for (s,b,a,o,r) in stepthrough(m, policy, "s,b,a,o,r", max_steps=100)
#    println("s: $s, b: $([pdf(b,s) for s in states(m)]), a: $a, o: $o")
    global rsum += r
end
println("Undiscounted reward was $rsum.")
