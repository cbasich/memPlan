using POMDPs, POMDPModelTools, QMDP, SARSOP, PointBasedValueIteration, POMDPSimulators, DiscreteValueIteration

include("SOMDP.jl")


#constants
δ = 1


struct FindPOMDP <: POMDP{DomainState, DomainAction, DomainState}
    ℳ::SOMDP
    ∅::DomainState #null observation
end

function FindPOMDP(ℳ::SOMDP)
    return FindPOMDP(ℳ, DomainState(-1, -1, '∅', -1, Vector{Integer}()))
end


domain_map_file = joinpath(@__DIR__, "..", "maps", "collapse_2.txt")
#PEOPLE_LOCATIONS = [(2,2), (4,7), (3,8)] # COLLAPSE 1
PEOPLE_LOCATIONS = [(7, 19), (10, 12), (6, 2)] # COLLAPSE 2
M = build_model(domain_map_file, PEOPLE_LOCATIONS)
ℳ = build_model(M, δ)

∅ = DomainState(-1, -1, '∅', -1, Vector{Integer}())
Ω = copy(ℳ.M.S)
push!(Ω, ∅)
A = copy(ℳ.M.A)
push!(A, DomainAction("QUERY"))

function POMDPs.transition(m::FindPOMDP, state::DomainState, action::DomainAction)
    if action.value == "QUERY"
        return Deterministic(state)
    else
        s,a = m.ℳ.M.Sindex[state], m.ℳ.M.Aindex[action]
        return SparseCat(m.ℳ.M.S, m.ℳ.M.T[s][a])
    end
end

function POMDPs.reward(m::FindPOMDP, state::DomainState, action::DomainAction)
    if action.value == "QUERY"
        return -0.2 * sum(state.𝒫)
    else
        s,a = m.ℳ.M.Sindex[state], m.ℳ.M.Aindex[action]
        return m.ℳ.M.R[s][a]
    end
end

function POMDPs.observation(m::FindPOMDP, action::DomainAction, successor::DomainState)
    if action.value == "QUERY"
        return Deterministic(successor)
    else
        η = eta(successor)
        return SparseCat([successor, m.∅], [η, 1-η])
    end
end

POMDPs.states(pomdp::FindPOMDP) = ℳ.M.S
POMDPs.actions(pomdp::FindPOMDP) = A
POMDPs.observations(pomdp::FindPOMDP) = Ω 
POMDPs.isterminal(pomdp::FindPOMDP, s::DomainState) = terminal(s)
POMDPs.discount(pomdp::FindPOMDP) = 0.9
POMDPs.initialstate(pomdp::FindPOMDP) = Deterministic(pomdp.ℳ.M.s₀)
POMDPs.stateindex(pomdp::FindPOMDP, state::DomainState) = pomdp.ℳ.M.Sindex[state]
POMDPs.actionindex(pomdp::FindPOMDP, action::DomainAction) =
    begin
        if action.value != "QUERY"
            pomdp.ℳ.M.Aindex[action]
        else
            return length(pomdp.ℳ.M.A) + 1
        end
    end

function POMDPs.obsindex(pomdp::FindPOMDP, obs::DomainState)
    if obs != pomdp.∅
        return pomdp.ℳ.M.Sindex[obs]
    else
        return length(pomdp.ℳ.M.S) + 1
    end
end

m = FindPOMDP(ℳ)

@time begin
    #solver = QMDPSolver(SparseValueIterationSolver(max_iterations=1000, belres=1e-3,verbose=true))
    solver = SARSOPSolver()
    #solver = PBVISolver(verbose=true)
    policy = @time SARSOP.solve(solver, m)
end

rsum = 0.0
rewards = Vector{Float64}()


for i in 1:100
    println(i)
    global rsum = 0.0
    for (s,b,a,o,r) in stepthrough(m, policy, "s,b,a,o,r", max_steps=100)
#        println("s: $s, a: $a, o: $o")
        global rsum += r
    end
    push!(rewards, rsum)
end
println("Average reward: $(mean(rewards)) ⨦ $(std(rewards))")
