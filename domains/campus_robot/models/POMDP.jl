using POMDPs, POMDPModelTools, QMDP, SARSOP, PointBasedValueIteration, POMDPSimulators, DiscreteValueIteration

include("SOMDP.jl")


#constants
δ = 1


# ================= DOMAIN CONFIGURATION BEGIN ================= 

MAP_PATH = joinpath(@__DIR__, "..", "maps", "two_buildings.txt")

# ================= DOMAIN CONFIGURATION END ================= 


struct CampusPOMDP <: POMDP{DomainState, DomainAction, DomainState}
    ℳ::SOMDP
    ∅::DomainState #null observation
end

function CampusPOMDP(ℳ::SOMDP)
    return CampusPOMDP(ℳ, DomainState(-10, -10, '⊥', '⊥'))
end


M = build_model(MAP_PATH, 's', 'g')
ℳ = build_model(M, δ)

∅ = DomainState(-10, -10, '⊥', '⊥')
Ω = copy(ℳ.M.S)
push!(Ω, ∅)
A = copy(ℳ.M.A)
push!(A, DomainAction("QUERY"))

function POMDPs.transition(m::CampusPOMDP, state::DomainState, action::DomainAction)
    if action.value == "QUERY"
        return Deterministic(state)
    else
        s,a = m.ℳ.M.Sindex[state], m.ℳ.M.Aindex[action]
        return SparseCat(m.ℳ.M.S, m.ℳ.M.T[s][a])
    end
end

function POMDPs.reward(m::CampusPOMDP, state::DomainState, action::DomainAction)
    if state.x == -1
        return -10
    elseif action.value == "QUERY"
        return -3.0
    else
        s,a = m.ℳ.M.Sindex[state], m.ℳ.M.Aindex[action]
        return m.ℳ.M.R[s][a]
    end
end

function POMDPs.observation(m::CampusPOMDP, action::DomainAction, successor::DomainState)
    if action.value == "QUERY"
        return Deterministic(successor)
    else
        η = eta(successor)
        return SparseCat([successor, m.∅], [η, 1-η])
    end
end

POMDPs.states(pomdp::CampusPOMDP) = ℳ.M.S
POMDPs.actions(pomdp::CampusPOMDP) = A
POMDPs.observations(pomdp::CampusPOMDP) = Ω 
POMDPs.isterminal(pomdp::CampusPOMDP, s::DomainState) = s == pomdp.ℳ.M.g
POMDPs.discount(pomdp::CampusPOMDP) = 1
POMDPs.initialstate(pomdp::CampusPOMDP) = Deterministic(pomdp.ℳ.M.s₀)
POMDPs.stateindex(pomdp::CampusPOMDP, state::DomainState) = pomdp.ℳ.M.Sindex[state]
POMDPs.actionindex(pomdp::CampusPOMDP, action::DomainAction) =
    begin
        if action.value != "QUERY"
            pomdp.ℳ.M.Aindex[action]
        else
            return length(pomdp.ℳ.M.A) + 1
        end
    end

function POMDPs.obsindex(pomdp::CampusPOMDP, obs::DomainState)
    if obs != pomdp.∅
        return pomdp.ℳ.M.Sindex[obs]
    else
        return length(pomdp.ℳ.M.S) + 1
    end
end

m = CampusPOMDP(ℳ)



# ================= SOLVER CONFIGURATION BEGIN ================= 

@time begin
    #solver = QMDPSolver(SparseValueIterationSolver(max_iterations=1000, belres=1e-3,verbose=false))
    solver = SARSOPSolver()
    #solver = PBVISolver(verbose=true)
    policy = @time SARSOP.solve(solver, m)
end


# ================= SOLVER CONFIGURATION END ================= 

rsum = 0.0
rewards = Vector{Float64}()

for i in 1:100
    println(i)
    global rsum = 0.0
    for (s,b,a,o,r) in stepthrough(m, policy, "s,b,a,o,r", max_steps=100)
#        println("s: $s, a: $a, o: $o, r: $r")
        global rsum += r
    end
    push!(rewards, rsum)
end
println("Average reward: $(mean(rewards)) ⨦ $(std(rewards))")
