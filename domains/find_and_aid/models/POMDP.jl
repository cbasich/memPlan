using POMDPs, POMDPModelTools, QMDP, SARSOP, PointBasedValueIteration
using POMDPSimulators, DiscreteValueIteration
using Infiltrator, LinearAlgebra
include("SOMDP.jl")


#constants
δ = 1


# ================= DOMAIN CONFIGURATION BEGIN =================

domain_map_file = joinpath(@__DIR__, "..", "maps", "collapse_1.txt")
PEOPLE_LOCATIONS = [(2,2), (4,7), (3,8)] # COLLAPSE 1
# PEOPLE_LOCATIONS = [(7, 19), (10, 12), (6, 2)] # COLLAPSE 2

# ================= DOMAIN CONFIGURATION END =================


struct FindPOMDP <: POMDP{DomainState, DomainAction, DomainState}
    ℳ::SOMDP
    ∅::DomainState #null observation
    people_locations::Vector{Tuple{Int, Int}}
end

function FindPOMDP(ℳ::SOMDP, people_locations::Vector{Tuple{Int, Int}})
    return FindPOMDP(ℳ, DomainState(-1, -1, '∅', -1, Vector{Integer}()), people_locations)
end


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
        if state.x == 0
            return -1.0*sum(state.𝒫)
        end
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
POMDPs.discount(pomdp::FindPOMDP) = .9
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

m = FindPOMDP(ℳ, PEOPLE_LOCATIONS)



# ================= SOLVER CONFIGURATION BEGIN =================
SOLVER = "SARSOP"             # Change this to select solver.

@time begin
    if SOLVER == "QMDP"
        solver = QMDPSolver(SparseValueIterationSolver(max_iterations=1000,
                                                  belres=1e-3,verbose=true))
    elseif SOLVER == "SARSOP"
        solver = SARSOPSolver()
    elseif SOLVER == "PBVI"
        solver = PBVISolver(verbose=true)
    end

    policy = @time SARSOP.solve(solver, m)
end
# ================= SOLVER CONFIGURATION END =================

# ================= RUN POLICY ON POMDP ======================
rsum = 0.0
rewards = Vector{Float64}()

@time for i in 1:10
    global rsum = 0.0
    for (s,b,a,o,r) in stepthrough(m, policy, "s,b,a,o,r", max_steps=100)
        println("s: $s, a: $a, o: $o")
        r = POMDPs.reward(m, s, a)
        global rsum += r

        ### To inspect alpha vector information, uncomment below:
        # alphas = policy.alphas
        # belief = b.b
        #
        # alpha_values = [dot(α, belief) for α in alphas]
        # best_action = argmax(alpha_values)
        #
        # println("Returning best action: $(policy.action_map[best_action] == a)")
    end
    push!(rewards, rsum)
end

println("Average reward: $(mean(rewards)) ⨦ $(std(rewards))")
