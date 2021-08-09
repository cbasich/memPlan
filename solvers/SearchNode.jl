mutable struct SearchNode
    children::Vector{SearchNode}
    immediateReward::Float64
    prob::Float64
    stepsToGo::Integer
    futureReward::Float64
    numVisits::Integer
    init::Bool
    solve::Bool
end
function SearchNode(prob::Float64, stepsToGo::Integer)
    return SearchNode(Vector{SearchNode}(),
                      0.0, prob, stepsToGo,
                      -Inf, 0, false, false)
end

function reset!(node::SearchNode, prob::Float64, stepsToGo::Integer)
    node = SearchNode(prob, stepsToGo)
end

function get_expected_reward_estimate(node::SearchNode)
    return node.immediateReward + node.futureReward
end
