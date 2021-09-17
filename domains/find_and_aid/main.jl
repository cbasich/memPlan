
## STANDARD LIBRARIES



## LOCAL DIRECTORIES
include(joinpath(@__DIR__, "models", "SOMDP.jl"))


## PARAMS
SOLVER = "laostar"
SIM = false
SIM_COUNT = 1
VERBOSE = false
DEPTH = 1

MAP_PATH = joinpath(@__DIR__, "maps", "collapse_1.txt")
PEOPLE_LOCATIONS = [(2,2), (4,7), (3,8)] # COLLAPSE 1
# PEOPLE_LOCATIONS = [(7, 19), (10, 12), (6, 2)] # COLLAPSE 2


## MAIN SCRIPT
println("Building MDP...")
M = build_model(MAP_PATH, PEOPLE_LOCATIONS)
println("Solving MDP...")
𝒱 = solve_model(M)
println("Building SOMDP...")
ℳ = build_model(M, DEPTH)
println("Solving SOMDP...")
solver = @time solve(ℳ, 𝒱, SOLVER)

if SIM
    println("Simulating...")
    simulate(ℳ, 𝒱, solver, SIM_COUNT, VERBOSE)
end
