
## STANDARD LIBRARIES



## LOCAL DIRECTORIES
include(joinpath(@__DIR__, "models", "SOMDP.jl"))


## PARAMS
MAP_PATH = joinpath(@__DIR__, "maps", "collapse_1.txt")
SOLVER = "laostar"
SIM = false
SIM_COUNT = 1
VERBOSE = false
DEPTH = 1


## MAIN SCRIPT
println("Building MDP...")
M = build_model(MAP_PATH)
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
