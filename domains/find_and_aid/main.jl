
## STANDARD LIBRARIES



## LOCAL DIRECTORIES
include(joinpath(@__DIR, "models", "SOMDP.jl"))


## PARAMS
MAP_PATH = joinpath(@__DIR__, "maps", "collapse_3751139720432660911.txt")
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
solver = @time solve(ℳ, SOLVER)

if SIM
    println("Simulating...")
    simulate(ℳ, 𝒱, solver, SIM_COUNT, VERBOSE)
end
