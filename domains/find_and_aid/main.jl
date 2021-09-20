module Runner

export main
## STANDARD LIBRARIES

## LOCAL DIRECTORIES
include(joinpath(@__DIR__, "models", "SOMDP.jl"))

function main()
    ## PARAMS
    SOLVER = "laostar"
    SIM = true 
    SIM_COUNT = 100
    VERBOSE = false
    DEPTH = 2

#    MAP_PATH = joinpath(@__DIR__, "maps", "collapse_1.txt")
    MAP_PATH = joinpath(@__DIR__, "maps", "collapse_2.txt")
#    PEOPLE_LOCATIONS = [(2,2), (4,7), (3,8)] # COLLAPSE 1
    PEOPLE_LOCATIONS = [(7, 19), (10, 12), (6, 2)] # COLLAPSE 2


    ## MAIN SCRIPT
    @time begin
        println("Building MDP...")
        M = build_model(MAP_PATH, PEOPLE_LOCATIONS)
        println("Solving MDP...")
        @time 𝒱 = solve_model(M)
        @time println("Building SOMDP...")
        ℳ = build_model(M, DEPTH)
        println("Solving SOMDP...")
        solver = @time solve(ℳ, 𝒱, SOLVER)
    end

    if SIM
        println("Simulating...")
        simulate(ℳ, 𝒱, solver, SIM_COUNT, VERBOSE)
    end

end

end

Runner.main()
