function generate_map(h::Int, w::Int)
    seed = abs(rand(Int))
    MT = MersenneTwister(seed)
    save_path = joinpath(@__DIR__, "maps", "collapse_$seed.txt")
    io = open(save_path, "w")
    for i = 1:h
        for j = 1:w
            if i == 1 || i == h
                write(io, 'X')
            elseif j == 1 || j == w
                write(io, 'X')
            else
                p = rand(MT)
                if p < 0.3
                    write(io, 'X')
                elseif p < 0.6
                    write(io, '0')
                elseif p < 0.8
                    write(io, '1')
                elseif p < 0.9
                    write(io, '2')
                else
                    write(io, '3')
                end
            end
            write(io, ' ')
        end
        write(io, '\n')
    end
    close(io)
end

generate_map(10,10)
