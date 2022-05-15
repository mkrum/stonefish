using Chess
using Chess.PGN
using DataFrames
using JLD2

function main()
    file = open("lichess_db_standard_rated_2016-09.pgn", "r")
    
    events::Vector{String} = []
    white_elos::Vector{Int} = []
    black_elos::Vector{Int}  = []
    results::Vector{String} = []
    moves::Vector{String} = []
    
    i = 0
    while true
        reader = PGNReader(file)
        game = readgame(reader; annotations=true)
        readline(file)

        if headervalue(game, "Termination") != "Normal"
            continue
        end

        event = headervalue(game, "Event")

        if event == "?"
            break
        end

        if (length(event) < 15) | (event[1:15] != "Rated Classical")
            continue
        end

        try
            white_elo = parse(Int, headervalue(game, "WhiteElo"))
            black_elo = parse(Int, headervalue(game, "BlackElo"))
            result = headervalue(game, "Result")
            move_str = string(game)[10:end]

            push!(events, event)
            push!(white_elos, white_elo)
            push!(black_elos, black_elo)
            push!(results, result)
            push!(moves, move_str)
            i += 1
        catch
            println("Missing data")
        end

        if i % 1000 == 0
            println(i)

            df = DataFrame(event=events, move=moves, result=results, white_elo=white_elos, black_elo=black_elos)
    
            jldopen("df.jld2", "w") do file
                file["df"] = df
            end
        end
    end

    df = DataFrame(event=events, move=moves, result=results, white_elo=white_elos, black_elo=black_elos)
    
    df
end
