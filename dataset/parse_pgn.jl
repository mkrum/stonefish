

using Redis
using ArgParse
using Distributed

using Chess
using Chess.PGN
using DataFrames
using JLD2

"""
    pop_chunk_from_list(conn::RedisConnection, list::Symbol,
    chunk_size::Int)::Vector{String}

Pops a chunk of data of size `chunk_size` from a list named `list` stored on
our redis server we are connected to with `conn`.
"""
function pop_chunk_from_list(
    conn::RedisConnection,
    list::Symbol,
    chunk_size::Int,
)::Vector{String}

    pipeline = open_pipeline(conn)
    # Gets values a chunk_sized subset of our list
    lrange(pipeline, list, 0, (chunk_size - 1))
    # Removes that same portion
    ltrim(pipeline, list, chunk_size, -1)

    data = read_pipeline(pipeline)[1]
    return data
end

function load_games(path::String)::Vector{String}

    conn = RedisConnection()

    file = open(path, "r")
    games::Vector{String} = []
    buffer::String = ""

    line = readline(file; keep=true)
    while line != nothing
        if line[1] == '['
            buffer = buffer * line
        end

        if line[1] == '1'
            buffer = buffer * line
            push!(games, buffer)
            buffer = ""
        end

        if length(line) > 2
            if (line[2] == '1') | (line[2] == '0')
                buffer = buffer * line
                push!(games, buffer)
                buffer = ""
            end
        end

        if length(games) == 1e6
            lpush(conn, :games, games)
            games = []
        end

        line = readline(file; keep=true)
    end
    lpush(conn, :games, games)
end

function parse_game(chunk::Int=100000)

    conn = RedisConnection()

    n = 0
    game_strs::Vector{String} = []
    while true

        game_strings = pop_chunk_from_list(conn, :games, chunk)

        if length(game_strings) == 0
            println(n)
            return
        end

        for game_string ∈ game_strings

            game = gamefromstring(game_string; annotations=true)

            if headervalue(game, "Termination") != "Normal"
                continue
            end

            try
                event = headervalue(game, "Event")
                white_elo = headervalue(game, "WhiteElo")
                black_elo = headervalue(game, "BlackElo")
                result = headervalue(game, "Result")
                move_str = string(game)[10:end]

                game_str = move_str * "," * result * "," * white_elo * "," * black_elo *","  * event
                push!(game_strs, game_str)

                if length(game_strs) == 1_000
                    lpush(conn, :data, game_strs)
                    game_strs = []
                end
            catch
                println("Missing data")
            end
        end
    end
    lpush(conn, :data, game_strs)
end

function to_dataset(file_path::String, chunk::Int=100000)

    conn = RedisConnection()

    file = open(file_path, "w")

    while true
        game_strings = pop_chunk_from_list(conn, :data, chunk)

        if length(game_strings) == 0
            return
        end

        for g in game_strings
            write(file, g * "\n")
        end
    end
end

function main()

    s = ArgParseSettings()
    @add_arg_table s begin
        "--load"
        help = ".pgn file to load into the job queue"
        "--parse"
        help = "Runs the worker process for the PGN -> FEN processing"
        action = :store_true
        "--to_dataset"
        help = "Converts the data into a dataset"
    end
    args = parse_args(s)

    file = args["load"]
    output_file = args["to_dataset"]

    if file != nothing
        load_games(file)
    elseif args["parse"]
        parse_game()
    elseif output_file != nothing
        to_dataset(output_file)
    end
end

main()
