
using Distributed
using ArgParse
using Redis
using Chess
using Chess.UCI
using Chess.PGN


"""
    populate_fen_queue(path::String)

Fills the "games" list in our redis server with the list of raw pgn games
stored in `path`. 
"""
function populate_fen_queue(path::String)
    conn = RedisConnection()

    file = open(path, "r")
    lines = readlines(file)
    for chunk in Iterators.partition(lines, Int(1e6))
        fen_lines = collect(filter(x -> length(x) > 1 && x[1] == '1', chunk))
        @time lpush(conn, :games, fen_lines)
    end
end

"""
    populate_stockfish_queue()

Takes all the unique boards stored as keys in the DB and put them into a list
in the DB called "boards".
"""
function populate_stockfish_queue()
    conn = RedisConnection()
    #data = collect(keys(conn, "*"))

    # chunk these operations to avoid timeout
    for chunk in Iterators.partition(keys(conn, "*"), Int(1e5))
        println(length(chunk))
        # add them to the "boards list"
        lpush(conn, :boards, Vector{String}(chunk))
        # remove them as a key->count pair
        del(conn, Vector{String}(chunk))
    end
end

"""
    parse_game_string(game_string::String)::Vector{String}

Converts the raw PGN game `game_string` into a list of FEN boards also
represented as strings.
"""
function parse_game_string(game_string::String)::Vector{String}
    game = gamefromstring(game_string)

    # Do not add the last board to the mix, because there is no move for it!
    raw_boards = boards(game)[1:end-1]
    fen_boards = collect(map(fen, raw_boards))
    return fen_boards
end

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

"""
    fen_worker(chunk::Int = 1000)

Continually pops a batch of data from "games", gets all of their unique boards,
and then stores them back into the database as their fen representation.
"""
function fen_worker(chunk::Int = 100000)
    conn = RedisConnection()

    println("working...")

    n = 0
    while true

        game_strings = pop_chunk_from_list(conn, :games, chunk)

        # Out of data
        if length(game_strings) == 0
            println(n)
            return
        end

        pipeline = open_pipeline(conn)

        for game_string ∈ game_strings
            fen_boards = parse_game_string(game_string)

            for f in fen_boards
                incr(pipeline, f)
                n += 1

                # Log progress every thousand boards
                if n % 100000 == 0
                    println(n)
                end

            end
        end

        # Flush the pipeline
        read_pipeline(pipeline)
    end
end

"""
    get_move(engine::Engine, fen_board::String)::String

Returns the bestmove found by the engine for this board state. This is using
Stockfish with a search depth of 15. According to
http://web.ist.utl.pt/diogo.ferreira/papers/ferreira13impact.pdf, this should
correspond to an ELO of about 2563. Picked this to balance chess performance
and speed. These seem to clock in anywhere between [.01, .3].
"""
function get_move(engine::Engine, fen_board::String)::String
    board = fromfen(fen_board)
    setboard(engine, board)
    move = search(engine, "go depth 2").bestmove
    return tostring(move)
end

"""
    stockfish_worker(chunk_size::Int = 100)

Worker process, gets pops chunks of boards from the redis server, runs them
through the engine, and then stores the board and the move as a key value pair
"""
function stockfish_worker(chunk_size::Int = 100000)
    conn = RedisConnection()

    # create our engine
    sf = runengine("stockfish")

    pipeline = open_pipeline(conn)

    n = 0
    while true
        boards = pop_chunk_from_list(conn, :boards, chunk_size)

        # No more data left
        if length(boards) == 0
            println(n)
            return
        end

        for b ∈ boards
            move = get_move(sf, b)
            set(pipeline, b, move)

            n += 1
            # Log process every 100 boards
            if n % 100 == 0
                println(n)
            end

        end

        read_pipeline(pipeline)
    end

end

"""
    to_dataset(path::String)

Converts the key value board move pairs stored in our DB into a text file
format saved at `path` to use as a dataset.
"""
function to_dataset(path::String)
    conn = RedisConnection()

    boards = collect(keys(conn, "*"))

    # remove our two job queues, just in case
    boards = collect(filter(x -> (x != "boards") & (x != "games"), boards))

    moves = []
    for board_chunk ∈ Iterators.partition(boards, Int(1e6))
        new_moves = mget(conn, Vector{String}(board_chunk))
        moves = [moves; new_moves]
    end

    io = open(path, "w")

    for (board, move) in zip(boards, moves)
        rep = board * "," * move * "\n"
        write(io, rep)
    end
    close(io)
end

function main()

    s = ArgParseSettings()
    @add_arg_table s begin
        "--load"
        help = ".pgn file to load into the job queue"
        "--fen_worker"
        help = "Runs the worker process for the PGN -> FEN processing"
        action = :store_true
        "--stock"
        help = "Sets up the job queque for the stockfish workers"
        action = :store_true
        "--stock_worker"
        help = "Runs the worker process for FEN -> bestmove"
        action = :store_true
        "--to_dataset"
        help = "Converts the data into a dataset"
    end
    args = parse_args(s)

    file = args["load"]
    output_file = args["to_dataset"]

    if file != nothing
        populate_fen_queue(file)
    elseif args["stock"]
        populate_stockfish_queue()
    elseif args["fen_worker"]
        fen_worker()
    elseif args["stock_worker"]
        stockfish_worker()
    elseif output_file != nothing
        to_dataset(output_file)
    end

end

main()
