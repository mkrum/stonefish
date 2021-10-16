
using Distributed
using ArgParse
using Redis
using Chess
using Chess.UCI
using Chess.PGN


"""
Fills the "games" list in our redis server with the list of raw pgn games.
These will be parsed into the unique FEN board states by the fen_workers. 
"""
function populate_fen_queue(path::String)
    conn = RedisConnection()

    file = open(path, "r")
    lines = readlines(file)
    non_empty_lines = collect(filter(x -> length(x) > 1, lines))
    fen_lines = collect(filter(x -> x[1] == '1', non_empty_lines))

    @time lpush(conn, :games, fen_lines)
end

"""
Takes all the unique boards found in the FEN step and loads them into a job
queue called "boards"
"""
function populate_stockfish_queue()
    conn = RedisConnection()
    data = collect(keys(conn, "*"))

    for chunk in Iterators.partition(data, Int(1e6))
        lpush(conn, :boards, Vector{String}(chunk))
        del(conn, Vector{String}(chunk))
    end
end

"""
Converts the raw PGN game into a list of FEN boards
"""
function parse_game_string(game_string::String)::Vector{String}
    game = gamefromstring(game_string)

    # Do not add the last board to the mix, because there is no move for it!
    raw_boards = boards(game)[1:end-1]
    fen_boards = collect(map(fen, raw_boards))
    return fen_boards
end

"""
Pops a chunk of data from a list stored on our redis server
"""
function pop_chunk_from_list(
    conn::RedisConnection,
    list::Symbol,
    chunk_size::Int,
)::Vector{String}
    pipeline = open_pipeline(conn)
    lrange(pipeline, list, 0, (chunk_size - 1))
    ltrim(pipeline, list, chunk_size, -1)
    data = read_pipeline(pipeline)[1]
    return data
end

"""
Continually pops a batch of data from "games", gets all of their unique boards,
and then stores them back into the database as their fen representation.
"""
function fen_worker(chunk::Int = 1000)
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
Returns the bestmove found by the engine for this board state
"""
function get_move(engine::Engine, fen_board::String)::String
    board = fromfen(fen_board)
    setboard(engine, board)
    move = search(engine, "go depth 15").bestmove
    return tostring(move)
end

"""
Worker process, gets pops chunks of boards from the redis server, runs them
through the engine, and then stores the board and the move as a key value pair
"""
function stockfish_worker(chunk_size::Int = 100)
    conn = RedisConnection()

    # create our engine
    sf = runengine("stockfish")

    pipeline = open_pipeline(conn)

    n = 0
    while true
        boards = pop_chunk_from_list(conn, :boards, chunk_size)

        if length(boards) == 0
            println(n)
            return
        end

        for b ∈ boards
            move = get_move(sf, b)
            set(pipeline, b, move)

            n += 1
            if n % 100 == 0
                println(n)
            end

        end

        read_pipeline(pipeline)
    end

end

"""
Tokenize board
"""
function tokenize_board(b::Board)
    out_str = ""
    for ri ∈ 1:8
        r = SquareRank(ri)
        for fi ∈ 1:8
            f = SquareFile(fi)
            p = pieceon(b, f, r)
            if isok(p)
                out_str *= tochar(p) * ","
            else
                out_str *= "e,"
            end
        end
    end

    if sidetomove(b) == WHITE::PieceColor
        out_str *= "w,"
    else
        out_str *= "b,"
    end

    return out_str
end

"""
Tokenize move
"""
function tokenize_move(m::Move)
    return tostring(from(m)) * "," * tostring(to(m)) * (ispromotion(m) ? tochar(promotion(m)) : "")
end

"""
Converts the key value pairs 
"""
function to_dataset(path::String)
    conn = RedisConnection()

    sf = runengine("stockfish")

    boards = collect(keys(conn, "*"))

    # remove our two job queues, just in case
    boards = collect(filter(x -> (x != "boards") & (x != "games"), boards))
    moves = mget(conn, boards)

    io = open(path, "w")

    for (board, move) in zip(boards, moves)
        board = fromfen(board) 
        move = movefromstring(move)
        rep = tokenize_board(board) * tokenize_move(move) * "\n"
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
