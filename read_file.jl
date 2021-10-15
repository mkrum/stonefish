
using Distributed

N_procs = 2
addprocs(N_procs)

@everywhere using Redis
@everywhere using Chess
@everywhere using Chess.PGN


@everywhere function worker(lines)
    conn = RedisConnection()
    
    n = 0
    for l ∈ lines
        sg = gamefromstring(l)

        for b ∈ boards(sg)
            f = fen(b)
            incr(conn, f)
            n += 1

            if n % 1000 == 0
                println(n)
            end
        end
    end
end

path = "lichess_db_standard_rated_2013-01.pgn"
file = open(path, "r")
lines = readlines(file)
non_empty_lines = collect(filter(x -> length(x) > 1, lines))
fen_lines = collect(filter(x -> x[1] == '1', non_empty_lines))

splits = Int(ceil(length(fen_lines) / N_procs))

split_lines = collect(Iterators.partition(fen_lines, splitsk))

procs = []
for i ∈ 1:N_procs
    push!(procs, @spawnat (i+1) worker(split_lines[i]))
end

for p ∈ procs
    fetch(p)
end
