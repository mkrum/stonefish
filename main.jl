
using Chess
using Chess.UCI

sf = runengine("stockfish")

function engine_game(engine)
    g = Game()
    while !isterminal(g)
        setboard(engine, g)
        move = search(engine, "go nodes 10000").bestmove
        domove!(g, move)
    end
    g
end

engine_game(sf)
