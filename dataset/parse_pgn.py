import chess.pgn
import chess
import sys

def mirror_square(sq):
    row, col = sq[0], int(sq[1])
    return row + str(9 - col)

def mirror_move(m):
    move_str = str(m)
    from_sq, to_sq, promo = mirror_square(move_str[:2]), mirror_square(move_str[2:4]), move_str[4:]
    return chess.Move.from_uci(from_sq + to_sq + promo)


def parse_pgn(pgn, output_file):
    node = pgn.game()

    boards = []
    moves = []
    while node: 
        boards.append(node.board())
        node = node.next()
        if node:
            moves.append(node.move)

    boards = boards[:-1]
    for (idx, (b, m)) in enumerate(zip(boards, moves)):

        if (idx % 2) != 0:
            b = b.mirror()
            m = mirror_move(m)
        
        output_file.write(f'{b.fen()},{str(m)}\n')

if __name__ == '__main__':
    path = sys.argv[1]
    pgn = open(path, "r")
    output_file = open(f'../data/{path[:-4]}.csv', "w")
    
    while True:
        pgn_file = chess.pgn.read_game(pgn)
        parse_pgn(pgn_file, output_file)
