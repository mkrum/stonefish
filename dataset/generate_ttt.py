import copy
import random

import pyspiel
from open_spiel.python.algorithms import minimax


def enumerate_tree(state):
    states = [state]
    for a in state.legal_actions():
        new_state = copy.copy(state)
        new_state.apply_action(a)
        if not new_state.is_terminal():
            states += enumerate_tree(new_state)
    return states


game = pyspiel.load_game("tic_tac_toe")

state = game.new_initial_state()

out = enumerate_tree(state)
out = {str(o): o for o in out}

states = list(out.values())
random.shuffle(states)

for s in states:
    flattened_str = str(s).replace("\n", "")
    _, action = minimax.alpha_beta_search(game, state=s)
    print(f"{flattened_str},{action}")
