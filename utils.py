import random
import numpy as np
import copy
import json

CARDS = [i for i in range(1, 104)]

# Number of Bullheads on each card
BULLHEAD = dict.fromkeys(range(1, 105), 1)
BULLHEAD.update(dict.fromkeys(range(5, 104, 10), 2))
BULLHEAD.update(dict.fromkeys(range(10, 104, 10), 3))
BULLHEAD.update(dict.fromkeys(range(11, 104, 11), 5))
BULLHEAD[55] = 7
BULLHEAD[0] = 0  # For calculating bullhead purpose


def lowest_bullhead_row(board):
    row_bullhead = []
    for row in board:
        bullhead_sum = 0
        for card in row:
            bullhead_sum += BULLHEAD[card]
        row_bullhead.append(bullhead_sum)

    return row_bullhead.index(min(row_bullhead))

def get_score(player_board):
    point = 0
    for card in player_board:
        point += BULLHEAD[card]

    return point
