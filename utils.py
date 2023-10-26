import random
import numpy as np
import copy
import json

CARDS = [i for i in range(1, 105)]

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


def update_state(board, played_cards, real_game = False):
    hero_action = played_cards[0]
    next_outcome = 0
    played_cards = sorted(played_cards)

    for card in played_cards:
        enter_position = [0, 0]
        for i in range(4):
            rightmost_card = max(board[i])
            if card > rightmost_card > enter_position[1]:
                enter_position[0] = i
                enter_position[1] = rightmost_card

        if enter_position == [0, 0]:

            if real_game:
                take_row = input("Which row does the player take? ")
                take_row = int(take_row) - 1
            else:
                take_row = lowest_bullhead_row(board)

            if card == hero_action:
                for i in range(5):
                    if board[take_row][i] != 0:
                        next_outcome -= BULLHEAD[board[take_row][i]]
                    else:
                        break

            board[take_row] = [0] * 6
            board[take_row][0] = card

        else:
            enter_slot = board[enter_position[0]].index(max(board[enter_position[0]])) + 1
            if enter_slot == 5:

                if card == hero_action:
                    for i in range(5):
                        if board[enter_position[0]][i] != 0:
                            next_outcome -= BULLHEAD[board[enter_position[0]][i]]
                        else:
                            break

                board[enter_position[0]] = [0] * 6
                board[enter_position[0]][0] = card
            else:
                board[enter_position[0]][enter_slot] = card
    return board, next_outcome


def choose_from_outcomes(outcomes):
    best_action = list(outcomes.keys())[0]
    best_mean = - float("inf")

    for action, outcome in outcomes.items():
        if np.mean(outcome) > best_mean:
            best_action = action
            best_mean = np.mean(outcome)
    return best_action
