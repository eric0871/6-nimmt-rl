from utils import *


class BasePlayer:

    def __init__(self, name):
        self.name = name
        self.hand = []
        self.cum_bullhead = 0
        self.cur_bullhead = 0
        self.current_action = 0

    def draw(self, card):
        self.hand.append(card)

    def choose_action(self, board, played_cards, available):
        raise NotImplementedError

    def get_score(self):
        return self.cum_bullhead

    def add_penalty(self, card):
        self.cur_bullhead += BULLHEAD[card]
        self.cum_bullhead += BULLHEAD[card]

    def calculate_reward(self, rank=0):
        end_game_bonus = 0

        if rank == 1:
            end_game_bonus = -25
        elif rank == 2:
            end_game_bonus = -15
        elif rank == 3:
            end_game_bonus = -7
        elif rank == 4:
            end_game_bonus = -3

        return (((self.cur_bullhead + end_game_bonus) + 25) / 50)

    def prepare_for_next_round(self):
        self.hand = []
        self.cum_bullhead = 0

    def prepare_for_next_turn(self):
        self.cur_bullhead = 0
