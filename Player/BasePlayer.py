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

    def choose_action(self, board, played_cards):
        raise NotImplementedError

    def get_score(self):
        return self.cum_bullhead

    def add_penalty(self, card):
        self.cur_bullhead += BULLHEAD[card]
        self.cum_bullhead += BULLHEAD[card]

    def prepare_for_next_round(self):
        self.hand = []
        self.cum_bullhead = 0

    def prepare_for_next_turn(self):
        self.cur_bullhead = 0
