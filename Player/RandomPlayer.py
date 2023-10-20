from utils import *
from Player.BasePlayer import BasePlayer


class RandomPlayer(BasePlayer):

    def __init__(self, name):
        super().__init__(name)

    def choose_action(self, board, played_cards):
        random.shuffle(self.hand)
        action = self.hand.pop()
        self.current_action = action
        return action
