from utils import *
from Player.BasePlayer import BasePlayer


class RulePlayer(BasePlayer):

    def __init__(self, name):
        super().__init__(name)

    def choose_action(self, board, played_cards):
        max_cards = []
        empty_positions = []
        for i in range(4):
            max_card = max(board[i])
            max_cards.append(max_card)
            max_card_index = board[i].index(max_card)
            empty_positions.append(4-max_card_index)

            for j in range(4-max_card_index):
                safe_card = max_card + j + 1
                if safe_card in self.hand:
                    self.hand.remove(safe_card)
                    return safe_card

        for i in range(4):
            if empty_positions[i] == 4:
                try:
                    next_largest = min(x for x in max_cards if x > max_cards[i])
                except ValueError:
                    next_largest = 105

                for j in range(max_cards[i]+1, next_largest):
                    if j in self.hand:
                        self.hand.remove(j)
                        return j

        for i in range(4):
            if empty_positions[i] == 3:
                try:
                    next_largest = min(min(x for x in max_cards if x > max_cards[i]), max_cards[i]+12)
                except ValueError:
                    next_largest = 105

                for j in range(max_cards[i]+1, next_largest):
                    if j in self.hand:
                        self.hand.remove(j)
                        return j

        for i in range(4):
            if empty_positions[i] == 2:
                try:
                    next_largest = min(min(x for x in max_cards if x > max_cards[i]), max_cards[i]+6)
                except ValueError:
                    next_largest = 105

                for j in range(max_cards[i]+1, next_largest):
                    if j in self.hand:
                        self.hand.remove(j)
                        return j

        for i in range(4):
            if empty_positions[i] == 1:
                safe_card = max_cards[i]+2
                if safe_card in self.hand:
                    self.hand.remove(safe_card)
                    return safe_card

        lowest_max_row = min(max_cards)
        for i in range(lowest_max_row-1, 0, -1):
            if i in self.hand:
                self.hand.remove(i)
                return i

        random.shuffle(self.hand)
        action = self.hand.pop()
        self.current_action = action
        return action

