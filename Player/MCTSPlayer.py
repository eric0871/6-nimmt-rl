from utils import *
from Player.BasePlayer import BasePlayer
import math


class MCTSPlayer(BasePlayer):
    def __init__(self, name, num_of_players=5):
        super().__init__(name)
        self.available_cards = list(range(1, 105))
        self.num_of_players = num_of_players

    def start(self):
        for card in self.hand:
            self.available_cards.remove(card)

        for row in range(4):
            self.available_cards.remove(self.board[row][0])

        for i in range(9):
            print("current board:", self.board)
            action = self.choose_action(self.board, None)
            print("The best action is:", action)

            other_players_actions = input("Enter other player's actions:").split()
            other_players_actions = [int(card) for card in other_players_actions]

            for card in other_players_actions:
                self.available_cards.remove(card)

            actions = [action] + other_players_actions
            self.board, _ = update_state(self.board, actions, real_game=True)

        print("Game Over")
        return

    def choose_action(self, b, played_cards):
        self.board = b

        n = len(self.hand)
        n_mc = min(50000, 3000 * math.factorial(n))
        outcomes = {action: [] for action in self.hand}
        for _ in range(n_mc):
            simulated_env = Simulator(copy.deepcopy(self.board), copy.deepcopy(self.hand),
                                      copy.deepcopy(self.available_cards), self.num_of_players)
            action, outcome = simulated_env.play_out()
            outcomes[action].append(outcome)
        best_action = choose_from_outcomes(outcomes)
        self.hand.remove(best_action)

        self.display_outcomes(outcomes)

        return best_action

    def display_outcomes(self, outcomes):
        for action in outcomes:
            print(action, round(np.mean(outcomes[action]), 2))

    def prepare_for_next_round(self):
        self.hand = []
        self.cum_bullhead = 0
        self.available_cards = list(range(1, 105))

class Simulator():
    def __init__(self, board, hero_hand, available_cards, num_of_players):
        self.board = board
        self.hero_hand = hero_hand
        self.available_cards = available_cards
        self.num_of_players = num_of_players

    def play_out(self):
        other_player_hands = self.initialize_other_player_hands()
        outcome = 0
        initial_action = None

        for turn in range(len(self.hero_hand)):
            actions = []

            hero_action = random.sample(self.hero_hand, 1)[0]
            self.hero_hand.remove(hero_action)
            actions.append(hero_action)

            if turn == 0:
                initial_action = hero_action

            for i in range(len(other_player_hands)):
                # other_player_action = self.choose_action_for_others(other_player_hands)
                other_player_action = random.sample(other_player_hands[i], 1)[0]
                actions.append(other_player_action)
                other_player_hands[i].remove(other_player_action)
            self.board, next_outcome = update_state(self.board, actions)
            outcome += next_outcome

        return initial_action, outcome

    def initialize_other_player_hands(self):
        other_player_hands = []
        for _ in range(self.num_of_players - 1):
            random_hand = random.sample(self.available_cards, len(self.hero_hand))
            other_player_hands.append(random_hand)
            for card in random_hand:
                self.available_cards.remove(card)
        return other_player_hands

    def choose_action_for_others(self, other_player_hands, method='random'):
        for hands in other_player_hands:
            pass


board = [
    [95, 0, 0, 0, 0, 0],
    [87, 0, 0, 0, 0, 0],
    [80, 0, 0, 0, 0, 0],
    [74, 0, 0, 0, 0, 0]
]
hand = [12,23,36,40,63,70,71,77,78,101]

agent = MCTSPlayer('Eric', 5)
agent.hand = hand
agent.board = board
agent.start()
