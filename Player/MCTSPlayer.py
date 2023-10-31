from utils import *
from Player.BasePlayer import BasePlayer
import math
import time
import multiprocessing


class MCTSPlayer(BasePlayer):
    def __init__(self, name, method, temperature=1.0, num_of_players=5):
        super().__init__(name)
        self.available_cards = list(range(1, 105))
        self.num_of_players = num_of_players
        self.method = method
        self.temperature = temperature
        self.played_cards = []

    def start(self):
        for card in self.hand:
            self.available_cards.remove(card)

        for row in range(4):
            self.available_cards.remove(self.board[row][0])
            self.played_cards.append(self.board[row][0])

        for i in range(9):
            print("current board:", self.board)
            action = self.choose_action(self.board, None)
            print("The best action is: ", action)

            other_players_actions = self.get_user_input()

            self.played_cards.append(action)
            for card in other_players_actions:
                self.available_cards.remove(card)
                self.played_cards.append(card)

            actions = [action] + other_players_actions
            self.board, _ = update_state(self.board, actions, real_game=True)

        print("Game Over")
        return

    def choose_action(self, b, played_cards):
        self.board = b

        n = len(self.hand)
        n_mc = 100 + 15 * (10 - n)
        outcomes = {action: [] for action in self.hand}
        with multiprocessing.Pool() as pool:
            results = pool.map(self.single_mcts_run, [None for _ in range(n_mc)])
            for action, outcome in results:
                outcomes[action].append(outcome)

        best_action = choose_from_outcomes(outcomes)
        self.hand.remove(best_action)

        self.display_outcomes(outcomes)

        return best_action

    def single_mcts_run(self, dummy=None):
        simulated_env = Simulator(copy.deepcopy(self.board), copy.deepcopy(self.hand),
                                  copy.deepcopy(self.available_cards), copy.deepcopy(self.played_cards), self.num_of_players, method=self.method,
                                  temperature=self.temperature)
        action, outcome = simulated_env.play_out()
        return action, outcome

    def display_outcomes(self, outcomes):
        for action in outcomes:
            print(action, round(np.mean(outcomes[action]), 2))

    def prepare_for_next_round(self):
        self.hand = []
        self.cum_bullhead = 0
        self.available_cards = list(range(1, 105))

    def remember_played(self, played_cards_this_round):
        for card in played_cards_this_round:
            if card in self.available_cards:
                self.available_cards.remove(card)

    def get_user_input(self):
        while True:
            other_players_actions = input("Enter other players' actions: ").split()

            # Validate input length
            if len(other_players_actions) != 4:
                print("Invalid input. Please enter exactly 4 numbers.")
                continue

            # Convert to integers
            try:
                other_players_actions = [int(num) for num in other_players_actions]
            except ValueError:
                print("Invalid input. Please enter only numbers.")
                continue

            # Check if all numbers are in the list
            if all(num in self.available_cards for num in other_players_actions):
                return other_players_actions
            else:
                print("One or more cards are not available. Please re-enter.")

class Simulator():
    def __init__(self, board, hero_hand, available_cards, played_cards, num_of_players, method, temperature=1.0):
        self.board = board
        self.hero_hand = hero_hand
        self.available_cards = available_cards
        self.num_of_players = num_of_players
        self.method = method
        self.available_cards_other = list(range(1, 105))
        self.other_player_hands = []
        self.temperature = temperature
        self.played_cards = played_cards

    def play_out(self):
        self.other_player_hands = self.initialize_other_player_hands()
        outcome = 0
        initial_action = None

        for turn in range(len(self.hero_hand)):
            actions = []

            hero_action = random.sample(self.hero_hand, 1)[0]
            actions.append(hero_action)

            if turn == 0:
                initial_action = hero_action

            other_player_actions = self.choose_action_for_others(self.method)
            actions += other_player_actions
            self.hero_hand.remove(hero_action)

            for action in actions:
                self.played_cards.append(action)

            self.board, next_outcome = update_state(self.board, actions)
            outcome += next_outcome

        return initial_action, outcome

    def initialize_other_player_hands(self):
        other_hands = []
        for _ in range(self.num_of_players - 1):
            random_hand = random.sample(self.available_cards, len(self.hero_hand))
            other_hands.append(random_hand)
            for card in random_hand:
                self.available_cards.remove(card)
        return other_hands

    def choose_action_for_others(self, method):
        actions = []
        if method == 'random':
            for i in range(len(self.other_player_hands)):
                other_player_action = random.sample(self.other_player_hands[i], 1)[0]
                actions.append(other_player_action)
                self.other_player_hands[i].remove(other_player_action)

        if method == 'real':
            for i in range(len(self.other_player_hands)):
                self.available_cards_other = list(range(1, 105))
                cards_used = self.other_player_hands[i] + self.played_cards
                cards_used = [x for x in cards_used if x != 0]
                cards_used = set(cards_used)
                for card in cards_used:
                    self.available_cards_other.remove(card)

                outcomes = {action: [] for action in self.other_player_hands[i]}
                n = len(self.other_player_hands[i])
                n_mc = 100 + 15 * (10 - n)

                for _ in range(n_mc):
                    simulated_env = Simulator(copy.deepcopy(self.board), copy.deepcopy(self.other_player_hands[i]),
                                              copy.deepcopy(self.available_cards_other), copy.deepcopy(self.played_cards), self.num_of_players,
                                              method='random')
                    action, outcome = simulated_env.play_out()
                    outcomes[action].append(outcome)

                best_action = choose_from_outcomes_softmax(outcomes, temperature=self.temperature)
                actions.append(best_action)
                self.other_player_hands[i].remove(best_action)

        return actions


if __name__ == '__main__':
    board = [
        [77, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0],
        [32, 0, 0, 0, 0, 0],
        [70, 0, 0, 0, 0, 0]
    ]
    hand = [13, 14, 26, 42, 74, 75, 86, 96, 97, 103]

    agent = MCTSPlayer('Eric', 'real', 5)
    agent.hand = hand
    agent.board = board
    agent.start()
