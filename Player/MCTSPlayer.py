from utils import *
from Player.BasePlayer import BasePlayer
import math
import time
import concurrent.futures


def display_outcomes(outcomes):
    for action in outcomes:
        print(action, round(np.mean(outcomes[action]), 2))


class MCTSPlayer(BasePlayer):
    def __init__(self, name, method, num_of_players=5):
        super().__init__(name)
        self.available_cards = list(range(1, 105))
        self.num_of_players = num_of_players
        self.method = method

    def start(self):
        for card in self.hand:
            self.available_cards.remove(card)

        for row in range(4):
            self.available_cards.remove(self.board[row][0])

        for i in range(9):
            print("current board:", self.board)
            action = self.choose_action(self.board, None)
            print("The best action is: ", action)

            other_players_actions = input("Enter other players' actions: ").split()
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
        n_mc = min(100, 50 * math.factorial(n))
        outcomes = {action: [] for action in self.hand}
        for _ in range(n_mc):
            #print(_)
            start_time = time.time()

            simulated_env = Simulator(copy.deepcopy(self.board), copy.deepcopy(self.hand),
                                      copy.deepcopy(self.available_cards), self.num_of_players, method=self.method)
            action, outcome = simulated_env.play_out()
            outcomes[action].append(outcome)

            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Code executed in: {elapsed_time:.5f} seconds")

        best_action = choose_from_outcomes(outcomes)
        self.hand.remove(best_action)

        display_outcomes(outcomes)

        return best_action

    def single_mcts_run(self):
        simulated_env = Simulator(copy.deepcopy(self.board), copy.deepcopy(self.hand),
                                  copy.deepcopy(self.available_cards), self.num_of_players, method=self.method)
        action, outcome = simulated_env.play_out()
        return action, outcome

    def prepare_for_next_round(self):
        self.hand = []
        self.cum_bullhead = 0
        self.available_cards = list(range(1, 105))


class Simulator():
    def __init__(self, board, hero_hand, available_cards, num_of_players, method):
        self.board = board
        self.hero_hand = hero_hand
        self.available_cards = available_cards
        self.num_of_players = num_of_players
        self.method = method

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

            other_player_actions, other_player_hands = self.choose_action_for_others(other_player_hands, self.method)
            actions += other_player_actions

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

    def choose_action_for_others(self, other_player_hands, method):
        actions = []
        if method == 'random':
            for i in range(len(other_player_hands)):
                other_player_action = random.sample(other_player_hands[i], 1)[0]
                actions.append(other_player_action)
                other_player_hands[i].remove(other_player_action)

        if method == 'real':
            for i in range(len(other_player_hands)):
                n = len(other_player_hands[i])
                n_mc = min(100, 50 * math.factorial(n))
                available_cards_other = list(range(1, 105))
                cards_used = other_player_hands[i] + np.array(self.board).ravel().tolist()
                cards_used = [x for x in cards_used if x != 0]
                cards_used = set(cards_used)
                for card in cards_used:
                    available_cards_other.remove(card)
                outcomes = {action: [] for action in other_player_hands[i]}

                for _ in range(n_mc):
                    simulated_env = Simulator(copy.deepcopy(self.board), copy.deepcopy(other_player_hands[i]),
                                              copy.deepcopy(available_cards_other), self.num_of_players,
                                              method='random')
                    action, outcome = simulated_env.play_out()
                    outcomes[action].append(outcome)

                best_action = choose_from_outcomes_softmax(outcomes)
                actions.append(best_action)
                other_player_hands[i].remove(best_action)

        return actions, other_player_hands


board = [
    [41, 0, 0, 0, 0, 0],
    [43, 0, 0, 0, 0, 0],
    [67, 0, 0, 0, 0, 0],
    [39, 0, 0, 0, 0, 0]
]
hand = [7,9,10,26,38,46,65,81,88,93]

agent = MCTSPlayer('Eric', 'real', 5)
agent.hand = hand
agent.board = board
agent.start()
