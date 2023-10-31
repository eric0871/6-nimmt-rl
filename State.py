from utils import *
import multi_elo
import matplotlib.pyplot as plt

class State:

    def __init__(self, card_pool):
        self.original_card_pool = card_pool
        self.card_pool = copy.copy(self.original_card_pool)
        self.players = []
        self.board = [[0] * 6 for _ in range(4)]
        self.played_cards = []
        self.starting_hand_size = 10
        self.scoreboard = []
        self.deterministic = False
        self.available_cards = list(range(1, 105))
        self.elos = []
        self.elo_k = 32

    def add_player(self, player):
        self.players.append(player)

    def set_initial_elos(self, elo=1200, k=32):
        self.elos = [elo] * len(self.players)
        self.elo_k = k

    def deal(self):
        for p in self.players:
            p.prepare_for_next_round()

            if not self.deterministic:
                random.shuffle(self.card_pool)

            for _ in range(self.starting_hand_size):
                p.draw(self.card_pool.pop())
            # print(sorted(p.hand))

            if isinstance(p, MCTSPlayer):
                for c in p.hand:
                    p.available_cards.remove(c)

        for row in range(4):
            start_card = self.card_pool.pop()
            self.board[row][0] = start_card
            #p5.available_cards.remove(start_card)

        # print(self.board)

    def play(self, output_result=False):
        self.scoreboard = [0] * len(self.players)

        self.deal()
        for turn in range(1, self.starting_hand_size+1):
            current_played_cards = []
            current_state = copy.deepcopy(self.board)

            for p in self.players:
                current_played_cards.append((p, p.choose_action(self.board, self.played_cards)))

            for c in current_played_cards:
                if c in p5.available_cards:
                    p5.available_cards.remove(c)

            self.add_cards_to_board(current_played_cards)
            # print(self.board)

            next_state = copy.deepcopy(self.board)

            for i, p in enumerate(self.players):

                p.prepare_for_next_turn()

        self.update_scoreboard()

    def add_cards_to_board(self, current_played_cards):
        current_played_cards = sorted(current_played_cards, key=lambda x: x[1])

        for p, card in current_played_cards:
            enter_position = [0, 0]
            for i in range(4):
                rightmost_card = max(self.board[i])
                if card > rightmost_card > enter_position[1]:
                    enter_position[0] = i
                    enter_position[1] = rightmost_card

            if enter_position == [0, 0]:
                take_row = lowest_bullhead_row(self.board)
                self.take_that(p, take_row)
                self.board[take_row][0] = card

            else:
                enter_slot = self.board[enter_position[0]].index(max(self.board[enter_position[0]])) + 1
                if enter_slot == 5:
                    self.take_that(p, enter_position[0])
                    self.board[enter_position[0]][0] = card
                else:
                    self.board[enter_position[0]][enter_slot] = card

        self.played_cards = sorted(self.played_cards)

    def update_scoreboard(self):
        for i, p in enumerate(self.players):
            self.scoreboard[i] = p.get_score()

    def take_that(self, p, row):
        for i in range(5):
            if self.board[row][i] != 0:
                self.played_cards.append(self.board[row][i])
                p.add_penalty(self.board[row][i])
                self.board[row][i] = 0
            else:
                return

    def prepare_for_next_game(self):
        self.scoreboard = []
        self.card_pool = copy.copy(self.original_card_pool)
        self.board = [[0] * 6 for _ in range(4)]
        self.played_cards = []

    def calculate_ranks(self, arr):
        # Sort the list in ascending order
        sorted_arr = sorted(arr)
        
        # Create a dictionary to store the rank of each element
        rank_dict = {value: index + 1 for index, value in enumerate(sorted_arr)}
        
        # Calculate the rank for each element in the original list
        ranks = [rank_dict[value] for value in arr]
        return ranks


from Player import *
import json

if __name__ == '__main__':
    # Create the initial game state with a deck of cards
    state = State(CARDS)

    # Create 5 players and add them to the game state
    p1 = MCTSPlayer('Player 1', 'random')
    p2 = RulePlayer('Player 2')
    p3 = RulePlayer('Player 3')
    p4 = RandomPlayer('Player 4')
    p5 = RandomPlayer('Player 5')
    state.add_player(p1)
    state.add_player(p2)
    state.add_player(p3)
    state.add_player(p4)
    state.add_player(p5)

    state.set_initial_elos(elo=1200, k=32)

    num_of_games = 800  # Number of games to simulate
    elo_history = []

    # Iterate through a series of game episodes
    for episode in range(1, num_of_games+1):
        if episode % 100 == 0:
            print('episode:', episode)

        # Play the game episode
        state.play()
        # Calculate the ranks of players in the current episode
        ranks = state.calculate_ranks(state.scoreboard)

        # Update Elo ratings based on the ranks
        players = [multi_elo.EloPlayer(place=place, elo=old_elo) for place, old_elo in zip(ranks, state.elos)]
        state.elos = multi_elo.calc_elo(players, state.elo_k)
        elo_history.append(state.elos)
        # Prepare the game state for the next episode
        state.prepare_for_next_game()

    elo_history = np.array(elo_history)
    # for col in range(elo_history.shape[1]):
    #     plt.plot(elo_history[:, col], label=f'Player {col + 1}')

    # plt.xlabel('X-axis')
    # plt.ylabel('Y-axis')
    # plt.legend()
    # plt.title('Multiple Line Plot')

    # # Show the plot or save it to a file
    # plt.show()