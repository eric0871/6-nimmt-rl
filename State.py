from utils import *


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

    def add_player(self, player):
        self.players.append(player)


    def deal(self):
        for p in self.players:
            p.prepare_for_next_round()

            if not self.deterministic:
                random.shuffle(self.card_pool)

            for _ in range(self.starting_hand_size):
                p.draw(self.card_pool.pop())
            # print(sorted(p.hand))

        for row in range(4):
            self.board[row][0] = self.card_pool.pop()

        # print(self.board)

    def play(self, output_result=False):
        self.scoreboard = [0] * len(self.players)

        self.deal()

        for turn in range(1, self.starting_hand_size+1):
            current_played_cards = []
            current_state = copy.deepcopy(self.board)

            for p in self.players:
                current_played_cards.append((p, p.choose_action(self.board, self.played_cards)))

            played_card_test = [item[1] for item in current_played_cards]
            # print(played_card_test)
            self.add_cards_to_board(current_played_cards)
            # print(self.board)

            next_state = copy.deepcopy(self.board)

            for i, p in enumerate(self.players):
                if isinstance(p, QlearnPlayer):
                    if turn != 10:
                        reward = p.calculate_reward()
                    else:
                        self.update_scoreboard()
                        sorted_indices = np.argsort(self.scoreboard)
                        ranks = np.argsort(sorted_indices) + 1
                        reward = p.calculate_reward(ranks[i])
                        p.epsilon *= 0.99

                    p.learn(current_state, reward, next_state)

                if isinstance(p, DeepQPlayer):
                    if turn != 10:
                        reward = p.calculate_reward()
                        done = False
                    else:
                        self.update_scoreboard()
                        sorted_indices = np.argsort(self.scoreboard)
                        ranks = np.argsort(sorted_indices) + 1
                        reward = p.calculate_reward(ranks[i])
                        done = True

                    p.remember(current_state, p.current_action, reward, next_state, done)
                    p.train()

                p.prepare_for_next_turn()

        self.update_scoreboard()
        # print(self.scoreboard)

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


from Player import *
import json

if __name__ == '__main__':
    state = State(CARDS)

    # Deep Q Learning Model Initialization
    model = QNetwork().to(device)
    target_model = QNetwork().to(device)
    target_model.load_state_dict(model.state_dict())

    p1 = RulePlayer('Player 1')
    p2 = DeepQPlayer('Player 2', model, target_model)
    p3 = DeepQPlayer('Player 3', model, target_model)
    p4 = RulePlayer('Player 4')
    p5 = RandomPlayer('Player 5')
    state.add_player(p1)
    state.add_player(p2)
    state.add_player(p3)
    state.add_player(p4)
    state.add_player(p5)

    # Train Qlearn Player
    # num_of_games = 1000000
    #
    # for episode in range(num_of_games):
    #     print(episode)
    #     state.play()
    #     state.prepare_for_next_game()
    #
    # print("Learning Completed")
    #

    num_of_games = 30000

    final_scores = [0, 0, 0, 0, 0]
    for episode in range(num_of_games):
        print(episode)
        state.play()
        final_scores = [a + b for a, b in zip(final_scores, state.scoreboard)]
        state.prepare_for_next_game()
        if episode % 100 == 0:
            p2.update_target_network()
    final_scores = [x / num_of_games for x in final_scores]
    print (final_scores)

    num_of_games = 5000

    final_scores = [0, 0, 0, 0, 0]
    for episode in range(num_of_games):
        print(episode)
        state.play()
        final_scores = [a + b for a, b in zip(final_scores, state.scoreboard)]
        state.prepare_for_next_game()
        if episode % 100 == 0:
            p2.update_target_network()
    final_scores = [x / num_of_games for x in final_scores]
    print (final_scores)

    torch.save(model, "deepqlearn.pth")