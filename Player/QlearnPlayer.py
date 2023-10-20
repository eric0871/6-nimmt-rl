from utils import *
from Player.BasePlayer import BasePlayer


class QlearnPlayer(BasePlayer):
    # with open("Q.json", "r") as file:
    #     Q = json.load(file)
    Q = {}

    def __init__(self, name):
        super().__init__(name)
        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 0.1

    def get_Q(self, state, action):
        return self.Q.get(str((state, action)), 0.0)

    def choose_action(self, board, played_cards):
        if random.uniform(0, 1) < self.epsilon:
            random.shuffle(self.hand)
            action = self.hand.pop()
            self.current_action = action
            return action

        else:
            q_values = [self.get_Q(board, action) for action in self.hand]
            max_q_value = max(q_values)
            action = self.hand.pop(q_values.index(max_q_value))
            self.current_action = action
            return action

    def learn(self, state, reward, next_state):
        old_q_value = self.get_Q(state, self.current_action)
        future_rewards = [self.get_Q(next_state, next_action) for next_action in self.hand]
        max_future_reward = max(future_rewards) if future_rewards else 0.0
        # Q-learning update formula
        new_q_value = old_q_value + self.alpha * (reward + self.gamma * max_future_reward - old_q_value)
        self.Q[str((state, self.current_action))] = new_q_value

    def calculate_reward(self, rank=0):
        end_game_bonus = 0

        if rank == 1:
            end_game_bonus = 20
        elif rank == 2:
            end_game_bonus = 7
        elif rank == 3:
            end_game_bonus = 2
        elif rank == 4:
            end_game_bonus = 1

        if self.cur_bullhead == 0:
            return 0.5 + end_game_bonus
        else:
            return self.cur_bullhead + end_game_bonus
