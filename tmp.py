from utils import *
from Player.BasePlayer import BasePlayer
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


class DeepQPlayer(BasePlayer):
    memory = []

    def __init__(self, name, model, target_model, learning_rate=0.001, discount_factor=0.99, exploration=1.0, exploration_decay=0.995, exploration_min=0.01):
        super().__init__(name)
        self.model = model
        self.target_model = target_model
        self.gamma = discount_factor
        self.epsilon = exploration
        self.epsilon_decay = exploration_decay
        self.epsilon_min = exploration_min
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        self.batch_size = 128

    def choose_action(self, board, played_cards):
        # Exploration
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.hand)
        # Exploitation
        with torch.no_grad():
            board = torch.tensor(board, dtype=torch.float32).to(device)
            board = torch.flatten(board)
            q_values = self.model(board)
            print(board)
            print(q_values[10])
            # Filter q_values for only the cards in hand and select the card with max Q value
            q_values_hand = {card: q_values[card] for card in self.hand}
            best_card = min(q_values_hand, key=q_values_hand.get)
            self.current_action = best_card
            return best_card

    def remember(self, state, action, reward, next_state, done, hand):
        self.memory.append((state, action, reward, next_state, done, hand))

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        batch = np.random.choice(self.memory, self.batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(states, dtype=torch.float32).to(device)
        actions = torch.tensor(actions, dtype=torch.int64).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
        dones = torch.tensor(dones, dtype=torch.float32).to(device)

        q_values = self.model(states)
        q_values = q_values.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        next_q_values = self.target_model(next_states)
        next_q_values, _ = next_q_values.max(dim=1)

        targets = rewards + (1 - dones) * self.gamma * next_q_values
        loss = self.loss_fn(q_values, targets.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.memory = []
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def calculate_reward(self):
        if self.cur_bullhead == 0:
            return 0
        else:
            return self.cur_bullhead



class QNetwork(nn.Module):
    def __init__(self, hidden_dim=64):
        super(QNetwork, self).__init__()

        self.fc_layers = nn.Sequential(
            nn.Linear(24, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 104)
        )

    def forward(self, board):
        q_values = self.fc_layers(board)
        return q_values

# class ReplayBuffer:
#     def __init__(self, capacity=10000):
#         self.buffer = deque(maxlen=capacity)
#
#     def push(self, state, action, reward, next_state, done):
#         self.buffer.append((state, action, reward, next_state, done))
#
#     def sample(self, batch_size):
#         state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch size))
#         return state, action, reward, next_state, done
#
#     def __len__(self):
#         return len(self.buffer)