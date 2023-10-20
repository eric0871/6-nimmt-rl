from utils import *
from Player.BasePlayer import BasePlayer
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque


class DeepQPlayer(BasePlayer):

    def __init__(self, name, model, learning_rate=1e-3, discount_factor=0.99, exploration=1.0, exploration_decay=0.995, exploration_min=0.01):
        super().__init__(name)
        self.model = model
        self.memory = []
        self.gamma = discount_factor
        self.epsilon = exploration
        self.epsilon_decay = exploration_decay
        self.epsilon_min = exploration_min
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        self.batch_size = 500

    def choose_action(self, board, played_cards):
        # Exploration
        if np.random.rand() <= self.epsilon:
            return np.random.choice(range(10))  # Assuming 10 possible actions
        # Exploitation
        with torch.no_grad():
            q_values = self.model(board)
        return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self, batch):
        if len(self.memory) < batch:
            return

        batch = np.random.choice(self.memory, self.batch_size, replace=False)

        for state, action, reward, next_state, done in batch:
            target = reward
            if not done:
                target += self.gamma * torch.max(self.model(next_state[0], next_state[1]))

            current_q = self.model(state[0], state[1])[action]
            loss = self.loss_fn(current_q, target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super(QNetwork, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(4*5*64, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, board):
        board_out = self.conv_layers(board)
        q_values = self.fc_layers(board_out)
        return self.fc(q_values)

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