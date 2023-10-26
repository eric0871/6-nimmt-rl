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

    def __init__(self, name, model, target_model, learning_rate=0.0001, discount_factor=0.99, exploration=1.0, exploration_decay=0.995, exploration_min=0.01):
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
            self.current_action = np.random.choice(self.hand)
            self.hand.remove(self.current_action)
            return self.current_action
        # Exploitation
        with torch.no_grad():
            # Format Board
            board_nn = [row[:-1] for row in board]
            resized_board = [item for sublist in board_nn for item in sublist]
            positions_for_card = []
            bullhead_per_row = []
            for line in board:
                positions_for_card.append(line.count(0))
                bullhead_sum = 0
                for element in line:
                    bullhead_sum += BULLHEAD[element]
                bullhead_per_row.append(bullhead_sum)

            hand_nn = copy.deepcopy(self.hand)
            hand_nn.extend([0] * (10 - len(hand_nn)))

            large_state_n = hand_nn + resized_board + positions_for_card + bullhead_per_row
            feed = torch.tensor(large_state_n, dtype=torch.float32).to(device)
            q_values = self.model(feed)
            #print(board)
            #print(q_values[10])

            # Filter q_values for only the cards in hand and select the card with max Q value
            q_values_hand = {card: q_values[card-1] for card in self.hand}
            best_card = min(q_values_hand, key=q_values_hand.get)
            self.hand.remove(best_card)
            self.current_action = best_card
            return best_card

    def remember(self, state, action, reward, next_state, done, hand):
        self.memory.append((state, action, reward, next_state, done, hand))

    def train(self):
        if len(self.memory) < 1000:
            return

        batch = np.random.choice(len(self.memory), self.batch_size, replace=False)
        batch = [self.memory[i] for i in batch]

        states, actions, rewards, next_states, dones, hands = zip(*batch)
        #print("hand:", hands[0])
        #print("action:", actions[0])
        # print(states[0], actions[0], rewards[0], next_states[0])
        states = torch.tensor(states, dtype=torch.float32).to(device)
        actions = torch.tensor(actions, dtype=torch.int64).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
        dones = torch.tensor(dones, dtype=torch.float32).to(device)

        action_masks = []
        #print(hands[0])
        for each_hand in hands:
            if len(each_hand) == 0:
                mask = [0]*104
                action_masks.append(mask)
            else:
                mask = [float('inf')]*104
                for playable_card in each_hand:
                    mask[playable_card-1] = 0
                action_masks.append(mask)
        action_masks = torch.tensor(action_masks).to(device)

        q_values = self.model(states)
        #print('training q:', q_values[0])
        q_values = q_values.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        #print("gathered q:", q_values[0])
        #print("q_values:", q_values)

        next_q_values = self.target_model(next_states)
        #print('next_q_values:', next_q_values[0])
        masked_next_q_values = next_q_values + action_masks
        #print('masked_next_q_values:', masked_next_q_values[0])
        min_next_q_values, _ = masked_next_q_values.min(dim=1)
        #print('min_next_q_values:', min_next_q_values[0])
        targets = rewards + (1 - dones) * self.gamma * min_next_q_values
        #print ('reward:', rewards[0])
        #print('targets:', targets[0])

        loss = self.loss_fn(q_values, targets.detach())

        self.optimizer.zero_grad()
        loss.backward()
        print(loss)
        self.optimizer.step()

        self.memory = []
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())


class QNetwork(nn.Module):
    def __init__(self, hidden_dim=64):
        super(QNetwork, self).__init__()

        self.fc_layers = nn.Sequential(
            nn.Linear(38, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 104)
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