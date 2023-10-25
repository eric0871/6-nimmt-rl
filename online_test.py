from Player import *
from utils import *

model = QNetwork().to(device)
model.load_state_dict(torch.load('deepqlearn3.pth'))

board = [
    [95, 97, 98, 100, 101, 0],
    [66, 75, 94, 0, 0, 0],
    [53, 85, 86, 93, 0, 0],
    [90, 91, 96, 103, 104, 0],
]

# board = [
#     [94, 99, 102, 0, 0, 0],
#     [72, 75, 76, 53, 66, 0],
#     [65, 0, 0, 0, 0, 0],
#     [55, 79, 81, 88, 0, 0]
# ]


hand = [44,55]

board = torch.tensor(board, dtype=torch.float32).to(device)
board = board[:,:-1]
board = torch.flatten(board)
q_values = model(board)
q_values_hand = {card: round(float(q_values[card-1]),4) for card in hand}
best_card = min(q_values_hand, key=q_values_hand.get)
print(q_values_hand)
print(best_card)
