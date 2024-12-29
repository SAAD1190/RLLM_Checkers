import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import checkers

class CheckersModel(nn.Module):
    def __init__(self):
        super(CheckersModel, self).__init__()
        self.fc1 = nn.Linear(5, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train_self_play():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = CheckersModel().to(device)
    optimizer = optim.NAdam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    rewards_per_generation = []
    losses_per_generation = []

    learning_rate = 0.5
    discount_factor = 0.95
    exploration = 0.95

    for generation in tqdm(range(200)):
        data = []
        labels = []
        total_reward = 0
        total_loss = 0

        for game_round in range(10):
            game = checkers.Checkers()
            player = 1
            temp_data = []
            done = False

            while not done:
                if player == 1:
                    leafs = game.minmax(player, RL=True)
                    if not leafs:
                        break

                    # Pre-convert the list of arrays to a NumPy array
                    leaf_features = np.array([leaf[2][:5] for leaf in leafs], dtype=np.float32)

                    # Convert the NumPy array to a PyTorch tensor
                    Leaf = torch.tensor(leaf_features, dtype=torch.float32, device=device)

                    scores = model(Leaf).detach().cpu().numpy()

                    if random.random() < exploration:
                        move_idx = random.randint(0, len(leafs) - 1)
                    else:
                        move_idx = np.argmax(scores)

                    move = leafs[move_idx][0]
                    reward = game.GetScore(verbose=False, player=player)

                    temp_data.append(leafs[move_idx][2][:5])
                    game.PushMove(move)
                    total_reward += reward

                    end = game.EndGame()
                    if end != 0:
                        done = True

                else:
                    # Self-play for player -1
                    leafs = game.minmax(player, RL=True)
                    if not leafs:
                        break

                    Leaf = torch.tensor([leaf[2][:5] for leaf in leafs], dtype=torch.float32, device=device)
                    scores = model(Leaf).detach().cpu().numpy()

                    if random.random() < exploration:
                        move_idx = random.randint(0, len(leafs) - 1)
                    else:
                        move_idx = np.argmax(scores)

                    move = leafs[move_idx][0]
                    reward = game.GetScore(verbose=False, player=player)

                    temp_data.append(leafs[move_idx][2][:5])
                    game.PushMove(move)
                    total_reward += reward

                    end = game.EndGame()
                    if end != 0:
                        done = True

                player = -player

            if temp_data:
                temp_array = np.vstack(temp_data).astype(np.float32)
                temp_tensor = torch.tensor(temp_array, dtype=torch.float32, device=device)
                old_predictions = model(temp_tensor).detach()
                optimal_future_value = torch.ones_like(old_predictions, device=device)
                temp_labels = old_predictions + learning_rate * (
                    total_reward + discount_factor * optimal_future_value - old_predictions
                )
                data.extend(temp_data)
                labels.extend(temp_labels.cpu().numpy())

        data_tensor = torch.tensor(np.vstack(data), dtype=torch.float32, device=device)
        labels_tensor = torch.tensor(np.array(labels, dtype=np.float32), dtype=torch.float32, device=device).view(-1, 1)

        dataset = TensorDataset(data_tensor, labels_tensor)
        dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

        for batch_data, batch_labels in dataloader:
            optimizer.zero_grad()
            predictions = model(batch_data)
            loss = criterion(predictions, batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        rewards_per_generation.append(total_reward / 10)
        losses_per_generation.append(total_loss / len(dataloader))

        torch.save(model.state_dict(), "models/self_play_model.pth")

    # Plot rewards and losses
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(rewards_per_generation, marker='o', linestyle='-')
    plt.title('Average Reward Per Generation')
    plt.xlabel('Generation')
    plt.ylabel('Average Reward')

    plt.subplot(1, 2, 2)
    plt.plot(losses_per_generation, marker='o', linestyle='-')
    plt.title('Average Loss Per Generation')
    plt.xlabel('Generation')
    plt.ylabel('Average Loss')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    train_self_play()