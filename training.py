import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import checkers

def concatenate(array1, array2):
    return array1 + array2

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

    def GetModel(Oppenent):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        model = CheckersModel().to(device)
        optimizer = torch.optim.NAdam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        winrates = []
        learning_rate = 0.5
        discount_factor = 0.95
        exploration = 0.95
        win = 0
        lose = 0
        draw = 0

        for generations in tqdm(range(200)):
            data = []
            labels = []
            for g in range(10):
                temp_data = []
                game = checkers.Checkers()
                player = 1
                count = 0
                while True:
                    count += 1
                    end2 = 0
                    if count > 1000:
                        draw += 1
                        break

                    if player == 1:
                        leafs = game.minmax(player, RL=True)
                        Leaf = torch.zeros((len(leafs), 5), device=device)
                        for l in range(len(leafs)):
                            tensor = torch.tensor(leafs[l][2][:5], dtype=torch.float32, device=device)
                            Leaf[l] = tensor
                        scores = model(Leaf).detach().cpu().numpy()
                        if len(scores) == 0:
                            end2 = -player
                            continue
                        i = np.argmax(scores)
                        game.PushMove(leafs[i][0])
                        tab = leafs[i][2][:5]
                        temp_data.append(tab)
                    elif player == -1:
                        if Oppenent == "random":
                            leafs = game.GetValidMoves(player)
                            if len(leafs) == 0:
                                end2 = -player
                                continue
                            move = random.choice(leafs)
                            game.PushMove(move)
                        elif Oppenent == "minmax":
                            moves = game.minmax(player)
                            if len(moves) == 0:
                                end2 = -player
                                continue
                            if random.random() >= exploration:
                                move = random.choice(game.GetValidMoves(player))
                            else:
                                move = random.choice(moves)
                            game.PushMove(move)
                        elif Oppenent == "itself":
                            leafs = game.minmax(player, RL=True)
                            Leaf = torch.zeros((len(leafs), 5), device=device)
                            for l in range(len(leafs)):
                                tensor = torch.tensor(leafs[l][2][:5], dtype=torch.float32, device=device)
                                Leaf[l] = tensor
                            scores = model(Leaf).detach().cpu().numpy()
                            if len(scores) == 0:
                                end2 = -player
                                continue
                            if random.random() >= exploration:
                                move = random.choice(leafs)[0]
                            else:
                                i = np.argmax(scores)
                                game.PushMove(leafs[i][0])
                        else:
                            raise ValueError("Invalid opponent type")

                    end = game.EndGame()

                    if end == 1 or end2 == 1:
                        win += 1
                        reward = 10
                        temp_array = np.vstack(temp_data[1:]).astype(np.float32)
                        temp_tensor = torch.tensor(temp_array, dtype=torch.float32, device=device)
                        old_prediction = model(temp_tensor).detach()
                        optimal_future_value = torch.ones_like(old_prediction, device=device)
                        temp_labels = old_prediction + learning_rate * (reward + discount_factor * optimal_future_value - old_prediction)
                        data.extend(temp_data[1:])
                        labels.extend(temp_labels.cpu().numpy())
                        break

                    elif end == -1 or end2 == -1:
                        lose += 1
                        reward = -10
                        temp_array = np.vstack(temp_data[1:]).astype(np.float32)
                        temp_tensor = torch.tensor(temp_array, dtype=torch.float32, device=device)
                        old_prediction = model(temp_tensor).detach()
                        optimal_future_value = -torch.ones_like(old_prediction, device=device)
                        temp_labels = old_prediction + learning_rate * (reward + discount_factor * optimal_future_value - old_prediction)
                        data.extend(temp_data[1:])
                        labels.extend(temp_labels.cpu().numpy())
                        break

                    player = -player

            # Convert data and labels to tensors
            data_array = np.vstack(data).astype(np.float32)
            data_tensor = torch.tensor(data_array, dtype=torch.float32, device=device)
            labels_tensor = torch.tensor(labels, dtype=torch.float32, device=device).view(-1, 1)

            # Dataset and DataLoader for training
            dataset = TensorDataset(data_tensor, labels_tensor)
            dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

            for epoch in range(16):
                for batch_data, batch_labels in dataloader:
                    optimizer.zero_grad()
                    predictions = model(batch_data)
                    loss = criterion(predictions, batch_labels)
                    loss.backward()
                    optimizer.step()

            winrate = int((win) / (win + draw + lose) * 100)
            winrates.append(winrate)

            # Save the model
            torch.save(model.state_dict(), f"models/{Oppenent}_model.pth")

        # Plot win rates
        plt.plot(range(len(winrates)), winrates, marker='o', linestyle='-')
        plt.title('Rates of Win')
        plt.xlabel('Generations')
        plt.ylabel('Win Rate [%]')
        plt.show()

if __name__ == "__main__":
    Oppenent = "itself"  # Choose opponent: "random", "minmax", "itself"
    print(Oppenent)
    GetModel(Oppenent=Oppenent)