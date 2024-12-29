import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import checkers
from transformers import GPT2LMHeadModel, GPT2Tokenizer

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

import ast

def query_llm_for_actions(current_state, previous_state, llm_model, tokenizer, game, debug=False):
    """
    Query the LLM for the best actions based on the current and previous states.
    Validate and return the best 7 valid actions.
    """
    # Prepare input prompt
    prompt = (
        "You are playing checkers. Based on the board state, provide the best 7 actions "
        "in the format ((start_x, start_y), (end_x, end_y)). "
        "Previous state:\n"
        f"{previous_state}\n"
        "Current state:\n"
        f"{current_state}\n"
        "Actions:"
    )

    # Tokenize input and ensure proper formatting
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).input_ids
    attention_mask = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).attention_mask

    # Generate output with max_new_tokens
    outputs = llm_model.generate(inputs, max_new_tokens=50, attention_mask=attention_mask)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if debug:
        print("Generated text:", generated_text)

    # Extract and parse actions
    suggested_actions = []
    try:
        actions_text = generated_text.split("Actions:")[-1]
        for line in actions_text.strip().split("\n"):
            try:
                action = eval(line.strip())  # Safely parse tuple
                if isinstance(action, tuple) and len(action) == 2:
                    suggested_actions.append(action)
            except (ValueError, SyntaxError):
                if debug:
                    print(f"Skipping invalid action: {line.strip()}")
    except Exception as e:
        if debug:
            print("Error parsing LLM actions:", e)

    # Validate actions against the environment
    valid_moves = game.GetValidMoves(1)  # Assuming player 1's turn
    filtered_actions = [action for action in suggested_actions if action in valid_moves]

    # If fewer than 7 valid actions are provided, pad with random valid moves
    while len(filtered_actions) < 7 and valid_moves:
        random_action = random.choice(valid_moves)
        if random_action not in filtered_actions:
            filtered_actions.append(random_action)

    if debug:
        print("Final filtered actions:", filtered_actions)

    return filtered_actions[:7]





def train_with_llm():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = CheckersModel().to(device)
    optimizer = optim.NAdam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Load GPT-2 model and tokenizer
    llm_model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    rewards_per_generation = []
    losses_per_generation = []

    learning_rate = 0.5
    discount_factor = 0.95

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
            previous_state = None

            while not done:
                current_state = game.board.copy()

                if player == 1:
                    actions = query_llm_for_actions(current_state, previous_state, llm_model, tokenizer, game)

                    if not actions:
                        break

                    action = actions[0]  # Use the first action suggested by the LLM
                    reward = game.GetScore(verbose=False, player=player)

                    temp_data.append(game.GetFeatures(player))
                    game.PushMove(action)
                    total_reward += reward

                    end = game.EndGame()
                    if end != 0:
                        done = True

                else:
                    # Opponent plays random valid moves for simplicity
                    valid_moves = game.GetValidMoves(player)
                    if not valid_moves:
                        break
                    action = random.choice(valid_moves)
                    game.PushMove(action)

                    end = game.EndGame()
                    if end != 0:
                        done = True

                previous_state = current_state
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

        torch.save(model.state_dict(), "models/llm_self_play_model.pth")

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
    train_with_llm()