import checkers
import matplotlib.pyplot as plt
import torch
from transformers import GPTNeoXForCausalLM, GPT2Tokenizer
import numpy as np
import random
from tqdm import tqdm
import os

# Initialize GPT-NeoX Model
model_name = "EleutherAI/gpt-neo-2.7B"
model = GPTNeoXForCausalLM.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

def concatenate(array1, array2):
    for i in range(len(array2)):
        array1.append(array2[i])
    return array1  

def get_top_moves_from_llm(board_state, num_moves=3):
    """
    Get the top N moves from GPT-NeoX model based on the current board state.
    """
    prompt = f"Game board: {board_state}\nSuggest the top {num_moves} best moves:"
    
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Generate LLM output
    outputs = model.generate(**inputs, max_length=150, num_return_sequences=1)
    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract moves from output
    lines = prediction.split("\n")
    moves = [line.split(":")[-1].strip() for line in lines if ":" in line][-num_moves:]  # Extract last `num_moves`
    return moves

def evaluate_moves_and_choose_best(game, candidate_moves):
    """
    Evaluate the 3 moves and choose the best based on game simulation.
    """
    best_move = None
    best_score = float('-inf')

    for move in candidate_moves:
        temp_game = game.clone()  # Clone the game to simulate without affecting the actual game
        try:
            temp_game.PushMove(move)  # Apply the move
            score = temp_game.evaluate_board()  # Assume a method exists to evaluate the board state
        except:
            score = -float('inf')  # If move is invalid, assign very low score
        
        if score > best_score:
            best_score = score
            best_move = move

    return best_move

def train_checkers_model(Opponent="itself"):
    winrates, avg_losses, avg_rewards = [], [], []
    learning_rate = 0.5
    discount_factor = 0.95
    win, lose, draw = 0, 0, 0

    for generations in tqdm(range(5)):
        data = []
        generation_losses, generation_rewards = [], []

        for g in range(10):
            temp_data = []
            game = checkers.Checkers()
            player = 1
            count = 0

            while True:
                count += 1
                if count > 1000:  # Draw condition
                    draw += 1
                    break

                if player == 1:
                    # Get the board state
                    board_state = game.get_board_state()

                    # Get the top 3 moves from the LLM
                    candidate_moves = get_top_moves_from_llm(board_state, num_moves=3)
                    
                    # Choose the best move from the 3
                    move = evaluate_moves_and_choose_best(game, candidate_moves)

                    if move:
                        game.PushMove(move)
                        temp_data.append((board_state, move))
                else:
                    # Opponent's move (random or heuristic-based)
                    legal_moves = game.get_legal_moves()
                    move = random.choice(legal_moves) if legal_moves else None
                    if move:
                        game.PushMove(move)

                end = game.EndGame()
                if end in [1, -1]:
                    reward = 10 if end == 1 else -10
                    win += (end == 1)
                    lose += (end == -1)
                    generation_rewards.append(reward)

                    temp_tensor = torch.tensor([t[0] for t in temp_data])
                    old_prediction = model(temp_tensor).detach().numpy()
                    optimal_futur_value = np.ones_like(old_prediction) * (1 if end == 1 else -1)
                    loss = learning_rate * (reward + discount_factor * optimal_futur_value - old_prediction)
                    generation_losses.append(loss.mean().item())

                    temp_labels = old_prediction + loss
                    data.extend([t[0] for t in temp_data])
                    labels = np.vstack((np.zeros(1), temp_labels))
                    break

                player = -player

        data_tensor = torch.tensor(data)
        labels_tensor = torch.tensor(labels[1:])
        model.train()  # Enable training mode
        # Example optimizer usage (e.g., Adam)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(model(data_tensor), labels_tensor)
        loss.backward()
        optimizer.step()

        winrate = int(win / (win + draw + lose + 1e-5) * 100)
        winrates.append(winrate)
        avg_losses.append(np.mean(generation_losses))
        avg_rewards.append(np.mean(generation_rewards))

        model_dir = "models"
        os.makedirs(model_dir, exist_ok=True)
        model.save_pretrained(os.path.join(model_dir, f"{Opponent}.gpt"))

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.plot(range(len(winrates)), winrates, marker='o', linestyle='-', label='Win Rate')
    plt.title('Win Rates')
    plt.xlabel('Generations')
    plt.ylabel('Wins [%]')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(range(len(avg_losses)), avg_losses, marker='o', linestyle='-', color='orange', label='Avg Loss')
    plt.title('Average Loss per Generation')
    plt.xlabel('Generations')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(range(len(avg_rewards)), avg_rewards, marker='o', linestyle='-', color='green', label='Avg Reward')
    plt.title('Average Reward per Generation')
    plt.xlabel('Generations')
    plt.ylabel('Reward')
    plt.legend()

    plt.tight_layout()
    plt.show()

if "__main__" == __name__:
    train_checkers_model()