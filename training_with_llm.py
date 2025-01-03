import checkers
import matplotlib.pyplot as plt
import torch
from transformers import GPTNeoXForCausalLM, GPT2Tokenizer
import numpy as np
import random
from tqdm import tqdm
import os

#Initialize GPT-NeoX Model
model_name = "EleutherAI/gpt-neo-2.7B"
model = GPTNeoXForCausalLM.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

def concatenate(array1, array2):
    for i in range(len(array2)):
        array1.append(array2[i])
    return array1  

def get_move_from_llm(board_state):
    """
    This function prepares the board state as text input for the GPT-NeoX model
    and retrieves the predicted move from the model.
    """
    prompt = f"Game board: {board_state}\nPlayer 1's move:"
    
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Get model prediction
    outputs = model.generate(**inputs, max_length=50, num_return_sequences=1)
    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract the predicted move from the text (you would need to fine-tune this part)
    move = prediction.split('\n')[-1]
    return move

def train_checkers_model(Opponent="itself"):
    data = [] 
    labels = np.zeros(1)
    winrates = []
    avg_losses = []  # Stocker les moyennes des pertes
    avg_rewards = []  # Stocker les moyennes des récompenses
    learning_rate = 0.5
    discount_factor = 0.95
    exploration = 0.95
    win, lose, draw = 0, 0, 0

    for generations in tqdm(range(25)):
        data = []
        generation_losses = []  # Liste des pertes pour cette génération
        generation_rewards = []  # Liste des récompenses pour cette génération

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
                else:
                    if player == 1: 
                        # Get the board state
                        board_state = game.get_board_state()
                        # Get the predicted move from GPT-NeoX
                        move = get_move_from_llm(board_state)
                        game.PushMove(move)
                        temp_data.append(board_state)
                    elif player == -1:
                        # Handle opponent moves
                        ...

                end = game.EndGame()

                if end == 1 or end2 == 1:
                    win += 1
                    reward = 10
                    generation_rewards.append(reward)  # Enregistrer la récompense
                    temp_tensor = torch.constant(temp_data[1:])
                    old_prediction = model.predict_on_batch(temp_tensor)
                    optimal_futur_value = np.ones(old_prediction.shape)
                    loss = learning_rate * (reward + discount_factor * optimal_futur_value - old_prediction)
                    generation_losses.append(loss.mean().item())  # Enregistrer la perte moyenne
                    temp_labels = old_prediction + loss
                    data = concatenate(data, temp_data[1:])
                    labels = np.vstack((labels, temp_labels))
                    break

                elif end == -1 or end2 == -1:
                    lose += 1
                    reward = -10
                    generation_rewards.append(reward)  # Enregistrer la récompense
                    temp_tensor = torch.constant(temp_data[1:])
                    old_prediction = model.predict_on_batch(temp_tensor)
                    optimal_futur_value = -1 * np.ones(old_prediction.shape)
                    loss = learning_rate * (reward + discount_factor * optimal_futur_value - old_prediction)
                    generation_losses.append(loss.mean().item())  # Enregistrer la perte moyenne
                    temp_labels = old_prediction + loss
                    data = concatenate(data, temp_data[1:])
                    labels = np.vstack((labels, temp_labels))
                    break

                player = -player 
        
        data = torch.constant(data)
        model.fit(data[1:], labels[2:], epochs=16, batch_size=256, verbose=0)
        labels = np.zeros(1)
        winrate = int((win)/(win+draw+lose)*100)
        winrates.append(winrate)
        
        # Calculer les moyennes des pertes et des récompenses
        avg_losses.append(np.mean(generation_losses))
        avg_rewards.append(np.mean(generation_rewards))

        # Save model in both formats
        model_dir = "models"
        os.makedirs(model_dir, exist_ok=True)
        keras_path = os.path.join(model_dir, f"{Opponent}.keras")
        model.save(keras_path)

    indices = list(range(len(winrates)))
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(indices, winrates, marker='o', linestyle='-', label='Win Rate')
    plt.title('Win Rates')
    plt.xlabel('Generations')
    plt.ylabel('Wins [%]')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(indices, avg_losses, marker='o', linestyle='-', color='orange', label='Avg Loss')
    plt.title('Average Loss per Generation')
    plt.xlabel('Generations')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(indices, avg_rewards, marker='o', linestyle='-', color='green', label='Avg Reward')
    plt.title('Average Reward per Generation')
    plt.xlabel('Generations')
    plt.ylabel('Reward')
    plt.legend()

    plt.tight_layout()
    plt.show()

if "__main__" == __name__:
    train_checkers_model()
