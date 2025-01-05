# State structure
"""
The state is represented by a 10x10 board (self.board), which is implemented as a NumPy array.
0: Empty cell // 1: White pawn // 2: White queen // -1: Black pawn // -2: Black queen //
"""
# Action structure
"""
Pawn moves:
((start_x, start_y), (end_x, end_y))
((4, 5), (3, 6))  # A move from (4, 5) to (3, 6)

Queen moves:
[(start_x, start_y), (intermediate_x, intermediate_y), ..., (end_x, end_y)]
[(5, 2), (3, 4), (1, 6)]  # A queen moves from (5, 2) to (3, 4) and then to (1, 6)
"""

# Reward structure
"""
Positive reward (+100): When a player wins (all opponent pieces are removed).
Negative reward (-100): When a player loses.
Intermediate rewards:
Capturing pieces increases the player’s score.
Losing pieces decreases the player’s score.
"""

# Features structure
"""
Used to evaluate board states and train a reinforcement learning (RL) agent.
The GetFeatures() function extracts features from the current board to inform decisions:

Feature vector (features[]):
Whether the player has won.
Number of player’s pawns and queens.
Number of opponent’s pawns and queens.
Pieces positioned in strategic areas (e.g., far rows, middle rows).
"""
############################################################################################
# Train the agent using the LLM to limit action space and optimize training process


# Step 1: Initilize the LLM
# Step 2: Get board state and features at each time step
# Step 3: Prompt the LLM to get the 3 best action with respect to action structure, output is a list of 3 actions (action space of three actions)
# Step 4: Input the 3 actions to the agent training process. The agent will select the best action from the 3 actions. training is done using the Q-learning algorithm

############################################################################################



import checkers
import numpy as np
import tensorflow as tf
from keras import Sequential, regularizers
from keras.layers import Dense
from tqdm import tqdm
import random
import os
import matplotlib.pyplot as plt
import torch
import random
from transformers import GPTNeoForCausalLM, AutoTokenizer


# Load the GPT-Neo model and tokenizer
model_name = "EleutherAI/gpt-neo-1.3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = GPTNeoForCausalLM.from_pretrained(model_name)


def format_board_state(board_state):
    """
    Convert the board state into a human-readable string for the LLM prompt.
    """
    formatted_board = "\n".join([" ".join([f"{int(cell):2}" for cell in row]) for row in board_state])
    return formatted_board


def get_top_3_actions(board_state, player):
    """
    Get the top 3 recommended actions using the GPT-Neo model.
    """
    formatted_board = format_board_state(board_state)
    prompt = f"""
    You are a checkers expert. Given the board state below, suggest only the top 3 recommended moves for player {player}.

    Board state (10x10):
    {formatted_board}

    Please return only the moves list in this format:
    [
    ((4, 5), (3, 6)),
    ((6, 1), (7, 0)),
    [(5, 2), (3, 4), (1, 6)]
    ]
"""


    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
        model.resize_token_embeddings(len(tokenizer))

    # Encode the prompt with attention mask
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512, padding="max_length")

    # Generate output with max_new_tokens and an attention mask
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=100,
            temperature=0.7,
            num_return_sequences=1,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )

    # Decode the output text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract actions from the output text
    start_idx = generated_text.find("[")
    end_idx = generated_text.find("]")
    if start_idx == -1 or end_idx == -1:
        print("Warning: Could not extract valid actions from output. Returning random default actions.")
        return [random.choice([((4, 5), (3, 6)), ((6, 1), (7, 0)), [(5, 2), (3, 4), (1, 6)]])]

    action_str = generated_text[start_idx:end_idx + 1]
    try:
        actions = eval(action_str)
    except Exception as e:
        print(f"Error parsing actions: {e}")
        actions = [random.choice([((4, 5), (3, 6)), ((6, 1), (7, 0)), [(5, 2), (3, 4), (1, 6)]])]
    
    return actions[:3]  # Return only the top 3 actions




def build_model():
    """
    Build a simple Q-learning neural network model.
    """
    model = Sequential()
    model.add(Dense(32, activation='relu', input_dim=5))  # 5 features
    model.add(Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(1, activation='linear'))  # Q-value output
    model.compile(optimizer='adam', loss='mse')
    return model


def train_checkers_model_with_llm(Opponent="itself"):
    model = build_model()

    winrates = []
    avg_losses = []
    avg_rewards = []
    learning_rate = 0.5
    discount_factor = 0.95
    exploration = 0.9  # Decreases over time for exploitation
    win, lose, draw = 0, 0, 0

    for generation in tqdm(range(5)):
        data = []
        labels = []
        total_reward = 0
        total_loss = 0
        num_moves = 0

        for _ in range(10):  # Number of games per generation
            game = checkers.Checkers()
            player = 1
            count = 0
            while True:
                count += 1
                if count > 1000:  # Draw condition
                    draw += 1
                    break

                board_state = game.board
                # Step 1: Get LLM-suggested actions
                top_3_actions = get_top_3_actions(board_state.tolist(), player)

                # Step 2: Evaluate actions using the model
                q_values = []
                for action in top_3_actions:
                    features = game.GetFeatures(player)
                    input_features = tf.constant(features[:5], shape=(1, 5), dtype=tf.float32)
                    q_value = model.predict(input_features, verbose=0)
                    q_values.append(q_value[0][0])

                # Step 3: Choose best action or explore
                if random.random() < exploration:
                    chosen_action = random.choice(top_3_actions)  # Exploration
                else:
                    chosen_action = top_3_actions[np.argmax(q_values)]  # Exploitation

                # Step 4: Apply the chosen action
                game.PushMove(chosen_action)

                # Calculate reward
                end_game = game.EndGame()
                if end_game == 1:
                    reward = 100  # Player 1 wins
                    win += 1
                elif end_game == -1:
                    reward = -100  # Player 1 loses
                    lose += 1
                else:
                    reward = 0

                # Record data for training
                features = game.GetFeatures(player)
                q_update = reward + discount_factor * np.max(q_values)
                data.append(features[:5])
                labels.append(q_update)

                total_reward += reward
                num_moves += 1

                if end_game != 0:
                    break

                # Switch to the other player
                player *= -1

        # Train the model
        data = np.array(data)
        labels = np.array(labels)
        history = model.fit(data, labels, epochs=16, batch_size=32, verbose=0)
        avg_loss = np.mean(history.history['loss'])
        avg_losses.append(avg_loss)
        avg_rewards.append(total_reward / num_moves if num_moves else 0)

        # Update exploration rate
        exploration *= 0.95

        winrate = int((win) / (win + draw + lose) * 100) if (win + draw + lose) else 0
        winrates.append(winrate)

        # Save model
        model_dir = "models"
        os.makedirs(model_dir, exist_ok=True)
        keras_path = os.path.join(model_dir, f"{Opponent}_llm.keras")
        model.save(keras_path)

    # Plot Win Rate
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(winrates)), winrates, marker='o', label="Win Rate")
    plt.title('Win Rate per Generation')
    plt.xlabel('Generations')
    plt.ylabel('Win Rate (%)')
    plt.legend()

    # Plot Average Loss per Generation
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(avg_losses)), avg_losses, marker='o', color='red', label="Average Loss")
    plt.title('Average Loss per Generation')
    plt.xlabel('Generations')
    plt.ylabel('Loss')
    plt.legend()

    # Plot Average Reward per Generation
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(avg_rewards)), avg_rewards, marker='o', color='green', label="Average Reward")
    plt.title('Average Reward per Generation')
    plt.xlabel('Generations')
    plt.ylabel('Reward')
    plt.legend()

    plt.show()


if __name__ == "__main__":
    train_checkers_model_with_llm()