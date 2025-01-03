import checkers
import matplotlib.pyplot as plt
from keras import Sequential, regularizers
from keras.layers import Dense
import tensorflow as tf
import numpy as np
import random
from tqdm import tqdm
import os
from transformers import GPTNeoForCausalLM, GPT2Tokenizer

# Initialize LLM (GPT-Neo 2.7B)
model_name = "EleutherAI/gpt-neo-2.7B"
llm_model = GPTNeoForCausalLM.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Keras model for board evaluation
def create_keras_model():
    model = Sequential()
    model.add(Dense(32, activation='relu', input_dim=5))
    model.add(Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.1)))
    model.add(Dense(1, activation='linear', kernel_regularizer=regularizers.l2(0.1)))
    model.compile(optimizer='nadam', loss='binary_crossentropy', metrics=["acc"])
    return model

def get_top_moves_from_llm(board_state, num_moves=3):
    """
    Get the top N moves from the LLM based on the current board state.
    """
    prompt = f"Game board: {board_state}\nSuggest the top {num_moves} best moves:"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = llm_model.generate(**inputs, max_length=150, num_return_sequences=1)
    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
    lines = prediction.split("\n")
    moves = [line.split(":")[-1].strip() for line in lines if ":" in line][-num_moves:]
    return moves

def train_checkers_model(Opponent="itself"):
    model = create_keras_model()
    winrates, avg_losses, avg_rewards = [], [], []
    win, lose, draw = 0, 0, 0

    for generations in tqdm(range(5)):
        data, labels = [], []
        generation_losses, generation_rewards = [], []

        for g in range(10):
            temp_data = []
            game = checkers.Checkers()
            player = 1
            count = 0

            while True:
                count += 1
                if count > 1000:
                    draw += 1
                    break

                if player == 1:
                    # Get the board state and ask LLM for top 3 moves
                    board_state = game.get_board_state()
                    candidate_moves = get_top_moves_from_llm(board_state, num_moves=3)

                    # Evaluate the 3 moves and choose the best
                    leafs = []
                    for move in candidate_moves:
                        # Get board tensor representation
                        board_tensor = tf.constant(np.array(move[:5]), dtype=tf.float32)
                        leafs.append((move, board_tensor))

                    if len(leafs) == 0:
                        lose += 1
                        break

                    # Get model scores for the 3 moves
                    Leaf = tf.stack([leaf[1] for leaf in leafs])
                    scores = model.predict_on_batch(Leaf)
                    best_index = np.argmax(scores)
                    best_move = leafs[best_index][0]
                    game.PushMove(best_move)
                    temp_data.append(leafs[best_index][1])

                else:
                    legal_moves = game.GetValidMoves(player)
                    move = random.choice(legal_moves) if legal_moves else None
                    if move:
                        game.PushMove(move)
                    else:
                        win += 1
                        break

                end = game.EndGame()
                if end in [1, -1]:
                    reward = 10 if end == 1 else -10
                    win += (end == 1)
                    lose += (end == -1)
                    generation_rewards.append(reward)

                    temp_tensor = tf.constant(temp_data[1:], dtype=tf.float32)
                    old_prediction = model.predict_on_batch(temp_tensor)
                    optimal_future_value = np.ones_like(old_prediction) * (1 if end == 1 else -1)
                    temp_labels = old_prediction + 0.5 * (reward + 0.95 * optimal_future_value - old_prediction)
                    data.extend(temp_data[1:])
                    labels.extend(temp_labels)
                    break

                player = -player

        if data:
            data_tensor = tf.constant(np.array(data), dtype=tf.float32)
            labels_tensor = tf.constant(np.array(labels), dtype=tf.float32)

            # Train Keras model
            history = model.fit(data_tensor, labels_tensor, epochs=16, batch_size=256, verbose=0)
            avg_losses.append(np.mean(history.history['loss']))
            avg_rewards.append(np.mean(generation_rewards))

        winrate = int(win / (win + draw + lose + 1e-5) * 100)
        winrates.append(winrate)

        # Save model
        model_dir = "models"
        os.makedirs(model_dir, exist_ok=True)
        keras_path = os.path.join(model_dir, f"{Opponent}.keras")
        model.save(keras_path)

    # Plot results
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(winrates)), winrates, marker='o', label='Win Rate')
    plt.title('Win Rate per Generation')
    plt.xlabel('Generations')
    plt.ylabel('Win Rate [%]')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(range(len(avg_losses)), avg_losses, marker='o', color='red', label='Average Loss')
    plt.title('Average Loss per Generation')
    plt.xlabel('Generations')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(range(len(avg_rewards)), avg_rewards, marker='o', color='green', label='Average Reward')
    plt.title('Average Reward per Generation')
    plt.xlabel('Generations')
    plt.ylabel('Reward')
    plt.legend()
    plt.show()

if "__main__" == __name__:
    train_checkers_model()