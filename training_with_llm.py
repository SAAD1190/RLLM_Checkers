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
llm_model = GPTNeoForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).cuda()  # Enable GPU & FP16 for memory efficiency
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Keras model for board evaluation
def create_keras_model():
    model = Sequential()
    model.add(Dense(32, activation='relu', input_dim=5))
    model.add(Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.1)))
    model.add(Dense(1, activation='linear', kernel_regularizer=regularizers.l2(0.1)))
    model.compile(optimizer='nadam', loss='mean_squared_error', metrics=["mae"])
    return model

def board_to_string(board):
    """
    Convert the checkers board into a formatted string.
    """
    board_str = "\n".join([" ".join([str(int(cell)) for cell in row]) for row in board])
    return board_str

def get_top_moves_from_llm(board, player, num_moves=3):
    """
    Get the top N moves from LLM based on the current board state.
    """
    player_name = "white" if player == 1 else "black"
    
    # Convert board state to string
    formatted_board = board_to_string(board)
    
    # Add example moves to the prompt
    example_moves = """
    Examples of valid moves:
    - (2, 3) -> (4, 5)
    - (6, 7) -> (5, 6)
    - (1, 2) -> (3, 4)
    """
    
    # Create the prompt
    prompt = f"""
    Current board state:
    {formatted_board}
    Player: {player_name}
    Suggest the top {num_moves} best moves for {player_name} in the format (x1, y1) -> (x2, y2).
    {example_moves}
    """
    
    print(f"LLM Prompt: {prompt[:300]}...")  # For debugging
    
    try:
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.cuda() for k, v in inputs.items()}  # Ensure inputs are on GPU
        outputs = llm_model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.7,
            top_k=50,
            pad_token_id=tokenizer.eos_token_id,
        )
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"LLM Output: {prediction}")
    except Exception as e:
        print(f"Error in LLM generation: {e}")
        return []  # Return empty moves if generation fails

    # Extract moves from the LLM output
    lines = prediction.split("\n")
    parsed_moves = []
    for line in lines:
        if "->" in line:
            move = parse_llm_move(line)
            if move:
                parsed_moves.append(move)
        if len(parsed_moves) >= num_moves:
            break

    return parsed_moves

def parse_llm_move(move_str):
    """
    Parse the LLM output string into a list of tuples (move format for checkers).
    Example input: '(2, 3) -> (4, 5)'
    Example output: [(2, 3), (4, 5)]
    """
    try:
        moves = move_str.replace("->", ",").replace("(", "").replace(")", "").split(",")
        coords = [(int(moves[i].strip()), int(moves[i + 1].strip())) for i in range(0, len(moves), 2)]
        return coords
    except Exception as e:
        print(f"Failed to parse LLM move: {move_str}, error: {e}")
        return None


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
            player = 1  # Start with player 1 (white)
            count = 0

            while True:
                count += 1
                if count > 1000:
                    draw += 1
                    break

                # Get the board for the current player
                board = game.board
                print(f"Board for player {player}: \n{board}")

                # Get the top 3 moves from the LLM
                candidate_moves = get_top_moves_from_llm(board, player, num_moves=3)

                if len(candidate_moves) == 0:
                    lose += 1
                    break

                # Get model scores for the 3 moves
                scores = []
                for move in candidate_moves:
                    compressed_move = game.CompressBoard(player, game.board)
                    tensor_move = tf.constant(compressed_move.flatten(), dtype=tf.float32)
                    score = model.predict_on_batch(tf.reshape(tensor_move, (1, 5)))
                    scores.append(score)

                # Choose the best move based on scores
                best_index = np.argmax(scores)
                best_move = candidate_moves[best_index]

                # Apply the best move
                game.PushMove(best_move)
                temp_data.append(compressed_move)

                # Check if the game has ended
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

                player = -player  # Switch player after each turn

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