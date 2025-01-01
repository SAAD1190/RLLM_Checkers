import checkers
import matplotlib.pyplot as plt
from keras.models import load_model
import tensorflow as tfw
import numpy as np
import random
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Function to load the model
def load_checkers_model(model_path):
    if os.path.exists(model_path):
        try:
            model = load_model(model_path)
            logging.info("Model loaded successfully!")
            return model
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            return None
    else:
        logging.error(f"Model file not found at: {model_path}")
        return None

# Function to get a move using the model
def get_move(game, player, model):
    leafs = game.minmax(player, RL=True)  # Generate possible moves
    if len(leafs) == 0:
        return None

    Leaf = tfw.zeros((len(leafs), 5))
    for l in range(len(leafs)):
        tensor = leafs[l][2]
        Leaf = tfw.tensor_scatter_nd_update(Leaf, [[l]], [tensor[:5]])

    scores = model.predict_on_batch(Leaf)
    i = np.argmax(scores)
    return leafs[i][0]

# Function to play a single game
def play_checkers_single_game(model):
    game = checkers.Checkers()
    player = -1
    NumberOfMoves = 0

    while True:
        player = -player

        moves = game.GetValidMoves(player, Filter=True)
        if len(moves) == 0:
            return -player  # Winner

        move = get_move(game, player, model)
        if move is None:
            return -player  # Winner

        game.PushMove(move)

        end = game.EndGame()
        if end == 1 or end == -1:
            return end

        NumberOfMoves += 2
        if NumberOfMoves > 300:
            return 0  # Draw

# Function to play multiple games
def play_multiple_games(model_path, num_games=10):
    model = load_checkers_model(model_path)
    if model is None:
        logging.error("Model could not be loaded. Exiting.")
        return

    results = {1: 0, -1: 0, 0: 0}  # Track wins (1: Player 1, -1: Player 2, 0: Draw)

    for game_num in range(1, num_games + 1):
        winner = play_checkers_single_game(model)
        results[winner] += 1
        logging.info(f"Game {game_num}: Winner is Player {winner if winner != 0 else 'Draw'}")

    # Print summary
    print("\nSummary of Results:")
    print(f"Player 1 Wins: {results[1]}")
    print(f"Player 2 Wins: {results[-1]}")
    print(f"Draws: {results[0]}")

# Main execution
if __name__ == "__main__":
    model_path = "models/itself.keras"
    play_multiple_games(model_path, num_games=10)