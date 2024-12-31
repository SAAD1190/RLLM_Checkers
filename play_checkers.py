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
def play_checkers(model_path, verbose=False):
    model = load_checkers_model(model_path)
    if model is None:
        logging.error("Model could not be loaded. Exiting game.")
        return

    game = checkers.Checkers()
    if verbose:
        game.Show()

    player = -1
    NumberOfMoves = 0

    while True:
        player = -player

        moves = game.GetValidMoves(player, Filter=True)
        if len(moves) == 0:
            winner = -player
            logging.info(f"Player {winner} wins by default (no moves left for Player {player}).")
            break

        move = get_move(game, player, model)
        if move is None:
            winner = -player
            logging.info(f"Player {winner} wins (no valid moves for Player {player}).")
            break

        game.PushMove(move)

        if verbose:
            logging.info(f"Player {player} move: {move}")
            game.Show()

        end = game.EndGame()
        if end == 1 or end == -1:
            logging.info(f"Game Over! Player {end} wins.")
            break

        NumberOfMoves += 2
        if NumberOfMoves > 300:
            logging.info("Game Over! Draw (move limit reached).")
            break

# Main execution
if __name__ == "__main__":
    model_path = "models/itself500.keras"
    play_checkers(model_path, verbose=True)