import pygame
import numpy as np
import tensorflow as tf
from keras.models import load_model
import logging
from checkers import Checkers  # Importing the provided environment

# Initialize Pygame and configure logging
pygame.init()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Game constants
WIDTH, HEIGHT = 800, 800  # Adjusted to match a 10x10 board
ROWS, COLS = 10, 10
SQUARE_SIZE = WIDTH // COLS

# Colors
LIGHT_BROWN = (238, 214, 175)
DARK_BROWN = (139, 69, 19)
WHITE = (255, 255, 255)
RED = (255, 0, 0)

# Initialize display
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Checkers Game")

# Draw the checkers board
def draw_board(win):
    for row in range(ROWS):
        for col in range(COLS):
            color = LIGHT_BROWN if (row + col) % 2 == 0 else DARK_BROWN
            pygame.draw.rect(win, color, (col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))

# Draw the pieces on the board
def draw_pieces(win, game):
    for row in range(ROWS):
        for col in range(COLS):
            piece = game.board[row][col]
            if piece == 1:  # White pawn
                pygame.draw.circle(win, WHITE, (col * SQUARE_SIZE + SQUARE_SIZE // 2, row * SQUARE_SIZE + SQUARE_SIZE // 2), SQUARE_SIZE // 3)
            elif piece == 2:  # White queen
                pygame.draw.circle(win, WHITE, (col * SQUARE_SIZE + SQUARE_SIZE // 2, row * SQUARE_SIZE + SQUARE_SIZE // 2), SQUARE_SIZE // 2, 2)
            elif piece == -1:  # Red pawn
                pygame.draw.circle(win, RED, (col * SQUARE_SIZE + SQUARE_SIZE // 2, row * SQUARE_SIZE + SQUARE_SIZE // 2), SQUARE_SIZE // 3)
            elif piece == -2:  # Red queen
                pygame.draw.circle(win, RED, (col * SQUARE_SIZE + SQUARE_SIZE // 2, row * SQUARE_SIZE + SQUARE_SIZE // 2), SQUARE_SIZE // 2, 2)

# Load the AI model
def load_checkers_model(model_path):
    try:
        model = load_model(model_path)
        logging.info("Model loaded successfully!")
        return model
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        return None

# Get AI move using the trained model
def get_ai_move(game, player, model):
    leafs = game.minmax(player, RL=True)
    if len(leafs) == 0:
        logging.warning("No valid moves available for the AI.")
        return None

    valid_leafs = []
    for leaf in leafs:
        if isinstance(leaf[2], np.ndarray) and len(leaf[2]) >= 5:  # Ensure at least 5 features for the model input
            features = leaf[2][:5]  # Compress board to match model input size
            valid_leafs.append((leaf[0], features))
        else:
            logging.warning(f"Skipping invalid board state with shape {leaf[2].shape if isinstance(leaf[2], np.ndarray) else 'unknown'}.")

    if not valid_leafs:
        logging.error("No valid board states for AI to evaluate.")
        return None

    Leaf = np.array([v[1] for v in valid_leafs])  # Collect compressed features for all valid leafs
    scores = model.predict_on_batch(Leaf)  # Use the model to predict scores
    i = np.argmax(scores)  # Choose the move with the highest score
    return valid_leafs[i][0]  # Return the corresponding move




# Handle human player moves
def handle_player_move(game, player):
    moves = game.GetValidMoves(player, Filter=True)
    if not moves:
        logging.warning("No valid moves available for the player.")
        return None

    selected_piece = None
    selected_move = None

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                logging.info("Game quit by player.")
                return None

            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                row, col = pos[1] // SQUARE_SIZE, pos[0] // SQUARE_SIZE

                if selected_piece is None:
                    for move in moves:
                        if move[0] == (row, col):
                            selected_piece = (row, col)
                            break
                else:
                    for move in moves:
                        if move[0] == selected_piece and move[1] == (row, col):
                            selected_move = move
                            return selected_move

# Main game loop
def main(model_path):
    clock = pygame.time.Clock()
    game = Checkers()
    model = load_checkers_model(model_path)

    if model is None:
        logging.error("Model could not be loaded. Exiting game.")
        return

    running = True
    player = 1  # Human starts as white

    while running:
        clock.tick(30)
        draw_board(WIN)
        draw_pieces(WIN, game)
        pygame.display.update()

        if player == 1:  # Human turn
            move = handle_player_move(game, player)
            if move is not None:
                game.PushMove(move)
                player = -1
        else:  # AI turn
            ai_move = get_ai_move(game, player, model)
            if ai_move:
                game.PushMove(ai_move)
            else:
                logging.error("AI could not make a move. Ending game.")
                break
            player = 1

        # Check for game end
        end = game.EndGame()
        if end != 0:
            logging.info(f"Game Over! Player {end} wins.")
            running = False

    pygame.quit()

# Run the game
if __name__ == "__main__":
    model_path = "models/itself.keras"
    main(model_path)