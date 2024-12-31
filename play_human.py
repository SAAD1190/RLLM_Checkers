import pygame
import numpy as np
import tensorflow as tf
from keras.models import load_model
import logging
import checkers

# Initialize Pygame and configure logging
pygame.init()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Game constants
WIDTH, HEIGHT = 600, 600
ROWS, COLS = 8, 8
SQUARE_SIZE = WIDTH // COLS

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

# Initialize display
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Checkers Game")

# Draw the checkers board
def draw_board(win):
    win.fill(WHITE)
    for row in range(ROWS):
        for col in range(row % 2, COLS, 2):
            pygame.draw.rect(win, BLACK, (col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))

# Draw the pieces on the board
def draw_pieces(win, game, selected_piece=None):
    for row in range(ROWS):
        for col in range(COLS):
            piece = game.board[row][col]
            if piece == 1:  # White piece
                # Add a black outline for visibility
                pygame.draw.circle(win, BLACK, (col * SQUARE_SIZE + SQUARE_SIZE // 2, row * SQUARE_SIZE + SQUARE_SIZE // 2), SQUARE_SIZE // 3 + 2)
                pygame.draw.circle(win, WHITE, (col * SQUARE_SIZE + SQUARE_SIZE // 2, row * SQUARE_SIZE + SQUARE_SIZE // 2), SQUARE_SIZE // 3)
            elif piece == -1:  # Red piece
                pygame.draw.circle(win, RED, (col * SQUARE_SIZE + SQUARE_SIZE // 2, row * SQUARE_SIZE + SQUARE_SIZE // 2), SQUARE_SIZE // 3)

    # Highlight the selected piece
    if selected_piece:
        pygame.draw.circle(win, BLUE, (selected_piece[1] * SQUARE_SIZE + SQUARE_SIZE // 2, selected_piece[0] * SQUARE_SIZE + SQUARE_SIZE // 2), SQUARE_SIZE // 3 + 4, 3)

# Load the AI model
def load_checkers_model(model_path):
    try:
        model = load_model(model_path)
        logging.info("Model loaded successfully!")
        return model
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        return None

# Get AI move using the model
def get_ai_move(game, player, model):
    leafs = game.minmax(player, RL=True)  # Generate possible moves
    if len(leafs) == 0:
        return None

    Leaf = tf.zeros((len(leafs), 5))
    for l in range(len(leafs)):
        tensor = leafs[l][2]
        Leaf = tf.tensor_scatter_nd_update(Leaf, [[l]], [tensor[:5]])

    scores = model.predict_on_batch(Leaf)
    i = np.argmax(scores)
    return leafs[i][0]

# Handle human player moves
def handle_player_move(game, player):
    moves = game.GetValidMoves(player, Filter=True)
    if not moves:
        return None

    selected_piece = None
    selected_move = None

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None

            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                row, col = pos[1] // SQUARE_SIZE, pos[0] // SQUARE_SIZE

                if selected_piece is None:
                    # Select a piece
                    for move in moves:
                        if move[0] == (row, col):
                            selected_piece = (row, col)
                            break
                else:
                    # Select a destination
                    for move in moves:
                        if move[0] == selected_piece and move[1] == (row, col):
                            selected_move = move
                            return selected_move

# Main game loop
def main(model_path):
    clock = pygame.time.Clock()
    game = checkers.Checkers()
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