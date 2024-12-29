import torch
import numpy as np
import random


class Checkers:
    def __init__(self) -> None:
        """Initialize the Checkers board and scores."""
        self.board = np.zeros((10, 10))
        for i in [0, 2]:
            for j in range(0, 9, 2):
                self.board[i + 1][j] = -1
                self.board[9 - i][j] = 1
            for j in range(1, 10, 2):
                self.board[i][j] = -1
                self.board[9 - i - 1][j] = 1
        self.score1 = 20
        self.score2 = 20

    def show_board(self):
        """Display the current state of the board."""
        print(self.board)

    def end_game(self):
        """Check if the game has ended."""
        player1, player2 = 0, 0
        for line in self.board:
            for case in line:
                if case > 0:
                    player1 += case
                elif case < 0:
                    player2 -= case
        if player1 == 0 and player2 == 0:
            return 2  # Draw
        elif player2 == 0:
            return 1  # Player 1 wins
        elif player1 == 0:
            return -1  # Player 2 wins
        return 0  # Game not over

    def get_valid_moves(self, player=1):
        """Get valid moves for the current player, including eating and promotion."""
        moves = []
        for i in range(10):
            for j in range(10):
                if self.board[i][j] == player:
                    moves.extend(self.get_moves_for_pawn(i, j, player))
                elif self.board[i][j] == 2 * player:  # King
                    moves.extend(self.get_moves_for_king(i, j, player))
        return moves

    def get_moves_for_pawn(self, x, y, player):
        """Generate valid moves for a pawn."""
        moves = []
        # Normal moves
        if self.is_on_board(x + player, y + 1) and self.is_empty(x + player, y + 1):
            moves.append(((x, y), (x + player, y + 1)))
        if self.is_on_board(x + player, y - 1) and self.is_empty(x + player, y - 1):
            moves.append(((x, y), (x + player, y - 1)))

        # Capture moves
        if (
            self.is_on_board(x + 2 * player, y + 2)
            and self.is_empty(x + 2 * player, y + 2)
            and self.is_opponent_piece(x + player, y + 1, player)
        ):
            moves.append(((x, y), (x + 2 * player, y + 2)))
        if (
            self.is_on_board(x + 2 * player, y - 2)
            and self.is_empty(x + 2 * player, y - 2)
            and self.is_opponent_piece(x + player, y - 1, player)
        ):
            moves.append(((x, y), (x + 2 * player, y - 2)))

        return moves

    def get_moves_for_king(self, x, y, player):
        """Generate valid moves for a king."""
        moves = []
        directions = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            while self.is_on_board(nx, ny) and self.is_empty(nx, ny):
                moves.append(((x, y), (nx, ny)))
                nx += dx
                ny += dy
        return moves

    def is_on_board(self, x, y):
        """Check if a position is on the board."""
        return 0 <= x < 10 and 0 <= y < 10

    def is_empty(self, x, y):
        """Check if a position is empty."""
        return self.board[x][y] == 0

    def is_opponent_piece(self, x, y, player):
        """Check if a position contains an opponent's piece."""
        return self.board[x][y] * player < 0

    def push_move(self, move):
        """Make the move on the board."""
        start, end = move
        self.board[end[0]][end[1]] = self.board[start[0]][start[1]]
        self.board[start[0]][start[1]] = 0


def play_self_with_model(model_path):
    """Play a game of Checkers against itself using the trained RL model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(model_path, map_location=device)
    model.eval()

    game = Checkers()
    game.show_board()

    player = 1
    while True:
        # Check for game end
        result = game.end_game()
        if result != 0:
            if result == 1:
                print("Player 1 wins!")
            elif result == -1:
                print("Player 2 wins!")
            else:
                print("Draw!")
            break

        # Get valid moves
        valid_moves = game.get_valid_moves(player)
        if not valid_moves:
            print(f"Player {player} has no moves! Game over.")
            break

        # Predict the best move using the RL model
        board_states = [game.compress_board(player).to(device) for move in valid_moves]
        scores = [model(state).item() for state in board_states]
        best_move = valid_moves[np.argmax(scores)]

        # Apply the move
        game.push_move(best_move)
        print(f"Player {player} moves: {best_move}")
        game.show_board()

        # Switch player
        player *= -1


if __name__ == "__main__":
    play_self_with_model("models/trained_model.pth")