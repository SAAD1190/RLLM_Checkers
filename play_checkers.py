import torch
import numpy as np
from checkers import Checkers
import checkers
from training import CheckersModel

def play_checkers(model_path, num_games=10, verbose=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the trained model
    model = CheckersModel().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    results = {"win": 0, "lose": 0, "draw": 0}

    for game_num in range(num_games):
        game = Checkers()
        player = 1
        moves = 0

        if verbose:
            print(f"\nGame {game_num + 1}")

        while True:
            moves += 1
            leafs = game.GetValidMoves(player)
            if not leafs:
                # If no valid moves, the other player wins
                results["win" if player == -1 else "lose"] += 1
                if verbose:
                    print(f"Player {player} loses due to no valid moves.")
                break

            # Filter and evaluate valid moves
            valid_moves = [
                leaf[2][:5]
                for leaf in leafs
                if len(leaf) > 2 and isinstance(leaf[2], (list, np.ndarray)) and len(leaf[2]) >= 5
            ]

            if not valid_moves:
                results["win" if player == -1 else "lose"] += 1
                if verbose:
                    print(f"Player {player} loses due to invalid move data.")
                break

            Leaf = torch.tensor(valid_moves, dtype=torch.float32, device=device)
            scores = model(Leaf).detach().cpu().numpy()
            best_move_idx = np.argmax(scores)
            best_move = leafs[best_move_idx]

            game.PushMove(best_move)

            if verbose:
                print(f"Player {player} moved: {best_move}")
                game.Show()

            # Check for game end
            end = game.EndGame()
            if end == 1:
                results["win"] += 1
                if verbose:
                    print("Player 1 wins!")
                break
            elif end == -1:
                results["lose"] += 1
                if verbose:
                    print("Player -1 wins!")
                break
            elif moves > 1000:
                results["draw"] += 1
                if verbose:
                    print("Game ended in a draw!")
                break

            # Switch players
            player = -player

    print("\nResults:")
    print(f"Wins: {results['win']}")
    print(f"Loses: {results['lose']}")
    print(f"Draws: {results['draw']}")

if __name__ == "__main__":
    model_path = "models\self_play_model.pth"
    play_checkers(model_path, num_games=5, verbose=True)