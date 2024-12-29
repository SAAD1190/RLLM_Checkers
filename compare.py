import checkers
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import random

# Load PyTorch models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

Random = torch.load("models/random_model.pth", map_location=device)
Minmax = torch.load("models/minmax_model.pth", map_location=device)
Itself = torch.load("models/itself_model.pth", map_location=device)

Random.eval()
Minmax.eval()
Itself.eval()

def playRandomMinMax(Bot1="random", Bot2="random", verbose=False):
    end2 = 0
    game = checkers.Checkers()
    if verbose:
        game.Show()
    player = -1
    NumberOfMoves = 0
    while True:
        player = -player
        if player == 1:
            a = game.CopyBoard()
            moves = game.GetValidMoves(player, Filter=True)
            if len(moves) == 0:
                if verbose:
                    print(f"player {player} lose")
                return -player

            if Bot1 == "random":
                move = random.choice(moves)
            elif Bot1 == "minmax":
                move = random.choice(game.minmax(1))
            elif Bot1 in ["Trandom", "Tminmax", "Titself"]:
                if Bot1 == "Trandom":
                    model = Random
                elif Bot1 == "Tminmax":
                    model = Minmax
                elif Bot1 == "Titself":
                    model = Itself
                else:
                    raise ValueError(f"Invalid bot type {Bot1}")

                leafs = game.minmax(player, RL=True)  # move | score | features
                if len(leafs) == 0:
                    end2 = -player
                    continue

                Leaf = torch.tensor([leaf[2][:5] for leaf in leafs], dtype=torch.float32, device=device)
                scores = model(Leaf).detach().cpu().numpy()
                i = np.argmax(scores)
                move = leafs[i][0]
            else:
                raise ValueError(f"Unknown bot type {Bot1}")

            game.PushMove(move)
            if game.CompareBoard(a, game.board):
                print(f"move did not play {move}")
                print(game.board)
                break

            if verbose:
                print(move)
                game.GetScore(verbose=True)
                game.GetFeatures(player, verbose=True)
                game.Show()

            end = game.EndGame()
            if end == 1 or end2 == 1:
                if verbose:
                    print("white win")
                return 1
            elif end == -1 or end2 == -1:
                if verbose:
                    print("black win")
                return -1

        elif player == -1:
            a = game.CopyBoard()
            moves = game.GetValidMoves(player, Filter=True)
            if len(moves) == 0:
                if verbose:
                    print(f"player {player} lose")
                return -player

            if Bot2 == "random":
                move = random.choice(moves)
            elif Bot2 == "minmax":
                move = random.choice(game.minmax(-1))
            elif Bot2 in ["Trandom", "Tminmax", "Titself"]:
                if Bot2 == "Trandom":
                    model = Random
                elif Bot2 == "Tminmax":
                    model = Minmax
                elif Bot2 == "Titself":
                    model = Itself
                else:
                    raise ValueError(f"Invalid bot type {Bot2}")

                leafs = game.minmax(player, RL=True)  # move | score | features
                if len(leafs) == 0:
                    end2 = -player
                    continue

                Leaf = torch.tensor([leaf[2][:5] for leaf in leafs], dtype=torch.float32, device=device)
                scores = model(Leaf).detach().cpu().numpy()
                i = np.argmax(scores)
                move = leafs[i][0]
            else:
                raise ValueError(f"Unknown bot type {Bot2}")

            game.PushMove(move)
            if game.CompareBoard(a, game.board):
                print(f"move did not play {move}")
                print(game.board)
                break

            if verbose:
                print(move)
                game.GetScore(verbose=True)
                game.GetFeatures(player, verbose=True)
                game.Show()

            end = game.EndGame()
            if end == 1 or end2 == 1:
                if verbose:
                    print("white win")
                return 1
            elif end == -1 or end2 == -1:
                if verbose:
                    print("black win")
                return -1

        NumberOfMoves += 2
        if NumberOfMoves > 300:
            return 0


def CompareBots(player1, player2, NumberPlays):
    black = 0
    white = 0
    draw = 0
    for i in range(NumberPlays):
        if i % (NumberPlays // 10) == ((NumberPlays // 10) - 1):
            print(f"[{i+1}/{NumberPlays}]%")
        ret = playRandomMinMax(Bot1=player1, Bot2=player2, verbose=False)
        if ret == 1:
            white += 1
        elif ret == -1:
            black += 1
        elif ret == 0:
            draw += 1
        else:
            raise ValueError(f"Unexpected return value {ret}")

    black_percent = (black / NumberPlays) * 100
    white_percent = (white / NumberPlays) * 100
    draw_percent = (draw / NumberPlays) * 100

    labels = ["Black", "White", "Draw"]
    sizes = [black_percent, white_percent, draw_percent]
    colors = ["lightcoral", "lightskyblue", "green"]

    plt.figure(figsize=(8, 6))
    plt.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=140)
    plt.axis("equal")  # Ensures the pie chart is a circle
    plt.title(f"{player1} (White) vs {player2} (Black)")
    plt.show()


if __name__ == "__main__":
    player1 = "Trandom"
    player2 = "Tminmax"
    print(player1, player2)
    CompareBots(player1=player1, player2=player2, NumberPlays=100)
    print(player1, player2)