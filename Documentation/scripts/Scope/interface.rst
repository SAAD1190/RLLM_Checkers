Interactive Interface
=====================

This section provides an overview of the interactive interface for playing the checkers game, which combines a graphical board display and AI-assisted gameplay.

Overview
--------

The interactive interface is built using **Pygame** for visualizing the board and game progress. It supports both human and AI players, allowing users to experience the game in real-time. The AI utilizes the trained reinforcement learning model for decision-making.

Features
--------

- **Graphical Board Visualization**:
  - The board is displayed with an 8x8 grid using light and dark brown squares.
  - Pieces are represented as white and red circles for the two players.

- **AI-Powered Gameplay**:
  - The AI leverages the trained model to evaluate moves and make decisions.
  - Supports dynamic action filtering and efficient decision-making.

- **Human Player Interaction**:
  - Users can select pieces and make moves through mouse clicks.
  - Real-time validation of moves ensures compliance with game rules.

Usage
-----

1. **Run the Game**:
   Execute the script with the path to the trained model:
   
   .. code-block:: bash
      
      python checkers_game.py

2. **Gameplay Instructions**:
   - Human player starts as white pieces.
   - Click on a piece to select it, then click on a valid destination to make a move.
   - The AI plays as red and makes decisions automatically.

3. **Game End**:
   - The game ends when one player wins or no valid moves remain.
   - The result is logged to the console.

Enhancements
------------

Future updates will focus on:

- **Improved Visualization**:
  - Highlight valid moves for human players.
  - Add animations for piece movement and captures.

- **Rules Compliance**:
  - Ensure all game rules (e.g., forced captures) are enforced.
  - Display error messages for invalid moves.

- **AI Customization**:
  - Allow users to switch between AI strategies.
  - Provide difficulty levels for the AI.

Dependencies
------------

The interactive interface requires the following libraries:

- **Pygame**: For graphical display and user interaction.
- **TensorFlow/Keras**: To load and use the trained model.
- **Checkers Module**: Custom module implementing game logic.

Important Note
------------

The tensorflow version is specified to avoid compatibility issues with the trained model. it should be the same as the one that trained and stored .keras file.

Run the following command to install dependencies:

.. code-block:: bash

   pip install pygame tensorflow==2.11.0