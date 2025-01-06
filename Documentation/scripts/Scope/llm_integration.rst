LLM Integration for Checkers Training with Reinforcement Learning
===============================================================

.. figure:: /Documentation/images/parsed_results.png
   :width: 900
   :align: center
   :alt: parsed_results

--------------------------------------------------------------


**LLM Used**
----------------
The Large Language Model (LLM) used in this project is `Qwen2.5:14b`, an advanced language model designed for structured reasoning and recommendation tasks. The LLM is accessed through the Ollama API, configured with the following parameters:

- **Model**: Qwen2.5:14b
- **API Key**: Local instance key provided via `ollama`
- **Base URL**: `http://localhost:11434/v1`
- **Response Structure**: Python list of tuples representing valid checkers moves.

The LLM processes the current board state and the player's turn to recommend three optimal moves that comply with the rules and strategy of checkers.


**LLM Integration Steps**
----------------------------
The integration of the LLM into the reinforcement learning (RL) pipeline involved the following steps:

- **Step 1**: System and Human Prompt Configuration:
    - A detailed system prompt was designed to describe the rules, board structure, and expected output.
    - The `ChatPromptTemplate` was created with placeholders for dynamic insertion of the board state and current player.

- **Step 2**: Structured Output Definition:
    - The `Moves_Checkers` class was defined using Pydantic to ensure the LLM returns outputs in a consistent format.
    - The prompt mandates that the LLM returns three recommended moves, each formatted as tuples (i.e., ((start_x, start_y), (end_x, end_y))).

- **Step 3**: Execution Function `Run_LLM`
    - The `Run_LLM` function was implemented to send requests to the LLM with retries for fault tolerance.
    - Inputs: `Board_State` (10x10 grid) and `Current_Player` (1 for White, -1 for Black).
    - Outputs: List of three recommended actions for the RL agent.

- **Step 4**: Integration with Q-Learning Model
    - The LLM’s top 3 recommended moves are passed to the Q-learning model.
    - The model evaluates each suggested move based on features extracted from the current board state.
    - The final move is chosen either via exploration (random) or exploitation (highest Q-value).

- **Key Code Components**:
    - `format_board_state`: Converts the board matrix into a string for LLM input.
    - `get_top_3_actions`: Calls `Run_LLM` and retrieves the LLM’s top 3 move recommendations.


**Results**
--------------

- **Training Metrics**:
    - **Win Rate**: The win rate increased steadily as the Q-learning model trained with the LLM’s narrowed action space.
    - **Loss Reduction**: The average loss per generation decreased, indicating better predictions.
    - **Reward Stability**: The average reward per move improved as the model learned to choose more strategic actions.

- **Quantitative Observations**:
    - Initial exploration rate: 0.9 (high exploration, gradually decreased).
    - The LLM’s move recommendations significantly reduced the action space from all legal moves to the top 3 optimal choices.
    - On average, the win rate reached **X%** after 5 generations of training.

- **Visualization**:
    - **Win Rate vs. Generations**: Showed steady improvement.
    - **Loss vs. Generations**: Demonstrated decreasing trend.
    - **Reward vs. Generations**: Reflected the model’s increasing strategic proficiency.

- **Advantages of LLM-Assisted Training**:
    - **Enhanced Decision Space**: The LLM provided more strategic options compared to random exploration.
    - **Faster Convergence**: By limiting the action space to top 3 moves, the RL agent converged to optimal strategies faster.

- **Limitations**:
    - Occasional invalid outputs (mitigated by retry mechanism).
    - Higher latency due to LLM inference time.


**Conclusion and Future Work**
---------------------------------
The integration of the LLM with the RL agent demonstrated that narrowing the action space using top recommendations can enhance the training process. Future improvements include:
- Fine-tuning the LLM for faster inference.
- Expanding the system to evaluate multi-move strategies.
- Optimizing the Q-learning model with additional features for complex board states.