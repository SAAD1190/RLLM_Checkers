Implementation Details
========================

.. figure:: /Documentation/images/architecture.png
   :width: 700
   :align: center
   :alt: architecture

--------------------------------------------------------------

This section provides detailed information about the architecture, training process, and the integration of the Language Model (LLM) as part of the reinforcement learning agent for playing checkers.

Agent Architecture
-------------------

The reinforcement learning agent is designed with a neural network to approximate the Q-values for state-action pairs. Below are the key components of the agent architecture:

- **Input Layer:**
  - Accepts a 5-dimensional feature vector representing the game state.

- **Hidden Layers:**
  - Two dense layers:
    - First layer: 32 neurons with ReLU activation.
    - Second layer: 16 neurons with ReLU activation and L2 regularization (regularization parameter = 0.1).

- **Output Layer:**
  - A single neuron with ReLU activation, predicting the Q-value for the input state.

- **Optimizer and Loss Function:**
  - Optimizer: Nadam (Nesterov-accelerated adaptive moment estimation).
  - Loss Function: Binary cross-entropy.

Training Process
----------------

The training process involves a hybrid approach combining reinforcement learning and supervised fine-tuning. Below are the detailed steps:

1. **Self-Play:**
   - The agent plays against itself or other opponents (e.g., random moves, minimax strategy).

2. **Exploration vs. Exploitation:**
   - An exploration probability (initially 0.95) determines whether the agent explores random actions or exploits the policy learned so far.

3. **Reward Assignment:**
   - Rewards are assigned based on the game outcome:
     - Win: +10
     - Loss: -10
     - Draw: Neutral (implicit handling).

4. **Temporal Difference Learning:**
   - The Q-values are updated using the temporal difference formula:
     
     .. math::
        Q(s, a) \leftarrow Q(s, a) + \alpha \cdot \left( r + \gamma \cdot \max_a Q(s', a) - Q(s, a) \right)
     
     Where:
     - \( \alpha \): Learning rate (set to 0.5).
     - \( \gamma \): Discount factor (set to 0.95).
     - \( r \): Reward for the current state.
     - \( Q(s', a) \): Predicted future Q-values.

5. **Supervised Fine-Tuning:**
   - Labels are updated Q-values, and the agent fine-tunes the model with backpropagation.

6. **Performance Tracking:**
   - Win rates, losses, and rewards are tracked across generations.

LLM Integration
----------------

The LLM acts as a core component to dynamically limit the action space. Its integration is as follows:

1. **Action Space Filtering:**
   - For each game state, the LLM evaluates all possible actions and filters them to the most promising \( n \) actions.

2. **Efficient Decision-Making:**
   - The RL agent focuses only on the reduced action space, leading to faster and more effective decision-making.

3. **Seamless Interaction:**
   - The LLM works alongside the RL agent in real-time, ensuring minimal latency while filtering actions.

Generations
-----------

The training process runs for multiple generations, each consisting of several games. Key details include:

- **Generations:**
  - A total of 25 generations.

- **Games per Generation:**
  - Each generation involves 10 games against a chosen opponent.

- **Exploration Decay:**
  - The exploration probability decreases over generations, allowing the agent to rely more on exploitation.

Next Steps
----------

- **Interface Details:**
  Learn how to interact with the trained model and visualize its performance in the next section.