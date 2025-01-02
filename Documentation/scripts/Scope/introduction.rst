Introduction
========================

.. figure:: /Documentation/images/intro.png
   :width: 500
   :align: center
   :alt: Image explaining Prompt Analyzer introduction

--------------------------------------------------------------

Overview
--------

This project implements a reinforcement learning agent to play the game of checkers. The goal is to design a self-improving agent capable of learning optimal strategies through self-play and dynamic decision-making. The core innovation of this project lies in the integration of a Language Model (LLM) to act as an action space limiter, enabling efficient and focused decision-making.

Motivation
----------

Traditional reinforcement learning approaches often struggle with large action spaces, especially in complex games like checkers. By incorporating an LLM to filter and prioritize actions, this project:

- Reduces computational overhead.
- Enhances the agent’s decision-making efficiency.
- Introduces a novel hybrid approach combining reinforcement learning and natural language processing techniques.

Goals
-----

The primary objectives of this project include:

1. Developing a reinforcement learning agent capable of self-play and iterative improvement.
2. Demonstrating the effectiveness of LLMs in reducing the action space in real-time.
3. Evaluating the agent's performance through metrics like win rates and reward distributions.

Key Features
------------

- **Self-Play Reinforcement Learning:**
  The agent learns by playing against itself, improving iteratively with each generation.

- **LLM Integration:**
  The LLM acts as a core component to limit the action space dynamically, ensuring the agent considers only the most promising moves.

- **Customizable Opponents:**
  The agent can train against various types of opponents, including random moves, minimax strategies, and itself.

- **Performance Metrics:**
  The system tracks win rates, losses, and rewards over generations, providing insights into the agent's learning progress.


Next Steps
----------

To go deeper, proceed to the following sections:

- **Implementation:** Learn about the agent’s architecture, training process, and the integration of the LLM.
- **Interface:** Discover how to interact with the trained model and visualize its performance.
