LLM backed Chess Reinforcement Learning Documentation
=============================================

.. figure:: /Documentation/images/pilot.png
   :width: 500
   :align: center
   :alt: Documentation Cover

--------------------------------------------------------


Checkers Reinforcement Learning Model
=====================================

This documentation provides an overview of the training process and architecture of a reinforcement learning model designed to play the game of checkers. It incorporates a hybrid Deep Q-Learning approach, combined with custom self-play and evaluation mechanisms, with a core integration of a Language Model (LLM) to limit the action space effectively.

Key Features
------------

- **Reinforcement Learning (RL)**: Utilizes temporal difference learning with a neural network to approximate Q-values for state-action pairs.
- **Self-Play**: The model trains against itself to iteratively improve its gameplay.
- **Custom Opponents**: Supports training against various opponents, including random moves, minimax strategies, and itself.
- **LLM-Driven Action Space Limitation**: A Language Model (LLM) acts as an integral part of the architecture, dynamically reducing the action space to the most promising options.
- **Exploration-Exploitation Balance**: Implements a dynamic exploration parameter to balance random exploration and policy exploitation.
- **Performance Metrics**: Tracks win rates over generations for performance evaluation.



.. toctree::
   :maxdepth: 2
   :caption: Introduction

   Documentation/scripts/Scope/introduction.rst

.. toctree::
   :maxdepth: 2
   :caption: Implementation

   Documentation/scripts/Scope/implementation.rst

.. toctree::
   :maxdepth: 2
   :caption: Interface

   Documentation/scripts/Scope/interface.rst