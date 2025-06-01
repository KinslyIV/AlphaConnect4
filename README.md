# AlphaConnect4: AlphaZero for Connect Four

## Overview
AlphaConnect4 is a reinforcement learning project that implements the AlphaZero algorithm to master the game of Connect Four. The project leverages deep neural networks and Monte Carlo Tree Search (MCTS) to train an agent capable of playing Connect Four at a high level, both against other Algorithms and human opponents.

## Project Structure
- **alpha_zero.py**: Core AlphaZero training loop and self-play logic.
- **mcts.py**: Monte Carlo Tree Search implementation.
- **model.py**: Neural network (ResNet) architecture for policy and value prediction.
- **connectx.py**: Game logic for Connect Four.
- **train_alphaconnectx.py**: Script to train the AlphaZero agent.
- **play.py**: Script to play against the trained agent using a graphical interface.
- **my_agents/**: Contains agent wrappers and pre-trained models.
- **eval.ipynb**: Jupyter notebook for evaluating the trained agent against various opponents.

## How It Works
1. **Self-Play**: The AlphaZero agent plays many games in parallel against itself using MCTS guided by its neural network. Game states, policies, and outcomes are stored for training.
2. **Neural Network**: The Neural network contains two heads Policy and Value head. It takes an encoded game state as input with the current player. The Policy head outputs a probability distribution over all actions. Whereas the value head outputs the value of the state (How good that state is) These are used to guide the MCTS. 
2. **Training**: The neural network is trained on the collected data during selfplay to predict the best move (policy) and the expected outcome (value) from any given state.
3. **Evaluation**: The trained agent is evaluated against baseline agents (e.g., minimax, negamax) and itself to measure performance improvements.
4. **Gameplay**: You can play against the trained agent using the provided graphical interface or evaluate it in the notebook.

## Results
- The AlphaZero agent, after training for 20 iterations each of 150 self play games with increasing buffer sizes and learning rate scheduling using the onecycle learning rate schedular, is able to compete against humans an minimax algorithms.
- In evaluation (see `eval.ipynb`), the agent (with 400 monte carlo simulations per move) consistently wins (When first player) or draws (when playing second) against minimax with depth 5 and negamax agents, demonstrating robust learning and generalization.

## Getting Started
1. **Install dependencies**: `pip install -r requirements.txt`
2. **Train the agent**: `python train_alphaconnectx.py`
3. **Play against the agent**: `python play.py`
4. **Evaluate the agent**: Open `eval.ipynb` in Jupyter and run the cells.

## Key Features
- Parallelized self-play for efficient data generation.
- Customizable neural network architecture.
- Subtree reuse in the Monte Carlo subtree

## Acknowledgements
- Inspired by DeepMind's AlphaZero.
- Utilizes PyTorch, NumPy, and Pygame for core functionality.

