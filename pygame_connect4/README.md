# Connect 4 Game

A Connect 4 game implementation using Pygame that supports both human and AI players.

## Features

- Graphical user interface using Pygame
- Support for human players using mouse input
- Support for AI players through a simple interface
- Win detection in all directions (horizontal, vertical, and diagonal)
- Board state accessible as a 1D numpy array for AI training

## Installation

1. Make sure you have Python 3.7+ installed
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Playing the Game

Run the game with:
```bash
python connect4.py
```

- Use your mouse to move the piece horizontally
- Click to drop the piece in the selected column
- The game will automatically detect wins and switch players
- After a win, the game will reset after 3 seconds

### Using with AI

The `Connect4` class provides methods for AI integration:

- `get_board_state()`: Returns the current board state as a 1D numpy array
- `play_ai_move(col)`: Makes a move for the AI player
- `is_valid_move(col)`: Checks if a move is valid
- `check_win()`: Checks if the current player has won

Example AI integration:
```python
game = Connect4()
# Get current board state
board_state = game.get_board_state()
# Make AI move
game.play_ai_move(column_number)
```

## Game Rules

- Players take turns dropping pieces into columns
- The first player to connect 4 pieces horizontally, vertically, or diagonally wins
- If the board fills up without a winner, the game is a draw 