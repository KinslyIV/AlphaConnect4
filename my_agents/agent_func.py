
import numpy as np
import torch


model_v2  = None
model_v1 = None
model_v3 = None

# Load your trained model
# model_v2 = DQN.load("C:\\Users\\Immat\\OneDrive\\Documents\\My Projects\\R_Learining\\connectx_2_1.zip", device='cpu')
# model_v3 = DQN.load("C:\\Users\\Immat\\OneDrive\\Documents\\My Projects\\R_Learining\\connectx_3.zip", device='cpu')


def agent_3(observation, configuration):
    board = np.array(observation["board"], dtype=np.int8)
    valid_actions = [c for c in range(configuration['columns']) if board[c] == 0]

    # Convert board to model input
    state_tensor = torch.tensor(board, dtype=torch.float32, device='cpu').unsqueeze(0)

    # Get Q-values from the model
    with torch.no_grad():
        q_values = model_v3.q_net(state_tensor).numpy().flatten()

    # Mask invalid actions
    q_values_filtered = np.full_like(q_values, -np.inf, device='cpu')
    for a in valid_actions:
        q_values_filtered[a] = q_values[a]

    # Return best valid action
    action = int(np.argmax(q_values_filtered))
    # print("Agent Action : ", action)
    return action


# Define the Kaggle-compatible agent function
def agent_2(observation, configuration):
    board = np.array(observation["board"], dtype=np.int8)
    valid_actions = [c for c in range(configuration['columns']) if board[c] == 0]

    # Convert board to model input
    state_tensor = torch.tensor(board, dtype=torch.float32, device='cpu').unsqueeze(0)

    # Get Q-values from the model
    with torch.no_grad():
        q_values = model_v2.q_net(state_tensor).numpy().flatten()

    # Mask invalid actions
    q_values_filtered = np.full_like(q_values, -np.inf, device='cpu')
    for a in valid_actions:
        q_values_filtered[a] = q_values[a]

    # Return best valid action
    action = int(np.argmax(q_values_filtered))
    # print("Agent Action : ", action)
    return action


def agent_1(observation, configuration):
    board = np.array(observation["board"], dtype=np.int8)
    valid_actions = [c for c in range(configuration['columns']) if board[c] == 0]

    # Convert board to model input
    state_tensor = torch.tensor(board, dtype=torch.float32, device='cpu').unsqueeze(0)

    # Get Q-values from the model
    with torch.no_grad():
        q_values = model_v1.q_net(state_tensor).numpy().flatten()

    # Mask invalid actions
    q_values_filtered = np.full_like(q_values, -np.inf, device='cpu')
    for a in valid_actions:
        q_values_filtered[a] = q_values[a]

    # Return best valid action
    action = int(np.argmax(q_values_filtered))
    # print("Agent Action : ", action)
    return action


def minimax_agent(observation, configuration):
    """
    Connect4 agent using minimax algorithm with alpha-beta pruning.
    
    Args:
        observation: The current game state
        configuration: Game configuration
        max_depth: Maximum search depth (adjust to balance speed vs performance)
    
    Returns:
        The column to play in
    """
    max_depth=5
    board = np.array(observation["board"])
    board = board.reshape(configuration['rows'], configuration['columns'])
    player = observation["mark"]
    opponent = 1 if player == 2 else 2
    
    # Get valid moves (empty columns)
    valid_moves = [c for c in range(configuration['columns']) if board[0][c] == 0]
    
    # If only one valid move, return it immediately
    if len(valid_moves) == 1:
        return valid_moves[0]
    
    # For each valid move, compute the minimax value
    best_score = -float('inf')
    best_move = valid_moves[0]  # Default to first valid move
    
    for col in valid_moves:
        # Create a copy of the board
        board_copy = np.copy(board)
        
        # Find the row where the piece will land
        for row in range(configuration['rows']-1, -1, -1):
            if board_copy[row][col] == 0:
                board_copy[row][col] = player
                break
        
        # Calculate the score using minimax with alpha-beta pruning
        score = minimax(board_copy, max_depth-1, -float('inf'), float('inf'), False, player, opponent, configuration)
        
        # Update best move if we found a better score
        if score > best_score:
            best_score = score
            best_move = col
    
    return best_move

def minimax(board, depth, alpha, beta, is_maximizing, player, opponent, configuration):
    """
    Minimax algorithm with alpha-beta pruning for Connect4.
    
    Args:
        board: Current board state
        depth: Current search depth
        alpha: Alpha value for pruning
        beta: Beta value for pruning
        is_maximizing: True if maximizing player's turn
        player: Player number (1 or 2)
        opponent: Opponent number (1 or 2)
        configuration: Game configuration
    
    Returns:
        The score for the current board state
    """
    # Check terminal states
    if check_win(board, player, configuration):
        return 1000  # Player wins
    if check_win(board, opponent, configuration):
        return -1000  # Opponent wins
    if np.all(board[0] != 0):  # Top row is full
        return 0  # Draw
    if depth == 0:
        return evaluate_board(board, player, opponent, configuration)
    
    if is_maximizing:
        max_score = -float('inf')
        current_player = player
    else:
        max_score = float('inf')
        current_player = opponent
    
    # For each valid move
    for col in range(configuration['columns']):
        if board[0][col] != 0:  # Column is full
            continue
        
        # Create a copy of the board and make the move
        board_copy = np.copy(board)
        for row in range(configuration['rows']-1, -1, -1):
            if board_copy[row][col] == 0:
                board_copy[row][col] = current_player
                break
        
        # Recursive call
        if is_maximizing:
            score = minimax(board_copy, depth-1, alpha, beta, False, player, opponent, configuration)
            max_score = max(max_score, score)
            alpha = max(alpha, score)
        else:
            score = minimax(board_copy, depth-1, alpha, beta, True, player, opponent, configuration)
            max_score = min(max_score, score)
            beta = min(beta, score)
        
        # Alpha-beta pruning
        if beta <= alpha:
            break
    
    return max_score

def check_win(board, player, configuration):
    """Check if player has won"""
    # Check horizontal
    for row in range(configuration['rows']):
        for col in range(configuration['columns'] - configuration['inarow'] + 1):
            if all(board[row][col+i] == player for i in range(configuration['inarow'])):
                return True
    
    # Check vertical
    for row in range(configuration['rows'] - configuration['inarow'] + 1):
        for col in range(configuration['columns']):
            if all(board[row+i][col] == player for i in range(configuration['inarow'])):
                return True
    
    # Check positive diagonal
    for row in range(configuration['rows'] - configuration['inarow'] + 1):
        for col in range(configuration['columns'] - configuration['inarow'] + 1):
            if all(board[row+i][col+i] == player for i in range(configuration['inarow'])):
                return True
    
    # Check negative diagonal
    for row in range(configuration['inarow'] - 1, configuration['rows']):
        for col in range(configuration['columns'] - configuration['inarow'] + 1):
            if all(board[row-i][col+i] == player for i in range(configuration['inarow'])):
                return True
    
    return False

def evaluate_board(board, player, opponent, configuration):
    """
    Heuristic evaluation of board state for non-terminal positions.
    Gives higher scores to boards where the player has more potential winning positions.
    """
    score = 0
    
    # Check all potential winning lines
    # Horizontal
    for row in range(configuration['rows']):
        for col in range(configuration['columns'] - configuration['inarow'] + 1):
            window = [board[row][col+i] for i in range(configuration['inarow'])]
            score += evaluate_window(window, player, opponent)
    
    # Vertical 
    for row in range(configuration['rows'] - configuration['inarow'] + 1):
        for col in range(configuration['columns']):
            window = [board[row+i][col] for i in range(configuration['inarow'])]
            score += evaluate_window(window, player, opponent)
    
    # Positive diagonal
    for row in range(configuration['rows'] - configuration['inarow'] + 1):
        for col in range(configuration['columns'] - configuration['inarow'] + 1):
            window = [board[row+i][col+i] for i in range(configuration['inarow'])]
            score += evaluate_window(window, player, opponent)
    
    # Negative diagonal
    for row in range(configuration['inarow'] - 1, configuration['rows']):
        for col in range(configuration['columns'] - configuration['inarow'] + 1):
            window = [board[row-i][col+i] for i in range(configuration['inarow'])]
            score += evaluate_window(window, player, opponent)
    
    # Prefer center column
    center_col = configuration['columns'] // 2
    center_count = np.sum(board[:, center_col] == player)
    score += center_count * 3
    
    return score

def evaluate_window(window, player, opponent):
    """
    Evaluate a window of 4 positions and assign a score.
    Windows with more player pieces get higher scores.
    Windows with opponent pieces get lower scores.
    """
    score = 0
    
    player_count = window.count(player)
    opponent_count = window.count(opponent)
    empty_count = window.count(0)
    
    if player_count == 4:
        score += 100
    elif player_count == 3 and empty_count == 1:
        score += 5
    elif player_count == 2 and empty_count == 2:
        score += 2
    
    if opponent_count == 3 and empty_count == 1:
        score -= 4  # Block opponent's potential win
    
    return score

