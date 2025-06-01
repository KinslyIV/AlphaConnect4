from typing import Any
import numpy as np

class Game:
    def __init__(self, config : Any | None= None) -> None:
        self.rows = getattr(config, "rows", 6) 
        self.columns  = getattr(config, "columns", 7) 
        self.inarow = getattr(config, "inarow", 4)
        self.action_size = int(self.columns)
        self.config = config

    def get_initial_state(self):
        return np.array(np.zeros((self.rows, self.columns), dtype=np.int8))
    
    def get_valid_moves(self, state):
        return state[0] == 0
    

    def get_next_state(self, state, action, player):
        if not state[0][action]:
            new_state = state.copy()
            col = action
            row = self.get_next_row(col, new_state)
            new_state[row][col] = player
            return new_state
        print(state)
        return None


    def get_next_row(self, col, state):
        for i in range(self.rows-1, -1, -1):
            if state[i][col] == 0:
                return i
        return -1
            
    # @DeprecationWarning
    # def make_move(self, action, player):
    #     if self.get_valid_moves(self.current_state)[action]:
    #         col = action
    #         row = self.get_next_row(col, self.current_state)
    #         self.current_state[row][col] = player
    #         return True
    #     return False
    

    def check_win(self, state, action):
        col = action
        # Find the row where the last piece was placed
        for row in range(self.rows):
            if state[row][col] != 0:
                break
        else:
            return False  # No piece in this column

        player = state[row][col]
        inarow = self.inarow

        def count_dir(delta_row, delta_col):
            r, c = row + delta_row, col + delta_col
            count = 0
            while 0 <= r < self.rows and 0 <= c < self.columns and state[r][c] == player:
                count += 1
                r += delta_row
                c += delta_col
            return count

        # Check all 4 directions
        directions = [ (0,1), (1,0), (1,1), (1,-1) ]
        for dr, dc in directions:
            count = 1
            count += count_dir(dr, dc)
            count += count_dir(-dr, -dc)
            if count >= inarow:
                return True
        return False
    

    def get_result_and_terminated(self, state, action):
        # Check if current player wins after this action
        if self.check_win(state, action):
            return 1, True  # Win

        # Check for draw (no valid moves left)
        if not np.any(state[0] == 0):
            return 0, True  # Draw

        # Otherwise, game is not terminated
        return 0, False
    

    def encode_state(self, state, player):

        if player == 2:
            swapped_board = state.copy()
            swapped_board[state == 1] = -1    # Temporarily mark player 1 as -1
            swapped_board[state == 2] = 1     # Replace player 2 with 1
            swapped_board[swapped_board == -1] = 2  # Replace temporary -1 with 2
            state = swapped_board

        channel_1 = (state == 1).astype(float)
        channel_2 = (state == 2).astype(float)
        channel_3 = (state == 0).astype(float)

        return np.stack([channel_1, channel_2, channel_3])
    
