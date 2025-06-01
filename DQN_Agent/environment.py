import gymnasium as gym
from kaggle_environments import make
from gymnasium.core import Env
from kaggle_environments.utils import Struct
import numpy as np
from agents.agent_func import minimax_agent 


class ConnectX(Env):
    def __init__(self, config : dict | None  = None, agents : list | None = None):
        super().__init__()
        self.env = make('connectx', 
                        configuration= {"rows": getattr(config, "rows", 6), 
                                        "columns": getattr(config, "columns", 7), 
                                        "inarow": getattr(config, "inarow", 4),
                                        "episodeSteps": getattr(config, "episodeSteps", None),
                                        "agentTimeout": getattr(config, "agentTimeout", None),
                                        "actTimeout": getattr(config, "actTimeout", None),
                                        "runTimeout": getattr(config, "runTimeout", None)
                                        },
                        steps=[],
                        debug=True)
        self.configuration = self.env.configuration
        self.board_rows = self.env.configuration.rows
        self.board_columns = self.env.configuration.columns
        self.agents : list | None = agents
        self.action_space = gym.spaces.Discrete(self.board_columns)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=2,
            shape=(self.board_rows * self.board_columns,),
            dtype=np.int32
        )
        self.observation : Struct
        self.current_player : bool = False

    
    def step(self, action : int):

        action = int(action)

        if self.observation.board[action] != 0:
            return self.observation.board, -10, False, True, {"invalid": True}

        info = self.env.step([action, None])[0]
        status = getattr(info, 'status')
        self.observation = getattr(info, "observation")
        board = self._get_state(self.observation)
        # reward = self._get_reward(action, info.reward, status)
        reward = info.reward

        if status != "DONE":
            try:
                info2 = self.env.step([None, minimax_agent(self.observation, self.configuration)])[0]
                status = getattr(info2, 'status')
                self.observation = getattr(info2, "observation")
                board = self._get_state(self.observation)

                if status == "DONE" and getattr(info2, "reward", 0) == -1:
                    reward = -10
            except Exception as e:
                # Handle potential error if environment is already done
                print(f"Error during opponent move: {e}")
                terminated = True
                truncated = True
        
        additional_info = getattr(info, 'info')
        terminated = status == 'DONE'
        truncated = getattr(self.observation, 'remainingOverageTime') == 0 and not terminated

        # print(board, reward, terminated, truncated, additional_info)
        
        return board, reward, terminated, truncated, additional_info
    
    def _get_state(self, obs):
        state = getattr(obs, 'board')
        return np.array(state)
    
    def _get_reward(self, action: int, reward: int, status: str):
        board = self._get_state(self.observation).reshape(
            (self.board_rows, self.board_columns)
        )
        mark = self.observation.mark

        reward_value = 0

        if status == "DONE":
            if reward == 1:
                reward_value += 10  # Win reward
            elif reward == -1:
                reward_value -= 10  # Lose penalty
            elif reward == 0:
                reward_value += 0.5  # Small reward for draw
            return reward_value

        # I Intermediate rewards - scaled appropriately
        # Reward for 3 in a row (75% of win reward)
        if self.made_3_streak(board, action, mark):
            reward_value += 5
        
        # # Check for 2 in a row (25% of win reward)
        # if self.made_2_streak(board, action, mark):
        #     reward_value += 2.5
        
        # Bonus: block opponent's potential win (50% of win reward)
        if self.is_block_move(board, action, mark):
            reward_value += 
        
        # Small penalty for moves that don't create advantages
        if reward_value == 0:
            reward_value -= 0.1  # Slight penalty to encourage meaningful moves
        
        return reward_value
    

    def _opponent_policy(self):
        # Simple random opponent
        valid_moves = [c for c in range(self.configuration.columns)
                       if self.observation.board[c] == 0]
        return int(np.random.choice(valid_moves)) if valid_moves else 0

    def reset(self, seed=None):
        info = self.env.reset()[0]
        self.observation = getattr(info, "observation")
        state = self._get_state(self.observation)
        additional_info = getattr(info, 'info')
        return state, additional_info

    def render(self, mode: str = "ipython"):
        self.env.render(mode)
        

    def is_block_move(self, game_board, col, player):
        board = np.copy(game_board)
        ROWS, COLS = len(board), len(board[0])
        opponent = 1 if player == 2 else 2

        # Find the row where the disc just landed
        row = next((r for r in range(ROWS) if board[r][col] != 0), ROWS) - 1
        if row < 0:
            return False  # Invalid move; column was full

        directions = [
            (0, 1),   # horizontal
            (1, 0),   # vertical
            (1, 1),   # diagonal /
            (1, -1)   # diagonal \
        ]

        def count_streak(r, c, dr, dc, target):
            count = 0
            for _ in range(3):
                r += dr
                c += dc
                if 0 <= r < ROWS and 0 <= c < COLS and board[r][c] == target:
                    count += 1
                else:
                    break
            return count

        # Temporarily remove the move to simulate pre-block state
        board[row][col] = 0

        for dr, dc in directions:
            before = count_streak(row, col, -dr, -dc, opponent)
            after = count_streak(row, col, dr, dc, opponent)

            if before + after >= 3:
                board[row][col] = player
                return True

        board[row][col] = player
        return False
    

    def made_3_streak(self, board, col, player):
        ROWS, COLS = len(board), len(board[0])

        # Find the row where the disc just landed
        row = next((r for r in range(ROWS) if board[r][col] != 0), ROWS) - 1
        if row < 0:
            return False  # Invalid move; column was full

        directions = [
            (0, 1),   # horizontal
            (1, 0),   # vertical
            (1, 1),   # diagonal /
            (1, -1)   # diagonal \
        ]

        def count_same(r, c, dr, dc):
            count = 0
            for _ in range(2):  # look for up to 2 more in one direction
                r += dr
                c += dc
                if 0 <= r < ROWS and 0 <= c < COLS and board[r][c] == player:
                    count += 1
                else:
                    break
            return count

        for dr, dc in directions:
            total = 1 + count_same(row, col, dr, dc) + count_same(row, col, -dr, -dc)
            if total == 3:
                return True

        return False

