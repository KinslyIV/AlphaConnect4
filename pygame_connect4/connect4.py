import pygame
import numpy as np
import sys
import time

class Connect4:
    def __init__(self, rows=6, cols=7):
        self.rows = rows
        self.cols = cols
        self.board = np.zeros((rows, cols), dtype=int)
        self.game_over = False
        self.current_player = 1  # 1 for player 1, 2 for player 2
        self.winning_line = None  # Store winning line coordinates
        self.game_result = None  # Store game result: "win", "draw", or None
        
        # Pygame setup
        self.SQUARESIZE = 100
        self.width = self.cols * self.SQUARESIZE
        self.height = (self.rows + 1) * self.SQUARESIZE
        self.size = (self.width, self.height)
        self.RADIUS = int(self.SQUARESIZE/2 - 5)
        
        # Colors - Using brighter, more vibrant colors
        self.BACKGROUND = (240, 240, 255)  # Light blue background
        self.BOARD_COLOR = (0, 100, 255)   # Bright blue for board
        self.BLACK = (0, 0, 0)
        self.RED = (255, 50, 50)          # Brighter red
        self.YELLOW = (255, 255, 0)       # Bright yellow
        self.WHITE = (255, 255, 255)
        
        pygame.init()
        self.screen = pygame.display.set_mode(self.size)
        pygame.display.set_caption("Connect 4")
        self.font = pygame.font.SysFont("monospace", 75)
        self.small_font = pygame.font.SysFont("monospace", 40)
        
        # Animation variables
        self.animation_in_progress = False
        self.animation_piece = None
        self.animation_col = None
        self.animation_row = None
        self.animation_y = 0
        self.animation_speed = 15

    def get_board_state(self):
        """Return the current board state as a 1D numpy array"""
        return self.board.flatten()

    def is_valid_move(self, col):
        """Check if the move is valid"""
        return self.board[0][col] == 0

    def get_next_open_row(self, col):
        """Get the next available row in the given column"""
        for r in range(self.rows-1, -1, -1):
            if self.board[r][col] == 0:
                return r
        return None

    def drop_piece(self, col):
        """Drop a piece in the specified column"""
        if self.is_valid_move(col):
            row = self.get_next_open_row(col)
            if row is not None:
                self.start_animation(col, row)
                return True
        return False

    def start_animation(self, col, row):
        """Start the piece dropping animation"""
        self.animation_in_progress = True
        self.animation_col = col
        self.animation_row = row
        self.animation_y = 0
        self.animation_piece = self.current_player

    def check_draw(self):
        """Check if the game is a draw"""
        return np.all(self.board != 0)

    def update_animation(self):
        """Update the piece dropping animation"""
        if self.animation_in_progress:
            self.animation_y += self.animation_speed
            target_y = self.animation_row * self.SQUARESIZE + self.SQUARESIZE
            
            if self.animation_y >= target_y:
                self.animation_in_progress = False
                self.board[self.animation_row][self.animation_col] = self.animation_piece
                if self.check_win():
                    self.game_over = True
                    self.game_result = "win"
                elif self.check_draw():
                    self.game_over = True
                    self.game_result = "draw"
                else:
                    self.current_player = 3 - self.current_player
                return True
        return False

    def check_win(self):
        """Check if the current player has won and store winning line"""
        # Check horizontal
        for c in range(self.cols-3):
            for r in range(self.rows):
                if (self.board[r][c] == self.current_player and
                    self.board[r][c+1] == self.current_player and
                    self.board[r][c+2] == self.current_player and
                    self.board[r][c+3] == self.current_player):
                    self.winning_line = [(r, c), (r, c+3)]
                    return True

        # Check vertical
        for c in range(self.cols):
            for r in range(self.rows-3):
                if (self.board[r][c] == self.current_player and
                    self.board[r+1][c] == self.current_player and
                    self.board[r+2][c] == self.current_player and
                    self.board[r+3][c] == self.current_player):
                    self.winning_line = [(r, c), (r+3, c)]
                    return True

        # Check positive diagonal
        for c in range(self.cols-3):
            for r in range(self.rows-3):
                if (self.board[r][c] == self.current_player and
                    self.board[r+1][c+1] == self.current_player and
                    self.board[r+2][c+2] == self.current_player and
                    self.board[r+3][c+3] == self.current_player):
                    self.winning_line = [(r, c), (r+3, c+3)]
                    return True

        # Check negative diagonal
        for c in range(self.cols-3):
            for r in range(3, self.rows):
                if (self.board[r][c] == self.current_player and
                    self.board[r-1][c+1] == self.current_player and
                    self.board[r-2][c+2] == self.current_player and
                    self.board[r-3][c+3] == self.current_player):
                    self.winning_line = [(r, c), (r-3, c+3)]
                    return True

        return False

    def draw_game_result(self):
        """Draw the game result message at the top of the board"""
        if self.game_result == "win":
            text = f"Player {self.current_player} Wins!"
            color = self.RED if self.current_player == 1 else self.YELLOW
        else:  # draw
            text = "Game Draw!"
            color = self.WHITE

        # Draw a background rectangle for the text
        pygame.draw.rect(self.screen, self.BACKGROUND, (0, 0, self.width, self.SQUARESIZE))
        
        # Draw the text
        label = self.font.render(text, 1, color)
        text_rect = label.get_rect(center=(self.width/2, self.SQUARESIZE/2))
        self.screen.blit(label, text_rect)

        # Draw "Click to continue" text
        continue_text = self.small_font.render("Click to continue", 1, self.BLACK)
        continue_rect = continue_text.get_rect(center=(self.width/2, self.SQUARESIZE/2 + 30))
        self.screen.blit(continue_text, continue_rect)

    def draw_board(self):
        """Draw the game board"""
        self.screen.fill(self.BACKGROUND)
        
        # Draw the board
        for c in range(self.cols):
            for r in range(self.rows):
                pygame.draw.rect(self.screen, self.BOARD_COLOR,
                               (c*self.SQUARESIZE, r*self.SQUARESIZE+self.SQUARESIZE,
                                self.SQUARESIZE, self.SQUARESIZE))
                pygame.draw.circle(self.screen, self.BACKGROUND,
                                 (int(c*self.SQUARESIZE+self.SQUARESIZE/2),
                                  int(r*self.SQUARESIZE+self.SQUARESIZE+self.SQUARESIZE/2)),
                                 self.RADIUS)

        # Draw the pieces
        for c in range(self.cols):
            for r in range(self.rows):
                if self.board[r][c] == 1:
                    pygame.draw.circle(self.screen, self.RED,
                                     (int(c*self.SQUARESIZE+self.SQUARESIZE/2),
                                      int(r*self.SQUARESIZE+self.SQUARESIZE+self.SQUARESIZE/2)),
                                     self.RADIUS)
                elif self.board[r][c] == 2:
                    pygame.draw.circle(self.screen, self.YELLOW,
                                     (int(c*self.SQUARESIZE+self.SQUARESIZE/2),
                                      int(r*self.SQUARESIZE+self.SQUARESIZE+self.SQUARESIZE/2)),
                                     self.RADIUS)

        # Draw the animated piece
        if self.animation_in_progress:
            color = self.RED if self.animation_piece == 1 else self.YELLOW
            pygame.draw.circle(self.screen, color,
                             (int(self.animation_col*self.SQUARESIZE+self.SQUARESIZE/2),
                              int(self.animation_y+self.SQUARESIZE/2)),
                             self.RADIUS)

        # Draw winning line
        if self.winning_line:
            start_pos = (int(self.winning_line[0][1]*self.SQUARESIZE+self.SQUARESIZE/2),
                        int(self.winning_line[0][0]*self.SQUARESIZE+self.SQUARESIZE+self.SQUARESIZE/2))
            end_pos = (int(self.winning_line[1][1]*self.SQUARESIZE+self.SQUARESIZE/2),
                      int(self.winning_line[1][0]*self.SQUARESIZE+self.SQUARESIZE+self.SQUARESIZE/2))
            pygame.draw.line(self.screen, self.WHITE, start_pos, end_pos, 10)

        # Draw the hovering piece
        if not self.game_over and not self.animation_in_progress:
            posx = pygame.mouse.get_pos()[0]
            pygame.draw.circle(self.screen, self.RED if self.current_player == 1 else self.YELLOW,
                             (posx, int(self.SQUARESIZE/2)), self.RADIUS)

        # Draw game result if game is over
        if self.game_over:
            self.draw_game_result()

        pygame.display.update()

    def play_human_move(self):
        """Handle human player's move"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.MOUSEMOTION and not self.game_over and not self.animation_in_progress:
                pygame.draw.rect(self.screen, self.BACKGROUND, (0, 0, self.width, self.SQUARESIZE))
                posx = event.pos[0]
                pygame.draw.circle(self.screen, self.RED if self.current_player == 1 else self.YELLOW,
                                 (posx, int(self.SQUARESIZE/2)), self.RADIUS)
                pygame.display.update()

            if event.type == pygame.MOUSEBUTTONDOWN:
                if self.game_over:
                    self.reset_game()
                elif not self.animation_in_progress:
                    posx = event.pos[0]
                    col = int(posx // self.SQUARESIZE)
                    self.drop_piece(col)

    def play_ai_move(self, col):
        """Handle AI player's move"""
        if not self.game_over and not self.animation_in_progress:
            self.drop_piece(col)

    def reset_game(self):
        """Reset the game state"""
        self.board = np.zeros((self.rows, self.cols), dtype=int)
        self.game_over = False
        self.current_player = 1
        self.winning_line = None
        self.animation_in_progress = False
        self.game_result = None

def main():
    game = Connect4()
    clock = pygame.time.Clock()
    
    while True:
        game.draw_board()
        game.play_human_move()
        
        if game.animation_in_progress:
            game.update_animation()
        
        clock.tick(60)  # Cap the frame rate at 60 FPS

if __name__ == "__main__":
    main() 