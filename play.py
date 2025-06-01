import numpy as np
import pygame
from connectx import Game
from pygame_connect4.connect4 import Connect4
from my_agents.agent import Agent, MCTSAgent
from my_agents.agent_func import minimax_agent
from mcts import MCTS
import time


def main():
    game = Connect4()
    config = {'rows': game.rows, 'columns': game.cols, 'inarow': 4}
    clock = pygame.time.Clock()
    agent1 = MCTSAgent(game= Game(), config=config, path="my_agents/models/agent_1001.pth")
    agent2 = Agent(minimax_agent, config=config)
    
    while True:
        # Draw the current state
        game.draw_board()
        
        # Process player input and events
        # human_made_move = game.play_human_move()

        
        
        # Update any ongoing animations
        animation_completed = True
        if game.animation_in_progress:
            animation_completed = game.update_animation()

        if animation_completed and game.current_player == 1 and not game.game_over:
            start = time.time_ns()
            agent2_move = agent2.get_move(game.board.flatten(), game.current_player)
            print(f"Move 1 time {time.time_ns() - start}")
            game.play_ai_move(agent2_move)
        
        # AI's turn - only after human has moved and animation is complete
        if animation_completed and game.current_player == 2 and not game.game_over:
            start = time.time_ns()
            agent1_move = agent1.get_move(game.board.flatten(), game.current_player)
            print(f"Move 2 time {time.time_ns() - start}")
            game.play_ai_move(agent1_move)
        
        # Cap the frame rate
        clock.tick(60)

if __name__ == "__main__":
    main() 