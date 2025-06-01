import base64
import pickle
import zlib
import numpy as np
import torch
from stable_baselines3 import DQN
from connectx import Game
from mcts import MCTS
from model import ResNet


class Agent:
    def __init__(self, agent, config: dict):
        self.agent = agent
        self.config = config
        

    def get_move(self, board: np.ndarray, mark : int):
        observation = {'board': board, 
                       'mark': mark}
        action = self.agent(observation, self.config)
        return action


class MCTSAgent(Agent):
    def __init__(self, game : Game , config: dict, path):
        super().__init__(None, config)
        # game = Game()
        device = torch.device('cpu')
        model = ResNet(game=game, num_hidden=64, num_resBlocks=8, device=device)
        model.state_dict()
        model = torch.load(path, weights_only=False)
        model.to(device)
        model.eval()
        self.mcts = MCTS(model=model, game=Game(config=config), n_loops=400, device=device, c=2.0)
        self.agent = lambda observation, config: np.argmax(self.mcts.run(observation, config)[0]) # type: ignore



class AlphaConnect4: 
    
    def __init__(self, game : Game, path, n_sims):
        self.device = torch.device('cpu')
        self.model = torch.load(path, weights_only=False, map_location=torch.device('cpu'))
        self.mcts = MCTS(model=self.model, game=game, n_loops=n_sims, device=self.device, c=2)


    def make_move(self, observation, configuration):
        action = int(np.argmax(self.mcts.run(observation, configuration)[0]))
        # logger.info("Called for action")
        return action
        




