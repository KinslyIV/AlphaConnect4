
import copy
import random
import torch
from connectx import Game
import numpy as np
import math
from model import ResNet
import logging


logger = logging.getLogger(__name__)
logging.basicConfig(filename='mcts_log.log', encoding='utf-8', level=logging.DEBUG)



class Node:
    def __init__(self, 
                 state, 
                 game : Game, 
                 last_action : int, 
                 player :int, 
                 parent : 'Node | None',
                 prob_dist : list[float], 
                 c : float = 2,
                 prior : float = 0) -> None:
        self.state = state
        self.n_visits : int = 0
        self.game = game
        self.valid_moves_bool = self.game.get_valid_moves(self.state)
        self.valid_actions = [i for i, action in enumerate(self.valid_moves_bool) if action == True]
        self.action_to_expand = copy.deepcopy(self.valid_actions)
        self.parent : Node | None = parent
        self.children_list : list[Node] = []
        self.is_terminal : bool = self.game.get_result_and_terminated(self.state, last_action)[1]
        self.c = c
        self.prior = prior
        self.prob_dist : list[float] = prob_dist
        self.value_1 : float = 0
        self.value_2 : float = 0
        self.value : float = 0
        self.last_action = last_action
        self.current_player = player
        self.opponent = 3 - player


    def __repr__(self) -> str:
        return str(self.state)
        pass

    @DeprecationWarning
    def get_value(self, root_player):
        if root_player == 1:
            return self.value_1
        else:
            return self.value_2
        
    @DeprecationWarning
    def back_prop_value(self, root_player, value):
        if root_player == 1:
            self.value_1 += value
            self.value_2 -= value
        else:
            self.value_2 += value
            self.value_1 -= value
        
    @property
    def q_value(self):
        if self.n_visits == 0:
            print("Why are you here ??")
            return 0
        return self.value / self.n_visits 

    def children_cpuct(self): # Not correct for the parent node value has to be flipped

        if self.n_visits == 0:
            print("Komisch!!!")
            return float('inf')
        
        return [ - child.q_value + self.c  * child.prior * (math.sqrt(self.n_visits) / (child.n_visits + 1)) for child in self.expanded_children]
    
    @property
    def expanded_children(self):
        return self.children_list
    
    @property
    def is_fully_expanded(self):
        return len(self.expanded_children) == len(self.valid_actions)
    
    def expand_node(self):
        if self.is_fully_expanded:
            raise Exception("Trying to Expand fully expanded node or Terminal Node")
            
        else:

            action = random.choice(self.action_to_expand)
                
            prior = self.prob_dist[action]
            next_state = self.game.get_next_state(self.state, action, self.current_player)
            p = self.valid_moves_bool.astype(float) / np.sum(self.valid_moves_bool)
            new_child = Node(next_state, self.game, action, self.opponent,prob_dist=p, parent=self, prior=prior, c=self.c)
            self.children_list.append(new_child)
            self.action_to_expand.remove(action)
            return new_child



class MCTS:
    def __init__(self, model : ResNet,  
                 device,
                 game : Game, 
                 n_loops, 
                 c : float
                 ) -> None:
        self.device = device
        self.game = game
        self.model = model
        self.n_loops = n_loops
        self.root_node : Node
        self.c = c

    def init_root(self, player, state: np.ndarray, c : float):
        p = self.game.get_valid_moves(state).astype(float)
        p = p / np.sum(p)
        
        self.root_node = Node(state, self.game, prob_dist=p, last_action=-1, player=player, parent=None, c=c)


    def use_subtree(self, player, state):
        new_root = None
        current_node = self.root_node
        for child in current_node.expanded_children:

            if np.array_equal(child.state, state) and child.current_player == player:
                    new_root = child
                    break
            
            for g_child in child.expanded_children: # Change this during play for subtree reuse

                if np.array_equal(g_child.state, state) and g_child.current_player == player:
                    new_root = g_child
                    break

        if new_root is not None:
            self.root_node = new_root
            self.root_node.parent = None
            self.root_node.last_action = -1
            # self.decay_stats(self.root_node)
        return new_root
            


    def select(self):
        current_node  = self.root_node
        while(True):
            selected_child = None

            for child in current_node.expanded_children:
                if child.is_terminal:
                    return child

            if not current_node.is_fully_expanded:
                return current_node
            
            # Use max with a key function for efficiency
            selected_child = current_node.expanded_children[np.argmax(current_node.children_cpuct())]
            
            if selected_child is None:
                print("Huh")
                return current_node
            
            current_node = selected_child


    def expand(self, node : Node):
        if node.is_fully_expanded and node is self.root_node:
            return node

        if node.is_terminal:
            raise Exception("Trying to expand disability")
        
        return node.expand_node()
    

    def simulate(self, node : Node):
        if node.is_terminal:
            return node.game.get_result_and_terminated(node.state, node.last_action)[0]
        
        value, last_player = self.random_simulation(node.state, node.current_player)

        if value == 0:
            return value
        if last_player == node.current_player:
            return -value
        else:
            return value
        
        
    def random_simulation(self, state, c_player):
        next_state, action = self.random_play(state, c_player)
        value, terminated = self.game.get_result_and_terminated(next_state, action)
        
        while(not terminated):
            c_player = 1 if c_player == 2 else 2
            next_state, action = self.random_play(next_state, c_player)
            value, terminated = self.game.get_result_and_terminated(next_state, action)

        return value, c_player

    def random_play(self, state, c_player):
        valid_actions = np.flatnonzero(self.game.get_valid_moves(state))
        action = np.random.choice(valid_actions)
        return self.game.get_next_state(state, action, c_player), action
        

    def backprop(self, node : Node, value):
        
        while(node != None):
            node.n_visits += 1
            node.value += value
            value = -value
            if node.parent is None:
                break
            else:
                node = node.parent

    def decay_stats(self, node, factor=0.5):
        node.n_visits = int(node.n_visits * factor)
        node.q_value *= factor
        for child in node.children:
            if child is not None:
                self.decay_stats(child, factor)


    @torch.no_grad
    def run(self, observation, config):

        board = np.array(observation["board"], dtype=np.int8)
        board = board.reshape(self.game.rows, self.game.columns)
        player = observation["mark"]

        # self.init_root(player, state=board, c=self.c)

        if getattr(self, 'root_node', None) is None:
            self.init_root(player, board, c=self.c)
        else:
            if not self.use_subtree(player, board):
                self.init_root(player, board, self.c)
        
        policy = None
        value = None

        node = self.root_node

        for i in range(self.n_loops):

            if node.is_terminal:
                expanded = node
            else:
                if node.is_fully_expanded and not node == self.root_node:
                    print("Let's think")
                expanded = self.expand(node)

            if expanded.is_terminal:
                result = -expanded.game.get_result_and_terminated(expanded.state, expanded.last_action)[0]
                self.backprop(expanded, result)
                node = self.select()
                continue

            policy, value  = self.model(
                torch.tensor(self.game.encode_state(expanded.state, expanded.current_player), dtype=torch.float32, device=self.device).unsqueeze(0)
            )
            policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy() # type: ignore
            policy *= node.valid_moves_bool

            
            if np.sum(policy) > 0:
                prob_dist = policy / np.sum(policy)
            else:
                # Fallback to uniform distribution over valid moves
                prob_dist = np.zeros_like(policy)
                prob_dist[node.valid_moves_bool] = 1.0 / np.sum(node.valid_moves_bool)
                
            value = value.item()

            expanded.prob_dist = prob_dist
            
            self.backprop(expanded, value)

            node = self.select()
        

        visit_counts = np.zeros(self.game.action_size, dtype=np.float32)

        for child in self.root_node.expanded_children:
            if child is not None:
                visit_counts[child.last_action] = child.n_visits

        visit_counts *= self.root_node.valid_moves_bool

        if np.sum(visit_counts) > 0:
            prob_dist = visit_counts / np.sum(visit_counts)
        else:
            prob_dist = self.root_node.valid_moves_bool.astype(np.float32)
            prob_dist /= np.sum(prob_dist)
    
        # Return the action that led to this child
        # logger.warning("Running Fine here")
        return prob_dist, self.root_node.q_value
        



