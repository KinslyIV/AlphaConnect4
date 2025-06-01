import random
import numpy as np
import torch
from connectx import Game
from mcts import MCTS
from model import ResNet
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import multiprocessing
from functools import partial
import copy
import os


os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)



class AlphaZeroDataset(Dataset):
    def __init__(self, memory, device):
        self.data = memory  # list of (state, policy, value)
        self.device = device

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        state, policy, value = self.data[idx]
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)  # Shape: [C, H, W] or flat vector
        policy_tensor = torch.tensor(policy, dtype=torch.float32, device=self.device)  # Action probabilities
        value_tensor = torch.tensor(value, dtype=torch.float32, device=self.device)
        return state_tensor, policy_tensor, value_tensor
    

class AlphaZero:
    def __init__(self, game : Game, args, config : dict) -> None:
        self.game = game
        self.config = config
        self.args = args
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model = ResNet(game=game, struct=self.args["struct"], device=torch.device('cpu'))
        self.mcts = MCTS(model=model, game=game, n_loops=args['n_mcts_loops'], device=torch.device('cpu'), c=self.args['c'])
        self.net = model
        self.memory = []
        self.lr = self.args['lr']
        self.optimizer = optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-5) # back to e-4
        self.n_self_play = args['n_self_play']
        self.epochs = self.args['epochs']
        self.batch_size = self.args['batch_size']
        self.n_iter = self.args['n_iter']
        self.init_buffer_size = args['init_buffer_size']
        self.temp = args['temp']
        self.n_workers = args['n_workers']
        self.coef = self.args['coef']
        self.coef_it = self.args['coef_change_it']
        self.tau = self.args['tau'] # n_move to use temp
        self.final_buffer_size = self.args['final_buffer_size']
        self.buffer_inc_last_step = self.args['buffer_inc_last_step']
        self.c_coef = 1
        self.buffer_size_inc = self.args['buffer_size_inc']
        self.actual_buffer_size = self.init_buffer_size
        # total_steps = ((self.max_buffer_size +  * self.buffer_size_inc) // self.batch_size)) * self.n_iter * self.epochs
        self.schedular = optim.lr_scheduler.OneCycleLR(self.optimizer,
                                                    max_lr=self.args['max_lr'],      #1e-3,
                                                    total_steps= self.compute_total_steps() ,
                                                    pct_start=0.3,        # 30% ramp-up, 70% cool-down
                                                    anneal_strategy='cos',  # Cosine decay (recommended)
                                                    div_factor=self.args['div_factor_schedular'],       #  25.0,       # Initial LR = max_lr / 25
                                                    final_div_factor=self.args['final_div_factor_schedular']         #1e4  # Final LR = max_lr / final_div_factor
                                                    )
        

    def self_play(self):
        self.net.eval()
        self.net.to(torch.device('cpu'))
        
        # Prepare model state dict for processes
        model_state_dict = {
            'weights': self.net.state_dict(),
            'args': self.args
        }
        
        # Determine number of processes
        n_workers = self.args.get('n_workers', max(1, multiprocessing.cpu_count() - 1))
        print(f"Using {n_workers} workers for parallel self-play")

        play_args = [
        (
            copy.deepcopy(self.game), 
            model_state_dict,
            self.config
        ) for _ in range(self.n_self_play)
        ]
        
        # Execute games in parallel
        with multiprocessing.Pool(processes=n_workers) as pool:
            results = list(tqdm(
                pool.imap_unordered(play_single_game_wrapper, play_args),
                total=int(self.n_self_play),
                desc="Self-play games"
            ))
        
        # Collect all game experiences
        for game_memory in results:
            self.memory.extend(game_memory)

            

    def train(self):
        self.net.to(self.device)
        dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        self.net.train()

        for epoch in range(self.epochs):
            total_policy_loss = 0
            total_value_loss = 0
            total_batches = 0
            
            for states, target_policies, target_values in dataloader:
                self.optimizer.zero_grad()
                
                pred_policies, pred_values = self.net(states)

                # Value loss: MSE between predicted value and final result
                value_loss = F.mse_loss(pred_values, target_values.view(-1, 1))

                # Policy loss: Cross-entropy between MCTS policy and predicted log policy
                pred_log_policies = F.log_softmax(pred_policies, dim=1)
                policy_loss = -(target_policies * pred_log_policies).sum(dim=1).mean()


                # Total loss
                loss = value_loss + policy_loss
                loss.backward()
                self.optimizer.step()
    

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_batches += 1

                self.schedular.step()
            avg_policy_loss = total_policy_loss / total_batches
            avg_value_loss = total_value_loss / total_batches
            total_loss = avg_policy_loss + avg_value_loss
            
            print(f"""Epoch {epoch+1}: Value Loss = {avg_value_loss:.4f}, Policy Loss = {avg_policy_loss:.4f}, Total_loss = {total_loss}, \nLearning_rate = {self.schedular.get_last_lr()[0]: .6f}""")

        

    def learn(self):
        for i in range(self.n_iter):
            print(f"=== Iteration {i+1}/{self.n_iter} ===")
            
            self.self_play()
            # Trim memory if needed
            if len(self.memory) > self.actual_buffer_size:
                # Keep a balanced mix of recent and older experiences
                keep_recent = int(self.actual_buffer_size * 0.7)
                keep_old = self.actual_buffer_size - keep_recent
                old_samples = self.memory[:-keep_recent]
                old_samples = random.sample(old_samples, min(keep_old, len(old_samples)))
                self.memory = old_samples + self.memory[-keep_recent:]
                
            # Create dataset with all accumulated data
            self.dataset = AlphaZeroDataset(self.memory, self.device)

            print(f"\nCurrent Max Bufer size: {self.actual_buffer_size}, Mixing Coef: {self.c_coef}")

            self.train()

            self.actual_buffer_size += self.buffer_size_inc

            if i <= self.buffer_inc_last_step - 1:
                self.actual_buffer_size = int(self.init_buffer_size + (i + 1) * (self.final_buffer_size - self.init_buffer_size / (self.buffer_inc_last_step - 1)))  # 14 steps to reach 20000 from 10000
            else:
                self.actual_buffer_size = self.final_buffer_size


            if i >= self.coef_it and self.c_coef > 0.65 :
                self.c_coef -= 0.05
                self.args['coef'] = self.c_coef

            torch.save(self.net.state_dict(), f"agents/models/check_points/agent_10_{i}.pt")


    def compute_total_steps(self):
        total_steps = 0
        for i in range(1, self.n_iter + 1):
            buffer_size = i * self.buffer_size_inc + self.init_buffer_size
            steps = buffer_size // self.batch_size
            steps *= self.epochs
            total_steps += steps
        return total_steps


def play_single_game_wrapper(args):
    return play_single_game(*args)


def play_single_game(game : Game, model_state_dict, config, device='cpu'):
        """Play a single game and return the replay buffer data."""
        # Initialize resources for this process

        args = model_state_dict['args']

        model = ResNet(
            game=game,
            struct=args['struct'],
            device=torch.device(device)
        )
        model.load_state_dict(model_state_dict['weights'], strict=True)
        model.eval()
        
        mcts = MCTS(model=model, game=game, n_loops=args['n_mcts_loops'], device=torch.device(device), c=args['c'])
        
        replay_buffer = []
        state = game.get_initial_state()
        current_player = 1
        opp = 2
        move_number = 0
        coef = args['coef']
        temp = args['temp']
        tau = args['tau']
        
        while True:
            prob_dist, pred_value = mcts.run({'board': state, 'mark': current_player}, config)
            
            # Safely handle action selection
            if prob_dist is None or np.any(np.isnan(prob_dist)):
                # Fallback to uniform distribution over valid moves
                valid_moves = game.get_valid_moves(state)
                prob_dist = np.zeros_like(prob_dist)
                prob_dist[valid_moves] = 1.0 / np.sum(valid_moves)

            if move_number > tau:
                temp = 0
            
            if temp == 0:
                action = np.argmax(prob_dist)

            else:
                prob_dist = prob_dist ** (1 / temp)
                prob_dist /= np.sum(prob_dist)
                action = np.random.choice(game.action_size, p=prob_dist)

            
            replay_buffer.append({
                'state': game.encode_state(state, current_player),
                'policy': prob_dist,
                'player': current_player,
                'q_value': pred_value
            })

            state = game.get_next_state(state, action, current_player)
            reward, terminated = game.get_result_and_terminated(state, action)


            move_number+=1
            
            if terminated:
                break
                
            current_player = opp
            opp = 3 - current_player

        replay_buffer.append({
                'state': game.encode_state(state, current_player),
                'policy': prob_dist,
                'player': current_player,
                'q_value': pred_value
            })
        
        # Add value information
        for sample in replay_buffer:
            sample_player = sample["player"]
            if sample_player == current_player:
                result = coef * reward + (1-coef)*sample['q_value']
            else: 
                result = coef * -reward + (1-coef)*sample['q_value']
            
            sample["result"] = result
        
        return [(sample["state"], sample["policy"], sample["result"]) for sample in replay_buffer]


