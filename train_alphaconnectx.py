import multiprocessing
import torch
from alpha_zero import AlphaZero
from connectx import Game


def main():
    agent = AlphaZero(Game(), {
                'struct': [64 for _ in range(6)], # Shape of inner layer of model
                'n_mcts_loops': 300,
                'n_self_play': 150,
                'epochs': 10,
                'batch_size': 64,
                'lr':1e-4,
                'n_iter': 20,
                'temp': 1,
                'init_buffer_size': 10000, # The initial max buffer size
                'n_workers': 11,
                'c': 3, # c for the mcts
                'coef': 1, # The mixing coeficient used to mix end game result and root qvalue to get value target 
                'coef_change_it':10, # iteration step at which mixing starts
                'tau': 10, # n_moves to use temp 
                'min_coef': 0.5, # not used
                'buffer_size_inc': 500, # amount to increment the max buffer size every iteration
                'final_buffer_size': 20000,
                'buffer_inc_last_step': 15,
                'max_lr': 5e-4, # one cycle max lr
                'div_factor_schedular': 10, #
                'final_div_factor_schedular': 1e3, 

    },  
                    config = {'rows': 6, 'cols': 7, 'inarow': 4}
    )

    agent.learn()
    # torch.save(, "alphazero_model.pth")
    torch.save(agent.net, "agents/models/agent_1002.pth")



if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()