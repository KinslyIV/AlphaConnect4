from stable_baselines3 import DQN
from environment import ConnectX


env = ConnectX()
model = DQN(
    policy="MlpPolicy",
    env=env,
    batch_size=128,
    train_freq=1,
    gamma=0.99,
    learning_rate=3e-4,
    buffer_size=100000,
    learning_starts=2000, 
    exploration_fraction=0.3,  # Added: explore more aggressively
    exploration_final_eps=0.05, # Added: maintain some exploration
    policy_kwargs=dict(net_arch=[256, 256, 128]),
    verbose=1,
    target_update_interval=150,
    device='cuda'
)

model.learn(total_timesteps=int(1e6), progress_bar=True, log_interval=1000)

model_name = 'connectx_3'
model.save(model_name)