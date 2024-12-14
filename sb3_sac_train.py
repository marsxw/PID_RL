# %%
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import SAC
from environment import CustomPID
from stable_baselines3.common.callbacks import EvalCallback
import torch
import numpy as np
import random
import os
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

os.chdir(os.path.dirname(os.path.realpath(__file__)))
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

env = CustomPID()
max_episode_steps = env.max_episode_steps
env = DummyVecEnv([lambda: Monitor(TimeLimit(env, max_episode_steps))])
model = SAC(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    batch_size=256,
    ent_coef=0.05    # 控制熵正则化项
)

eval_callback = EvalCallback(
    env,
    best_model_save_path='./logs_sac/',
    log_path='./logs_sac/',
    eval_freq=max_episode_steps*5,
    deterministic=True,
    render=False
)

model.learn(total_timesteps=50000, callback=eval_callback)
