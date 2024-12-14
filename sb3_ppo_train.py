# %%
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from gymnasium.wrappers import TimeLimit
from environment import CustomPID
import torch
import numpy as np
import random
import os

os.chdir(os.path.dirname(os.path.realpath(__file__)))
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# 初始化环境
env = CustomPID()
max_episode_steps = env.max_episode_steps
env = DummyVecEnv([lambda: Monitor(TimeLimit(env, max_episode_steps))])

# 创建 PPO 模型
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    batch_size=256,
    n_steps=2048,            # 每次梯度更新使用的步数
    gamma=0.99,              # 折扣因子
    gae_lambda=0.95,         # 广义优势估计的衰减因子
    clip_range=0.2,          # PPO 剪切范围
    ent_coef=0.01            # 控制熵正则化项
)

# 评估回调
eval_callback = EvalCallback(
    env,
    best_model_save_path='./logs_ppo/',
    log_path='./logs_ppo/',
    eval_freq=max_episode_steps * 5,
    deterministic=True,
    render=False
)

# 开始训练
model.learn(total_timesteps=5000000, callback=eval_callback)
