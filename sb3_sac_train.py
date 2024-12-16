# %%
from stable_baselines3.common.callbacks import BaseCallback
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import SAC
from environment import CustomPID
from stable_baselines3.common.callbacks import EvalCallback
import torch
import numpy as np
import random
import os
import joblib
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
# 打印网络结构
print("Actor Network (Policy):")
print(model.policy.actor)  # 策略网络
print("\nCritic Network (Q-Value):")
print(model.policy.critic)  # 价值网络
#%%
log_dir = './logs_sac/'

eval_callback = EvalCallback(
    env,
    best_model_save_path=log_dir,
    log_path=log_dir,
    eval_freq=max_episode_steps*5,
    deterministic=True,
    render=False
)


class SaveActionsCallback(BaseCallback):
    def __init__(self, env, save_path, verbose=0):
        super(SaveActionsCallback, self).__init__(verbose)
        self.env = env
        self.save_path = save_path
        self.all_actions = []  # 存储所有回合的动作

    def _on_step(self) -> bool:
        # 获取动作和 done 标志
        actions = self.env.get_attr('actions')[0]
        info = self.env.get_attr('info')[0]
        if info.get("TimeLimit.truncated", True) or info.get("terminated", True):
            self.all_actions.append(actions)

        return True

    def _on_training_end(self) -> None:
        # 保存所有回合的动作
        joblib.dump(self.all_actions, self.save_path)
        print(f"Actions saved to {self.save_path}")


action_callback = SaveActionsCallback(env, save_path=log_dir + "/actions.npy")

model.learn(total_timesteps=50000, callback=[eval_callback, action_callback])

# %%
