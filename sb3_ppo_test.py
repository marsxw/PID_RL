# %%
# %%
import matplotlib.pyplot as plt
import numpy as np
import torch
import random
from stable_baselines3 import PPO
from environment import CustomPID
import os

# 设置当前工作路径
os.chdir(os.path.dirname(os.path.realpath(__file__)))

# 设置随机种子
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# 初始化环境
env = CustomPID()

# 绘制训练过程中的结果
data = np.load('logs_ppo/evaluations.npz')
timesteps = data['timesteps']
results = data['results']
ep_lengths = data['ep_lengths']

plt.plot(timesteps, results[:, 0])
plt.xlabel('Timesteps')
plt.ylabel('Average reward')
plt.grid(True)
plt.show()

# plt.plot(timesteps, ep_lengths[:, 0])
# plt.xlabel('Timesteps')
# plt.ylabel('Episode lengths')将
# plt.grid(True)

# 测试训练好的模型
model = PPO.load('logs_ppo/best_model.zip')
obs, terminated, truncated, total_reward = env.reset(), False, False, 0

for i in range(1, env.max_episode_steps + 1):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    if terminated or truncated:
        break

print("Total reward:", total_reward)

# 绘制系统输出响应曲线
plt.plot(env.t[:env.step_num], env.y[:env.step_num], label='y')
plt.plot(env.t[:env.step_num], np.ones(env.step_num) * env.setpoint, '--', label='setpoint')
plt.xlabel('Time (t)')
plt.ylabel('System output (y)')
plt.title('PID System Response')
plt.grid(True)
plt.legend()
plt.xlim(0, env.sim_time)
plt.show()
# %%
