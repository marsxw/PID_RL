# %%
import matplotlib.pyplot as plt
import numpy as np
import torch
import random
from stable_baselines3 import SAC
from environment import CustomPID
import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


env = CustomPID()
# 绘制训练过程中的结果
data = np.load('logs_sac/evaluations.npz')
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
# plt.ylabel('ep_lengths')
# plt.grid(True)
# plt.show()

# 测试训练好的模型
model = SAC.load('logs_sac/best_model.zip')
obs, terminated, total_reward = env.reset(), False, 0
for i in range(1, env.max_episode_steps+1):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    if terminated or truncated:
        break
print("total_reward", total_reward)

plt.plot(env.t[:env.step_num], env.y[:env.step_num], label='y')
plt.plot(env.t[:env.step_num], np.ones(env.step_num) * env.setpoint, '--', label='setpoint')
plt.xlabel('t')
plt.ylabel('y')
plt.title('pid')
plt.grid(True)
plt.legend()
plt.xlim(0, env.sim_time)
plt.show()
# %%