# %%
import matplotlib.pyplot as plt
from scipy.signal import TransferFunction, lsim
import numpy as np
from scipy.signal import lsim, TransferFunction
import gymnasium
from gymnasium import spaces


class CustomPID(gymnasium.Env):
    def __init__(self, time_step=0.01, sim_time=1, setpoint=1):
        """
        Constructor for the CustomPID environment.

        Parameters
        ----------
        time_step : float
            The time step of the simulation.
        sim_time : float
            The total simulation time.
        setpoint : float
            The target value that the system should reach.
        """
        super(CustomPID, self).__init__()
        self.setpoint = setpoint
        self.time_step = time_step
        self.sim_time = sim_time
        self.max_episode_steps = int(sim_time/time_step)
        self.t = np.linspace(0, self.sim_time, self.max_episode_steps+1)
        self.y = None  # 系统输出 每个时刻的输出
        self.u = None  # 系统输入 每个时刻的输入

        num1 = [0.02 * 0.005 * 903]
        den1 = [1/3600, 0.9/60, 1]
        num2 = [1]
        den2 = [1, 0]
        num3 = [1]
        den3 = [1/5519, 0.4/74.29, 1]
        num = np.polymul(num1, num2)
        num = np.polymul(num, num3)
        den = np.polymul(den1, den2)
        den = np.polymul(den, den3)
        self.system = TransferFunction(num, den)

        self.action_space = spaces.Box(low=np.array([0, 0, 0]), high=np.array([100, 1, 1]), shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)

        self.previous_error = None  # 前一个误差
        self.integral = None  # 积分项
        self.step_num = None
        self.previous_action = None

    def _scale_reward(self, reward, min_value, max_value):
        '''
            归一化reward
        '''
        reward = np.clip(reward, min_value, max_value)
        return (reward - min_value) / (max_value - min_value)

    def get_observation(self):
        return np.array([self.y[self.step_num], self.setpoint - self.y[self.step_num]])

    def reset(self, *args, **kwargs):
        self.step_num = 0
        self.integral = 0
        self.previous_error = 0
        self.previous_action = []

        self.y = np.zeros_like(self.t)
        self.u = np.zeros_like(self.t)
        self.y[0] = np.random.uniform(0, .01*self.setpoint)  # 初始值添加一点噪音
        return self.get_observation()

    def step(self, action):
        self.step_num += 1
        # 计算 PID 输出
        error = self.setpoint - self.y[self.step_num-1]
        self.integral += error * self.time_step
        derivative = (error - self.previous_error) / self.time_step
        self.u[self.step_num] = action[0] * error + action[1] * self.integral + action[2] * derivative
        self.previous_error = error

        # 更新系统当前值
        _, y_response, _ = lsim(self.system, U=self.u[: self.step_num + 1], T=self.t[: self.step_num + 1])
        self.y[self.step_num] = y_response[-1]  # 获取当前时刻的输出值

        # 计算误差奖励 误差越小奖励越大
        r_error = self._scale_reward(-abs(self.y[self.step_num] - self.setpoint), -self.setpoint, 0)
        if abs(self.y[self.step_num] - self.setpoint) < 0.05:
            r_error = 1

        # 计算贴近目标奖励
        # r_attach = 0
        # if abs(self.y[self.step_num] - self.setpoint) < 0.1:  # 达到目标值的阈值时,步数越小,奖励越大
        #     r_attach = self._scale_reward(-self.step_num, -self.max_episode_steps, 0)
        # 平滑action奖励
        # action_reward = 0
        # if len(self.previous_action) > 0:
        #     action_dt = np.abs(np.array(action) - np.array(self.previous_action[-1]))/self.action_space.high
        #     action_reward = self._scale_reward(-np.mean(action_dt), -1, 0)
        self.previous_action.append(action)
        # reward = .5*r_error + .4*r_attach + .1*action_reward
        reward = r_error

        terminated = False
        # 超调则结束 则结束
        if self.y[self.step_num]/self.setpoint > 1.2:
            terminated = True
            reward = 0

        truncated = self.step_num >= self.max_episode_steps
        info = {
            "TimeLimit.truncated": truncated,
            "terminated": terminated
        }
        return self.get_observation(), reward, terminated, truncated, info

    def convert_action(self, action):
        '''
            -1~1 的动作转换为实际值
        '''
        action = np.clip(action, -1, 1)
        action = np.array(action)
        action = (action+1)/2
        return action * self.action_space.high


if __name__ == '__main__':

    env = CustomPID()
    env.reset()
    total_reward = 0
    terminated = False
    while not terminated:
        action = [27, .01, 0]
        obs, reward, terminated, truncated, info = env.step(action)
        # print(obs, reward, terminated, truncated, info)
        total_reward += reward
        if terminated or truncated:
            break

    print(total_reward)
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