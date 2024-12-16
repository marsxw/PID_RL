# %%
import matplotlib.pyplot as plt
from scipy.signal import TransferFunction, lsim
import numpy as np
from scipy.signal import lsim, TransferFunction
import gymnasium
from gymnasium import spaces


class CustomPID(gymnasium.Env):
    def __init__(self, time_step=0.01, sim_time=1):
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
        self.time_step = time_step
        self.sim_time = sim_time
        self.max_episode_steps = int(sim_time/time_step)
        self.t = np.linspace(0, self.sim_time, self.max_episode_steps+1)
        self.y = None  # 系统输出 每个时刻的输出
        self.u = None  # 系统输入 每个时刻的输入
        self.setpoint = None  # 系统目标

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
        self.actions = None
        self.info = {}

    def _scale_reward(self, reward, min_value, max_value):
        '''
            归一化reward
        '''
        reward = np.clip(reward, min_value, max_value)
        return (reward - min_value) / (max_value - min_value)

    def get_observation(self):
        return np.array([self.y[self.step_num], self.setpoint[self.step_num] - self.y[self.step_num]])

    def reset(self,  setpoint=None,  *args, **kwargs):
        self.step_num = 0
        self.integral = 0
        self.previous_error = 0
        self.actions = []
        self.info = {}

        if setpoint is not None:
            self.setpoint = setpoint  # 使用传入的 setpoint
        else:
            # 随机生成阶跃 方波 正弦信号
            # A = np.random.uniform(0, 1)  # 随机目标幅值
            A = 1
            phase = np.random.uniform(0, 2 * np.pi)
            sampel_type = np.random.randint(0, 3)
            if sampel_type == 0:  # 阶跃信号
                self.setpoint = np.random.choice([-1, 1])*A*np.ones_like(self.t)
            elif sampel_type == 1:  # 正弦信号
                freq = 5
                self.setpoint = A * np.sin(2 * np.pi * freq * self.t + phase)
            else:  # 方波信号
                freq = .5  # 周期2
                self.setpoint = np.sin(2 * np.pi * freq * self.t + phase)
                self.setpoint = A * np.where(self.setpoint > 0, 1, -1)

        self.y = np.zeros_like(self.t)
        self.u = np.zeros_like(self.t)
        self.y[0] = np.random.uniform(0, .01*self.setpoint[0])  # 初始值添加一点噪音
        return self.get_observation()

    def step(self, action):
        self.step_num += 1
        self.actions.append(action)

        # 计算 PID 输出
        error = self.setpoint[self.step_num-1] - self.y[self.step_num-1]
        self.integral += error * self.time_step
        derivative = (error - self.previous_error) / self.time_step
        self.u[self.step_num] = action[0] * error + action[1] * self.integral + action[2] * derivative
        self.previous_error = error

        # 更新系统当前值
        _, y_response, _ = lsim(self.system, U=self.u[: self.step_num + 1], T=self.t[: self.step_num + 1])
        self.y[self.step_num] = y_response[-1]  # 获取当前时刻的输出值

        # 计算误差奖励 误差越小奖励越大
        reward = self._scale_reward(-abs(self.y[self.step_num] - self.setpoint[self.step_num]), -self.setpoint[self.step_num], 0)
        if abs(self.y[self.step_num] - self.setpoint[self.step_num-1]) < 0.05:
            reward = 1

        terminated = False
        # 超调则结束 则结束
        if abs(self.y[self.step_num]/self.setpoint[self.step_num-1]) > 1.2:
            terminated = True
            reward = 0

        truncated = self.step_num >= self.max_episode_steps
        self.info = {
            "TimeLimit.truncated": truncated,
            "terminated": terminated
        }
        return self.get_observation(), reward, terminated, truncated, self.info

    def convert_action(self, action):
        '''
            -1~1 的动作转换为实际值
        '''
        action = np.clip(action, -1, 1)
        action = np.array(action)
        action = (action+1)/2
        return action * self.action_space.high


if __name__ == '__main__':

    env = CustomPID(sim_time=2)
    env.reset()
    total_reward = 0
    terminated = False
    while not terminated:
        action = [30, .01, 0]
        obs, reward, terminated, truncated, info = env.step(action)
        # print(obs, reward, terminated, truncated, info)
        total_reward += reward
        if terminated or truncated:
            break

    print(total_reward)
    plt.plot(env.t[:env.step_num], env.y[:env.step_num], label='y')
    plt.plot(env.t[:env.step_num],  env.setpoint[:env.step_num], '--', label='setpoint')
    plt.xlabel('t')
    plt.ylabel('y')
    plt.title('pid')
    plt.grid(True)
    plt.legend()
    plt.xlim(0, env.sim_time)
    plt.show()

# %%
env.y[env.step_num]/env.setpoint[env.step_num]
