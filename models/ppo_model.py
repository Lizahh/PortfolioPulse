# models/ppo_model.py
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

class DummyPortfolioEnv(gym.Env):
    def __init__(self):
        super(DummyPortfolioEnv, self).__init__()
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(5,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(5,))
        self.state = np.random.rand(5)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.random.rand(5)
        info = {}  # you can add custom info later if needed
        return self.state, info

    def step(self, action):
        action = np.array(action).flatten()
        reward = np.dot(self.state, action)
        self.state = np.random.rand(5)
        
        terminated = False     # used to signal end of episode (e.g., if account balance hits zero)
        truncated = False      # used for time limit end
        info = {}
        
        return self.state, reward, terminated, truncated, info

def train_model():
    env = DummyPortfolioEnv()
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=5000)
    model.save("models/ppo_portfolio_model")
