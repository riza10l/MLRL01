import gymnasium as gym
import numpy as np
import pandas as pd # <-- benerin di sini
from stable_baselines3 import PPO
#algo rl

class TradingEnv(gym.Env):
    def __init__(self,df):
        super(TradingEnv,self).__init__()
        self.df=df
        self.current_step = 0
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(low=-np.inf , high=np.inf, shape=(5,), dtype=np.float32)
    def reset(self, seed=None, options=None):
        self.current_step = 0
        return self._get_observation(), {}
    def step(self,action): 
        self.current_step += 1
        reward = 0
        done = self.current_step >= len(self.df)-6 # Biar nggak bablas
        obs = self._get_observation()
        return obs, reward, done, False, {}

    def _get_observation(self): 
        return np.array(self.df.iloc[self.current_step : self.current_step+5]['close'].values, dtype=np.float32)

#load data cuy
df = pd.read_csv("data05.csv")

#nyetup env yuks
env = TradingEnv(df)
model = PPO('MlpPolicy', env, verbose=1)

#"ill keep evolving till i die" ahh machine
print("Sek mas ngilmu dulu....")
model.learn(total_timesteps=10000)

#Save dulu cik model rl nya
model.save("ppo_trading_gold")
print("done cees, dah pinter")
