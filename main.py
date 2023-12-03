#!/usr/bin/env python3

import gymnasium as gym
from envs import AIRbotPlayEnv
from stable_baselines3 import PPO
import numpy as np

env = AIRbotPlayEnv.AIRbotPlayEnv()

# Train with new policy
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=2000)
model.save("saved_models/PPO")

# # Load the existing model
# loaded_model = A2C.load("saved_models/A2C_green",verbose = 1)

# obs, _info = env.reset()
# for i in range(100):
#     action, _state = loaded_model.predict(obs, deterministic=True) 
#     obs, reward, done, truncated, info = env.step(int(action))
    
