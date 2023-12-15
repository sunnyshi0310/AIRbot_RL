#!/usr/bin/env python3

import gymnasium as gym
import numpy as np
from envs.AIRbotPlayEnv import AIRbotPlayEnv
from stable_baselines3 import PPO
from threading import Thread


def train(train_id):
    env = AIRbotPlayEnv("./pick_place_configs_isaac.json")
    env.set_id(train_id)
    env.set_total_record(200)  # 总共多少step
    # Train with new policy
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=2000)
    model.save(f"saved_models/PPO_{train_id}")


train_id = 0
total_episodes = 3
for _ in range(total_episodes):
    print(f"Training: {train_id}")
    try:
        train(train_id)
    except:
        train_id += 1

# # Load the existing model
# loaded_model = A2C.load("saved_models/A2C_green",verbose = 1)

# obs, _info = env.reset()
# for i in range(100):
#     action, _state = loaded_model.predict(obs, deterministic=True)
#     obs, reward, done, truncated, info = env.step(int(action))
