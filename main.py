#!/usr/bin/env python3

from envs.AIRbotPlayEnv import AIRbotPlayEnv
from stable_baselines3 import PPO
import os
import numpy as np


# train_id = 3
# total_episodes = 1
# for _ in range(total_episodes):
#     print(f"Training: {train_id}")
#     try:
#         train(train_id)
#     except:
#         train_id += 1

if __name__ == "__main__":

    def init_env():
        env = AIRbotPlayEnv("./pick_place_configs_isaac.json")
        env.set_id(train_id)
        env.set_total_record(0)  # 总共记录多少step
        return env

    def train(train_id):
        env = init_env()
        # model = PPO.load(f"saved_models/PPO_arm{train_id}", env=env)
        model = PPO("MlpPolicy", env, verbose=1)
        model.learn(total_timesteps=1e5)
        model.save(f"saved_models/PPO_arm{train_id}")
        env.close()
        return model

    def evaluate(train_id, model_path=None):
        models_dir = "saved_models"
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        env = init_env()
        if model_path is None:
            model = PPO.load(f"saved_models/PPO_arm{train_id}", env=env)
        else:
            model = PPO.load(model_path, env=env)
        episodes = 10
        all_score = 0
        end_error = 0
        for episode in range(1, episodes + 1):
            obs, info = env.reset()
            done = False
            score = 0
            while not done:
                action, _ = model.predict(obs)  # 使用model来预测动作,返回预测的动作和下一个状态
                last_obs = obs.copy()
                # print(env.current_pose, env.target_pose)
                obs, reward, done, _, info = env.step(action)
                print(last_obs, action, reward)
                # print(reward)
                score += reward
                all_score += score
            end_error += np.linalg.norm(obs)
            print("Episode:{} Score:{}".format(episode, score))
        print("Average score:{}".format(all_score / episodes))
        print("Average end_error:{}".format(end_error / episodes))
        # print("Average step:{}".format(env._total_record / episodes))
        print("Original target error:", np.linalg.norm(env.target_pose))
        env.close()

    train_id = 0
    # train(train_id)
    evaluate(train_id,"saved_models/action27")
