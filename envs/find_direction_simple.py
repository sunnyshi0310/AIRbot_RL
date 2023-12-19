#!/usr/bin/env python3

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import time


class FindDirection(gym.Env):
    def __init__(self):
        # self.action_space = spaces.Discrete(6)
        # self._action_to_direction = {
        #     0: np.array([1, 0, 0]),
        #     1: np.array([-1, 0, 0]),
        #     2: np.array([0, 1, 0]),
        #     3: np.array([0, -1, 0]),
        #     4: np.array([0, 0, 1]),
        #     5: np.array([0, 0, -1]),
        # }

        self.action_space = spaces.Discrete(27)
        values = [0, 1, -1]
        self._action_to_direction = {}
        cnt = 0
        for i in values:
            for j in values:
                for k in values:
                    self._action_to_direction[cnt] = np.array([i, j, k])
                    cnt += 1  # 所有可能方向的排列组合

        self.obs_range = np.array([[-10, -10, -10], [10, 10, 10]])
        self.observation_space = spaces.Box(
            low=self.obs_range[0],
            high=self.obs_range[1],
            dtype=np.double,
        )

        self._recorder = {
            "time": [],
            "observation": [],
            "action": [],
            "reward": [],
            "num": 0,
        }
        self._time_base = time.time()
        self._stable_times = 0
        self._target_stable_num = 1
        self._steps_per_episode = 0

    def step(self, action):
        direction = self._action_to_direction[int(action)]
        self.current_pose += direction

        # 区域限制
        obs_range = self.get_observation_range()
        self.current_pose = np.clip(self.current_pose, obs_range[0], obs_range[1])
        # Get observation (bias: target_pose - current_pose)
        observation = self._get_obs()

        test = False
        if test:
            print("target_pose", self.target_pose)
            print("last_pose", self.last_pose)
            print("last_observation", self.last_observation)
            print("move_direction", direction)
            print("current_pose", self.current_pose)
            print("current_observation", observation)

        terminated = False
        if np.linalg.norm(observation) < np.linalg.norm(self.last_observation):
            reward = 1 * (100 - np.linalg.norm(observation))
        else:
            reward = -100
            print("not perfect")
        if np.linalg.norm(observation) < 1:
            # print("attempt times:", self._steps_per_episode)
            reward += 1000
            terminated = True
        if test:
            print("reward", reward)
            print(" ")
            time.sleep(1)

        # print("reward", reward)
        self.last_observation = observation.copy()
        self.last_pose = self.current_pose.copy()
        # record the steps
        self._recorder["num"] += 1
        self._steps_per_episode += 1
        # print("num:", self._recorder["num"])

        truncated = False
        info = {}

        return observation, reward, terminated, truncated, info

    def _get_obs(self):
        return self.target_pose - self.current_pose

    def get_observation_range(self):
        return self.obs_range

    def go_to_init_pose(self):
        self.current_pose = self.init_pose.copy()

    def pose_check(self, pose):
        if (pose < self.obs_range[0]).any() or (pose > self.obs_range[1]).any():
            return False
        return True

    def set_init_pose(self, init_pose: np.ndarray):
        self.pose_check(init_pose)
        self.init_pose = init_pose

    def set_target_pose(self, target_pose: np.ndarray):
        self.pose_check(target_pose)
        self.target_pose = target_pose

    def set_id(self, id):
        self._id = id

    def set_total_record(self, total_record):
        self._total_record = total_record

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # print("Reset")
        self._steps_per_episode = 0
        self.set_init_pose(np.random.randint(-10, 10, size=3))
        self.set_target_pose(np.random.randint(-10, 10, size=3))
        self.go_to_init_pose()
        observation = self._get_obs()
        self.last_observation = observation.copy()
        self.last_pose = self.current_pose.copy()
        info = {}
        return observation, info

    def render(self):
        pass


if __name__ == "__main__":
    from stable_baselines3 import PPO
    from stable_baselines3.common.evaluation import evaluate_policy
    import os

    def init_env():
        env = FindDirection()
        env.set_id(train_id)
        env.set_total_record(0)  # 总共多少step
        env.set_init_pose(np.random.randint(-10, 10, size=3))
        env.set_target_pose(np.random.randint(-10, 10, size=3))
        return env

    def train(train_id):
        env = init_env()
        # model = PPO.load(f"saved_models/PPO_find_dirction{train_id}", env=env)
        model = PPO("MlpPolicy", env, verbose=1)
        model.learn(total_timesteps=1e5)
        model.save(f"saved_models/PPO_find_dirction{train_id}")
        env.close()
        return model

    def evaluate(train_id):
        models_dir = "saved_models"
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        env = init_env()
        model = PPO.load(f"saved_models/PPO_find_dirction{train_id}", env=env)
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
    evaluate(train_id)
