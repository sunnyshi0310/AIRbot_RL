#!/usr/bin/env python3

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import time


class ObservationConfig(object):
    def __init__(self, config) -> None:
        eval(f"self.config{config}")()

    def config1(self):
        self.obs_range = np.array([[-5, -5, -3], [5, 5, 3]])
        self.observation_space = spaces.Box(
            low=self.obs_range[0],
            high=self.obs_range[1],
            dtype=np.int0,
        )

    def config2(self):
        self.obs_range = np.array([[-5, -5, -3], [5, 5, 3]])
        self.observation_space = spaces.Box(
            low=self.obs_range[0],
            high=self.obs_range[1],
            dtype=np.double,
        )


class ActionConfig(object):
    def __init__(self, config) -> None:
        eval(f"self.config{config}")()

    def config1(self):
        self.action_space = spaces.Discrete(27)
        values = [0, 1, -1]
        self.action_to_direction = {}
        cnt = 0
        for i in values:
            for j in values:
                for k in values:
                    self.action_to_direction[cnt] = np.array([i, j, k])
                    cnt += 1  # 所有可能方向的排列组合
        self._action_to_direction = lambda action: self.action_to_direction[action]

    def config2(self):
        self.action_space = spaces.Discrete(6)
        self.action_to_direction = {
            0: np.array([1, 0, 0]),
            1: np.array([-1, 0, 0]),
            2: np.array([0, 1, 0]),
            3: np.array([0, -1, 0]),
            4: np.array([0, 0, 1]),
            5: np.array([0, 0, -1]),
        }
        self._action_to_direction = lambda action: self.action_to_direction[action]

    def config3(self):
        self.action_space = spaces.Discrete(7)
        self.action_to_direction = {
            0: np.array([1, 0, 0]),
            1: np.array([-1, 0, 0]),
            2: np.array([0, 1, 0]),
            3: np.array([0, -1, 0]),
            4: np.array([0, 0, 1]),
            5: np.array([0, 0, -1]),
            6: np.array([0, 0, 0]),
        }
        self._action_to_direction = lambda action: self.action_to_direction[action]

    def config4(self):
        self.action_space = spaces.Box(
            low=np.array([-1, -1, -1]),
            high=np.array([1, 1, 1]),
            dtype=np.intc,  # 但实际上仍是float
        )
        self.action_to_direction = None
        # action_round = np.round(action).astype(np.int8)

    def _action_to_direction(self, action):
        return action


class FindDirection(gym.Env):
    def __init__(self, observation_config=1, action_config=1):
        self.action_space = spaces.Discrete(6)
        self._action_to_direction = {
            0: np.array([1, 0, 0]),
            1: np.array([-1, 0, 0]),
            2: np.array([0, 1, 0]),
            3: np.array([0, -1, 0]),
            4: np.array([0, 0, 1]),
            5: np.array([0, 0, -1]),
        }

        self.obs_range = np.array([[-5, -5, -3], [5, 5, 3]])
        self.observation_space = spaces.Box(
            low=self.obs_range[0],
            high=self.obs_range[1],
            dtype=np.int0,
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

    def get_observation_range(self):
        return self.obs_range

    def go_to_init_pose(self):
        self.current_pose = np.zeros(3)

    def set_target_pose(self, target_pose: np.ndarray):
        if (target_pose < self.obs_range[0]).any() or (
            target_pose > self.obs_range[1]
        ).any():
            raise ValueError("Target pose is out of range")
        self.target_pose = target_pose

    def set_id(self, id):
        self._id = id

    def set_total_record(self, total_record):
        self._total_record = total_record

    def step(self, action):

        direction = self._action_to_direction[int(action)]
        self.current_pose += direction

        # 区域限制
        obs_range = self.get_observation_range()
        self.current_pose = np.clip(self.current_pose, obs_range[0], obs_range[1])
        # Get observation (bias: target_pose - current_pose)
        observation = self._get_obs()

        # Calculate reward
        max_steps = 175
        reward = -np.linalg.norm(observation) * 10

        # print("reward", reward)
        self.last_observation = observation.copy()
        # record the steps
        self._recorder["num"] += 1
        self._steps_per_episode += 1
        # print("num:", self._recorder["num"])

        truncated = False
        info = {}

        terminated = False

        if np.linalg.norm(observation) < 2:
            reward += 1000
            terminated = True
        elif self._steps_per_episode == max_steps:  # threshold
            terminated = True

        return observation, reward, terminated, truncated, info

    def _get_obs(self):
        return self.target_pose - self.current_pose

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        print("Reset")
        self._steps_per_episode = 0
        self.go_to_init_pose()  # 这个位置是不是每次可以随机设置？
        # self.set_target_pose(np.random.randint(-1000, 1000, size=(3,)) / 100)
        observation = self._get_obs()
        self.last_observation = observation.copy()
        info = {}
        return observation, info

    def render(self):
        pass


if __name__ == "__main__":
    from stable_baselines3 import PPO
    from stable_baselines3.common.evaluation import evaluate_policy
    import os

    def init_env():
        env = FindDirection(1, 2)
        env.set_id(train_id)
        env.set_total_record(0)  # 总共多少step
        env.set_target_pose(np.array([4, 4, 2]))
        return env

    def train(train_id):
        models_dir = "saved_models"
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        env = init_env()
        # Train with new policy
        # logdir = ''
        # if not os.path.exists(logdir):
        #     os.makedirs(logdir)
        # model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logdir)
        model = PPO("MlpPolicy", env, verbose=1)
        # model.load(f"saved_models/PPO_find_dirction{train_id}")
        # # 训练之前随机的 policy，可以获得的平均 reward 比较低
        # mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
        # print(f"Before training: mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

        model.learn(total_timesteps=100000)
        model.save(f"{models_dir}/PPO_find_dirction{train_id}")
        # 评估训练后的 policy
        # env.reset()
        # mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
        # print(f"After training: mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

        env.close()

        return model

    def evaluate(train_id):
        models_dir = "saved_models"
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        env = init_env()
        model = PPO("MlpPolicy", env, verbose=1)
        model.load(f"saved_models/PPO_find_dirction{train_id}")
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

    train_id = 3
    total_episodes = 1
    train(train_id)
    # evaluate(train_id)
