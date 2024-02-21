#!/usr/bin/env python3
from env_config import Observation, Action, Environment
import numpy as np
from gymnasium import spaces


class ObsSim2Real(Observation):

    def config1(self):
        self.range = np.array([[-1280, -720], [1280, 720]])
        self.space = spaces.Box(
            low=self.range[0],
            high=self.range[1],
            dtype=np.integer,
        )


class ActionSim2Real(Action):

    def config1(self):
        """sim"""
        self.range = np.array([[-1, -1], [1, 1]])
        self.space = spaces.Box(
            low=np.array(self.range[0]),
            high=np.array(self.range[1]),
            dtype=np.intc,  # 但实际上仍是float
        )
        real_cmd_range = np.array([[-10, -10], [10, 10]])

        def sim2real(action):
            return real_cmd_range[0] + (action - self.range[0]) / (
                self.range[1] - self.range[0]
            ) * (real_cmd_range[1] - real_cmd_range[0])

    def config2(self):
        """real"""
        self.config1()
        real_cmd_range = np.array([[-10, -10], [10, 10]])

# class EnvSim2Real(Environment):

#     def __init__(
#         self, observation_config: ObsSim2Real, action_config: ActionSim2Real
#     ) -> None:
#         super().__init__(observation_config, action_config)
