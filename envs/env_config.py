#!/usr/bin/env python3
from robot_tools.trajer import TrajsRecorder
from typing import List
from copy import deepcopy
from gymnasium import spaces
import numpy as np

"""环境配置基类，用于方便多样化环境设置与调整."""


class IOBase(object):
    def __init__(self, config) -> None:
        self._id = config
        self._space = None
        self._range = None

    def configure(self, config=None):
        if config is None:
            config = self.id
        eval(f"self.config{config}()")

    @property
    def space(self):
        return self._space

    @space.setter
    def space(self, space):
        self._space = space

    @property
    def range(self):
        return self._range

    @range.setter
    def range(self, range: np.ndarray):
        self._range = range

    @property
    def id(self):
        return self._id


class Observation(IOBase):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.configure()

    def config1(self):
        raise NotImplementedError

    def convert(self, observation, *args, **kwargs):
        """Convert observation to the format used by RL."""
        return observation


class Action(IOBase):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.configure()

    def config1(self):
        raise NotImplementedError

    def convert(self, action, *args, **kwargs):
        """Convert action to the format used by robot."""
        return action


class Environment(object):
    def __init__(self, observation_config: Observation, action_config: Action) -> None:
        self.obs_cfg = observation_config
        self.act_cfg = action_config
        self._current_observation = None
        self._current_action = None
        self._current_reward = None
        self._current_traj_id = None
        self._last_observation = None
        self._tr = None
        # SB3必须要求的属性
        self.observation_space = self.obs_cfg.space
        self.action_space = self.act_cfg.space

    def record_start(self, path, experiment_id, max_record_steps, end_exit=False):
        """
        Start recording the trajectory which will be saved to path_experiment_id.json.
        Args:
            path: The path to save the record data.
            experiment_id: The id of the experiment.
            max_record_steps: The total record steps of the experiment. If <= 0, not record.
            end_exit: Whether to exit after the record is finished.
        """
        if max_record_steps <= 0:
            return
        self._tr = TrajsRecorder(
            ["observation", "action", "reward"],
            path + f"_{experiment_id}.json",
        )
        self._experiment_id = experiment_id
        self._max_record_steps = max_record_steps
        self._end_exit = end_exit

    def record_add_new(self, features_name: List[str]):
        """Add new features name to the record."""
        self._tr.add_new_features(features_name)
        # 将特征名创建成类的属性
        for name in features_name:
            setattr(self, f"cur_{name}", None)

    def record_add_cur(self, debug=False) -> bool:
        """Add current features data to the record."""
        if self._tr is None:
            return False
        if debug:
            assert self._current_observation is not None
            assert self._current_action is not None
            assert self._current_reward is not None
            assert self._current_traj_id is not None
            print(
                f"obs: {self._current_observation}, act: {self._current_action}, rwd: {self._current_reward}"
            )
        self._tr.features_add(
            self._current_observation, self._current_action, self._current_reward
        )
        if debug:
            self._current_observation = None
            self._current_action = None
            self._current_reward = None
            self._current_traj_id = None
        if self._tr.features_add_cnt == self._max_record_steps:
            self._tr.save()
            if self._end_exit:
                exit("Record finished.")
        return True

    def record_save(self):
        """Save the recorded data."""
        self._tr.save()

    def record_working(self):
        """Whether the record has been started."""
        return self._tr is not None

    @property
    def cur_obs(self):
        """Current observation before action."""
        return self._current_observation

    @cur_obs.setter
    def cur_obs(self, observation):
        self._last_observation = deepcopy(self._current_observation)
        self._current_observation = observation

    @property
    def cur_act(self):
        """Current action that will be applied."""
        return self._current_action

    @cur_act.setter
    def cur_act(self, action):
        self._current_action = action

    @property
    def cur_rwd(self):
        """Current reward after action."""
        return self._current_reward

    @cur_rwd.setter
    def cur_rwd(self, reward):
        self._current_reward = reward

    @property
    def cur_traj_id(self):
        """Current trajectory id."""
        return self._current_traj_id

    @cur_traj_id.setter
    def cur_traj_id(self, traj_id):
        self._current_traj_id = traj_id

    @property
    def last_obs(self):
        """Last observation relative to cur_obs."""
        return self._last_observation


if __name__ == "__main__":

    class ObsCustom(Observation):
        def config1(self):
            self.range = np.array([[-5, -5, -3], [5, 5, 3]])
            self.space = spaces.Box(
                low=self.range[0],
                high=self.range[1],
                dtype=np.integer,
            )

    class ActionCustom(Action):
        def config1(self):
            self.space = spaces.Discrete(27)
            values = [0, 1, -1]
            self._action_to_direction = {}
            cnt = 0
            for i in values:
                for j in values:
                    for k in values:
                        self._action_to_direction[cnt] = np.array([i, j, k])
                        cnt += 1  # 所有可能方向的排列组合
            self.convert = lambda action: self._action_to_direction[action]

    class EnvCustom(Environment):
        def __init__(self, obs_cfg, act_cfg):
            super().__init__(obs_cfg, act_cfg)
            print("Observation space:", self.obs_cfg.space)
            print("Action space:", self.act_cfg.space)

    env = EnvCustom(ObsCustom(1), ActionCustom(1))
