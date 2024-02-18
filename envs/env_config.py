#!/usr/bin/env python3
from robot_tools.trajer import TrajsRecorder
from typing import List
from copy import deepcopy

"""环境配置基类，用于方便多样化环境设置与调整."""

class Observation(object):
    def __init__(self, config) -> None:
        self.range = None
        self.space = None
        self.id = config
        eval(f"self.config{config}")()

    def config1(self):
        raise NotImplementedError

    def convert(self, observation, *args, **kwargs):
        """Convert observation to the format used by RL."""
        return observation

class Action(object):
    def __init__(self, config) -> None:
        self.space = None
        self.id = config
        eval(f"self.config{config}")()

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
            ["observation", "action", "reward"], path + f"_{experiment_id}.json",
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
        return self._tr is not None

    @property
    def cur_obs(self):
        """Current observation before action."""
        return self._current_observation

    @property.setter
    def cur_obs(self, observation):
        self._last_observation = deepcopy(self._current_observation)
        self._current_observation = observation

    @property
    def cur_act(self):
        """Current action that will be applied."""
        return self._current_action

    @property.setter
    def cur_act(self, action):
        self._current_action = action

    @property
    def cur_rwd(self):
        """Current reward after action."""
        return self._current_reward

    @property.setter
    def cur_rwd(self, reward):
        self._current_reward = reward

    @property
    def cur_traj_id(self):
        """Current trajectory id."""
        return self._current_traj_id

    @property.setter
    def cur_traj_id(self, traj_id):
        self._current_traj_id = traj_id

    @property
    def last_obs(self):
        """Last observation relative to cur_obs."""
        return self._last_observation


if __name__ == "__main__":
    import gymnasium as gym

    class Example(Environment):

        def __init__(self, obs_cfg, act_cfg):
            super().__init__(obs_cfg, act_cfg)
