#!/usr/bin/env python3

import numpy as np

import tf_conversions
import gymnasium as gym
from gymnasium import spaces
import rospy
from geometry_msgs.msg import TransformStamped
from gazebo_msgs.msg import ModelStates
from std_srvs.srv import Empty
import numpy as np
from airbot_play_control.control import RoboticArmAgent, ChooseGripper


def tf_to_xyzrpy(tf: TransformStamped):
    """将TransformStamped格式的数据转换为xyz和rpy"""
    xyz = [
        tf.transform.translation.x,
        tf.transform.translation.y,
        tf.transform.translation.z,
    ]
    rpy = list(
        tf_conversions.transformations.euler_from_quaternion(
            [
                tf.transform.rotation.x,
                tf.transform.rotation.y,
                tf.transform.rotation.z,
                tf.transform.rotation.w,
            ]
        )
    )
    Obs2D = np.array([xyz[0], xyz[1], rpy[2]])
    return Obs2D


import json


def json_process(file_path, write=None, log=False):
    """读取/写入json文件"""

    if write is not None:
        with open(file_path, "w") as f_obj:
            json.dump(write, f_obj)
        if log:
            print("写入数据为：", write)
    else:
        with open(file_path) as f_obj:
            write = json.load(f_obj)
        if log:
            print("加载数据为：", write)
    return write


class BuildingBlocksInterface(RoboticArmAgent):
    """搭积木任务的细分操作接口"""

    def load_config(self, config):
        """加载配置文件"""
        if isinstance(config, str):
            config = json_process(config)
        # Define the pick pose
        self.pick_pose = [
            config["PICK_POSITION_X"],
            config["PICK_POSITION_Y"],
            config["PICK_POSITION_Z"],
            config["PICK_ROLL"],
            config["PICK_PITCH"],
            config["PICK_YAW"],
        ]
        self._pick_base_z = config["PICK_GRASP_Z"]
        if config["PICK_JOINT"] == "AUTO":
            self.pick_joint = self.change_pose_to_joints(self.pick_pose)
        else:  # use the pre-defined joint
            self.pick_joint = config["PICK_JOINT"]
        if config["PLACE_POSITION_Z"] == "PICK_GRASP_Z":
            config["PLACE_POSITION_Z"] = self._pick_base_z
        self.place_xyz = [
            config["PLACE_POSITION_X"],
            config["PLACE_POSITION_Y"],
            config["PLACE_POSITION_Z"],
        ]
        if config["PLACE_ROLL"] == "PICK_ROLL":
            config["PLACE_ROLL"] = config["PICK_ROLL"]
        if config["PLACE_PITCH"] == "PICK_PITCH":
            config["PLACE_PITCH"] = config["PICK_PITCH"]
        self.place_rpy = [
            config["PLACE_ROLL"],
            config["PLACE_PITCH"],
            config["PLACE_YAW"],
        ]
        self.pick_cnt = 0  # 记录pick动作的次数，也代表着循环的次数
        self.place_cnt = 0  # 记录place动作的次数，也代表着已经放置的积木的数量

    def _set_vision_attention(self, attention = None):
        if attention is None:
            attention = self.get_current_stage()
        elif attention != "pause":
            raise Exception("attention must be None or pause")
        rospy.set_param("/vision_attention", attention)

    def get_current_stage(self):
        """获取当前的状态（不包括pause）"""
        if self.pick_cnt == self.place_cnt:
            if self.pick_cnt % 2 == 0:
                return "pick0"
            else:
                return "pick1"
        elif self.pick_cnt == self.place_cnt + 1:
            if self.place_cnt % 2 == 0:
                return "place0"
            else:
                return "place1"

    def go_to_pick_pose(self):
        self.set_and_go_to_pose_target(
            self.pick_pose, None, "0", 0.5, return_enable=True
        )

    def pick_detect(self):
        self._set_vision_attention("pick0")
        rospy.sleep(0.2)

    def go_down_to_grasp(self):
        self.go_to_single_axis_target(2, self._pick_base_z, sleep_time=1)
        self.gripper_control(1, 1.2)
        self.pick_cnt += 1

    def lift_up_to_place_height(self):
        self.go_to_single_axis_target(2, self._pick_base_z + 0.05, sleep_time=1)

    def go_to_place_pose(self):
        self.set_and_go_to_pose_target(
            self.place_xyz, self.place_rpy, "0", 0.5, return_enable=True
        )

    def pick_detect_over(self):
        """pick动作组合"""
        self._set_vision_attention("pause")
        self.go_down_to_grasp()
        rospy.sleep(0.5)
        self.lift_up_to_place_height()
        rospy.sleep(0.5)
        self.go_to_place_pose()

    def place_detect(self):
        self._set_vision_attention("place0")
        rospy.sleep(0.2)

    def go_down_to_place(self):
        self.go_to_single_axis_target(2, self._pick_base_z, sleep_time=1)
        self.gripper_control(0, 1.2)
        self.place_cnt += 1

    def go_up_to_avoid_collision(self):
        self.go_to_single_axis_target(2, self._pick_base_z + 0.05, sleep_time=1)

    def place_detect_over(self):
        """place动作组合"""
        self._set_vision_attention("pause")
        self.go_down_to_place()
        rospy.sleep(0.5)
        self.go_up_to_avoid_collision()
        rospy.sleep(0.5)

    def test(self):
        """动作循环测试"""
        self.go_to_pick_pose()

        self.pick_detect()
        self.pick_detect_over()
        self.place_detect()
        self.place_detect_over()


class BuildingBlocksEnv(gym.Env):
    def __init__(self, config_path):
        # Define ros related itemss
        NODE_NAME = "AIRbot"
        rospy.init_node(NODE_NAME, anonymous=True)
        rospy.Subscriber(
            "target_TF", TransformStamped, self._feedback_callback, queue_size=1
        )
        # 读取配置文件
        config = json_process(config_path)
        self._sim_type = config["SIM_TYPE"]

        # 选择夹爪类型
        if "gazebo" in self._sim_type:
            sim_type_g = "gazebo"
        else:
            sim_type_g = self._sim_type
        self.gripper_control = ChooseGripper(sim_type=sim_type_g)()

        # Use Moveit to control the robot arm
        other_config = None if sim_type_g != "gazbeo" else ("", "airbot_play_arm")

        self.arm = RoboticArmAgent(
            node_name=NODE_NAME,
            gripper=(4, self.gripper_control),
            other_config=other_config,
        )

        self.obs_range = np.array([[-10, -10, -10], [10, 10, 10]])
        self.observation_space = spaces.Box(
            low=self.obs_range[0],
            high=self.obs_range[1],
            dtype=np.double,
        )

        """
        The following dictionary maps abstract actions from `self.action_space` to
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self.action_space = spaces.Discrete(27)
        values = [0, 1, -1]
        self._action_to_direction = {}
        cnt = 0
        for i in values:
            for j in values:
                for k in values:
                    self._action_to_direction[cnt] = np.array([i, j, k])
                    cnt += 1  # 所有可能方向的排列组合

        self.sleep_time = 2  # make sure the previous action is done
        self.cube_counter = 0

        # count the properly settled cube
        self._pixel_error = TransformStamped()  # initilize the pixel info

        # Define the pick pose
        self.pick_pose = [
            config["PICK_POSITION_X"],
            config["PICK_POSITION_Y"],
            config["PICK_POSITION_Z"],
            config["PICK_ROLL"],
            config["PICK_PITCH"],
            config["PICK_YAW"],
        ]
        self._pick_base_z = config["PICK_GRASP_Z"]
        if config["PICK_JOINT"] == "AUTO":
            self.pick_joint = self.arm.change_pose_to_joints(self.pick_pose)
        else:  # use the pre-defined joint
            self.pick_joint = config["PICK_JOINT"]
        if config["PLACE_POSITION_Z"] == "PICK_GRASP_Z":
            config["PLACE_POSITION_Z"] = self._pick_base_z
        self.place_xyz = [
            config["PLACE_POSITION_X"],
            config["PLACE_POSITION_Y"],
            config["PLACE_POSITION_Z"],
        ]
        if config["PLACE_ROLL"] == "PICK_ROLL":
            config["PLACE_ROLL"] = config["PICK_ROLL"]
        if config["PLACE_PITCH"] == "PICK_PITCH":
            config["PLACE_PITCH"] = config["PICK_PITCH"]
        self.place_rpy = [
            config["PLACE_ROLL"],
            config["PLACE_PITCH"],
            config["PLACE_YAW"],
        ]

        self._kp = config["KP"]
        str_to_bool = lambda x: True if x == "True" else False
        self._not_pick = str_to_bool(config["NOT_PICK"])

        self._recorder = {
            "time": [],
            "observation": [],
            "action": [],
            "reward": [],
            "num": 0,
        }
        self._time_base = rospy.get_time()
        self._no_target_check = 0

    def set_id(self, id):
        self._id = id

    def set_total_record(self, total_record):
        self._total_record = total_record

    def go_to_pick_pose(self):
        self.arm.go_to_named_or_joint_target(self.pick_joint, sleep_time=1)

    def set_vision_attention(self, attention):
        rospy.set_param("/vision_attention", attention)

    def step(self, action):
        print("observation", self.last_observation)
        print("action", action)

        # Take action
        direction = self._action_to_direction[int(action)]
        inc = direction * np.abs(self.last_observation) / self._kp
        inc[2] *= self._kp  # yaw无放缩
        pos_inc = [inc[0], inc[1], 0]
        rot_inc = [0, 0, inc[2]]

        # last表示给定值是基于上次目标值的增量
        self.arm.set_and_go_to_pose_target(
            pos_inc, rot_inc, "last", 0.5, return_enable=True
        )
        # print("pos_inc", pos_inc)
        # print("rot_inc", rot_inc)
        self.no_target = False
        if self.last_pixel_error == self._pixel_error:
            self._no_target_check += 1
            if self._no_target_check > 4:
                print("no target")
                self.no_target = True
                reward = -2000
        else:
            self.last_pixel_error = self._pixel_error
            self._no_target_check = 0

        observation = self._get_obs()  # deviations x, y ,yaw
        # print("observation", observation)
        self.last_observation = observation.copy()

        # Calculate reward
        if np.linalg.norm(observation) < np.linalg.norm(self.last_observation):
            reward = 1 * (100 - np.linalg.norm(observation))
        else:
            reward = -100
            print("not perfect")

        # record the steps
        self._recorder["num"] += 1
        print("num:", self._recorder["num"])
        # record the data
        if self._total_record > 0:
            self._recorder["time"].append(rospy.get_time() - self._time_base)
            self._recorder["observation"].append(self.last_observation.tolist())
            self._recorder["action"].append(inc.tolist())
            self._recorder["reward"].append(reward)
            if self._recorder["num"] == self._total_record:
                json_process(f"./data_real_{self._id}.json", write=self._recorder)
                raise Exception("stop")
        terminated = False
        truncated = False
        info = {}

        if self.no_target or (
            abs(observation[0]) < 6 and abs(observation[1]) < 3 and observation[2] < 0.1
        ):  # threshold
            reward += 1000
            if not self._not_pick:
                print("Start to pick up")
                # start the pick up-action
                # move down the robot arm
                self.arm.go_to_single_axis_target(
                    2, self._pick_base_z, sleep_time=1
                )  # 首先到达可抓取的高度位置(z单轴移动)
                # close the gripper
                self.gripper_control(1, 1.2)
                # lift up the arm
                self.arm.go_to_single_axis_target(
                    2, self._pick_base_z + 0.05, sleep_time=1
                )

                # move to the desired position
                self.arm.set_and_go_to_pose_target(
                    self.place_xyz,
                    self.place_rpy,
                    "0",
                    self.sleep_time,
                    return_enable=True,
                )

                if self.cube_counter == 0:
                    self.set_vision_attention("place0")
                else:
                    self.set_vision_attention("place1")

                # open the gripper
                self.gripper_control(0, 1)
                # lift up the arm
                self.arm.go_to_single_axis_target(
                    2, self._pick_base_z + 0.05, sleep_time=1
                )

                # move back to pick up area
                self.go_to_pick_pose()

                # Define the termination condition
                if self.cube_counter == 0:
                    self.set_vision_attention("pick1")
                else:
                    self.set_vision_attention("pause")
                    terminated = True
                self.cube_counter += 1
            else:
                terminated = True

        return self._adjust_obs(observation), reward, terminated, truncated, info

    def _get_obs(self):
        # deviations x, y ,yaw
        return tf_to_xyzrpy(self._pixel_error)

    def _adjust_obs(self, obs: np.ndarray):
        # adjust the observation from [-2000, 2000] to the range of [-10, 10]
        obs = obs.copy() / 200.0
        obs[2] *= 200.0
        return obs

    def _feedback_callback(self, msg: TransformStamped):
        self._pixel_error = msg

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        print("Reset")
        if self._sim_type != "real":
            self.arm.go_to_named_or_joint_target("Home", sleep_time=0.5)
        # Reset the cubes and robot
        if self._sim_type == "gazebo":
            rospy.wait_for_service("/gazebo/reset_world")
            reset_world = rospy.ServiceProxy("/gazebo/reset_world", Empty)
            reset_world()
        elif self._sim_type == "isaac":
            rospy.set_param("/reset_isaac", True)
        elif self._sim_type == "gibson":
            rospy.set_param("/reset_gibson", True)

        # reset the robot arm to pick pose
        self.go_to_pick_pose()
        # transit to pick mode for vision part
        self.set_vision_attention("pick0")
        rospy.sleep(0.2)
        self.cube_counter = 0
        observation = self._get_obs()
        self.last_observation = observation.copy()
        self.last_pixel_error = self._pixel_error
        info = {}
        return self._adjust_obs(observation), info

    def render(self):
        pass


if __name__ == "__main__":
    pass
    # from stable_baselines3.common.env_checker import check_env

    # env = AIRbotPlayEnv()
    # check_env(env, warn = True)
