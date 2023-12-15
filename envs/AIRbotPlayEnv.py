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
from math import pi


def tf_to_xyzrpy(tf: TransformStamped):
    """将TransformStamped格式的数据转换为xyz和rpy"""
    if not tf:
        # when there are no cubes in the camera, generate random actions to move the camera
        Obs2D = (np.random.rand(3) - 0.5) * 1000.0
        Obs2D[2] = 0  # make sure the yaw angle is valid
    else:
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


class AIRbotPlayEnv(gym.Env):
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

        # Define the observation space and action space
        # Observations are deviations from the correct pose, including x, y and yaw
        self.observation_space = spaces.Box(
            low=np.array([-1000.0, -1000.0, -pi]),
            high=np.array([1000.0, 1000.0, pi]),
            dtype=np.double,
        )
        self.last_observation = np.zeros(3)

        # We have 6 actions, corresponding to "right", "left","up",  "down", "clockwise", "counterclock"
        self.action_space = spaces.Discrete(
            12
        )  # improved later with complicated actions.

        """
        The following dictionary maps abstract actions from `self.action_space` to
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            0: np.array([1, 0, 0]),
            1: np.array([-1, 0, 0]),
            2: np.array([0, 1, 0]),
            3: np.array([0, -1, 0]),
            4: np.array([0, 0, 1]),
            5: np.array([0, 0, -1]),
            6: np.array([0.3, 0, 0]),
            7: np.array([-0.3, 0, 0]),
            8: np.array([0, 0.3, 0]),
            9: np.array([0, -0.3, 0]),
            10: np.array([0, 0, 0.3]),
            11: np.array([0, 0, -0.3]),
        }
        self.step_size = np.array(
            [0.005, 0.005, 0.005]
        )  # action step size on x- y- yaw direction.
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

        self._recorder = {
            "time": [],
            "observation": [],
            "action": [],
            "reward": [],
            "num": 0,
        }
        self._time_base = rospy.get_time()

    def set_id(self, id):
        self._id = id
    
    def set_total_record(self, total_record):
        self._total_record = total_record

    def go_to_pick_pose(self):
        self.arm.go_to_named_or_joint_target(self.pick_joint, sleep_time=1)

    def set_vision_attention(self, attention):
        rospy.set_param("/vision_attention", attention)

    def step(self, action):
        print("action", action)
        # Take action
        direction = self._action_to_direction[action]
        inc = direction * self.step_size
        pos_inc = [inc[0], inc[1], 0]
        rot_inc = [0, 0, inc[2]]
        # last表示给定值是基于上次目标值的增量
        self.arm.set_and_go_to_pose_target(
            pos_inc, rot_inc, "last", self.sleep_time, return_enable=True
        )

        # rospy.sleep(self.sleep_time)
        observation = self._get_obs()  # deviations x, y ,yaw
        print("observation", observation)
        reward = -np.linalg.norm(
            observation
        )  # -(1-self.cube_counter)*500.0  np.linalg.norm(self.last_observation)?
        print("reward", reward)
        self.last_observation = observation.copy()

        # record the data
        self._recorder["time"].append(rospy.get_time() - self._time_base)
        self._recorder["observation"].append(self.last_observation.tolist())
        self._recorder["action"].append(inc.tolist())
        self._recorder["reward"].append(reward)
        self._recorder["num"] += 1
        print("num:", self._recorder["num"])
        if self._recorder["num"] == self._total_record:
            json_process(f"./data_{self._id}.json", write=self._recorder)
            raise Exception("stop")

        terminated = False
        truncated = False
        info = {}

        if (
            abs(observation[0]) < 6 and abs(observation[1]) < 3 and observation[2] < 0.1
        ):  # threshold
            print("Start to pick up")
            # start the pick up-action
            # move down the robot arm
            self.arm.go_to_single_axis_target(
                2, self._pick_base_z, sleep_time=1
            )  # 首先到达可抓取的高度位置(z单轴移动)
            self.gripper_control(1, self.sleep_time)
            # lift up the arm
            self.arm.go_to_single_axis_target(2, self._pick_base_z + 0.05, sleep_time=1)

            # move to the desired position
            self.arm.set_and_go_to_pose_target(
                self.place_xyz, self.place_rpy, "0", self.sleep_time, return_enable=True
            )

            if self.cube_counter == 0:
                self.set_vision_attention("place0")
            else:
                self.set_vision_attention("place1")

            self.gripper_control(0, self.sleep_time)
            # lift up the arm
            self.arm.go_to_single_axis_target(2, self._pick_base_z + 0.05, sleep_time=1)

            # move back to pick up area
            self.arm.set_and_go_to_pose_target(
                self.pick_pose[0:3],
                self.pick_pose[3:6],
                "0",
                self.sleep_time,
                return_enable=True,
            )

            # Define the termination condition
            if self.cube_counter == 0:
                self.set_vision_attention("pick1")
            else:
                self.set_vision_attention("pause")
                terminated = True

            self.cube_counter += 1

        return observation, reward, terminated, truncated, info

    def _get_obs(self):
        # deviations x, y ,yaw
        return tf_to_xyzrpy(self._pixel_error)

    def _feedback_callback(self, msg: TransformStamped):
        self._pixel_error = msg

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        print("Reset")
        self.arm.go_to_named_or_joint_target("Home", sleep_time=0.5)
        # Reset the cubes and robot
        if self._sim_type == "gazebo":
            rospy.wait_for_service("/gazebo/reset_world")
            reset_world = rospy.ServiceProxy("/gazebo/reset_world", Empty)
            reset_world()
        elif self._sim_type == "isaac":
            rospy.set_param("/reset_isaac", True)
        else:
            rospy.set_param("/reset_gibson", True)

        # reset the robot arm to pick pose
        self.go_to_pick_pose()
        # transit to pick mode for vision part
        self.set_vision_attention("pick0")
        rospy.sleep(0.2)
        self.cube_counter = 0
        observation = self._get_obs()
        self.last_observation = observation.copy()
        info = {}
        return observation, info

    def render(self):
        pass


# from stable_baselines3.common.env_checker import check_env

# env = AIRbotPlayEnv()
# check_env(env, warn = True)
