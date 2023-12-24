#!/usr/bin/env python3

import gymnasium as gym
from gymnasium import spaces

import rospy
from geometry_msgs.msg import TransformStamped
from std_srvs.srv import Empty

import numpy as np
from airbot_play_control.control import RoboticArmAgent, ChooseGripper
from robot_tools import conversions, transformations, recorder
from threading import Event
from copy import deepcopy


class BuildingBlocksInterface(RoboticArmAgent):
    """搭积木任务的细分操作接口"""

    def configure(self, config):
        """参数配置"""
        if isinstance(config, str):
            config = recorder.json_process(config)
        # Define the pick pose
        self.pick_pose = [
            config["PICK_POSITION_X"],
            config["PICK_POSITION_Y"],
            config["PICK_POSITION_Z"],
            config["PICK_ROLL"],
            config["PICK_PITCH"],
            config["PICK_YAW"],
        ]
        self._pick_grasp_z = config["PICK_GRASP_Z"]
        if config["PICK_JOINT"] == "AUTO":
            self.pick_joint = self.change_pose_to_joints(self.pick_pose)
        else:  # use the pre-defined joint
            self.pick_joint = config["PICK_JOINT"]
        if config["PLACE_POSITION_Z"] == "PICK_GRASP_Z":
            config["PLACE_POSITION_Z"] = self._pick_grasp_z
        self._place_base_z = config["PLACE_POSITION_Z"]
        self.place_xyz = [
            config["PLACE_POSITION_X"],
            config["PLACE_POSITION_Y"],
            config["PLACE_POSITION_Z"],
        ]
        if config["PLACE_ROLL"] == "PICK_ROLL":
            config["PLACE_ROLL"] = config["PICK_ROLL"]
        if config["PLACE_PITCH"] == "PICK_PITCH":
            config["PLACE_PITCH"] = config["PICK_PITCH"]
        if config["PLACE_YAW"] == "PICK_YAW":
            config["PLACE_YAW"] = config["PICK_YAW"]
        self.place_rpy = [
            config["PLACE_ROLL"],
            config["PLACE_PITCH"],
            config["PLACE_YAW"],
        ]
        self._cube_height = config["CUBE_HEIGHT"]
        self._detect_gap_place = 0.003
        self.pick_cnt = 0  # 记录pick动作的次数，也代表着循环的次数
        self.place_cnt = 0  # 记录place动作的次数，也代表着已经放置的积木的数量
        rospy.Subscriber(
            "target_TF", TransformStamped, self._feedback_callback, queue_size=1
        )
        self._pixel_error = None
        self._vision_event = Event()

    def _feedback_callback(self, msg: TransformStamped):
        trans_list = conversions.transform_to_list(msg.transform)
        rpy = transformations.euler_from_quaternion(trans_list[3:7])
        self._pixel_error = (trans_list[0], trans_list[1], rpy[2])
        self._vision_event.set()

    def _set_vision_attention(self, attention=None):
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

    def get_what_see(self, wait=False):
        """
        获取视觉反馈信息:
            wait: 是否等待最新的视觉反馈
        """
        if wait:
            self._vision_event.wait()
        self._vision_event.clear()
        return self._pixel_error

    def go_to_pick_pose(self):
        self.go_to_named_or_joint_target(self.pick_joint, sleep_time=1)

    def pick_detect(self):
        self._set_vision_attention()
        rospy.sleep(0.2)

    def go_down_to_grasp(self):
        self.go_to_single_axis_target(2, self._place_base_z, sleep_time=1)
        self.gripper_control(1, 1.2)
        self.pick_cnt += 1

    def lift_up_to_place_height(self):
        """抬起到放置检测时的高度（该函数调用后将会更新place的相关Z轴位置参数）"""
        delta_z = self._cube_height * self.place_cnt
        self.target_place_z = self._place_base_z + delta_z
        self.detect_place_z = self.target_place_z + self._detect_gap_place
        if self.place_cnt == 0:
            delta_z += self._cube_height + self._detect_gap_place
        elif self.place_cnt == 1:
            delta_z += self._detect_gap_place
        self.go_to_single_axis_target(2, self._place_base_z + delta_z, sleep_time=1)

    def go_to_place_detect_pose(self):
        self.place_detect_xyz = (
            self.place_xyz[0],
            self.place_xyz[1],
            self.detect_place_z,
        )
        self.set_and_go_to_pose_target(self.place_detect_xyz, self.place_rpy, "0", 0.5)

    def pick_detect_over(self):
        """pick动作组合"""
        self._set_vision_attention("pause")
        self.go_down_to_grasp()
        rospy.sleep(0.5)
        self.lift_up_to_place_height()
        rospy.sleep(0.5)
        self.go_to_place_detect_pose()

    def place_detect(self):
        self._set_vision_attention()
        rospy.sleep(0.2)

    def go_down_to_place(self):
        self.go_to_single_axis_target(2, self.target_place_z, sleep_time=1)
        self.gripper_control(0, 1.2)
        self.place_cnt += 1

    def go_up_to_avoid_collision(self):
        pose1 = deepcopy(self.last_target_pose.pose)
        pose1.position.z = self.last_xyz[2] + self._cube_height + 0.01  # 先上升
        pose2 = deepcopy(pose1)
        delta_y = 0.075
        # 然后向右/左移动一段距离(取决于在哪侧搭建)
        if self.place_xyz[1] > self.pick_pose[1]:
            pose2.position.y = self.last_xyz[1] - delta_y
        else:
            pose2.position.y = self.last_xyz[1] + delta_y
        self.set_and_go_to_way_points([pose1, pose2], allowed_fraction=0.6)

    def place_detect_over(self):
        """place动作组合"""
        self._set_vision_attention("pause")
        self.go_down_to_place()
        rospy.sleep(0.5)
        self.go_up_to_avoid_collision()
        rospy.sleep(0.5)

    def test(self, episodes=10):
        """动作循环测试"""
        for _ in range(episodes):
            print("pick_cnt:", self.pick_cnt)
            print("current_stage—1:", self.get_current_stage())
            self.go_to_pick_pose(), print("go_to_pick_pose_finish")
            self.pick_detect(), print("pick_detect_finish")
            self.pick_detect_over(), print("pick_detect_over_finish")
            print("current_stage-2:", self.get_current_stage())
            self.place_detect(), print("place_detect_finish")
            self.place_detect_over(), print("place_detect_over_finish")
            print("place_cnt:", self.place_cnt)
            print(" ")
        print("test finished")


class BuildingBlocksEnv(gym.Env):
    def __init__(self, config_path):
        # Define ros related itemss
        NODE_NAME = "AIRbot"
        rospy.init_node(NODE_NAME, anonymous=True)
        rospy.Subscriber(
            "target_TF", TransformStamped, self._feedback_callback, queue_size=1
        )
        # 读取配置文件
        config = recorder.json_process(config_path)
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
                recorder.json_process(
                    f"./data_real_{self._id}.json", write=self._recorder
                )
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
    NODE_NAME = "BuildingBlocksInterfaceTest"
    rospy.init_node(NODE_NAME)

    config = recorder.json_process("./pick_place_configs_isaac.json")
    sim_type = config["SIM_TYPE"]
    # 选择夹爪类型
    if "gazebo" in sim_type:
        sim_type_g = "gazebo"
    else:
        sim_type_g = sim_type
    gripper_control = ChooseGripper(sim_type=sim_type_g)()

    # Use Moveit to control the robot arm
    other_config = None if sim_type_g != "gazbeo" else ("", "airbot_play_arm")

    bbi = BuildingBlocksInterface(
        node_name=NODE_NAME,
        gripper=(4, gripper_control),
        other_config=other_config,
    )
    bbi.configure(config)
    bbi.test(6)
