#!/usr/bin/env python3

import gymnasium as gym
import numpy as np
from threading import Event
from copy import deepcopy

import rospy
from geometry_msgs.msg import TransformStamped
from std_srvs.srv import Empty

from airbot_play_control.control import RoboticArmAgent, ChooseGripper
from robot_tools import conversions, transformations, recorder
from env_config_sim2real import ObsSim2Real, ActionSim2Real, Environment


class BuildingBlocksInterface(RoboticArmAgent):
    """搭积木任务的细分操作接口"""

    def configure(self, config):
        """参数配置"""
        if isinstance(config, str):
            config = recorder.json_process(config)
        # Define the pick pose
        self.pick_pose = config["PICK_POSITION"] + config["PICK_ORIENTATION"]
        self._pick_grasp_z = config["PICK_GRASP_Z"]
        if config["PICK_JOINT"] == "AUTO":
            self.pick_joint = self.change_pose_to_joints(self.pick_pose)
        else:  # use the pre-defined joint
            self.pick_joint = config["PICK_JOINT"]
        if config["PLACE_POSITION"][2] == "PICK_GRASP_Z":
            config["PLACE_POSITION"][2] = self._pick_grasp_z
        self._place_base_z = config["PLACE_POSITION"][2]
        self.place_xyz = config["PLACE_POSITION"]
        names = ["PICK_ROLL", "PICK_PITCH", "PICK_YAW"]
        for i in range(3):
            if config["PLACE_ORIENTATION"][i] == names[i]:
                config["PLACE_ORIENTATION"][i] = config["PICK_ORIENTATION"][i]
        self.place_rpy = config["PLACE_ORIENTATION"]
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

    def get_what_see(self, wait=0):
        """
        获取视觉反馈信息:
            wait: 是否等待最新的视觉反馈
        """
        if wait != 0:
            if wait > 0:
                self._vision_event.wait(wait)
            else:
                self._vision_event.wait()
        self._vision_event.clear()
        return self._pixel_error

    def go_to_pick_pose(self):
        self.go_to_named_or_joint_target(self.pick_joint, sleep_time=1)

    def detect(self):
        self._set_vision_attention()
        rospy.sleep(0.2)

    def go_down_to_grasp(self):
        self.go_to_single_axis_target(2, self._place_base_z, sleep_time=1)
        self.gripper_control(1, 1.2)
        self.pick_cnt += 1

    def lift_up_to_place_detect_height(self):
        """抬起到放置检测时的高度（该函数调用后将会更新place的相关Z轴位置参数）"""
        if self.place_cnt == 0:
            self.target_place_z = self._place_base_z
            self.detect_place_z = (
                self.target_place_z + self._cube_height + self._detect_gap_place
            )
        else:
            self.target_place_z = (
                self._place_base_z + self._cube_height * self.place_cnt
            )
            self.detect_place_z = self.target_place_z + self._detect_gap_place
        self.go_to_single_axis_target(2, self.detect_place_z, sleep_time=1)

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
        self.lift_up_to_place_detect_height()
        rospy.sleep(0.5)
        self.go_to_place_detect_pose()

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
            self.detect(), print("pick_detect_finish")
            self.pick_detect_over(), print("pick_detect_over_finish")
            print("current_stage-2:", self.get_current_stage())
            self.detect(), print("place_detect_finish")
            self.place_detect_over(), print("place_detect_over_finish")
            print("place_cnt:", self.place_cnt)
            print(" ")
        print("test finished")


class BuildingBlocksEnv(gym.Env, Environment):
    def __init__(self, config_path, obs_cfg, act_cfg):
        """机械臂搭积木RL环境"""
        super().__init__(obs_cfg, act_cfg)
        print(self.obs_cfg._id, self.act_cfg.id)

        """机械臂任务相关初始化"""
        # 读取配置文件
        config = recorder.json_process(config_path)
        self._sim_type = config["SIM_TYPE"]
        # 选择夹爪类型
        if "gazebo" in self._sim_type:
            sim_type_g = "gazebo"
        else:
            sim_type_g = self._sim_type
        self.gripper_control = ChooseGripper(sim_type=sim_type_g)()
        other_config = None if sim_type_g != "gazbeo" else ("", "airbot_play_arm")
        # 初始化机械臂搭积木控制接口
        node_name = rospy.get_name()
        if node_name == "":
            node_name = "building_blocks_env"
            rospy.init_node(node_name)
        self.arm = BuildingBlocksInterface(
            node_name=node_name,
            gripper=(4, self.gripper_control),
            other_config=other_config,
        )
        self.arm.configure(config)

        self._kp = config["KP"]
        self._pick_kp = config["PICK_KP"]
        self._place_kp = config["PLACE_KP"]
        self._pick_tolerance = np.array(config["PICK_TOLERANCE"])
        self._place_tolerance = np.array(config["PLACE_TOLERANCE"])

        str_to_bool = lambda x: True if x == "True" else False
        self._not_pick = str_to_bool(config["NOT_PICK"])

        self.sleep_time = 2  # make sure the previous action is done

        self.reset_cnt = 0

        self._time_base = rospy.get_time()
        self._no_target_check = 0

    def success_judge(self, observation: np.ndarray):
        """判断是否成功"""
        stage = self.arm.get_current_stage()
        if "pick" in stage:
            bias = self._pick_tolerance - np.abs(observation)
            # print("error", bias)
            return (bias >= 0).all()
        elif "place" in stage:
            bias = self._place_tolerance - np.abs(observation)
            # print("error", bias)
            return (bias >= 0).all()
        else:
            raise Exception("stage error")

    def step(self, action):
        self.cur_act = action

        # Get the current stage
        current_stage = self.arm.get_current_stage()

        # Take action
        act_converted = self.act_cfg.convert(action)

        # 移动一步
        self.arm.set_and_go_to_pose_target(
            act_converted, "last", 0.7, return_enable=True
        )

        observation = self._get_obs()  # deviations x, y ,yaw
        self.no_target = False
        if (self.last_obs == observation).all():
            self._no_target_check += 1
            if self._no_target_check > 4:
                print("no target")
                self.no_target = True
                reward = -2000
        else:
            # print("observation", observation)
            self._no_target_check = 0

        # Calculate reward
        if np.linalg.norm(observation) < np.linalg.norm(self.last_obs):
            reward = 1 * (100 - np.linalg.norm(observation))
        else:
            reward = -100
            print("not perfect")

        terminated = False
        truncated = False
        info = {}
        # abs(observation[0]) < 6 and abs(observation[1]) < 3 and observation[2] < 0.1
        if self.no_target or (self.success_judge(observation)):  # threshold
            reward += 1000
            if not self._not_pick:
                if "pick" in current_stage:
                    print("Start to pick block")
                    # start the pick up-action
                    self.arm.pick_detect_over()
                    self.arm.detect()  # start place detect
                elif "place" in current_stage:
                    print("Start to place block")
                    # start the place-action
                    self.arm.place_detect_over()
                    self.arm.go_to_pick_pose()
                    self.arm.detect()  # start pick detect
                # Define the termination condition
                if self.arm.place_cnt == 6:
                    terminated = True
            else:
                terminated = True
        self.cur_rwd = reward

        # record the data
        self.record_add_cur(debug=True)
        return self.obs_cfg.convert(observation), reward, terminated, truncated, info

    def _get_obs(self):
        # deviations: x, y ,yaw
        return np.array(self.arm.get_what_see())

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        print("Reset")
        reset_scene = True
        if reset_scene:
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
        self.arm.go_to_pick_pose()
        # transit to pick mode for vision part
        self.arm.detect()
        rospy.sleep(0.2)
        self.cur_traj_id = self.reset_cnt
        self.cur_obs = self._get_obs()
        info = {}
        self.reset_cnt += 1
        return self.obs_cfg.convert(self.last_obs), info

    def render(self):
        pass


if __name__ == "__main__":
    env = BuildingBlocksEnv(
        "./pick_place_configs_isaac_new.json",
        ObsSim2Real(1),
        ActionSim2Real(1),
    )
