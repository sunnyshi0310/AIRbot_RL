#!/usr/bin/env python3

import rospy
from airbot_play_control.control import ChooseGripper
from robot_tools import recorder
from robot_tools.painter import Painter2D
from building_blocks_env import BuildingBlocksInterface
import numpy as np


def get_random_spiral_trajs(
    experiment_id=0, epochs=10, points_num=100, add_z=True, add_yaw=False
) -> np.ndarray:
    base = np.ones((1, 3))
    start = experiment_id * epochs
    end = (experiment_id + 1) * epochs

    for i in range(start, end):
        np.random.seed(i)
        a, b, num_points, turns = 0.1, 0.1, points_num, 3
        end_phase = 3 / 2 * np.pi
        roation = np.random.uniform(0, 2 * np.pi)

        spiral_points = Painter2D.get_spiral_points(
            a, b, num_points, turns, end_phase=end_phase, points_allocate_mode="turn"
        )
        spiral_points = Painter2D.reverse_points(spiral_points)
        spiral_points = Painter2D.rotate_points(spiral_points, roation)

        # 目标位置xyz: [0.1540,0.0000,-0.2340]
        # 初始位置xyz: [0.2740,-0.0000,-0.2340]
        arm_max_xy = np.array([0.2740, 0])
        arm_min_xy = np.array([0.1540, 0])
        arm_max_d = np.abs((arm_max_xy[0] - arm_min_xy[0]))

        spiral_r = np.max(np.abs(spiral_points[0]))
        spiral_max_r = arm_max_d / 2
        spiral_min_r = spiral_max_r / 3
        spiral_ran_r = np.random.uniform(spiral_min_r, spiral_max_r)
        factor = spiral_r / spiral_ran_r
        init_max_xy = arm_max_xy - spiral_ran_r
        init_min_xy = arm_min_xy + spiral_ran_r

        spiral_points /= np.array([factor, factor / 2])
        spiral_points += np.random.uniform(init_min_xy, init_max_xy)

        # 增加z轴坐标
        if add_z:
            dim = -0.2340
        # 增加随机旋转
        elif add_yaw:
            dim = np.random.uniform(-np.pi / 4, np.pi / 4, size=len(spiral_points))

        spiral_points = Painter2D.append_one_dim(spiral_points, dim)
        base = np.concatenate((base, spiral_points), axis=0)
    assert (
        len(base) == epochs * points_num + 1
    ), f"len(spiral_points): {len(spiral_points)}"
    return base[1:]


def get_agent(node_name, path_pre=".") -> BuildingBlocksInterface:
    config = recorder.json_process(f"{path_pre}/pick_place_configs_isaac_new.json")
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
        node_name=node_name,
        gripper=(4, gripper_control),
        other_config=other_config,
    )
    bbi.configure(config)
    return bbi


if __name__ == "__main__":
    """
    1. 启动MoveIt
    roslaunch airbot_play_launch airbot_play_moveit.launch target_moveit_config:=airbot_play_v2_1_config use_basic:=true use_rviz:=true
    2. 启动仿真环境
    cd path/to/airbot_play_isaac_sim
    ./sim/start_sim.sh -pc v2_1_p
    3. 启动视觉识别
    rosrun airbot_play_vision_cpp ros_demo --not_show 0 --use_real 0 --negative 1 --sim_type isaac
    4. 运行本文件
    python ./envs/action_test.py
    """
    NODE_NAME = "BuildingBlocksInterfaceTest"
    rospy.init_node(NODE_NAME)
    bbi = get_agent()
    TEST_ID = 1
    if TEST_ID == 0:
        bbi.test(2)
    elif TEST_ID == 1:
        bbi.go_to_pick_pose()
        experiment_id = 0
        trajs = get_random_spiral_trajs(experiment_id, epochs=10, points_num=3)
        for way_point in trajs:
            print("当前目标：", way_point)
            bbi.set_and_go_to_pose_target(way_point, sleep_time=1.5)