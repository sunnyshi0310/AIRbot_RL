#!/usr/bin/env python3

import rospy
from airbot_play_control.control import ChooseGripper
from robot_tools import recorder
from robot_tools.painter import Painter2D
from building_blocks_env import BuildingBlocksInterface
import numpy as np


def get_move_limit(robot_type="real"):
    # 适当缩小范围，避免极限越界
    if robot_type == "isaac":
        # 初始位置xyz: [0.2740,-0.0000,-0.2340]
        # 最下位姿：[0.15,-0.0000,-0.2340]
        arm_max_xy = np.array([0.26, 0])
        arm_min_xy = np.array([0.15, 0])
        hw_ratio = 1280 / 720
    elif robot_type == "real":
        # 初始位置xyz: [0.2740,0.0000,0.03]
        # 最下位姿：xyz: [0.1740,0.0000,0.03]
        arm_max_xy = np.array([0.26, 0])
        arm_min_xy = np.array([0.174, 0])
        hw_ratio = 640 / 480
    arm_max_d = np.abs((arm_max_xy[0] - arm_min_xy[0]))
    arm_max_r = arm_max_d / 2
    arm_max_xy[1] = arm_max_r
    arm_min_xy[1] = -arm_max_r
    return arm_max_xy, arm_min_xy, arm_max_d, hw_ratio


def get_random_spiral_trajs(
    experiment_id=0, epochs=10, points_num=100, add_z=None, add_yaw=False, test_from=0
) -> np.ndarray:
    start = experiment_id * epochs + test_from
    end = (experiment_id + 1) * epochs

    if add_z is None and not add_yaw:
        base = np.ones((1, 2))
    else:
        base = np.ones((1, 3))

    for i in range(start, end):
        np.random.seed(i)
        a, b, num_points, turns = 0.1, 0.1, points_num, 3
        end_phase = 3 / 2 * np.pi
        roation = np.random.uniform(0, 2 * np.pi)

        if add_z is not None:  # 增加随机高度
            dim = np.random.uniform(add_z, add_z + 0.02, size=points_num)
        elif add_yaw:  # 增加随机旋转
            dim = np.random.uniform(-np.pi / 4, np.pi / 4, size=points_num)
        else:
            dim = None

        spiral_points = Painter2D.get_spiral_points(
            a, b, num_points, turns, end_phase=end_phase, points_allocate_mode="turn"
        )
        spiral_points = Painter2D.reverse_points(spiral_points)
        spiral_points = Painter2D.rotate_points(spiral_points, roation)

        arm_max_xy, arm_min_xy, arm_max_d, hw_ratio = get_move_limit()

        spiral_r = np.max(np.abs(spiral_points[0]))
        spiral_max_r = arm_max_d / 2
        spiral_min_r = spiral_max_r / 3  # 避免移动距离过小
        spiral_ran_r_x = np.random.uniform(spiral_min_r, spiral_max_r)
        # 1280/720=1.777...; 640 / 480=1.333...
        spiral_ran_r_y = np.random.uniform(spiral_min_r, spiral_max_r * hw_ratio)
        factor_x = spiral_r / spiral_ran_r_x
        factor_y = spiral_r / spiral_ran_r_y
        spiral_ran_r_xy = np.array([spiral_ran_r_x, spiral_ran_r_y])
        # 根据生成的轨迹范围确定初始位置范围
        init_max_xy = arm_max_xy - spiral_ran_r_xy
        init_min_xy = arm_min_xy + spiral_ran_r_xy

        spiral_points /= np.array([factor_x, factor_y])
        spiral_points += np.random.uniform(init_min_xy, init_max_xy)

        if dim is not None:
            spiral_points = Painter2D.append_one_dim(spiral_points, dim)
        base = np.concatenate((base, spiral_points), axis=0)
    # # test
    # assert (
    #     len(base) == epochs * points_num + 1
    # ), f"len(spiral_points): {len(spiral_points)}"
    return base[1:]


def get_agent(node_name, path_pre=".", file_name=None) -> BuildingBlocksInterface:
    if file_name is None:
        file_name = "pick_place_configs_isaac_new.json"
    config = recorder.json_process(f"{path_pre}/{file_name}")
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
    bbi:BuildingBlocksInterface = get_agent()
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
