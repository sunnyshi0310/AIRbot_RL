#!/usr/bin/env python3
import rospy
from action_test import get_random_spiral_trajs, get_agent, get_move_limit
from robot_tools.trajer import TrajsRecorder, TrajTools
from robot_tools import pather
import numpy as np
import argparse


if __name__ == "__main__":

    # 参数解析
    parser = argparse.ArgumentParser(description="Record data")
    parser.add_argument(
        "-jpp", "--just_pick_pose", action="store_true", help="Just go to pick pose"
    )
    parser.add_argument(
        "-rt", "--robot_type", type=str, default="real", help="Robot type: real or sim"
    )
    parser.add_argument(
        "-sp", "--save_path", type=str, default="", help="Save path"
    )
    args, unknown = parser.parse_known_args()

    NODE_NAME = "record_data_node"
    rospy.init_node(NODE_NAME)

    robot_type = args.robot_type
    print(f"robot_type: {robot_type}")
    if robot_type == "real":
        file_name = "pick_place_configs_real.json"
    elif robot_type == "isaac":
        file_name = "pick_place_configs_isaac_new.json"
    prefix = pather.get_current_dir(__file__, upper=1)
    args.save_path = f"{prefix}/trajs_recorder.json" if args.save_path == "" else args.save_path
    bbi = get_agent(NODE_NAME, prefix, file_name=file_name)
    bbi.go_to_pick_pose()  # 初始控制real: [0.274, 0.0, 0.03] aloha二值刚好接触地面
    if args.just_pick_pose:
        rospy.spin()
        exit(0)

    # 初始化轨迹记录器
    feature_names = ["observation", "action", "raw_target"]
    tr = TrajsRecorder(feature_names, args.save_path)

    # 添加随机初始偏移
    arm_max_xy, arm_min_xy, arm_max_d = get_move_limit()
    init_xy = np.random.uniform(arm_min_xy, arm_max_xy)
    bbi.detect()  # 使能视觉检测

    # 初始控制命令
    temp_position = list(bbi.pick_pose[:3])
    temp_orientation = list(bbi.pick_pose[3:])
    # base_target_xy_yaw = np.array(list(temp_position[:2]) + [bbi.pick_pose[5]])
    base_target_xy_yaw = np.array(temp_position[:2])  # 无需考虑yaw/z

    experiment_id = 0  # 决定随机种子
    trajs = get_random_spiral_trajs(
        experiment_id, epochs=20, points_num=10, add_z=None, add_yaw=False, test_from=0
    )
    # 重复动作（因为会得到略有不同的观测）
    duplicate = 5
    trajs = TrajTools.repeat_trajs(trajs, duplicate, "v")
    # trajs_lenth = len(trajs)
    # assert trajs_lenth == 200 * 5
    # 计算相对动作
    ref_trajs = TrajTools.insert_point(trajs, 0, base_target_xy_yaw, "v", True)
    inc_trajs = trajs - ref_trajs

    tr.feature_add(0, feature_names[1], inc_trajs.tolist(), all=True)
    tr.feature_add(0, feature_names[2], trajs.tolist(), all=True)
    # init obs
    obs = bbi.get_what_see()

    cnt = 1
    for way_point in trajs:
        print(
            f"Point_{cnt} 后验动作：",
            list(way_point[:2]) + [temp_position[2]],
            "先验观测：",
            obs,
        )
        tr.feature_add(0, feature_names[0], obs)
        temp_position[:2] = way_point[:2]
        # temp_orientation[2] = way_point[2]
        bbi.set_and_go_to_pose_target(temp_position, temp_orientation, sleep_time=1.5)
        obs_temp = bbi.get_what_see()
        if obs == obs_temp:
            print(f"Warning_{cnt}: 观测未更新，很可能发生遮挡！")
        else:
            obs = obs_temp
        cnt += 1

    tr.save()