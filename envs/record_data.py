#!/usr/bin/env python3
import rospy
from action_test import get_random_spiral_trajs, get_agent
from robot_tools.trajer import TrajsRecorder
import numpy as np


if __name__ == "__main__":

    NODE_NAME = "record_data_node"
    rospy.init_node(NODE_NAME)

    feature_names = ["observation", "action"]
    tr = TrajsRecorder(feature_names, "trajs_recorder.json", count="num")

    bbi = get_agent(NODE_NAME)
    bbi.go_to_pick_pose()  # 初始控制
    bbi.detect()  # 使能视觉检测

    # 初始控制命令
    temp_position = list(bbi.pick_pose[:3])
    temp_orientation = list(bbi.pick_pose[3:])
    base_target_xy_yaw = np.array(list(temp_position[:2]) + [bbi.pick_pose[5]])

    experiment_id = 0
    trajs = get_random_spiral_trajs(
        experiment_id, epochs=2, points_num=2, add_z=False, add_yaw=True
    )
    # 计算相对动作
    ref_trajs = np.delete(np.insert(trajs, 0, [base_target_xy_yaw], axis=0), -1, axis=0)
    inc_trajs = trajs - ref_trajs

    tr.feature_add(0, feature_names[1], inc_trajs.tolist(), all=True)
    # init obs
    obs = bbi.get_what_see()

    for way_point in trajs:
        print("Test 后摇动作：", way_point, "前摇观测：", obs)
        tr.feature_add(0, feature_names[0], obs)
        temp_position[:2] = way_point[:2]
        # temp_orientation[2] = way_point[2]
        bbi.set_and_go_to_pose_target(temp_position, temp_orientation, sleep_time=1.5)
        obs = bbi.get_what_see()

    tr.save("trajs_recorder.json")
