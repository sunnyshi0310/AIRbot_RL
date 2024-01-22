from airbot_play_control.control import RoboticArmAgent
import rospy

# NODE_NAME = "test_node"
# rospy.init_node(NODE_NAME)

# arm = RoboticArmAgent(node_name=NODE_NAME,other_config=("", "airbot_play_arm"))

# arm.go_to_named_or_joint_target([0.003175794615375, -1.142793469954583, 2.467404658086127, 1.558211449229078, 0.255114596298973, -1.558618725616465])
# arm.go_to_named_or_joint_target([-1.077810334297183, -0.807377130630933, 2.376355603974666, -0.005763212916910, -1.077841248478552, 0.012177962725514])
# arm.go_to_named_or_joint_target([1.080807884083372, -0.807377130630933, 2.376355603974666, 0.005721842735885, 1.080838576351668, -0.012158439683643])

import numpy as np

# print(np.array([1,2,3]) * np.array([1,2,3]))

# x = [[1,2,3],[4,5,6],[7,8,9]]
# x_np = np.array(x)
# print(x_np.T)
# print(np.zeros((3,3)))
# print(np.concatenate((x_np.T,np.zeros((3,3))),axis=1))
# print(np.delete(x_np,0,axis=1))
# print(np.delete(x_np,-1,axis=1))

# Xtrue = np.zeros((4, 6))
# step = 0
# print(Xtrue[step, :])
# print(Xtrue[step, :].T)
# print(Xtrue[step, :][:, np.newaxis])

# x0 = np.array([0.5, -0.75])

# print([np.array([1,2,3,4,5,6])[0::2],np.array([1,2,3,4,5,6])[1::2]])
# print(np.hstack([np.array([1,2,3,4,5,6]),np.array([1,2,3,4,5,6])]))

# import numpy as np

# # 创建一个二维数组
# array_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# # 计算每行的范数
# norms = np.max(array_2d, axis=1, keepdims=True)

# # 对每行进行归一化
# normalized_array = array_2d / norms

# print("原始数组：")
# print(array_2d)
# print("\n每行归一化后的数组：")
# print(normalized_array)

a = np.array(([12,3,4],[1,2,3]))
a[0] = [1,2,3]  # 列表可以直接赋值给数组
print(a)
