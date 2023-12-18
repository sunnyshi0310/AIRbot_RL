from airbot_play_control.control import RoboticArmAgent
import rospy

NODE_NAME = "test_node"
rospy.init_node(NODE_NAME)

arm = RoboticArmAgent(node_name=NODE_NAME,other_config=("", "airbot_play_arm"))

arm.go_to_named_or_joint_target([0.003175794615375, -1.142793469954583, 2.467404658086127, 1.558211449229078, 0.255114596298973, -1.558618725616465])
arm.go_to_named_or_joint_target([-1.077810334297183, -0.807377130630933, 2.376355603974666, -0.005763212916910, -1.077841248478552, 0.012177962725514])
arm.go_to_named_or_joint_target([1.080807884083372, -0.807377130630933, 2.376355603974666, 0.005721842735885, 1.080838576351668, -0.012158439683643])
