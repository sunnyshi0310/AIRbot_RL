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
from airbot_play_control.control import RoboticArmAgent,AirbotPlayConfig
from math import pi

from moveit_commander import MoveGroupCommander


def gripper_control(pp,sleep_time=0):
    airbot_play_gripper = MoveGroupCommander("airbot_play_gripper")
    airbot_play_gripper.set_max_acceleration_scaling_factor(0.1)
    airbot_play_gripper.set_max_acceleration_scaling_factor(0.1)
    
    if pp:  # pp=1, pick
        SCALE_CLOSE = 0.7
        delta = 0.04
        CLOSE = (-SCALE_CLOSE,SCALE_CLOSE,SCALE_CLOSE,-SCALE_CLOSE + delta)
        airbot_play_gripper.go(CLOSE)
    else:
        airbot_play_gripper.set_named_target("open")
        airbot_play_gripper.go(wait=True)
    rospy.sleep(sleep_time)

                
def tf_to_xyzrpy(tf:TransformStamped):
    """ 将TransformStamped格式的数据转换为xyz和rpy """
    if not tf:
        # when there are no cubes in the camera, generate random actions to move the camera
        Obs2D = (np.random.rand(3)-0.5)*1000.0 
        Obs2D[2] = 0 # make sure the yaw angle is valid
    else:
        xyz = [tf.transform.translation.x,tf.transform.translation.y,tf.transform.translation.z]
        rpy = list(tf_conversions.transformations.euler_from_quaternion([tf.transform.rotation.x,tf.transform.rotation.y,
                                                                tf.transform.rotation.z,tf.transform.rotation.w]))
        Obs2D = np.array([xyz[0],xyz[1],rpy[2]])
    return Obs2D

    
class AIRbotPlayEnv(gym.Env):
    def __init__(self):
        # Define ros related itemss
        NODE_NAME = 'AIRbot'
        rospy.init_node(NODE_NAME, anonymous = True)
        rospy.Subscriber('target_TF',TransformStamped,self._feedback_callback,queue_size=1)

        # Use Moveit to control the robot arm
        self.arm = RoboticArmAgent(control_mode=AirbotPlayConfig.normal,node_name=NODE_NAME,other_config = ("","airbot_play_arm"))

        # Define the observation space and action space
        # Observations are deviations from the correct pose, including x, y and yaw
        self.observation_space = spaces.Box(
            low = np.array([-1000.0, -1000.0, -pi]),
            high = np.array([1000.0, 1000.0, pi]),
            dtype=np.double
        )

        # We have 6 actions, corresponding to "right", "left","up",  "down", "clockwise", "counterclock"
        self.action_space = spaces.Discrete(6) # improved later with complicated actions.

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
            5: np.array([0, 0, -1])            
        }
        self.step_size = np.array([0.005,0.005,0.005]) # action step size on x- y- yaw direction.
        self.sleep_time = 2 # make sure the previous action is done
        self.cube_counter = 0; # count the properly settled cube
        self._pixel_error = TransformStamped() # initilize the pixel info
        
        self.pick_pose = [0.59961,-0.17347,0.54,0.0,1.57,0.0]
        self._pick_base_z = 0.469 # for gazebo
        
        
    def step(self, action):
        print('action',action)
        # Take action
        direction = self._action_to_direction[action]
        inc = direction * self.step_size
        pos_inc = [inc[0],inc[1],0]       
        rot_inc = [0, 0, inc[2]]
        # last表示给定值是基于上次目标值的增量
        self.arm.set_and_go_to_pose_target(pos_inc,rot_inc,'last',self.sleep_time,return_enable =True)
        
        # rospy.sleep(self.sleep_time)
        observation = self._get_obs() # deviations x, y ,yaw
        print('observation',observation)
        reward = -np.linalg.norm(observation)#-(1-self.cube_counter)*500.0
        print("reward", reward)        
        
        terminated = False
        truncated = False
        info = {}
                
        if abs(observation[0])< 6 and abs(observation[1])< 3 and observation[2] < 0.1: # threshold
            print('Start to  pick up')            
            # start the pick up-action
            # move down the robot arm
            self.arm.go_to_single_axis_target(2,self._pick_base_z,sleep_time=1)  # 首先到达可抓取的高度位置(z单轴移动)
            gripper_control(1,self.sleep_time)
            # lift up the arm
            self.arm.go_to_single_axis_target(2, self._pick_base_z+0.05, sleep_time=1)
                        
            # move to the desired position
            place_xyz = [0.5,-0.4,0.469]
            place_rpy = [0.0,1.57,0.0]
            self.arm.set_and_go_to_pose_target(place_xyz,place_rpy,'0',self.sleep_time,return_enable =True)
            
            if self.cube_counter == 0:
                rospy.set_param("/vision_attention",'place0') # set the vision mode
            else:
                rospy.set_param("/vision_attention",'place1') # set the vision mode
            
            gripper_control(0,self.sleep_time)
            # lift up the arm
            self.arm.go_to_single_axis_target(2, self._pick_base_z+0.05, sleep_time=1)
            
            # move back to pick up area
            self.arm.set_and_go_to_pose_target(self.pick_pose[0:3],self.pick_pose[3:6],'0',self.sleep_time,return_enable =True)
            
            # Define the termination condition
            if self.cube_counter == 0:
                rospy.set_param("/vision_attention",'pick1') # set the vision mode
            else:
                rospy.set_param("/vision_attention",'pause') # set the vision mode
                terminated = True           
                
            self.cube_counter +=1
                       
        return observation, reward, terminated, truncated, info
        
    def _get_obs(self):
        # deviations x, y ,yaw
        return tf_to_xyzrpy(self._pixel_error)
        
    def _feedback_callback(self, msg:TransformStamped):
        self._pixel_error = msg
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        print('Reset')
        # Reset the cubes and robot
        rospy.wait_for_service('/gazebo/reset_world')
        reset_world = rospy.ServiceProxy('/gazebo/reset_world', Empty)
        reset_world()
        
        # reset the robot arm to pick pose
        self.arm.go_to_named_or_joint_target(self.arm.change_pose_to_joints(self.pick_pose),sleep_time=2)   
        rospy.set_param("/vision_attention",'pick0') # transit to pick mode for vision part 
        self.cube_counter = 0    
        observation = self._get_obs() 
        info = {}    
        return observation, info
    
    def render(self):
        pass
    
# from stable_baselines3.common.env_checker import check_env

# env = AIRbotPlayEnv()
# check_env(env, warn = True)
