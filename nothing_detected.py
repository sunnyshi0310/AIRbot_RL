import rospy
from geometry_msgs.msg import TransformStamped

rospy.init_node('nothing_detected', anonymous=True)

tf_pub = rospy.Publisher('/target_TF', TransformStamped, queue_size=1)
xy_bias = [100.0, 100.0]
target_TF = TransformStamped()
received_TF = None
rate = rospy.Rate(5)  # 5hz
while not rospy.is_shutdown():
    try:
        received_TF:TransformStamped = rospy.wait_for_message('/target_TF', TransformStamped, timeout=1.5)
    except rospy.exceptions.ROSException:
        rospy.loginfo("Nothing detected")
        if received_TF is None: continue
        translation = received_TF.transform.translation
        if translation.x == 0.0: translation.x = 0.0
        else: translation.x *= xy_bias[0] / abs(translation.x)
        if translation.y == 0.0: translation.y = 0.0
        else: translation.y *= xy_bias[1] / abs(translation.y)
        target_TF.transform.translation = translation
        target_TF.transform.rotation = received_TF.transform.rotation  # keep the same
        tf_pub.publish(target_TF)
    else: rate.sleep()
