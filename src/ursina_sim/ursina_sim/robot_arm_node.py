#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from pymoveit2 import MoveIt2
from pymoveit2.robots import kuka

class RobotArmNode(Node):
    def __init__(self):
        super().__init__("robot_arm_node")

        # Initialize MoveIt2 interface
        self.moveit2 = MoveIt2(
            node=self,
            joint_names=ur5.joint_names(),
            base_link_name=ur5.base_link_name(),
            end_effector_name=ur5.end_effector_name(),
            group_name=ur5.MOVE_GROUP_ARM,
        )

        # Subscriber: listen for target poses from sim_node
        self.subscription = self.create_subscription(
            PoseStamped,
            "arm_target_pose",
            self.target_pose_callback,
            10,
        )

        self.get_logger().info("robot_arm_node started with pymoveit2")

    def target_pose_callback(self, msg: PoseStamped):
        self.get_logger().info(f"Received target pose: {msg.pose.position}")
        # Send the target to MoveIt2
        self.moveit2.move_to_pose(msg.pose)
        self.moveit2.wait_until_executed()

def main(args=None):
    rclpy.init(args=args)
    node = RobotArmNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()
