import rclpy
from rclpy.node import Node

from moveit_msgs.srv import GetPositionIK, GetMotionPlan
from moveit_msgs.msg import PositionIKRequest, MotionPlanRequest

from geometry_msgs.msg import PoseStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

class MoveIt2PythonInterface(Node):
    def __init__(self, group_name, joint_names):
        super().__init__('moveit2_python_interface')
        
        self.group_name = group_name
        self.joint_names = joint_names
        
        # IK service client
        self.ik_cli = self.create_client(GetPositionIK, 'compute_ik')
        while not self.ik_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for IK service...')

        # Motion plan service client
        self.plan_cli = self.create_client(GetMotionPlan, 'plan_kinematic_path')
        while not self.plan_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for MotionPlan service...')

    def get_ik(self, pose: PoseStamped):
        """Call MoveIt2 IK service to get joint positions for target pose"""
        req = GetPositionIK.Request()
        req.ik_request.group_name = self.group_name
        req.ik_request.pose_stamped = pose
        future = self.ik_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        result = future.result()
        if result.error_code.val == 1:  # SUCCESS
            return list(result.solution.joint_state.position)
        else:
            self.get_logger().warn(f"IK failed with code {result.error_code.val}")
            return None

    def plan_trajectory(self, start_joints, target_pose: PoseStamped):
        """Call MoveIt2 MotionPlan service to get trajectory"""
        req = GetMotionPlan.Request()
        mp_req = MotionPlanRequest()
        mp_req.group_name = self.group_name
        mp_req.start_state.joint_state.position = start_joints
        mp_req.goal_constraints.append(
            PositionIKRequest().pose_stamped  # or construct a proper constraint
        )
        req.motion_plan_request = mp_req
        future = self.plan_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        result = future.result()
        if result.error_code.val == 1:  # SUCCESS
            return result.trajectory
        else:
            self.get_logger().warn(f"Motion plan failed with code {result.error_code.val}")
            return None

    def simple_trajectory_point(self, joint_positions, dt=0.05):
        """Convert joint positions to a ROS2 JointTrajectory with a single point"""
        traj_msg = JointTrajectory()
        traj_msg.joint_names = self.joint_names
        point = JointTrajectoryPoint()
        point.positions = joint_positions
        point.velocities = [0.0] * len(joint_positions)
        point.time_from_start.sec = int(dt)
        point.time_from_start.nanosec = int((dt - int(dt)) * 1e9)
        traj_msg.points.append(point)
        return traj_msg