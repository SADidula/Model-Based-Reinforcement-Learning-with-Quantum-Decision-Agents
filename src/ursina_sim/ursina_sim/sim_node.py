import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy

from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker
from trajectory_msgs.msg import JointTrajectory

from .bot_dpg_her_logic import BotDPGHER
from .qrl_dpg_her_logic import QRLDPGHERAgent  # classical DDPG variant (if present)
from .dpg_her_logic import RLDPGHERAgent       # classical DDPG+HER (updated to accept achieved_next)
from .quantum_dpg_her_logic import QRLDPGHERAgent as QuantumRLDPGHERAgent  # quantum DDPG+HER

from .polygon_path import PolygonPath
from .moveit_interface import MoveIt2PythonInterface
from .metrics_tracker import MetricsTracker
from .robotic_config import RoboticConfig

import math
import numpy as np
import os
import csv
import time
from typing import Any, Tuple, Optional, Dict, List
import argparse


def offset_polygon(vertices, offset_x=0.0, offset_y=0.0):
    return [(x + offset_x, y + offset_y) for x, y in vertices]


def _pentagon_vertices(circle_radius: float = 10.0):
    points = []
    for i in range(0, 5):
        x = circle_radius * math.sin(2 * math.pi * i / 5)
        y = circle_radius * math.sin(2 * math.pi * i / 5) * math.cos(2 * math.pi * i / 5)
        points.append((round(x, 2), round(y, 2)))
    return points


def _figure_8_vertices(resolution: int = 5, circle_radius: float = 10.0):
    points = []
    for i in range(0, resolution):
        x = circle_radius * math.sin(2 * math.pi * i / resolution)
        y = circle_radius * math.sin(2 * math.pi * i / resolution) * math.cos(2 * math.pi * i / resolution)
        points.append((round(x, 2), round(y, 2)))
    return points


def _star_vertices(resolution: int = 5, circle_radius: float = 10.0):
    points = []
    for i in range(0, resolution):
        x = circle_radius * math.cos((2 * math.pi * i) / resolution + math.pi / 2)
        y = circle_radius * math.sin((2 * math.pi * i) / resolution + math.pi / 2)
        points.append((round(x, 2), round(y, 2)))
        p = circle_radius * 0.5 * math.cos((2 * math.pi * i) / resolution + math.pi / 2 + 2 * math.pi / 10)
        q = circle_radius * 0.5 * math.sin((2 * math.pi * i) / resolution + math.pi / 2 + 2 * math.pi / 10)
        points.append((round(p, 2), round(q, 2)))
    return points


def _circle_vertices(resolution: int = 8, circle_radius: float = 10.0):
    return [
        (circle_radius * math.cos(theta), circle_radius * math.sin(theta))
        for theta in np.linspace(0, 2 * math.pi, num=resolution, endpoint=False)
    ]


def _triangle_vertices():
    return [
        (0.0, 10.0),
        (-8.66, -5.0),
        (8.66, -5.0),
    ]


def _square_vertices():
    return [
        (0.0, 0.0),
        (0.0, 10.0),
        (10.0, 10.0),
        (10.0, 0.0),
    ]


def limit_polygon_to_workspace(vertices, max_radius=0.5):
    points = np.array(vertices, dtype=float)
    center = points.mean(axis=0)
    vectors = points - center
    distances = np.linalg.norm(vectors, axis=1)
    scale_factors = np.minimum(1.0, max_radius / (distances + 1e-9))
    scaled_points = center + vectors * scale_factors[:, np.newaxis]
    offset = -scaled_points.mean(axis=0)
    final_points = scaled_points + offset
    return [tuple(map(float, p)) for p in final_points]


def scale_polygon_around_center(vertices, scale_factor):
    points = np.array(vertices, dtype=float)
    center = points.mean(axis=0)
    new_points = center + scale_factor * (points - center)
    return [tuple(map(float, p)) for p in new_points]


def clamp_to_workspace(x, y, max_radius=0.5):
    r = math.hypot(x, y)
    if r > max_radius:
        scale = max_radius / r
        x *= scale
        y *= scale
    return x, y


class PathFollowerSimulator(Node):
    def __init__(self):
        super().__init__('path_follower_simulator')

        # load robot configs
        config = RoboticConfig(file_name="robotic_config.json").load_config()
        self.group_name = config["group_name"]
        self.joint_names = config["joint_names"]
        self.base_link = config["base_link"]
        self.safety_radius = config["safety_radius"]

        # Publishers
        self.bot_pose_pub = self.create_publisher(PoseStamped, 'bot_pose', 10)
        self.bot_marker_pub = self.create_publisher(Marker, '/bot_marker', 10)
        self.traj_pub = self.create_publisher(
            JointTrajectory,
            '/joint_trajectory_controller/joint_trajectory',
            10
        )

        marker_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL
        )
        self.path_marker_pub = self.create_publisher(Marker, '/main_path_marker', marker_qos)
        self.boundary_markers_pub = self.create_publisher(Marker, '/boundary_markers', marker_qos)
        self.target_pub = self.create_publisher(PoseStamped, '/arm_target_pose', 10)

        # Build polygon path from CLI (may be overridden by multi-path scheduler)
        main_polygon_vertices = self._build_polygon_from_args()
        main_polygon_vertices = limit_polygon_to_workspace(main_polygon_vertices, max_radius=args.max_radius)
        main_polygon_vertices = offset_polygon(main_polygon_vertices, offset_x=args.offset_x, offset_y=args.offset_y)

        # Boundaries (optional)
        self.boundary_spec: List[Tuple[float, float]] = []
        self.path_polygon = PolygonPath(main_polygon_vertices)
        self.boundary_polygons = [
            (PolygonPath(scale_polygon_around_center(main_polygon_vertices, scale)), penalty)
            for scale, penalty in self.boundary_spec
        ]

        # Timers
        self.simulation_timer = self.create_timer(0.05, self._update_simulation_step)
        self.visualization_timer = self.create_timer(1.0, self._publish_all_path_markers)

        # Agent
        self.agent = self._build_agent_from_args()  # sets self.agent_key

        # Stop condition
        self._stop_requested = False
        self.stop_after_steps = getattr(args, "stop_after_steps", None)

        # Multi-path schedule setup
        self.multi_paths: List[str] = []
        self.steps_per_path: Optional[int] = None
        if getattr(args, "multi_paths", None):
            self.multi_paths = [p.strip().lower() for p in str(args.multi_paths).split(",") if p.strip()]
            self.steps_per_path = int(args.steps_per_path or 0) or int(args.stop_after_steps or 0)
            if not self.steps_per_path:
                raise ValueError("When using --multi_paths, you must set --steps_per_path or --stop_after_steps.")
            self.current_path_index = 0
            # Override initial shape
            args.path = self.multi_paths[0]
            self._rebuild_path_for_current_arg_path()

        # Checkpoint naming
        self._update_file_prefixes_for_current_path()

        # Noise wrapper
        self._wrap_agent_select_action_for_noise()

        # Quantum warmup
        if hasattr(self.agent, "warmup_forward") and not args.no_warmup:
            try:
                self.agent.warmup_forward()
            except Exception as e:
                if args.verbose:
                    self.get_logger().warn(f"Quantum warmup skipped: {e}")

        # Load policy unless skipped
        if not args.skip_load:
            try:
                self.agent.load_policy(self.actor_ckpt_path, self.critic_ckpt_path)
            except Exception as e:
                if args.verbose:
                    self.get_logger().warn(f"Policy load failed or not found: {e}")

        # Bot wrapper
        self.bot = BotDPGHER(
            rl_agent=self.agent,
            main_path=self.path_polygon,
            boundary_penalties=self.boundary_polygons,
            max_speed=0.5,
            max_turn_speed=12.0
        )

        # State, viz, metrics, moveit
        self.initial_bot_pose = (float(getattr(self.bot, 'x', 0.0)), float(getattr(self.bot, 'y', 0.0)))
        self.sim_step_counter = 0

        self._prepare_visualization_markers()
        try:
            self._publish_all_path_markers()
        except Exception:
            pass

        self.metrics_tracker = MetricsTracker()
        self._update_metrics_csv_path()

        self.moveit_iface = MoveIt2PythonInterface(group_name=self.group_name, joint_names=self.joint_names)

        # Episode
        self.episode_step_count = 0

    def _build_polygon_from_args(self):
        shape = args.path
        res = int(args.resolution)
        rad = float(args.radius)
        if shape == "circle":
            verts = _circle_vertices(resolution=res, circle_radius=rad)
        elif shape == "figure8":
            verts = _figure_8_vertices(resolution=res, circle_radius=rad)
        elif shape == "triangle":
            verts = _triangle_vertices()
        elif shape == "square":
            verts = _square_vertices()
        elif shape == "pentagon":
            verts = _pentagon_vertices(circle_radius=rad)
        elif shape == "star":
            star_res = max(5, res)
            verts = _star_vertices(resolution=star_res, circle_radius=rad)
        else:
            verts = _circle_vertices(resolution=res, circle_radius=rad)
        return verts

    def _rebuild_path_for_current_arg_path(self):
        main_polygon_vertices = self._build_polygon_from_args()
        main_polygon_vertices = limit_polygon_to_workspace(main_polygon_vertices, max_radius=args.max_radius)
        main_polygon_vertices = offset_polygon(main_polygon_vertices, offset_x=args.offset_x, offset_y=args.offset_y)
        self.path_polygon = PolygonPath(main_polygon_vertices)
        self.boundary_polygons = [
            (PolygonPath(scale_polygon_around_center(main_polygon_vertices, scale)), penalty)
            for scale, penalty in self.boundary_spec
        ]
        self._prepare_visualization_markers()
        try:
            self._publish_all_path_markers()
        except Exception:
            pass

    def _build_agent_from_args(self):
        agent_choice = str(args.agent).lower()
        is_quantum = agent_choice in ("quantum", "qrl")

        if agent_choice == "rl":
            agent = RLDPGHERAgent(state_dim=4)
            self.agent_key = "rl"
        elif is_quantum:
            backend = args.backend if args.backend not in (None, "None", "none", "") else None
            shots = None if args.shots in (None, -1) else int(args.shots)
            agent = QuantumRLDPGHERAgent(
                state_dim=4,
                backend=backend,
                shots=shots
            )
            self.agent_key = "qrl"
        else:
            if args.verbose:
                self.get_logger().warn(f"Unknown agent '{args.agent}', defaulting to RL.")
            agent = QRLDPGHERAgent(state_dim=4)
            self.agent_key = "rl_no_her"

        mode = str(args.mode).lower()
        if mode not in ("train", "eval"):
            mode = "train"
        if args.verbose:
            self.get_logger().info(f"Agent: {agent_choice} | Mode: {mode}")
        return agent

    def _wrap_agent_select_action_for_noise(self):
        try:
            original_select = self.agent.select_action
            noise = 0.0 if args.mode.lower() == "eval" else float(args.noise_std)

            def select_action_with_noise(state, goal=None, noise_std=None):
                if goal is None:
                    goal = np.array([0.0, 0.0], dtype=np.float32)
                try:
                    return original_select(state, goal, noise if noise_std is None else noise_std)
                except TypeError:
                    return original_select(state, goal)

            self.agent.select_action = select_action_with_noise  # type: ignore[attr-defined]
        except Exception as e:
            if args.verbose:
                self.get_logger().warn(f"Failed to wrap agent.select_action for noise: {e}")

    def _rebind_bot_to_agent(self):
        # reassign the agent to the bot and reset bot internals safely
        self.bot.rl_agent = self.agent
        # reset previous step memory on bot so transitions don't cross agents
        if hasattr(self.bot, "previous_state"):
            self.bot.previous_state = None
        if hasattr(self.bot, "previous_action"):
            self.bot.previous_action = None
        if hasattr(self.bot, "previous_goal"):
            self.bot.previous_goal = None
        if hasattr(self.bot, "previous_achieved"):
            self.bot.previous_achieved = None

    def moveit_command(self):
        dx, dy, dz = 0.005, 0.0, 0.1
        x_target, y_target = clamp_to_workspace(self.bot.x + dx, self.bot.y + dy, max_radius=0.5)
        test_pose = PoseStamped()
        test_pose.header.frame_id = self.base_link
        test_pose.header.stamp = self.get_clock().now().to_msg()
        test_pose.pose.position.x = x_target
        test_pose.pose.position.y = y_target
        test_pose.pose.position.z = dz
        test_pose.pose.orientation.w = 1.0

        joints = self.moveit_iface.get_ik(test_pose)
        if joints is not None:
            traj_msg = self.moveit_iface.simple_trajectory_point(joints, dt=0.1)
            self.traj_pub.publish(traj_msg)
            if args.verbose:
                self.get_logger().info(f"Test MoveIt command published: {joints}")
        else:
            if args.verbose:
                self.get_logger().warn("Test MoveIt IK failed!")

    def _prepare_visualization_markers(self):
        self.main_path_marker = self._as_marker(
            self.path_polygon, "main_path", 1, color=(1.0, 1.0, 1.0, 1.0), width=0.07
        )
        color_palette = [
            (1.0, 0.0, 0.0, 0.5),
            (1.0, 0.5, 0.0, 0.4),
            (0.0, 1.0, 1.0, 0.3),
            (1.0, 0.0, 1.0, 0.2),
        ]
        self.boundary_markers = []
        for idx, (boundary_path, penalty) in enumerate(self.boundary_polygons):
            color = color_palette[idx % len(color_palette)]
            boundary_marker = self._as_marker(
                boundary_path, "boundary", idx + 10, color=color, width=0.05
            )
            self.boundary_markers.append(boundary_marker)

    def _as_marker(self, polygon_path, ns, marker_id, color, width):
        marker = Marker()
        marker.header.frame_id = 'world'
        marker.ns = ns
        marker.id = marker_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = width
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = color
        verts = polygon_path.as_points()
        marker.points = []
        for pt in verts + [verts[0]]:
            point = Point()
            point.x = float(pt[0])
            point.y = float(pt[1])
            point.z = 0.0
            marker.points.append(point)
        return marker

    def _publish_all_path_markers(self):
        now = self.get_clock().now().to_msg()
        self.main_path_marker.header.stamp = now
        for boundary_marker in self.boundary_markers:
            boundary_marker.header.stamp = now
        self.path_marker_pub.publish(self.main_path_marker)
        for marker in self.boundary_markers:
            self.boundary_markers_pub.publish(marker)

    def _reset_bot_position(self):
        if hasattr(self.bot, 'reset') and callable(getattr(self.bot, 'reset')):
            self.bot.reset()
        else:
            self.bot.x, self.bot.y = self.initial_bot_pose
            for attr in ('vx', 'vy', 'v', 'speed', 'omega', 'yaw_rate'):
                if hasattr(self.bot, attr):
                    setattr(self.bot, attr, 0.0)
            for attr in ('theta', 'yaw', 'heading'):
                if hasattr(self.bot, attr):
                    setattr(self.bot, attr, 0.0)
        self.get_logger().info(f'Bot position reset.')

    def _extract_reward_done(self, step_output: Any) -> Tuple[float, bool]:
        reward: Optional[float] = None
        done = False

        if isinstance(step_output, (tuple, list)):
            if len(step_output) >= 2 and isinstance(step_output[1], (int, float)) and not isinstance(step_output[1], bool):
                reward = float(step_output[1])
            if reward is None:
                for item in step_output:
                    if isinstance(item, (int, float)) and not isinstance(item, bool):
                        reward = float(item)
                        break
            for item in step_output:
                if isinstance(item, bool):
                    done = done or item
            if len(step_output) >= 4:
                term = step_output[2] if isinstance(step_output[2], bool) else False
                trunc = step_output[3] if isinstance(step_output[3], bool) else False
                done = done or bool(term) or bool(trunc)
            for item in step_output:
                if isinstance(item, dict):
                    done = (
                        bool(item.get('done', False)) or
                        bool(item.get('terminated', False)) or
                        bool(item.get('truncated', False)) or
                        done
                    )
        elif isinstance(step_output, dict):
            if step_output.get('reward', None) is not None:
                reward = float(step_output['reward'])
            done = (
                bool(step_output.get('done', False)) or
                bool(step_output.get('terminated', False)) or
                bool(step_output.get('truncated', False))
            )

        if reward is None:
            for src in (getattr(self, 'bot', None), getattr(self, 'agent', None)):
                if src is None:
                    continue
                for attr in ('last_reward', 'reward'):
                    if hasattr(src, attr):
                        try:
                            reward = float(getattr(src, attr))
                            break
                        except Exception:
                            pass
                if reward is not None:
                    break

        for src in (getattr(self, 'bot', None), getattr(self, 'agent', None)):
            if src is None:
                continue
            for attr in ('done', 'terminated', 'truncated', 'is_done', 'episode_done', 'episode_terminated'):
                if hasattr(src, attr):
                    try:
                        done = bool(getattr(src, attr)) or done
                    except Exception:
                        pass

        if reward is None:
            reward = 0.0

        return float(reward), bool(done)

    def _update_file_prefixes_for_current_path(self):
        self.path_key = str(args.path).lower()
        self.file_prefix = f"{self.agent_key}_{self.path_key}"
        self.actor_ckpt_path = args.actor_path if args.actor_path else f"{self.file_prefix}_actor.pt"
        self.critic_ckpt_path = args.critic_path if args.critic_path else f"{self.file_prefix}_critic.pt"

    def _update_metrics_csv_path(self):
        self.metrics_csv_path = os.path.join(
            os.getcwd(),
            f"{self.file_prefix}" + ("_eval_metrics.csv" if args.mode.lower() == "eval" else "_metrics.csv")
        )
        os.makedirs(os.path.dirname(self.metrics_csv_path) or ".", exist_ok=True)

    def _advance_to_next_path(self):
        if not self.multi_paths:
            # No multi-run schedule: stop
            self._request_stop(already_flushed=False)
            return

        self.current_path_index += 1
        if self.current_path_index >= len(self.multi_paths):
            # Completed all shapes
            self._request_stop(already_flushed=False)
            return

        # Switch to next path
        next_shape = self.multi_paths[self.current_path_index]
        args.path = next_shape

        # Rebuild polygon and markers
        self._rebuild_path_for_current_arg_path()

        # Optionally reset agent for this path
        if getattr(args, "reset_agent_per_path", False):
            # Rebuild agent fresh
            self.agent = self._build_agent_from_args()
            # Wrap noise again
            self._wrap_agent_select_action_for_noise()
            # Do not load any prior checkpoints unless user explicitly passed paths
            if not args.skip_load and args.actor_path and args.critic_path:
                try:
                    self.agent.load_policy(args.actor_path, args.critic_path)
                except Exception as e:
                    if args.verbose:
                        self.get_logger().warn(f"Agent load skipped/failed on path switch: {e}")
            # Rebind bot to new agent
            self._rebind_bot_to_agent()

        # Update file prefixes, metrics path
        self._update_file_prefixes_for_current_path()
        self._update_metrics_csv_path()

        # Reset bot pose and counters
        self._reset_bot_position()
        self.sim_step_counter = 0
        self.episode_step_count = 0

        self.get_logger().info(f"Switched to next path: {next_shape} for {self.steps_per_path} steps. Reset agent: {getattr(args, 'reset_agent_per_path', False)}")

    def _write_metrics_row(self, row: Optional[Dict[str, float]]):
        if row is None:
            return
        file_exists = os.path.exists(self.metrics_csv_path) and os.path.getsize(self.metrics_csv_path) > 0
        with open(self.metrics_csv_path, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=MetricsTracker.CSV_FIELDS)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    def _request_stop(self, already_flushed: bool = False):
        if getattr(self, "_stop_requested", False):
            return
        self._stop_requested = True

        try:
            if not already_flushed:
                metrics_row = self.metrics_tracker.flush_window()
                self._write_metrics_row(metrics_row)
                if args.verbose and metrics_row is not None:
                    self.get_logger().info(f'Final metrics flushed.')
        except Exception as e:
            if args.verbose:
                self.get_logger().warn(f"Final metrics flush failed: {e}")

        try:
            self.agent.save_policy(self.actor_ckpt_path, self.critic_ckpt_path)
            if args.verbose:
                self.get_logger().info(f'Final checkpoints saved: {self.actor_ckpt_path}, {self.critic_ckpt_path}')
        except Exception as e:
            if args.verbose:
                self.get_logger().warn(f"Final checkpoint save failed: {e}")

        try:
            self.simulation_timer.cancel()
        except Exception:
            pass
        try:
            self.visualization_timer.cancel()
        except Exception:
            pass

        self.get_logger().info(f"Stopping at step {self.sim_step_counter}. Shutting down rclpy...")
        try:
            rclpy.shutdown()
        except Exception:
            pass

    def _compute_path_error(self) -> float:
        try:
            bx, by = float(self.bot.x), float(self.bot.y)
        except Exception:
            return 0.0
        try:
            _, (cx, cy) = self.path_polygon.find_closest_segment(bx, by)
        except Exception:
            return 0.0
        return math.hypot(bx - cx, by - cy)

    def _update_simulation_step(self):
        dt = 0.05
        self.sim_step_counter += 1

        t0 = time.perf_counter()
        step_output = self.bot.step(dt)
        step_time_s = time.perf_counter() - t0

        reward, done = self._extract_reward_done(step_output)
        max_ep_steps = int(getattr(args, "max_episode_steps", 300))
        if not done and (self.episode_step_count + 1) >= max_ep_steps:
            done = True
            if args.verbose:
                self.get_logger().info(f"Episode forced done at {max_ep_steps} steps.")

        # Compute error for metrics
        error_mag = self._compute_path_error()
        error_for_metrics = max(error_mag, 1e-9)
        self.metrics_tracker.record_step(reward=reward, step_time_s=step_time_s, done=done, error=error_for_metrics)

        # Per-step metrics
        try:
            per_step_row = self.metrics_tracker.compute_step_metrics(last_reward=reward, last_step_time_s=step_time_s)
            self._write_metrics_row(per_step_row)
        except Exception as e:
            if args.verbose:
                self.get_logger().warn(f"Per-step metrics write failed: {e}")

        # Train if needed
        if args.mode.lower() == "train":
            if (self.sim_step_counter % max(1, int(args.train_every))) == 0:
                try:
                    self.agent.train()
                except Exception as e:
                    if args.verbose:
                        self.get_logger().warn(f"Agent.train() failed at step {self.sim_step_counter}: {e}")

        # Publish pose and marker
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = 'world'
        pose_msg.pose.position.x = float(self.bot.x)
        pose_msg.pose.position.y = float(self.bot.y)
        pose_msg.pose.position.z = 0.0
        pose_msg.pose.orientation.w = 1.0
        self.bot_pose_pub.publish(pose_msg)

        bot_marker = Marker()
        bot_marker.header.frame_id = 'world'
        bot_marker.header.stamp = self.get_clock().now().to_msg()
        bot_marker.ns = 'bot'
        bot_marker.id = 0
        bot_marker.type = Marker.CUBE
        bot_marker.action = Marker.ADD
        bot_marker.pose.position.x = float(self.bot.x)
        bot_marker.pose.position.y = float(self.bot.y)
        bot_marker.pose.position.z = 0.0
        bot_marker.scale.x = bot_marker.scale.y = bot_marker.scale.z = 0.05
        bot_marker.color.a = 1.0
        bot_marker.color.r = 0.0
        bot_marker.color.g = 1.0
        bot_marker.color.b = 0.0
        self.bot_marker_pub.publish(bot_marker)
        
        target_pose = PoseStamped()
        target_pose.header.stamp = self.get_clock().now().to_msg()
        target_pose.header.frame_id = 'world'
        target_pose.pose.position.x = float(self.bot.x)
        target_pose.pose.position.y = float(self.bot.y)
        target_pose.pose.position.z = 0.1
        target_pose.pose.orientation.w = 1.0
        self.target_pub.publish(target_pose)

        try:
            self._publish_all_path_markers()
            self.moveit_command()
        except Exception:
            pass

        just_flushed = False
        if done:
            self.episode_step_count = 0
            if getattr(args, "flush_on_episode_end", False):
                try:
                    metrics_row = self.metrics_tracker.flush_window()
                    self._write_metrics_row(metrics_row)
                    just_flushed = True
                    if args.verbose and metrics_row is not None:
                        self.get_logger().info(f'Episode metrics flushed.')
                    self._reset_bot_position()
                except Exception as e:
                    if args.verbose:
                        self.get_logger().warn(f"Metrics flush on episode end failed: {e}")
        else:
            self.episode_step_count += 1

        # Periodic save and reset
        if (self.sim_step_counter % max(1, int(args.save_every))) == 0:
            if not just_flushed:
                metrics_row = self.metrics_tracker.flush_window()
                self._write_metrics_row(metrics_row)
                if args.verbose and metrics_row is not None:
                    self.get_logger().info(f'Metrics saved at step {self.sim_step_counter}.')
                self._reset_bot_position()
            try:
                self.agent.save_policy(self.actor_ckpt_path, self.critic_ckpt_path)
                if args.verbose:
                    self.get_logger().info(f'Saved agent to {self.actor_ckpt_path} and {self.critic_ckpt_path}')
            except Exception as e:
                if args.verbose:
                    self.get_logger().warn(f"Policy save failed: {e}")

        # Multi-path stop and advance
        if self.multi_paths:
            if self.sim_step_counter >= int(self.steps_per_path):
                # Save current path snapshot before switching
                try:
                    metrics_row = self.metrics_tracker.flush_window()
                    self._write_metrics_row(metrics_row)
                    self.agent.save_policy(self.actor_ckpt_path, self.critic_ckpt_path)
                except Exception as e:
                    if args.verbose:
                        self.get_logger().warn(f"Pre-switch save failed: {e}")
                self._advance_to_next_path()
                return

        # Single-run stop
        if (getattr(self, "stop_after_steps", None) is not None and
                self.sim_step_counter >= int(self.stop_after_steps) and
                not self.multi_paths):
            self._request_stop(already_flushed=just_flushed)


def main():
    parser = argparse.ArgumentParser(description="Run Ursina Simulation (train/eval RL or Quantum RL on chosen path)")
    parser.add_argument("--max_radius", type=float, default=0.8, help="Maximum workspace radius")
    parser.add_argument("--offset_x", type=float, default=0.0, help="X offset of workspace center")
    parser.add_argument("--offset_y", type=float, default=0.0, help="Y offset of workspace center")
    parser.add_argument("--verbose", action="store_true", help="Show logs of the agent")
    parser.add_argument("--agent", choices=["rl", "qrl", "rl_no_her"], default="qrl", help="Agent: rl (classical) or qrl (quantum DPG+HER) or rl_no_her(classical no HER)")
    parser.add_argument("--mode", choices=["train", "eval"], default="train", help="Mode")
    parser.add_argument("--noise_std", type=float, default=0.1, help="Exploration noise (train). Eval forces 0.0")
    parser.add_argument("--n_qubits", type=int, default=3, help="Qubits for quantum agent")
    parser.add_argument("--n_layers", type=int, default=1, help="Layers for quantum circuit")
    parser.add_argument("--backend", type=str, default=None, help="PennyLane backend (e.g., lightning.qubit)")
    parser.add_argument("--shots", type=int, default=None, help="Shots for quantum backend (None for analytic)")
    parser.add_argument("--no_warmup", action="store_true", help="Disable quantum warm-up forwards")
    parser.add_argument("--actor_path", type=str, default=None, help="Actor weights path")
    parser.add_argument("--critic_path", type=str, default=None, help="Critic weights path")
    parser.add_argument("--skip_load", action="store_true", help="Do not load weights at startup")
    parser.add_argument("--train_every", type=int, default=1, help="Call agent.train() every N steps in train mode")
    parser.add_argument("--save_every", type=int, default=1000, help="Save metrics and policy every N steps")
    parser.add_argument("--path", choices=["circle", "figure8", "triangle", "square", "pentagon", "star"], default="circle", help="Path shape")
    parser.add_argument("--resolution", type=int, default=24, help="Resolution for circle/figure8/star")
    parser.add_argument("--radius", type=float, default=1.0, help="Radius/scale for generated shapes")
    parser.add_argument("--max_episode_steps", type=int, default=1000, help="Force episode end after N steps")
    parser.add_argument("--stop_after_steps", type=int, default=None, help="Stop simulation after N global steps")
    parser.add_argument("--flush_on_episode_end", action="store_true", help="Flush metrics on each episode end")

    # New: multi-path schedule
    parser.add_argument("--multi_paths", type=str, default=None, help="Comma-separated list of paths to run sequentially (e.g., circle,triangle,square)")
    parser.add_argument("--steps_per_path", type=int, default=None, help="Global steps per path when using --multi_paths")
    parser.add_argument("--reset_agent_per_path", action="store_true", help="Reset (re-instantiate) the agent for each path when using --multi_paths")

    global args
    args = parser.parse_args()

    rclpy.init(args=None)
    node = PathFollowerSimulator()
    rclpy.spin(node)
    node.destroy_node()
    try:
        rclpy.shutdown()
    except Exception:
        pass


if __name__ == '__main__':
    main()