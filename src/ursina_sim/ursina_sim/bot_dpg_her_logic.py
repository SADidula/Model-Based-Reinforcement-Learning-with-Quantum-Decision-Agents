import math
import random
import torch
from typing import Tuple, List
from .qrl_dpg_her_logic import QRLDPGHERAgent

class BotDPGHER:
    """
    This bot is controlled by a QRL DPG+HER agent.
    It moves in a simulated world, tries to follow a given path,
    and gets rewards (or penalties) based on performance and boundary crossings.
    """
    def __init__(
        self,
        rl_agent: QRLDPGHERAgent,
        main_path,
        boundary_penalties: List = None,
        max_speed: float = 6.0,
        max_turn_speed: float = 20.0,
        initial_position: Tuple[float, float] = (0.0, 0.0)
    ):
        """
        Args:
            rl_agent: The learning brain (policy network).
            main_path: PolygonPath providing geometric reference.
            boundary_penalties: List of (PolygonPath, penalty_multiplier)
            max_speed: forward speed per sim step.
            max_turn_speed: rotation radians/sec per sim step.
            initial_position: (x, y) spawn/start.
        """
        self.rl_agent = rl_agent
        self.main_path = main_path
        self.boundary_penalties = boundary_penalties or []
        self.max_speed = float(max_speed)
        self.max_turn_speed = float(max_turn_speed)
        self.x, self.y = float(initial_position[0]), float(initial_position[1])
        self.heading = 0.0  # facing angle (radians, 0 is +x)

        # Reward shaping coefficients (tunable)
        # Error weights (penalize distance and heading deviation)
        self.w_dist = 0.2
        self.w_head = 0.1
        # Forward progress reward scale (higher -> stronger push to move forward)
        self.progress_coef = 1.0
        # Decays progress reward as the bot drifts from the path
        self.progress_decay = 0.3
        # "On track" bonus thresholds
        self.bonus_distance_threshold = 1.0
        self.bonus_heading_threshold = math.radians(10.0)
        self.on_track_bonus = 2.0

        # Tracking and learning (for RL/stateful features)
        self.previous_state = None
        self.previous_action = None
        self.previous_goal = None
        self.previous_achieved = None

        self.total_angle_progress = 0.0
        self.laps_completed = 0
        self.recent_positions = []

    @staticmethod
    def wrap_angle(angle: float) -> float:
        """
        Normalize an angle to [-pi, pi].
        """
        return (angle + math.pi) % (2 * math.pi) - math.pi

    def compute_path_errors(self) -> Tuple[float, float]:
        """
        Determine current path-following errors: lateral (distance) and heading.
        Returns:
            distance_error: signed distance to path
            heading_error: angle difference to path tangent
        """
        distance_error = self.main_path.radial_error(self.x, self.y)
        target_tangent = self.main_path.tangent_heading(self.x, self.y)
        heading_error = self.wrap_angle(self.heading - target_tangent)
        return distance_error, heading_error

    def penalty_zone_index(self, x, y) -> int:
        """
        Find which penalty boundary (if any) is violated, and return its index.
        Returns:
            The index of the first boundary outside which the bot is located,
            or -1 if inside all boundaries (i.e., no penalty now).
        Note:
            This assumes boundary.find_closest_segment returns a truthy value when in-range.
            If your PolygonPath supports a point-in-polygon test, prefer that here.
        """
        for idx, (boundary, _penalty) in enumerate(self.boundary_penalties):
            if not boundary.find_closest_segment(x, y):
                return idx
        return -1

    def compute_reward(self, distance_error: float, heading_error: float, angle_delta: float, delta_time: float) -> float:
        """
        Compute per-step reward:

        1) Boundary violation: strong penalty, proportional to boundary penalty and distance.
        2) Forward progress: reward proportional to motion along the local tangent,
           decayed by how far off the path we are.
           progress ~= max(0, cos(heading_error)) * speed * dt
        3) Smoothness penalties: penalize lateral/heading errors.
        4) Small on-track bonus when within tight thresholds.

        Important:
        - All time-dependent terms scale with delta_time (dt) to be step-size agnostic.
        - angle_delta is not used for per-step reward here (only for lap tracking).
        """
        # 1) Boundary violation
        penalty_index = self.penalty_zone_index(self.x, self.y)
        if penalty_index != -1:
            _, penalty_multiplier = self.boundary_penalties[penalty_index]
            return -penalty_multiplier - 3.0 * abs(distance_error)

        # 2) Forward progress along the path tangent (non-negative)
        #    Rewards forward motion (aligned with tangent); discourages backward motion implicitly.
        forward_speed = self.max_speed * max(0.0, math.cos(heading_error))  # project velocity onto tangent
        forward_progress = forward_speed * delta_time
        # decay progress when far from path
        progress_decay = math.exp(-self.progress_decay * abs(distance_error))
        progress_reward = self.progress_coef * forward_progress * progress_decay

        # 3) Error penalties (scaled by dt to match progress scaling)
        error_penalty = (self.w_dist * abs(distance_error) + self.w_head * abs(heading_error)) * delta_time

        step_reward = progress_reward - error_penalty

        # 4) On-track bonus if within tight bounds
        if abs(distance_error) < self.bonus_distance_threshold and abs(heading_error) < self.bonus_heading_threshold:
            step_reward += self.on_track_bonus

        return float(step_reward)

    def step(self, delta_time: float, episode_done: bool = False) -> None:
        """
        Do one simulation step:
        - Observe path/distance/heading errors
        - Ask RL agent for action
        - Execute movement
        - Compute rewards
        - Feed experience back to agent's replay buffer

        Note:
        - Stores self.last_reward and self.last_done so external code can read them.
        """
        # Observe state (errors to path)
        distance_error, heading_error = self.compute_path_errors()
        target_angle = self.main_path.tangent_heading(self.x, self.y)
        rl_state = self.rl_agent.state_from_errors(distance_error, heading_error, target_angle)

        # Agent proposes an action (normalized steering command)
        rl_action = self.rl_agent.select_action(rl_state.cpu().numpy())
        steering_command = float(rl_action[0]) * self.max_turn_speed

        # Move bot using kinematics
        self.heading = self.wrap_angle(self.heading + steering_command * delta_time)
        self.x += math.cos(self.heading) * self.max_speed * delta_time
        self.y += math.sin(self.heading) * self.max_speed * delta_time

        # After moving: get rewards and new features
        new_distance_error, new_heading_error = self.compute_path_errors()
        new_target_angle = self.main_path.tangent_heading(self.x, self.y)

        # Track how much progress was made along the path (angularly from previous)
        if not hasattr(self, 'prev_angle_progress'):
            angle_delta = 0.0
        else:
            angle_delta = self.wrap_angle(new_target_angle - getattr(self, 'prev_angle_progress', 0.0))

        # Compute reward (no extra angle_delta terms here; avoid double-counting)
        reward = self.compute_reward(new_distance_error, new_heading_error, angle_delta, delta_time)

        # Give reward for a full lap (keep lap tracking via angle_delta)
        self.total_angle_progress += angle_delta
        if abs(self.total_angle_progress) > 2 * math.pi:
            reward += 50.0
            self.total_angle_progress = 0.0
            self.laps_completed += 1

        # Expose reward/done to outside readers (e.g., simulator metrics)
        self.last_reward = float(reward)
        self.last_done = bool(episode_done)

        # Prepare experience for HER (hindsight relabeling)
        goal = torch.tensor([0.0, 0.0], dtype=torch.float32)      # Target is zero error
        achieved_goal = torch.tensor([new_distance_error, new_heading_error], dtype=torch.float32)

        # Add transition to experience buffer
        if self.previous_state is not None:
            transition = (
                self.previous_state.cpu().numpy()[0],
                self.previous_action,
                reward,
                rl_state.cpu().numpy()[0],
                self.previous_goal.cpu().numpy(),
                float(episode_done),
                self.previous_achieved.cpu().numpy()
            )
            self.rl_agent.remember(transition)

        # Update for next step
        self.previous_state = rl_state
        self.previous_action = rl_action
        self.previous_goal = goal
        self.previous_achieved = achieved_goal
        self.prev_angle_progress = new_target_angle

        self.recent_positions.append((self.x, self.y))
        if len(self.recent_positions) > 1024:
            self.recent_positions.pop(0)

    def finish_episode(self):
        """
        Tell the RL agent to learn/update after an episode. (Call at end of episode.)
        """
        self.rl_agent.train()

    def reset(self, location: Tuple[float, float] = None):
        """
        Resets the bot state and location.
        If location is given, spawns there. Otherwise, random near-polygon center.
        """
        if location is not None:
            self.x, self.y = location
        else:
            # Random spawn near the center of the polygon (within 30% of polygon span)
            if hasattr(self.main_path, 'centroid'):
                cx, cy = self.main_path.centroid()
            else:
                verts = self.main_path.as_points()
                cx = sum([v[0] for v in verts]) / len(verts)
                cy = sum([v[1] for v in verts]) / len(verts)
            verts = self.main_path.as_points()
            min_span = min(
                max([v[0] for v in verts]) - min([v[0] for v in verts]),
                max([v[1] for v in verts]) - min([v[1] for v in verts])
            )
            r = random.uniform(0, min_span * 0.3)
            angle = random.uniform(-math.pi, math.pi)
            self.x = cx + r * math.cos(angle)
            self.y = cy + r * math.sin(angle)
        self.heading = random.uniform(-math.pi, math.pi)
        self.prev_angle_progress = 0.0
        self.previous_state = None
        self.previous_action = None
        self.previous_goal = None
        self.previous_achieved = None
        self.total_angle_progress = 0.0
        self.laps_completed = 0
        self.recent_positions.clear()