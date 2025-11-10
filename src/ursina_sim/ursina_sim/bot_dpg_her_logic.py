import math
import random
import torch
from typing import Tuple, List
from .dpg_her_logic import RLDPGHERAgent
from .quantum_dpg_her_logic import QRLDPGHERAgent

class BotDPGHER:
    """
    DPG+HER-controlled bot:
    - Computes distance and heading error to local path tangent
    - Moves with bounded turn rate and fixed forward speed
    - Rewards: progress along tangent, strong cross-track/heading penalties, on-track bonus
    - Supports HER through achieved and achieved_next in remember()
    """

    def __init__(
        self,
        rl_agent: RLDPGHERAgent | QRLDPGHERAgent,
        main_path,
        boundary_penalties: List = None,
        max_speed: float = 6.0,
        max_turn_speed: float = 20.0,
        initial_position: Tuple[float, float] = (0.0, 0.0),
        # workspace clamp
        x_min: float = -6.0, x_max: float = 6.0, y_min: float = -6.0, y_max: float = 6.0
    ):
        self.rl_agent = rl_agent
        self.main_path = main_path
        self.boundary_penalties = boundary_penalties or []
        self.max_speed = float(max_speed)
        self.max_turn_speed = float(max_turn_speed)
        self.x, self.y = float(initial_position[0]), float(initial_position[1])
        self.heading = 0.0

        # reward shaping
        self.w_dist = 0.25
        self.w_head = 0.12
        self.progress_coef = 1.0
        self.progress_decay = 0.35
        self.bonus_distance_threshold = 0.6
        self.bonus_heading_threshold = math.radians(8.0)
        self.on_track_bonus = 2.0

        # workspace
        self.x_min, self.x_max = float(x_min), float(x_max)
        self.y_min, self.y_max = float(y_min), float(y_max)

        # tracking
        self.previous_state = None
        self.previous_action = None
        self.previous_goal = None
        self.previous_achieved = None

        self.total_angle_progress = 0.0
        self.laps_completed = 0
        self.recent_positions = []

    @staticmethod
    def wrap_angle(angle: float) -> float:
        return (angle + math.pi) % (2 * math.pi) - math.pi

    def _clamp_workspace(self):
        clamped = False
        if self.x < self.x_min:
            self.x = self.x_min; clamped = True
        elif self.x > self.x_max:
            self.x = self.x_max; clamped = True
        if self.y < self.y_min:
            self.y = self.y_min; clamped = True
        elif self.y > self.y_max:
            self.y = self.y_max; clamped = True
        if clamped:
            # small penalty by reducing bonus opportunity next step (implicit)
            pass

    def compute_path_errors(self):
        distance_error = self.main_path.radial_error(self.x, self.y)
        target_tangent = self.main_path.tangent_heading(self.x, self.y)
        heading_error = self.wrap_angle(self.heading - target_tangent)
        return distance_error, heading_error, target_tangent

    def penalty_zone_index(self, x, y) -> int:
        for idx, (boundary, _penalty) in enumerate(self.boundary_penalties):
            if not boundary.find_closest_segment(x, y):
                return idx
        return -1

    def compute_reward(self, distance_error: float, heading_error: float, delta_time: float) -> float:
        penalty_index = self.penalty_zone_index(self.x, self.y)
        if penalty_index != -1:
            _, penalty_multiplier = self.boundary_penalties[penalty_index]
            return -penalty_multiplier - 3.0 * abs(distance_error)

        forward_speed = self.max_speed * max(0.0, math.cos(heading_error))
        forward_progress = forward_speed * delta_time
        progress_decay = math.exp(-self.progress_decay * abs(distance_error))
        progress_reward = self.progress_coef * forward_progress * progress_decay

        error_penalty = (self.w_dist * abs(distance_error) + self.w_head * abs(heading_error)) * delta_time

        step_reward = progress_reward - error_penalty

        if abs(distance_error) < self.bonus_distance_threshold and abs(heading_error) < self.bonus_heading_threshold:
            step_reward += self.on_track_bonus

        return float(step_reward)

    def step(self, delta_time: float, episode_done: bool = False):
        dist_err, head_err, target_angle = self.compute_path_errors()
        rl_state = self.rl_agent.state_from_errors(dist_err, head_err, target_angle)

        rl_action = self.rl_agent.select_action(rl_state.cpu().numpy())
        steering_command = float(rl_action[0]) * self.max_turn_speed

        # integrate
        self.heading = self.wrap_angle(self.heading + steering_command * delta_time)
        self.x += math.cos(self.heading) * self.max_speed * delta_time
        self.y += math.sin(self.heading) * self.max_speed * delta_time
        self._clamp_workspace()

        new_dist_err, new_head_err, new_target_angle = self.compute_path_errors()

        # angle progress for laps
        if not hasattr(self, 'prev_angle_progress'):
            angle_delta = 0.0
        else:
            angle_delta = self.wrap_angle(new_target_angle - getattr(self, 'prev_angle_progress', 0.0))

        reward = self.compute_reward(new_dist_err, new_head_err, delta_time)

        self.total_angle_progress += angle_delta
        if abs(self.total_angle_progress) > 2 * math.pi:
            reward += 50.0
            self.total_angle_progress = 0.0
            self.laps_completed += 1

        self.last_reward = float(reward)
        self.last_done = bool(episode_done)

        goal = torch.tensor([0.0, 0.0], dtype=torch.float32)
        achieved = torch.tensor([new_dist_err, new_head_err], dtype=torch.float32)

        # store transition (use achieved as achieved_next, bot-step switches to next_state immediately)
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
            self.rl_agent.remember(transition, achieved_next=achieved.cpu().numpy())

        self.previous_state = rl_state
        self.previous_action = rl_action
        self.previous_goal = goal
        self.previous_achieved = torch.tensor([dist_err, head_err], dtype=torch.float32)
        self.prev_angle_progress = new_target_angle

        self.recent_positions.append((self.x, self.y))
        if len(self.recent_positions) > 1024:
            self.recent_positions.pop(0)

    def finish_episode(self):
        self.rl_agent.train()

    def reset(self, location: Tuple[float, float] = None):
        if location is not None:
            self.x, self.y = location
        else:
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