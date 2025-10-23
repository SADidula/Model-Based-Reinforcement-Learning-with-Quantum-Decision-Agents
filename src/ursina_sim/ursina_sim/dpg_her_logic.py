import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import math
from typing import Optional, Tuple, List

DEFAULT_GAMMA = 0.99
DEFAULT_TARGET_UPDATE = 0.005
DEFAULT_ACTOR_LR = 1e-4
DEFAULT_CRITIC_LR = 1e-3
DEFAULT_REPLAY_BUFFER_SIZE = 100000
DEFAULT_BATCH_SIZE = 64

# ----------------- Networks (goal-conditioned) -----------------

class Actor(nn.Module):
    """
    Actor(s, g) -> a in [-1, 1]
    Architecture mirrors the simpler QRL style for stability.
    """
    def __init__(self, state_dim: int, goal_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim + goal_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )

    def forward(self, state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, goal], dim=-1)
        return self.model(x)


class Critic(nn.Module):
    """
    Critic(s, g, a) -> Q
    """
    def __init__(self, state_dim: int, goal_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim + goal_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state: torch.Tensor, goal: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, goal, action], dim=-1)
        return self.model(x)

# ----------------- Replay Buffer (HER-ready) -----------------

class HERReplayBuffer:
    """
    Stores transitions with achieved and achieved_next for HER.
    Transition: (state, action, reward, next_state, goal, done, achieved, achieved_next)
    """
    def __init__(self, capacity=DEFAULT_REPLAY_BUFFER_SIZE):
        self.buffer = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.buffer)

    def push(self, transition, achieved_next: Optional[np.ndarray] = None):
        # (state, action, reward, next_state, goal, done, achieved)
        s, a, r, s2, goal, done, achieved = transition
        s = np.asarray(s, dtype=np.float32).reshape(-1)
        a = np.asarray(a, dtype=np.float32).reshape(-1)
        s2 = np.asarray(s2, dtype=np.float32).reshape(-1)
        goal = np.asarray(goal, dtype=np.float32).reshape(-1)
        achieved = np.asarray(achieved, dtype=np.float32).reshape(-1)
        if achieved_next is None:
            achieved_next = achieved.copy()
        else:
            achieved_next = np.asarray(achieved_next, dtype=np.float32).reshape(-1)
        self.buffer.append((s, a, float(r), s2, goal, float(done), achieved, achieved_next))

    def sample(self, batch_size: int, device: torch.device):
        batch = random.sample(self.buffer, min(len(self.buffer), batch_size))
        s, a, r, s2, goal, d, ach, ach_n = zip(*batch)
        states = torch.tensor(np.stack(s), dtype=torch.float32, device=device)
        actions = torch.tensor(np.stack(a), dtype=torch.float32, device=device)
        rewards_env = torch.tensor(np.stack(r), dtype=torch.float32, device=device).view(-1, 1)
        next_states = torch.tensor(np.stack(s2), dtype=torch.float32, device=device)
        goals = torch.tensor(np.stack(goal), dtype=torch.float32, device=device)
        dones = torch.tensor(np.stack(d), dtype=torch.float32, device=device).view(-1, 1)
        achieved = torch.tensor(np.stack(ach), dtype=torch.float32, device=device)
        achieved_next = torch.tensor(np.stack(ach_n), dtype=torch.float32, device=device)
        return states, actions, rewards_env, next_states, goals, dones, achieved, achieved_next

# ----------------- Agent -----------------

def _wrap_angle_t(x: torch.Tensor) -> torch.Tensor:
    return (x + math.pi) % (2 * math.pi) - math.pi

class RLDPGHERAgent:
    """
    Goal-conditioned DDPG with simple HER, tuned to behave like the original QRL agent:
      - Single actor/critic, soft targets
      - Targets use environment reward only (stable like QRL)
      - HER relabels goals in-batch (no reward change), gentle early
      - Actor uses original goals for first steps, then relabeled goals
    API:
      - select_action(state, goal, noise_std=None)
      - remember(transition, achieved_next=None)
      - state_from_errors(distance_error, heading_error, tangent_angle)
    """
    def __init__(
        self,
        state_dim: int,
        goal_dim: int = 2,
        n_actions: int = 1,
        gamma: float = DEFAULT_GAMMA,
        tau: float = DEFAULT_TARGET_UPDATE,
        lr_actor: float = DEFAULT_ACTOR_LR,
        lr_critic: float = DEFAULT_CRITIC_LR,
        buffer_size: int = DEFAULT_REPLAY_BUFFER_SIZE,
        batch_size: int = DEFAULT_BATCH_SIZE,
        device: Optional[torch.device] = None,
    ):
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.discount = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.action_dim = int(n_actions)
        self.state_dim = int(state_dim)
        self.goal_dim = int(goal_dim)

        self.actor = Actor(self.state_dim, self.goal_dim, self.action_dim).to(self.device)
        self.actor_target = Actor(self.state_dim, self.goal_dim, self.action_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(self.state_dim, self.goal_dim, self.action_dim).to(self.device)
        self.critic_target = Critic(self.state_dim, self.goal_dim, self.action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.replay_buffer = HERReplayBuffer(capacity=buffer_size)

        # reward shaping for path following (kept for potential future use)
        self.angle_idx = 1            # heading error index in goal vector [distance_error, heading_error]
        self.w_dist = 3.0             # cross-track weight
        self.w_head = 1.0             # heading weight
        self.tol_head = math.radians(5.0)

        # exploration and HER schedule (gentler early)
        self.reward_scale = 1.0
        self.her_ratio = 0.3          # start lower, will anneal up
        self.action_noise_std = 0.05   # modest noise like QRL

        # bookkeeping
        self.total_steps = 0
        self.total_episodes = 0

    # ------------ API ------------

    def select_action(self, state_obs, goal, action_noise_std: Optional[float] = None):
        # state_obs: np.ndarray (state_dim,), goal: np.ndarray (goal_dim,)
        s = torch.as_tensor(state_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        g = torch.as_tensor(goal, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            a = self.actor(s, g).cpu().numpy()[0]
        std = self.action_noise_std if action_noise_std is None else action_noise_std
        if std and std > 0:
            a = a + std * np.random.randn(*a.shape)
        return np.clip(a, -1.0, 1.0)

    def remember(self, transition, achieved_next: Optional[np.ndarray] = None):
        self.replay_buffer.push(transition, achieved_next=achieved_next)
        self.total_steps += 1

    def state_from_errors(self, distance_error: float, heading_error: float, tangent_angle: float):
        sin_phase = math.sin(tangent_angle)
        cos_phase = math.cos(tangent_angle)
        state = torch.tensor([distance_error, heading_error, sin_phase, cos_phase],
                             dtype=torch.float32, device=self.device)
        return state

    def _soft_update(self, src: nn.Module, tgt: nn.Module):
        for p, tp in zip(src.parameters(), tgt.parameters()):
            tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)

    # ------------ Train ------------

    def train(self, her_ratio: Optional[float] = None):
        if len(self.replay_buffer) < self.batch_size:
            return

        # Anneal HER ratio up a bit over time
        if (self.total_steps % 4000) == 0 and self.her_ratio < 0.8:
            self.her_ratio = min(0.8, self.her_ratio + 0.05)

        her_r = self.her_ratio if her_ratio is None else float(her_ratio)
        states, actions, rewards_env, next_states, goals, dones, achieved, achieved_next = \
            self.replay_buffer.sample(self.batch_size, self.device)

        # HER relabeling: replace some goals with achieved_next (do not change reward definition)
        mask = (torch.rand(states.size(0), 1, device=self.device) < her_r)
        relabeled_goals = torch.where(mask.expand_as(goals), achieved_next, goals)

        # Use environment reward only for target (like QRL)
        rewards = rewards_env

        # Target Q using relabeled goals (stable env reward)
        with torch.no_grad():
            next_action = self.actor_target(next_states, relabeled_goals)
            target_q = rewards + (1.0 - dones) * self.discount * self.critic_target(next_states, relabeled_goals, next_action)

        # Critic update
        q_pred = self.critic(states, relabeled_goals, actions)
        critic_loss = nn.MSELoss()(q_pred, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor update: stabilize early using original goals, then switch to relabeled
        goals_for_actor = goals if self.total_steps < 5000 else relabeled_goals
        a_pred = self.actor(states, goals_for_actor)
        actor_loss = -self.critic(states, goals_for_actor, a_pred).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft updates
        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)

    # ------------ Save/Load ------------

    def save_policy(self, actor_path="rl_dpg_actor.pt", critic_path="rl_dpg_critic.pt"):
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    def load_policy(self, actor_path="rl_dpg_actor.pt", critic_path="rl_dpg_critic.pt"):
        try:
            self.actor.load_state_dict(torch.load(actor_path, map_location=self.device))
            self.critic.load_state_dict(torch.load(critic_path, map_location=self.device))
            self.actor_target.load_state_dict(self.actor.state_dict())
            self.critic_target.load_state_dict(self.critic.state_dict())
            print("Loaded saved DPG+HER policy.")
        except Exception:
            print("No saved DPG+HER policy found, starting fresh.")