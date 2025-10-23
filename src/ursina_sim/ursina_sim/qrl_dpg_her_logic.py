import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import math
from typing import List, Dict, Any, Optional

# --------- Default Hyperparameters ---------
DEFAULT_GAMMA = 0.99             # Discount factor for future rewards (gamma)
DEFAULT_TARGET_UPDATE = 0.005        # Soft update interpolation factor (tau)
DEFAULT_ACTOR_LR = 1e-4              # Actor neural network learning rate
DEFAULT_CRITIC_LR = 1e-3             # Critic neural network learning rate
DEFAULT_REPLAY_BUFFER_SIZE = 100000  # Replay buffer length
DEFAULT_BATCH_SIZE = 64              # Minibatch size for training

# --------- Neural Network Definitions ---------

class Actor(nn.Module):
    """
    The Actor network produces continuous actions for the environment,
    given the current state as input.
    """
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim), nn.Tanh()  # Outputs in [-1, 1]
        )

    def forward(self, state):
        return self.model(state)

class Critic(nn.Module):
    """
    The Critic network estimates the value (Q-value) of a (state, action) pair.
    """
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        # Concatenate state and action along the last dimension
        input_tensor = torch.cat([state, action], dim=-1)
        return self.model(input_tensor)

# --------- Experience Replay Buffer with HER ---------

class HERReplayBuffer:
    """
    Experience replay buffer for RL, with support for storing achieved goals (HER).
    Stores up to `capacity` transitions.
    """
    def __init__(self, capacity=DEFAULT_REPLAY_BUFFER_SIZE):
        self.buffer = deque(maxlen=capacity)

    def push(self, transition):
        # Each transition: (state, action, reward, next_state, goal, done, achieved)
        self.buffer.append(transition)

    def sample(self, batch_size, device):
        batch = random.sample(self.buffer, min(len(self.buffer), batch_size))
        states, actions, rewards, next_states, goals, dones, achieved = zip(*batch)
        states = torch.FloatTensor(np.stack(states)).to(device)
        actions = torch.FloatTensor(np.stack(actions)).to(device)
        rewards = torch.FloatTensor(np.stack(rewards)).reshape(-1, 1).to(device)
        next_states = torch.FloatTensor(np.stack(next_states)).to(device)
        goals = torch.FloatTensor(np.stack(goals)).to(device)
        dones = torch.FloatTensor(np.stack(dones)).reshape(-1, 1).to(device)
        achieved = torch.FloatTensor(np.stack(achieved)).to(device)
        return states, actions, rewards, next_states, goals, dones, achieved

    def __len__(self):
        return len(self.buffer)

# --------- The Agent Class ---------

class QRLDPGHERAgent:
    """
    Deep Deterministic Policy Gradient agent with Hindsight Experience Replay.
    Trains Actor (policy) and Critic (value) networks to learn continuous control.
    """
    def __init__(
        self,
        state_dim,
        n_actions=1,
        gamma=DEFAULT_GAMMA,
        tau=DEFAULT_TARGET_UPDATE,
        lr_actor=DEFAULT_ACTOR_LR,
        lr_critic=DEFAULT_CRITIC_LR,
        buffer_size=DEFAULT_REPLAY_BUFFER_SIZE,
        batch_size=DEFAULT_BATCH_SIZE,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            state_dim: Dimension of the state input
            n_actions: Number of output action dimensions
            gamma: Discount for future rewards
            tau: Soft target network interpolation factor
            lr_actor: Learning rate for the actor
            lr_critic: Learning rate for the critic
            buffer_size: Max transitions to keep in memory
            batch_size: How many transitions to use per training step
            device: torch device ('cpu' or 'cuda')
        """
        self.device = (
            device if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.discount = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.action_dim = int(n_actions)
        self.state_dim = int(state_dim)

        # Actor and Target Actor networks
        self.actor = Actor(self.state_dim, self.action_dim).to(self.device)
        self.actor_target = Actor(self.state_dim, self.action_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        # Critic and Target Critic networks
        self.critic = Critic(self.state_dim, self.action_dim).to(self.device)
        self.critic_target = Critic(self.state_dim, self.action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Setup optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # Setup experience replay buffer (HER-enabled)
        self.replay_buffer = HERReplayBuffer(capacity=buffer_size)

        # Bookkeeping
        self.total_steps = 0
        self.total_episodes = 0

    def select_action(self, state_obs, goal=None, action_noise_std=0.1):
        """
        Given a state, return a (noisy) action for exploration.
        Args:
            state_obs: numpy array or torch tensor, shape (state_dim,)
            goal: unused in this implementation (for future HER potential)
            action_noise_std: standard deviation for exploration noise
        Returns:
            Clipped action (numpy array in [-1, 1])
        """
        # Ensure state is a (1, state_dim) tensor on the correct device
        state_tensor = (
            state_obs.to(self.device).unsqueeze(0) if isinstance(state_obs, torch.Tensor) and state_obs.dim() == 1
            else torch.FloatTensor(np.array(state_obs, dtype=np.float32)).unsqueeze(0).to(self.device)
        )
        action = self.actor(state_tensor).detach().cpu().numpy()[0]
        if action_noise_std and action_noise_std > 0:
            action = action + action_noise_std * np.random.randn(*action.shape)
        return np.clip(action, -1.0, 1.0)

    def remember(self, transition):
        """
        Store a transition to the replay buffer, converting values to numpy arrays as needed.
        Args:
            transition: (state, action, reward, next_state, goal, done, achieved)
        """
        s, a, r, s2, goal, done, achieved = transition
        s = np.asarray(s, dtype=np.float32).reshape(-1)
        a = np.asarray(a, dtype=np.float32).reshape(-1)
        s2 = np.asarray(s2, dtype=np.float32).reshape(-1)
        goal = np.asarray(goal, dtype=np.float32).reshape(-1)
        achieved = np.asarray(achieved, dtype=np.float32).reshape(-1)
        self.replay_buffer.push((s, a, float(r), s2, goal, float(done), achieved))
        self.total_steps += 1

    def state_from_errors(self, distance_error, heading_error, tangent_angle):
        """
        Returns a tensor state representation for the agent, to be used in networks.
        Combines distance error, heading error, and sin/cos of tangent phase for smooth cyclic input.
        """
        sin_phase = math.sin(tangent_angle)
        cos_phase = math.cos(tangent_angle)
        state = torch.tensor([distance_error, heading_error, sin_phase, cos_phase],
                             dtype=torch.float32, device=self.device)
        return state

    def _soft_update(self, source_net, target_net):
        """
        Soft-updates target parameters from the source network, using tau.
        """
        for param, target_param in zip(source_net.parameters(), target_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def train(self, her_ratio=0.8):
        """
        Performs a training update for the RL agent, including HER-style goal relabeling.
        Args:
            her_ratio: Probability of substituting the actual achieved goal for the desired one.
        """
        if len(self.replay_buffer) < self.batch_size:
            return  # Not enough experience for a minibatch yet

        states, actions, rewards, next_states, goals, dones, achieved = self.replay_buffer.sample(
            self.batch_size, self.device)

        # HER relabeling: mix in achieved goals with probability her_ratio
        prob = (torch.rand(states.size(0), 1, device=self.device) < her_ratio)
        goals = goals.unsqueeze(-1) if goals.dim() == 1 else goals
        achieved = achieved.unsqueeze(-1) if achieved.dim() == 1 else achieved
        relabel_mask = prob.expand_as(goals)
        relabeled_goals = torch.where(relabel_mask, achieved, goals)

        # Critic loss: Bellman equation target
        with torch.no_grad():
            next_action = self.actor_target(next_states)
            target_q = rewards + (1.0 - dones) * self.discount * self.critic_target(next_states, next_action)
        critic_prediction = self.critic(states, actions)
        critic_loss = nn.MSELoss()(critic_prediction, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor loss: maximize expected Q-value under policy
        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Polyak (soft) update of the target networks for stability
        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)

        self.total_episodes += 1

    def save_policy(self, actor_path="qrl_dpg_actor.pt", critic_path="qrl_dpg_critic.pt"):
        """
        Save the actor and critic network weights to disk.
        """
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    def load_policy(self, actor_path="qrl_dpg_actor.pt", critic_path="qrl_dpg_critic.pt"):
        """
        Load actor and critic network weights from disk, if they exist.
        """
        try:
            self.actor.load_state_dict(torch.load(actor_path, map_location=self.device))
            self.critic.load_state_dict(torch.load(critic_path, map_location=self.device))
            print("Loaded saved DPG+HER policy.")
        except Exception:
            print("No saved DPG+HER policy found, starting fresh.")