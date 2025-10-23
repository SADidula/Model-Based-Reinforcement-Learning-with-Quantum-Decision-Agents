import math
import random
from collections import deque
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Optional PennyLane import
try:
    import pennylane as qml
    PENNYLANE_AVAILABLE = True
except Exception:
    PENNYLANE_AVAILABLE = False


# --------- Default Hyperparameters (mirror original QRL) ---------
DEFAULT_GAMMA = 0.99
DEFAULT_TARGET_UPDATE = 0.005
DEFAULT_ACTOR_LR = 1e-4
DEFAULT_CRITIC_LR = 1e-3
DEFAULT_REPLAY_BUFFER_SIZE = 100000
DEFAULT_BATCH_SIZE = 64


# --------- Simple HER Replay Buffer (same shape as original) ---------
class HERReplayBuffer:
    """
    Stores transitions:
      (state, action, reward, next_state, goal, done, achieved)
    """
    def __init__(self, capacity: int = DEFAULT_REPLAY_BUFFER_SIZE):
        self.buffer = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.buffer)

    def push(self, transition: Tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray, float, np.ndarray]):
        s, a, r, s2, g, d, ach = transition
        s = np.asarray(s, dtype=np.float32).reshape(-1)
        a = np.asarray(a, dtype=np.float32).reshape(-1)
        s2 = np.asarray(s2, dtype=np.float32).reshape(-1)
        g = np.asarray(g, dtype=np.float32).reshape(-1)
        d = float(d)
        ach = np.asarray(ach, dtype=np.float32).reshape(-1)
        self.buffer.append((s, a, float(r), s2, g, d, ach))

    def sample(self, batch_size: int, device: torch.device):
        batch = random.sample(self.buffer, min(len(self.buffer), batch_size))
        s, a, r, s2, g, d, ach = zip(*batch)
        states = torch.tensor(np.stack(s), dtype=torch.float32, device=device)
        actions = torch.tensor(np.stack(a), dtype=torch.float32, device=device)
        rewards = torch.tensor(np.stack(r), dtype=torch.float32, device=device).view(-1, 1)
        next_states = torch.tensor(np.stack(s2), dtype=torch.float32, device=device)
        goals = torch.tensor(np.stack(g), dtype=torch.float32, device=device)
        dones = torch.tensor(np.stack(d), dtype=torch.float32, device=device).view(-1, 1)
        achieved = torch.tensor(np.stack(ach), dtype=torch.float32, device=device)
        return states, actions, rewards, next_states, goals, dones, achieved


# --------- Hybrid quantum-classical VQC block ---------
class VQCLayer(nn.Module):
    """
    Small hybrid layer:
      - Linear encoder maps input_dim -> n_qubits angles (scaled)
      - PennyLane circuit with StronglyEntanglingLayers
      - Linear head to out_dim
    If PennyLane is not available, falls back to a simple MLP head.
    """
    def __init__(
        self,
        input_dim: int,
        out_dim: int,
        n_qubits: int = 6,
        n_layers: int = 1,
        head_hidden: int = 128,
        backend: Optional[str] = None,
        shots: Optional[int] = None,
        input_scale: float = math.pi,
    ):
        super().__init__()
        self.use_quantum = PENNYLANE_AVAILABLE
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.input_scale = float(input_scale)

        # Shared encoder from input to qubit angles
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, n_qubits),
            nn.Tanh()
        )

        if self.use_quantum:
            if backend is None:
                backend = "default.qubit"
            self.dev = qml.device(backend, wires=n_qubits, shots=shots)

            def circuit(inputs, weights):
                qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
                qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
                return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

            weight_shapes = {"weights": (n_layers, n_qubits, 3)}
            diff_method = "adjoint" if shots is None else "parameter-shift"
            qnode = qml.QNode(circuit, self.dev, interface="torch", diff_method=diff_method)
            self.vqc = qml.qnn.TorchLayer(qnode, weight_shapes)
            head_in = n_qubits
        else:
            # Fallback: classical feature extractor
            self.vqc = nn.Sequential(
                nn.Linear(n_qubits, head_hidden),
                nn.ReLU(),
                nn.Linear(head_hidden, n_qubits),
                nn.ReLU(),
            )
            head_in = n_qubits

        self.head = nn.Sequential(
            nn.Linear(head_in, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, out_dim),
        )

        # Optional classical bypass with learnable blend (can help early convergence)
        self.classical_bypass = nn.Linear(input_dim, out_dim)
        self.alpha_q = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            x = x.view(x.size(0), -1)
        ang = self.encoder(x) * self.input_scale
        qfeat = self.vqc(ang)
        q_out = self.head(qfeat)
        c_out = self.classical_bypass(x)
        alpha = torch.clamp(self.alpha_q, 0.0, 1.0)
        return alpha * q_out + (1.0 - alpha) * c_out


# --------- Actor/Critic using VQC blocks (match original QRL API) ---------
class QActor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int,
                 n_qubits=6, n_layers=1, hidden=128, backend=None, shots=None):
        super().__init__()
        self.vqc = VQCLayer(
            input_dim=state_dim,
            out_dim=action_dim,
            n_qubits=n_qubits,
            n_layers=n_layers,
            head_hidden=hidden,
            backend=backend,
            shots=shots,
        )
        self.out_act = nn.Tanh()

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.out_act(self.vqc(state))


class QCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int,
                 n_qubits=6, n_layers=1, hidden=128, backend=None, shots=None):
        super().__init__()
        in_dim = state_dim + action_dim
        self.vqc = VQCLayer(
            input_dim=in_dim,
            out_dim=1,
            n_qubits=n_qubits,
            n_layers=n_layers,
            head_hidden=hidden,
            backend=backend,
            shots=shots,
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        return self.vqc(x)


# --------- Agent (Quantum-inspired variant of original QRL) ---------
class QRLDPGHERAgent:
    """
    Quantum-inspired DDPG agent with HER-ready buffer.
    Mirrors the original qrl_dpg_her_logic.py API and training behavior:
      - Actor(s)->a, Critic([s,a])->Q
      - Targets trained on environment reward
      - Simple HER goal storage (unused by networks here, but kept for compatibility)
      - Soft target updates, modest exploration noise
    """
    def __init__(
        self,
        state_dim: int,
        n_actions: int = 1,
        gamma: float = DEFAULT_GAMMA,
        tau: float = DEFAULT_TARGET_UPDATE,
        lr_actor: float = DEFAULT_ACTOR_LR,
        lr_critic: float = DEFAULT_CRITIC_LR,
        buffer_size: int = DEFAULT_REPLAY_BUFFER_SIZE,
        batch_size: int = DEFAULT_BATCH_SIZE,
        device: Optional[torch.device] = None,
        # Quantum params
        n_qubits: int = 6,
        n_layers: int = 1,
        hidden: int = 128,
        backend: Optional[str] = None,
        shots: Optional[int] = None,
    ):
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.discount = float(gamma)
        self.tau = float(tau)
        self.batch_size = int(batch_size)
        self.action_dim = int(n_actions)
        self.state_dim = int(state_dim)

        # Models
        self.actor = QActor(self.state_dim, self.action_dim, n_qubits, n_layers, hidden, backend, shots).to(self.device)
        self.actor_target = QActor(self.state_dim, self.action_dim, n_qubits, n_layers, hidden, backend, shots).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = QCritic(self.state_dim, self.action_dim, n_qubits, n_layers, hidden, backend, shots).to(self.device)
        self.critic_target = QCritic(self.state_dim, self.action_dim, n_qubits, n_layers, hidden, backend, shots).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optims
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # Buffer
        self.replay_buffer = HERReplayBuffer(capacity=buffer_size)

        # Bookkeeping
        self.total_steps = 0
        self.total_episodes = 0

    # ---------- API (same as original) ----------
    def select_action(self, state_obs, goal=None, action_noise_std: float = 0.1):
        """
        Same signature as original QRL agent.
        goal is unused (kept for compatibility with callers passing it).
        """
        if isinstance(state_obs, torch.Tensor):
            if state_obs.dim() == 1:
                state_tensor = state_obs.to(self.device).unsqueeze(0)
            else:
                state_tensor = state_obs.to(self.device)
        else:
            state_tensor = torch.as_tensor(state_obs, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy()[0]
        if action_noise_std and action_noise_std > 0:
            action = action + action_noise_std * np.random.randn(*action.shape)
        return np.clip(action, -1.0, 1.0)

    def remember(self, transition):
        """
        transition: (state, action, reward, next_state, goal, done, achieved)
        """
        self.replay_buffer.push(transition)
        self.total_steps += 1

    def state_from_errors(self, distance_error: float, heading_error: float, tangent_angle: float):
        """
        Matches original helper to produce a 4D state.
        """
        sin_phase = math.sin(tangent_angle)
        cos_phase = math.cos(tangent_angle)
        state = torch.tensor([distance_error, heading_error, sin_phase, cos_phase],
                             dtype=torch.float32, device=self.device)
        return state

    def _soft_update(self, source_net: nn.Module, target_net: nn.Module):
        for p, tp in zip(source_net.parameters(), target_net.parameters()):
            tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)

    def train(self, her_ratio: float = 0.8):
        """
        Keep the same simple DDPG target as original qrl agent (env reward only).
        HER data is stored but not used to alter reward; you can extend later.
        """
        if len(self.replay_buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, goals, dones, achieved = self.replay_buffer.sample(
            self.batch_size, self.device
        )

        # Critic target (env reward)
        with torch.no_grad():
            next_action = self.actor_target(next_states)
            target_q = rewards + (1.0 - dones) * self.discount * self.critic_target(next_states, next_action)

        # Critic update
        q_pred = self.critic(states, actions)
        critic_loss = nn.MSELoss()(q_pred, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        # Actor update
        actor_out = self.actor(states)
        actor_loss = -self.critic(states, actor_out).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()

        # Soft updates
        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)

    def save_policy(self, actor_path="qrl_dpg_actor.pt", critic_path="qrl_dpg_critic.pt"):
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    def load_policy(self, actor_path="qrl_dpg_actor.pt", critic_path="qrl_dpg_critic.pt"):
        try:
            self.actor.load_state_dict(torch.load(actor_path, map_location=self.device))
            self.critic.load_state_dict(torch.load(critic_path, map_location=self.device))
            self.actor_target.load_state_dict(self.actor.state_dict())
            self.critic_target.load_state_dict(self.critic.state_dict())
            print("Loaded saved Quantum DPG+HER policy.")
        except Exception:
            print("No saved Quantum DPG+HER policy found, starting fresh.")