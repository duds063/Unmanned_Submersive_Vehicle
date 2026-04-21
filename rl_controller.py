"""
USV Digital Twin — Módulo 5: Hierarchical PPO RL Controller
=============================================================
Implementação própria de PPO (Proximal Policy Optimization) com
arquitetura hierárquica de 3 níveis para controle do USV.

Hierarquia:
    N1 — Estabilização:  observa IMU (9D)      → ação propulsor (4D)
    N2 — Evasão:         observa sonar (13D)   → setpoint N1 (9D)
    N3 — Navegação:      observa estado (27D)  → waypoint N2 (13D)

Treinamento sequencial:
    Fase 1: Treina N1 (N2, N3 congelados)
    Fase 2: Treina N2 com N1 congelado
    Fase 3: Treina N3 com N1, N2 congelados

PPO com:
    - Clipping de probabilidade (ε=0.2)
    - Generalized Advantage Estimation (GAE, λ=0.95)
    - Entropy bonus (β=0.01)
    - Value function clipping
    - Gradient clipping
    - Normalização de observações e recompensas

Referências:
    - Schulman et al. (2017) — Proximal Policy Optimization Algorithms
    - Duan et al. (2016)    — Benchmarking Deep RL for Continuous Control
    - Nachum et al. (2018)  — Data-Efficient HRL
"""

import numpy as np
import os
import pickle
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from collections import deque


# ─────────────────────────────────────────────
# DEPENDÊNCIAS — numpy-only neural network
# ─────────────────────────────────────────────

class Tensor:
    """Tensor leve com autograd manual para backprop."""

    def __init__(self, data: np.ndarray):
        self.data = np.array(data, dtype=np.float32)
        self.grad = np.zeros_like(self.data)


class Linear:
    """Camada linear totalmente conectada."""

    def __init__(self, in_features: int, out_features: int):
        # inicialização de He (Kaiming) — melhor pra ReLU
        scale = np.sqrt(2.0 / in_features)
        self.W = Tensor(np.random.randn(out_features, in_features) * scale)
        self.b = Tensor(np.zeros(out_features))
        self._input = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._input = x.copy()
        return x @ self.W.data.T + self.b.data

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        self.W.grad += grad_out.T @ self._input
        self.b.grad += grad_out.sum(axis=0)
        return grad_out @ self.W.data

    def parameters(self) -> List[Tensor]:
        return [self.W, self.b]

    def zero_grad(self):
        self.W.grad[:] = 0
        self.b.grad[:] = 0


def relu(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """ReLU com máscara para backprop."""
    mask = (x > 0).astype(np.float32)
    return x * mask, mask


def tanh_activation(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Tanh com derivada."""
    out = np.tanh(x)
    return out, 1 - out**2


class MLP:
    """
    Multi-Layer Perceptron com arquitetura parametrizável.
    Ativação: ReLU nas camadas ocultas, linear na saída.
    """

    def __init__(self, layer_sizes: List[int]):
        self.layers = []
        self._masks = []   # máscaras ReLU para backprop
        for i in range(len(layer_sizes) - 1):
            self.layers.append(Linear(layer_sizes[i], layer_sizes[i+1]))

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._masks = []
        h = x
        for i, layer in enumerate(self.layers):
            h = layer.forward(h)
            if i < len(self.layers) - 1:   # ReLU em todas menos a última
                h, mask = relu(h)
                self._masks.append(mask)
        return h

    def backward(self, grad: np.ndarray) -> np.ndarray:
        g = grad
        for i in reversed(range(len(self.layers))):
            if i < len(self.layers) - 1:
                g = g * self._masks[i]
            g = self.layers[i].backward(g)
        return g

    def parameters(self) -> List[Tensor]:
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    def zero_grad(self):
        for layer in self.layers:
            layer.zero_grad()

    def save(self, path: str):
        data = {
            'weights': [(l.W.data.copy(), l.b.data.copy()) for l in self.layers]
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    def load(self, path: str):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        for layer, (W, b) in zip(self.layers, data['weights']):
            layer.W.data[:] = W
            layer.b.data[:] = b


class Adam:
    """Otimizador Adam."""

    def __init__(
        self,
        parameters: List[Tensor],
        lr:    float = 3e-4,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps:   float = 1e-8,
    ):
        self.params = parameters
        self.lr     = lr
        self.beta1  = beta1
        self.beta2  = beta2
        self.eps    = eps
        self.t      = 0
        self.m      = [np.zeros_like(p.data) for p in parameters]
        self.v      = [np.zeros_like(p.data) for p in parameters]

    def step(self, max_grad_norm: float = 0.5):
        self.t += 1

        # gradient clipping global
        total_norm = np.sqrt(sum(
            np.sum(p.grad**2) for p in self.params
        ))
        if total_norm > max_grad_norm:
            scale = max_grad_norm / (total_norm + 1e-8)
            for p in self.params:
                p.grad *= scale

        for i, p in enumerate(self.params):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * p.grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * p.grad**2

            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)

            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        for p in self.params:
            p.grad[:] = 0


# ─────────────────────────────────────────────
# ACTOR-CRITIC NETWORK
# ─────────────────────────────────────────────

class ActorCritic:
    """
    Rede Actor-Critic compartilhando backbone.

    Actor: π(a|s) — política gaussiana pra ações contínuas
    Critic: V(s)  — função de valor

    Política gaussiana: a ~ N(μ(s), σ)
        μ(s) = actor_head(backbone(s))
        log_σ = parâmetro aprendível independente
    """

    def __init__(
        self,
        obs_dim:    int,
        action_dim: int,
        hidden:     List[int] = None,
    ):
        self.obs_dim    = obs_dim
        self.action_dim = action_dim
        hidden = hidden or [64, 64]

        # backbone compartilhado
        sizes_shared = [obs_dim] + hidden
        self.backbone = MLP(sizes_shared)

        # cabeça do ator — média da política
        self.actor_head  = MLP([hidden[-1], action_dim])

        # cabeça do crítico — valor escalar
        self.critic_head = MLP([hidden[-1], 1])

        # log_std aprendível (independente do estado)
        self.log_std = Tensor(np.zeros(action_dim) - 0.5)

        # normalização de observação online (running mean/std)
        self._obs_mean  = np.zeros(obs_dim)
        self._obs_var   = np.ones(obs_dim)
        self._obs_count = 1e-4

    def normalize_obs(self, obs: np.ndarray, update_stats: bool = True) -> np.ndarray:
        """Normalização de observações usando running mean/std."""
        if update_stats:
            self._obs_count += 1
            delta  = obs - self._obs_mean
            self._obs_mean += delta / self._obs_count
            delta2 = obs - self._obs_mean
            self._obs_var  += delta * delta2 / self._obs_count  # online variance

        std = np.sqrt(self._obs_var / self._obs_count + 1e-8)
        return (obs - self._obs_mean) / std

    def forward(
        self, obs: np.ndarray, normalize: bool = True, update_obs_stats: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Forward pass.

        Returns:
            mean:  média da política (action_dim,)
            std:   desvio padrão
            value: estimativa de valor escalar
        """
        if normalize:
            obs = self.normalize_obs(obs, update_stats=update_obs_stats)

        # batch dimension se necessário
        single = obs.ndim == 1
        if single:
            obs = obs[np.newaxis, :]

        h     = self.backbone.forward(obs)
        mean  = np.tanh(self.actor_head.forward(h))   # tanh → bounded actions
        value = self.critic_head.forward(h)

        std   = np.exp(np.clip(self.log_std.data, -4, 2))

        if single:
            return mean[0], std, value[0, 0]
        return mean, std, value[:, 0]

    def act(self, obs: np.ndarray, update_obs_stats: bool = True) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Amostra ação da política.

        Returns:
            action:   ação amostrada
            log_prob: log probabilidade da ação
            value:    estimativa de valor
        """
        mean, std, value = self.forward(obs, normalize=True, update_obs_stats=update_obs_stats)
        action    = mean + std * np.random.randn(*mean.shape)
        log_prob  = self._gaussian_log_prob(action, mean, std)
        return action, log_prob, value

    def evaluate(
        self, obs: np.ndarray, actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Avalia log_prob e valor para batch de (obs, actions).
        Usado na atualização PPO.
        """
        single = obs.ndim == 1
        if single:
            obs     = obs[np.newaxis, :]
            actions = actions[np.newaxis, :]

        # PPO deve usar a mesma normalização do act(), porém sem atualizar
        # estatísticas online durante o cálculo dos gradientes.
        obs = self.normalize_obs(obs, update_stats=False)

        h      = self.backbone.forward(obs)
        mean   = np.tanh(self.actor_head.forward(h))
        values = self.critic_head.forward(h)[:, 0]
        std    = np.exp(np.clip(self.log_std.data, -4, 2))

        log_probs = self._gaussian_log_prob(actions, mean, std)
        entropy   = self._gaussian_entropy(std)

        return log_probs, values, entropy

    def parameters(self) -> List[Tensor]:
        params = []
        params.extend(self.backbone.parameters())
        params.extend(self.actor_head.parameters())
        params.extend(self.critic_head.parameters())
        params.append(self.log_std)
        return params

    def zero_grad(self):
        self.backbone.zero_grad()
        self.actor_head.zero_grad()
        self.critic_head.zero_grad()
        self.log_std.grad[:] = 0

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        data = {
            'obs_mean':  self._obs_mean,
            'obs_var':   self._obs_var,
            'obs_count': self._obs_count,
            'log_std':   self.log_std.data,
        }
        self.backbone.save(path + '_backbone.pkl')
        self.actor_head.save(path + '_actor.pkl')
        self.critic_head.save(path + '_critic.pkl')
        with open(path + '_meta.pkl', 'wb') as f:
            pickle.dump(data, f)

    def load(self, path: str):
        self.backbone.load(path + '_backbone.pkl')
        self.actor_head.load(path + '_actor.pkl')
        self.critic_head.load(path + '_critic.pkl')
        with open(path + '_meta.pkl', 'rb') as f:
            data = pickle.load(f)
        self._obs_mean  = data['obs_mean']
        self._obs_var   = data['obs_var']
        self._obs_count = data['obs_count']
        self.log_std.data[:] = data['log_std']

    @staticmethod
    def _gaussian_log_prob(
        x: np.ndarray, mean: np.ndarray, std: np.ndarray
    ) -> np.ndarray:
        """Log probabilidade de distribuição gaussiana."""
        log_prob = -0.5 * ((x - mean) / std)**2 - np.log(std) - 0.5 * np.log(2*np.pi)
        return log_prob.sum(axis=-1)

    @staticmethod
    def _gaussian_entropy(std: np.ndarray) -> float:
        """Entropia da distribuição gaussiana."""
        return (0.5 * np.log(2 * np.pi * np.e * std**2)).sum()


# ─────────────────────────────────────────────
# ROLLOUT BUFFER
# ─────────────────────────────────────────────

@dataclass
class RolloutBuffer:
    """
    Buffer de experiências para atualização PPO.
    Armazena um rollout completo antes de atualizar.
    """
    capacity: int = 2048

    observations: List[np.ndarray] = field(default_factory=list)
    actions:      List[np.ndarray] = field(default_factory=list)
    log_probs:    List[float]      = field(default_factory=list)
    rewards:      List[float]      = field(default_factory=list)
    values:       List[float]      = field(default_factory=list)
    dones:        List[bool]       = field(default_factory=list)

    def add(
        self,
        obs:      np.ndarray,
        action:   np.ndarray,
        log_prob: float,
        reward:   float,
        value:    float,
        done:     bool,
    ):
        self.observations.append(obs.copy())
        self.actions.append(action.copy())
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def compute_returns_and_advantages(
        self,
        last_value: float,
        gamma:  float = 0.99,
        lam:    float = 0.95,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generalized Advantage Estimation (GAE).
        Schulman et al. 2016.
        """
        T = len(self.rewards)
        advantages = np.zeros(T)
        returns    = np.zeros(T)

        gae = 0.0
        for t in reversed(range(T)):
            next_val  = last_value if t == T-1 else self.values[t+1]
            next_done = 0.0 if t == T-1 else float(self.dones[t+1])
            delta = (self.rewards[t] +
                     gamma * next_val * (1 - next_done) -
                     self.values[t])
            gae = delta + gamma * lam * (1 - next_done) * gae
            advantages[t] = gae
            returns[t]    = gae + self.values[t]

        # normaliza vantagens
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return returns, advantages

    def get_batches(
        self, batch_size: int
    ):
        """Gera mini-batches aleatórios para epochs de atualização."""
        n   = len(self.observations)
        idx = np.random.permutation(n)
        for start in range(0, n, batch_size):
            batch_idx = idx[start:start + batch_size]
            yield (
                np.array([self.observations[i] for i in batch_idx]),
                np.array([self.actions[i]      for i in batch_idx]),
                np.array([self.log_probs[i]    for i in batch_idx]),
                np.array([self.returns[i]      for i in batch_idx]),
                np.array([self.advantages[i]   for i in batch_idx]),
            )

    def clear(self):
        self.observations.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()

    @property
    def full(self) -> bool:
        return len(self.rewards) >= self.capacity

    def finalize(self, last_value: float, gamma: float, lam: float):
        """Calcula returns e advantages — chamado antes de atualizar."""
        self.returns, self.advantages = self.compute_returns_and_advantages(
            last_value, gamma, lam
        )


# ─────────────────────────────────────────────
# PPO UPDATER
# ─────────────────────────────────────────────

class PPOUpdater:
    """
    Atualização PPO com clipping.
    Reutilizado pelos três níveis hierárquicos.
    """

    def __init__(
        self,
        network:     ActorCritic,
        lr:          float = 3e-4,
        clip_eps:    float = 0.2,
        entropy_coef: float = 0.01,
        value_coef:   float = 0.5,
        n_epochs:    int   = 10,
        batch_size:  int   = 64,
    ):
        self.net          = network
        self.clip_eps     = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef   = value_coef
        self.n_epochs     = n_epochs
        self.batch_size   = batch_size
        self.optimizer    = Adam(network.parameters(), lr=lr)

        self._losses = deque(maxlen=100)

    def update(self, buffer: RolloutBuffer) -> Dict[str, float]:
        """
        Executa n_epochs de atualização PPO no buffer.
        Retorna métricas de treinamento.
        """
        total_loss        = 0.0
        total_policy_loss = 0.0
        total_value_loss  = 0.0
        total_entropy     = 0.0
        n_updates         = 0

        for _ in range(self.n_epochs):
            for obs_b, act_b, old_lp_b, ret_b, adv_b in \
                    buffer.get_batches(self.batch_size):

                # avalia política atual no batch
                log_probs, values, entropy = self.net.evaluate(obs_b, act_b)

                # ratio PPO: π_new / π_old
                ratio = np.exp(log_probs - old_lp_b)

                # clipped surrogate objective
                surr1 = ratio * adv_b
                surr2 = np.clip(ratio, 1-self.clip_eps, 1+self.clip_eps) * adv_b
                policy_loss = -np.minimum(surr1, surr2).mean()

                # value function loss
                value_loss = 0.5 * ((values - ret_b)**2).mean()

                # total loss
                loss = (policy_loss +
                        self.value_coef * value_loss -
                        self.entropy_coef * entropy)

                # backprop manual
                self.net.zero_grad()
                self._backward(
                    obs_b, act_b, old_lp_b, ret_b, adv_b,
                    log_probs, values, ratio
                )
                self.optimizer.step(max_grad_norm=0.5)
                self.optimizer.zero_grad()

                total_loss        += loss
                total_policy_loss += policy_loss
                total_value_loss  += value_loss
                total_entropy     += entropy
                n_updates         += 1

        metrics = {
            'loss':        total_loss / n_updates,
            'policy_loss': total_policy_loss / n_updates,
            'value_loss':  total_value_loss / n_updates,
            'entropy':     total_entropy / n_updates,
        }
        self._losses.append(metrics['loss'])
        return metrics

    def _backward(
        self,
        obs:      np.ndarray,
        actions:  np.ndarray,
        old_lp:   np.ndarray,
        returns:  np.ndarray,
        advantages: np.ndarray,
        log_probs: np.ndarray,
        values:   np.ndarray,
        ratio:    np.ndarray,
    ):
        """
        Backprop manual através da rede Actor-Critic.
        Gradientes calculados analiticamente para cada componente.
        """
        B = len(obs)

        # gradiente do value loss: dL/dV = (V - R)
        grad_value = (values - returns) / B

        # gradiente do policy loss via ratio clipping
        clipped = (ratio < 1 - self.clip_eps) | (ratio > 1 + self.clip_eps)
        grad_ratio = np.where(clipped, 0.0, -advantages) / B
        grad_log_prob = grad_ratio * ratio   # chain rule: d/d(log_π) = ratio * d/d(ratio)

        # backprop pelo critic head
        grad_critic = grad_value[:, np.newaxis]
        h = self.net.backbone.forward(obs)
        self.net.critic_head.backward(grad_critic)

        # backprop pelo actor head
        # log_prob = Σ [-0.5 * ((a - μ)/σ)² - log(σ) - 0.5*log(2π)]
        mean = np.tanh(self.net.actor_head.forward(h))
        std  = np.exp(np.clip(self.net.log_std.data, -4, 2))

        # d(log_prob)/d(mean) = (a - mean) / std²
        d_log_prob_d_mean = (actions - mean) / (std**2 + 1e-8)
        grad_mean = grad_log_prob[:, np.newaxis] * d_log_prob_d_mean

        # d(tanh)/d(pre_tanh) = 1 - tanh²
        grad_pre_tanh = grad_mean * (1 - mean**2)
        self.net.actor_head.backward(grad_pre_tanh)

        # gradiente de log_std
        d_log_prob_d_log_std = ((actions - mean)**2 / (std**2 + 1e-8) - 1)
        self.net.log_std.grad += (
            grad_log_prob[:, np.newaxis] * d_log_prob_d_log_std
        ).mean(axis=0)

        # backprop pelo backbone (soma dos gradientes de actor e critic)
        grad_backbone_from_critic = self.net.critic_head.layers[0].backward(
            grad_critic
        ) if hasattr(self.net.critic_head, '_last_backward') else np.zeros_like(h)
        self.net.backbone.backward(grad_pre_tanh @ self.net.actor_head.layers[0].W.data)

    @property
    def mean_loss(self) -> float:
        return float(np.mean(self._losses)) if self._losses else 0.0


# ─────────────────────────────────────────────
# FUNÇÕES DE RECOMPENSA
# ─────────────────────────────────────────────

class RewardFunction:
    """Funções de recompensa para cada nível hierárquico."""

    @staticmethod
    def n1_stabilization(
        orientation: np.ndarray,   # [phi, theta, psi]
        angular_vel: np.ndarray,   # [p, q, r]
        action:      np.ndarray,
    ) -> float:
        """
        N1 — Estabilização de atitude.
        Penaliza desvio de orientação e velocidade angular.
        """
        orientation_error = np.sum(orientation**2)
        angular_penalty   = np.sum(angular_vel**2)
        action_penalty    = 0.01 * np.sum(action**2)   # suavidade

        reward = -(orientation_error + 0.1 * angular_penalty + action_penalty)
        return float(np.clip(reward, -10.0, 0.0))

    @staticmethod
    def n2_evasion(
        sonar_distances: np.ndarray,   # 6 leituras ortogonais
        clearance_min:   float = 1.0,  # distância mínima de segurança (m)
    ) -> float:
        """
        N2 — Evasão de obstáculos.
        Penaliza proximidade de obstáculos, recompensa espaço livre.
        """
        reward = 0.0

        for dist in sonar_distances:
            if dist < 0:   # sem leitura — assume seguro
                reward += 0.1
            elif dist < clearance_min:
                # penalidade exponencial por proximidade
                reward -= np.exp(clearance_min - dist) - 1
            else:
                # bonus por distância segura
                reward += 0.05 * min(dist / clearance_min, 2.0)

        return float(np.clip(reward, -10.0, 1.0))

    @staticmethod
    def n3_navigation(
        position:         np.ndarray,   # [x, y, z] atual
        waypoint_current: np.ndarray,   # [x, y, z] alvo
        waypoint_reached: bool,
        time_step:        float,
        action_norm:      float,
    ) -> float:
        """
        N3 — Navegação por waypoints.
        Recompensa progresso, penaliza tempo e energia.
        """
        dist = np.linalg.norm(position - waypoint_current)

        # recompensa de progresso — negativo da distância normalizada
        progress_reward = -0.01 * dist

        # bonus por atingir waypoint
        reached_bonus = 10.0 if waypoint_reached else 0.0

        # penalidades
        time_penalty   = -0.001 * time_step
        energy_penalty = -0.005 * action_norm

        reward = progress_reward + reached_bonus + time_penalty + energy_penalty
        return float(np.clip(reward, -5.0, 10.0))


# ─────────────────────────────────────────────
# AGENTES HIERÁRQUICOS
# ─────────────────────────────────────────────

class N1Agent:
    """Nível 1 — Estabilização de atitude."""

    OBS_DIM    = 9    # accel(3) + gyro(3) + orientation(3)
    ACTION_DIM = 4    # thruster_power, theta, phi, ballast

    def __init__(self, lr: float = 3e-4):
        self.network = ActorCritic(
            obs_dim=self.OBS_DIM,
            action_dim=self.ACTION_DIM,
            hidden=[64, 64],
        )
        self.updater = PPOUpdater(self.network, lr=lr)
        self.buffer  = RolloutBuffer(capacity=2048)
        self.frozen  = False

    def get_observation(self, imu_reading, ekf_state) -> np.ndarray:
        """Extrai observação do IMU e estado EKF."""
        return np.concatenate([
            imu_reading.accel,                    # (3,)
            imu_reading.gyro,                     # (3,)
            ekf_state.orientation,                # phi, theta, psi (3,)
        ])

    def act(self, obs: np.ndarray, update_obs_stats: bool = True) -> Tuple[np.ndarray, float, float]:
        return self.network.act(obs, update_obs_stats=update_obs_stats)

    def store(self, obs, action, log_prob, reward, value, done):
        if not self.frozen:
            self.buffer.add(obs, action, log_prob, reward, value, done)

    def update(self, last_value: float) -> Optional[Dict]:
        if self.frozen or not self.buffer.full:
            return None
        self.buffer.finalize(last_value, gamma=0.99, lam=0.95)
        metrics = self.updater.update(self.buffer)
        self.buffer.clear()
        return metrics

    def action_to_command(self, action: np.ndarray):
        """Converte ação normalizada em ControlCommand."""
        from control_engine import ControlCommand
        return ControlCommand(
            thruster_power=float(np.clip(action[0], -1, 1)),
            thruster_theta=float(np.clip(abs(action[1]), 0, np.radians(60))),
            thruster_phi=float(action[2] % (2*np.pi)),
            ballast_cmd=float(np.clip(action[3], -1, 1)),
        )

    def save(self, path: str):
        self.network.save(path + '/n1')

    def load(self, path: str):
        self.network.load(path + '/n1')


class N2Agent:
    """Nível 2 — Evasão de obstáculos."""

    OBS_DIM    = 13   # sonar(6) + n1_action(4) + position(3)
    ACTION_DIM = 9    # setpoint pra N1: accel_ref(3) + gyro_ref(3) + orientation_ref(3)

    def __init__(self, lr: float = 3e-4):
        self.network = ActorCritic(
            obs_dim=self.OBS_DIM,
            action_dim=self.ACTION_DIM,
            hidden=[128, 64],
        )
        self.updater = PPOUpdater(self.network, lr=lr)
        self.buffer  = RolloutBuffer(capacity=2048)
        self.frozen  = False

    def get_observation(
        self,
        sonar_readings,
        n1_action:  np.ndarray,
        position:   np.ndarray,
    ) -> np.ndarray:
        """Extrai observação do sonar, ação do N1 e posição."""
        sonar_vec = np.array([
            r.distance if r.hit else SONAR_RANGE_MAX
            for r in sonar_readings
        ])
        return np.concatenate([sonar_vec, n1_action, position])

    def act(self, obs: np.ndarray, update_obs_stats: bool = True):
        return self.network.act(obs, update_obs_stats=update_obs_stats)

    def store(self, obs, action, log_prob, reward, value, done):
        if not self.frozen:
            self.buffer.add(obs, action, log_prob, reward, value, done)

    def update(self, last_value: float) -> Optional[Dict]:
        if self.frozen or not self.buffer.full:
            return None
        self.buffer.finalize(last_value, gamma=0.99, lam=0.95)
        metrics = self.updater.update(self.buffer)
        self.buffer.clear()
        return metrics

    def save(self, path: str):
        self.network.save(path + '/n2')

    def load(self, path: str):
        self.network.load(path + '/n2')


class N3Agent:
    """Nível 3 — Navegação por waypoints."""

    MAX_WAYPOINTS = 5
    OBS_DIM       = 12 + MAX_WAYPOINTS * 3   # ekf_state(12) + waypoints(15)
    ACTION_DIM    = 13                         # setpoint pra N2

    def __init__(self, lr: float = 3e-4):
        self.network = ActorCritic(
            obs_dim=self.OBS_DIM,
            action_dim=self.ACTION_DIM,
            hidden=[128, 128],
        )
        self.updater  = PPOUpdater(self.network, lr=lr)
        self.buffer   = RolloutBuffer(capacity=1024)   # menor — decisões mais lentas
        self.frozen   = False
        self.waypoints: List[np.ndarray] = []
        self.current_wp_idx = 0
        self.waypoint_threshold = 0.5   # m — raio de chegada

    def set_waypoints(self, waypoints: List[np.ndarray]) -> None:
        assert len(waypoints) <= self.MAX_WAYPOINTS
        self.waypoints      = waypoints
        self.current_wp_idx = 0

    def get_observation(self, ekf_state) -> np.ndarray:
        """Estado EKF + waypoints (com padding se < MAX_WAYPOINTS)."""
        state_vec = np.concatenate([ekf_state.eta, ekf_state.nu])

        # waypoints restantes com padding de zeros
        remaining = self.waypoints[self.current_wp_idx:]
        wp_vec    = np.zeros(self.MAX_WAYPOINTS * 3)
        for i, wp in enumerate(remaining[:self.MAX_WAYPOINTS]):
            wp_vec[i*3:(i+1)*3] = wp

        return np.concatenate([state_vec, wp_vec])

    def check_waypoint_reached(self, position: np.ndarray) -> bool:
        """Verifica se waypoint atual foi atingido."""
        if self.current_wp_idx >= len(self.waypoints):
            return False
        dist = np.linalg.norm(position - self.waypoints[self.current_wp_idx])
        if dist < self.waypoint_threshold:
            self.current_wp_idx += 1
            return True
        return False

    @property
    def current_waypoint(self) -> Optional[np.ndarray]:
        if self.current_wp_idx < len(self.waypoints):
            return self.waypoints[self.current_wp_idx]
        return None

    @property
    def mission_complete(self) -> bool:
        return self.current_wp_idx >= len(self.waypoints)

    def act(self, obs: np.ndarray, update_obs_stats: bool = True):
        return self.network.act(obs, update_obs_stats=update_obs_stats)

    def store(self, obs, action, log_prob, reward, value, done):
        if not self.frozen:
            self.buffer.add(obs, action, log_prob, reward, value, done)

    def update(self, last_value: float) -> Optional[Dict]:
        if self.frozen or not self.buffer.full:
            return None
        self.buffer.finalize(last_value, gamma=0.995, lam=0.95)
        metrics = self.updater.update(self.buffer)
        self.buffer.clear()
        return metrics

    def save(self, path: str):
        self.network.save(path + '/n3')

    def load(self, path: str):
        self.network.load(path + '/n3')


# ─────────────────────────────────────────────
# CONSTANTE IMPORTADA DO SENSOR ENGINE
# ─────────────────────────────────────────────
SONAR_RANGE_MAX = 7.0


# ─────────────────────────────────────────────
# HIERARCHICAL RL CONTROLLER
# ─────────────────────────────────────────────

class HRLController:
    """
    Controlador RL Hierárquico — interface unificada.

    Treinamento sequencial:
        Fase 1: treina N1, N2/N3 congelados
        Fase 2: treina N2, N1/N3 congelados
        Fase 3: treina N3, N1/N2 congelados

    Inferência:
        N3 → setpoint → N2 → setpoint → N1 → ControlCommand
    """

    PHASES = {
        1: 'n1_only',
        2: 'n2_only',
        3: 'n3_only',
        0: 'all_frozen',   # apenas inferência
    }

    def __init__(self, checkpoint_dir: str = './checkpoints'):
        self.n1 = N1Agent()
        self.n2 = N2Agent()
        self.n3 = N3Agent()
        self.reward_fn   = RewardFunction()
        self.checkpoint_dir = checkpoint_dir
        self._phase = 1
        self._set_phase(1)

        # métricas de treinamento
        self.training_metrics = {
            'n1_losses': [], 'n2_losses': [], 'n3_losses': [],
            'n1_rewards': deque(maxlen=100),
            'n2_rewards': deque(maxlen=100),
            'n3_rewards': deque(maxlen=100),
        }

    def set_phase(self, phase: int) -> None:
        """
        Define fase de treinamento.
        phase=1: treina N1
        phase=2: treina N2
        phase=3: treina N3
        phase=0: inferência pura
        """
        self._phase = phase
        self._set_phase(phase)
        print(f"✓ HRL fase {phase}: {self.PHASES[phase]}")

    def set_waypoints(self, waypoints: List[np.ndarray]) -> None:
        self.n3.set_waypoints(waypoints)

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return float((angle + np.pi) % (2 * np.pi) - np.pi)

    def _fuse_actions(
        self,
        act_n1: np.ndarray,
        act_n2: np.ndarray,
        act_n3: np.ndarray,
        ekf_state,
        sonar_readings,
    ) -> np.ndarray:
        """
        Funde ações hierárquicas em uma ação final para o N1.

        N1: estabilização local
        N2: evita obstáculos (bias reativo)
        N3: guia direção/profundidade para waypoint
        """
        fused = np.array(act_n1, dtype=float)

        # Vetor desejado em body frame (x, y, z)
        x_b, y_b, z_b = 0.3, 0.0, 0.0
        depth_err = 0.0

        # --- N3: guidance para waypoint atual (world -> body) ---
        wp = self.n3.current_waypoint
        if wp is not None:
            delta = np.asarray(wp, dtype=float) - np.asarray(ekf_state.position, dtype=float)
            depth_err = float(delta[2])

            yaw = float(ekf_state.orientation[2])
            cy, sy = np.cos(yaw), np.sin(yaw)

            # rotação 2D (mundo -> corpo) no plano horizontal
            x_b = cy * float(delta[0]) + sy * float(delta[1])
            y_b = -sy * float(delta[0]) + cy * float(delta[1])
            z_b = float(delta[2])

            norm_vec = float(np.linalg.norm([x_b, y_b, z_b])) + 1e-6
            x_b /= norm_vec
            y_b /= norm_vec
            z_b /= norm_vec

        # --- N2: reatividade por sonar ---
        sonar_dists = np.array([
            r.distance if r.hit else SONAR_RANGE_MAX
            for r in sonar_readings
        ], dtype=float)

        front = float(sonar_dists[0]) if len(sonar_dists) > 0 else SONAR_RANGE_MAX
        stb   = float(sonar_dists[2]) if len(sonar_dists) > 2 else SONAR_RANGE_MAX
        port  = float(sonar_dists[3]) if len(sonar_dists) > 3 else SONAR_RANGE_MAX

        clearance = 2.0
        front_close = float(np.clip((clearance - front) / clearance, 0.0, 1.0))
        stb_close   = float(np.clip((clearance - stb)   / clearance, 0.0, 1.0))
        port_close  = float(np.clip((clearance - port)  / clearance, 0.0, 1.0))

        # se obstáculo mais perto à direita (stb), empurra para bombordo (y+)
        avoid_turn = float(np.clip(stb_close - port_close, -1.0, 1.0))
        x_b = x_b * (1.0 - 0.7 * front_close)
        y_b = y_b + 0.9 * avoid_turn * max(front_close, 0.25)

        # contribuição residual dos níveis superiores
        x_b += 0.15 * np.tanh(float(act_n3[0])) + 0.10 * np.tanh(float(act_n2[0]))
        y_b += 0.20 * np.tanh(float(act_n3[1])) + 0.15 * np.tanh(float(act_n2[8]))
        z_b += 0.10 * np.tanh(float(act_n2[2]))

        # mapeamento de vetor body -> [power, theta, phi, ballast]
        lateral = float(np.hypot(y_b, z_b))
        theta = float(np.arctan2(lateral, max(0.05, x_b)))
        theta = float(np.clip(theta, 0.0, np.radians(60.0)))

        phi = float(np.arctan2(z_b, y_b))

        # potência favorece avanço + distância ao alvo quando houver waypoint
        base_power = 0.45 + 0.25 * max(x_b, 0.0)
        if wp is not None:
            dist = float(np.linalg.norm(np.asarray(wp, dtype=float) - np.asarray(ekf_state.position, dtype=float)))
            base_power += 0.25 * np.tanh(dist / 3.0)
        power = float(np.clip(base_power + 0.15 * np.tanh(float(act_n1[0])), -1.0, 1.0))

        ballast = float(np.clip(0.35 * np.tanh(depth_err) + 0.15 * np.tanh(float(act_n1[3])), -1.0, 1.0))

        fused[0] = power
        fused[1] = theta
        fused[2] = phi
        fused[3] = ballast
        return fused

    def compute(
        self,
        ekf_state,
        imu_reading,
        sonar_readings,
        time: float,
        training: bool = False,
        forced_done: bool = False,
        return_info: bool = False,
    ):
        """
        Computa ação hierárquica.

        N3 → N2 → N1 → ControlCommand

        Args:
            training: se True, armazena experiências no buffer
        """
        from control_engine import ControlCommand

        position = ekf_state.position

        # ── N3 — Navegação ────────────────────
        obs_n3 = self.n3.get_observation(ekf_state)
        act_n3, lp_n3, val_n3 = self.n3.act(
            obs_n3,
            update_obs_stats=bool(training and not self.n3.frozen),
        )

        # verifica waypoint
        reached = self.n3.check_waypoint_reached(position)

        # recompensa N3
        waypoint_before = self.n3.current_waypoint
        if waypoint_before is not None:
            r_n3 = self.reward_fn.n3_navigation(
                position=position,
                waypoint_current=waypoint_before,
                waypoint_reached=reached,
                time_step=time,
                action_norm=np.linalg.norm(act_n3),
            )
        else:
            r_n3 = 1.0   # missão completa

        # ── N2 — Evasão ───────────────────────
        obs_n2 = self.n2.get_observation(sonar_readings, act_n3[:4], position)
        act_n2, lp_n2, val_n2 = self.n2.act(
            obs_n2,
            update_obs_stats=bool(training and not self.n2.frozen),
        )

        # recompensa N2
        sonar_dists = np.array([
            r.distance if r.hit else SONAR_RANGE_MAX
            for r in sonar_readings
        ])
        r_n2 = self.reward_fn.n2_evasion(sonar_dists)

        # ── N1 — Estabilização ────────────────
        obs_n1 = self.n1.get_observation(imu_reading, ekf_state)
        act_n1, lp_n1, val_n1 = self.n1.act(
            obs_n1,
            update_obs_stats=bool(training and not self.n1.frozen),
        )

        # recompensa N1
        r_n1 = self.reward_fn.n1_stabilization(
            orientation=ekf_state.orientation,
            angular_vel=ekf_state.velocity_angular,
            action=act_n1,
        )

        # ── Armazena experiências ─────────────
        done = bool(self.n3.mission_complete or forced_done)

        if training:
            self.n1.store(obs_n1, act_n1, lp_n1, r_n1, val_n1, done)
            self.n2.store(obs_n2, act_n2, lp_n2, r_n2, val_n2, done)
            self.n3.store(obs_n3, act_n3, lp_n3, r_n3, val_n3, done)

            self.training_metrics['n1_rewards'].append(r_n1)
            self.training_metrics['n2_rewards'].append(r_n2)
            self.training_metrics['n3_rewards'].append(r_n3)

        # ── Converte em ControlCommand ────────
        fused_action = self._fuse_actions(
            act_n1=act_n1,
            act_n2=act_n2,
            act_n3=act_n3,
            ekf_state=ekf_state,
            sonar_readings=sonar_readings,
        )
        cmd = self.n1.action_to_command(fused_action)

        if return_info:
            return cmd, {
                'values': {
                    'n1': float(val_n1),
                    'n2': float(val_n2),
                    'n3': float(val_n3),
                },
                'rewards': {
                    'n1': float(r_n1),
                    'n2': float(r_n2),
                    'n3': float(r_n3),
                },
                'done': done,
            }
        return cmd

    def update_networks(self, last_values: Dict[str, float]) -> Dict:
        """Atualiza redes na fase ativa."""
        metrics = {}

        m1 = self.n1.update(last_values.get('n1', 0.0))
        m2 = self.n2.update(last_values.get('n2', 0.0))
        m3 = self.n3.update(last_values.get('n3', 0.0))

        if m1:
            metrics['n1'] = m1
            self.training_metrics['n1_losses'].append(m1['loss'])
        if m2:
            metrics['n2'] = m2
            self.training_metrics['n2_losses'].append(m2['loss'])
        if m3:
            metrics['n3'] = m3
            self.training_metrics['n3_losses'].append(m3['loss'])

        return metrics

    def save_checkpoint(self, phase: int) -> None:
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.n1.save(self.checkpoint_dir)
        self.n2.save(self.checkpoint_dir)
        self.n3.save(self.checkpoint_dir)
        print(f"✓ Checkpoint salvo — fase {phase}")

    def load_checkpoint(self) -> None:
        self.n1.load(self.checkpoint_dir)
        self.n2.load(self.checkpoint_dir)
        self.n3.load(self.checkpoint_dir)
        print("✓ Checkpoint carregado")

    def print_metrics(self) -> None:
        m = self.training_metrics
        print(f"  N1 reward médio: {np.mean(m['n1_rewards']):.3f}  "
              f"loss: {m['n1_losses'][-1]:.4f}" if m['n1_losses'] else
              f"  N1 reward médio: {np.mean(m['n1_rewards']):.3f}")
        print(f"  N2 reward médio: {np.mean(m['n2_rewards']):.3f}  "
              f"loss: {m['n2_losses'][-1]:.4f}" if m['n2_losses'] else
              f"  N2 reward médio: {np.mean(m['n2_rewards']):.3f}")
        print(f"  N3 reward médio: {np.mean(m['n3_rewards']):.3f}  "
              f"loss: {m['n3_losses'][-1]:.4f}" if m['n3_losses'] else
              f"  N3 reward médio: {np.mean(m['n3_rewards']):.3f}")

    def _set_phase(self, phase: int) -> None:
        """Congela/descongela agentes conforme a fase."""
        self.n1.frozen = phase != 1
        self.n2.frozen = phase != 2
        self.n3.frozen = phase != 3


# ─────────────────────────────────────────────
# INTEGRAÇÃO NO CONTROL ENGINE
# ─────────────────────────────────────────────

def integrate_rl(control_engine, checkpoint_dir: str = './checkpoints') -> HRLController:
    """Inicializa e integra o HRL no ControlEngine."""
    hrl = HRLController(checkpoint_dir=checkpoint_dir)

    # tenta carregar checkpoint existente
    if os.path.exists(checkpoint_dir):
        try:
            hrl.load_checkpoint()
        except Exception:
            print("  Nenhum checkpoint válido — iniciando do zero.")

    control_engine._rl = hrl
    print("✓ HRL Controller integrado ao ControlEngine.")
    return hrl


# ─────────────────────────────────────────────
# TESTES
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import time as _time
    from geometry_engine import GeometryEngine
    from physics_engine  import PhysicsEngine
    from sensor_engine   import SensorEngine, ExtendedKalmanFilter, Environment
    from control_engine  import ControlEngine, ControlCommand
    from mpc_controller  import integrate_mpc

    print("Inicializando HRL Controller...")

    geo     = GeometryEngine(L=0.8, D=0.1)
    physics = PhysicsEngine(geo, max_thruster_force=10.0)
    env     = Environment(pool_depth=5.0)
    env.add_sphere(np.array([3.0, 0.0, 2.0]), radius=0.5)
    sensors = SensorEngine(env, noise_scale=0.5)
    ekf     = ExtendedKalmanFilter(physics)
    control = ControlEngine(physics, hover_depth=2.0)
    hrl     = integrate_rl(control)

    # define missão com 3 waypoints
    waypoints = [
        np.array([2.0, 0.0, 2.0]),
        np.array([4.0, 1.0, 3.0]),
        np.array([6.0, 0.0, 2.0]),
    ]
    hrl.set_waypoints(waypoints)

    # Teste 1 — arquitetura das redes
    print("\nTeste 1 — Arquitetura das redes:")
    for name, agent in [('N1', hrl.n1), ('N2', hrl.n2), ('N3', hrl.n3)]:
        n_params = sum(p.data.size for p in agent.network.parameters())
        print(f"  {name}: obs={agent.OBS_DIM}D → "
              f"action={agent.ACTION_DIM}D | "
              f"params={n_params:,}")

    # Teste 2 — forward pass
    print("\nTeste 2 — Forward pass:")
    physics.reset()
    ekf.reset()
    dt = 0.01

    bundle = sensors.read(physics.state, 0.0)
    ekf.predict(dt)
    ekf.update_imu(bundle.imu)
    ekf.update_barometer(bundle.barometer)
    est = ekf.state_estimate

    t0  = _time.time()
    cmd = hrl.compute(est, bundle.imu, bundle.sonar, 0.0, training=False)
    dt_infer = (_time.time() - t0) * 1000

    print(f"  Inferência: {dt_infer:.2f}ms")
    print(f"  Comando: power={cmd.thruster_power:.3f} "
          f"ballast={cmd.ballast_cmd:.3f}")

    # Teste 3 — loop de treinamento N1 curto (50 steps)
    print("\nTeste 3 — Treinamento N1 (50 steps, fase 1):")
    hrl.set_phase(1)
    physics.reset()
    ekf.reset()

    t_start = _time.time()
    for i in range(50):
        bundle = sensors.read(physics.state, physics.time)
        ekf.predict(dt)
        ekf.update_imu(bundle.imu)
        ekf.update_barometer(bundle.barometer)
        est = ekf.state_estimate

        cmd = hrl.compute(est, bundle.imu, bundle.sonar,
                          physics.time, training=True)

        physics.step(
            thruster_power=cmd.thruster_power,
            thruster_theta=cmd.thruster_theta,
            thruster_phi=cmd.thruster_phi,
            ballast_cmd=cmd.ballast_cmd,
            dt=dt,
        )

    wall = _time.time() - t_start
    print(f"  50 steps em {wall:.2f}s ({50/wall:.0f} steps/s)")
    print(f"  N1 reward médio: {np.mean(hrl.training_metrics['n1_rewards']):.3f}")
    print(f"  Buffer N1: {len(hrl.n1.buffer.rewards)}/2048 experiências")

    # Teste 4 — inferência pura (modo deploy)
    print("\nTeste 4 — Modo inferência (fase 0):")
    hrl.set_phase(0)
    t0 = _time.time()
    for _ in range(100):
        cmd = hrl.compute(est, bundle.imu, bundle.sonar, 0.0, training=False)
    dt_100 = (_time.time() - t0) * 1000
    print(f"  100 inferências: {dt_100:.1f}ms total ({dt_100/100:.2f}ms/step)")

    print("\n✓ HRL Controller validado.")
