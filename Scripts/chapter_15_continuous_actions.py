# Chapter 15: Continuous Action Space
# DDPG and TD3 for continuous control tasks

# Install required packages:
# !pip install torch gymnasium[classic_control] gymnasium[mujoco] matplotlib numpy -q

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Optional
from collections import deque
import random
import copy
from tqdm import tqdm

# Configure matplotlib for proper font rendering
plt.switch_backend("Agg")
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "PingFang SC", "Arial Unicode MS", "Hiragino Sans GB"]
plt.rcParams["axes.unicode_minus"] = False

print("Chapter 15: Continuous Action Space")
print("=" * 50)

# 1. INTRODUCTION TO CONTINUOUS CONTROL
print("\n1. Continuous Action Spaces")
print("-" * 30)

print("""
CONTINUOUS ACTION SPACES:

1. MOTIVATION:
   - Many real-world tasks require continuous control
   - Examples: robot arm control, autonomous driving, game playing
   - Discrete actions are often insufficient or inefficient

2. CHALLENGES:
   - Cannot enumerate all possible actions
   - Q-learning requires max_a Q(s,a) operation
   - Policy gradients work but have high variance

3. SOLUTIONS:
   - Actor-Critic with continuous policy
   - Deterministic Policy Gradients (DPG)
   - Deep Deterministic Policy Gradients (DDPG)
   - Twin Delayed DDPG (TD3)

4. KEY INNOVATIONS:
   - Deterministic policies: μ(s) → a
   - Critic learns Q(s,a) for continuous actions
   - Actor learns deterministic policy
   - Exploration through noise injection

5. DDPG COMPONENTS:
   - Actor: μ_θ(s) outputs deterministic actions
   - Critic: Q_φ(s,a) estimates action values
   - Target networks for stability
   - Experience replay for sample efficiency
   - Noise for exploration

6. TD3 IMPROVEMENTS:
   - Twin critics to reduce overestimation
   - Delayed policy updates
   - Target policy smoothing
""")

# 2. ORNSTEIN-UHLENBECK NOISE
print("\n2. Exploration Noise for Continuous Actions")
print("-" * 30)

class OrnsteinUhlenbeckNoise:
    """Ornstein-Uhlenbeck process for temporally correlated exploration noise.
    
    Generates temporally correlated noise that's more suitable for
    continuous control than independent Gaussian noise.
    """
    
    def __init__(self, action_dim: int, mu: float = 0.0, theta: float = 0.15,
                 sigma: float = 0.2, dt: float = 1e-2):
        """Initialize OU noise process.
        
        Args:
            action_dim: Dimension of action space
            mu: Long-term mean (equilibrium value)
            theta: Mean reversion rate
            sigma: Noise scale
            dt: Time step
        """
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        
        self.state = np.ones(action_dim) * mu
        
    def reset(self) -> None:
        """Reset noise to initial state."""
        self.state = np.ones(self.action_dim) * self.mu
    
    def sample(self) -> np.ndarray:
        """Sample next noise value.
        
        Returns:
            Noise sample
        """
        dx = self.theta * (self.mu - self.state) * self.dt + \
             self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.action_dim)
        
        self.state += dx
        return self.state

class GaussianNoise:
    """Simple Gaussian noise for exploration."""
    
    def __init__(self, action_dim: int, sigma: float = 0.1):
        """Initialize Gaussian noise.
        
        Args:
            action_dim: Dimension of action space
            sigma: Standard deviation of noise
        """
        self.action_dim = action_dim
        self.sigma = sigma
    
    def reset(self) -> None:
        """Reset noise (no-op for Gaussian noise)."""
        pass
    
    def sample(self) -> np.ndarray:
        """Sample noise.
        
        Returns:
            Gaussian noise sample
        """
        return np.random.normal(0, self.sigma, self.action_dim)

# Test noise processes
print("Testing noise processes...")
ou_noise = OrnsteinUhlenbeckNoise(action_dim=2)
gaussian_noise = GaussianNoise(action_dim=2, sigma=0.1)

print(f"OU noise sample: {ou_noise.sample()}")
print(f"Gaussian noise sample: {gaussian_noise.sample()}")

# 3. DDPG IMPLEMENTATION
print("\n3. DDPG Implementation")
print("-" * 30)

class Actor(nn.Module):
    """Actor network for DDPG.
    
    Outputs deterministic actions given states.
    """
    
    def __init__(self, state_dim: int, action_dim: int, max_action: float,
                 hidden_dim: int = 256):
        """Initialize actor network.
        
        Args:
            state_dim: State space dimension
            action_dim: Action space dimension
            max_action: Maximum action value for scaling
            hidden_dim: Hidden layer dimension
        """
        super(Actor, self).__init__()
        
        self.max_action = max_action
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # Output in [-1, 1]
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass through actor.
        
        Args:
            state: Input state tensor
            
        Returns:
            Deterministic action scaled to action space
        """
        return self.max_action * self.network(state)

class Critic(nn.Module):
    """Critic network for DDPG.
    
    Estimates Q-values for state-action pairs.
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        """Initialize critic network.
        
        Args:
            state_dim: State space dimension
            action_dim: Action space dimension
            hidden_dim: Hidden layer dimension
        """
        super(Critic, self).__init__()
        
        # State processing layers
        self.state_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Combined state-action processing
        self.combined_net = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward pass through critic.
        
        Args:
            state: Input state tensor
            action: Input action tensor
            
        Returns:
            Q-value estimate
        """
        state_features = self.state_net(state)
        combined = torch.cat([state_features, action], dim=1)
        return self.combined_net(combined)

class ReplayBuffer:
    """Experience replay buffer for DDPG."""
    
    def __init__(self, capacity: int = 100000):
        """Initialize replay buffer.
        
        Args:
            capacity: Maximum buffer capacity
        """
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state: np.ndarray, action: np.ndarray, reward: float,
             next_state: np.ndarray, done: bool) -> None:
        """Add experience to buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """Sample batch from buffer.
        
        Args:
            batch_size: Size of batch to sample
            
        Returns:
            Tuple of batched tensors
        """
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        
        return (
            torch.FloatTensor(state),
            torch.FloatTensor(action),
            torch.FloatTensor(reward),
            torch.FloatTensor(next_state),
            torch.BoolTensor(done)
        )
    
    def __len__(self) -> int:
        """Return current buffer size."""
        return len(self.buffer)

class DDPGAgent:
    """Deep Deterministic Policy Gradient agent."""
    
    def __init__(self, state_dim: int, action_dim: int, max_action: float,
                 lr_actor: float = 0.001, lr_critic: float = 0.001,
                 gamma: float = 0.99, tau: float = 0.005,
                 noise_type: str = 'ou', noise_scale: float = 0.1,
                 device: str = None):
        """Initialize DDPG agent.
        
        Args:
            state_dim: State space dimension
            action_dim: Action space dimension
            max_action: Maximum action value
            lr_actor: Actor learning rate
            lr_critic: Critic learning rate
            gamma: Discount factor
            tau: Soft update coefficient
            noise_type: Type of noise ('ou' or 'gaussian')
            noise_scale: Scale of exploration noise
            device: Device to run on
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        
        self.device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Networks
        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer()
        
        # Noise
        if noise_type == 'ou':
            self.noise = OrnsteinUhlenbeckNoise(action_dim, sigma=noise_scale)
        else:
            self.noise = GaussianNoise(action_dim, sigma=noise_scale)
        
        # Training statistics
        self.actor_losses = []
        self.critic_losses = []
        self.episode_rewards = []
    
    def select_action(self, state: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """Select action using current policy.
        
        Args:
            state: Current state
            add_noise: Whether to add exploration noise
            
        Returns:
            Selected action
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy().flatten()
        
        if add_noise:
            noise = self.noise.sample()
            action = np.clip(action + noise, -self.max_action, self.max_action)
        
        return action
    
    def store_transition(self, state: np.ndarray, action: np.ndarray, reward: float,
                        next_state: np.ndarray, done: bool) -> None:
        """Store transition in replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def soft_update(self, target: nn.Module, source: nn.Module) -> None:
        """Soft update target network parameters.
        
        Args:
            target: Target network
            source: Source network
        """
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def train(self, batch_size: int = 64) -> Optional[Tuple[float, float]]:
        """Train the agent.
        
        Args:
            batch_size: Batch size for training
            
        Returns:
            Tuple of (actor_loss, critic_loss) if training occurred
        """
        if len(self.replay_buffer) < batch_size:
            return None
        
        # Sample batch
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)
        
        # Move to device
        state = state.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        next_state = next_state.to(self.device)
        done = done.to(self.device)
        
        # Critic update
        with torch.no_grad():
            next_action = self.actor_target(next_state)
            target_q = self.critic_target(next_state, next_action)
            target_q = reward.unsqueeze(1) + (1 - done.unsqueeze(1).float()) * self.gamma * target_q
        
        current_q = self.critic(state, action)
        critic_loss = F.mse_loss(current_q, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Actor update
        predicted_action = self.actor(state)
        actor_loss = -self.critic(state, predicted_action).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Soft update target networks
        self.soft_update(self.actor_target, self.actor)
        self.soft_update(self.critic_target, self.critic)
        
        # Store losses
        self.actor_losses.append(actor_loss.item())
        self.critic_losses.append(critic_loss.item())
        
        return actor_loss.item(), critic_loss.item()

print("DDPG implementation complete.")

# 4. TD3 IMPLEMENTATION
print("\n4. TD3 Implementation")
print("-" * 30)

class TD3Agent:
    """Twin Delayed Deep Deterministic Policy Gradient agent.
    
    Improves upon DDPG with:
    1. Twin critics to reduce overestimation
    2. Delayed policy updates
    3. Target policy smoothing
    """
    
    def __init__(self, state_dim: int, action_dim: int, max_action: float,
                 lr_actor: float = 0.001, lr_critic: float = 0.001,
                 gamma: float = 0.99, tau: float = 0.005,
                 policy_delay: int = 2, policy_noise: float = 0.2,
                 noise_clip: float = 0.5, exploration_noise: float = 0.1,
                 device: str = None):
        """Initialize TD3 agent.
        
        Args:
            policy_delay: Frequency of delayed policy updates
            policy_noise: Target policy smoothing noise
            noise_clip: Range to clip target policy noise
            exploration_noise: Exploration noise scale
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.policy_delay = policy_delay
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.exploration_noise = exploration_noise
        
        self.device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Networks
        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        
        # Twin critics
        self.critic1 = Critic(state_dim, action_dim).to(self.device)
        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic2 = Critic(state_dim, action_dim).to(self.device)
        self.critic2_target = copy.deepcopy(self.critic2)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr_critic)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr_critic)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer()
        
        # Training step counter
        self.total_it = 0
        
        # Training statistics
        self.actor_losses = []
        self.critic_losses = []
        self.episode_rewards = []
    
    def select_action(self, state: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """Select action using current policy."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy().flatten()
        
        if add_noise:
            noise = np.random.normal(0, self.exploration_noise, size=self.action_dim)
            action = np.clip(action + noise, -self.max_action, self.max_action)
        
        return action
    
    def store_transition(self, state: np.ndarray, action: np.ndarray, reward: float,
                        next_state: np.ndarray, done: bool) -> None:
        """Store transition in replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def soft_update(self, target: nn.Module, source: nn.Module) -> None:
        """Soft update target network parameters."""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def train(self, batch_size: int = 64) -> Optional[Tuple[float, float]]:
        """Train the agent with TD3 improvements."""
        if len(self.replay_buffer) < batch_size:
            return None
        
        self.total_it += 1
        
        # Sample batch
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)
        
        # Move to device
        state = state.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        next_state = next_state.to(self.device)
        done = done.to(self.device)
        
        # Target policy smoothing: add clipped noise to target actions
        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            next_action = (self.actor_target(next_state) + noise).clamp(
                -self.max_action, self.max_action
            )
            
            # Twin critics: take minimum Q-value
            target_q1 = self.critic1_target(next_state, next_action)
            target_q2 = self.critic2_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward.unsqueeze(1) + (1 - done.unsqueeze(1).float()) * self.gamma * target_q
        
        # Update critics
        current_q1 = self.critic1(state, action)
        current_q2 = self.critic2(state, action)
        
        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)
        
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        
        # Delayed policy update
        actor_loss = 0
        if self.total_it % self.policy_delay == 0:
            # Actor update
            predicted_action = self.actor(state)
            actor_loss = -self.critic1(state, predicted_action).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # Soft update target networks
            self.soft_update(self.actor_target, self.actor)
            self.soft_update(self.critic1_target, self.critic1)
            self.soft_update(self.critic2_target, self.critic2)
            
            actor_loss = actor_loss.item()
        
        # Store losses
        critic_loss = (critic1_loss.item() + critic2_loss.item()) / 2
        self.critic_losses.append(critic_loss)
        if actor_loss != 0:
            self.actor_losses.append(actor_loss)
        
        return actor_loss, critic_loss

print("TD3 implementation complete.")

# 5. TRAINING ON CONTINUOUS CONTROL TASK
print("\n5. Training on Continuous Control")
print("-" * 30)

def train_continuous_agent(agent, env: gym.Env, n_episodes: int = 500,
                         max_steps: int = 1000, start_training: int = 1000) -> List[Dict]:
    """Train continuous control agent.
    
    Args:
        agent: DDPG or TD3 agent
        env: Continuous control environment
        n_episodes: Number of training episodes
        max_steps: Maximum steps per episode
        start_training: Episode to start training
        
    Returns:
        List of episode statistics
    """
    episode_stats = []
    
    print(f"Training agent for {n_episodes} episodes...")
    
    for episode in range(n_episodes):
        state, _ = env.reset()
        episode_reward = 0
        
        # Reset noise
        if hasattr(agent, 'noise'):
            agent.noise.reset()
        
        for step in range(max_steps):
            # Select action
            action = agent.select_action(state, add_noise=True)
            
            # Take step
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store transition
            agent.store_transition(state, action, reward, next_state, done)
            
            # Train agent
            if episode >= start_training:
                losses = agent.train()
            else:
                losses = None
            
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        # Store episode statistics
        episode_stat = {
            'episode': episode,
            'reward': episode_reward,
            'steps': step + 1
        }
        
        if losses:
            episode_stat['actor_loss'] = losses[0]
            episode_stat['critic_loss'] = losses[1]
        
        episode_stats.append(episode_stat)
        agent.episode_rewards.append(episode_reward)
        
        # Logging
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean([s['reward'] for s in episode_stats[-50:]])
            print(f"Episode {episode+1}: Avg Reward = {avg_reward:.2f}")
    
    return episode_stats

# Create environment
env_name = 'Pendulum-v1'
env = gym.make(env_name)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

print(f"Environment: {env_name}")
print(f"State dimension: {state_dim}")
print(f"Action dimension: {action_dim}")
print(f"Max action: {max_action}")

# Train DDPG
print("\nTraining DDPG agent...")
ddpg_agent = DDPGAgent(state_dim, action_dim, max_action, 
                       noise_type='gaussian', noise_scale=0.1)
ddpg_stats = train_continuous_agent(ddpg_agent, env, n_episodes=300, start_training=50)

print(f"\nDDPG training completed!")
final_ddpg_rewards = [s['reward'] for s in ddpg_stats[-50:]]
print(f"DDPG final average reward: {np.mean(final_ddpg_rewards):.2f}")

# Train TD3
print("\nTraining TD3 agent...")
td3_agent = TD3Agent(state_dim, action_dim, max_action, exploration_noise=0.1)
td3_stats = train_continuous_agent(td3_agent, env, n_episodes=300, start_training=50)

print(f"\nTD3 training completed!")
final_td3_rewards = [s['reward'] for s in td3_stats[-50:]]
print(f"TD3 final average reward: {np.mean(final_td3_rewards):.2f}")

# 6. EVALUATION AND COMPARISON
print("\n6. Evaluation and Analysis")
print("-" * 30)

def evaluate_continuous_agent(agent, env: gym.Env, n_episodes: int = 10) -> float:
    """Evaluate trained agent."""
    total_reward = 0
    
    for _ in range(n_episodes):
        state, _ = env.reset()
        episode_reward = 0
        
        for _ in range(1000):
            action = agent.select_action(state, add_noise=False)  # No exploration
            state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            
            if terminated or truncated:
                break
        
        total_reward += episode_reward
    
    return total_reward / n_episodes

# Evaluate both agents
ddpg_performance = evaluate_continuous_agent(ddpg_agent, env)
td3_performance = evaluate_continuous_agent(td3_agent, env)

print(f"\nEvaluation Results:")
print(f"DDPG: {ddpg_performance:.2f} average reward")
print(f"TD3:  {td3_performance:.2f} average reward")

# 7. VISUALIZATION
print("\n7. Creating Visualizations")
print("-" * 30)

# Create comprehensive plots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Learning curves comparison
ddpg_rewards = [s['reward'] for s in ddpg_stats]
td3_rewards = [s['reward'] for s in td3_stats]

window_size = 25
if len(ddpg_rewards) >= window_size:
    ddpg_smooth = np.convolve(ddpg_rewards, np.ones(window_size)/window_size, mode='valid')
    td3_smooth = np.convolve(td3_rewards, np.ones(window_size)/window_size, mode='valid')
    
    ax1.plot(ddpg_smooth, label='DDPG', linewidth=2, alpha=0.8)
    ax1.plot(td3_smooth, label='TD3', linewidth=2, alpha=0.8)
else:
    ax1.plot(ddpg_rewards, label='DDPG', linewidth=2, alpha=0.8)
    ax1.plot(td3_rewards, label='TD3', linewidth=2, alpha=0.8)

ax1.set_xlabel('Episode')
ax1.set_ylabel('Episode Reward')
ax1.set_title('Learning Curves: DDPG vs TD3')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Loss comparison
if ddpg_agent.critic_losses and td3_agent.critic_losses:
    # Smooth losses
    ddpg_critic_smooth = np.convolve(ddpg_agent.critic_losses, np.ones(50)/50, mode='valid')
    td3_critic_smooth = np.convolve(td3_agent.critic_losses, np.ones(50)/50, mode='valid')
    
    ax2.plot(ddpg_critic_smooth, label='DDPG Critic', alpha=0.8)
    ax2.plot(td3_critic_smooth, label='TD3 Critic', alpha=0.8)
    ax2.set_xlabel('Training Step')
    ax2.set_ylabel('Critic Loss')
    ax2.set_title('Critic Loss Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

# Plot 3: Action distribution analysis
# Sample actions from trained policies
test_states = []
ddpg_actions = []
td3_actions = []

for _ in range(100):
    state, _ = env.reset()
    test_states.append(state)
    
    ddpg_action = ddpg_agent.select_action(state, add_noise=False)
    td3_action = td3_agent.select_action(state, add_noise=False)
    
    ddpg_actions.append(ddpg_action[0])  # First action dimension
    td3_actions.append(td3_action[0])

ax3.hist(ddpg_actions, bins=20, alpha=0.7, label='DDPG', density=True)
ax3.hist(td3_actions, bins=20, alpha=0.7, label='TD3', density=True)
ax3.set_xlabel('Action Value')
ax3.set_ylabel('Density')
ax3.set_title('Action Distribution Comparison')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Performance comparison
methods = ['DDPG', 'TD3']
performances = [ddpg_performance, td3_performance]
final_rewards = [np.mean(final_ddpg_rewards), np.mean(final_td3_rewards)]

x = np.arange(len(methods))
width = 0.35

ax4.bar(x - width/2, performances, width, label='Evaluation', alpha=0.8)
ax4.bar(x + width/2, final_rewards, width, label='Final Training', alpha=0.8)
ax4.set_xlabel('Method')
ax4.set_ylabel('Average Reward')
ax4.set_title('Performance Comparison')
ax4.set_xticks(x)
ax4.set_xticklabels(methods)
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('pytorch_rl_tutorial/chapter_15_continuous_control_results.png', dpi=150, bbox_inches='tight')
print("Results saved to chapter_15_continuous_control_results.png")

# 8. CONTINUOUS CONTROL ANALYSIS
print("\n8. Continuous Control Analysis")
print("-" * 30)

print("""
CONTINUOUS CONTROL INSIGHTS:

1. DDPG CHARACTERISTICS:
   ✓ Pioneered deterministic policy gradients for continuous control
   ✓ Uses experience replay and target networks from DQN
   ✓ Requires exploration noise (OU or Gaussian)
   ✗ Can suffer from overestimation bias
   ✗ Sensitive to hyperparameters

2. TD3 IMPROVEMENTS:
   ✓ Twin critics reduce overestimation bias
   ✓ Delayed policy updates improve stability
   ✓ Target policy smoothing reduces variance
   ✓ More robust to hyperparameters
   ✓ Better sample efficiency

3. KEY DESIGN DECISIONS:
   - Deterministic vs stochastic policies
   - Exploration noise type and scale
   - Network architecture choices
   - Target network update frequency

4. COMMON CHALLENGES:
   - Exploration in continuous spaces
   - Hyperparameter sensitivity
   - Sample efficiency
   - Stability across different environments

5. APPLICATIONS:
   - Robotics and robot control
   - Autonomous vehicles
   - Game playing (e.g., racing games)
   - Industrial process control
   - Financial trading strategies

6. FUTURE DIRECTIONS:
   - Soft Actor-Critic (SAC) for maximum entropy
   - Distributional methods for better estimates
   - Model-based approaches for sample efficiency
   - Meta-learning for faster adaptation
""")

print(f"\nPerformance Summary:")
print(f"DDPG: {ddpg_performance:.2f} evaluation reward, {np.mean(final_ddpg_rewards):.2f} final training")
print(f"TD3:  {td3_performance:.2f} evaluation reward, {np.mean(final_td3_rewards):.2f} final training")
print(f"TD3 improvement: {((td3_performance - ddpg_performance) / abs(ddpg_performance) * 100):.1f}%")

print("\nChapter 15 Complete! ✓")
print("Continuous action space methods successfully implemented")
print("DDPG and TD3 comparison demonstrates importance of algorithmic improvements")
print("Ready to explore trust region methods (Chapter 16)")

env.close()
