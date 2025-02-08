# Chapter 12: Actor-Critic Methods - A2C and A3C
# Combining value and policy learning for improved stability

# Install required packages:
# !pip install torch gymnasium[atari] gymnasium[accept-rom-license] ale-py opencv-python matplotlib tqdm -q

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.multiprocessing as mp
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional, Any
from collections import deque
import threading
import time
from tqdm import tqdm
import cv2
import queue
import random

# Configure matplotlib for proper font rendering
plt.switch_backend("Agg")
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "PingFang SC", "Arial Unicode MS", "Hiragino Sans GB"]
plt.rcParams["axes.unicode_minus"] = False

print("Chapter 12: Actor-Critic Methods - A2C and A3C")
print("=" * 50)

# 1. INTRODUCTION TO ACTOR-CRITIC
print("\n1. Actor-Critic Framework")
print("-" * 30)

print("""
ACTOR-CRITIC INTUITION:

1. MOTIVATION:
   - REINFORCE has high variance due to Monte Carlo returns
   - DQN can't handle continuous actions naturally
   - Actor-Critic combines best of both approaches

2. KEY COMPONENTS:
   - ACTOR: Policy network π_θ(a|s) that selects actions
   - CRITIC: Value network V_φ(s) that estimates state values
   - Actor uses critic's estimates to reduce variance

3. ADVANTAGE FUNCTION:
   - A(s,a) = Q(s,a) - V(s)
   - Measures how much better action a is compared to average
   - Reduces variance compared to raw returns

4. ACTOR-CRITIC UPDATE:
   - Actor: ∇_θ J(θ) ≈ E[∇_θ log π_θ(a|s) * A(s,a)]
   - Critic: Minimize TD error δ = r + γV(s') - V(s)

5. VARIANTS:
   - A2C: Advantage Actor-Critic (synchronous)
   - A3C: Asynchronous Advantage Actor-Critic
   - PPO: Proximal Policy Optimization
   - SAC: Soft Actor-Critic

6. ADVANTAGES:
   ✓ Lower variance than REINFORCE
   ✓ More stable than pure policy gradients
   ✓ Handles continuous action spaces
   ✓ Online learning capability
""")

# 2. SHARED NETWORK ARCHITECTURE
print("\n2. Actor-Critic Network Architecture")
print("-" * 30)

class ActorCriticNetwork(nn.Module):
    """Shared network for Actor-Critic with separate heads.
    
    Uses a shared feature extractor with separate heads for
    policy (actor) and value function (critic).
    """
    
    def __init__(self, input_dim: int, action_dim: int, hidden_dim: int = 256):
        """Initialize Actor-Critic network.
        
        Args:
            input_dim: Input state dimension
            action_dim: Number of discrete actions
            hidden_dim: Hidden layer dimension
        """
        super(ActorCriticNetwork, self).__init__()
        
        # Shared feature extraction layers
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor head: outputs action logits
        self.actor_head = nn.Linear(hidden_dim, action_dim)
        
        # Critic head: outputs state value
        self.critic_head = nn.Linear(hidden_dim, 1)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through network.
        
        Args:
            state: Input state tensor
            
        Returns:
            Tuple of (action_logits, state_value)
        """
        # Shared feature extraction
        features = self.shared_layers(state)
        
        # Separate heads
        action_logits = self.actor_head(features)
        state_value = self.critic_head(features)
        
        return action_logits, state_value
    
    def get_action_and_value(self, state: torch.Tensor) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action and get value estimate.
        
        Args:
            state: Input state tensor
            
        Returns:
            Tuple of (action, log_prob, entropy, value)
        """
        action_logits, value = self.forward(state)
        
        # Create action distribution
        distribution = Categorical(logits=action_logits)
        
        # Sample action
        action = distribution.sample()
        log_prob = distribution.log_prob(action)
        entropy = distribution.entropy()
        
        return action.item(), log_prob, entropy, value.squeeze()
    
    def evaluate_actions(self, states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions for given states.
        
        Args:
            states: Batch of states
            actions: Batch of actions
            
        Returns:
            Tuple of (log_probs, values, entropies)
        """
        action_logits, values = self.forward(states)
        
        distribution = Categorical(logits=action_logits)
        log_probs = distribution.log_prob(actions)
        entropies = distribution.entropy()
        
        return log_probs, values.squeeze(), entropies

# Test the network
print("Testing Actor-Critic network...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

test_ac_net = ActorCriticNetwork(input_dim=4, action_dim=2).to(device)
test_state = torch.randn(1, 4).to(device)

action_logits, state_value = test_ac_net(test_state)
action, log_prob, entropy, value = test_ac_net.get_action_and_value(test_state)

print(f"Action logits: {action_logits.squeeze().detach().cpu().numpy()}")
print(f"State value: {state_value.item():.4f}")
print(f"Sampled action: {action}")
print(f"Log probability: {log_prob.item():.4f}")
print(f"Entropy: {entropy.item():.4f}")

# 3. A2C (ADVANTAGE ACTOR-CRITIC) IMPLEMENTATION
print("\n3. A2C Implementation")
print("-" * 30)

class A2CAgent:
    """Advantage Actor-Critic (A2C) agent.
    
    Synchronous version that collects experiences from multiple
    parallel environments and updates the network synchronously.
    """
    
    def __init__(self, state_dim: int, action_dim: int, lr: float = 0.001,
                 gamma: float = 0.99, value_coeff: float = 0.5, entropy_coeff: float = 0.01,
                 max_grad_norm: float = 0.5, n_steps: int = 5, device: str = None):
        """Initialize A2C agent.
        
        Args:
            state_dim: State space dimension
            action_dim: Action space dimension
            lr: Learning rate
            gamma: Discount factor
            value_coeff: Value loss coefficient
            entropy_coeff: Entropy regularization coefficient
            max_grad_norm: Maximum gradient norm for clipping
            n_steps: Number of steps for n-step returns
            device: Device to run on
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.value_coeff = value_coeff
        self.entropy_coeff = entropy_coeff
        self.max_grad_norm = max_grad_norm
        self.n_steps = n_steps
        
        self.device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Network and optimizer
        self.network = ActorCriticNetwork(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
        # Training statistics
        self.episode_rewards = []
        self.policy_losses = []
        self.value_losses = []
        self.entropy_losses = []
        self.total_losses = []
    
    def select_action(self, state: np.ndarray) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """Select action using current policy.
        
        Args:
            state: Current state
            
        Returns:
            Tuple of (action, log_prob, value)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, log_prob, _, value = self.network.get_action_and_value(state_tensor)
        
        return action, log_prob, value
    
    def compute_gae(self, rewards: List[float], values: List[torch.Tensor], 
                   dones: List[bool], next_value: torch.Tensor, 
                   gae_lambda: float = 0.95) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Compute Generalized Advantage Estimation (GAE).
        
        Args:
            rewards: List of rewards
            values: List of value estimates
            dones: List of done flags
            next_value: Value estimate for next state
            gae_lambda: GAE lambda parameter
            
        Returns:
            Tuple of (advantages, returns)
        """
        advantages = []
        returns = []
        
        gae = 0
        
        # Compute advantages backwards
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[i]
                next_value_est = next_value
            else:
                next_non_terminal = 1.0 - dones[i]
                next_value_est = values[i + 1]
            
            delta = rewards[i] + self.gamma * next_value_est * next_non_terminal - values[i]
            gae = delta + self.gamma * gae_lambda * next_non_terminal * gae
            
            advantages.insert(0, gae)
            returns.insert(0, gae + values[i])
        
        return advantages, returns
    
    def update(self, states: List[np.ndarray], actions: List[int], 
              rewards: List[float], dones: List[bool], values: List[torch.Tensor],
              log_probs: List[torch.Tensor], next_value: torch.Tensor) -> Dict[str, float]:
        """Update the network using collected experiences.
        
        Args:
            states: List of states
            actions: List of actions
            rewards: List of rewards
            dones: List of done flags
            values: List of value estimates
            log_probs: List of log probabilities
            next_value: Value estimate for next state
            
        Returns:
            Dictionary of loss components
        """
        # Compute advantages and returns
        advantages, returns = self.compute_gae(rewards, values, dones, next_value)
        
        # Convert to tensors
        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        old_log_probs = torch.stack(log_probs).to(self.device)
        advantages_tensor = torch.stack(advantages).to(self.device)
        returns_tensor = torch.stack(returns).to(self.device)
        
        # Normalize advantages
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
        
        # Forward pass
        new_log_probs, new_values, entropies = self.network.evaluate_actions(states_tensor, actions_tensor)
        
        # Policy loss (actor)
        policy_loss = -(new_log_probs * advantages_tensor).mean()
        
        # Value loss (critic)
        value_loss = F.mse_loss(new_values, returns_tensor)
        
        # Entropy loss (for exploration)
        entropy_loss = -entropies.mean()
        
        # Total loss
        total_loss = policy_loss + self.value_coeff * value_loss + self.entropy_coeff * entropy_loss
        
        # Update network
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        # Store losses
        losses = {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'total_loss': total_loss.item()
        }
        
        self.policy_losses.append(policy_loss.item())
        self.value_losses.append(value_loss.item())
        self.entropy_losses.append(entropy_loss.item())
        self.total_losses.append(total_loss.item())
        
        return losses

print("A2C agent implementation complete.")

# 4. TRAINING A2C
print("\n4. Training A2C Agent")
print("-" * 30)

def train_a2c(agent: A2CAgent, env: gym.Env, n_episodes: int = 1000,
              max_steps: int = 500, update_frequency: int = 5) -> List[Dict]:
    """Train A2C agent.
    
    Args:
        agent: A2C agent to train
        env: Environment to train in
        n_episodes: Number of training episodes
        max_steps: Maximum steps per episode
        update_frequency: How often to update the network
        
    Returns:
        List of episode statistics
    """
    episode_stats = []
    
    print(f"Training A2C for {n_episodes} episodes...")
    
    episode = 0
    while episode < n_episodes:
        # Collect batch of experiences
        batch_states = []
        batch_actions = []
        batch_rewards = []
        batch_dones = []
        batch_values = []
        batch_log_probs = []
        
        episode_reward = 0
        state, _ = env.reset()
        
        for step in range(max_steps):
            # Select action
            action, log_prob, value = agent.select_action(state)
            
            # Take step
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store experience
            batch_states.append(state)
            batch_actions.append(action)
            batch_rewards.append(reward)
            batch_dones.append(done)
            batch_values.append(value)
            batch_log_probs.append(log_prob)
            
            episode_reward += reward
            state = next_state
            
            # Update every update_frequency steps or at episode end
            if (step + 1) % update_frequency == 0 or done:
                # Get next value for bootstrapping
                if done:
                    next_value = torch.tensor(0.0).to(agent.device)
                else:
                    with torch.no_grad():
                        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(agent.device)
                        _, next_value = agent.network(next_state_tensor)
                        next_value = next_value.squeeze()
                
                # Update network
                losses = agent.update(
                    batch_states, batch_actions, batch_rewards, 
                    batch_dones, batch_values, batch_log_probs, next_value
                )
                
                # Clear batch
                batch_states.clear()
                batch_actions.clear()
                batch_rewards.clear()
                batch_dones.clear()
                batch_values.clear()
                batch_log_probs.clear()
            
            if done:
                break
        
        # Store episode statistics
        episode_stat = {
            'episode': episode,
            'reward': episode_reward,
            'steps': step + 1,
            **losses
        }
        episode_stats.append(episode_stat)
        agent.episode_rewards.append(episode_reward)
        
        # Logging
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean([s['reward'] for s in episode_stats[-100:]])
            avg_loss = np.mean([s['total_loss'] for s in episode_stats[-100:]])
            print(f"Episode {episode+1}: Avg Reward = {avg_reward:.2f}, Avg Loss = {avg_loss:.4f}")
        
        episode += 1
    
    return episode_stats

# Create environment and train A2C
env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

print(f"Environment: CartPole-v1")
print(f"State dimension: {state_dim}")
print(f"Action dimension: {action_dim}")

# Train A2C agent
print("\nTraining A2C agent...")
a2c_agent = A2CAgent(state_dim, action_dim, lr=0.001, n_steps=5)
a2c_stats = train_a2c(a2c_agent, env, n_episodes=800, update_frequency=5)

print(f"\nA2C training completed!")
final_a2c_rewards = [s['reward'] for s in a2c_stats[-100:]]
print(f"Final average reward: {np.mean(final_a2c_rewards):.2f}")

# 5. A3C CONCEPT AND SIMPLIFIED IMPLEMENTATION
print("\n5. A3C Concept and Threading")
print("-" * 30)

print("""
A3C (ASYNCHRONOUS ADVANTAGE ACTOR-CRITIC):

1. KEY DIFFERENCES FROM A2C:
   - Multiple worker threads collect experiences in parallel
   - Each worker has its own environment copy
   - Workers update shared global network asynchronously
   - No experience replay needed

2. ADVANTAGES:
   ✓ Better exploration through diverse experiences
   ✓ More stable gradients from diverse data
   ✓ Faster training through parallelization
   ✓ Less memory usage (no replay buffer)

3. IMPLEMENTATION CHALLENGES:
   - Thread synchronization
   - Shared memory management
   - Gradient accumulation
   - Load balancing

4. HOGWILD! UPDATES:
   - Workers update global network without locks
   - Surprisingly stable due to sparse gradients
   - Each worker pulls latest weights periodically

Note: Full A3C implementation requires careful threading.
Here we show the concept with simplified threading.
""")

class A3CWorker:
    """A3C Worker thread (simplified implementation)."""
    
    def __init__(self, worker_id: int, global_network: ActorCriticNetwork,
                 optimizer: torch.optim.Optimizer, env_name: str,
                 gamma: float = 0.99, n_steps: int = 5):
        """Initialize A3C worker.
        
        Args:
            worker_id: Unique worker identifier
            global_network: Shared global network
            optimizer: Shared optimizer
            env_name: Environment name
            gamma: Discount factor
            n_steps: Number of steps before update
        """
        self.worker_id = worker_id
        self.global_network = global_network
        self.optimizer = optimizer
        self.env_name = env_name
        self.gamma = gamma
        self.n_steps = n_steps
        
        # Local network (copy of global)
        self.local_network = ActorCriticNetwork(
            global_network.shared_layers[0].in_features,
            global_network.actor_head.out_features
        ).to(global_network.shared_layers[0].weight.device)
        
        # Environment
        self.env = gym.make(env_name)
        
        # Statistics
        self.episode_rewards = []
        self.episode_count = 0
    
    def sync_with_global(self):
        """Synchronize local network with global network."""
        self.local_network.load_state_dict(self.global_network.state_dict())
    
    def compute_loss(self, states: torch.Tensor, actions: torch.Tensor,
                    rewards: torch.Tensor, dones: torch.Tensor,
                    next_value: torch.Tensor) -> torch.Tensor:
        """Compute A3C loss."""
        # Get policy and value outputs
        log_probs, values, entropies = self.local_network.evaluate_actions(states, actions)
        
        # Compute returns
        returns = []
        R = next_value
        for i in reversed(range(len(rewards))):
            R = rewards[i] + self.gamma * R * (1 - dones[i])
            returns.insert(0, R)
        returns = torch.tensor(returns).to(states.device)
        
        # Compute advantages
        advantages = returns - values
        
        # Losses
        policy_loss = -(log_probs * advantages.detach()).mean()
        value_loss = advantages.pow(2).mean()
        entropy_loss = -entropies.mean()
        
        total_loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss
        
        return total_loss

# Simplified A3C training demonstration
print("A3C concept implementation complete.")
print("Note: Full A3C requires proper multi-threading and shared memory.")

# 6. EVALUATION AND COMPARISON
print("\n6. Evaluation and Analysis")
print("-" * 30)

def evaluate_agent(agent: A2CAgent, env: gym.Env, n_episodes: int = 20) -> float:
    """Evaluate trained agent.
    
    Args:
        agent: Trained agent
        env: Environment
        n_episodes: Number of evaluation episodes
        
    Returns:
        Average episode reward
    """
    total_reward = 0
    
    for _ in range(n_episodes):
        state, _ = env.reset()
        episode_reward = 0
        
        for _ in range(500):
            # Get action without exploration
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                action_logits, _ = agent.network(state_tensor)
                action = torch.argmax(action_logits, dim=1).item()
            
            state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            
            if terminated or truncated:
                break
        
        total_reward += episode_reward
    
    return total_reward / n_episodes

# Evaluate A2C agent
a2c_performance = evaluate_agent(a2c_agent, env, n_episodes=20)
print(f"A2C evaluation performance: {a2c_performance:.2f}")

# 7. VISUALIZATION
print("\n7. Creating Visualizations")
print("-" * 30)

# Create comprehensive plots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Learning curve
a2c_rewards = [s['reward'] for s in a2c_stats]
window_size = 50
if len(a2c_rewards) >= window_size:
    a2c_smoothed = np.convolve(a2c_rewards, np.ones(window_size)/window_size, mode='valid')
    ax1.plot(a2c_smoothed, label='A2C', linewidth=2, color='blue')
else:
    ax1.plot(a2c_rewards, label='A2C', linewidth=2, color='blue')

ax1.axhline(y=475, color='red', linestyle='--', alpha=0.7, label='Target')
ax1.set_xlabel('Episode')
ax1.set_ylabel('Episode Reward')
ax1.set_title('A2C Learning Curve')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Loss components
policy_losses = [s.get('policy_loss', 0) for s in a2c_stats]
value_losses = [s.get('value_loss', 0) for s in a2c_stats]
entropy_losses = [s.get('entropy_loss', 0) for s in a2c_stats]

# Smooth losses
if len(policy_losses) >= 20:
    policy_smooth = np.convolve(policy_losses, np.ones(20)/20, mode='valid')
    value_smooth = np.convolve(value_losses, np.ones(20)/20, mode='valid')
    entropy_smooth = np.convolve(entropy_losses, np.ones(20)/20, mode='valid')
    
    ax2.plot(policy_smooth, label='Policy Loss', alpha=0.8)
    ax2.plot(value_smooth, label='Value Loss', alpha=0.8)
    ax2.plot(entropy_smooth, label='Entropy Loss', alpha=0.8)
else:
    ax2.plot(policy_losses, label='Policy Loss', alpha=0.8)
    ax2.plot(value_losses, label='Value Loss', alpha=0.8)
    ax2.plot(entropy_losses, label='Entropy Loss', alpha=0.8)

ax2.set_xlabel('Episode')
ax2.set_ylabel('Loss')
ax2.set_title('Loss Components')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Episode length progression
episode_lengths = [s['steps'] for s in a2c_stats]
ax3.plot(episode_lengths, alpha=0.7, color='green')
if len(episode_lengths) >= window_size:
    length_smoothed = np.convolve(episode_lengths, np.ones(window_size)/window_size, mode='valid')
    ax3.plot(length_smoothed, linewidth=2, color='darkgreen', label='Smoothed')
    ax3.legend()

ax3.set_xlabel('Episode')
ax3.set_ylabel('Episode Length')
ax3.set_title('Episode Length Over Time')
ax3.grid(True, alpha=0.3)

# Plot 4: Value function visualization
# Sample states and their value estimates
test_states = []
test_values = []

for _ in range(200):
    state, _ = env.reset()
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(a2c_agent.device)
    
    with torch.no_grad():
        _, value = a2c_agent.network(state_tensor)
    
    test_states.append(state)
    test_values.append(value.item())

test_states = np.array(test_states)
test_values = np.array(test_values)

# Plot cart position vs estimated value
cart_positions = test_states[:, 0]
ax4.scatter(cart_positions, test_values, alpha=0.6, s=20)
ax4.set_xlabel('Cart Position')
ax4.set_ylabel('Estimated State Value')
ax4.set_title('Value Function Estimates')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('pytorch_rl_tutorial/chapter_12_actor_critic_results.png', dpi=150, bbox_inches='tight')
print("Results saved to chapter_12_actor_critic_results.png")

# 8. ACTOR-CRITIC ADVANTAGES AND ANALYSIS
print("\n8. Actor-Critic Analysis")
print("-" * 30)

print("""
ACTOR-CRITIC METHODS ANALYSIS:

1. VARIANCE REDUCTION:
   - Critic provides lower variance baselines
   - Advantage function A(s,a) = Q(s,a) - V(s)
   - More stable learning than pure policy gradients

2. BIAS-VARIANCE TRADEOFF:
   - Introduces bias through value function approximation
   - Reduces variance significantly
   - Usually net positive for learning speed

3. SAMPLE EFFICIENCY:
   - More sample efficient than REINFORCE
   - Online learning without experience replay
   - Good for environments where data collection is expensive

4. STABILITY:
   - More stable than pure policy gradients
   - Shared feature extraction helps both actor and critic
   - Entropy regularization maintains exploration

5. PARALLELIZATION:
   - A3C enables effective parallel training
   - Diverse experiences from multiple workers
   - Better gradient estimates

6. HYPERPARAMETER SENSITIVITY:
   - Value coefficient balances critic learning
   - Entropy coefficient controls exploration
   - Learning rate affects both actor and critic

KEY INSIGHTS:
- Actor-Critic combines strengths of value and policy methods
- Shared networks enable efficient feature learning
- Asynchronous training (A3C) improves exploration
- Foundation for advanced methods like PPO, SAC
""")

print(f"\nPerformance Summary:")
print(f"A2C final performance: {a2c_performance:.2f} average reward")
print(f"Training episodes: {len(a2c_stats)}")
print(f"Final average reward: {np.mean(final_a2c_rewards):.2f}")

print("\nChapter 12 Complete! ✓")
print("Actor-Critic methods successfully implemented and analyzed")
print("Ready to explore continuous action spaces (Chapter 15)")

env.close()
