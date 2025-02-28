# Chapter 16: Trust Region Methods
# Proximal Policy Optimization (PPO) and trust region concepts

# Install required packages:
# !pip install torch gymnasium[mujoco] matplotlib numpy tqdm tensorboard -q

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional, Union
from collections import deque
from tqdm import tqdm
import math

# Configure matplotlib for proper font rendering
plt.switch_backend("Agg")
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "PingFang SC", "Arial Unicode MS", "Hiragino Sans GB"]
plt.rcParams["axes.unicode_minus"] = False

print("Chapter 16: Trust Region Methods")
print("=" * 50)

# 1. INTRODUCTION TO TRUST REGION METHODS
print("\n1. Trust Region Policy Optimization")
print("-" * 30)

print("""
TRUST REGION METHODS MOTIVATION:

1. POLICY GRADIENT CHALLENGES:
   - Large policy updates can be destructive
   - Poor step size choice leads to performance collapse
   - Vanilla policy gradients are sample inefficient
   - High variance in gradient estimates

2. TRUST REGION INTUITION:
   - Limit policy changes per update
   - Stay within a "trust region" where linear approximation is valid
   - Ensure monotonic policy improvement
   - Balance exploration vs exploitation

3. MATHEMATICAL FOUNDATION:
   - Constrain KL divergence between old and new policies
   - D_KL(π_old || π_new) ≤ δ
   - Surrogate objective with constraint
   - Natural policy gradients

4. TRUST REGION POLICY OPTIMIZATION (TRPO):
   - Theoretically sound policy improvement
   - Conjugate gradient for constraint optimization
   - Line search for step size
   - Guarantees monotonic improvement

5. PROXIMAL POLICY OPTIMIZATION (PPO):
   - Simpler implementation than TRPO
   - Clipped surrogate objective
   - First-order optimization (Adam)
   - Maintains trust region benefits

6. PPO ADVANTAGES:
   ✓ Simple to implement and tune
   ✓ Sample efficient
   ✓ Stable training
   ✓ Works well across many domains
   ✓ Good balance of performance and simplicity
""")

# 2. GENERALIZED ADVANTAGE ESTIMATION (GAE)
print("\n2. Generalized Advantage Estimation")
print("-" * 30)

class GAECalculator:
    """Generalized Advantage Estimation for variance reduction.
    
    GAE balances bias and variance in advantage estimates using
    exponentially-weighted average of n-step advantages.
    """
    
    def __init__(self, gamma: float = 0.99, gae_lambda: float = 0.95):
        """Initialize GAE calculator.
        
        Args:
            gamma: Discount factor
            gae_lambda: GAE lambda parameter (bias-variance tradeoff)
        """
        self.gamma = gamma
        self.gae_lambda = gae_lambda
    
    def compute_gae(self, rewards: List[float], values: List[float], 
                   dones: List[bool], next_value: float = 0.0) -> Tuple[List[float], List[float]]:
        """Compute GAE advantages and returns.
        
        Args:
            rewards: List of rewards
            values: List of value estimates
            dones: List of done flags
            next_value: Value estimate for next state
            
        Returns:
            Tuple of (advantages, returns)
        """
        advantages = []
        returns = []
        
        # Initialize
        gae = 0
        next_val = next_value
        
        # Compute backwards through episode
        for i in reversed(range(len(rewards))):
            # TD error: δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
            if i == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[i]
                next_value_est = next_val
            else:
                next_non_terminal = 1.0 - dones[i]
                next_value_est = values[i + 1]
            
            delta = rewards[i] + self.gamma * next_value_est * next_non_terminal - values[i]
            
            # GAE: A_t = δ_t + γλ * A_{t+1}
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            
            # Return: G_t = A_t + V(s_t)
            return_val = gae + values[i]
            
            advantages.insert(0, gae)
            returns.insert(0, return_val)
        
        return advantages, returns

# Test GAE calculation
print("Testing GAE calculation...")
gae_calc = GAECalculator(gamma=0.99, gae_lambda=0.95)

# Dummy data
test_rewards = [1.0, 0.5, 0.0, 1.0, 0.0]
test_values = [0.8, 0.6, 0.4, 0.9, 0.1]
test_dones = [False, False, False, False, True]

advantages, returns = gae_calc.compute_gae(test_rewards, test_values, test_dones)
print(f"Advantages: {[f'{a:.3f}' for a in advantages]}")
print(f"Returns: {[f'{r:.3f}' for r in returns]}")

# 3. PPO ACTOR-CRITIC NETWORK
print("\n3. PPO Actor-Critic Network")
print("-" * 30)

class PPOActorCritic(nn.Module):
    """PPO Actor-Critic network for both discrete and continuous actions."""
    
    def __init__(self, state_dim: int, action_dim: int, continuous: bool = False,
                 hidden_dim: int = 64):
        """Initialize PPO Actor-Critic network.
        
        Args:
            state_dim: State space dimension
            action_dim: Action space dimension
            continuous: Whether action space is continuous
            hidden_dim: Hidden layer dimension
        """
        super(PPOActorCritic, self).__init__()
        
        self.continuous = continuous
        self.action_dim = action_dim
        
        # Shared feature extraction
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        
        # Actor head
        if continuous:
            # For continuous actions: output mean and log_std
            self.action_mean = nn.Linear(hidden_dim, action_dim)
            self.action_log_std = nn.Parameter(torch.zeros(action_dim))
        else:
            # For discrete actions: output action logits
            self.action_head = nn.Linear(hidden_dim, action_dim)
        
        # Critic head
        self.value_head = nn.Linear(hidden_dim, 1)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=0.01 if module == self.value_head else 1.0)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through network.
        
        Args:
            state: Input state tensor
            
        Returns:
            Tuple of (action_distribution_params, value)
        """
        features = self.shared_layers(state)
        
        if self.continuous:
            action_mean = self.action_mean(features)
            action_std = torch.exp(self.action_log_std)
            return (action_mean, action_std), self.value_head(features)
        else:
            action_logits = self.action_head(features)
            return action_logits, self.value_head(features)
    
    def get_action_and_value(self, state: torch.Tensor, action: torch.Tensor = None) -> Tuple:
        """Get action and value, optionally evaluate given action.
        
        Args:
            state: Input state tensor
            action: Optional action to evaluate
            
        Returns:
            Tuple of (action, log_prob, entropy, value)
        """
        if self.continuous:
            (action_mean, action_std), value = self.forward(state)
            
            # Create normal distribution
            dist = Normal(action_mean, action_std)
            
            if action is None:
                action = dist.sample()
            
            log_prob = dist.log_prob(action).sum(-1)
            entropy = dist.entropy().sum(-1)
            
            return action, log_prob, entropy, value.squeeze(-1)
        else:
            action_logits, value = self.forward(state)
            
            # Create categorical distribution
            dist = Categorical(logits=action_logits)
            
            if action is None:
                action = dist.sample()
            
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()
            
            return action, log_prob, entropy, value.squeeze(-1)

# Test PPO network
print("Testing PPO Actor-Critic network...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Test discrete actions
discrete_net = PPOActorCritic(state_dim=4, action_dim=2, continuous=False).to(device)
test_state = torch.randn(1, 4).to(device)

action, log_prob, entropy, value = discrete_net.get_action_and_value(test_state)
print(f"Discrete - Action: {action.item()}, Log_prob: {log_prob.item():.4f}, Value: {value.item():.4f}")

# Test continuous actions
continuous_net = PPOActorCritic(state_dim=3, action_dim=1, continuous=True).to(device)
test_state_cont = torch.randn(1, 3).to(device)

action_cont, log_prob_cont, entropy_cont, value_cont = continuous_net.get_action_and_value(test_state_cont)
print(f"Continuous - Action: {action_cont.item():.4f}, Log_prob: {log_prob_cont.item():.4f}, Value: {value_cont.item():.4f}")

# 4. PPO AGENT IMPLEMENTATION
print("\n4. PPO Agent Implementation")
print("-" * 30)

class PPOAgent:
    """Proximal Policy Optimization agent."""
    
    def __init__(self, state_dim: int, action_dim: int, continuous: bool = False,
                 lr: float = 3e-4, gamma: float = 0.99, gae_lambda: float = 0.95,
                 clip_epsilon: float = 0.2, value_coeff: float = 0.5, 
                 entropy_coeff: float = 0.01, max_grad_norm: float = 0.5,
                 ppo_epochs: int = 4, batch_size: int = 64, device: str = None):
        """Initialize PPO agent.
        
        Args:
            state_dim: State space dimension
            action_dim: Action space dimension
            continuous: Whether action space is continuous
            lr: Learning rate
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_epsilon: PPO clipping parameter
            value_coeff: Value loss coefficient
            entropy_coeff: Entropy bonus coefficient
            max_grad_norm: Maximum gradient norm for clipping
            ppo_epochs: Number of PPO epochs per update
            batch_size: Mini-batch size
            device: Device to run on
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.continuous = continuous
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coeff = value_coeff
        self.entropy_coeff = entropy_coeff
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        
        self.device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Network and optimizer
        self.network = PPOActorCritic(state_dim, action_dim, continuous).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
        # GAE calculator
        self.gae_calculator = GAECalculator(gamma, gae_lambda)
        
        # Storage for rollouts
        self.rollout_buffer = {
            'states': [],
            'actions': [],
            'log_probs': [],
            'rewards': [],
            'dones': [],
            'values': []
        }
        
        # Training statistics
        self.episode_rewards = []
        self.policy_losses = []
        self.value_losses = []
        self.entropy_losses = []
        self.total_losses = []
        self.clip_fractions = []
        self.kl_divergences = []
    
    def select_action(self, state: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """Select action using current policy.
        
        Args:
            state: Current state
            
        Returns:
            Tuple of (action, log_prob, value)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, log_prob, _, value = self.network.get_action_and_value(state_tensor)
        
        if self.continuous:
            return action.cpu().numpy().flatten(), log_prob.item(), value.item()
        else:
            return action.item(), log_prob.item(), value.item()
    
    def store_transition(self, state: np.ndarray, action: Union[int, np.ndarray], 
                        log_prob: float, reward: float, done: bool, value: float) -> None:
        """Store transition in rollout buffer.
        
        Args:
            state: Current state
            action: Action taken
            log_prob: Log probability of action
            reward: Reward received
            done: Whether episode ended
            value: Value estimate
        """
        self.rollout_buffer['states'].append(state)
        self.rollout_buffer['actions'].append(action)
        self.rollout_buffer['log_probs'].append(log_prob)
        self.rollout_buffer['rewards'].append(reward)
        self.rollout_buffer['dones'].append(done)
        self.rollout_buffer['values'].append(value)
    
    def compute_advantages_and_returns(self, next_value: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """Compute advantages and returns using GAE.
        
        Args:
            next_value: Value estimate for next state
            
        Returns:
            Tuple of (advantages, returns)
        """
        advantages, returns = self.gae_calculator.compute_gae(
            self.rollout_buffer['rewards'],
            self.rollout_buffer['values'],
            self.rollout_buffer['dones'],
            next_value
        )
        
        return np.array(advantages), np.array(returns)
    
    def update(self, next_value: float = 0.0) -> Dict[str, float]:
        """Update policy using PPO.
        
        Args:
            next_value: Value estimate for next state
            
        Returns:
            Dictionary of training statistics
        """
        # Compute advantages and returns
        advantages, returns = self.compute_advantages_and_returns(next_value)
        
        # Convert rollout data to tensors
        states = torch.FloatTensor(self.rollout_buffer['states']).to(self.device)
        
        if self.continuous:
            actions = torch.FloatTensor(self.rollout_buffer['actions']).to(self.device)
        else:
            actions = torch.LongTensor(self.rollout_buffer['actions']).to(self.device)
        
        old_log_probs = torch.FloatTensor(self.rollout_buffer['log_probs']).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Training statistics
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0
        total_clip_fraction = 0
        total_kl_div = 0
        
        # PPO epochs
        for epoch in range(self.ppo_epochs):
            # Create mini-batches
            indices = torch.randperm(len(states))
            
            for start_idx in range(0, len(states), self.batch_size):
                end_idx = start_idx + self.batch_size
                batch_indices = indices[start_idx:end_idx]
                
                # Mini-batch data
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Forward pass
                _, new_log_probs, entropies, values = self.network.get_action_and_value(
                    batch_states, batch_actions)
                
                # Ratio for PPO clipping
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # Clipped surrogate objective
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(values, batch_returns)
                
                # Entropy loss (for exploration)
                entropy_loss = -entropies.mean()
                
                # Total loss
                total_loss = policy_loss + self.value_coeff * value_loss + self.entropy_coeff * entropy_loss
                
                # Update network
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # Statistics
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy_loss.item()
                
                # Clip fraction (percentage of ratios clipped)
                clip_fraction = ((ratio < (1 - self.clip_epsilon)) | (ratio > (1 + self.clip_epsilon))).float().mean()
                total_clip_fraction += clip_fraction.item()
                
                # KL divergence (for monitoring)
                kl_div = (batch_old_log_probs - new_log_probs).mean()
                total_kl_div += kl_div.item()
        
        # Average statistics
        n_updates = self.ppo_epochs * math.ceil(len(states) / self.batch_size)
        
        stats = {
            'policy_loss': total_policy_loss / n_updates,
            'value_loss': total_value_loss / n_updates,
            'entropy_loss': total_entropy_loss / n_updates,
            'clip_fraction': total_clip_fraction / n_updates,
            'kl_divergence': total_kl_div / n_updates
        }
        
        # Store statistics
        self.policy_losses.append(stats['policy_loss'])
        self.value_losses.append(stats['value_loss'])
        self.entropy_losses.append(stats['entropy_loss'])
        self.clip_fractions.append(stats['clip_fraction'])
        self.kl_divergences.append(stats['kl_divergence'])
        
        # Clear rollout buffer
        for key in self.rollout_buffer:
            self.rollout_buffer[key].clear()
        
        return stats

print("PPO agent implementation complete.")

# 5. TRAINING PPO
print("\n5. Training PPO Agent")
print("-" * 30)

def train_ppo(agent: PPOAgent, env: gym.Env, n_episodes: int = 1000,
              rollout_length: int = 2048, update_frequency: int = 2048) -> List[Dict]:
    """Train PPO agent.
    
    Args:
        agent: PPO agent to train
        env: Environment to train in
        n_episodes: Number of training episodes
        rollout_length: Length of rollouts before update
        update_frequency: How often to update (in steps)
        
    Returns:
        List of episode statistics
    """
    episode_stats = []
    
    print(f"Training PPO for {n_episodes} episodes...")
    
    episode = 0
    total_steps = 0
    
    while episode < n_episodes:
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        
        while True:
            # Select action
            if agent.continuous:
                action, log_prob, value = agent.select_action(state)
            else:
                action, log_prob, value = agent.select_action(state)
            
            # Take step
            if agent.continuous:
                next_state, reward, terminated, truncated, _ = env.step(action)
            else:
                next_state, reward, terminated, truncated, _ = env.step(action)
            
            done = terminated or truncated
            
            # Store transition
            agent.store_transition(state, action, log_prob, reward, done, value)
            
            episode_reward += reward
            episode_length += 1
            total_steps += 1
            state = next_state
            
            # Update policy
            if total_steps % update_frequency == 0:
                # Get next value for bootstrapping
                if done:
                    next_value = 0.0
                else:
                    with torch.no_grad():
                        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(agent.device)
                        _, _, _, next_value = agent.network.get_action_and_value(next_state_tensor)
                        next_value = next_value.item()
                
                # Update
                update_stats = agent.update(next_value)
                
                print(f"Step {total_steps}: Policy Loss = {update_stats['policy_loss']:.4f}, "
                      f"Value Loss = {update_stats['value_loss']:.4f}, "
                      f"Clip Fraction = {update_stats['clip_fraction']:.4f}")
            
            if done:
                break
        
        # Store episode statistics
        episode_stat = {
            'episode': episode,
            'reward': episode_reward,
            'length': episode_length,
            'total_steps': total_steps
        }
        episode_stats.append(episode_stat)
        agent.episode_rewards.append(episode_reward)
        
        # Logging
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean([s['reward'] for s in episode_stats[-50:]])
            print(f"Episode {episode+1}: Avg Reward = {avg_reward:.2f}")
        
        episode += 1
    
    return episode_stats

# Create environment and train PPO
env_name = 'CartPole-v1'
env = gym.make(env_name)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
continuous = False

print(f"Environment: {env_name}")
print(f"State dimension: {state_dim}")
print(f"Action dimension: {action_dim}")
print(f"Continuous: {continuous}")

# Train PPO agent
print("\nTraining PPO agent...")
ppo_agent = PPOAgent(state_dim, action_dim, continuous=continuous, 
                     clip_epsilon=0.2, ppo_epochs=4, batch_size=64)
ppo_stats = train_ppo(ppo_agent, env, n_episodes=500, update_frequency=2048)

print(f"\nPPO training completed!")
final_ppo_rewards = [s['reward'] for s in ppo_stats[-50:]]
print(f"Final average reward: {np.mean(final_ppo_rewards):.2f}")

# 6. VISUALIZATION AND ANALYSIS
print("\n6. Creating Visualizations")
print("-" * 30)

# Create comprehensive plots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Learning curve
ppo_rewards = [s['reward'] for s in ppo_stats]
window_size = 25
if len(ppo_rewards) >= window_size:
    ppo_smoothed = np.convolve(ppo_rewards, np.ones(window_size)/window_size, mode='valid')
    ax1.plot(ppo_smoothed, linewidth=2, color='blue', label='PPO')
else:
    ax1.plot(ppo_rewards, linewidth=2, color='blue', label='PPO')

ax1.axhline(y=475, color='red', linestyle='--', alpha=0.7, label='Target')
ax1.set_xlabel('Episode')
ax1.set_ylabel('Episode Reward')
ax1.set_title('PPO Learning Curve')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Loss components
if ppo_agent.policy_losses:
    policy_smooth = np.convolve(ppo_agent.policy_losses, np.ones(10)/10, mode='valid')
    value_smooth = np.convolve(ppo_agent.value_losses, np.ones(10)/10, mode='valid')
    
    ax2.plot(policy_smooth, label='Policy Loss', alpha=0.8)
    ax2.plot(value_smooth, label='Value Loss', alpha=0.8)
    ax2.set_xlabel('Update')
    ax2.set_ylabel('Loss')
    ax2.set_title('PPO Loss Components')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

# Plot 3: Clipping analysis
if ppo_agent.clip_fractions:
    ax3.plot(ppo_agent.clip_fractions, alpha=0.8, color='green')
    ax3.axhline(y=0.1, color='red', linestyle='--', alpha=0.7, label='Target (~0.1)')
    ax3.set_xlabel('Update')
    ax3.set_ylabel('Clip Fraction')
    ax3.set_title('PPO Clipping Behavior')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

# Plot 4: KL divergence
if ppo_agent.kl_divergences:
    ax4.plot(ppo_agent.kl_divergences, alpha=0.8, color='purple')
    ax4.set_xlabel('Update')
    ax4.set_ylabel('KL Divergence')
    ax4.set_title('Policy Change (KL Divergence)')
    ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('pytorch_rl_tutorial/chapter_16_ppo_results.png', dpi=150, bbox_inches='tight')
print("Results saved to chapter_16_ppo_results.png")

# 7. PPO ANALYSIS
print("\n7. PPO and Trust Region Analysis")
print("-" * 30)

print("""
PPO INSIGHTS AND ANALYSIS:

1. CLIPPING MECHANISM:
   - Clip ratio prevents large policy updates
   - Typical clip fraction should be ~0.1-0.3
   - Higher clip fractions indicate aggressive updates
   - Lower fractions suggest conservative updates

2. TRUST REGION ENFORCEMENT:
   - PPO approximates trust region constraint via clipping
   - Simpler than TRPO's conjugate gradient method
   - More robust to hyperparameter choices
   - Maintains policy improvement guarantees (approximately)

3. HYPERPARAMETER SENSITIVITY:
   - Clip epsilon (ε): typically 0.1-0.3
   - PPO epochs: 3-10 updates per rollout
   - Batch size: larger is often better
   - Learning rate: 3e-4 is common starting point

4. ADVANTAGES OVER OTHER METHODS:
   ✓ More stable than vanilla policy gradients
   ✓ Simpler than TRPO
   ✓ Better sample efficiency than REINFORCE
   ✓ Works well across many domains
   ✓ Easy to implement and tune

5. COMMON ISSUES:
   - Value function overfitting
   - Insufficient exploration
   - Hyperparameter sensitivity
   - Environment-specific tuning needed

6. BEST PRACTICES:
   - Use GAE for advantage estimation
   - Normalize advantages
   - Clip gradients for stability
   - Monitor clip fraction and KL divergence
   - Use orthogonal weight initialization

7. THEORETICAL FOUNDATION:
   - Surrogate objective with constraint
   - Conservative policy improvement
   - Prevents catastrophic policy collapse
   - Balances exploration and exploitation
""")

# Performance analysis
print(f"\nPerformance Summary:")
print(f"PPO final performance: {np.mean(final_ppo_rewards):.2f} average reward")
print(f"Training episodes: {len(ppo_stats)}")
if ppo_agent.clip_fractions:
    print(f"Average clip fraction: {np.mean(ppo_agent.clip_fractions[-10:]):.3f}")
if ppo_agent.kl_divergences:
    print(f"Average KL divergence: {np.mean(ppo_agent.kl_divergences[-10:]):.6f}")

print("\nChapter 16 Complete! ✓")
print("Trust region methods and PPO successfully implemented")
print("PPO demonstrates excellent balance of simplicity and performance")
print("Ready to explore advanced topics in subsequent chapters")

env.close()
