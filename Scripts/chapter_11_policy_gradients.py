# Chapter 11: Policy Gradients
# REINFORCE algorithm and policy-based reinforcement learning

# Install required packages:
# !pip install torch gymnasium[all] matplotlib numpy tqdm -q

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
from collections import deque
from tqdm import tqdm
import random

# Configure matplotlib for proper font rendering
plt.switch_backend("Agg")
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "PingFang SC", "Arial Unicode MS", "Hiragino Sans GB"]
plt.rcParams["axes.unicode_minus"] = False

print("Chapter 11: Policy Gradients")
print("=" * 50)

# 1. INTRODUCTION TO POLICY GRADIENTS
print("\n1. From Value-Based to Policy-Based RL")
print("-" * 30)

print("""
POLICY GRADIENTS INTUITION:

1. VALUE-BASED RL (DQN):
   - Learn value function Q(s,a)
   - Derive policy indirectly: π(s) = argmax_a Q(s,a)
   - Works well for discrete actions
   - Can suffer from overestimation bias

2. POLICY-BASED RL:
   - Learn policy directly: π_θ(a|s)
   - No value function needed (though can be used)
   - Natural for continuous action spaces
   - Can learn stochastic policies

3. ADVANTAGES OF POLICY GRADIENTS:
   ✓ Handle continuous action spaces naturally
   ✓ Learn stochastic policies (useful for exploration)
   ✓ Better convergence properties in some cases
   ✓ Can incorporate prior knowledge about policy structure

4. DISADVANTAGES:
   ✗ Higher variance in gradient estimates
   ✗ Can be sample inefficient
   ✗ Sensitive to learning rate
   ✗ Can get stuck in local optima

5. POLICY GRADIENT THEOREM:
   ∇_θ J(θ) = E[∇_θ log π_θ(a|s) * G_t]
   
   Where:
   - J(θ) is the policy performance
   - G_t is the return from time t
   - This allows us to update policy parameters directly
""")

# 2. POLICY NETWORK IMPLEMENTATION
print("\n2. Policy Network Architecture")
print("-" * 30)

class PolicyNetwork(nn.Module):
    """Neural network for policy function approximation.
    
    Maps states to action probabilities for discrete action spaces.
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        """Initialize policy network.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Number of discrete actions
            hidden_dim: Hidden layer dimension
        """
        super(PolicyNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
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
        """Forward pass through policy network.
        
        Args:
            state: Input state tensor
            
        Returns:
            Action logits (before softmax)
        """
        return self.network(state)
    
    def get_action_probs(self, state: torch.Tensor) -> torch.Tensor:
        """Get action probabilities.
        
        Args:
            state: Input state tensor
            
        Returns:
            Action probabilities (after softmax)
        """
        logits = self.forward(state)
        return F.softmax(logits, dim=-1)
    
    def sample_action(self, state: torch.Tensor) -> Tuple[int, torch.Tensor]:
        """Sample action from policy.
        
        Args:
            state: Input state tensor
            
        Returns:
            Tuple of (action, log_probability)
        """
        logits = self.forward(state)
        distribution = Categorical(logits=logits)
        action = distribution.sample()
        log_prob = distribution.log_prob(action)
        
        return action.item(), log_prob
    
    def get_log_prob(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Get log probability of given action.
        
        Args:
            state: Input state tensor
            action: Action tensor
            
        Returns:
            Log probability of action
        """
        logits = self.forward(state)
        distribution = Categorical(logits=logits)
        return distribution.log_prob(action)

# Test policy network
print("Testing policy network...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create test network
test_policy = PolicyNetwork(state_dim=4, action_dim=2).to(device)
test_state = torch.randn(1, 4).to(device)

# Test forward pass
logits = test_policy(test_state)
probs = test_policy.get_action_probs(test_state)
action, log_prob = test_policy.sample_action(test_state)

print(f"Test state shape: {test_state.shape}")
print(f"Action logits: {logits.squeeze().detach().cpu().numpy()}")
print(f"Action probabilities: {probs.squeeze().detach().cpu().numpy()}")
print(f"Sampled action: {action}, log_prob: {log_prob.item():.4f}")

# 3. REINFORCE ALGORITHM
print("\n3. REINFORCE Algorithm Implementation")
print("-" * 30)

class REINFORCEAgent:
    """REINFORCE agent implementation.
    
    Monte Carlo policy gradient algorithm that updates policy
    based on complete episode returns.
    """
    
    def __init__(self, state_dim: int, action_dim: int, lr: float = 0.01,
                 gamma: float = 0.99, use_baseline: bool = True, device: str = None):
        """Initialize REINFORCE agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Number of discrete actions
            lr: Learning rate
            gamma: Discount factor
            use_baseline: Whether to use baseline for variance reduction
            device: Device to run on
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.use_baseline = use_baseline
        
        self.device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Policy network
        self.policy_net = PolicyNetwork(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        
        # Episode storage
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
        self.episode_log_probs = []
        
        # Baseline (running average of returns)
        self.baseline = 0.0
        self.baseline_lr = 0.01
        
        # Tracking
        self.episode_returns = []
        self.policy_losses = []
        self.episode_count = 0
    
    def select_action(self, state: np.ndarray) -> int:
        """Select action using current policy.
        
        Args:
            state: Current state
            
        Returns:
            Selected action
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action, log_prob = self.policy_net.sample_action(state_tensor)
        
        # Store for training
        self.episode_states.append(state)
        self.episode_actions.append(action)
        self.episode_log_probs.append(log_prob)
        
        return action
    
    def store_reward(self, reward: float) -> None:
        """Store reward for current step.
        
        Args:
            reward: Reward received
        """
        self.episode_rewards.append(reward)
    
    def compute_returns(self) -> List[float]:
        """Compute discounted returns for episode.
        
        Returns:
            List of discounted returns
        """
        returns = []
        G = 0
        
        # Compute returns backward through episode
        for reward in reversed(self.episode_rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)
        
        return returns
    
    def update_policy(self) -> float:
        """Update policy using REINFORCE algorithm.
        
        Returns:
            Policy loss value
        """
        if not self.episode_rewards:
            return 0.0
        
        # Compute returns
        returns = self.compute_returns()
        
        # Convert to tensors
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        log_probs_tensor = torch.stack(self.episode_log_probs)
        
        # Normalize returns for numerical stability
        if len(returns) > 1:
            returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-8)
        
        # Apply baseline if enabled
        if self.use_baseline:
            returns_tensor = returns_tensor - self.baseline
        
        # Compute policy loss
        # REINFORCE loss: -log(π(a|s)) * G
        policy_loss = -(log_probs_tensor * returns_tensor).mean()
        
        # Update policy
        self.optimizer.zero_grad()
        policy_loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Update baseline
        if self.use_baseline:
            episode_return = sum(self.episode_rewards)
            self.baseline += self.baseline_lr * (episode_return - self.baseline)
        
        # Store statistics
        self.policy_losses.append(policy_loss.item())
        self.episode_returns.append(sum(self.episode_rewards))
        self.episode_count += 1
        
        return policy_loss.item()
    
    def end_episode(self) -> float:
        """End current episode and update policy.
        
        Returns:
            Policy loss from update
        """
        loss = self.update_policy()
        
        # Clear episode data
        self.episode_states.clear()
        self.episode_actions.clear()
        self.episode_rewards.clear()
        self.episode_log_probs.clear()
        
        return loss
    
    def get_action_distribution(self, state: np.ndarray) -> np.ndarray:
        """Get action probability distribution for given state.
        
        Args:
            state: Input state
            
        Returns:
            Action probabilities
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = self.policy_net.get_action_probs(state_tensor)
        return probs.squeeze().cpu().numpy()

print("REINFORCE agent implementation complete.")

# 4. TRAINING REINFORCE ON CARTPOLE
print("\n4. Training REINFORCE on CartPole")
print("-" * 30)

def train_reinforce(agent: REINFORCEAgent, env: gym.Env, n_episodes: int = 1000,
                   max_steps: int = 500, target_reward: float = 450.0) -> List[Dict]:
    """Train REINFORCE agent.
    
    Args:
        agent: REINFORCE agent to train
        env: Environment to train in
        n_episodes: Number of training episodes
        max_steps: Maximum steps per episode
        target_reward: Target average reward for early stopping
        
    Returns:
        List of episode statistics
    """
    episode_stats = []
    recent_rewards = deque(maxlen=100)
    
    print(f"Training REINFORCE for up to {n_episodes} episodes...")
    
    for episode in range(n_episodes):
        state, _ = env.reset()
        episode_reward = 0
        
        # Run episode
        for step in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            agent.store_reward(reward)
            episode_reward += reward
            
            state = next_state
            
            if terminated or truncated:
                break
        
        # Update policy at end of episode
        policy_loss = agent.end_episode()
        recent_rewards.append(episode_reward)
        
        # Store episode statistics
        episode_stat = {
            'episode': episode,
            'reward': episode_reward,
            'policy_loss': policy_loss,
            'steps': step + 1,
            'baseline': agent.baseline if agent.use_baseline else 0
        }
        episode_stats.append(episode_stat)
        
        # Logging
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(recent_rewards)
            print(f"Episode {episode+1}: Avg Reward = {avg_reward:.2f}, "
                  f"Policy Loss = {policy_loss:.4f}, Baseline = {agent.baseline:.2f}")
            
            # Early stopping
            if avg_reward >= target_reward:
                print(f"\nTarget reward {target_reward} reached! Stopping training.")
                break
    
    return episode_stats

# Create environment and agent
env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

print(f"Environment: CartPole-v1")
print(f"State dimension: {state_dim}")
print(f"Action dimension: {action_dim}")

# Train REINFORCE with baseline
print("\nTraining REINFORCE with baseline...")
reinforce_agent = REINFORCEAgent(state_dim, action_dim, lr=0.01, use_baseline=True)
reinforce_stats = train_reinforce(reinforce_agent, env, n_episodes=800)

print(f"\nREINFORCE training completed!")
final_rewards = [s['reward'] for s in reinforce_stats[-100:]]
print(f"Final average reward: {np.mean(final_rewards):.2f}")

# Create comparison visualization
print("\n5. Creating Visualizations")
print("-" * 30)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Learning curve
rewards = [s['reward'] for s in reinforce_stats]
window_size = 50
if len(rewards) >= window_size:
    smoothed = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
    ax1.plot(smoothed, linewidth=2)
else:
    ax1.plot(rewards, linewidth=2)

ax1.axhline(y=475, color='red', linestyle='--', alpha=0.7, label='Target')
ax1.set_xlabel('Episode')
ax1.set_ylabel('Episode Reward')
ax1.set_title('REINFORCE Learning Curve')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Policy loss
losses = [s['policy_loss'] for s in reinforce_stats]
ax2.plot(losses, alpha=0.7)
ax2.set_xlabel('Episode')
ax2.set_ylabel('Policy Loss')
ax2.set_title('Policy Loss During Training')
ax2.grid(True, alpha=0.3)

# Plot 3: Baseline evolution
baselines = [s['baseline'] for s in reinforce_stats]
ax3.plot(baselines, linewidth=2)
ax3.set_xlabel('Episode')
ax3.set_ylabel('Baseline Value')
ax3.set_title('Baseline Evolution')
ax3.grid(True, alpha=0.3)

# Plot 4: Episode length
episode_lengths = [s['steps'] for s in reinforce_stats]
ax4.plot(episode_lengths, alpha=0.7)
ax4.set_xlabel('Episode')
ax4.set_ylabel('Episode Length')
ax4.set_title('Episode Length Over Time')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('pytorch_rl_tutorial/chapter_11_policy_gradients_results.png', dpi=150, bbox_inches='tight')
print("Results saved to chapter_11_policy_gradients_results.png")

print("\nChapter 11 Complete! ✓")
print("Policy gradient methods successfully implemented")
print("Ready to explore Actor-Critic methods (Chapter 12)")

env.close()