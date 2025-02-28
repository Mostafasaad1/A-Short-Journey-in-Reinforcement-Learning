#!/usr/bin/env python3
"""
Chapter 18: Advanced Exploration

This chapter demonstrates advanced exploration techniques in RL including
Intrinsic Curiosity Module (ICM), Noisy Networks, and Bootstrapped DQN.
These methods help agents explore efficiently in sparse reward environments.

Key concepts covered:
- Intrinsic Curiosity Module with forward/inverse dynamics
- Noisy Networks with factorized Gaussian noise
- Bootstrapped DQN for uncertainty-driven exploration
- Count-based exploration methods
- Exploration bonus design and tuning


"""

# Chapter 18: Advanced Exploration
# Start by installing the required packages:
# !pip install torch gymnasium[atari] gymnasium[accept-rom-license] ale-py opencv-python matplotlib tqdm -q

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import numpy as np
import matplotlib.pyplot as plt
from collections import deque, defaultdict
import random
import cv2
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import math

# Configure matplotlib for non-interactive mode
plt.switch_backend("Agg")
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "PingFang SC", "Arial Unicode MS", "Hiragino Sans GB"]
plt.rcParams["axes.unicode_minus"] = False

# Sparse Reward Environment for Exploration
class SparseGridWorld:
    """Grid world with very sparse rewards to test exploration."""
    
    def __init__(self, size=10, num_goals=1, max_steps=200):
        self.size = size
        self.num_goals = num_goals
        self.max_steps = max_steps
        self.action_space = 4  # up, down, left, right
        
        # Place goals randomly
        self.goals = set()
        while len(self.goals) < num_goals:
            goal = (np.random.randint(size), np.random.randint(size))
            if goal != (0, 0):  # Don't place goal at start
                self.goals.add(goal)
        
        self.reset()
    
    def reset(self):
        """Reset to start position."""
        self.pos = [0, 0]
        self.step_count = 0
        self.visited = set()
        self.visited.add(tuple(self.pos))
        return self._get_state()
    
    def _get_state(self):
        """Get state representation."""
        # Create 2D observation
        state = np.zeros((self.size, self.size), dtype=np.float32)
        state[self.pos[0], self.pos[1]] = 1.0  # Agent position
        
        # Add goals
        for goal in self.goals:
            state[goal] = 0.5
        
        return state.flatten()
    
    def step(self, action):
        """Take action and return next state, reward, done."""
        old_pos = tuple(self.pos)
        
        # Execute action
        if action == 0 and self.pos[0] > 0:  # up
            self.pos[0] -= 1
        elif action == 1 and self.pos[0] < self.size - 1:  # down
            self.pos[0] += 1
        elif action == 2 and self.pos[1] > 0:  # left
            self.pos[1] -= 1
        elif action == 3 and self.pos[1] < self.size - 1:  # right
            self.pos[1] += 1
        
        new_pos = tuple(self.pos)
        self.visited.add(new_pos)
        self.step_count += 1
        
        # Calculate reward
        reward = 0.0
        
        # Goal reward
        if new_pos in self.goals:
            reward = 10.0
            done = True
        else:
            done = self.step_count >= self.max_steps
            reward = -0.01  # Small time penalty
        
        return self._get_state(), reward, done, {
            "pos": new_pos,
            "visited_count": len(self.visited),
            "goal_reached": new_pos in self.goals
        }
    
    def get_visitation_count(self, pos):
        """Get number of times position was visited."""
        return 1 if pos in self.visited else 0


# Intrinsic Curiosity Module (ICM)
class ICMNetwork(nn.Module):
    """Intrinsic Curiosity Module with forward and inverse dynamics."""
    
    def __init__(self, state_dim, action_dim, feature_dim=64, hidden_dim=128):
        super(ICMNetwork, self).__init__()
        
        # Feature extraction
        self.feature_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
            nn.ReLU()
        )
        
        # Inverse dynamics model: predict action from state transitions
        self.inverse_model = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Forward dynamics model: predict next state from current state and action
        self.forward_model = nn.Sequential(
            nn.Linear(feature_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        )
        
    def forward(self, state, next_state, action):
        """Compute ICM losses and intrinsic reward."""
        # Extract features
        state_feat = self.feature_net(state)
        next_state_feat = self.feature_net(next_state)
        
        # Inverse dynamics loss
        concat_feat = torch.cat([state_feat, next_state_feat], dim=-1)
        predicted_action = self.inverse_model(concat_feat)
        
        # Forward dynamics loss
        action_onehot = F.one_hot(action, num_classes=4).float()
        forward_input = torch.cat([state_feat, action_onehot], dim=-1)
        predicted_next_feat = self.forward_model(forward_input)
        
        # Compute losses
        inverse_loss = F.cross_entropy(predicted_action, action)
        forward_loss = F.mse_loss(predicted_next_feat, next_state_feat.detach())
        
        # Intrinsic reward is forward model error
        intrinsic_reward = forward_loss.detach()
        
        return inverse_loss, forward_loss, intrinsic_reward


# Noisy Networks
class NoisyLinear(nn.Module):
    """Noisy linear layer with factorized Gaussian noise."""
    
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        
        # Weight parameters
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.Tensor(out_features, in_features))
        
        # Bias parameters
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        self.register_buffer('bias_epsilon', torch.Tensor(out_features))
        
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        """Initialize parameters."""
        mu_range = 1 / math.sqrt(self.in_features)
        
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))
    
    def reset_noise(self):
        """Reset noise for factorized Gaussian noise."""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def _scale_noise(self, size):
        """Scale noise using sign(x)sqrt(|x|)."""
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())
    
    def forward(self, input):
        """Forward pass with noisy weights."""
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        
        return F.linear(input, weight, bias)


class NoisyDQN(nn.Module):
    """DQN with Noisy Networks for exploration."""
    
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(NoisyDQN, self).__init__()
        
        self.feature_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Noisy layers for exploration
        self.noisy1 = NoisyLinear(hidden_dim, hidden_dim)
        self.noisy2 = NoisyLinear(hidden_dim, action_dim)
        
    def forward(self, state):
        """Forward pass through noisy network."""
        features = self.feature_net(state)
        x = F.relu(self.noisy1(features))
        q_values = self.noisy2(x)
        return q_values
    
    def reset_noise(self):
        """Reset noise in all noisy layers."""
        self.noisy1.reset_noise()
        self.noisy2.reset_noise()


# Bootstrapped DQN
class BootstrappedDQN(nn.Module):
    """Bootstrapped DQN with multiple Q-heads for uncertainty estimation."""
    
    def __init__(self, state_dim, action_dim, num_heads=5, hidden_dim=128):
        super(BootstrappedDQN, self).__init__()
        
        self.num_heads = num_heads
        
        # Shared feature extractor
        self.shared_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Multiple Q-heads
        self.q_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim)
            ) for _ in range(num_heads)
        ])
        
    def forward(self, state, head_idx=None):
        """Forward pass through shared features and selected head(s)."""
        shared_features = self.shared_net(state)
        
        if head_idx is not None:
            # Use specific head
            return self.q_heads[head_idx](shared_features)
        else:
            # Return all heads
            q_values = []
            for head in self.q_heads:
                q_values.append(head(shared_features))
            return torch.stack(q_values, dim=1)  # [batch, num_heads, actions]
    
    def get_uncertainty(self, state):
        """Get uncertainty estimate from ensemble variance."""
        with torch.no_grad():
            all_q_values = self.forward(state)  # [batch, num_heads, actions]
            uncertainty = torch.var(all_q_values, dim=1)  # [batch, actions]
            return uncertainty.mean(dim=-1)  # [batch]


# Count-based exploration
class CountBasedExplorer:
    """Count-based exploration using visitation counts."""
    
    def __init__(self, state_dim, count_bonus_coeff=0.1):
        self.state_counts = defaultdict(int)
        self.count_bonus_coeff = count_bonus_coeff
        
    def get_count_bonus(self, state):
        """Get exploration bonus based on visitation count."""
        # Convert state to hashable representation
        state_key = tuple(state.flatten().round(2).tolist())  # Round for discretization
        
        self.state_counts[state_key] += 1
        count = self.state_counts[state_key]
        
        # Exploration bonus inversely proportional to sqrt(count)
        bonus = self.count_bonus_coeff / math.sqrt(count)
        return bonus
    
    def get_visitation_stats(self):
        """Get statistics about state visitations."""
        if not self.state_counts:
            return {"unique_states": 0, "avg_visits": 0, "max_visits": 0}
        
        visits = list(self.state_counts.values())
        return {
            "unique_states": len(self.state_counts),
            "avg_visits": np.mean(visits),
            "max_visits": max(visits)
        }


class ExplorationAgent:
    """Agent combining multiple exploration techniques."""
    
    def __init__(self, env, exploration_method="icm", lr=3e-4):
        self.env = env
        self.exploration_method = exploration_method
        self.state_dim = len(env.reset())
        self.action_dim = env.action_space

        # Define a factory function for Q-network
        def make_q_network():
            if exploration_method == "icm" or exploration_method == "count":
                return nn.Sequential(
                    nn.Linear(self.state_dim, 128),
                    nn.ReLU(),
                    nn.Linear(128, 128),
                    nn.ReLU(),
                    nn.Linear(128, self.action_dim)
                )
            elif exploration_method == "noisy":
                return NoisyDQN(self.state_dim, self.action_dim)
            elif exploration_method == "bootstrapped":
                return BootstrappedDQN(self.state_dim, self.action_dim, num_heads=5)
        
        # Create networks
        self.q_network = make_q_network()
        self.target_network = None if exploration_method == "bootstrapped" else make_q_network()

        if self.target_network:
            self.target_network.load_state_dict(self.q_network.state_dict())

        # ICM specific
        if exploration_method == "icm":
            self.icm = ICMNetwork(self.state_dim, self.action_dim)
            self.icm_optimizer = optim.Adam(self.icm.parameters(), lr=lr)

        # Count-based exploration
        if exploration_method == "count":
            self.count_explorer = CountBasedExplorer(self.state_dim)

        # Common components
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.memory = deque(maxlen=10000)
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

        # Statistics
        self.episode_rewards = []
        self.intrinsic_rewards = []
        self.exploration_bonuses = []

        
    def select_action(self, state, training=True):
        """Select action using exploration strategy."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        if self.exploration_method == "icm":
            # Use epsilon-greedy with ICM
            if training and np.random.random() < self.epsilon:
                return np.random.randint(self.action_dim)
            else:
                with torch.no_grad():
                    q_values = self.q_network(state_tensor)
                    return q_values.argmax().item()
        
        elif self.exploration_method == "noisy":
            # Noisy networks provide exploration through noise
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
                return q_values.argmax().item()
        
        elif self.exploration_method == "bootstrapped":
            # Use Thompson sampling with bootstrapped Q-values
            if training:
                # Sample random head
                self.current_head = np.random.randint(self.q_network.num_heads)
            
            with torch.no_grad():
                q_values = self.q_network(state_tensor, head_idx=self.current_head)
                return q_values.argmax().item()
        
        elif self.exploration_method == "count":
            # Epsilon-greedy with count-based exploration
            if training and np.random.random() < self.epsilon:
                return np.random.randint(self.action_dim)
            else:
                with torch.no_grad():
                    q_values = self.q_network(state_tensor)
                    return q_values.argmax().item()
    
    def train_step(self, batch_size=32):
        """Perform one training step."""
        if len(self.memory) < batch_size:
            return {"loss": 0, "intrinsic_reward": 0}
        
        # Sample batch
        batch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor([e[0] for e in batch])
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor([e[3] for e in batch])
        dones = torch.BoolTensor([e[4] for e in batch])
        
        # Compute intrinsic rewards and additional losses
        intrinsic_reward = 0
        additional_loss = 0
        
        if self.exploration_method == "icm":
            # ICM forward and inverse losses
            inverse_loss, forward_loss, intrinsic_reward = self.icm(states, next_states, actions)
            additional_loss = inverse_loss + forward_loss
            
            # Add intrinsic reward to extrinsic reward
            intrinsic_reward_values = []
            for i in range(len(batch)):
                with torch.no_grad():
                    _, _, ir = self.icm(states[i:i+1], next_states[i:i+1], actions[i:i+1])
                    intrinsic_reward_values.append(ir.item())
            
            intrinsic_rewards_tensor = torch.FloatTensor(intrinsic_reward_values)
            rewards = rewards + 0.1 * intrinsic_rewards_tensor  # Scale intrinsic reward
        
        elif self.exploration_method == "count":
            # Add count-based exploration bonus
            count_bonuses = []
            for state in states:
                bonus = self.count_explorer.get_count_bonus(state.numpy())
                count_bonuses.append(bonus)
            
            count_bonuses_tensor = torch.FloatTensor(count_bonuses)
            rewards = rewards + count_bonuses_tensor
        
        # Q-learning update
        if self.exploration_method == "bootstrapped":
            # Train all heads
            total_loss = 0
            for head_idx in range(self.q_network.num_heads):
                current_q_values = self.q_network(states, head_idx=head_idx)
                current_q_values = current_q_values.gather(1, actions.unsqueeze(1))
                
                with torch.no_grad():
                    next_q_values = self.q_network(next_states, head_idx=head_idx)
                    max_next_q_values = next_q_values.max(1)[0]
                    target_q_values = rewards + (0.99 * max_next_q_values * ~dones)
                
                loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
                total_loss += loss
            
            total_loss /= self.q_network.num_heads
        else:
            # Standard DQN update
            current_q_values = self.q_network(states)
            current_q_values = current_q_values.gather(1, actions.unsqueeze(1))
            
            with torch.no_grad():
                if self.target_network:
                    next_q_values = self.target_network(next_states)
                else:
                    next_q_values = self.q_network(next_states)
                
                max_next_q_values = next_q_values.max(1)[0]
                target_q_values = rewards + (0.99 * max_next_q_values * ~dones)
            
            total_loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Add additional losses
        total_loss += additional_loss
        
        # Update networks
        self.optimizer.zero_grad()
        if self.exploration_method == "icm":
            self.icm_optimizer.zero_grad()
        
        total_loss.backward()
        
        self.optimizer.step()
        if self.exploration_method == "icm":
            self.icm_optimizer.step()
        
        # Reset noise for noisy networks
        if self.exploration_method == "noisy" and hasattr(self.q_network, 'reset_noise'):
            self.q_network.reset_noise()
        
        return {
            "loss": total_loss.item(),
            "intrinsic_reward": intrinsic_reward if isinstance(intrinsic_reward, (int, float)) else intrinsic_reward.item() if hasattr(intrinsic_reward, 'item') else 0
        }
    
    def train_episode(self):
        """Train for one episode."""
        state = self.env.reset()
        total_reward = 0
        total_intrinsic_reward = 0
        steps = 0
        
        while True:
            action = self.select_action(state, training=True)
            next_state, reward, done, info = self.env.step(action)
            
            # Store experience
            self.memory.append((state, action, reward, next_state, done))
            
            # Training step
            train_result = self.train_step()
            
            total_reward += reward
            total_intrinsic_reward += train_result["intrinsic_reward"]
            steps += 1
            
            state = next_state
            
            if done:
                break
        
        # Update epsilon for epsilon-greedy methods
        if self.exploration_method in ["icm", "count"]:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        # Update target network
        if self.target_network and len(self.episode_rewards) % 100 == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.episode_rewards.append(total_reward)
        self.intrinsic_rewards.append(total_intrinsic_reward)
        
        return {
            "episode_reward": total_reward,
            "intrinsic_reward": total_intrinsic_reward,
            "steps": steps,
            "goal_reached": info.get("goal_reached", False),
            "visited_states": info.get("visited_count", 0)
        }


def compare_exploration_methods(num_episodes=1000):
    """Compare different exploration methods."""
    print("=== Chapter 18: Advanced Exploration ===")
    print("Comparing exploration methods in sparse reward environment...\n")
    
    # Create environment
    env = SparseGridWorld(size=8, num_goals=1, max_steps=100)
    print(f"Environment: {env.size}x{env.size} grid world")
    print(f"Goals: {env.goals}")
    print(f"Max steps per episode: {env.max_steps}\n")
    
    # Test different exploration methods
    methods = ["icm", "noisy", "bootstrapped", "count"]
    results = {}
    
    for method in methods:
        print(f"Training with {method.upper()} exploration...")
        
        # Create agent
        agent = ExplorationAgent(env, exploration_method=method, lr=1e-3)
        
        # Training statistics
        success_rates = []
        avg_rewards = []
        exploration_efficiency = []
        
        for episode in range(num_episodes):
            result = agent.train_episode()
            
            # Log progress
            if (episode + 1) % 100 == 0:
                recent_rewards = agent.episode_rewards[-100:]
                recent_successes = sum(1 for i in range(max(0, len(agent.episode_rewards) - 100), len(agent.episode_rewards)) 
                                     if i < len(agent.episode_rewards))
                
                # Test success rate
                test_successes = 0
                for _ in range(20):
                    test_state = env.reset()
                    test_steps = 0
                    while test_steps < env.max_steps:
                        test_action = agent.select_action(test_state, training=False)
                        test_state, _, test_done, test_info = env.step(test_action)
                        test_steps += 1
                        if test_done and test_info.get("goal_reached", False):
                            test_successes += 1
                            break
                        if test_done:
                            break
                
                success_rate = test_successes / 20
                avg_reward = np.mean(recent_rewards)
                
                success_rates.append(success_rate)
                avg_rewards.append(avg_reward)
                
                # Calculate exploration efficiency
                if method == "count" and hasattr(agent, 'count_explorer'):
                    stats = agent.count_explorer.get_visitation_stats()
                    efficiency = stats["unique_states"] / (episode + 1)
                    exploration_efficiency.append(efficiency)
                else:
                    exploration_efficiency.append(0)
                
                print(f"  Episode {episode+1:4d} | Success Rate: {success_rate:5.1%} | "
                      f"Avg Reward: {avg_reward:6.2f}")
        
        results[method] = {
            "success_rates": success_rates,
            "avg_rewards": avg_rewards,
            "exploration_efficiency": exploration_efficiency,
            "final_success_rate": success_rates[-1] if success_rates else 0,
            "agent": agent
        }
        
        print(f"  Final success rate: {results[method]['final_success_rate']:.1%}\n")
    
    # Create comparison plots
    create_exploration_plots(results)
    
    # Analyze exploration behavior
    analyze_exploration_behavior(results, env)
    
    return results


def create_exploration_plots(results):
    """Create comparison plots for exploration methods."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Advanced Exploration Methods Comparison', fontsize=16)
    
    methods = list(results.keys())
    colors = ['blue', 'red', 'green', 'orange']
    
    # Success rates
    for i, method in enumerate(methods):
        episodes = np.arange(100, len(results[method]['success_rates']) * 100 + 1, 100)
        axes[0, 0].plot(episodes, results[method]['success_rates'], 
                       color=colors[i], linewidth=2, label=method.upper())
    
    axes[0, 0].set_title('Success Rate Over Training')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Success Rate')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim(0, 1)
    
    # Average rewards
    for i, method in enumerate(methods):
        episodes = np.arange(100, len(results[method]['avg_rewards']) * 100 + 1, 100)
        axes[0, 1].plot(episodes, results[method]['avg_rewards'], 
                       color=colors[i], linewidth=2, label=method.upper())
    
    axes[0, 1].set_title('Average Reward Over Training')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Average Reward')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Final performance comparison
    final_success = [results[method]['final_success_rate'] for method in methods]
    bars = axes[1, 0].bar(methods, final_success, color=colors)
    axes[1, 0].set_title('Final Success Rate Comparison')
    axes[1, 0].set_ylabel('Success Rate')
    axes[1, 0].set_ylim(0, 1)
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.1%}', ha='center', va='bottom')
    
    # Learning curves (smoothed)
    window = 3
    for i, method in enumerate(methods):
        success_data = results[method]['success_rates']
        if len(success_data) >= window:
            smoothed = np.convolve(success_data, np.ones(window)/window, mode='valid')
            episodes = np.arange(window * 100, (len(smoothed) + window) * 100, 100)
            axes[1, 1].plot(episodes, smoothed, color=colors[i], linewidth=3, 
                           label=f'{method.upper()} (smoothed)')
    
    axes[1, 1].set_title('Smoothed Learning Curves')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Success Rate')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('pytorch_rl_tutorial/advanced_exploration_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\nComparison plots saved as 'advanced_exploration_comparison.png'")


def analyze_exploration_behavior(results, env):
    """Analyze exploration behavior of different methods."""
    print("\n=== Exploration Behavior Analysis ===")
    
    for method, result in results.items():
        print(f"\n{method.upper()} Exploration Analysis:")
        agent = result["agent"]
        
        # Test exploration coverage
        state = env.reset()
        visited_positions = set()
        
        for _ in range(200):  # Test for 200 steps
            action = agent.select_action(state, training=False)
            state, _, done, info = env.step(action)
            visited_positions.add(info["pos"])
            
            if done:
                state = env.reset()
        
        coverage = len(visited_positions) / (env.size * env.size)
        print(f"  State space coverage: {coverage:.1%} ({len(visited_positions)}/{env.size * env.size} states)")
        
        # Method-specific analysis
        if method == "count" and hasattr(agent, 'count_explorer'):
            stats = agent.count_explorer.get_visitation_stats()
            print(f"  Unique states visited during training: {stats['unique_states']}")
            print(f"  Average visits per state: {stats['avg_visits']:.2f}")
            print(f"  Max visits to single state: {stats['max_visits']}")
        
        elif method == "bootstrapped":
            # Test uncertainty-based exploration
            print(f"  Number of Q-heads: {agent.q_network.num_heads}")
            
            # Sample some states and check uncertainty
            uncertainties = []
            for _ in range(50):
                test_state = env.reset()
                state_tensor = torch.FloatTensor(test_state).unsqueeze(0)
                uncertainty = agent.q_network.get_uncertainty(state_tensor)
                uncertainties.append(uncertainty.item())
            
            print(f"  Average Q-value uncertainty: {np.mean(uncertainties):.4f}")
            print(f"  Uncertainty std: {np.std(uncertainties):.4f}")
        
        elif method == "icm":
            print(f"  ICM feature dimension: {agent.icm.feature_net[2].out_features}")
            print(f"  Average intrinsic reward: {np.mean(agent.intrinsic_rewards):.4f}")
        
        elif method == "noisy":
            print(f"  Noisy network layers: {len([m for m in agent.q_network.modules() if isinstance(m, NoisyLinear)])}")
        
        print(f"  Final success rate: {result['final_success_rate']:.1%}")


if __name__ == "__main__":
    print("Chapter 18: Advanced Exploration")
    print("="*50)
    print("This chapter demonstrates advanced exploration techniques for RL.")
    print("We'll compare ICM, Noisy Networks, Bootstrapped DQN, and Count-based methods.\n")
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Compare exploration methods
    results = compare_exploration_methods(num_episodes=800)
    
    print("\n" + "="*50)
    print("Key Concepts Demonstrated:")
    print("- Intrinsic Curiosity Module (ICM) with forward/inverse dynamics")
    print("- Noisy Networks with factorized Gaussian noise")
    print("- Bootstrapped DQN for uncertainty-driven exploration")
    print("- Count-based exploration bonuses")
    print("- Comparison of exploration efficiency in sparse reward settings")
    print("\nAdvanced exploration is crucial for:")
    print("- Environments with sparse or delayed rewards")
    print("- Large state spaces requiring systematic exploration")
    print("- Tasks where random exploration is insufficient")
    print("- Sample-efficient learning in complex domains")
    print("\nNext: Chapter 19 - Reinforcement Learning with Human Feedback")
