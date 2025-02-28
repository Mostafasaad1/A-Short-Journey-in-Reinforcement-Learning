#!/usr/bin/env python3
"""
Chapter 22: Multi-Agent RL

This chapter demonstrates multi-agent reinforcement learning using Independent DQN (IDQN)
and Multi-Agent Deep Deterministic Policy Gradient (MADDPG). We'll implement cooperative
and competitive scenarios to show the challenges of non-stationarity and credit assignment.

Key concepts covered:
- Independent DQN for decentralized learning
- MADDPG with centralized training, decentralized execution
- Cooperative and competitive multi-agent scenarios
- Experience replay in multi-agent settings
- Communication and coordination mechanisms


"""

# Chapter 22: Multi-Agent RL
# Start by installing the required packages:
# !pip install torch pettingzoo[all] matplotlib numpy -q

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import numpy as np
import matplotlib.pyplot as plt
from collections import deque, namedtuple
import random
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import itertools

# Configure matplotlib for non-interactive mode
plt.switch_backend("Agg")
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "PingFang SC", "Arial Unicode MS", "Hiragino Sans GB"]
plt.rcParams["axes.unicode_minus"] = False

# Multi-Agent Environment
class SimpleSpreadEnvironment:
    """Simplified multi-agent environment similar to PettingZoo's simple_spread.
    
    Agents must spread out to cover all landmarks while avoiding collisions.
    This is a cooperative task requiring coordination.
    """
    
    def __init__(self, num_agents=3, num_landmarks=3, world_size=2.0, agent_size=0.05, landmark_size=0.1):
        self.num_agents = num_agents
        self.num_landmarks = num_landmarks
        self.world_size = world_size
        self.agent_size = agent_size
        self.landmark_size = landmark_size
        
        # Action space: [no_action, move_left, move_right, move_up, move_down]
        self.action_space = 5
        self.action_map = {
            0: np.array([0.0, 0.0]),    # no action
            1: np.array([-0.1, 0.0]),   # left
            2: np.array([0.1, 0.0]),    # right
            3: np.array([0.0, 0.1]),    # up
            4: np.array([0.0, -0.1])    # down
        }
        
        self.reset()
    
    def reset(self):
        """Reset environment to initial state."""
        # Initialize agent positions randomly
        self.agent_positions = np.random.uniform(
            -self.world_size/2, self.world_size/2, 
            size=(self.num_agents, 2)
        )
        
        # Initialize landmark positions randomly
        self.landmark_positions = np.random.uniform(
            -self.world_size/2, self.world_size/2,
            size=(self.num_landmarks, 2)
        )
        
        # Track which agent is closest to each landmark
        self.landmark_assignments = [-1] * self.num_landmarks
        
        self.step_count = 0
        self.max_steps = 100
        
        return self._get_observations()
    
    def _get_observations(self):
        """Get observations for each agent."""
        observations = []
        
        for i in range(self.num_agents):
            obs = []
            
            # Agent's own position
            obs.extend(self.agent_positions[i])
            
            # Relative positions of other agents
            for j in range(self.num_agents):
                if i != j:
                    relative_pos = self.agent_positions[j] - self.agent_positions[i]
                    obs.extend(relative_pos)
            
            # Relative positions of landmarks
            for landmark_pos in self.landmark_positions:
                relative_pos = landmark_pos - self.agent_positions[i]
                obs.extend(relative_pos)
            
            observations.append(np.array(obs, dtype=np.float32))
        
        return observations
    
    def step(self, actions):
        """Take a step in the environment."""
        actions = np.array(actions)
        
        # Apply actions to update agent positions
        for i, action in enumerate(actions):
            if action < len(self.action_map):
                self.agent_positions[i] += self.action_map[action]
        
        # Keep agents within world bounds
        self.agent_positions = np.clip(
            self.agent_positions, 
            -self.world_size/2 + self.agent_size,
            self.world_size/2 - self.agent_size
        )
        
        # Calculate rewards
        rewards = self._calculate_rewards()
        
        self.step_count += 1
        done = self.step_count >= self.max_steps
        
        # Additional info
        info = {
            'collisions': self._count_collisions(),
            'coverage': self._calculate_coverage(),
            'avg_distance_to_landmarks': self._avg_distance_to_landmarks()
        }
        
        return self._get_observations(), rewards, done, info
    
    def _calculate_rewards(self):
        """Calculate rewards for cooperative task."""
        rewards = [0.0] * self.num_agents
        
        # Find closest agent to each landmark
        for landmark_idx, landmark_pos in enumerate(self.landmark_positions):
            distances = [np.linalg.norm(agent_pos - landmark_pos) 
                        for agent_pos in self.agent_positions]
            closest_agent = np.argmin(distances)
            min_distance = distances[closest_agent]
            
            # Reward based on distance (closer is better)
            landmark_reward = max(0, 1.0 - min_distance)
            
            # Give reward to the closest agent
            rewards[closest_agent] += landmark_reward
        
        # Penalty for collisions
        collision_penalty = -0.1 * self._count_collisions()
        for i in range(self.num_agents):
            rewards[i] += collision_penalty
        
        # Small penalty for time
        for i in range(self.num_agents):
            rewards[i] -= 0.01
        
        return rewards
    
    def _count_collisions(self):
        """Count number of agent collisions."""
        collisions = 0
        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                distance = np.linalg.norm(self.agent_positions[i] - self.agent_positions[j])
                if distance < 2 * self.agent_size:
                    collisions += 1
        return collisions
    
    def _calculate_coverage(self):
        """Calculate how well landmarks are covered."""
        coverage_threshold = 0.2
        covered_landmarks = 0
        
        for landmark_pos in self.landmark_positions:
            min_distance = min(np.linalg.norm(agent_pos - landmark_pos) 
                             for agent_pos in self.agent_positions)
            if min_distance < coverage_threshold:
                covered_landmarks += 1
        
        return covered_landmarks / self.num_landmarks
    
    def _avg_distance_to_landmarks(self):
        """Calculate average distance from agents to nearest landmarks."""
        total_distance = 0
        
        for landmark_pos in self.landmark_positions:
            min_distance = min(np.linalg.norm(agent_pos - landmark_pos) 
                             for agent_pos in self.agent_positions)
            total_distance += min_distance
        
        return total_distance / self.num_landmarks
    
    def render(self, title="Multi-Agent Environment"):
        """Visualize the environment."""
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Draw world boundaries
        world_bound = self.world_size / 2
        ax.set_xlim(-world_bound, world_bound)
        ax.set_ylim(-world_bound, world_bound)
        
        # Draw landmarks
        for i, pos in enumerate(self.landmark_positions):
            circle = plt.Circle(pos, self.landmark_size, color='red', alpha=0.7, label='Landmarks' if i == 0 else "")
            ax.add_patch(circle)
            ax.text(pos[0], pos[1], str(i), ha='center', va='center', fontweight='bold')
        
        # Draw agents
        colors = ['blue', 'green', 'orange', 'purple', 'brown']
        for i, pos in enumerate(self.agent_positions):
            color = colors[i % len(colors)]
            circle = plt.Circle(pos, self.agent_size, color=color, alpha=0.8, 
                              label=f'Agent {i}' if i < 3 else "")
            ax.add_patch(circle)
        
        ax.set_title(title)
        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Y coordinate')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        return fig


# Independent DQN Agent
class DQNAgent:
    """Independent DQN agent for multi-agent RL."""
    
    def __init__(self, obs_dim, action_dim, hidden_dim=64, lr=1e-3, epsilon=1.0):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.epsilon = epsilon
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
        # Q-network
        self.q_network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Target network
        self.target_network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.memory = deque(maxlen=10000)
        
    def select_action(self, obs, training=True):
        """Select action using epsilon-greedy policy."""
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            q_values = self.q_network(obs_tensor)
            return q_values.argmax().item()
    
    def store_experience(self, obs, action, reward, next_obs, done):
        """Store experience in replay buffer."""
        self.memory.append((obs, action, reward, next_obs, done))
    
    def train(self, batch_size=32):
        """Train the Q-network."""
        if len(self.memory) < batch_size:
            return 0.0
        
        # Sample batch
        batch = random.sample(self.memory, batch_size)
        obs_batch = torch.FloatTensor([e[0] for e in batch])
        action_batch = torch.LongTensor([e[1] for e in batch])
        reward_batch = torch.FloatTensor([e[2] for e in batch])
        next_obs_batch = torch.FloatTensor([e[3] for e in batch])
        done_batch = torch.BoolTensor([e[4] for e in batch])
        
        # Current Q-values
        current_q_values = self.q_network(obs_batch).gather(1, action_batch.unsqueeze(1))
        
        # Target Q-values
        with torch.no_grad():
            next_q_values = self.target_network(next_obs_batch).max(1)[0]
            target_q_values = reward_batch + (0.99 * next_q_values * ~done_batch)
        
        # Loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()
    
    def update_target_network(self):
        """Update target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())


# MADDPG Components
class Actor(nn.Module):
    """Actor network for MADDPG."""
    
    def __init__(self, obs_dim, action_dim, hidden_dim=64):
        super(Actor, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, obs):
        return self.network(obs)


class Critic(nn.Module):
    """Centralized critic for MADDPG."""
    
    def __init__(self, total_obs_dim, total_action_dim, hidden_dim=128):
        super(Critic, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(total_obs_dim + total_action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, obs, actions):
        input_tensor = torch.cat([obs, actions], dim=-1)
        return self.network(input_tensor)


class MADDPGAgent:
    """MADDPG agent with centralized critic."""
    
    def __init__(self, agent_id, obs_dim, action_dim, total_obs_dim, total_action_dim, lr=1e-3):
        self.agent_id = agent_id
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # Actor networks
        self.actor = Actor(obs_dim, action_dim)
        self.target_actor = Actor(obs_dim, action_dim)
        self.target_actor.load_state_dict(self.actor.state_dict())
        
        # Critic networks (centralized)
        self.critic = Critic(total_obs_dim, total_action_dim)
        self.target_critic = Critic(total_obs_dim, total_action_dim)
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        # Exploration noise
        self.exploration_noise = 0.1
        self.noise_decay = 0.999
        
    def select_action(self, obs, training=True):
        """Select action using actor network."""
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            action_probs = self.actor(obs_tensor).squeeze()
            
            if training:
                # Add exploration noise
                noise = torch.randn_like(action_probs) * self.exploration_noise
                action_probs = F.softmax(action_probs + noise, dim=-1)
                action = Categorical(action_probs).sample().item()
            else:
                action = action_probs.argmax().item()
        
        return action
    
    def update_networks(self, tau=0.01):
        """Soft update of target networks."""
        # Update target actor
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        
        # Update target critic
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        
        # Decay exploration noise
        self.exploration_noise = max(0.01, self.exploration_noise * self.noise_decay)


class MADDPG:
    """Multi-Agent DDPG implementation."""
    
    def __init__(self, num_agents, obs_dims, action_dims, lr=1e-3):
        self.num_agents = num_agents
        self.obs_dims = obs_dims
        self.action_dims = action_dims
        
        total_obs_dim = sum(obs_dims)
        total_action_dim = sum(action_dims)
        
        # Create agents
        self.agents = []
        for i in range(num_agents):
            agent = MADDPGAgent(
                agent_id=i,
                obs_dim=obs_dims[i],
                action_dim=action_dims[i],
                total_obs_dim=total_obs_dim,
                total_action_dim=total_action_dim,
                lr=lr
            )
            self.agents.append(agent)
        
        # Shared replay buffer
        self.memory = deque(maxlen=50000)
        
    def select_actions(self, observations, training=True):
        """Select actions for all agents."""
        actions = []
        for i, agent in enumerate(self.agents):
            action = agent.select_action(observations[i], training)
            actions.append(action)
        return actions
    
    def store_experience(self, observations, actions, rewards, next_observations, done):
        """Store experience in shared replay buffer."""
        self.memory.append((observations, actions, rewards, next_observations, done))
    
    def train(self, batch_size=64):
        """Train all agents using MADDPG."""
        if len(self.memory) < batch_size:
            return [0.0] * self.num_agents
        
        # Sample batch
        batch = random.sample(self.memory, batch_size)
        
        obs_batch = []
        action_batch = []
        reward_batch = []
        next_obs_batch = []
        done_batch = []
        
        for experience in batch:
            obs, actions, rewards, next_obs, done = experience
            obs_batch.append(obs)
            action_batch.append(actions)
            reward_batch.append(rewards)
            next_obs_batch.append(next_obs)
            done_batch.append(done)
        
        # Convert to tensors
        obs_batch = torch.FloatTensor(obs_batch)  # [batch_size, num_agents, obs_dim]
        action_batch = torch.LongTensor(action_batch)  # [batch_size, num_agents]
        reward_batch = torch.FloatTensor(reward_batch)  # [batch_size, num_agents]
        next_obs_batch = torch.FloatTensor(next_obs_batch)  # [batch_size, num_agents, obs_dim]
        done_batch = torch.BoolTensor(done_batch)  # [batch_size]
        
        losses = []
        
        # Train each agent
        for agent_idx, agent in enumerate(self.agents):
            # Critic update
            agent.critic_optimizer.zero_grad()
            
            # Current Q-values
            current_obs = obs_batch.view(batch_size, -1)  # Flatten observations
            
            # Convert actions to one-hot for critic input
            current_actions_onehot = []
            for i in range(self.num_agents):
                actions_onehot = F.one_hot(action_batch[:, i], num_classes=self.action_dims[i]).float()
                current_actions_onehot.append(actions_onehot)
            current_actions = torch.cat(current_actions_onehot, dim=1)
            
            current_q_values = agent.critic(current_obs, current_actions)
            
            # Target Q-values
            with torch.no_grad():
                next_obs_flat = next_obs_batch.view(batch_size, -1)
                
                # Get target actions from all agents
                target_actions_onehot = []
                for i, target_agent in enumerate(self.agents):
                    target_action_probs = target_agent.target_actor(next_obs_batch[:, i])
                    target_actions_onehot.append(target_action_probs)
                target_actions = torch.cat(target_actions_onehot, dim=1)
                
                target_q_values = agent.target_critic(next_obs_flat, target_actions)
                target_q_values = reward_batch[:, agent_idx] + (0.99 * target_q_values.squeeze() * ~done_batch)
            
            critic_loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
            critic_loss.backward()
            agent.critic_optimizer.step()
            
            # Actor update
            agent.actor_optimizer.zero_grad()
            
            # Get current policy actions
            policy_actions_onehot = []
            for i, policy_agent in enumerate(self.agents):
                if i == agent_idx:
                    policy_action_probs = agent.actor(obs_batch[:, i])
                else:
                    with torch.no_grad():
                        policy_action_probs = policy_agent.actor(obs_batch[:, i])
                policy_actions_onehot.append(policy_action_probs)
            policy_actions = torch.cat(policy_actions_onehot, dim=1)
            
            actor_loss = -agent.critic(current_obs, policy_actions).mean()
            actor_loss.backward()
            agent.actor_optimizer.step()
            
            # Update target networks
            agent.update_networks()
            
            total_loss = critic_loss.item() + actor_loss.item()
            losses.append(total_loss)
        
        return losses


def train_independent_dqn(num_episodes=2000):
    """Train independent DQN agents."""
    print("=== Training Independent DQN ===")
    
    # Create environment and agents
    env = SimpleSpreadEnvironment(num_agents=3, num_landmarks=3)
    
    # Get observation dimensions
    sample_obs = env.reset()
    obs_dim = len(sample_obs[0])
    action_dim = env.action_space
    
    print(f"Environment: {env.num_agents} agents, {env.num_landmarks} landmarks")
    print(f"Observation dim: {obs_dim}, Action dim: {action_dim}")
    
    # Create independent agents
    agents = [DQNAgent(obs_dim, action_dim, lr=1e-3) for _ in range(env.num_agents)]
    
    # Training statistics
    episode_rewards = []
    coverage_history = []
    collision_history = []
    
    for episode in range(num_episodes):
        observations = env.reset()
        episode_reward = [0.0] * env.num_agents
        done = False
        
        while not done:
            # Select actions
            actions = [agent.select_action(obs) for agent, obs in zip(agents, observations)]
            
            # Take step
            next_observations, rewards, done, info = env.step(actions)
            
            # Store experiences
            for i, agent in enumerate(agents):
                agent.store_experience(
                    observations[i], actions[i], rewards[i], next_observations[i], done
                )
                episode_reward[i] += rewards[i]
            
            # Train agents
            if episode > 100:  # Start training after some exploration
                for agent in agents:
                    agent.train()
            
            observations = next_observations
        
        # Update target networks
        if episode % 100 == 0:
            for agent in agents:
                agent.update_target_network()
        
        # Store statistics
        avg_reward = np.mean(episode_reward)
        episode_rewards.append(avg_reward)
        coverage_history.append(info['coverage'])
        collision_history.append(info['collisions'])
        
        # Logging
        if (episode + 1) % 200 == 0:
            recent_reward = np.mean(episode_rewards[-100:])
            recent_coverage = np.mean(coverage_history[-100:])
            recent_collisions = np.mean(collision_history[-100:])
            
            print(f"Episode {episode+1:4d} | Avg Reward: {recent_reward:6.2f} | "
                  f"Coverage: {recent_coverage:5.1%} | Collisions: {recent_collisions:4.1f}")
    
    return agents, env, {
        'rewards': episode_rewards,
        'coverage': coverage_history,
        'collisions': collision_history
    }


def train_maddpg(num_episodes=1500):
    """Train MADDPG agents."""
    print("\n=== Training MADDPG ===")
    
    # Create environment
    env = SimpleSpreadEnvironment(num_agents=3, num_landmarks=3)
    
    # Get dimensions
    sample_obs = env.reset()
    obs_dims = [len(obs) for obs in sample_obs]
    action_dims = [env.action_space] * env.num_agents
    
    print(f"Environment: {env.num_agents} agents, {env.num_landmarks} landmarks")
    print(f"Observation dims: {obs_dims}, Action dims: {action_dims}")
    
    # Create MADDPG
    maddpg = MADDPG(env.num_agents, obs_dims, action_dims, lr=1e-3)
    
    # Training statistics
    episode_rewards = []
    coverage_history = []
    collision_history = []
    
    for episode in range(num_episodes):
        observations = env.reset()
        episode_reward = [0.0] * env.num_agents
        done = False
        
        while not done:
            # Select actions
            actions = maddpg.select_actions(observations, training=True)
            
            # Take step
            next_observations, rewards, done, info = env.step(actions)
            
            # Store experience
            maddpg.store_experience(observations, actions, rewards, next_observations, done)
            
            # Update episode rewards
            for i in range(env.num_agents):
                episode_reward[i] += rewards[i]
            
            observations = next_observations
        
        # Train agents
        if episode > 100:  # Start training after some exploration
            maddpg.train()
        
        # Store statistics
        avg_reward = np.mean(episode_reward)
        episode_rewards.append(avg_reward)
        coverage_history.append(info['coverage'])
        collision_history.append(info['collisions'])
        
        # Logging
        if (episode + 1) % 200 == 0:
            recent_reward = np.mean(episode_rewards[-100:])
            recent_coverage = np.mean(coverage_history[-100:])
            recent_collisions = np.mean(collision_history[-100:])
            
            print(f"Episode {episode+1:4d} | Avg Reward: {recent_reward:6.2f} | "
                  f"Coverage: {recent_coverage:5.1%} | Collisions: {recent_collisions:4.1f}")
    
    return maddpg, env, {
        'rewards': episode_rewards,
        'coverage': coverage_history,
        'collisions': collision_history
    }


def compare_multi_agent_methods():
    """Compare Independent DQN and MADDPG."""
    print("=== Chapter 22: Multi-Agent RL ===")
    print("Comparing Independent DQN and MADDPG on cooperative task...\n")
    
    # Train both methods
    print("Training Independent DQN...")
    idqn_agents, idqn_env, idqn_stats = train_independent_dqn(num_episodes=1500)
    
    print("Training MADDPG...")
    maddpg_agents, maddpg_env, maddpg_stats = train_maddpg(num_episodes=1200)
    
    # Final evaluation
    print("\n=== Final Evaluation ===")
    
    # Test Independent DQN
    idqn_test_rewards = []
    idqn_test_coverage = []
    idqn_test_collisions = []
    
    for _ in range(50):
        observations = idqn_env.reset()
        total_reward = [0.0] * idqn_env.num_agents
        done = False
        
        while not done:
            actions = [agent.select_action(obs, training=False) 
                      for agent, obs in zip(idqn_agents, observations)]
            observations, rewards, done, info = idqn_env.step(actions)
            
            for i in range(idqn_env.num_agents):
                total_reward[i] += rewards[i]
        
        idqn_test_rewards.append(np.mean(total_reward))
        idqn_test_coverage.append(info['coverage'])
        idqn_test_collisions.append(info['collisions'])
    
    # Test MADDPG
    maddpg_test_rewards = []
    maddpg_test_coverage = []
    maddpg_test_collisions = []
    
    for _ in range(50):
        observations = maddpg_env.reset()
        total_reward = [0.0] * maddpg_env.num_agents
        done = False
        
        while not done:
            actions = maddpg_agents.select_actions(observations, training=False)
            observations, rewards, done, info = maddpg_env.step(actions)
            
            for i in range(maddpg_env.num_agents):
                total_reward[i] += rewards[i]
        
        maddpg_test_rewards.append(np.mean(total_reward))
        maddpg_test_coverage.append(info['coverage'])
        maddpg_test_collisions.append(info['collisions'])
    
    # Print results
    print(f"Independent DQN Performance:")
    print(f"  Average Reward: {np.mean(idqn_test_rewards):.3f} ± {np.std(idqn_test_rewards):.3f}")
    print(f"  Coverage Rate:  {np.mean(idqn_test_coverage):.1%} ± {np.std(idqn_test_coverage):.1%}")
    print(f"  Avg Collisions: {np.mean(idqn_test_collisions):.2f} ± {np.std(idqn_test_collisions):.2f}")
    
    print(f"\nMADDPG Performance:")
    print(f"  Average Reward: {np.mean(maddpg_test_rewards):.3f} ± {np.std(maddpg_test_rewards):.3f}")
    print(f"  Coverage Rate:  {np.mean(maddpg_test_coverage):.1%} ± {np.std(maddpg_test_coverage):.1%}")
    print(f"  Avg Collisions: {np.mean(maddpg_test_collisions):.2f} ± {np.std(maddpg_test_collisions):.2f}")
    
    # Create comparison plots
    create_multi_agent_plots(idqn_stats, maddpg_stats)
    
    # Visualize final behaviors
    visualize_agent_behaviors(idqn_agents, idqn_env, maddpg_agents, maddpg_env)
    
    return {
        'idqn': {'agents': idqn_agents, 'env': idqn_env, 'stats': idqn_stats},
        'maddpg': {'agents': maddpg_agents, 'env': maddpg_env, 'stats': maddpg_stats}
    }


def create_multi_agent_plots(idqn_stats, maddpg_stats):
    """Create comparison plots for multi-agent methods."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Multi-Agent RL Comparison: IDQN vs MADDPG', fontsize=16)
    
    # Smooth the curves
    window = 50
    
    def smooth(data, window_size):
        if len(data) < window_size:
            return data
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')
    
    # Episode rewards
    idqn_rewards_smooth = smooth(idqn_stats['rewards'], window)
    maddpg_rewards_smooth = smooth(maddpg_stats['rewards'], window)
    
    axes[0, 0].plot(idqn_rewards_smooth, 'b-', linewidth=2, label='Independent DQN')
    axes[0, 0].plot(maddpg_rewards_smooth, 'r-', linewidth=2, label='MADDPG')
    axes[0, 0].set_title('Average Episode Reward')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Average Reward')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Coverage rate
    idqn_coverage_smooth = smooth(idqn_stats['coverage'], window)
    maddpg_coverage_smooth = smooth(maddpg_stats['coverage'], window)
    
    axes[0, 1].plot(idqn_coverage_smooth, 'b-', linewidth=2, label='Independent DQN')
    axes[0, 1].plot(maddpg_coverage_smooth, 'r-', linewidth=2, label='MADDPG')
    axes[0, 1].set_title('Landmark Coverage Rate')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Coverage Rate')
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Collision rate
    idqn_collisions_smooth = smooth(idqn_stats['collisions'], window)
    maddpg_collisions_smooth = smooth(maddpg_stats['collisions'], window)
    
    axes[1, 0].plot(idqn_collisions_smooth, 'b-', linewidth=2, label='Independent DQN')
    axes[1, 0].plot(maddpg_collisions_smooth, 'r-', linewidth=2, label='MADDPG')
    axes[1, 0].set_title('Average Collisions per Episode')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Number of Collisions')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Performance comparison
    final_episodes = 100
    idqn_final_reward = np.mean(idqn_stats['rewards'][-final_episodes:])
    maddpg_final_reward = np.mean(maddpg_stats['rewards'][-final_episodes:])
    idqn_final_coverage = np.mean(idqn_stats['coverage'][-final_episodes:])
    maddpg_final_coverage = np.mean(maddpg_stats['coverage'][-final_episodes:])
    
    methods = ['Independent DQN', 'MADDPG']
    rewards = [idqn_final_reward, maddpg_final_reward]
    coverages = [idqn_final_coverage, maddpg_final_coverage]
    
    x = np.arange(len(methods))
    width = 0.35
    
    bars1 = axes[1, 1].bar(x - width/2, rewards, width, label='Average Reward', alpha=0.8)
    bars2 = axes[1, 1].bar(x + width/2, coverages, width, label='Coverage Rate', alpha=0.8)
    
    axes[1, 1].set_title('Final Performance Comparison')
    axes[1, 1].set_ylabel('Performance')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(methods)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.2f}', ha='center', va='bottom')
    
    for bar in bars2:
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.1%}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('pytorch_rl_tutorial/chapter_22_multi_agent_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\nMulti-agent comparison plots saved as 'multi_agent_comparison.png'")


def visualize_agent_behaviors(idqn_agents, idqn_env, maddpg_agents, maddpg_env):
    """Visualize final agent behaviors."""
    print("\n=== Agent Behavior Visualization ===")
    
    # Test Independent DQN
    observations = idqn_env.reset()
    
    fig1 = idqn_env.render("Independent DQN - Initial State")
    fig1.savefig('pytorch_rl_tutorial/idqn_initial.png', dpi=300, bbox_inches='tight')
    plt.close(fig1)
    
    # Run for a few steps
    for _ in range(20):
        actions = [agent.select_action(obs, training=False) 
                  for agent, obs in zip(idqn_agents, observations)]
        observations, _, done, _ = idqn_env.step(actions)
        if done:
            break
    
    fig2 = idqn_env.render("Independent DQN - After 20 Steps")
    fig2.savefig('pytorch_rl_tutorial/idqn_final.png', dpi=300, bbox_inches='tight')
    plt.close(fig2)
    
    # Test MADDPG
    observations = maddpg_env.reset()
    
    fig3 = maddpg_env.render("MADDPG - Initial State")
    fig3.savefig('pytorch_rl_tutorial/maddpg_initial.png', dpi=300, bbox_inches='tight')
    plt.close(fig3)
    
    # Run for a few steps
    for _ in range(20):
        actions = maddpg_agents.select_actions(observations, training=False)
        observations, _, done, _ = maddpg_env.step(actions)
        if done:
            break
    
    fig4 = maddpg_env.render("MADDPG - After 20 Steps")
    fig4.savefig('pytorch_rl_tutorial/maddpg_final.png', dpi=300, bbox_inches='tight')
    plt.close(fig4)
    
    print("Agent behavior visualizations saved:")
    print("  - idqn_initial.png, idqn_final.png")
    print("  - maddpg_initial.png, maddpg_final.png")


if __name__ == "__main__":
    print("Chapter 22: Multi-Agent RL")
    print("="*40)
    print("This chapter demonstrates multi-agent reinforcement learning.")
    print("We'll compare Independent DQN and MADDPG on a cooperative task.\n")
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Compare multi-agent methods
    results = compare_multi_agent_methods()
    
    print("\n" + "="*40)
    print("Key Concepts Demonstrated:")
    print("- Independent DQN for decentralized learning")
    print("- MADDPG with centralized training, decentralized execution")
    print("- Cooperative multi-agent task (landmark coverage)")
    print("- Non-stationarity challenges in multi-agent learning")
    print("- Credit assignment in cooperative settings")
    print("\nMulti-agent RL applications:")
    print("- Autonomous vehicle coordination")
    print("- Robot swarm coordination")
    print("- Game playing (team sports, strategy games)")
    print("- Resource allocation and scheduling")
    print("- Traffic light control systems")
    print("\nCongratulations! You've completed all 22 chapters of the")
    print("PyTorch Reinforcement Learning Tutorial series!")
