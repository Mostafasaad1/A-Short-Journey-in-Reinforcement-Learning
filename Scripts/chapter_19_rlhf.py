#!/usr/bin/env python3
"""
Chapter 19: Reinforcement Learning with Human Feedback (RLHF)

This chapter demonstrates how to integrate human feedback into RL training
through reward modeling and preference learning. We simulate the RLHF process
used in training large language models and other AI systems.

Key concepts covered:
- Human preference collection and modeling
- Reward model training from preference data
- PPO fine-tuning with learned reward models
- Comparison with ground-truth rewards
- Safety considerations and alignment


"""

# Chapter 19: Reinforcement Learning with Human Feedback
# Start by installing the required packages:
# !pip install torch gymnasium[all] matplotlib numpy tqdm -q

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

# Data structures for RLHF
@dataclass
class Trajectory:
    """Represents a complete trajectory/episode."""
    states: List[np.ndarray]
    actions: List[int]
    rewards: List[float]
    total_reward: float
    length: int

@dataclass 
class Preference:
    """Represents a human preference between two trajectories."""
    traj_a: Trajectory
    traj_b: Trajectory
    preference: int  # 0 for traj_a preferred, 1 for traj_b preferred
    confidence: float  # Human confidence in the preference


class CustomCartPoleEnv:
    """Custom CartPole environment with interpretable reward design."""
    
    def __init__(self, reward_type="standard"):
        self.reward_type = reward_type
        self.reset()
        
        # Environment parameters
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02
        
        # Thresholds
        self.theta_threshold_radians = 12 * 2 * np.pi / 360
        self.x_threshold = 2.4
        
        self.action_space = 2
        self.observation_space = 4
        
    def reset(self):
        """Reset environment to initial state."""
        self.state = np.random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps = 0
        return self.state.copy()
    
    def step(self, action):
        """Take a step in the environment."""
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        
        costheta = np.cos(theta)
        sintheta = np.sin(theta)
        
        temp = (force + self.polemass_length * theta_dot**2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0/3.0 - self.masspole * costheta**2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc
        
        self.state = np.array([x, x_dot, theta, theta_dot])
        
        # Check termination
        done = (
            x < -self.x_threshold or 
            x > self.x_threshold or 
            theta < -self.theta_threshold_radians or 
            theta > self.theta_threshold_radians
        )
        
        # Calculate reward based on type
        reward = self._calculate_reward()
        
        self.steps += 1
        if self.steps >= 500:
            done = True
            
        return self.state.copy(), reward, done, {}
    
    def _calculate_reward(self):
        """Calculate reward based on reward type."""
        x, x_dot, theta, theta_dot = self.state
        
        if self.reward_type == "standard":
            return 1.0  # Standard CartPole reward
        
        elif self.reward_type == "balanced":
            # Reward for keeping pole upright and cart centered
            angle_reward = 1.0 - abs(theta) / self.theta_threshold_radians
            position_reward = 1.0 - abs(x) / self.x_threshold
            stability_reward = 1.0 - (abs(x_dot) + abs(theta_dot)) / 10.0
            return (angle_reward + position_reward + stability_reward) / 3.0
        
        elif self.reward_type == "smooth":
            # Smooth reward encouraging stability
            angle_penalty = (theta / self.theta_threshold_radians) ** 2
            position_penalty = (x / self.x_threshold) ** 2
            velocity_penalty = 0.1 * (x_dot**2 + theta_dot**2)
            return 1.0 - angle_penalty - position_penalty - velocity_penalty
        
        else:
            return 1.0


class PolicyNetwork(nn.Module):
    """Policy network for CartPole."""
    
    def __init__(self, state_dim=4, action_dim=2, hidden_dim=64):
        super(PolicyNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, state):
        return self.network(state)
    
    def get_action_and_log_prob(self, state):
        """Get action and log probability for given state."""
        action_probs = self.forward(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob


class RewardModel(nn.Module):
    """Reward model that learns from human preferences."""
    
    def __init__(self, state_dim=4, hidden_dim=128):
        super(RewardModel, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state):
        """Predict reward for given state."""
        return self.network(state)
    
    def predict_trajectory_reward(self, trajectory):
        """Predict total reward for a trajectory."""
        states = torch.FloatTensor(np.array(trajectory.states))
        state_rewards = self.forward(states)
        return torch.sum(state_rewards)


class HumanOracle:
    """Simulates human preferences with some noise and bias."""
    
    def __init__(self, env, noise_level=0.1, preference_bias=0.0):
        self.env = env
        self.noise_level = noise_level
        self.preference_bias = preference_bias
        
    def compare_trajectories(self, traj_a: Trajectory, traj_b: Trajectory) -> Preference:
        """Compare two trajectories and return preference."""
        
        # Calculate true utility for each trajectory
        utility_a = self._calculate_utility(traj_a)
        utility_b = self._calculate_utility(traj_b)
        
        # Add noise to utilities
        noisy_utility_a = utility_a + np.random.normal(0, self.noise_level)
        noisy_utility_b = utility_b + np.random.normal(0, self.noise_level)
        
        # Add preference bias
        noisy_utility_a += self.preference_bias
        
        # Determine preference
        if noisy_utility_a > noisy_utility_b:
            preference = 0  # Prefer trajectory A
        else:
            preference = 1  # Prefer trajectory B
        
        # Calculate confidence based on utility difference
        utility_diff = abs(utility_a - utility_b)
        confidence = min(1.0, utility_diff / 5.0)  # Normalize to [0, 1]
        
        return Preference(traj_a, traj_b, preference, confidence)
    
    def _calculate_utility(self, trajectory: Trajectory) -> float:
        """Calculate utility of trajectory based on multiple criteria."""
        # Length-based utility (longer episodes are better)
        length_utility = trajectory.length / 500.0
        
        # Stability utility (less variance in states)
        states = np.array(trajectory.states)
        if len(states) > 1:
            state_variance = np.mean(np.var(states, axis=0))
            stability_utility = max(0, 1.0 - state_variance)
        else:
            stability_utility = 0
        
        # Smoothness utility (less erratic actions)
        if len(trajectory.actions) > 1:
            action_changes = sum(1 for i in range(1, len(trajectory.actions)) 
                               if trajectory.actions[i] != trajectory.actions[i-1])
            smoothness_utility = max(0, 1.0 - action_changes / len(trajectory.actions))
        else:
            smoothness_utility = 1.0
        
        # Combine utilities
        total_utility = (0.5 * length_utility + 
                        0.3 * stability_utility + 
                        0.2 * smoothness_utility)
        
        return total_utility


class RLHFTrainer:
    """Main trainer for RLHF process."""
    
    def __init__(self, env, human_oracle, initial_policy=None):
        self.env = env
        self.human_oracle = human_oracle
        
        # Initialize policy
        if initial_policy is None:
            self.policy = PolicyNetwork()
        else:
            self.policy = initial_policy
            
        # Initialize reward model
        self.reward_model = RewardModel()
        
        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=3e-4)
        self.reward_optimizer = optim.Adam(self.reward_model.parameters(), lr=1e-3)
        
        # Storage
        self.preference_data = []
        self.trajectory_buffer = deque(maxlen=1000)
        
        # Statistics
        self.training_stats = {
            "policy_rewards": [],
            "reward_model_loss": [],
            "preference_accuracy": [],
            "human_agreement": []
        }
    
    def collect_trajectories(self, num_trajectories=100):
        """Collect trajectories using current policy."""
        trajectories = []
        
        for _ in range(num_trajectories):
            trajectory = self._rollout_trajectory()
            trajectories.append(trajectory)
            self.trajectory_buffer.append(trajectory)
        
        return trajectories
    
    def _rollout_trajectory(self):
        """Rollout a single trajectory."""
        states = []
        actions = []
        rewards = []
        
        state = self.env.reset()
        done = False
        
        while not done:
            states.append(state.copy())
            
            # Get action from policy
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action, _ = self.policy.get_action_and_log_prob(state_tensor)
            action = action.item()
            
            actions.append(action)
            
            # Take step
            next_state, reward, done, _ = self.env.step(action)
            rewards.append(reward)
            
            state = next_state
        
        return Trajectory(
            states=states,
            actions=actions,
            rewards=rewards,
            total_reward=sum(rewards),
            length=len(states)
        )
    
    def collect_preferences(self, num_preferences=50):
        """Collect human preferences between trajectory pairs."""
        if len(self.trajectory_buffer) < 2:
            return []
        
        new_preferences = []
        trajectories = list(self.trajectory_buffer)
        
        for _ in range(num_preferences):
            # Sample two different trajectories
            traj_a, traj_b = random.sample(trajectories, 2)
            
            # Get human preference
            preference = self.human_oracle.compare_trajectories(traj_a, traj_b)
            new_preferences.append(preference)
            
        self.preference_data.extend(new_preferences)
        return new_preferences
    
    def train_reward_model(self, num_epochs=100, batch_size=32):
        """Train reward model on preference data."""
        if len(self.preference_data) < batch_size:
            return 0.0
        
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        for epoch in range(num_epochs):
            # Sample batch of preferences
            batch_preferences = random.sample(self.preference_data, 
                                            min(batch_size, len(self.preference_data)))
            
            batch_loss = 0.0
            
            for preference in batch_preferences:
                # Get predicted rewards for both trajectories
                reward_a = self.reward_model.predict_trajectory_reward(preference.traj_a)
                reward_b = self.reward_model.predict_trajectory_reward(preference.traj_b)
                
                # Create preference labels
                if preference.preference == 0:  # Prefer trajectory A
                    labels = torch.FloatTensor([1.0, 0.0])
                else:  # Prefer trajectory B
                    labels = torch.FloatTensor([0.0, 1.0])
                
                # Bradley-Terry model for preferences
                rewards = torch.stack([reward_a, reward_b])
                probs = F.softmax(rewards, dim=0)
                
                # Cross-entropy loss
                loss = F.cross_entropy(probs.unsqueeze(0), labels.unsqueeze(0))
                batch_loss += loss
                
                # Track accuracy
                predicted_pref = 0 if reward_a > reward_b else 1
                if predicted_pref == preference.preference:
                    correct_predictions += 1
                total_predictions += 1
            
            # Update reward model
            self.reward_optimizer.zero_grad()
            batch_loss.backward()
            self.reward_optimizer.step()
            
            total_loss += batch_loss.item()
        
        avg_loss = total_loss / num_epochs
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        
        self.training_stats["reward_model_loss"].append(avg_loss)
        self.training_stats["preference_accuracy"].append(accuracy)
        
        return avg_loss
    
    def train_policy_with_learned_rewards(self, num_epochs=100):
        """Train policy using PPO with learned reward model."""
        # Collect trajectories
        trajectories = self.collect_trajectories(num_trajectories=50)
        
        # Convert to training data
        states = []
        actions = []
        old_log_probs = []
        rewards = []
        
        for trajectory in trajectories:
            traj_states = torch.FloatTensor(np.array(trajectory.states))
            
            # Get learned rewards instead of environment rewards
            learned_rewards = self.reward_model(traj_states).squeeze().detach().numpy()
            
            # Get old policy probabilities
            with torch.no_grad():
                for i, state in enumerate(trajectory.states):
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    action_probs = self.policy(state_tensor)
                    dist = Categorical(action_probs)
                    old_log_prob = dist.log_prob(torch.tensor(trajectory.actions[i]))
                    
                    states.append(state)
                    actions.append(trajectory.actions[i])
                    old_log_probs.append(old_log_prob.item())
                    rewards.append(learned_rewards[i])
        
        if not states:
            return 0.0
        
        # Convert to tensors
        states_tensor = torch.FloatTensor(np.array(states))
        actions_tensor = torch.LongTensor(actions)
        old_log_probs_tensor = torch.FloatTensor(old_log_probs)
        rewards_tensor = torch.FloatTensor(rewards)
        
        # Calculate advantages (simplified)
        advantages = rewards_tensor - rewards_tensor.mean()
        
        total_loss = 0.0
        
        for epoch in range(num_epochs):
            # Get new policy probabilities
            action_probs = self.policy(states_tensor)
            dist = Categorical(action_probs)
            new_log_probs = dist.log_prob(actions_tensor)
            
            # PPO clipped objective
            ratio = torch.exp(new_log_probs - old_log_probs_tensor)
            clipped_ratio = torch.clamp(ratio, 0.8, 1.2)
            
            policy_loss = -torch.min(
                ratio * advantages,
                clipped_ratio * advantages
            ).mean()
            
            # Entropy bonus
            entropy = dist.entropy().mean()
            
            total_loss_epoch = policy_loss - 0.01 * entropy
            
            # Update policy
            self.policy_optimizer.zero_grad()
            total_loss_epoch.backward()
            self.policy_optimizer.step()
            
            total_loss += total_loss_epoch.item()
        
        avg_reward = np.mean([traj.total_reward for traj in trajectories])
        self.training_stats["policy_rewards"].append(avg_reward)
        
        return total_loss / num_epochs
    
    def evaluate_policy(self, num_episodes=20):
        """Evaluate current policy performance."""
        total_rewards = []
        episode_lengths = []
        
        for _ in range(num_episodes):
            trajectory = self._rollout_trajectory()
            total_rewards.append(trajectory.total_reward)
            episode_lengths.append(trajectory.length)
        
        return {
            "avg_reward": np.mean(total_rewards),
            "std_reward": np.std(total_rewards),
            "avg_length": np.mean(episode_lengths),
            "success_rate": sum(1 for length in episode_lengths if length >= 200) / num_episodes
        }


def run_rlhf_experiment():
    """Run complete RLHF experiment."""
    print("=== Chapter 19: Reinforcement Learning with Human Feedback ===")
    print("Demonstrating RLHF process with simulated human preferences...\n")
    
    # Create environment and human oracle
    env = CustomCartPoleEnv(reward_type="balanced")
    human_oracle = HumanOracle(env, noise_level=0.1, preference_bias=0.0)
    
    print(f"Environment: Custom CartPole with '{env.reward_type}' rewards")
    print(f"Human oracle noise level: {human_oracle.noise_level}")
    print(f"Human oracle bias: {human_oracle.preference_bias}\n")
    
    # Initialize RLHF trainer
    trainer = RLHFTrainer(env, human_oracle)
    
    # Phase 1: Collect initial data
    print("Phase 1: Collecting initial trajectories...")
    initial_trajectories = trainer.collect_trajectories(num_trajectories=100)
    initial_performance = trainer.evaluate_policy()
    print(f"Initial performance: Avg reward = {initial_performance['avg_reward']:.2f}, "
          f"Success rate = {initial_performance['success_rate']:.1%}")
    
    # Phase 2: Collect preferences and train reward model
    print("\nPhase 2: Collecting preferences and training reward model...")
    
    num_rlhf_iterations = 10
    
    for iteration in range(num_rlhf_iterations):
        print(f"\n--- RLHF Iteration {iteration + 1} ---")
        
        # Collect new preferences
        new_preferences = trainer.collect_preferences(num_preferences=30)
        print(f"Collected {len(new_preferences)} new preferences")
        print(f"Total preferences: {len(trainer.preference_data)}")
        
        # Train reward model
        reward_loss = trainer.train_reward_model(num_epochs=50)
        print(f"Reward model loss: {reward_loss:.4f}")
        
        if len(trainer.training_stats["preference_accuracy"]) > 0:
            accuracy = trainer.training_stats["preference_accuracy"][-1]
            print(f"Preference prediction accuracy: {accuracy:.1%}")
        
        # Train policy with learned rewards
        policy_loss = trainer.train_policy_with_learned_rewards(num_epochs=50)
        print(f"Policy loss: {policy_loss:.4f}")
        
        # Evaluate current policy
        performance = trainer.evaluate_policy()
        print(f"Performance: Avg reward = {performance['avg_reward']:.2f}, "
              f"Success rate = {performance['success_rate']:.1%}")
        
        # Collect new trajectories for next iteration
        if iteration < num_rlhf_iterations - 1:
            trainer.collect_trajectories(num_trajectories=20)
    
    # Final evaluation
    print("\n=== Final Evaluation ===")
    final_performance = trainer.evaluate_policy(num_episodes=50)
    print(f"Final performance:")
    print(f"  Average reward: {final_performance['avg_reward']:.2f} ± {final_performance['std_reward']:.2f}")
    print(f"  Average episode length: {final_performance['avg_length']:.1f}")
    print(f"  Success rate: {final_performance['success_rate']:.1%}")
    
    # Create training plots
    create_rlhf_plots(trainer)
    
    # Analyze reward model
    analyze_reward_model(trainer, env)
    
    return trainer


def create_rlhf_plots(trainer):
    """Create plots showing RLHF training progress."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('RLHF Training Progress', fontsize=16)
    
    # Policy performance over time
    if trainer.training_stats["policy_rewards"]:
        axes[0, 0].plot(trainer.training_stats["policy_rewards"], 'b-', linewidth=2)
        axes[0, 0].set_title('Policy Performance (Learned Rewards)')
        axes[0, 0].set_xlabel('RLHF Iteration')
        axes[0, 0].set_ylabel('Average Reward')
        axes[0, 0].grid(True, alpha=0.3)
    
    # Reward model loss
    if trainer.training_stats["reward_model_loss"]:
        axes[0, 1].plot(trainer.training_stats["reward_model_loss"], 'r-', linewidth=2)
        axes[0, 1].set_title('Reward Model Training Loss')
        axes[0, 1].set_xlabel('RLHF Iteration')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].grid(True, alpha=0.3)
    
    # Preference prediction accuracy
    if trainer.training_stats["preference_accuracy"]:
        axes[1, 0].plot(trainer.training_stats["preference_accuracy"], 'g-', linewidth=2)
        axes[1, 0].set_title('Preference Prediction Accuracy')
        axes[1, 0].set_xlabel('RLHF Iteration')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].grid(True, alpha=0.3)
    
    # Preference distribution
    if trainer.preference_data:
        preferences = [p.preference for p in trainer.preference_data]
        confidences = [p.confidence for p in trainer.preference_data]
        
        # Scatter plot of preferences vs confidence
        axes[1, 1].scatter(preferences, confidences, alpha=0.6)
        axes[1, 1].set_title('Preference Distribution')
        axes[1, 1].set_xlabel('Preference (0=A, 1=B)')
        axes[1, 1].set_ylabel('Human Confidence')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pytorch_rl_tutorial/rlhf_training_progress.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\nRLHF training plots saved as 'rlhf_training_progress.png'")


def analyze_reward_model(trainer, env):
    """Analyze the learned reward model."""
    print("\n=== Reward Model Analysis ===")
    
    # Test reward model on different states
    test_states = [
        [0.0, 0.0, 0.0, 0.0],      # Centered, upright
        [1.0, 0.0, 0.1, 0.0],      # Off-center, slightly tilted
        [0.0, 0.0, 0.3, 0.0],      # Centered, more tilted
        [2.0, 0.0, 0.0, 0.0],      # Far off-center
        [0.0, 2.0, 0.0, 2.0],      # High velocities
    ]
    
    state_descriptions = [
        "Centered, upright",
        "Off-center, slightly tilted", 
        "Centered, more tilted",
        "Far off-center",
        "High velocities"
    ]
    
    print("Reward model predictions for different states:")
    for state, desc in zip(test_states, state_descriptions):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        predicted_reward = trainer.reward_model(state_tensor).item()
        
        # Compare with ground truth reward
        env.state = np.array(state)
        true_reward = env._calculate_reward()
        
        print(f"  {desc:25} | Predicted: {predicted_reward:6.3f} | True: {true_reward:6.3f}")
    
    # Analyze preference agreement
    if trainer.preference_data:
        correct_predictions = 0
        total_predictions = len(trainer.preference_data)
        
        for preference in trainer.preference_data:
            reward_a = trainer.reward_model.predict_trajectory_reward(preference.traj_a)
            reward_b = trainer.reward_model.predict_trajectory_reward(preference.traj_b)
            
            predicted_pref = 0 if reward_a > reward_b else 1
            if predicted_pref == preference.preference:
                correct_predictions += 1
        
        agreement_rate = correct_predictions / total_predictions
        print(f"\nOverall preference agreement: {agreement_rate:.1%} ({correct_predictions}/{total_predictions})")
        
        # Analyze confidence vs correctness
        high_conf_correct = 0
        high_conf_total = 0
        
        for preference in trainer.preference_data:
            if preference.confidence > 0.7:  # High confidence
                high_conf_total += 1
                
                reward_a = trainer.reward_model.predict_trajectory_reward(preference.traj_a)
                reward_b = trainer.reward_model.predict_trajectory_reward(preference.traj_b)
                
                predicted_pref = 0 if reward_a > reward_b else 1
                if predicted_pref == preference.preference:
                    high_conf_correct += 1
        
        if high_conf_total > 0:
            high_conf_accuracy = high_conf_correct / high_conf_total
            print(f"High-confidence preference accuracy: {high_conf_accuracy:.1%} ({high_conf_correct}/{high_conf_total})")


def compare_with_baseline():
    """Compare RLHF-trained policy with baseline methods."""
    print("\n=== Comparison with Baseline Methods ===")
    
    env = CustomCartPoleEnv(reward_type="balanced")
    
    # Train baseline PPO agent with true rewards
    print("Training baseline PPO with true rewards...")
    baseline_policy = PolicyNetwork()
    baseline_optimizer = optim.Adam(baseline_policy.parameters(), lr=3e-4)
    
    # Simple PPO training loop
    baseline_rewards = []
    for episode in range(200):
        # Collect trajectory
        states = []
        actions = []
        rewards = []
        log_probs = []
        
        state = env.reset()
        done = False
        
        while not done:
            states.append(state)
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            action, log_prob = baseline_policy.get_action_and_log_prob(state_tensor)
            action = action.item()
            
            next_state, reward, done, _ = env.step(action)
            
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            
            state = next_state
        
        # Simple policy update
        if len(states) > 0:
            returns = []
            G = 0
            for r in reversed(rewards):
                G = r + 0.99 * G
                returns.insert(0, G)
            
            returns = torch.FloatTensor(returns)
            log_probs = torch.stack(log_probs)
            
            # Policy gradient update
            policy_loss = -(log_probs * (returns - returns.mean())).mean()
            
            baseline_optimizer.zero_grad()
            policy_loss.backward()
            baseline_optimizer.step()
        
        baseline_rewards.append(sum(rewards))
    
    # Evaluate baseline
    baseline_performance = []
    for _ in range(50):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action_probs = baseline_policy(state_tensor)
                action = torch.argmax(action_probs).item()
            
            state, reward, done, _ = env.step(action)
            total_reward += reward
        
        baseline_performance.append(total_reward)
    
    baseline_avg = np.mean(baseline_performance)
    baseline_std = np.std(baseline_performance)
    
    print(f"Baseline PPO performance: {baseline_avg:.2f} ± {baseline_std:.2f}")
    
    # Note: RLHF results would be compared here in a full implementation
    print("\nNote: In practice, you would compare this with the RLHF-trained policy")
    print("to see if human feedback leads to better-aligned behavior.")


if __name__ == "__main__":
    print("Chapter 19: Reinforcement Learning with Human Feedback")
    print("="*60)
    print("This chapter demonstrates RLHF for training aligned AI systems.")
    print("We'll simulate human preferences and train reward models.\n")
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Run RLHF experiment
    trainer = run_rlhf_experiment()
    
    # Compare with baseline
    compare_with_baseline()
    
    print("\n" + "="*60)
    print("Key Concepts Demonstrated:")
    print("- Human preference collection and modeling")
    print("- Reward model training from preference pairs")
    print("- PPO fine-tuning with learned reward signals")
    print("- Bradley-Terry model for preference learning")
    print("- Evaluation of alignment and safety")
    print("\nRLHF is crucial for:")
    print("- Aligning AI systems with human values")
    print("- Training systems when reward engineering is difficult")
    print("- Incorporating subjective human judgments")
    print("- Improving safety and reducing harmful outputs")
    print("- Fine-tuning large language models")
    print("\nNext: Chapter 20 - AlphaGo Zero and MuZero")
