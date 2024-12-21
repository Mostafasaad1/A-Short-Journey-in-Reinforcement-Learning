# Chapter 4: The Cross-Entropy Method (CEM)
# Model-free policy search for reinforcement learning

# Install required packages:
# !pip install torch gymnasium[all] matplotlib numpy -q

import torch
import torch.nn as nn
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
from collections import deque
import random

# Configure matplotlib for proper font rendering
plt.switch_backend("Agg")
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "PingFang SC", "Arial Unicode MS", "Hiragino Sans GB"]
plt.rcParams["axes.unicode_minus"] = False

print("Chapter 4: The Cross-Entropy Method (CEM)")
print("=" * 50)

# 1. INTRODUCTION TO CROSS-ENTROPY METHOD
print("\n1. What is the Cross-Entropy Method?")
print("-" * 30)

print("""
The Cross-Entropy Method (CEM) is a gradient-free optimization algorithm that:

1. Treats RL as an optimization problem over policy parameters
2. Uses sampling and selection to find good policy parameters
3. Updates parameters via maximum likelihood estimation
4. No gradients needed - works with any differentiable or non-differentiable policy

CEM Algorithm:
1. Sample N sets of policy parameters from current distribution
2. Evaluate each parameter set by running episodes
3. Select top K elite parameter sets (highest rewards)
4. Update parameter distribution using elite sets
5. Repeat until convergence

Advantages:
- Simple to implement and understand
- No gradient computation required
- Works with discrete and continuous action spaces
- Good exploration through parameter space sampling

Disadvantages:
- Sample inefficient compared to gradient methods
- Can get stuck in local optima
- Requires many episodes per iteration
""")

# 2. LINEAR POLICY FOR CARTPOLE
print("\n2. Linear Policy Implementation")
print("-" * 30)

class LinearPolicy:
    """A simple linear policy that maps states to action logits.
    
    For CartPole: state = [x, x_dot, theta, theta_dot] -> action logits [left, right]
    Action selection: sample from softmax distribution or take argmax
    """
    
    def __init__(self, state_dim: int, action_dim: int):
        """Initialize linear policy parameters.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Number of possible actions
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        # Initialize weights randomly - these are the parameters we'll optimize
        self.weights = np.random.randn(state_dim, action_dim) * 0.1
        self.bias = np.zeros(action_dim)
    
    def get_action(self, state: np.ndarray, deterministic: bool = False) -> int:
        """Select action given current state.
        
        Args:
            state: Environment state
            deterministic: If True, take argmax; if False, sample from distribution
            
        Returns:
            Selected action index
        """
        # Compute action logits: logits = state @ weights + bias
        logits = np.dot(state, self.weights) + self.bias
        
        if deterministic:
            return np.argmax(logits)
        else:
            # Convert logits to probabilities using softmax
            probs = self.softmax(logits)
            # Sample action from probability distribution
            return np.random.choice(self.action_dim, p=probs)
    
    def softmax(self, logits: np.ndarray) -> np.ndarray:
        """Compute softmax probabilities from logits."""
        exp_logits = np.exp(logits - np.max(logits))  # Subtract max for numerical stability
        return exp_logits / np.sum(exp_logits)
    
    def get_parameters(self) -> np.ndarray:
        """Get flattened parameter vector."""
        return np.concatenate([self.weights.flatten(), self.bias.flatten()])
    
    def set_parameters(self, params: np.ndarray) -> None:
        """Set parameters from flattened vector."""
        weight_size = self.state_dim * self.action_dim
        self.weights = params[:weight_size].reshape(self.state_dim, self.action_dim)
        self.bias = params[weight_size:]

# Test the linear policy
print("Testing linear policy...")
env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

policy = LinearPolicy(state_dim, action_dim)
print(f"State dimension: {state_dim}")
print(f"Action dimension: {action_dim}")
print(f"Parameter count: {len(policy.get_parameters())}")

# Test action selection
test_state = env.reset()[0]
action = policy.get_action(test_state)
print(f"Test state: {test_state}")
print(f"Selected action: {action}")

# 3. EPISODE EVALUATION
print("\n3. Episode Evaluation")
print("-" * 30)

def evaluate_policy(policy: LinearPolicy, env: gym.Env, n_episodes: int = 1) -> float:
    """Evaluate policy performance over multiple episodes.
    
    Args:
        policy: Policy to evaluate
        env: Environment to test in
        n_episodes: Number of episodes to run
        
    Returns:
        Average episode reward
    """
    total_reward = 0.0
    
    for episode in range(n_episodes):
        state, _ = env.reset()
        episode_reward = 0.0
        done = False
        
        while not done:
            action = policy.get_action(state, deterministic=True)  # Use deterministic for evaluation
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
        
        total_reward += episode_reward
    
    return total_reward / n_episodes

# Test policy evaluation
print("Evaluating random policy...")
random_reward = evaluate_policy(policy, env, n_episodes=5)
print(f"Random policy average reward: {random_reward:.2f}")

# 4. CROSS-ENTROPY METHOD IMPLEMENTATION
print("\n4. Cross-Entropy Method Implementation")
print("-" * 30)

class CrossEntropyMethod:
    """Cross-Entropy Method for policy optimization."""
    
    def __init__(self, state_dim: int, action_dim: int, 
                 n_samples: int = 50, elite_ratio: float = 0.2,
                 noise_scale: float = 1.0, noise_decay: float = 0.99):
        """Initialize CEM optimizer.
        
        Args:
            state_dim: State space dimension
            action_dim: Action space dimension
            n_samples: Number of parameter samples per iteration
            elite_ratio: Fraction of samples to use as elites
            noise_scale: Standard deviation for parameter sampling
            noise_decay: Decay rate for noise scale
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_samples = n_samples
        self.n_elite = int(n_samples * elite_ratio)
        self.noise_scale = noise_scale
        self.noise_decay = noise_decay
        
        # Initialize parameter distribution
        param_dim = state_dim * action_dim + action_dim  # weights + bias
        self.param_mean = np.zeros(param_dim)
        self.param_std = np.ones(param_dim) * noise_scale
        
        # Tracking
        self.iteration = 0
        self.best_reward = -np.inf
        self.reward_history = []
        self.elite_reward_history = []
    
    def sample_parameters(self) -> List[np.ndarray]:
        """Sample parameter vectors from current distribution.
        
        Returns:
            List of parameter vectors
        """
        samples = []
        for _ in range(self.n_samples):
            # Sample from multivariate normal distribution
            params = np.random.normal(self.param_mean, self.param_std)
            samples.append(params)
        return samples
    
    def evaluate_samples(self, param_samples: List[np.ndarray], env: gym.Env) -> List[float]:
        """Evaluate each parameter sample.
        
        Args:
            param_samples: List of parameter vectors to evaluate
            env: Environment for evaluation
            
        Returns:
            List of rewards corresponding to each parameter sample
        """
        rewards = []
        policy = LinearPolicy(self.state_dim, self.action_dim)
        
        for params in param_samples:
            policy.set_parameters(params)
            reward = evaluate_policy(policy, env, n_episodes=1)
            rewards.append(reward)
        
        return rewards
    
    def select_elites(self, param_samples: List[np.ndarray], 
                     rewards: List[float]) -> Tuple[List[np.ndarray], List[float]]:
        """Select elite parameter samples.
        
        Args:
            param_samples: All parameter samples
            rewards: Corresponding rewards
            
        Returns:
            Elite parameter samples and their rewards
        """
        # Sort by reward (descending)
        elite_indices = np.argsort(rewards)[-self.n_elite:]
        elite_params = [param_samples[i] for i in elite_indices]
        elite_rewards = [rewards[i] for i in elite_indices]
        
        return elite_params, elite_rewards
    
    def update_distribution(self, elite_params: List[np.ndarray]) -> None:
        """Update parameter distribution using elite samples.
        
        Args:
            elite_params: Elite parameter samples
        """
        # Convert to numpy array for easier computation
        elite_array = np.array(elite_params)
        
        # Update mean and standard deviation using maximum likelihood
        self.param_mean = np.mean(elite_array, axis=0)
        self.param_std = np.std(elite_array, axis=0) + 1e-6  # Add small epsilon for numerical stability
        
        # Apply noise decay
        self.param_std *= self.noise_decay
    
    def train(self, env: gym.Env, max_iterations: int = 100, 
             target_reward: float = 475.0) -> Dict:
        """Train policy using Cross-Entropy Method.
        
        Args:
            env: Environment to train in
            max_iterations: Maximum number of CEM iterations
            target_reward: Stop training when this reward is reached
            
        Returns:
            Training statistics
        """
        print(f"Starting CEM training...")
        print(f"Samples per iteration: {self.n_samples}, Elite count: {self.n_elite}")
        
        for iteration in range(max_iterations):
            self.iteration = iteration
            
            # Sample parameter vectors
            param_samples = self.sample_parameters()
            
            # Evaluate samples
            rewards = self.evaluate_samples(param_samples, env)
            
            # Select elites
            elite_params, elite_rewards = self.select_elites(param_samples, rewards)
            
            # Update distribution
            self.update_distribution(elite_params)
            
            # Track statistics
            avg_reward = np.mean(rewards)
            elite_avg_reward = np.mean(elite_rewards)
            max_reward = np.max(rewards)
            
            self.reward_history.append(avg_reward)
            self.elite_reward_history.append(elite_avg_reward)
            
            if max_reward > self.best_reward:
                self.best_reward = max_reward
            
            # Logging
            if (iteration + 1) % 10 == 0 or iteration == 0:
                print(f"Iteration {iteration+1:3d}: "
                      f"Avg={avg_reward:6.1f}, Elite={elite_avg_reward:6.1f}, "
                      f"Max={max_reward:6.1f}, Std={np.mean(self.param_std):.4f}")
            
            # Check convergence
            if elite_avg_reward >= target_reward:
                print(f"\nTarget reward {target_reward} reached! Stopping training.")
                break
        
        return {
            'best_reward': self.best_reward,
            'final_mean_reward': avg_reward,
            'final_elite_reward': elite_avg_reward,
            'iterations': iteration + 1,
            'reward_history': self.reward_history,
            'elite_reward_history': self.elite_reward_history
        }
    
    def get_best_policy(self) -> LinearPolicy:
        """Get policy with current best parameters.
        
        Returns:
            Policy using current parameter mean
        """
        policy = LinearPolicy(self.state_dim, self.action_dim)
        policy.set_parameters(self.param_mean)
        return policy

# 5. TRAINING WITH CEM
print("\n5. Training CartPole with CEM")
print("-" * 30)

# Initialize CEM optimizer
cem = CrossEntropyMethod(
    state_dim=state_dim,
    action_dim=action_dim,
    n_samples=50,      # Number of parameter samples per iteration
    elite_ratio=0.2,   # Use top 20% as elites
    noise_scale=1.0,   # Initial parameter noise
    noise_decay=0.99   # Reduce noise over time
)

# Train the policy
training_stats = cem.train(env, max_iterations=50, target_reward=450.0)

print(f"\nTraining completed!")
print(f"Best reward achieved: {training_stats['best_reward']:.1f}")
print(f"Final elite reward: {training_stats['final_elite_reward']:.1f}")
print(f"Iterations completed: {training_stats['iterations']}")

# 6. EVALUATION AND VISUALIZATION
print("\n6. Evaluating Trained Policy")
print("-" * 30)

# Get the best policy
best_policy = cem.get_best_policy()

# Evaluate over multiple episodes
final_reward = evaluate_policy(best_policy, env, n_episodes=10)
print(f"Final policy average reward (10 episodes): {final_reward:.1f}")

# 7. HYPERPARAMETER SENSITIVITY ANALYSIS
print("\n7. Hyperparameter Sensitivity Analysis")
print("-" * 30)

def sensitivity_analysis():
    """Test different hyperparameter combinations."""
    configs = [
        {'n_samples': 30, 'elite_ratio': 0.2, 'noise_scale': 1.0, 'name': 'Small N'},
        {'n_samples': 50, 'elite_ratio': 0.2, 'noise_scale': 1.0, 'name': 'Medium N'},
        {'n_samples': 100, 'elite_ratio': 0.2, 'noise_scale': 1.0, 'name': 'Large N'},
        {'n_samples': 50, 'elite_ratio': 0.1, 'noise_scale': 1.0, 'name': 'Low Elite'},
        {'n_samples': 50, 'elite_ratio': 0.3, 'noise_scale': 1.0, 'name': 'High Elite'},
        {'n_samples': 50, 'elite_ratio': 0.2, 'noise_scale': 0.5, 'name': 'Low Noise'},
        {'n_samples': 50, 'elite_ratio': 0.2, 'noise_scale': 2.0, 'name': 'High Noise'},
    ]
    
    results = []
    
    for config in configs:
        print(f"Testing {config['name']}...")
        cem_test = CrossEntropyMethod(
            state_dim=state_dim,
            action_dim=action_dim,
            n_samples=config['n_samples'],
            elite_ratio=config['elite_ratio'],
            noise_scale=config['noise_scale']
        )
        
        stats = cem_test.train(env, max_iterations=30, target_reward=400.0)
        results.append({
            'name': config['name'],
            'final_reward': stats['final_elite_reward'],
            'iterations': stats['iterations'],
            'converged': stats['final_elite_reward'] >= 400.0
        })
    
    return results

# Run sensitivity analysis (comment out if too slow)
# sensitivity_results = sensitivity_analysis()
# print("\nSensitivity Analysis Results:")
# for result in sensitivity_results:
#     print(f"{result['name']:12s}: Final={result['final_reward']:6.1f}, "
#           f"Iterations={result['iterations']:2d}, Converged={result['converged']}")

# 8. VISUALIZATION
print("\n8. Creating Visualizations")
print("-" * 30)

# Create comprehensive plots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Learning curve
ax1.plot(training_stats['reward_history'], label='Average Reward', alpha=0.7)
ax1.plot(training_stats['elite_reward_history'], label='Elite Average', linewidth=2)
ax1.axhline(y=475, color='r', linestyle='--', label='Target (475)')
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Reward')
ax1.set_title('CEM Learning Curve')
ax1.legend()
ax1.grid(True)

# Plot 2: Parameter evolution (mean values)
if len(cem.param_mean) <= 20:  # Only plot if not too many parameters
    ax2.bar(range(len(cem.param_mean)), cem.param_mean)
    ax2.set_xlabel('Parameter Index')
    ax2.set_ylabel('Parameter Value')
    ax2.set_title('Final Parameter Values')
    ax2.grid(True)
else:
    ax2.text(0.5, 0.5, f'Too many parameters\nto visualize\n({len(cem.param_mean)} total)', 
             ha='center', va='center', transform=ax2.transAxes)
    ax2.set_title('Parameter Visualization Skipped')

# Plot 3: Noise scale evolution
noise_evolution = [1.0 * (0.99 ** i) for i in range(len(training_stats['reward_history']))]
ax3.plot(noise_evolution)
ax3.set_xlabel('Iteration')
ax3.set_ylabel('Noise Scale')
ax3.set_title('Parameter Noise Decay')
ax3.grid(True)

# Plot 4: Action distribution visualization
test_states = []
test_actions = []
for _ in range(100):
    state, _ = env.reset()
    action = best_policy.get_action(state, deterministic=False)
    test_states.append(state)
    test_actions.append(action)

test_states = np.array(test_states)
action_0_mask = np.array(test_actions) == 0
action_1_mask = np.array(test_actions) == 1

# Plot cart position vs pole angle, colored by action
if len(test_states) > 0:
    ax4.scatter(test_states[action_0_mask, 0], test_states[action_0_mask, 2], 
               c='blue', alpha=0.6, label='Action 0 (Left)', s=20)
    ax4.scatter(test_states[action_1_mask, 0], test_states[action_1_mask, 2], 
               c='red', alpha=0.6, label='Action 1 (Right)', s=20)
    ax4.set_xlabel('Cart Position')
    ax4.set_ylabel('Pole Angle')
    ax4.set_title('Learned Policy Actions')
    ax4.legend()
    ax4.grid(True)

plt.tight_layout()
plt.savefig('pytorch_rl_tutorial/chapter_04_cem_results.png', dpi=150, bbox_inches='tight')
print("Results saved to chapter_04_cem_results.png")

# 9. COMPARISON WITH GRADIENT METHODS
print("\n9. CEM vs Gradient-Based Methods")
print("-" * 30)

print("""
Cross-Entropy Method vs Gradient-Based Methods:

CEM Advantages:
✓ No gradients required - works with non-differentiable policies
✓ Natural exploration through parameter space sampling
✓ Simple to implement and understand
✓ Robust to local optima in some cases
✓ Works well with small parameter spaces

CEM Disadvantages:
✗ Sample inefficient - needs many episodes per iteration
✗ Poor scaling to high-dimensional parameter spaces
✗ No direct use of value function or policy gradients
✗ Can be slow to converge

When to use CEM:
• Small to medium policy networks
• Non-differentiable policies or rewards
• Initial hyperparameter tuning
• Baseline comparison for other methods
• Environments where simulation is cheap

Gradient-based methods (DQN, Policy Gradients) are generally preferred
for larger networks and complex environments due to better sample efficiency.
""")

print(f"\nChapter 4 Complete! ✓")
print(f"Best CEM policy achieved {final_reward:.1f} average reward")
print(f"Ready to move on to Tabular Learning (Chapter 5)")

env.close()
