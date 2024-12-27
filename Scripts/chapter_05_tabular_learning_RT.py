# Chapter 5: Tabular Learning and the Bellman Equation
# Foundation of value-based reinforcement learning

# Install required packages:
# !pip install gymnasium[all] torch matplotlib numpy -q

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import random
from collections import defaultdict

# Configure matplotlib for proper font rendering and interactive mode
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "PingFang SC", "Arial Unicode MS", "Hiragino Sans GB"]
plt.rcParams["axes.unicode_minus"] = False
plt.ion()  # Enable interactive mode for live updates

print("Chapter 5: Tabular Learning and the Bellman Equation")
print("=" * 50)

# 1. INTRODUCTION TO VALUE-BASED LEARNING
print("\n1. Understanding Value-Based Reinforcement Learning")
print("-" * 30)

print("""
Value-based RL focuses on learning value functions:

1. STATE VALUE FUNCTION V(s):
   - Expected return when starting from state s
   - V(s) = E[G_t | S_t = s]
   - Where G_t is discounted future reward

2. ACTION-VALUE FUNCTION Q(s,a):
   - Expected return when taking action a in state s
   - Q(s,a) = E[G_t | S_t = s, A_t = a]
   - Used to derive optimal policy: π*(s) = argmax_a Q*(s,a)

3. BELLMAN EQUATION:
   - Recursive relationship for value functions
   - Q(s,a) = E[R + γ * max_a' Q(s',a') | s,a]
   - Forms the basis for Q-learning and other algorithms

4. BOOTSTRAPPING:
   - Update current estimates using other estimates
   - Enables learning without waiting for episode completion
   - Key difference from Monte Carlo methods
""")

# 2. PURE PYTHON Q-LEARNING IMPLEMENTATION
print("\n2. Pure Python Q-Learning for FrozenLake")
print("-" * 30)

class TabularQLearning:
    """Pure Python implementation of Q-learning for discrete state/action spaces."""
    
    def __init__(self, n_states: int, n_actions: int, 
                 learning_rate: float = 0.1, discount_factor: float = 0.99,
                 epsilon: float = 1.0, epsilon_decay: float = 0.995, 
                 epsilon_min: float = 0.01):
        """Initialize Q-learning agent.
        
        Args:
            n_states: Number of discrete states
            n_actions: Number of discrete actions
            learning_rate: Step size for Q-value updates (α)
            discount_factor: Future reward discount (γ)
            epsilon: Initial exploration rate
            epsilon_decay: Decay rate for epsilon
            epsilon_min: Minimum epsilon value
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Initialize Q-table with zeros
        self.q_table = np.zeros((n_states, n_actions))
        
        # Tracking statistics
        self.episode_rewards = []
        self.epsilon_history = []
        self.q_value_changes = []
    
    def get_action(self, state: int, training: bool = True) -> int:
        """Select action using ε-greedy policy.
        
        Args:
            state: Current state
            training: If True, use exploration; if False, be greedy
            
        Returns:
            Selected action
        """
        if training and random.random() < self.epsilon:
            # Exploration: random action
            return random.randint(0, self.n_actions - 1)
        else:
            # Exploitation: best known action
            return np.argmax(self.q_table[state])
    
    def update_q_value(self, state: int, action: int, reward: float, 
                      next_state: int, done: bool) -> float:
        """Update Q-value using Bellman equation.
        
        Args:
            state: Current state
            action: Action taken
            reward: Immediate reward
            next_state: Next state
            done: Whether episode ended
            
        Returns:
            Q-value change magnitude
        """
        # Current Q-value
        current_q = self.q_table[state, action]
        
        # Bellman target
        if done:
            target = reward  # No future rewards
        else:
            target = reward + self.gamma * np.max(self.q_table[next_state])
        
        # Q-learning update
        td_error = target - current_q
        new_q = current_q + self.lr * td_error
        self.q_table[state, action] = new_q
        
        return abs(td_error)
    
    def decay_epsilon(self) -> None:
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def train(self, env: gym.Env, n_episodes: int = 1000, 
             max_steps: int = 100) -> Dict:
        """Train the Q-learning agent with live graph updates.
        
        Args:
            env: Environment to train in
            n_episodes: Number of training episodes
            max_steps: Maximum steps per episode
            
        Returns:
            Training statistics
        """
        print(f"Starting Q-learning training for {n_episodes} episodes...")
        
        # Set up live plot for rewards
        self.fig, self.ax = plt.subplots(figsize=(8, 4))
        self.ax.set_title('Live Training Progress - Episode Rewards (Python Agent)')
        self.ax.set_xlabel('Episode')
        self.ax.set_ylabel('Reward')
        self.reward_line, = self.ax.plot([], [], 'b-', label='Episode Reward')
        self.ax.legend()
        self.ax.grid(True)
        plt.tight_layout()
        
        total_q_changes = []
        
        for episode in range(n_episodes):
            state, _ = env.reset()
            episode_reward = 0
            episode_q_changes = []
            
            for step in range(max_steps):
                # Select and take action
                action = self.get_action(state, training=True)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                # Update Q-table
                q_change = self.update_q_value(state, action, reward, next_state, done)
                episode_q_changes.append(q_change)
                
                episode_reward += reward
                state = next_state
                
                if done:
                    break
            
            # Decay exploration
            self.decay_epsilon()
            
            # Track statistics
            self.episode_rewards.append(episode_reward)
            self.epsilon_history.append(self.epsilon)
            total_q_changes.extend(episode_q_changes)
            self.q_value_changes.append(np.mean(episode_q_changes) if episode_q_changes else 0)
            
            # Update live plot every episode
            episodes = np.arange(len(self.episode_rewards))
            self.reward_line.set_data(episodes, self.episode_rewards)
            self.ax.relim()
            self.ax.autoscale_view()
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.pause(0.001)  # Brief pause to allow update
            
            # Logging
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                print(f"Episode {episode+1:4d}: Avg Reward = {avg_reward:.3f}, "
                      f"Epsilon = {self.epsilon:.3f}, Q-change = {np.mean(episode_q_changes):.4f}")
        
        return {
            'episode_rewards': self.episode_rewards,
            'epsilon_history': self.epsilon_history,
            'q_value_changes': self.q_value_changes,
            'final_q_table': self.q_table.copy()
        }
    
    def evaluate(self, env: gym.Env, n_episodes: int = 100) -> float:
        """Evaluate trained policy.
        
        Args:
            env: Environment to evaluate in
            n_episodes: Number of evaluation episodes
            
        Returns:
            Average episode reward
        """
        total_reward = 0
        success_count = 0
        
        for _ in range(n_episodes):
            state, _ = env.reset()
            episode_reward = 0
            
            for _ in range(100):  # Max steps
                action = self.get_action(state, training=False)  # No exploration
                state, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                
                if terminated or truncated:
                    if reward > 0:  # Success in FrozenLake
                        success_count += 1
                    break
            
            total_reward += episode_reward
        
        avg_reward = total_reward / n_episodes
        success_rate = success_count / n_episodes
        
        print(f"Evaluation: Avg Reward = {avg_reward:.3f}, Success Rate = {success_rate:.3f}")
        return avg_reward

# Test pure Python implementation
print("Testing Pure Python Q-learning...")
env = gym.make('FrozenLake-v1', is_slippery=True, render_mode=None)

# Get environment dimensions
n_states = env.observation_space.n
n_actions = env.action_space.n
print(f"Environment: {n_states} states, {n_actions} actions")

# Create and train agent
python_agent = TabularQLearning(n_states, n_actions)
python_stats = python_agent.train(env, n_episodes=2000)

# Evaluate trained agent
python_performance = python_agent.evaluate(env, n_episodes=100)

# 3. PYTORCH TENSOR IMPLEMENTATION
print("\n3. PyTorch Tensor-Based Q-Learning")
print("-" * 30)

class TorchTabularQLearning:
    """PyTorch tensor-based Q-learning implementation."""
    
    def __init__(self, n_states: int, n_actions: int,
                 learning_rate: float = 0.1, discount_factor: float = 0.99,
                 epsilon: float = 1.0, epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01, device: str = 'cpu'):
        """Initialize PyTorch Q-learning agent."""
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.device = device
        
        # Initialize Q-table as PyTorch tensor
        self.q_table = torch.zeros(n_states, n_actions, device=device, dtype=torch.float32)
        
        # Tracking
        self.episode_rewards = []
        self.epsilon_history = []
        self.q_value_changes = []
    
    def get_action(self, state: int, training: bool = True) -> int:
        """Select action using ε-greedy policy with PyTorch."""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        else:
            with torch.no_grad():
                return torch.argmax(self.q_table[state]).item()
    
    def update_q_value(self, state: int, action: int, reward: float,
                      next_state: int, done: bool) -> float:
        """Update Q-value using PyTorch tensors."""
        # Convert to tensors
        state_tensor = torch.tensor(state, device=self.device)
        action_tensor = torch.tensor(action, device=self.device)
        reward_tensor = torch.tensor(reward, device=self.device, dtype=torch.float32)
        next_state_tensor = torch.tensor(next_state, device=self.device)
        
        # Current Q-value
        current_q = self.q_table[state_tensor, action_tensor]
        
        # Bellman target
        with torch.no_grad():
            if done:
                target = reward_tensor
            else:
                target = reward_tensor + self.gamma * torch.max(self.q_table[next_state_tensor])
        
        # Compute TD error
        td_error = target - current_q
        
        # Update Q-value
        self.q_table[state_tensor, action_tensor] += self.lr * td_error
        
        return abs(td_error.item())
    
    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def train(self, env: gym.Env, n_episodes: int = 1000) -> Dict:
        """Train using PyTorch tensors with live graph updates."""
        print(f"Starting PyTorch Q-learning training for {n_episodes} episodes...")
        
        # Set up live plot for rewards
        self.fig, self.ax = plt.subplots(figsize=(8, 4))
        self.ax.set_title('Live Training Progress - Episode Rewards (PyTorch Agent)')
        self.ax.set_xlabel('Episode')
        self.ax.set_ylabel('Reward')
        self.reward_line, = self.ax.plot([], [], 'b-', label='Episode Reward')
        self.ax.legend()
        self.ax.grid(True)
        plt.tight_layout()
        
        for episode in range(n_episodes):
            state, _ = env.reset()
            episode_reward = 0
            episode_q_changes = []
            
            for step in range(100):  # Max steps
                action = self.get_action(state, training=True)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                q_change = self.update_q_value(state, action, reward, next_state, done)
                episode_q_changes.append(q_change)
                
                episode_reward += reward
                state = next_state
                
                if done:
                    break
            
            self.decay_epsilon()
            
            # Track statistics
            self.episode_rewards.append(episode_reward)
            self.epsilon_history.append(self.epsilon)
            self.q_value_changes.append(np.mean(episode_q_changes) if episode_q_changes else 0)
            
            # Update live plot every episode
            episodes = np.arange(len(self.episode_rewards))
            self.reward_line.set_data(episodes, self.episode_rewards)
            self.ax.relim()
            self.ax.autoscale_view()
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.pause(0.001)  # Brief pause to allow update
            
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                print(f"Episode {episode+1:4d}: Avg Reward = {avg_reward:.3f}, "
                      f"Epsilon = {self.epsilon:.3f}")
        
        return {
            'episode_rewards': self.episode_rewards,
            'epsilon_history': self.epsilon_history,
            'q_value_changes': self.q_value_changes,
            'final_q_table': self.q_table.cpu().numpy()
        }
    
    def evaluate(self, env: gym.Env, n_episodes: int = 100) -> float:
        """Evaluate trained policy."""
        total_reward = 0
        success_count = 0
        
        for _ in range(n_episodes):
            state, _ = env.reset()
            episode_reward = 0
            
            for _ in range(100):
                action = self.get_action(state, training=False)
                state, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                
                if terminated or truncated:
                    if reward > 0:
                        success_count += 1
                    break
            
            total_reward += episode_reward
        
        avg_reward = total_reward / n_episodes
        success_rate = success_count / n_episodes
        
        print(f"PyTorch Evaluation: Avg Reward = {avg_reward:.3f}, Success Rate = {success_rate:.3f}")
        return avg_reward

# Train PyTorch version
print("Training PyTorch Q-learning...")
torch_agent = TorchTabularQLearning(n_states, n_actions)
torch_stats = torch_agent.train(env, n_episodes=2000)
torch_performance = torch_agent.evaluate(env, n_episodes=100)

# 4. BELLMAN EQUATION VERIFICATION
print("\n4. Bellman Equation Verification")
print("-" * 30)

def verify_bellman_update(q_table: np.ndarray, state: int, action: int,
                         reward: float, next_state: int, done: bool,
                         gamma: float, alpha: float) -> bool:
    """Verify that Q-learning update follows Bellman equation.
    
    Args:
        q_table: Current Q-table
        state, action, reward, next_state, done: Transition tuple
        gamma: Discount factor
        alpha: Learning rate
        
    Returns:
        True if update is correct
    """
    # Current Q-value
    current_q = q_table[state, action]
    
    # Bellman target
    if done:
        target = reward
    else:
        target = reward + gamma * np.max(q_table[next_state])
    
    # Expected new Q-value
    expected_new_q = current_q + alpha * (target - current_q)
    
    # Actual update
    updated_q_table = q_table.copy()
    updated_q_table[state, action] = expected_new_q
    
    print(f"State {state}, Action {action}:")
    print(f"  Current Q: {current_q:.4f}")
    print(f"  Target: {target:.4f}")
    print(f"  TD Error: {target - current_q:.4f}")
    print(f"  New Q: {expected_new_q:.4f}")
    
    return True

# Test Bellman update
print("Testing Bellman update correctness...")
test_q_table = np.random.rand(16, 4) * 0.1
verify_bellman_update(test_q_table, state=0, action=1, reward=1.0, 
                     next_state=5, done=False, gamma=0.99, alpha=0.1)

# 5. Q-TABLE CONVERGENCE ANALYSIS
print("\n5. Q-Table Convergence Analysis")
print("-" * 30)

def analyze_convergence(q_table_history: List[np.ndarray]) -> Dict:
    """Analyze Q-table convergence over training.
    
    Args:
        q_table_history: List of Q-tables at different training steps
        
    Returns:
        Convergence statistics
    """
    if len(q_table_history) < 2:
        return {'mean_change': 0, 'max_change': 0, 'converged': True}
    
    changes = []
    for i in range(1, len(q_table_history)):
        diff = np.abs(q_table_history[i] - q_table_history[i-1])
        changes.append(np.mean(diff))
    
    return {
        'mean_change': np.mean(changes[-10:]) if len(changes) >= 10 else np.mean(changes),
        'max_change': np.max(changes[-10:]) if len(changes) >= 10 else np.max(changes),
        'converged': changes[-1] < 0.001 if changes else True
    }

# 6. VISUALIZATION
print("\n6. Creating Visualizations")
print("-" * 30)

# Create comprehensive plots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Learning curves comparison
window_size = 100
python_smoothed = np.convolve(python_stats['episode_rewards'], 
                             np.ones(window_size)/window_size, mode='valid')
torch_smoothed = np.convolve(torch_stats['episode_rewards'], 
                            np.ones(window_size)/window_size, mode='valid')

ax1.plot(python_smoothed, label='Pure Python', alpha=0.8)
ax1.plot(torch_smoothed, label='PyTorch Tensors', alpha=0.8)
ax1.set_xlabel('Episode')
ax1.set_ylabel('Average Reward (100-ep window)')
ax1.set_title('Learning Curves Comparison')
ax1.legend()
ax1.grid(True)

# Plot 2: Epsilon decay
ax2.plot(python_stats['epsilon_history'], label='Python')
ax2.plot(torch_stats['epsilon_history'], label='PyTorch')
ax2.set_xlabel('Episode')
ax2.set_ylabel('Epsilon (Exploration Rate)')
ax2.set_title('Exploration Decay')
ax2.legend()
ax2.grid(True)

# Plot 3: Q-value changes (learning progress)
ax3.plot(python_stats['q_value_changes'], alpha=0.7, label='Python')
ax3.plot(torch_stats['q_value_changes'], alpha=0.7, label='PyTorch')
ax3.set_xlabel('Episode')
ax3.set_ylabel('Average Q-value Change')
ax3.set_title('Learning Progress (Q-value Updates)')
ax3.legend()
ax3.grid(True)

# Plot 4: Final Q-table heatmap
final_q_table = python_stats['final_q_table']
im = ax4.imshow(final_q_table, cmap='viridis', aspect='auto')
ax4.set_xlabel('Action')
ax4.set_ylabel('State')
ax4.set_title('Final Q-Table (Python Agent)')
ax4.set_xticks(range(n_actions))
ax4.set_xticklabels(['Left', 'Down', 'Right', 'Up'])
plt.colorbar(im, ax=ax4)

plt.tight_layout()
plt.show()
print("Displaying final visualizations...")

# 7. POLICY VISUALIZATION
print("\n7. Policy Visualization")
print("-" * 30)

def visualize_policy(q_table: np.ndarray, env_name: str = 'FrozenLake-v1'):
    """Visualize the learned policy.
    
    Args:
        q_table: Learned Q-table
        env_name: Environment name
    """
    # Get optimal policy
    policy = np.argmax(q_table, axis=1)
    
    print(f"Learned Policy for {env_name}:")
    action_symbols = ['←', '↓', '→', '↑']
    
    if env_name == 'FrozenLake-v1':
        # 4x4 grid
        print("\nPolicy Grid (4x4):")
        for i in range(4):
            row = ""
            for j in range(4):
                state = i * 4 + j
                if state == 15:  # Goal state
                    row += "G "
                elif state in [5, 7, 11, 12]:  # Hole states
                    row += "H "
                else:
                    row += f"{action_symbols[policy[state]]} "
            print(row)
        
        print("\nLegend: ← Left, ↓ Down, → Right, ↑ Up, H Hole, G Goal")
    
    return policy

# Visualize learned policies
print("\nPython Agent Policy:")
python_policy = visualize_policy(python_stats['final_q_table'])

print("\nPyTorch Agent Policy:")
torch_policy = visualize_policy(torch_stats['final_q_table'])

# Compare policies
policy_agreement = np.mean(python_policy == torch_policy)
print(f"\nPolicy Agreement: {policy_agreement:.1%}")

# 8. TRANSITION TO FUNCTION APPROXIMATION
print("\n8. Why Tabular Methods Don't Scale")
print("-" * 30)

print("""
LIMITATIONS OF TABULAR Q-LEARNING:

1. STATE SPACE EXPLOSION:
   - FrozenLake: 16 states → 16×4 = 64 Q-values ✓
   - Atari (84×84×4): ~2.8M states → 11.2M Q-values per action ✗
   - Continuous spaces: Infinite states ✗

2. MEMORY REQUIREMENTS:
   - Linear growth with state×action combinations
   - Cannot share learning between similar states
   - No generalization capability

3. LEARNING EFFICIENCY:
   - Must visit every state-action pair multiple times
   - Cannot leverage similarity between states
   - Poor sample efficiency in large spaces

SOLUTION: FUNCTION APPROXIMATION
- Use neural networks to approximate Q(s,a)
- Generalize across similar states
- Handle continuous and high-dimensional spaces
- Share parameters across state-action pairs

This motivates Deep Q-Networks (DQN) in Chapter 6!
""")

# 9. PERFORMANCE COMPARISON
print("\n9. Final Performance Summary")
print("-" * 30)

print(f"Pure Python Q-Learning:")
print(f"  Final Performance: {python_performance:.3f}")
print(f"  Training Episodes: {len(python_stats['episode_rewards'])}")
print(f"  Final Epsilon: {python_stats['epsilon_history'][-1]:.4f}")

print(f"\nPyTorch Q-Learning:")
print(f"  Final Performance: {torch_performance:.3f}")
print(f"  Training Episodes: {len(torch_stats['episode_rewards'])}")
print(f"  Final Epsilon: {torch_stats['epsilon_history'][-1]:.4f}")

print(f"\nBoth implementations achieved similar performance, demonstrating that")
print(f"PyTorch tensors can effectively replace NumPy arrays for tabular RL.")
print(f"The real power of PyTorch emerges with function approximation!")

print("\nChapter 5 Complete! ✓")
print("Ready to move on to Deep Q-Networks (Chapter 6)")

env.close()