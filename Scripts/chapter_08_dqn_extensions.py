# Chapter 8: DQN Extensions
# Double DQN, Dueling DQN, and Prioritized Experience Replay

# Install required packages:
# !pip install torch gymnasium[atari] gymnasium[accept-rom-license] ale-py opencv-python matplotlib tqdm -q

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque, namedtuple
import random
import heapq
from typing import Tuple, List, Optional, Union
from tqdm import tqdm
import math

# Configure matplotlib for proper font rendering
plt.switch_backend("Agg")
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "PingFang SC", "Arial Unicode MS", "Hiragino Sans GB"]
plt.rcParams["axes.unicode_minus"] = False

print("Chapter 8: DQN Extensions")
print("=" * 50)

# 1. INTRODUCTION TO DQN LIMITATIONS
print("\n1. Understanding DQN Limitations")
print("-" * 30)

print("""
VANILLA DQN PROBLEMS:

1. OVERESTIMATION BIAS:
   - Q-learning is known to overestimate action values
   - max operator in Bellman equation introduces positive bias
   - Can lead to suboptimal policies

2. POOR REPRESENTATION:
   - Single stream network may not separate state value from advantage
   - Mixing state-dependent and action-dependent values
   - Inefficient learning of state values

3. UNIFORM SAMPLING:
   - All experiences treated equally in replay buffer
   - Important transitions may be rarely sampled
   - Inefficient use of experience data

SOLUTIONS:
1. Double DQN: Decouple action selection from evaluation
2. Dueling DQN: Separate state value and advantage streams
3. Prioritized Experience Replay: Sample important transitions more frequently
""")

# 2. DOUBLE DQN IMPLEMENTATION
print("\n2. Double DQN Implementation")
print("-" * 30)

class DoubleDQNAgent:
    """Double DQN agent that decouples action selection from evaluation.
    
    Key insight: Use main network for action selection and target network for evaluation.
    This reduces overestimation bias present in vanilla DQN.
    """
    
    def __init__(self, state_dim: int, action_dim: int, lr: float = 0.001,
                 gamma: float = 0.99, epsilon: float = 1.0, epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01, buffer_size: int = 10000,
                 batch_size: int = 32, target_update_freq: int = 100,
                 device: str = None):
        """Initialize Double DQN agent."""
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        self.device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Networks
        self.q_network = self._build_network().to(self.device)
        self.target_network = self._build_network().to(self.device)
        self.update_target_network()
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Replay buffer
        self.replay_buffer = deque(maxlen=buffer_size)
        
        # Tracking
        self.steps = 0
        self.losses = []
        self.episode_rewards = []
        
    def _build_network(self) -> nn.Module:
        """Build Q-network."""
        return nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_dim)
        )
    
    def get_action(self, state: np.ndarray, training: bool = True) -> int:
        """ε-greedy action selection."""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return q_values.argmax().item()
    
    def store_experience(self, state: np.ndarray, action: int, reward: float,
                        next_state: np.ndarray, done: bool) -> None:
        """Store experience in replay buffer."""
        self.replay_buffer.append((state, action, reward, next_state, done))
    
    def update_target_network(self) -> None:
        """Update target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def train_step(self) -> Optional[float]:
        """Perform Double DQN training step."""
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample batch
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # Current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Double DQN target computation
        with torch.no_grad():
            # Action selection using main network
            next_actions = self.q_network(next_states).argmax(1)
            # Q-value evaluation using target network
            next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10)
        self.optimizer.step()
        
        # Update target network
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.update_target_network()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()

print("Double DQN implementation complete.")

# 3. DUELING DQN IMPLEMENTATION
print("\n3. Dueling DQN Implementation")
print("-" * 30)

class DuelingDQNNetwork(nn.Module):
    """Dueling DQN network architecture.
    
    Separates state value V(s) and advantage A(s,a) streams:
    Q(s,a) = V(s) + A(s,a) - mean(A(s,·))
    
    This allows better learning of state values, especially when
    action choice doesn't matter much.
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        """Initialize Dueling DQN network.
        
        Args:
            state_dim: State space dimension
            action_dim: Action space dimension
            hidden_dim: Hidden layer dimension
        """
        super(DuelingDQNNetwork, self).__init__()
        
        self.action_dim = action_dim
        
        # Shared feature extraction layers
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Value stream: V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Advantage stream: A(s,a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through dueling network.
        
        Args:
            x: Input state tensor
            
        Returns:
            Q-values computed as Q(s,a) = V(s) + A(s,a) - mean(A(s,·))
        """
        # Shared features
        features = self.feature_layer(x)
        
        # Separate streams
        values = self.value_stream(features)  # [batch_size, 1]
        advantages = self.advantage_stream(features)  # [batch_size, action_dim]
        
        # Combine streams using the dueling formula
        # Q(s,a) = V(s) + (A(s,a) - mean_a(A(s,a)))
        q_values = values + (advantages - advantages.mean(dim=1, keepdim=True))
        
        return q_values

class DuelingDQNAgent:
    """Dueling DQN agent implementation."""
    
    def __init__(self, state_dim: int, action_dim: int, lr: float = 0.001,
                 gamma: float = 0.99, epsilon: float = 1.0, epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01, buffer_size: int = 10000,
                 batch_size: int = 32, target_update_freq: int = 100,
                 device: str = None):
        """Initialize Dueling DQN agent."""
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        self.device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Dueling networks
        self.q_network = DuelingDQNNetwork(state_dim, action_dim).to(self.device)
        self.target_network = DuelingDQNNetwork(state_dim, action_dim).to(self.device)
        self.update_target_network()
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Replay buffer
        self.replay_buffer = deque(maxlen=buffer_size)
        
        # Tracking
        self.steps = 0
        self.losses = []
        self.episode_rewards = []
    
    def get_action(self, state: np.ndarray, training: bool = True) -> int:
        """ε-greedy action selection."""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return q_values.argmax().item()
    
    def store_experience(self, state: np.ndarray, action: int, reward: float,
                        next_state: np.ndarray, done: bool) -> None:
        """Store experience in replay buffer."""
        self.replay_buffer.append((state, action, reward, next_state, done))
    
    def update_target_network(self) -> None:
        """Update target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def train_step(self) -> Optional[float]:
        """Perform Dueling DQN training step."""
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample batch
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # Current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Target Q-values (can use Double DQN here too)
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10)
        self.optimizer.step()
        
        # Update target network
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.update_target_network()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()

print("Dueling DQN implementation complete.")

# 4. PRIORITIZED EXPERIENCE REPLAY
print("\n4. Prioritized Experience Replay Implementation")
print("-" * 30)

class SumTree:
    """Sum tree data structure for efficient prioritized sampling.
    
    A complete binary tree where each leaf stores a priority and each internal
    node stores the sum of priorities in its subtree.
    """
    
    def __init__(self, capacity: int):
        """Initialize sum tree.
        
        Args:
            capacity: Maximum number of experiences
        """
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)  # Internal nodes + leaves
        self.data = np.zeros(capacity, dtype=object)  # Actual experiences
        self.write = 0  # Current write position
        self.n_entries = 0  # Current number of experiences
    
    def _propagate(self, idx: int, change: float) -> None:
        """Propagate priority change up the tree."""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)
    
    def _retrieve(self, idx: int, s: float) -> int:
        """Retrieve sample index based on priority."""
        left = 2 * idx + 1
        right = left + 1
        
        if left >= len(self.tree):
            return idx
        
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])
    
    def total(self) -> float:
        """Return sum of all priorities."""
        return self.tree[0]
    
    def add(self, priority: float, data: object) -> None:
        """Add experience with given priority."""
        idx = self.write + self.capacity - 1
        
        self.data[self.write] = data
        self.update(idx, priority)
        
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        
        if self.n_entries < self.capacity:
            self.n_entries += 1
    
    def update(self, idx: int, priority: float) -> None:
        """Update priority of experience."""
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)
    
    def get(self, s: float) -> Tuple[int, float, object]:
        """Get experience by priority value.
        
        Args:
            s: Priority value to search for
            
        Returns:
            Tuple of (tree_index, priority, data)
        """
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataIdx])

class PrioritizedReplayBuffer:
    """Prioritized Experience Replay buffer.
    
    Samples experiences based on their TD-error magnitude,
    allowing more important transitions to be learned from more frequently.
    """
    
    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4,
                 beta_increment: float = 0.001, epsilon: float = 1e-6):
        """Initialize prioritized replay buffer.
        
        Args:
            capacity: Buffer capacity
            alpha: Prioritization exponent (0 = uniform, 1 = full prioritization)
            beta: Importance sampling exponent (0 = no correction, 1 = full correction)
            beta_increment: Beta increment per sampling
            epsilon: Small constant to avoid zero priorities
        """
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        self.max_priority = 1.0
    
    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool) -> None:
        """Add experience with maximum priority."""
        experience = (state, action, reward, next_state, done)
        priority = self.max_priority ** self.alpha
        self.tree.add(priority, experience)
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """Sample batch with priorities.
        
        Returns:
            Tuple of (states, actions, rewards, next_states, dones, indices, weights)
        """
        batch = []
        indices = []
        weights = []
        priorities = []
        
        # Calculate segment size for stratified sampling
        segment = self.tree.total() / batch_size
        
        for i in range(batch_size):
            # Sample from segment
            s = random.uniform(segment * i, segment * (i + 1))
            idx, priority, data = self.tree.get(s)
            
            batch.append(data)
            indices.append(idx)
            priorities.append(priority)
        
        # Calculate importance sampling weights
        min_prob = min(priorities) / self.tree.total()
        max_weight = (min_prob * self.tree.n_entries) ** (-self.beta)
        
        for priority in priorities:
            prob = priority / self.tree.total()
            weight = (prob * self.tree.n_entries) ** (-self.beta)
            weights.append(weight / max_weight)
        
        # Increment beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Convert to tensors
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            torch.FloatTensor(states),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(next_states),
            torch.BoolTensor(dones),
            indices,
            torch.FloatTensor(weights)
        )
    
    def update_priorities(self, indices: List[int], priorities: List[float]) -> None:
        """Update priorities of sampled experiences."""
        for idx, priority in zip(indices, priorities):
            priority = (abs(priority) + self.epsilon) ** self.alpha
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self) -> int:
        """Return current buffer size."""
        return self.tree.n_entries

class PrioritizedDQNAgent:
    """DQN agent with Prioritized Experience Replay."""
    
    def __init__(self, state_dim: int, action_dim: int, lr: float = 0.001,
                 gamma: float = 0.99, epsilon: float = 1.0, epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01, buffer_size: int = 10000,
                 batch_size: int = 32, target_update_freq: int = 100,
                 alpha: float = 0.6, beta: float = 0.4, device: str = None):
        """Initialize Prioritized DQN agent."""
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        self.device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Networks
        self.q_network = self._build_network().to(self.device)
        self.target_network = self._build_network().to(self.device)
        self.update_target_network()
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Prioritized replay buffer
        self.replay_buffer = PrioritizedReplayBuffer(buffer_size, alpha, beta)
        
        # Tracking
        self.steps = 0
        self.losses = []
        self.episode_rewards = []
    
    def _build_network(self) -> nn.Module:
        """Build Q-network."""
        return nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_dim)
        )
    
    def get_action(self, state: np.ndarray, training: bool = True) -> int:
        """ε-greedy action selection."""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return q_values.argmax().item()
    
    def store_experience(self, state: np.ndarray, action: int, reward: float,
                        next_state: np.ndarray, done: bool) -> None:
        """Store experience in prioritized replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def update_target_network(self) -> None:
        """Update target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def train_step(self) -> Optional[float]:
        """Perform Prioritized DQN training step."""
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample batch with priorities
        states, actions, rewards, next_states, dones, indices, weights = self.replay_buffer.sample(self.batch_size)
        
        # Move to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        weights = weights.to(self.device)
        
        # Current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Target Q-values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute TD errors
        td_errors = target_q_values - current_q_values.squeeze()
        
        # Weighted loss (importance sampling)
        loss = (weights * (td_errors ** 2)).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10)
        self.optimizer.step()
        
        # Update priorities
        priorities = abs(td_errors.detach().cpu().numpy()) + 1e-6
        self.replay_buffer.update_priorities(indices, priorities)
        
        # Update target network
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.update_target_network()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()

print("Prioritized Experience Replay implementation complete.")

# 5. TRAINING AND COMPARISON
print("\n5. Training and Comparison")
print("-" * 30)

# Create environment
env = gym.make('CartPole-v1' , render_mode=None)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
print(f"Environment: CartPole-v1")
print(f"State dimension: {state_dim}")
print(f"Action dimension: {action_dim}")

def train_agent(agent, env, n_episodes: int = 500, max_steps: int = 500) -> List[float]:
    """Train agent and return episode rewards."""
    episode_rewards = []
    
    for episode in range(n_episodes):
        state, _ = env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            action = agent.get_action(state, training=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.store_experience(state, action, reward, next_state, done)
            loss = agent.train_step()
            
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        agent.episode_rewards.append(episode_reward)
        
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode+1}: Avg Reward = {avg_reward:.2f}, Epsilon = {agent.epsilon:.3f}")
    
    return episode_rewards

# Train all variants
print("\nTraining Double DQN...")
double_dqn = DoubleDQNAgent(state_dim, action_dim, epsilon_decay=0.995)
double_rewards = train_agent(double_dqn, env, n_episodes=300)

print("\nTraining Dueling DQN...")
dueling_dqn = DuelingDQNAgent(state_dim, action_dim, epsilon_decay=0.995)
dueling_rewards = train_agent(dueling_dqn, env, n_episodes=300)

print("\nTraining Prioritized DQN...")
prioritized_dqn = PrioritizedDQNAgent(state_dim, action_dim, epsilon_decay=0.995)
prioritized_rewards = train_agent(prioritized_dqn, env, n_episodes=300)

# 6. EVALUATION AND ANALYSIS
print("\n6. Evaluation and Analysis")
print("-" * 30)

def evaluate_agent(agent, env, n_episodes: int = 20) -> float:
    """Evaluate agent performance."""
    total_reward = 0
    for _ in range(n_episodes):
        state, _ = env.reset()
        episode_reward = 0
        
        for _ in range(500):
            action = agent.get_action(state, training=False)
            state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            if terminated or truncated:
                break
        
        total_reward += episode_reward
    
    return total_reward / n_episodes

# Evaluate all agents
double_performance = evaluate_agent(double_dqn, env)
dueling_performance = evaluate_agent(dueling_dqn, env)
prioritized_performance = evaluate_agent(prioritized_dqn, env)

print(f"\nFinal Performance:")
print(f"Double DQN:      {double_performance:.2f}")
print(f"Dueling DQN:     {dueling_performance:.2f}")
print(f"Prioritized DQN: {prioritized_performance:.2f}")

# 7. VISUALIZATION
print("\n7. Creating Visualizations")
print("-" * 30)

# Create comprehensive comparison plots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Learning curves
window_size = 50

def smooth_curve(data, window_size):
    if len(data) >= window_size:
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')
    return data

double_smooth = smooth_curve(double_rewards, window_size)
dueling_smooth = smooth_curve(dueling_rewards, window_size)
prioritized_smooth = smooth_curve(prioritized_rewards, window_size)

ax1.plot(double_smooth, label='Double DQN', alpha=0.8, linewidth=2)
ax1.plot(dueling_smooth, label='Dueling DQN', alpha=0.8, linewidth=2)
ax1.plot(prioritized_smooth, label='Prioritized DQN', alpha=0.8, linewidth=2)
ax1.set_xlabel('Episode')
ax1.set_ylabel('Average Reward')
ax1.set_title('Learning Curves Comparison')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Final performance comparison
agents = ['Double DQN', 'Dueling DQN', 'Prioritized DQN']
performances = [double_performance, dueling_performance, prioritized_performance]

bars = ax2.bar(agents, performances, alpha=0.7, color=['blue', 'orange', 'green'])
ax2.set_ylabel('Average Reward')
ax2.set_title('Final Performance Comparison')
ax2.grid(True, alpha=0.3)

# Add values on bars
for bar, perf in zip(bars, performances):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 5,
             f'{perf:.1f}', ha='center', va='bottom')

# Plot 3: Value and Advantage visualization (for Dueling DQN)
if hasattr(dueling_dqn.q_network, 'value_stream'):
    # Sample some states and analyze value/advantage separation
    test_states = []
    values = []
    advantages = []
    
    for _ in range(100):
        state, _ = env.reset()
        test_states.append(state)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(dueling_dqn.device)
        with torch.no_grad():
            features = dueling_dqn.q_network.feature_layer(state_tensor)
            value = dueling_dqn.q_network.value_stream(features).item()
            advantage = dueling_dqn.q_network.advantage_stream(features).squeeze().cpu().numpy()
            
            values.append(value)
            advantages.append(advantage)
    
    values = np.array(values)
    advantages = np.array(advantages)
    
    # Plot value distribution
    ax3.hist(values, bins=20, alpha=0.7, label='State Values V(s)')
    ax3.set_xlabel('Value')
    ax3.set_ylabel('Frequency')
    ax3.set_title('State Value Distribution (Dueling DQN)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot advantage distribution
    advantage_diff = advantages[:, 1] - advantages[:, 0]  # Right - Left
    ax4.hist(advantage_diff, bins=20, alpha=0.7, color='orange', label='Advantage Difference')
    ax4.set_xlabel('A(s, Right) - A(s, Left)')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Advantage Distribution (Dueling DQN)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
else:
    ax3.text(0.5, 0.5, 'Value/Advantage\nAnalysis\nNot Available', 
             ha='center', va='center', transform=ax3.transAxes)
    ax4.text(0.5, 0.5, 'Advantage\nDistribution\nNot Available', 
             ha='center', va='center', transform=ax4.transAxes)

plt.tight_layout()
plt.savefig('pytorch_rl_tutorial/chapter_08_dqn_extensions_results.png', dpi=150, bbox_inches='tight')
print("Results saved to chapter_08_dqn_extensions_results.png")

# 8. ABLATION STUDY
print("\n8. Ablation Study Summary")
print("-" * 30)

print("""
DQN EXTENSIONS ANALYSIS:

1. DOUBLE DQN:
   ✓ Reduces overestimation bias
   ✓ More stable learning
   ✓ Better final performance in many environments
   ✓ Simple modification to vanilla DQN

2. DUELING DQN:
   ✓ Better state value estimation
   ✓ Faster learning when actions don't matter much
   ✓ More robust to action space size
   ✓ Separates "how good is this state" from "how good is this action"

3. PRIORITIZED EXPERIENCE REPLAY:
   ✓ More efficient use of experience data
   ✓ Faster learning on important transitions
   ✓ Better sample efficiency
   ✗ Added computational complexity
   ✗ Additional hyperparameters to tune

COMBINATION EFFECTS:
- Double + Dueling DQN often work well together
- PER can be combined with any DQN variant
- Rainbow DQN combines all these improvements
""")

print(f"\nPerformance Summary:")
print(f"Double DQN:      {double_performance:.2f} average reward")
print(f"Dueling DQN:     {dueling_performance:.2f} average reward")
print(f"Prioritized DQN: {prioritized_performance:.2f} average reward")

print("\nChapter 8 Complete! ✓")
print("Key DQN extensions implemented and compared")
print("Ready to explore optimization techniques (Chapter 9)")

env.close()
