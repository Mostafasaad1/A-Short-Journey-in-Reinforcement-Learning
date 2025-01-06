# Chapter 6: Deep Q-Networks (DQN)
# Deep reinforcement learning for high-dimensional state spaces

# Install required packages:
# !pip install torch gymnasium[atari] gymnasium[accept-rom-license] ale-py opencv-python matplotlib tqdm tensorboard -q

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque, namedtuple
import random
import cv2
from tqdm import tqdm
import os
from typing import Tuple, List, Optional,Dict

# Configure matplotlib for proper font rendering
plt.switch_backend("Agg")
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "PingFang SC", "Arial Unicode MS", "Hiragino Sans GB"]
plt.rcParams["axes.unicode_minus"] = False

print("Chapter 6: Deep Q-Networks (DQN)")
print("=" * 50)

# 1. INTRODUCTION TO DQN
print("\n1. From Tabular to Deep Q-Learning")
print("-" * 30)

print("""
DEEP Q-NETWORKS (DQN) solve the limitations of tabular Q-learning:

1. FUNCTION APPROXIMATION:
   - Replace Q-table with neural network: Q(s,a; θ)
   - Handle high-dimensional state spaces (images, continuous states)
   - Generalize across similar states
   - Share parameters for efficient learning

2. KEY INNOVATIONS:
   - Experience Replay: Break correlation between consecutive samples
   - Target Network: Stabilize learning with fixed Q-targets
   - Frame Preprocessing: Convert raw pixels to meaningful features
   - Error Clipping: Handle large Q-value updates gracefully

3. CHALLENGES ADDRESSED:
   - Non-stationary targets (moving Q-values)
   - Correlated samples (sequential gameplay)
   - Unstable learning (catastrophic forgetting)
   - Sparse rewards (delayed gratification)

4. DQN ALGORITHM:
   1. Store transitions in replay buffer
   2. Sample random batch for training
   3. Compute Q-targets using target network
   4. Update main network via gradient descent
   5. Periodically update target network
""")

# 2. FRAME PREPROCESSING
print("\n2. Frame Preprocessing for Atari")
print("-" * 30)

class FrameProcessor:
    """Preprocesses Atari frames for DQN training.
    
    Converts raw RGB frames to grayscale, resizes, and stacks consecutive frames
    to provide temporal information to the agent.
    """
    
    def __init__(self, frame_height: int = 84, frame_width: int = 84, 
                 frame_stack: int = 4):
        """Initialize frame processor.
        
        Args:
            frame_height: Height of processed frame
            frame_width: Width of processed frame
            frame_stack: Number of consecutive frames to stack
        """
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)
    
    def reset(self) -> None:
        """Reset frame buffer."""
        self.frames.clear()
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process single frame.
        
        Args:
            frame: Raw RGB frame from environment
            
        Returns:
            Processed grayscale frame
        """
        # Convert to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        else:
            gray = frame
        
        # Resize frame
        resized = cv2.resize(gray, (self.frame_width, self.frame_height), 
                           interpolation=cv2.INTER_AREA)
        
        # Normalize to [0, 1]
        normalized = resized.astype(np.float32) / 255.0
        
        return normalized
    
    def add_frame(self, frame: np.ndarray) -> np.ndarray:
        """Add frame to buffer and return stacked frames.
        
        Args:
            frame: Raw frame from environment
            
        Returns:
            Stacked frames ready for neural network input
        """
        processed = self.process_frame(frame)
        self.frames.append(processed)
        
        # Pad with first frame if buffer not full
        while len(self.frames) < self.frame_stack:
            self.frames.append(processed)
        
        # Stack frames: (height, width, channels)
        stacked = np.stack(self.frames, axis=0)
        return stacked

# Test frame processing
print("Testing frame processor...")
processor = FrameProcessor()
test_frame = np.random.randint(0, 255, (210, 160, 3), dtype=np.uint8)
processed = processor.add_frame(test_frame)
print(f"Original frame shape: {test_frame.shape}")
print(f"Processed frame shape: {processed.shape}")
print(f"Processed frame range: [{processed.min():.2f}, {processed.max():.2f}]")

# 3. EXPERIENCE REPLAY BUFFER
print("\n3. Experience Replay Buffer")
print("-" * 30)

# Named tuple for storing transitions
Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done'])

class ReplayBuffer:
    """Experience replay buffer for DQN.
    
    Stores agent experiences and provides random sampling to break correlation
    between consecutive experiences during training.
    """
    
    def __init__(self, capacity: int = 100000):
        """Initialize replay buffer.
        
        Args:
            capacity: Maximum number of experiences to store
        """
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
    
    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool) -> None:
        """Store experience in buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        experience = Transition(state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """Sample random batch of experiences.
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            Tuple of batched tensors (states, actions, rewards, next_states, dones)
        """
        batch = random.sample(self.buffer, batch_size)
        
        # Unpack batch
        states = torch.FloatTensor([e.state for e in batch])
        actions = torch.LongTensor([e.action for e in batch])
        rewards = torch.FloatTensor([e.reward for e in batch])
        next_states = torch.FloatTensor([e.next_state for e in batch])
        dones = torch.BoolTensor([e.done for e in batch])
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self) -> int:
        """Return current buffer size."""
        return len(self.buffer)

# Test replay buffer
print("Testing replay buffer...")
buffer = ReplayBuffer(capacity=1000)

# Add some dummy experiences
for i in range(10):
    state = np.random.rand(4, 84, 84)
    action = random.randint(0, 3)
    reward = random.random()
    next_state = np.random.rand(4, 84, 84)
    done = random.random() < 0.1
    buffer.push(state, action, reward, next_state, done)

print(f"Buffer size: {len(buffer)}")

# Sample batch
if len(buffer) >= 5:
    batch = buffer.sample(5)
    print(f"Batch shapes: states={batch[0].shape}, actions={batch[1].shape}")

# 4. DQN NETWORK ARCHITECTURE
print("\n4. DQN Network Architecture")
print("-" * 30)

class DQNNetwork(nn.Module):
    """Deep Q-Network for Atari games.
    
    Convolutional neural network that processes stacked frames and outputs
    Q-values for each possible action.
    """
    
    def __init__(self, input_channels: int = 4, n_actions: int = 4):
        """Initialize DQN network.
        
        Args:
            input_channels: Number of input channels (stacked frames)
            n_actions: Number of possible actions
        """
        super(DQNNetwork, self).__init__()
        
        # Convolutional layers for feature extraction
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calculate size of flattened features
        # For 84x84 input: 84 -> 20 -> 9 -> 7
        conv_out_size = self._get_conv_out_size(input_channels)
        
        # Fully connected layers
        self.fc1 = nn.Linear(conv_out_size, 512)
        self.fc2 = nn.Linear(512, n_actions)
        
        # Initialize weights
        self._initialize_weights()
    
    def _get_conv_out_size(self, input_channels: int) -> int:
        """Calculate output size of convolutional layers."""
        x = torch.zeros(1, input_channels, 84, 84)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return int(np.prod(x.size()))
    
    def _initialize_weights(self) -> None:
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through network.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Q-values for each action
        """
        # Convolutional feature extraction
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x

# Test DQN network
print("Testing DQN network...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

dqn = DQNNetwork(input_channels=4, n_actions=4).to(device)
test_input = torch.randn(1, 4, 84, 84).to(device)
output = dqn(test_input)
print(f"Network output shape: {output.shape}")
print(f"Q-values: {output.squeeze().detach().cpu().numpy()}")

# Count parameters
total_params = sum(p.numel() for p in dqn.parameters())
trainable_params = sum(p.numel() for p in dqn.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# 5. DQN AGENT IMPLEMENTATION
print("\n5. DQN Agent Implementation")
print("-" * 30)

class DQNAgent:
    """Deep Q-Network agent for reinforcement learning."""
    
    def __init__(self, state_shape: Tuple[int, ...], n_actions: int,
                 lr: float = 0.0001, gamma: float = 0.99,
                 epsilon: float = 1.0, epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01, buffer_size: int = 100000,
                 batch_size: int = 32, target_update_freq: int = 1000,
                 device: str = None):
        """Initialize DQN agent.
        
        Args:
            state_shape: Shape of state space (channels, height, width)
            n_actions: Number of possible actions
            lr: Learning rate
            gamma: Discount factor
            epsilon: Initial exploration rate
            epsilon_decay: Decay rate for epsilon
            epsilon_min: Minimum epsilon value
            buffer_size: Replay buffer capacity
            batch_size: Training batch size
            target_update_freq: How often to update target network
            device: Device to run on ('cpu' or 'cuda')
        """
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Device setup
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Neural networks
        self.q_network = DQNNetwork(state_shape[0], n_actions).to(self.device)
        self.target_network = DQNNetwork(state_shape[0], n_actions).to(self.device)
        
        # Copy weights to target network
        self.update_target_network()
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Tracking
        self.steps = 0
        self.episode_rewards = []
        self.losses = []
        self.epsilon_history = []
    
    def get_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select action using ε-greedy policy.
        
        Args:
            state: Current state
            training: Whether in training mode (affects exploration)
            
        Returns:
            Selected action
        """
        if training and random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        
        # Convert state to tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Get Q-values
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        
        # Return action with highest Q-value
        return q_values.argmax().item()
    
    def store_experience(self, state: np.ndarray, action: int, reward: float,
                        next_state: np.ndarray, done: bool) -> None:
        """Store experience in replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def update_target_network(self) -> None:
        """Copy weights from main network to target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def train_step(self) -> Optional[float]:
        """Perform one training step.
        
        Returns:
            Training loss if update performed, None otherwise
        """
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Move to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q-values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss (Huber loss for stability)
        loss = F.smooth_l1_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients for stability
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
    
    def save(self, filepath: str) -> None:
        """Save model checkpoint."""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps
        }, filepath)
    
    def load(self, filepath: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']

# 6. TRAINING ENVIRONMENT SETUP
print("\n6. Setting Up Training Environment")
print("-" * 30)

# Create environment (using simpler environment for demonstration)
env_name = 'ALE/Pong-v5'  # You can change to other Atari games

try:
    env = gym.make(env_name, render_mode=None)
    print(f"Environment: {env_name}")
except:
    # Fallback to CartPole if Atari not available
    print("Atari environment not available, using CartPole for demonstration")
    env_name = 'CartPole-v1'
    env = gym.make(env_name, render_mode=None)

print(f"Action space: {env.action_space}")
print(f"Observation space: {env.observation_space}")

# Initialize components
if 'Pong' in env_name or 'ALE' in env_name:
    # Atari game
    frame_processor = FrameProcessor()
    state_shape = (4, 84, 84)  # (channels, height, width)
    n_actions = env.action_space.n
else:
    # CartPole or other simple environment
    frame_processor = None
    state_shape = (env.observation_space.shape[0],)  # Flatten state
    n_actions = env.action_space.n

print(f"State shape: {state_shape}")
print(f"Number of actions: {n_actions}")

# 7. TRAINING LOOP
print("\n7. DQN Training Loop")
print("-" * 30)

def train_dqn(agent: DQNAgent, env: gym.Env, n_episodes: int = 1000,
              max_steps: int = 1000, save_interval: int = 100) -> Dict:
    """Train DQN agent.
    
    Args:
        agent: DQN agent to train
        env: Environment to train in
        n_episodes: Number of training episodes
        max_steps: Maximum steps per episode
        save_interval: How often to save checkpoints
        
    Returns:
        Training statistics
    """
    print(f"Starting DQN training for {n_episodes} episodes...")
    
    episode_rewards = []
    losses = []
    
    for episode in tqdm(range(n_episodes), desc="Training"):
        state, _ = env.reset()
        
        # Process initial state
        if frame_processor:
            frame_processor.reset()
            state = frame_processor.add_frame(state)
        else:
            state = state.astype(np.float32)
        
        episode_reward = 0
        episode_losses = []
        
        for step in range(max_steps):
            # Select action
            action = agent.get_action(state, training=True)
            
            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Process next state
            if frame_processor:
                next_state = frame_processor.add_frame(next_state)
            else:
                next_state = next_state.astype(np.float32)
            
            # Store experience
            agent.store_experience(state, action, reward, next_state, done)
            
            # Train agent
            loss = agent.train_step()
            if loss is not None:
                episode_losses.append(loss)
            
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        # Track statistics
        episode_rewards.append(episode_reward)
        agent.episode_rewards.append(episode_reward)
        agent.epsilon_history.append(agent.epsilon)
        
        if episode_losses:
            avg_loss = np.mean(episode_losses)
            losses.append(avg_loss)
            agent.losses.append(avg_loss)
        
        # Logging
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            avg_loss = np.mean(losses[-50:]) if losses else 0
            print(f"Episode {episode+1:4d}: Avg Reward = {avg_reward:7.2f}, "
                  f"Avg Loss = {avg_loss:.4f}, Epsilon = {agent.epsilon:.3f}")
        
        # Save checkpoint
        if (episode + 1) % save_interval == 0:
            os.makedirs('pytorch_rl_tutorial/checkpoints', exist_ok=True)
            agent.save(f'pytorch_rl_tutorial/checkpoints/dqn_episode_{episode+1}.pth')
    
    return {
        'episode_rewards': episode_rewards,
        'losses': losses,
        'epsilon_history': agent.epsilon_history
    }

# Create and train agent (reduced episodes for demonstration)
print("Creating DQN agent...")

if 'CartPole' in env_name:
    # Simple MLP for CartPole
    class SimpleDQN(nn.Module):
        def __init__(self, input_size: int, n_actions: int):
            super(SimpleDQN, self).__init__()
            self.fc1 = nn.Linear(input_size, 128)
            self.fc2 = nn.Linear(128, 128)
            self.fc3 = nn.Linear(128, n_actions)
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            return self.fc3(x)
    
    # Replace DQN network for CartPole
    agent = DQNAgent(state_shape=(4,), n_actions=n_actions, 
                     epsilon_decay=0.995, target_update_freq=100)
    agent.q_network = SimpleDQN(4, n_actions).to(device)
    agent.target_network = SimpleDQN(4, n_actions).to(device)
    agent.update_target_network()
    agent.optimizer = optim.Adam(agent.q_network.parameters(), lr=0.001)
else:
    # Full DQN for Atari
    agent = DQNAgent(state_shape=state_shape, n_actions=n_actions)

# Train for fewer episodes for demonstration
training_stats = train_dqn(agent, env, n_episodes=200, max_steps=500)

print(f"\nTraining completed!")
print(f"Final average reward: {np.mean(training_stats['episode_rewards'][-10:]):.2f}")
print(f"Final epsilon: {agent.epsilon:.4f}")

# 8. EVALUATION AND VISUALIZATION
print("\n8. Evaluation and Results")
print("-" * 30)

# Evaluate trained agent
def evaluate_agent(agent: DQNAgent, env: gym.Env, n_episodes: int = 10) -> float:
    """Evaluate trained agent performance."""
    total_reward = 0
    
    for episode in range(n_episodes):
        state, _ = env.reset()
        
        if frame_processor:
            frame_processor.reset()
            state = frame_processor.add_frame(state)
        else:
            state = state.astype(np.float32)
        
        episode_reward = 0
        
        for _ in range(500):  # Max steps
            action = agent.get_action(state, training=False)  # No exploration
            state, reward, terminated, truncated, _ = env.step(action)
            
            if frame_processor:
                state = frame_processor.add_frame(state)
            else:
                state = state.astype(np.float32)
            
            episode_reward += reward
            
            if terminated or truncated:
                break
        
        total_reward += episode_reward
    
    return total_reward / n_episodes

# Evaluate performance
final_performance = evaluate_agent(agent, env, n_episodes=10)
print(f"Final evaluation performance: {final_performance:.2f}")

# Create visualization
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Learning curve
window_size = 50
if len(training_stats['episode_rewards']) >= window_size:
    smoothed_rewards = np.convolve(training_stats['episode_rewards'], 
                                  np.ones(window_size)/window_size, mode='valid')
    ax1.plot(smoothed_rewards)
else:
    ax1.plot(training_stats['episode_rewards'])
ax1.set_xlabel('Episode')
ax1.set_ylabel('Episode Reward')
ax1.set_title('DQN Learning Curve')
ax1.grid(True)

# Plot 2: Training loss
if training_stats['losses']:
    ax2.plot(training_stats['losses'], alpha=0.7)
    if len(training_stats['losses']) >= 20:
        smoothed_loss = np.convolve(training_stats['losses'], 
                                   np.ones(20)/20, mode='valid')
        ax2.plot(smoothed_loss, linewidth=2, label='Smoothed')
        ax2.legend()
ax2.set_xlabel('Training Step')
ax2.set_ylabel('Loss')
ax2.set_title('Training Loss')
ax2.grid(True)

# Plot 3: Epsilon decay
ax3.plot(agent.epsilon_history)
ax3.set_xlabel('Episode')
ax3.set_ylabel('Epsilon (Exploration Rate)')
ax3.set_title('Exploration Decay')
ax3.grid(True)

# Plot 4: Q-value distribution (sample)
if len(agent.replay_buffer) > 0:
    # Sample some states and compute Q-values
    sample_states, _, _, _, _ = agent.replay_buffer.sample(min(32, len(agent.replay_buffer)))
    with torch.no_grad():
        sample_q_values = agent.q_network(sample_states.to(device))
    
    q_values_np = sample_q_values.cpu().numpy()
    ax4.hist(q_values_np.flatten(), bins=30, alpha=0.7)
    ax4.set_xlabel('Q-value')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Q-value Distribution')
    ax4.grid(True)
else:
    ax4.text(0.5, 0.5, 'No Q-values\navailable', ha='center', va='center',
             transform=ax4.transAxes)

plt.tight_layout()
plt.savefig('pytorch_rl_tutorial/chapter_06_dqn_results.png', dpi=150, bbox_inches='tight')
print("Results saved to chapter_06_dqn_results.png")

# 9. KEY INNOVATIONS EXPLAINED
print("\n9. DQN Innovations Explained")
print("-" * 30)

print("""
DQN KEY INNOVATIONS AND THEIR IMPACT:

1. EXPERIENCE REPLAY:
   ✓ Breaks correlation between consecutive samples
   ✓ More efficient use of experiences (reuse data)
   ✓ Smooths out learning curve
   ✓ Enables offline learning from stored experiences

2. TARGET NETWORK:
   ✓ Provides stable targets for Q-value updates
   ✓ Prevents moving target problem
   ✓ Reduces overestimation bias
   ✓ Updated periodically (not every step)

3. FRAME PREPROCESSING:
   ✓ Converts RGB to grayscale (reduces dimensionality)
   ✓ Resizes frames to standard size (84x84)
   ✓ Stacks frames to provide temporal information
   ✓ Normalizes pixel values to [0,1]

4. GRADIENT CLIPPING:
   ✓ Prevents exploding gradients
   ✓ Stabilizes training
   ✓ Improves convergence

5. HUBER LOSS:
   ✓ Less sensitive to outliers than MSE
   ✓ Combines benefits of L1 and L2 loss
   ✓ More stable training

THESE INNOVATIONS MADE DEEP RL PRACTICAL!
""")

print("\nChapter 6 Complete! ✓")
print(f"DQN achieved {final_performance:.2f} average reward on {env_name}")
print("Ready to explore high-level RL libraries (Chapter 7)")

env.close()
