# Chapter 7: Higher-Level RL Libraries
# Using Stable Baselines3 for rapid prototyping and comparison

# Install required packages:
# !pip install stable-baselines3[extra] gymnasium[all] torch tensorboard matplotlib -q

import gymnasium as gym
import torch
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.dqn.policies import QNetwork
import torch.nn as nn
import time
from typing import Dict, List, Tuple, Any
import os
from pathlib import Path

# Configure matplotlib for proper font rendering
plt.switch_backend("Agg")
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "PingFang SC", "Arial Unicode MS", "Hiragino Sans GB"]
plt.rcParams["axes.unicode_minus"] = False

print("Chapter 7: Higher-Level RL Libraries")
print("=" * 50)

# 1. INTRODUCTION TO STABLE BASELINES3
print("\n1. Introduction to Stable Baselines3")
print("-" * 30)

print("""
STABLE BASELINES3 (SB3) provides:

1. PRODUCTION-READY ALGORITHMS:
   - DQN, Double DQN, Dueling DQN
   - PPO, A2C, SAC, TD3, DDPG
   - All with best practices built-in

2. KEY ABSTRACTIONS:
   - VecEnv: Vectorized environments for parallel training
   - Policies: Neural network architectures
   - Callbacks: Custom training hooks and logging
   - Wrappers: Environment modifications

3. ADVANTAGES:
   ✓ Well-tested implementations
   ✓ Extensive documentation
   ✓ Easy hyperparameter tuning
   ✓ Built-in logging and evaluation
   ✓ Custom network architectures
   ✓ GPU acceleration support

4. TRADE-OFFS:
   ✓ Rapid prototyping vs detailed control
   ✓ Abstraction vs understanding
   ✓ Convenience vs customization

Let's compare SB3 with our Chapter 6 DQN implementation!
""")

# 2. BASIC SB3 USAGE
print("\n2. Basic Stable Baselines3 Usage")
print("-" * 30)

# Create environment
env_name = 'CartPole-v1'
env = gym.make(env_name)

print(f"Environment: {env_name}")
print(f"Observation space: {env.observation_space}")
print(f"Action space: {env.action_space}")

# Create DQN agent with default settings
print("\nCreating SB3 DQN agent...")
sb3_dqn = DQN(
    "MlpPolicy",  # Use Multi-Layer Perceptron policy
    env,
    learning_rate=1e-3,
    buffer_size=10000,
    learning_starts=1000,
    batch_size=32,
    tau=1.0,  # Hard update (same as our implementation)
    gamma=0.99,
    train_freq=4,
    gradient_steps=1,
    target_update_interval=1000,
    exploration_fraction=0.1,
    exploration_initial_eps=1.0,
    exploration_final_eps=0.05,
    verbose=1,
    device='auto',
    seed=42
)

print(f"SB3 DQN policy: {sb3_dqn.policy}")
print(f"SB3 DQN device: {sb3_dqn.device}")

# Quick training
print("\nTraining SB3 DQN for 10,000 steps...")
start_time = time.time()
sb3_dqn.learn(total_timesteps=10000, progress_bar=True)
sb3_training_time = time.time() - start_time

print(f"SB3 training completed in {sb3_training_time:.2f} seconds")

# Evaluate SB3 agent
print("\nEvaluating SB3 DQN...")
sb3_mean_reward, sb3_std_reward = evaluate_policy(sb3_dqn, env, n_eval_episodes=10)
print(f"SB3 DQN: {sb3_mean_reward:.2f} +/- {sb3_std_reward:.2f}")

# 3. CUSTOM NETWORK ARCHITECTURES
print("\n3. Custom Network Architectures")
print("-" * 30)

class CustomDQNNetwork(nn.Module):
    """Custom DQN network architecture compatible with SB3."""
    
    def __init__(self, observation_space: gym.Space, action_space: gym.Space, 
                 features_dim: int = 256):
        super(CustomDQNNetwork, self).__init__()
        
        # Extract dimensions
        self.input_dim = observation_space.shape[0]
        self.n_actions = action_space.n
        self.features_dim = features_dim
        
        # Feature extraction layers
        self.features_extractor = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, features_dim),
            nn.ReLU()
        )
        
        # Q-value head
        self.q_net = nn.Linear(features_dim, self.n_actions)
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        features = self.features_extractor(observations)
        return self.q_net(features)

class CustomFeaturesExtractor(BaseFeaturesExtractor):
    """Custom features extractor for SB3."""
    
    def __init__(self, observation_space: gym.Space, features_dim: int = 256):
        super(CustomFeaturesExtractor, self).__init__(observation_space, features_dim)
        
        n_input = observation_space.shape[0]
        self.net = nn.Sequential(
            nn.Linear(n_input, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, features_dim),
            nn.ReLU()
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations)

# Create DQN with custom architecture
print("Creating DQN with custom architecture...")

policy_kwargs = dict(
    features_extractor_class=CustomFeaturesExtractor,
    features_extractor_kwargs=dict(features_dim=256),
    net_arch=[128, 128]  # Additional layers after feature extraction
)

custom_dqn = DQN(
    "MlpPolicy",
    env,
    policy_kwargs=policy_kwargs,
    learning_rate=1e-3,
    buffer_size=10000,
    learning_starts=1000,
    batch_size=32,
    verbose=1,
    device='auto',
    seed=42
)

print(f"Custom DQN created with policy: {custom_dqn.policy}")

# Train custom DQN
print("\nTraining custom DQN...")
custom_dqn.learn(total_timesteps=10000, progress_bar=True)

# Evaluate custom DQN
custom_mean_reward, custom_std_reward = evaluate_policy(custom_dqn, env, n_eval_episodes=10)
print(f"Custom DQN: {custom_mean_reward:.2f} +/- {custom_std_reward:.2f}")

# 4. VECTORIZED ENVIRONMENTS
print("\n4. Vectorized Environments")
print("-" * 30)

# Create vectorized environment for parallel training
print("Creating vectorized environment...")
n_envs = 4
vec_env = make_vec_env(env_name, n_envs=n_envs, seed=42)

print(f"Created {n_envs} parallel environments")
print(f"Vectorized env type: {type(vec_env)}")

# Create DQN for vectorized training
vec_dqn = DQN(
    "MlpPolicy",
    vec_env,
    learning_rate=1e-3,
    buffer_size=40000,  # Larger buffer for multiple envs
    learning_starts=1000,
    batch_size=32,
    verbose=1,
    device='auto',
    seed=42
)

# Train with vectorized environments
print("\nTraining with vectorized environments...")
start_time = time.time()
vec_dqn.learn(total_timesteps=40000, progress_bar=True)  # 4x environments
vec_training_time = time.time() - start_time

print(f"Vectorized training completed in {vec_training_time:.2f} seconds")

# Evaluate vectorized DQN
vec_mean_reward, vec_std_reward = evaluate_policy(vec_dqn, env, n_eval_episodes=10)
print(f"Vectorized DQN: {vec_mean_reward:.2f} +/- {vec_std_reward:.2f}")

# 5. CALLBACKS AND MONITORING
print("\n5. Callbacks and Advanced Training")
print("-" * 30)

# Create monitored environment
log_dir = "pytorch_rl_tutorial/sb3_logs/"
os.makedirs(log_dir, exist_ok=True)

monitored_env = Monitor(gym.make(env_name), log_dir)

# Stop training when reward threshold is reached
stop_callback = StopTrainingOnRewardThreshold(reward_threshold=400.0, verbose=1)

# Create evaluation callback
eval_callback = EvalCallback(
    monitored_env,
    best_model_save_path=log_dir + 'best_model/',
    log_path=log_dir + 'evaluations/',
    eval_freq=1000,
    deterministic=True,
    render=False,
    n_eval_episodes=5,
    callback_after_eval=stop_callback
)


# Combine callbacks
from stable_baselines3.common.callbacks import CallbackList
callback_list = CallbackList([eval_callback])

# Create DQN with callbacks
callback_dqn = DQN(
    "MlpPolicy",
    monitored_env,
    learning_rate=1e-3,
    buffer_size=10000,
    learning_starts=1000,
    batch_size=32,
    verbose=1,
    device='auto',
    seed=42,
    tensorboard_log=log_dir + 'tensorboard/'
)

print("Training with callbacks and monitoring...")
callback_dqn.learn(total_timesteps=20000, callback=callback_list, progress_bar=True)

# Evaluate callback DQN
callback_mean_reward, callback_std_reward = evaluate_policy(callback_dqn, monitored_env, n_eval_episodes=10)
print(f"Callback DQN: {callback_mean_reward:.2f} +/- {callback_std_reward:.2f}")

# 6. ALGORITHM COMPARISON
print("\n6. Algorithm Comparison")
print("-" * 30)

# Compare different algorithms on the same environment
algorithms = {
    'DQN': DQN,
    'PPO': PPO,
    'A2C': A2C
}

results = {}
training_times = {}

for name, AlgorithmClass in algorithms.items():
    print(f"\nTraining {name}...")
    
    # Create fresh environment
    test_env = gym.make(env_name, render_mode=None)
    
    # Create agent
    if name == 'DQN':
        agent = AlgorithmClass(
            "MlpPolicy", test_env, learning_rate=1e-3,
            buffer_size=10000, verbose=0, seed=42
        )
    else:  # PPO or A2C
        agent = AlgorithmClass(
            "MlpPolicy", test_env, learning_rate=3e-4,
            verbose=0, seed=42
        )
    
    # Train agent
    start_time = time.time()
    agent.learn(total_timesteps=15000)
    training_times[name] = time.time() - start_time
    
    # Evaluate agent
    mean_reward, std_reward = evaluate_policy(agent, test_env, n_eval_episodes=10)
    results[name] = (mean_reward, std_reward)
    
    print(f"{name}: {mean_reward:.2f} +/- {std_reward:.2f} (trained in {training_times[name]:.2f}s)")
    
    test_env.close()

# 7. EXTRACTING AND ANALYZING LEARNED NETWORKS
print("\n7. Network Analysis and Extraction")
print("-" * 30)

def analyze_network(model, env, n_samples: int = 1000):
    """Analyze learned Q-network behavior."""
    
    # Collect states and Q-values
    states = []
    q_values_list = []
    actions = []
    
    for _ in range(n_samples):
        # Get random state
        state, _ = env.reset()
        states.append(state)
        
        # Get Q-values
        obs_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = model.q_net(obs_tensor)
            q_values_list.append(q_values.squeeze().cpu().numpy())
            actions.append(q_values.argmax().item())
    
    states = np.array(states)
    q_values_array = np.array(q_values_list)
    actions = np.array(actions)
    
    return states, q_values_array, actions

# Analyze SB3 DQN
print("Analyzing SB3 DQN network...")
states, q_values, actions = analyze_network(sb3_dqn, env, n_samples=500)

print(f"State shape: {states.shape}")
print(f"Q-values shape: {q_values.shape}")
print(f"Action distribution: {np.bincount(actions)}")
print(f"Mean Q-values: {np.mean(q_values, axis=0)}")
print(f"Q-value std: {np.std(q_values, axis=0)}")

# 8. SAVING AND LOADING MODELS
print("\n8. Model Persistence")
print("-" * 30)

# Save models
model_dir = "pytorch_rl_tutorial/saved_models/"
os.makedirs(model_dir, exist_ok=True)

print("Saving models...")
sb3_dqn.save(model_dir + "sb3_dqn")
custom_dqn.save(model_dir + "custom_dqn")
vec_dqn.save(model_dir + "vec_dqn")

print("Models saved successfully!")

# Load and test saved model
print("Loading and testing saved model...")
loaded_dqn = DQN.load(model_dir + "sb3_dqn", env=env)
loaded_mean_reward, loaded_std_reward = evaluate_policy(loaded_dqn, env, n_eval_episodes=5)
print(f"Loaded DQN: {loaded_mean_reward:.2f} +/- {loaded_std_reward:.2f}")

# 9. VISUALIZATION AND COMPARISON
print("\n9. Results Visualization")
print("-" * 30)

# Create comprehensive comparison plots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Algorithm comparison
alg_names = list(results.keys())
alg_means = [results[name][0] for name in alg_names]
alg_stds = [results[name][1] for name in alg_names]

ax1.bar(alg_names, alg_means, yerr=alg_stds, capsize=5, alpha=0.7)
ax1.set_ylabel('Average Reward')
ax1.set_title('Algorithm Performance Comparison')
ax1.grid(True, alpha=0.3)

# Add performance numbers on bars
for i, (mean, std) in enumerate(zip(alg_means, alg_stds)):
    ax1.text(i, mean + std + 5, f'{mean:.1f}', ha='center', va='bottom')

# Plot 2: Training time comparison
time_names = list(training_times.keys())
time_values = list(training_times.values())

ax2.bar(time_names, time_values, alpha=0.7, color='orange')
ax2.set_ylabel('Training Time (seconds)')
ax2.set_title('Training Time Comparison')
ax2.grid(True, alpha=0.3)

# Add time numbers on bars
for i, time_val in enumerate(time_values):
    ax2.text(i, time_val + 0.5, f'{time_val:.1f}s', ha='center', va='bottom')

# Plot 3: Q-value distribution
ax3.hist(q_values.flatten(), bins=30, alpha=0.7, edgecolor='black')
ax3.set_xlabel('Q-value')
ax3.set_ylabel('Frequency')
ax3.set_title('Q-value Distribution (SB3 DQN)')
ax3.grid(True, alpha=0.3)

# Plot 4: Action-value relationship (for CartPole)
if env_name == 'CartPole-v1':
    # Plot cart position vs Q-values
    cart_pos = states[:, 0]
    q_diff = q_values[:, 1] - q_values[:, 0]  # Right - Left
    
    scatter = ax4.scatter(cart_pos, q_diff, alpha=0.6, c=actions, cmap='viridis')
    ax4.set_xlabel('Cart Position')
    ax4.set_ylabel('Q(Right) - Q(Left)')
    ax4.set_title('Learned Policy Visualization')
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    plt.colorbar(scatter, ax=ax4, label='Selected Action')

plt.tight_layout()
plt.savefig('pytorch_rl_tutorial/chapter_07_sb3_results.png', dpi=150, bbox_inches='tight')
print("Results saved to chapter_07_sb3_results.png")

# 10. COMPARISON WITH CHAPTER 6 IMPLEMENTATION
print("\n10. SB3 vs Custom Implementation Comparison")
print("-" * 30)

print("""
COMPARISON SUMMARY:

1. IMPLEMENTATION COMPLEXITY:
   ✓ SB3: ~10 lines of code for basic DQN
   ✓ Custom (Ch6): ~200+ lines for full implementation
   
2. PERFORMANCE:
   ✓ SB3: Optimized, well-tested algorithms
   ✓ Custom: Educational, full control over details
   
3. FLEXIBILITY:
   ✓ SB3: Easy hyperparameter tuning, custom networks
   ✓ Custom: Complete algorithmic control
   
4. DEBUGGING:
   ✓ SB3: Built-in logging, callbacks, monitoring
   ✓ Custom: Full visibility into training process
   
5. RESEARCH vs PRODUCTION:
   ✓ SB3: Perfect for rapid prototyping and production
   ✓ Custom: Better for understanding and novel research

RECOMMENDATIONS:
- Use SB3 for practical applications and baselines
- Implement custom algorithms for learning and research
- Combine both: SB3 for comparison, custom for innovation
""")

# Performance summary
print(f"\nPERFORMANCE SUMMARY:")
print(f"SB3 DQN (basic):      {sb3_mean_reward:.2f} ± {sb3_std_reward:.2f}")
print(f"SB3 DQN (custom):     {custom_mean_reward:.2f} ± {custom_std_reward:.2f}")
print(f"SB3 DQN (vectorized): {vec_mean_reward:.2f} ± {vec_std_reward:.2f}")
print(f"SB3 DQN (callbacks):  {callback_mean_reward:.2f} ± {callback_std_reward:.2f}")

print(f"\nTRAINING EFFICIENCY:")
print(f"Basic SB3 training:    {sb3_training_time:.2f} seconds (10k steps)")
print(f"Vectorized training:   {vec_training_time:.2f} seconds (40k steps, 4 envs)")
print(f"Speedup from vectorization: {(40000/10000) / (vec_training_time/sb3_training_time):.2f}x")

print("\nChapter 7 Complete! ✓")
print("Stable Baselines3 provides powerful abstractions for rapid RL development")
print("Ready to explore DQN extensions (Chapter 8)")

# Clean up
env.close()
vec_env.close()
monitored_env.close()
