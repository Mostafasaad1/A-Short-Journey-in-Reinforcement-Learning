# Chapter 9: Ways to Speed Up RL
# Optimization techniques for faster reinforcement learning

# Install required packages:
# !pip install torch gymnasium[all] numpy matplotlib tqdm -q

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.profiler import profile, record_function, ProfilerActivity
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import time
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Dict, Any, Optional
import psutil
import threading
from tqdm import tqdm

# Configure matplotlib for proper font rendering
plt.switch_backend("Agg")
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "PingFang SC", "Arial Unicode MS", "Hiragino Sans GB"]
plt.rcParams["axes.unicode_minus"] = False

print("Chapter 9: Ways to Speed Up RL")
print("=" * 50)

# 1. INTRODUCTION TO RL OPTIMIZATION
print("\n1. RL Performance Bottlenecks")
print("-" * 30)

print("""
COMMON RL BOTTLENECKS:

1. ENVIRONMENT INTERACTION:
   - Step-by-step environment execution
   - Single-threaded environment loops
   - I/O overhead in complex environments

2. NEURAL NETWORK COMPUTATION:
   - Forward/backward passes on CPU
   - Inefficient memory usage
   - Lack of batch processing

3. DATA TRANSFER:
   - CPU-GPU memory transfers
   - Serialization/deserialization overhead
   - Poor memory locality

4. TRAINING PIPELINE:
   - Sequential data collection and training
   - Underutilized parallel processing
   - Inefficient hyperparameter choices

OPTIMIZATION STRATEGIES:
1. GPU Acceleration
2. Vectorized Environments
3. JIT Compilation
4. Automatic Mixed Precision (AMP)
5. Parallel Data Collection
6. Efficient Memory Management
7. Profiling and Monitoring
""")

# 2. GPU ACCELERATION
print("\n2. GPU Acceleration")
print("-" * 30)

# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("CUDA not available - using CPU")

class OptimizedDQN(nn.Module):
    """GPU-optimized DQN network with efficient operations."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(OptimizedDQN, self).__init__()
        
        # Use efficient layer types
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(inplace=True),  # In-place operations save memory
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Initialize weights efficiently
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Efficient weight initialization."""
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with efficient tensor operations."""
        return self.network(x)

# Compare CPU vs GPU performance
def benchmark_gpu_cpu(state_dim: int = 4, action_dim: int = 2, 
                     batch_size: int = 1024, n_iterations: int = 100):
    """Benchmark GPU vs CPU performance."""
    
    # Create models
    cpu_model = OptimizedDQN(state_dim, action_dim).to('cpu')
    
    results = {'cpu_time': 0, 'gpu_time': 0, 'speedup': 1.0}
    
    # Generate test data
    test_input = torch.randn(batch_size, state_dim)
    
    # CPU benchmark
    cpu_input = test_input.to('cpu')
    start_time = time.time()
    
    for _ in range(n_iterations):
        with torch.no_grad():
            _ = cpu_model(cpu_input)
    
    cpu_time = time.time() - start_time
    results['cpu_time'] = cpu_time
    
    # GPU benchmark (if available)
    if torch.cuda.is_available():
        gpu_model = OptimizedDQN(state_dim, action_dim).to('cuda')
        gpu_input = test_input.to('cuda')
        
        # Warm up GPU
        for _ in range(10):
            _ = gpu_model(gpu_input)
        torch.cuda.synchronize()
        
        start_time = time.time()
        for _ in range(n_iterations):
            with torch.no_grad():
                _ = gpu_model(gpu_input)
        torch.cuda.synchronize()
        
        gpu_time = time.time() - start_time
        results['gpu_time'] = gpu_time
        results['speedup'] = cpu_time / gpu_time if gpu_time > 0 else 1.0
    
    return results

print("Benchmarking GPU vs CPU performance...")
benchmark_results = benchmark_gpu_cpu()
print(f"CPU time: {benchmark_results['cpu_time']:.4f}s")
print(f"GPU time: {benchmark_results['gpu_time']:.4f}s")
print(f"GPU speedup: {benchmark_results['speedup']:.2f}x")

# 3. VECTORIZED ENVIRONMENTS
print("\n3. Vectorized Environments")
print("-" * 30)

class VectorizedEnv:
    """Simple vectorized environment wrapper.
    
    Runs multiple environment instances in parallel to improve
    data collection efficiency.
    """
    
    def __init__(self, env_fns: List[callable]):
        """Initialize vectorized environment.
        
        Args:
            env_fns: List of functions that create environments
        """
        self.envs = [env_fn() for env_fn in env_fns]
        self.n_envs = len(self.envs)
        
        # Get environment properties
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
    
    def reset(self) -> np.ndarray:
        """Reset all environments.
        
        Returns:
            Stacked initial observations
        """
        obs = []
        for env in self.envs:
            ob, _ = env.reset()
            obs.append(ob)
        return np.array(obs)
    
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Step all environments.
        
        Args:
            actions: Array of actions for each environment
            
        Returns:
            Tuple of (observations, rewards, terminated, truncated)
        """
        observations = []
        rewards = []
        terminated = []
        truncated = []
        
        for env, action in zip(self.envs, actions):
            obs, reward, term, trunc, _ = env.step(action)
            
            # Auto-reset if episode finished
            if term or trunc:
                obs, _ = env.reset()
            
            observations.append(obs)
            rewards.append(reward)
            terminated.append(term)
            truncated.append(trunc)
        
        return np.array(observations), np.array(rewards), np.array(terminated), np.array(truncated)
    
    def close(self):
        """Close all environments."""
        for env in self.envs:
            env.close()

# Compare single vs vectorized environment performance
def benchmark_vectorized_envs(n_envs: int = 4, n_steps: int = 1000):
    """Benchmark single vs vectorized environment performance."""
    
    # Single environment
    env = gym.make('CartPole-v1')
    obs, info = env.reset()

    start_time = time.time()
    for _ in range(n_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()
    single_time = time.time() - start_time
    env.close()
    
    # Vectorized environments
    env_fns = [lambda: gym.make('CartPole-v1') for _ in range(n_envs)]
    vec_env = VectorizedEnv(env_fns)
    
    start_time = time.time()
    vec_env.reset()
    for _ in range(n_steps // n_envs):
        actions = np.array([vec_env.action_space.sample() for _ in range(n_envs)])
        vec_env.step(actions)
    vectorized_time = time.time() - start_time
    vec_env.close()
    
    speedup = single_time / vectorized_time if vectorized_time > 0 else 1.0
    
    return {
        'single_time': single_time,
        'vectorized_time': vectorized_time,
        'speedup': speedup,
        'n_envs': n_envs
    }

print("Benchmarking vectorized environments...")
vec_benchmark = benchmark_vectorized_envs(n_envs=4, n_steps=1000)
print(f"Single env time: {vec_benchmark['single_time']:.4f}s")
print(f"Vectorized time: {vec_benchmark['vectorized_time']:.4f}s")
print(f"Vectorization speedup: {vec_benchmark['speedup']:.2f}x")

# 4. JIT COMPILATION
print("\n4. JIT Compilation with TorchScript")
print("-" * 30)

class JITOptimizedDQN(nn.Module):
    """DQN network optimized for JIT compilation."""
    
    def __init__(self, state_dim: int, action_dim: int):
        super(JITOptimizedDQN, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass optimized for JIT."""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

def benchmark_jit_compilation(state_dim: int = 4, action_dim: int = 2,
                            batch_size: int = 256, n_iterations: int = 1000):
    """Benchmark JIT vs regular PyTorch model."""
    
    # Regular model
    regular_model = JITOptimizedDQN(state_dim, action_dim).to(device)
    
    # JIT compiled model
    jit_model = torch.jit.script(regular_model)
    
    # Test data
    test_input = torch.randn(batch_size, state_dim).to(device)
    
    # Warm up
    for _ in range(10):
        _ = regular_model(test_input)
        _ = jit_model(test_input)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Benchmark regular model
    start_time = time.time()
    for _ in range(n_iterations):
        with torch.no_grad():
            _ = regular_model(test_input)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    regular_time = time.time() - start_time
    
    # Benchmark JIT model
    start_time = time.time()
    for _ in range(n_iterations):
        with torch.no_grad():
            _ = jit_model(test_input)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    jit_time = time.time() - start_time
    
    speedup = regular_time / jit_time if jit_time > 0 else 1.0
    
    return {
        'regular_time': regular_time,
        'jit_time': jit_time,
        'speedup': speedup
    }

print("Benchmarking JIT compilation...")
jit_benchmark = benchmark_jit_compilation()
print(f"Regular model time: {jit_benchmark['regular_time']:.4f}s")
print(f"JIT model time: {jit_benchmark['jit_time']:.4f}s")
print(f"JIT speedup: {jit_benchmark['speedup']:.2f}x")

# 5. AUTOMATIC MIXED PRECISION (AMP)
print("\n5. Automatic Mixed Precision (AMP)")
print("-" * 30)

class AMPOptimizedAgent:
    """DQN agent with Automatic Mixed Precision support."""
    
    def __init__(self, state_dim: int, action_dim: int, use_amp: bool = True):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.use_amp = use_amp and torch.cuda.is_available()
        
        # Network
        self.q_network = OptimizedDQN(state_dim, action_dim).to(device)
        self.target_network = OptimizedDQN(state_dim, action_dim).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)
        
        # AMP components
        if self.use_amp:
            self.scaler = GradScaler()
        
        # Replay buffer
        self.replay_buffer = deque(maxlen=10000)
        
        self.steps = 0
        self.losses = []
    
    def train_step(self, batch_size: int = 32) -> Optional[float]:
        """Training step with optional AMP."""
        if len(self.replay_buffer) < batch_size:
            return None
        
        # Sample batch
        batch = random.sample(self.replay_buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.BoolTensor(dones).to(device)
        
        if self.use_amp:
            # Training with AMP
            with autocast():
                current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
                
                with torch.no_grad():
                    next_q_values = self.target_network(next_states).max(1)[0]
                    target_q_values = rewards + (0.99 * next_q_values * ~dones)
                
                loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
            
            # Backward pass with scaling
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # Regular training
            current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
            
            with torch.no_grad():
                next_q_values = self.target_network(next_states).max(1)[0]
                target_q_values = rewards + (0.99 * next_q_values * ~dones)
            
            loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        return loss.item()

def benchmark_amp(n_iterations: int = 1000, batch_size: int = 64):
    """Benchmark AMP vs regular training."""
    
    if not torch.cuda.is_available():
        print("AMP requires CUDA - skipping benchmark")
        return {'regular_time': 0, 'amp_time': 0, 'speedup': 1.0}
    
    # Create agents
    regular_agent = AMPOptimizedAgent(4, 2, use_amp=False)
    amp_agent = AMPOptimizedAgent(4, 2, use_amp=True)
    
    # Generate dummy data
    for _ in range(1000):
        state = np.random.randn(4)
        action = np.random.randint(0, 2)
        reward = np.random.randn()
        next_state = np.random.randn(4)
        done = np.random.random() < 0.1
        
        regular_agent.replay_buffer.append((state, action, reward, next_state, done))
        amp_agent.replay_buffer.append((state, action, reward, next_state, done))
    
    # Benchmark regular training
    start_time = time.time()
    for _ in range(n_iterations):
        regular_agent.train_step(batch_size)
    torch.cuda.synchronize()
    regular_time = time.time() - start_time
    
    # Benchmark AMP training
    start_time = time.time()
    for _ in range(n_iterations):
        amp_agent.train_step(batch_size)
    torch.cuda.synchronize()
    amp_time = time.time() - start_time
    
    speedup = regular_time / amp_time if amp_time > 0 else 1.0
    
    return {
        'regular_time': regular_time,
        'amp_time': amp_time,
        'speedup': speedup
    }

print("Benchmarking Automatic Mixed Precision...")
amp_benchmark = benchmark_amp()
if amp_benchmark['regular_time'] > 0:
    print(f"Regular training time: {amp_benchmark['regular_time']:.4f}s")
    print(f"AMP training time: {amp_benchmark['amp_time']:.4f}s")
    print(f"AMP speedup: {amp_benchmark['speedup']:.2f}x")
else:
    print("AMP benchmark skipped (requires CUDA)")

# 6. PARALLEL DATA COLLECTION
print("\n6. Parallel Data Collection")
print("-" * 30)

def collect_episode_data(env_name: str, n_steps: int) -> List[Tuple]:
    """Collect episode data in separate process."""
    env = gym.make(env_name)
    data = []
    
    state, _ = env.reset()
    for _ in range(n_steps):
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        data.append((state.copy(), action, reward, next_state.copy(), done))
        
        if done:
            state, _ = env.reset()
        else:
            state = next_state
    
    env.close()
    return data

def benchmark_parallel_collection(n_workers: int = 4, n_steps_per_worker: int = 250):
    """Benchmark sequential vs parallel data collection."""
    
    env_name = 'CartPole-v1'
    total_steps = n_workers * n_steps_per_worker
    
    # Sequential collection
    start_time = time.time()
    sequential_data = collect_episode_data(env_name, total_steps)
    sequential_time = time.time() - start_time
    
    # Parallel collection
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(collect_episode_data, env_name, n_steps_per_worker) 
                  for _ in range(n_workers)]
        
        parallel_data = []
        for future in futures:
            parallel_data.extend(future.result())
    
    parallel_time = time.time() - start_time
    
    speedup = sequential_time / parallel_time if parallel_time > 0 else 1.0
    
    return {
        'sequential_time': sequential_time,
        'parallel_time': parallel_time,
        'speedup': speedup,
        'n_workers': n_workers
    }

print("Benchmarking parallel data collection...")
parallel_benchmark = benchmark_parallel_collection()
print(f"Sequential collection time: {parallel_benchmark['sequential_time']:.4f}s")
print(f"Parallel collection time: {parallel_benchmark['parallel_time']:.4f}s")
print(f"Parallel speedup: {parallel_benchmark['speedup']:.2f}x")

# 7. MEMORY OPTIMIZATION
print("\n7. Memory Optimization")
print("-" * 30)

class MemoryEfficientReplayBuffer:
    """Memory-efficient replay buffer with optimized storage."""
    
    def __init__(self, capacity: int, state_dim: int):
        self.capacity = capacity
        self.state_dim = state_dim
        self.position = 0
        self.size = 0
        
        # Pre-allocate arrays for better memory efficiency
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=bool)
    
    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool) -> None:
        """Add experience to buffer."""
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = done
        
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """Sample batch efficiently."""
        indices = np.random.choice(self.size, batch_size, replace=False)
        
        return (
            torch.from_numpy(self.states[indices]),
            torch.from_numpy(self.actions[indices]),
            torch.from_numpy(self.rewards[indices]),
            torch.from_numpy(self.next_states[indices]),
            torch.from_numpy(self.dones[indices])
        )
    
    def __len__(self) -> int:
        return self.size

def benchmark_memory_efficiency():
    """Compare memory usage of different buffer implementations."""
    
    capacity = 10000
    state_dim = 4
    
    # Measure memory usage
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Create memory-efficient buffer
    efficient_buffer = MemoryEfficientReplayBuffer(capacity, state_dim)
    
    # Fill buffer
    for _ in range(capacity):
        state = np.random.randn(state_dim).astype(np.float32)
        action = np.random.randint(0, 2)
        reward = np.random.randn()
        next_state = np.random.randn(state_dim).astype(np.float32)
        done = np.random.random() < 0.1
        
        efficient_buffer.push(state, action, reward, next_state, done)
    
    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_usage = final_memory - initial_memory
    
    return {
        'memory_usage_mb': memory_usage,
        'buffer_size': len(efficient_buffer)
    }

print("Analyzing memory efficiency...")
memory_benchmark = benchmark_memory_efficiency()
print(f"Memory usage: {memory_benchmark['memory_usage_mb']:.2f} MB")
print(f"Buffer size: {memory_benchmark['buffer_size']} experiences")
print(f"Memory per experience: {memory_benchmark['memory_usage_mb'] * 1024 / memory_benchmark['buffer_size']:.2f} KB")

# 8. PROFILING AND MONITORING
print("\n8. Performance Profiling")
print("-" * 30)

def profile_training_step():
    """Profile a training step to identify bottlenecks."""
    
    # Create agent and data
    agent = AMPOptimizedAgent(4, 2, use_amp=False)
    
    # Fill replay buffer
    for _ in range(1000):
        state = np.random.randn(4)
        action = np.random.randint(0, 2)
        reward = np.random.randn()
        next_state = np.random.randn(4)
        done = np.random.random() < 0.1
        agent.replay_buffer.append((state, action, reward, next_state, done))
    
    # Profile training
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], 
                record_shapes=True) as prof:
        with record_function("training_loop"):
            for _ in range(10):
                agent.train_step()
    
    # Print profiling results
    print("\nProfiling Results (Top 10 operations):")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    
    return prof

# Run profiling (comment out if too verbose)
# print("Running performance profiling...")
# profiling_results = profile_training_step()

# 9. COMPREHENSIVE BENCHMARK
print("\n9. Comprehensive Performance Comparison")
print("-" * 30)

# Collect all benchmark results
benchmark_summary = {
    'GPU Speedup': benchmark_results['speedup'],
    'Vectorized Envs': vec_benchmark['speedup'],
    'JIT Compilation': jit_benchmark['speedup'],
    'AMP Training': amp_benchmark['speedup'] if amp_benchmark['speedup'] > 0 else 1.0,
    'Parallel Collection': parallel_benchmark['speedup']
}

print("\nPerformance Optimization Summary:")
for optimization, speedup in benchmark_summary.items():
    print(f"{optimization:20s}: {speedup:.2f}x speedup")

# Calculate theoretical combined speedup (multiplicative)
combined_speedup = 1.0
for speedup in benchmark_summary.values():
    combined_speedup *= speedup

print(f"\nTheoretical combined speedup: {combined_speedup:.2f}x")

# 10. VISUALIZATION
print("\n10. Creating Performance Visualizations")
print("-" * 30)

# Create comprehensive performance plots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Speedup comparison
optimizations = list(benchmark_summary.keys())
speedups = list(benchmark_summary.values())

bars = ax1.bar(optimizations, speedups, alpha=0.7, 
               color=['blue', 'orange', 'green', 'red', 'purple'])
ax1.set_ylabel('Speedup (x)')
ax1.set_title('Performance Optimization Speedups')
ax1.tick_params(axis='x', rotation=45)
ax1.grid(True, alpha=0.3)

# Add speedup values on bars
for bar, speedup in zip(bars, speedups):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.05,
             f'{speedup:.2f}x', ha='center', va='bottom')

# Plot 2: Cumulative speedup (theoretical)
cumulative_speedups = []
current_speedup = 1.0
for speedup in speedups:
    current_speedup *= speedup
    cumulative_speedups.append(current_speedup)

ax2.plot(range(len(optimizations)), cumulative_speedups, 'o-', linewidth=2, markersize=8)
ax2.set_xlabel('Optimization Stage')
ax2.set_ylabel('Cumulative Speedup (x)')
ax2.set_title('Cumulative Performance Improvement')
ax2.set_xticks(range(len(optimizations)))
ax2.set_xticklabels([opt.split()[0] for opt in optimizations], rotation=45)
ax2.grid(True, alpha=0.3)

# Plot 3: Memory usage analysis
memory_types = ['Baseline', 'Optimized Buffer', 'GPU Memory']
memory_usage = [100, memory_benchmark['memory_usage_mb'], 
                torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0]

ax3.bar(memory_types, memory_usage, alpha=0.7, color=['gray', 'green', 'blue'])
ax3.set_ylabel('Memory Usage (MB)')
ax3.set_title('Memory Usage Comparison')
ax3.grid(True, alpha=0.3)

# Plot 4: Training time comparison
training_scenarios = ['Sequential\n(Baseline)', 'GPU\nAccelerated', 'Vectorized\nEnvironments', 
                     'JIT\nCompiled', 'Full\nOptimized']

# Simulate training times (relative to baseline)
baseline_time = 100  # seconds
training_times = [
    baseline_time,
    baseline_time / benchmark_results['speedup'],
    baseline_time / vec_benchmark['speedup'],
    baseline_time / jit_benchmark['speedup'],
    baseline_time / combined_speedup
]

bars = ax4.bar(training_scenarios, training_times, alpha=0.7,
               color=['red', 'orange', 'yellow', 'lightgreen', 'green'])
ax4.set_ylabel('Training Time (seconds)')
ax4.set_title('Training Time Reduction')
ax4.grid(True, alpha=0.3)

# Add time values on bars
for bar, time_val in zip(bars, training_times):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + 2,
             f'{time_val:.0f}s', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('pytorch_rl_tutorial/chapter_09_optimization_results.png', dpi=150, bbox_inches='tight')
print("Results saved to chapter_09_optimization_results.png")

# 11. OPTIMIZATION RECOMMENDATIONS
print("\n11. Optimization Recommendations")
print("-" * 30)

print("""
RL OPTIMIZATION BEST PRACTICES:

1. ALWAYS USE GPU WHEN AVAILABLE:
   ✓ Move models and tensors to GPU
   ✓ Use batch processing for efficiency
   ✓ Monitor GPU memory usage

2. VECTORIZED ENVIRONMENTS:
   ✓ 4-8 parallel environments for most tasks
   ✓ Balance between speedup and memory usage
   ✓ Use SubprocVecEnv for CPU-intensive environments

3. JIT COMPILATION:
   ✓ Compile models after architecture is finalized
   ✓ Especially beneficial for inference
   ✓ Test thoroughly after compilation

4. AUTOMATIC MIXED PRECISION:
   ✓ Enable for modern GPUs (Volta, Turing, Ampere)
   ✓ Monitor for numerical instability
   ✓ Significant memory savings

5. MEMORY OPTIMIZATION:
   ✓ Pre-allocate buffers with appropriate dtypes
   ✓ Use in-place operations when possible
   ✓ Clear unnecessary variables

6. PROFILING:
   ✓ Regular profiling to identify bottlenecks
   ✓ Focus on hot paths in training loop
   ✓ Monitor both CPU and GPU utilization

PROJECTED SPEEDUP:
- Single optimization: 1.5-3x speedup
- Combined optimizations: 5-20x speedup
- Depends heavily on hardware and environment
""")

print(f"\nOptimization Impact Summary:")
print(f"Total theoretical speedup: {combined_speedup:.1f}x")
print(f"Memory optimization: {memory_benchmark['memory_usage_mb']:.1f}MB for 10k experiences")
print(f"Training time reduction: {(1 - 1/combined_speedup)*100:.0f}%")

print("\nChapter 9 Complete! ✓")
print("RL training significantly accelerated through optimization")
print("Ready to explore practical applications (Chapter 10)")
