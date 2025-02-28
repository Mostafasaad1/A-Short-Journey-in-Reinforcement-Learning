#!/usr/bin/env python3
"""
Chapter 17: Black-Box Optimizations in RL

This chapter demonstrates gradient-free optimization methods for RL, including
Covariance Matrix Adaptation Evolution Strategy (CMA-ES) and Genetic Algorithms.
These methods are particularly useful when gradients are unavailable or noisy.

Key concepts covered:
- CMA-ES for policy parameter optimization
- Genetic Algorithm implementations
- Population-based training strategies
- Comparison with gradient-based methods
- Parallel evaluation and scalability


"""

# Chapter 17: Black-Box Optimizations in RL
# Start by installing the required packages:
# !pip install torch gymnasium[box2d] cma numpy matplotlib -q

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
import multiprocessing as mp
from typing import List, Tuple, Dict, Any
import time
from dataclasses import dataclass

# Configure matplotlib for non-interactive mode
plt.switch_backend("Agg")
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "PingFang SC", "Arial Unicode MS", "Hiragino Sans GB"]
plt.rcParams["axes.unicode_minus"] = False

# CMA-ES Implementation
# Since the full CMA library might not be available, we'll implement a simplified version

class SimpleCMA:
    """Simplified Covariance Matrix Adaptation Evolution Strategy."""
    
    def __init__(self, initial_mean, initial_sigma=0.5, population_size=None):
        self.dimension = len(initial_mean)
        self.mean = np.array(initial_mean, dtype=np.float64)
        self.sigma = initial_sigma
        
        # Population size
        if population_size is None:
            self.population_size = 4 + int(3 * np.log(self.dimension))
        else:
            self.population_size = population_size
        
        # Selection parameters
        self.mu = self.population_size // 2
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = self.weights / np.sum(self.weights)
        self.mu_eff = 1.0 / np.sum(self.weights**2)
        
        # Adaptation parameters
        self.c_sigma = (self.mu_eff + 2) / (self.dimension + self.mu_eff + 5)
        self.d_sigma = 1 + 2 * max(0, np.sqrt((self.mu_eff - 1) / (self.dimension + 1)) - 1) + self.c_sigma
        self.c_c = (4 + self.mu_eff / self.dimension) / (self.dimension + 4 + 2 * self.mu_eff / self.dimension)
        self.c_1 = 2 / ((self.dimension + 1.3)**2 + self.mu_eff)
        self.c_mu = min(1 - self.c_1, 2 * (self.mu_eff - 2 + 1/self.mu_eff) / ((self.dimension + 2)**2 + self.mu_eff))
        
        # Dynamic parameters
        self.p_sigma = np.zeros(self.dimension)
        self.p_c = np.zeros(self.dimension)
        self.C = np.eye(self.dimension)
        self.eigen_eval = 0
        self.B = np.eye(self.dimension)
        self.D = np.ones(self.dimension)
        
        self.generation = 0
        self.fitness_history = []
        
    def ask(self):
        """Generate population of candidate solutions."""
        if self.generation % 10 == 0:  # Update eigendecomposition periodically
            self.D, self.B = np.linalg.eigh(self.C)
            self.D = np.sqrt(self.D)
        
        samples = []
        for _ in range(self.population_size):
            z = np.random.randn(self.dimension)
            y = self.B @ (self.D * z)
            x = self.mean + self.sigma * y
            samples.append(x)
        
        self._samples = np.array(samples)
        return self._samples
    
    def tell(self, fitness_values):
        """Update distribution based on fitness values."""
        fitness_values = np.array(fitness_values)
        
        # Sort by fitness (assuming maximization)
        idx = np.argsort(fitness_values)[::-1]
        
        # Select elite solutions
        elite_samples = self._samples[idx[:self.mu]]
        elite_fitness = fitness_values[idx[:self.mu]]
        
        # Update mean
        old_mean = self.mean.copy()
        self.mean = np.sum(self.weights[:, np.newaxis] * elite_samples, axis=0)
        
        # Update evolution paths
        y = (self.mean - old_mean) / self.sigma
        C_inv_sqrt = self.B @ np.diag(1.0 / self.D) @ self.B.T
        
        self.p_sigma = (1 - self.c_sigma) * self.p_sigma + \
                      np.sqrt(self.c_sigma * (2 - self.c_sigma) * self.mu_eff) * C_inv_sqrt @ y
        
        # Update step size
        self.sigma *= np.exp((self.c_sigma / self.d_sigma) * 
                            (np.linalg.norm(self.p_sigma) / np.sqrt(self.dimension) - 1))
        
        # Update covariance matrix
        h_sigma = int(np.linalg.norm(self.p_sigma) / 
                     np.sqrt(1 - (1 - self.c_sigma)**(2 * (self.generation + 1))) < 
                     1.4 + 2 / (self.dimension + 1))
        
        self.p_c = (1 - self.c_c) * self.p_c + \
                  h_sigma * np.sqrt(self.c_c * (2 - self.c_c) * self.mu_eff) * y
        
        # Rank-mu update
        delta_h_sigma = (1 - h_sigma) * self.c_c * (2 - self.c_c)
        
        self.C = (1 - self.c_1 - self.c_mu) * self.C + \
                self.c_1 * (np.outer(self.p_c, self.p_c) + delta_h_sigma * self.C)
        
        for i in range(self.mu):
            y_i = (elite_samples[i] - old_mean) / self.sigma
            self.C += self.c_mu * self.weights[i] * np.outer(y_i, y_i)
        
        self.generation += 1
        self.fitness_history.append(np.max(elite_fitness))
        
        return {
            "best_fitness": np.max(elite_fitness),
            "mean_fitness": np.mean(elite_fitness),
            "sigma": self.sigma,
            "generation": self.generation
        }


class GeneticAlgorithm:
    """Simple Genetic Algorithm implementation."""
    
    def __init__(self, dimension, population_size=50, mutation_rate=0.1, crossover_rate=0.8):
        self.dimension = dimension
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.generation = 0
        self.fitness_history = []
        
        # Initialize population
        self.population = [np.random.randn(dimension) for _ in range(population_size)]
        
    def evaluate_population(self, fitness_func, parallel=False):
        """Evaluate fitness of entire population."""
        if parallel:
            with mp.Pool() as pool:
                fitness_values = pool.map(fitness_func, self.population)
        else:
            fitness_values = [fitness_func(individual) for individual in self.population]
        
        return fitness_values
    
    def selection(self, fitness_values, num_parents):
        """Tournament selection."""
        parents = []
        
        for _ in range(num_parents):
            # Tournament size of 3
            tournament_idx = np.random.choice(len(fitness_values), 3, replace=False)
            tournament_fitness = [fitness_values[i] for i in tournament_idx]
            winner_idx = tournament_idx[np.argmax(tournament_fitness)]
            parents.append(self.population[winner_idx].copy())
        
        return parents
    
    def crossover(self, parent1, parent2):
        """Uniform crossover."""
        if np.random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        child1, child2 = parent1.copy(), parent2.copy()
        
        for i in range(len(parent1)):
            if np.random.random() < 0.5:
                child1[i], child2[i] = child2[i], child1[i]
        
        return child1, child2
    
    def mutation(self, individual):
        """Gaussian mutation."""
        mutated = individual.copy()
        
        for i in range(len(individual)):
            if np.random.random() < self.mutation_rate:
                mutated[i] += np.random.normal(0, 0.1)
        
        return mutated
    
    def evolve(self, fitness_values):
        """Evolve population for one generation."""
        # Selection
        num_parents = self.population_size // 2
        parents = self.selection(fitness_values, num_parents)
        
        # Create offspring
        offspring = []
        
        for i in range(0, len(parents) - 1, 2):
            child1, child2 = self.crossover(parents[i], parents[i + 1])
            offspring.extend([self.mutation(child1), self.mutation(child2)])
        
        # Fill remaining slots
        while len(offspring) < self.population_size:
            parent = random.choice(parents)
            offspring.append(self.mutation(parent))
        
        self.population = offspring[:self.population_size]
        self.generation += 1
        
        best_fitness = np.max(fitness_values)
        self.fitness_history.append(best_fitness)
        
        return {
            "best_fitness": best_fitness,
            "mean_fitness": np.mean(fitness_values),
            "generation": self.generation
        }


# Simplified BipedalWalker Environment
class SimpleBipedalWalker:
    """Simplified bipedal walker environment for demonstration."""
    
    def __init__(self):
        self.state_dim = 24
        self.action_dim = 4
        self.max_steps = 1600
        self.reset()
        
    def reset(self):
        """Reset environment."""
        self.step_count = 0
        self.state = np.random.randn(self.state_dim) * 0.1
        self.velocity = np.zeros(2)
        self.position = 0.0
        return self.state.copy()
    
    def step(self, action):
        """Take a step in the environment."""
        action = np.clip(action, -1, 1)  # Clip actions to valid range
        
        # Simplified dynamics
        self.velocity[0] += np.sum(action[:2]) * 0.1  # Forward velocity
        self.velocity[1] += np.sum(action[2:]) * 0.05  # Vertical velocity
        
        self.position += self.velocity[0] * 0.02
        
        # Update state
        self.state[:2] = self.velocity
        self.state[2] = self.position
        self.state[3:] += np.random.randn(self.state_dim - 3) * 0.01
        
        self.step_count += 1
        
        # Reward calculation
        reward = self.velocity[0] * 10  # Reward for moving forward
        reward -= np.sum(np.abs(action)) * 0.1  # Penalty for large actions
        reward -= abs(self.velocity[1]) * 5  # Penalty for vertical movement
        
        # Check termination
        done = self.step_count >= self.max_steps or abs(self.position) > 50
        
        return self.state.copy(), reward, done, {}
    
    def render(self):
        pass


class SimplePolicy(nn.Module):
    """Simple feedforward policy network."""
    
    def __init__(self, state_dim, action_dim, hidden_dims=[64, 64]):
        super(SimplePolicy, self).__init__()
        
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.Tanh()
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, action_dim))
        layers.append(nn.Tanh())  # Output in [-1, 1]
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, state):
        return self.network(state)
    
    def get_parameters_as_vector(self):
        """Get all parameters as a flat vector."""
        return torch.cat([p.flatten() for p in self.parameters()]).detach().numpy()
    
    def set_parameters_from_vector(self, vector):
        """Set parameters from a flat vector."""
        vector = torch.FloatTensor(vector)
        idx = 0
        
        for param in self.parameters():
            param_length = param.numel()
            param.data = vector[idx:idx + param_length].view(param.shape)
            idx += param_length


def evaluate_policy(policy_params, env_class=SimpleBipedalWalker, num_episodes=3, max_steps=1600):
    """Evaluate a policy with given parameters."""
    # Create policy and set parameters
    env = env_class()
    policy = SimplePolicy(env.state_dim, env.action_dim)
    policy.set_parameters_from_vector(policy_params)
    
    total_reward = 0
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                action = policy(state_tensor).squeeze().numpy()
            
            state, reward, done, _ = env.step(action)
            episode_reward += reward
            
            if done:
                break
        
        total_reward += episode_reward
    
    return total_reward / num_episodes


def train_with_cma_es(num_generations=100):
    """Train policy using CMA-ES."""
    print("=== CMA-ES Training ===")
    
    # Create sample policy to get parameter count
    env = SimpleBipedalWalker()
    sample_policy = SimplePolicy(env.state_dim, env.action_dim)
    param_count = len(sample_policy.get_parameters_as_vector())
    
    print(f"Policy parameters: {param_count}")
    print(f"State dimension: {env.state_dim}")
    print(f"Action dimension: {env.action_dim}")
    
    # Initialize CMA-ES
    initial_mean = np.zeros(param_count)
    cma = SimpleCMA(initial_mean, initial_sigma=1.0, population_size=50)
    
    best_fitness_history = []
    mean_fitness_history = []
    
    print("\nStarting CMA-ES optimization...")
    start_time = time.time()
    
    for generation in range(num_generations):
        # Generate candidate solutions
        candidates = cma.ask()
        
        # Evaluate population
        fitness_values = []
        for candidate in candidates:
            fitness = evaluate_policy(candidate, num_episodes=2)
            fitness_values.append(fitness)
        
        # Update CMA-ES
        stats = cma.tell(fitness_values)
        
        best_fitness_history.append(stats["best_fitness"])
        mean_fitness_history.append(stats["mean_fitness"])
        
        if generation % 10 == 0:
            elapsed = time.time() - start_time
            print(f"Generation {generation:3d} | Best: {stats['best_fitness']:8.2f} | "
                  f"Mean: {stats['mean_fitness']:8.2f} | Sigma: {stats['sigma']:.4f} | "
                  f"Time: {elapsed:.1f}s")
    
    print(f"\nCMA-ES training completed in {time.time() - start_time:.1f}s")
    
    # Get best solution
    final_candidates = cma.ask()
    final_fitness = [evaluate_policy(candidate, num_episodes=5) for candidate in final_candidates]
    best_idx = np.argmax(final_fitness)
    best_policy_params = final_candidates[best_idx]
    
    return best_policy_params, best_fitness_history, mean_fitness_history, cma


def train_with_genetic_algorithm(num_generations=200):
    """Train policy using Genetic Algorithm."""
    print("\n=== Genetic Algorithm Training ===")
    
    # Create sample policy to get parameter count
    env = SimpleBipedalWalker()
    sample_policy = SimplePolicy(env.state_dim, env.action_dim)
    param_count = len(sample_policy.get_parameters_as_vector())
    
    print(f"Policy parameters: {param_count}")
    
    # Initialize GA
    ga = GeneticAlgorithm(
        dimension=param_count,
        population_size=100,
        mutation_rate=0.15,
        crossover_rate=0.8
    )
    
    best_fitness_history = []
    mean_fitness_history = []
    
    print("\nStarting GA optimization...")
    start_time = time.time()
    
    for generation in range(num_generations):
        # Evaluate population
        fitness_values = []
        for individual in ga.population:
            fitness = evaluate_policy(individual, num_episodes=2)
            fitness_values.append(fitness)
        
        # Evolve population
        stats = ga.evolve(fitness_values)
        
        best_fitness_history.append(stats["best_fitness"])
        mean_fitness_history.append(stats["mean_fitness"])
        
        if generation % 20 == 0:
            elapsed = time.time() - start_time
            print(f"Generation {generation:3d} | Best: {stats['best_fitness']:8.2f} | "
                  f"Mean: {stats['mean_fitness']:8.2f} | Time: {elapsed:.1f}s")
    
    print(f"\nGA training completed in {time.time() - start_time:.1f}s")
    
    # Get best solution
    final_fitness = [evaluate_policy(individual, num_episodes=5) for individual in ga.population]
    best_idx = np.argmax(final_fitness)
    best_policy_params = ga.population[best_idx]
    
    return best_policy_params, best_fitness_history, mean_fitness_history, ga


def compare_black_box_methods():
    """Compare CMA-ES and GA performance."""
    print("=== Chapter 17: Black-Box Optimizations in RL ===")
    print("Comparing CMA-ES and Genetic Algorithm for policy optimization...\n")
    
    # Train with CMA-ES
    cma_best_params, cma_best_history, cma_mean_history, cma_optimizer = train_with_cma_es(num_generations=80)
    
    # Train with GA
    ga_best_params, ga_best_history, ga_mean_history, ga_optimizer = train_with_genetic_algorithm(num_generations=160)
    
    # Final evaluation
    print("\n=== Final Evaluation ===")
    
    # Test CMA-ES best policy
    cma_final_score = evaluate_policy(cma_best_params, num_episodes=10)
    print(f"CMA-ES final score: {cma_final_score:.2f}")
    
    # Test GA best policy
    ga_final_score = evaluate_policy(ga_best_params, num_episodes=10)
    print(f"GA final score: {ga_final_score:.2f}")
    
    # Create comparison plots
    create_comparison_plots(cma_best_history, cma_mean_history, ga_best_history, ga_mean_history)
    
    # Demonstrate policy behavior
    demonstrate_policies(cma_best_params, ga_best_params)
    
    return {
        "cma_best_params": cma_best_params,
        "ga_best_params": ga_best_params,
        "cma_final_score": cma_final_score,
        "ga_final_score": ga_final_score,
        "cma_optimizer": cma_optimizer,
        "ga_optimizer": ga_optimizer
    }


def create_comparison_plots(cma_best, cma_mean, ga_best, ga_mean):
    """Create comparison plots for black-box optimization methods."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Black-Box Optimization Comparison', fontsize=16)
    
    # Adjust generations for fair comparison
    cma_gens = np.arange(1, len(cma_best) + 1)
    ga_gens = np.arange(1, len(ga_best) + 1)
    
    # Best fitness comparison
    axes[0, 0].plot(cma_gens, cma_best, 'b-', linewidth=2, label='CMA-ES')
    axes[0, 0].plot(ga_gens, ga_best, 'r-', linewidth=2, label='GA')
    axes[0, 0].set_title('Best Fitness Over Generations')
    axes[0, 0].set_xlabel('Generation')
    axes[0, 0].set_ylabel('Best Fitness')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Mean fitness comparison
    axes[0, 1].plot(cma_gens, cma_mean, 'b--', linewidth=2, label='CMA-ES')
    axes[0, 1].plot(ga_gens, ga_mean, 'r--', linewidth=2, label='GA')
    axes[0, 1].set_title('Mean Population Fitness')
    axes[0, 1].set_xlabel('Generation')
    axes[0, 1].set_ylabel('Mean Fitness')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Learning curves (smoothed)
    window = 5
    cma_smooth = np.convolve(cma_best, np.ones(window)/window, mode='valid')
    ga_smooth = np.convolve(ga_best, np.ones(window)/window, mode='valid')
    
    axes[1, 0].plot(cma_smooth, 'b-', linewidth=3, label='CMA-ES (smoothed)')
    axes[1, 0].plot(ga_smooth, 'r-', linewidth=3, label='GA (smoothed)')
    axes[1, 0].set_title('Smoothed Learning Curves')
    axes[1, 0].set_xlabel('Generation')
    axes[1, 0].set_ylabel('Fitness')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Convergence rate
    cma_improvement = np.diff(cma_best)
    ga_improvement = np.diff(ga_best)
    
    axes[1, 1].plot(cma_improvement, 'b-', alpha=0.7, label='CMA-ES improvement')
    axes[1, 1].plot(ga_improvement, 'r-', alpha=0.7, label='GA improvement')
    axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[1, 1].set_title('Fitness Improvement per Generation')
    axes[1, 1].set_xlabel('Generation')
    axes[1, 1].set_ylabel('Fitness Change')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('chapter_17_black_box_optimization_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\nComparison plots saved as 'chapter_17_black_box_optimization_comparison.png'")


def demonstrate_policies(cma_params, ga_params):
    """Demonstrate the behavior of optimized policies."""
    print("\n=== Policy Behavior Demonstration ===")
    
    env = SimpleBipedalWalker()
    
    # Create policies
    cma_policy = SimplePolicy(env.state_dim, env.action_dim)
    ga_policy = SimplePolicy(env.state_dim, env.action_dim)
    
    cma_policy.set_parameters_from_vector(cma_params)
    ga_policy.set_parameters_from_vector(ga_params)
    
    # Test both policies
    for name, policy in [("CMA-ES", cma_policy), ("GA", ga_policy)]:
        print(f"\n{name} Policy Demonstration:")
        
        total_rewards = []
        total_steps = []
        
        for episode in range(5):
            state = env.reset()
            episode_reward = 0
            step_count = 0
            
            for step in range(1600):
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    action = policy(state_tensor).squeeze().numpy()
                
                state, reward, done, _ = env.step(action)
                episode_reward += reward
                step_count += 1
                
                if done:
                    break
            
            total_rewards.append(episode_reward)
            total_steps.append(step_count)
            
            print(f"  Episode {episode + 1}: Reward = {episode_reward:8.2f}, Steps = {step_count:4d}")
        
        avg_reward = np.mean(total_rewards)
        avg_steps = np.mean(total_steps)
        std_reward = np.std(total_rewards)
        
        print(f"  Average: Reward = {avg_reward:8.2f} ± {std_reward:6.2f}, Steps = {avg_steps:6.1f}")


def analyze_population_diversity(optimizer_results):
    """Analyze population diversity in evolutionary algorithms."""
    print("\n=== Population Diversity Analysis ===")
    
    cma_optimizer = optimizer_results["cma_optimizer"]
    ga_optimizer = optimizer_results["ga_optimizer"]
    
    # Analyze final populations
    print("\nFinal Population Analysis:")
    
    # CMA-ES diversity
    cma_samples = cma_optimizer.ask()
    cma_diversity = np.mean(np.std(cma_samples, axis=0))
    print(f"CMA-ES population diversity (std): {cma_diversity:.4f}")
    print(f"CMA-ES sigma: {cma_optimizer.sigma:.4f}")
    
    # GA diversity
    ga_population = np.array(ga_optimizer.population)
    ga_diversity = np.mean(np.std(ga_population, axis=0))
    print(f"GA population diversity (std): {ga_diversity:.4f}")
    print(f"GA mutation rate: {ga_optimizer.mutation_rate}")
    
    # Parameter statistics
    print(f"\nParameter Statistics:")
    print(f"CMA-ES mean parameter magnitude: {np.mean(np.abs(cma_optimizer.mean)):.4f}")
    print(f"GA mean parameter magnitude: {np.mean(np.abs(np.mean(ga_population, axis=0))):.4f}")


if __name__ == "__main__":
    print("Chapter 17: Black-Box Optimizations in RL")
    print("="*60)
    print("This chapter demonstrates gradient-free optimization methods.")
    print("We'll compare CMA-ES and Genetic Algorithms for policy optimization.\n")
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Run comparison
    results = compare_black_box_methods()
    
    # Additional analysis
    analyze_population_diversity(results)
    
    print("\n" + "="*60)
    print("Key Concepts Demonstrated:")
    print("- CMA-ES for continuous parameter optimization")
    print("- Genetic Algorithm with selection, crossover, mutation")
    print("- Population-based training strategies")
    print("- Gradient-free optimization for noisy fitness landscapes")
    print("- Comparison of evolutionary vs gradient-based methods")
    print("\nWhen to use black-box optimization:")
    print("- Non-differentiable or discontinuous objectives")
    print("- Noisy or stochastic environments")
    print("- Multi-modal optimization landscapes")
    print("- When gradient computation is expensive")
    print("- Robustness to hyperparameter choices")
    print("\nNext: Chapter 18 - Advanced Exploration")
