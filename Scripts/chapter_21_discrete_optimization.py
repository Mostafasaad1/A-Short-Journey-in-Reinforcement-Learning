#!/usr/bin/env python3
"""
Chapter 21: RL in Discrete Optimization

This chapter demonstrates how to apply RL to combinatorial optimization problems,
specifically the Traveling Salesman Problem (TSP) using Pointer Networks and
REINFORCE. We'll show how RL can learn to construct good solutions for NP-hard problems.

Key concepts covered:
- Pointer Networks with attention mechanisms
- Constructive approach to combinatorial optimization
- REINFORCE with baseline for sequence generation
- Graph representation and attention over nodes
- Comparison with classical heuristics


"""

# Chapter 21: RL in Discrete Optimization
# Start by installing the required packages:
# !pip install torch numpy matplotlib tqdm -q

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
import math
from typing import List, Tuple, Dict, Optional
import itertools
from dataclasses import dataclass

# torch.autograd.set_detect_anomaly(True)
# Configure matplotlib for non-interactive mode
plt.switch_backend("Agg")
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "PingFang SC", "Arial Unicode MS", "Hiragino Sans GB"]
plt.rcParams["axes.unicode_minus"] = False

# TSP Problem Definition
@dataclass
class TSPInstance:
    """Traveling Salesman Problem instance."""
    cities: np.ndarray  # City coordinates [n_cities, 2]
    distances: np.ndarray  # Distance matrix [n_cities, n_cities]
    n_cities: int
    
    @classmethod
    def random_instance(cls, n_cities: int, seed: Optional[int] = None):
        """Generate random TSP instance."""
        if seed is not None:
            np.random.seed(seed)
        
        # Generate random city coordinates
        cities = np.random.uniform(0, 1, size=(n_cities, 2))
        
        # Calculate distance matrix
        distances = np.zeros((n_cities, n_cities))
        for i in range(n_cities):
            for j in range(n_cities):
                if i != j:
                    distances[i, j] = np.linalg.norm(cities[i] - cities[j])
        
        return cls(cities=cities, distances=distances, n_cities=n_cities)
    
    def tour_length(self, tour: List[int]) -> float:
        """Calculate total length of a tour."""
        if len(tour) != self.n_cities:
            return float('inf')
        
        total_length = 0.0
        for i in range(len(tour)):
            from_city = tour[i]
            to_city = tour[(i + 1) % len(tour)]  # Return to start
            total_length += self.distances[from_city, to_city]
        
        return total_length
    
    def visualize_tour(self, tour: List[int], title: str = "TSP Tour"):
        """Visualize a tour."""
        plt.figure(figsize=(8, 8))
        
        # Plot cities
        plt.scatter(self.cities[:, 0], self.cities[:, 1], c='red', s=100, zorder=3)
        
        # Plot tour
        tour_coords = self.cities[tour]
        tour_coords = np.vstack([tour_coords, tour_coords[0]])  # Close the loop
        plt.plot(tour_coords[:, 0], tour_coords[:, 1], 'b-', linewidth=2, alpha=0.7)
        
        # Label cities
        for i, (x, y) in enumerate(self.cities):
            plt.annotate(str(i), (x, y), xytext=(5, 5), textcoords='offset points')
        
        plt.title(f"{title} (Length: {self.tour_length(tour):.3f})")
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        return plt.gcf()


# Attention Mechanism
class Attention(nn.Module):
    """Attention mechanism for Pointer Networks."""
    
    def __init__(self, hidden_dim: int):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        
        self.W_ref = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
        
    def forward(self, query: torch.Tensor, ref: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            query: [batch_size, hidden_dim] - current decoder state
            ref: [batch_size, seq_len, hidden_dim] - encoder outputs
            mask: [batch_size, seq_len] - mask for visited cities
        
        Returns:
            attention_weights: [batch_size, seq_len]
            context: [batch_size, hidden_dim]
        """
        batch_size, seq_len, _ = ref.size()
        
        # Expand query to match reference dimensions
        query_expanded = query.unsqueeze(1).expand(-1, seq_len, -1)  # [batch_size, seq_len, hidden_dim]
        
        # Calculate attention scores
        ref_projected = self.W_ref(ref)  # [batch_size, seq_len, hidden_dim]
        query_projected = self.W_q(query_expanded)  # [batch_size, seq_len, hidden_dim]
        
        scores = self.v(torch.tanh(ref_projected + query_projected)).squeeze(-1)  # [batch_size, seq_len]
        
        # Apply mask (set scores to -inf for visited cities)
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))
        
        # Softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)  # [batch_size, seq_len]
        
        # Calculate context vector
        context = torch.bmm(attention_weights.unsqueeze(1), ref).squeeze(1)  # [batch_size, hidden_dim]
        
        return attention_weights, context


class PointerNetwork(nn.Module):
    """Pointer Network for TSP using attention mechanism."""
    
    def __init__(self, input_dim: int = 2, hidden_dim: int = 128):
        super(PointerNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Encoder: embeds city coordinates
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Decoder LSTM
        self.decoder_lstm = nn.LSTMCell(hidden_dim, hidden_dim)
        
        # Attention mechanism
        self.attention = Attention(hidden_dim)
        
        # Initialize decoder state
        self.decoder_start = nn.Parameter(torch.randn(hidden_dim))
        
    def forward(self, cities: torch.Tensor, tour: List[int] = None):
        """
        Args:
            cities: [batch_size, n_cities, 2] - city coordinates
            tour: If provided, use teacher forcing
        
        Returns:
            log_probs: [batch_size, n_cities] - log probabilities for each step
            tour: [batch_size, n_cities] - generated tour
        """
        batch_size, n_cities, _ = cities.size()
        
        # Encode cities
        encoded_cities = self.encoder(cities)  # [batch_size, n_cities, hidden_dim]
        
        # Initialize decoder state
        h = self.decoder_start.unsqueeze(0).expand(batch_size, -1)  # [batch_size, hidden_dim]
        c = torch.zeros_like(h)  # [batch_size, hidden_dim]
        
        # Track visited cities
        mask = torch.zeros(batch_size, n_cities, dtype=torch.bool, device=cities.device)
        
        # Store outputs
        log_probs = []
        generated_tour = []
        
        for step in range(n_cities):
            # Get attention weights and context
            attention_weights, context = self.attention(h, encoded_cities, mask)
            
            # Store log probabilities
            log_probs.append(torch.log(attention_weights + 1e-10))
            
            if tour is not None:  # Teacher forcing
                next_city = tour[step]
                if isinstance(next_city, int):
                    next_city = torch.full((batch_size,), next_city, dtype=torch.long, device=cities.device)
            else:  # Sampling
                next_city = Categorical(attention_weights).sample()
            
            generated_tour.append(next_city)
            
            # Update mask
            mask = mask.detach().clone()
            mask.scatter_(1, next_city.unsqueeze(1), True)
            
            # Get embedding for next city
            next_city_embedding = encoded_cities.gather(
                1, next_city.unsqueeze(1).unsqueeze(2).expand(-1, -1, self.hidden_dim)
            ).squeeze(1)
            
            # Update decoder state
            h, c = self.decoder_lstm(next_city_embedding, (h, c))
        
        log_probs = torch.stack(log_probs, dim=1)  # [batch_size, n_cities, n_cities]
        
        if tour is None:
            generated_tour = torch.stack(generated_tour, dim=1)  # [batch_size, n_cities]
        
        return log_probs, generated_tour
    
    def generate_tour(self, cities: torch.Tensor):
        """Generate tour using greedy decoding."""
        self.eval()
        with torch.no_grad():
            _, tour = self.forward(cities)
        return tour


class TSPSolver:
    """TSP solver using Pointer Networks and REINFORCE."""
    
    def __init__(self, hidden_dim: int = 128, lr: float = 1e-3):
        self.model = PointerNetwork(input_dim=2, hidden_dim=hidden_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        self.training_stats = {
            "loss": [],
            "tour_length": [],
            "baseline": [],
            "advantage": []
        }
        
        # Baseline (moving average of tour lengths)
        self.baseline = None
        self.baseline_alpha = 0.05
        
    def solve(self, tsp_instance: TSPInstance, beam_width: int = 1) -> Tuple[List[int], float]:
        """Solve TSP instance."""
        cities_tensor = torch.FloatTensor(tsp_instance.cities).unsqueeze(0)
        
        if beam_width == 1:
            # Greedy decoding
            tour_tensor = self.model.generate_tour(cities_tensor)
            tour = tour_tensor.squeeze(0).tolist()
            length = tsp_instance.tour_length(tour)
        else:
            # Beam search
            tour, length = self._beam_search(tsp_instance, cities_tensor, beam_width)
        
        return tour, length
    
    def _beam_search(self, tsp_instance: TSPInstance, cities_tensor: torch.Tensor, beam_width: int):
        """Beam search for better solutions."""
        batch_size, n_cities, _ = cities_tensor.size()
        
        # Initialize beams
        beams = [{
            'tour': [],
            'log_prob': 0.0,
            'mask': torch.zeros(n_cities, dtype=torch.bool),
            'h': self.model.decoder_start.clone(),
            'c': torch.zeros_like(self.model.decoder_start)
        }]
        
        encoded_cities = self.model.encoder(cities_tensor.squeeze(0))  # [n_cities, hidden_dim]
        
        for step in range(n_cities):
            new_beams = []
            
            for beam in beams:
                if len(beam['tour']) == n_cities:
                    new_beams.append(beam)
                    continue
                
                # Get attention weights
                h = beam['h'].unsqueeze(0)
                attention_weights, _ = self.model.attention(
                    h, encoded_cities.unsqueeze(0), beam['mask'].unsqueeze(0)
                )
                attention_weights = attention_weights.squeeze(0)
                
                # Get top k actions
                top_k = min(beam_width, torch.sum(~beam['mask']).item())
                top_probs, top_actions = torch.topk(attention_weights, top_k)
                
                for i in range(top_k):
                    action = top_actions[i].item()
                    prob = top_probs[i].item()
                    
                    new_beam = {
                        'tour': beam['tour'] + [action],
                        'log_prob': beam['log_prob'] + math.log(prob + 1e-10),
                        'mask': beam['mask'].clone(),
                        'h': beam['h'].clone(),
                        'c': beam['c'].clone()
                    }
                    
                    new_beam['mask'][action] = True
                    
                    # Update LSTM state
                    next_city_embedding = encoded_cities[action]
                    new_beam['h'], new_beam['c'] = self.model.decoder_lstm(
                        next_city_embedding, (new_beam['h'], new_beam['c'])
                    )
                    
                    new_beams.append(new_beam)
            
            # Keep top beams
            beams = sorted(new_beams, key=lambda x: x['log_prob'], reverse=True)[:beam_width]
        
        # Return best tour
        best_beam = beams[0]
        tour = best_beam['tour']
        length = tsp_instance.tour_length(tour)
        
        return tour, length
    
    def train_step(self, batch_instances: List[TSPInstance]):
        """Perform one training step using REINFORCE."""
        self.model.train()
        
        batch_size = len(batch_instances)
        
        # Prepare batch
        cities_batch = torch.stack([
            torch.FloatTensor(instance.cities) for instance in batch_instances
        ])
        
        # Generate tours
        log_probs, tours = self.model(cities_batch)
        
        # Calculate tour lengths (rewards)
        tour_lengths = []
        for i, instance in enumerate(batch_instances):
            tour = tours[i].tolist()
            length = instance.tour_length(tour)
            tour_lengths.append(length)
        
        tour_lengths = torch.FloatTensor(tour_lengths)
        
        # Update baseline
        current_baseline = tour_lengths.mean().item()
        if self.baseline is None:
            self.baseline = current_baseline
        else:
            self.baseline = (1 - self.baseline_alpha) * self.baseline + self.baseline_alpha * current_baseline
        
        # Calculate advantages (negative because we want to minimize tour length)
        advantages = -(tour_lengths - self.baseline)
        
        # Calculate REINFORCE loss
        loss = 0
        for i in range(batch_size):
            tour = tours[i]
            tour_log_probs = []
            
            for step in range(len(tour)):
                city_idx = tour[step]
                step_log_prob = log_probs[i, step, city_idx]
                tour_log_probs.append(step_log_prob)
            
            tour_log_prob = torch.stack(tour_log_probs).sum()
            loss += -advantages[i] * tour_log_prob
        
        loss = loss / batch_size
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Store statistics
        self.training_stats["loss"].append(loss.item())
        self.training_stats["tour_length"].append(tour_lengths.mean().item())
        self.training_stats["baseline"].append(self.baseline)
        self.training_stats["advantage"].append(advantages.mean().item())
        
        return {
            "loss": loss.item(),
            "avg_tour_length": tour_lengths.mean().item(),
            "baseline": self.baseline,
            "avg_advantage": advantages.mean().item()
        }


# Classical TSP Heuristics for Comparison
class TSPHeuristics:
    """Classical heuristics for TSP."""
    
    @staticmethod
    def nearest_neighbor(tsp_instance: TSPInstance, start_city: int = 0) -> Tuple[List[int], float]:
        """Nearest neighbor heuristic."""
        n_cities = tsp_instance.n_cities
        tour = [start_city]
        unvisited = set(range(n_cities)) - {start_city}
        
        current_city = start_city
        
        while unvisited:
            nearest_city = min(unvisited, key=lambda city: tsp_instance.distances[current_city, city])
            tour.append(nearest_city)
            unvisited.remove(nearest_city)
            current_city = nearest_city
        
        length = tsp_instance.tour_length(tour)
        return tour, length
    
    @staticmethod
    def two_opt(tsp_instance: TSPInstance, initial_tour: List[int], max_iterations: int = 1000) -> Tuple[List[int], float]:
        """2-opt improvement heuristic."""
        tour = initial_tour.copy()
        best_length = tsp_instance.tour_length(tour)
        
        for iteration in range(max_iterations):
            improved = False
            
            for i in range(len(tour) - 1):
                for j in range(i + 2, len(tour)):
                    # Try swapping edges
                    new_tour = tour[:i+1] + tour[i+1:j+1][::-1] + tour[j+1:]
                    new_length = tsp_instance.tour_length(new_tour)
                    
                    if new_length < best_length:
                        tour = new_tour
                        best_length = new_length
                        improved = True
            
            if not improved:
                break
        
        return tour, best_length
    
    @staticmethod
    def random_tour(tsp_instance: TSPInstance) -> Tuple[List[int], float]:
        """Random tour baseline."""
        tour = list(range(tsp_instance.n_cities))
        random.shuffle(tour)
        length = tsp_instance.tour_length(tour)
        return tour, length


def train_tsp_solver(n_cities: int = 20, num_epochs: int = 200, batch_size: int = 32):
    """Train TSP solver using REINFORCE."""
    print(f"=== Chapter 21: RL in Discrete Optimization ===")
    print(f"Training Pointer Network for TSP with {n_cities} cities...\n")
    
    # Create solver
    solver = TSPSolver(hidden_dim=128, lr=1e-3)
    
    print(f"Model architecture:")
    print(f"  Input: City coordinates (2D)")
    print(f"  Encoder: Linear layers -> {solver.model.hidden_dim}D embeddings")
    print(f"  Decoder: LSTM with attention mechanism")
    print(f"  Output: Pointer to next city in tour")
    print(f"  Training: REINFORCE with baseline\n")
    
    # Training loop
    print("Starting training...")
    
    for epoch in range(num_epochs):
        # Generate batch of random TSP instances
        batch_instances = [
            TSPInstance.random_instance(n_cities, seed=epoch * batch_size + i)
            for i in range(batch_size)
        ]
        
        # Training step
        stats = solver.train_step(batch_instances)
        
        # Logging
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1:3d} | Loss: {stats['loss']:8.4f} | "
                  f"Avg Tour: {stats['avg_tour_length']:6.3f} | "
                  f"Baseline: {stats['baseline']:6.3f} | "
                  f"Advantage: {stats['avg_advantage']:6.3f}")
    
    print("\nTraining completed!")
    
    # Evaluation
    print("\n=== Evaluation ===")
    
    # Test on new instances
    test_instances = [TSPInstance.random_instance(n_cities, seed=1000 + i) for i in range(10)]
    
    rl_results = []
    nn_results = []
    opt_results = []
    random_results = []
    
    for i, instance in enumerate(test_instances):
        # RL solution
        rl_tour, rl_length = solver.solve(instance, beam_width=1)
        rl_results.append(rl_length)
        
        # Nearest neighbor
        nn_tour, nn_length = TSPHeuristics.nearest_neighbor(instance)
        nn_results.append(nn_length)
        
        # 2-opt on nearest neighbor
        opt_tour, opt_length = TSPHeuristics.two_opt(instance, nn_tour)
        opt_results.append(opt_length)
        
        # Random baseline
        random_tour, random_length = TSPHeuristics.random_tour(instance)
        random_results.append(random_length)
        
        if i < 3:  # Show detailed results for first 3 instances
            print(f"\nTest Instance {i+1}:")
            print(f"  RL (Pointer Net): {rl_length:.3f}")
            print(f"  Nearest Neighbor: {nn_length:.3f}")
            print(f"  NN + 2-opt:       {opt_length:.3f}")
            print(f"  Random:           {random_length:.3f}")
    
    # Summary statistics
    print(f"\nSummary over {len(test_instances)} test instances:")
    print(f"  RL (Pointer Net): {np.mean(rl_results):.3f} ± {np.std(rl_results):.3f}")
    print(f"  Nearest Neighbor: {np.mean(nn_results):.3f} ± {np.std(nn_results):.3f}")
    print(f"  NN + 2-opt:       {np.mean(opt_results):.3f} ± {np.std(opt_results):.3f}")
    print(f"  Random:           {np.mean(random_results):.3f} ± {np.std(random_results):.3f}")
    
    # Improvement ratios
    rl_vs_nn = np.mean(rl_results) / np.mean(nn_results)
    rl_vs_random = np.mean(rl_results) / np.mean(random_results)
    
    print(f"\nImprovement ratios:")
    print(f"  RL vs Nearest Neighbor: {rl_vs_nn:.3f}")
    print(f"  RL vs Random:           {rl_vs_random:.3f}")
    
    # Create training plots
    create_tsp_training_plots(solver)
    
    # Visualize example solutions
    visualize_solutions(solver, test_instances[0])
    
    return solver, test_instances


def create_tsp_training_plots(solver):
    """Create training progress plots."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('TSP Pointer Network Training Progress', fontsize=16)
    
    epochs = range(1, len(solver.training_stats["loss"]) + 1)
    
    # Training loss
    axes[0, 0].plot(epochs, solver.training_stats["loss"], 'b-', linewidth=2)
    axes[0, 0].set_title('Training Loss (REINFORCE)')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Average tour length
    axes[0, 1].plot(epochs, solver.training_stats["tour_length"], 'r-', linewidth=2, label='Tour Length')
    axes[0, 1].plot(epochs, solver.training_stats["baseline"], 'g--', linewidth=2, label='Baseline')
    axes[0, 1].set_title('Tour Length Over Training')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Average Tour Length')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Advantage values
    axes[1, 0].plot(epochs, solver.training_stats["advantage"], 'purple', linewidth=2)
    axes[1, 0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[1, 0].set_title('Average Advantage')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Advantage')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Learning progress (smoothed)
    window = 10
    if len(solver.training_stats["tour_length"]) >= window:
        smoothed_length = np.convolve(solver.training_stats["tour_length"], 
                                    np.ones(window)/window, mode='valid')
        smoothed_epochs = range(window, len(solver.training_stats["tour_length"]) + 1)
        
        axes[1, 1].plot(smoothed_epochs, smoothed_length, 'orange', linewidth=3)
        axes[1, 1].set_title(f'Smoothed Tour Length (window={window})')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Smoothed Tour Length')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pytorch_rl_tutorial/chapter_21_tsp_training_progress.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\nTraining plots saved as 'tsp_training_progress.png'")


def visualize_solutions(solver, tsp_instance):
    """Visualize different solution methods on the same instance."""
    print("\n=== Solution Visualization ===")
    
    # Generate different solutions
    rl_tour, rl_length = solver.solve(tsp_instance, beam_width=1)
    nn_tour, nn_length = TSPHeuristics.nearest_neighbor(tsp_instance)
    opt_tour, opt_length = TSPHeuristics.two_opt(tsp_instance, nn_tour)
    random_tour, random_length = TSPHeuristics.random_tour(tsp_instance)
    
    # Create subplot for each solution
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(f'TSP Solutions Comparison ({tsp_instance.n_cities} cities)', fontsize=16)
    
    solutions = [
        (rl_tour, rl_length, "RL (Pointer Network)"),
        (nn_tour, nn_length, "Nearest Neighbor"),
        (opt_tour, opt_length, "NN + 2-opt"),
        (random_tour, random_length, "Random")
    ]
    
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    
    for (tour, length, title), (row, col) in zip(solutions, positions):
        ax = axes[row, col]
        
        # Plot cities
        ax.scatter(tsp_instance.cities[:, 0], tsp_instance.cities[:, 1], 
                  c='red', s=100, zorder=3)
        
        # Plot tour
        tour_coords = tsp_instance.cities[tour]
        tour_coords = np.vstack([tour_coords, tour_coords[0]])  # Close the loop
        ax.plot(tour_coords[:, 0], tour_coords[:, 1], 'b-', linewidth=2, alpha=0.7)
        
        # Label cities
        for i, (x, y) in enumerate(tsp_instance.cities):
            ax.annotate(str(i), (x, y), xytext=(3, 3), textcoords='offset points', fontsize=8)
        
        ax.set_title(f"{title}\nLength: {length:.3f}")
        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Y coordinate')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('pytorch_rl_tutorial/chapter_21_tsp_solutions_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Solution comparison plots saved as 'tsp_solutions_comparison.png'")


def analyze_scalability():
    """Analyze how different methods scale with problem size."""
    print("\n=== Scalability Analysis ===")
    
    city_counts = [10, 15, 20, 25, 30]
    methods_results = {"RL": [], "NN": [], "NN+2opt": [], "Random": []}
    
    # Quick training for different problem sizes
    for n_cities in city_counts:
        print(f"\nTesting with {n_cities} cities...")
        
        # Train small model quickly
        solver = TSPSolver(hidden_dim=64, lr=1e-3)
        
        # Quick training (fewer epochs for larger problems)
        num_epochs = max(50, 200 - n_cities * 5)
        
        for epoch in range(num_epochs):
            batch_instances = [
                TSPInstance.random_instance(n_cities, seed=epoch * 8 + i)
                for i in range(8)
            ]
            solver.train_step(batch_instances)
        
        # Test on a few instances
        test_instances = [TSPInstance.random_instance(n_cities, seed=2000 + i) for i in range(5)]
        
        rl_lengths = []
        nn_lengths = []
        opt_lengths = []
        random_lengths = []
        
        for instance in test_instances:
            # RL solution
            rl_tour, rl_length = solver.solve(instance)
            rl_lengths.append(rl_length)
            
            # Classical methods
            nn_tour, nn_length = TSPHeuristics.nearest_neighbor(instance)
            nn_lengths.append(nn_length)
            
            opt_tour, opt_length = TSPHeuristics.two_opt(instance, nn_tour, max_iterations=100)
            opt_lengths.append(opt_length)
            
            random_tour, random_length = TSPHeuristics.random_tour(instance)
            random_lengths.append(random_length)
        
        methods_results["RL"].append(np.mean(rl_lengths))
        methods_results["NN"].append(np.mean(nn_lengths))
        methods_results["NN+2opt"].append(np.mean(opt_lengths))
        methods_results["Random"].append(np.mean(random_lengths))
        
        print(f"  RL: {np.mean(rl_lengths):.3f}")
        print(f"  NN: {np.mean(nn_lengths):.3f}")
        print(f"  NN+2opt: {np.mean(opt_lengths):.3f}")
    
    # Plot scalability
    plt.figure(figsize=(10, 6))
    
    for method, results in methods_results.items():
        plt.plot(city_counts, results, marker='o', linewidth=2, label=method)
    
    plt.title('TSP Solution Quality vs Problem Size')
    plt.xlabel('Number of Cities')
    plt.ylabel('Average Tour Length')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pytorch_rl_tutorial/chapter_21_tsp_scalability.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\nScalability analysis saved as 'tsp_scalability.png'")


if __name__ == "__main__":
    print("Chapter 21: RL in Discrete Optimization")
    print("="*50)
    print("This chapter demonstrates RL for combinatorial optimization.")
    print("We'll use Pointer Networks to solve the Traveling Salesman Problem.\n")
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Train TSP solver
    solver, test_instances = train_tsp_solver(n_cities=20, num_epochs=150, batch_size=32)
    
    # Analyze scalability
    analyze_scalability()
    
    print("\n" + "="*50)
    print("Key Concepts Demonstrated:")
    print("- Pointer Networks with attention mechanisms")
    print("- Constructive approach to combinatorial optimization")
    print("- REINFORCE with baseline for sequence generation")
    print("- Graph attention over city nodes")
    print("- Comparison with classical TSP heuristics")
    print("\nRL for discrete optimization is useful for:")
    print("- Routing and logistics problems")
    print("- Scheduling and resource allocation")
    print("- Circuit design and VLSI placement")
    print("- Protein folding and molecular design")
    print("- Any NP-hard combinatorial problem")
    print("\nNext: Chapter 22 - Multi-Agent RL")
