# Chapter 17: Black-Box Optimization in RL — Summary

## Theory Summary

Black-box (gradient-free) optimization is a family of methods that optimize policy parameters without access to analytical gradients. These are useful when gradients are unavailable, expensive, or unreliable (e.g., non-differentiable simulators or noisy evaluation). Two major families covered here:

- Evolution Strategies (ES) and Covariance Matrix Adaptation Evolution Strategy (CMA-ES): population-based continuous optimization that adapts a multivariate Gaussian search distribution via estimated covariances and step-size adaptation.

- Genetic Algorithms (GA): population-based heuristics using selection, crossover, and mutation to explore the parameter space.

Why use black-box methods in RL:

- Robustness to noisy or sparse reward signals
- Applicability to policies parameterized by non-differentiable components
- Good parallel scalability (population evaluations can be distributed)

Core concepts:

- Population generation: sample candidate parameters from a distribution (typically Gaussian).
- Evaluation: measure fitness (e.g., episode return) for each candidate via simulation.
- Selection & recombination (in GAs) or weighted recombination (in CMA-ES) to form new mean/parents.
- Covariance adaptation (CMA-ES) to increase exploration along productive directions.

CMA-ES canonical steps (informal):

1. Sample \(x_i = m + \sigma B D z_i\) where \(z_i\sim\mathcal{N}(0, I)\) and \(C = B D^2 B^T\).
2. Evaluate fitness \(f(x_i)\) and sort candidates.
3. Update mean \(m\) as a weighted average of top \(\mu\) solutions.
4. Update evolution paths and covariance \(C\) with rank-one and rank-\(\mu\) updates.
5. Adapt step size \(\sigma\) using the path length control.

GA canonical steps (informal):

1. Evaluate population fitness.
2. Select parents (e.g., tournament selection).
3. Apply crossover (e.g., uniform, single-point) to produce offspring.
4. Apply mutation (e.g., Gaussian perturbation) with some probability.
5. Replace population and iterate.

When to prefer black-box vs gradient-based methods:

- Use black-box when simulation noise, discontinuities, or multi-modal landscapes dominate, or when large-scale parallel resources are available. Gradient-based methods (PPO, DDPG, SAC) are often more sample efficient when gradients are reliable.

## Code Implementation Breakdown (file: `pytorch_rl_tutorial/chapter_17_black_box_optimization.py`)

This chapter contains robust, self-contained implementations of a simplified CMA-ES (`SimpleCMA`) and a basic Genetic Algorithm (`GeneticAlgorithm`), along with a small demonstration environment (`SimpleBipedalWalker`) and a simple policy network (`SimplePolicy`).

Key classes and functions:

- `SimpleCMA`
  - `__init__`: initializes mean (`self.mean`), sigma (`self.sigma`), population size and adaptation parameters (weights, mu_eff, c_sigma, d_sigma, c_c, c_1, c_mu).
  - `ask()`: samples a population of candidate parameter vectors by sampling z ~ N(0, I), mapping via current covariance decomposition, and scaling by sigma.
  - `tell(fitness_values)`: sorts candidates by fitness (assumes maximization), updates mean using weighted recombination of top `mu` individuals, updates evolution paths (`p_sigma`, `p_c`), adapts sigma using path length control, and performs rank-one and rank-mu updates to covariance matrix `C`.
  - Returns diagnostic stats including best fitness, mean fitness, sigma and generation count.

- `GeneticAlgorithm`
  - `evaluate_population()`: evaluates the current population's fitness, with optional parallel evaluation using multiprocessing.
  - `selection()`: tournament selection (tournament size 3) to pick parents.
  - `crossover()`: uniform crossover between two parents to create children, controlled by `crossover_rate`.
  - `mutation()`: Gaussian mutation applied per-gene with `mutation_rate` probability.
  - `evolve()`: orchestrates selection, crossover, mutation and population replacement and records generation-level statistics.

- `SimpleBipedalWalker`
  - A toy continuous control environment with `state_dim=24`, `action_dim=4`, and simplified dynamics; deterministic-ish transitions with stochastic perturbations. Serve as quick local environment for experiments without external Gym dependencies.

- `SimplePolicy(nn.Module)`
  - Small MLP with Tanh output in [-1,1]; includes helper methods:
    - `get_parameters_as_vector()`: flattens parameters to a single numpy vector.
    - `set_parameters_from_vector(vector)`: sets network parameter tensors from a flat vector.

- `evaluate_policy(policy_params, env_class=SimpleBipedalWalker, num_episodes=3)`: Instantiates a `SimplePolicy`, sets parameters from `policy_params`, runs `num_episodes` episodes in `SimpleBipedalWalker`, and returns average episode reward.

- `train_with_cma_es()` and `train_with_genetic_algorithm()`
  - Orchestrate optimization loops, print periodic stats, and finally return best parameters along with fitness histories and optimizer instances for further analysis.

- `compare_black_box_methods()`
  - Runs both optimizers, evaluates final solutions, generates comparison plots (`pytorch_rl_tutorial/black_box_optimization_comparison.png`) and demonstrates the final policies by running multiple episodes and printing per-episode rewards.

Implementation notes and engineering details:

- `SimpleCMA` uses a simplified eigen decomposition update scheme where eigen-decomposition of `C` is refreshed every 10 generations to construct `B` and `D` (square root of eigenvalues). This keeps computational cost bounded for moderate dimensionality.
- Fitness is treated as maximization; CMA's `tell()` sorts descending.
- `GeneticAlgorithm` uses a population-size-specified initialization and simple tournament selection. Crossover is uniform; mutation adds Gaussian noise per-gene with some probability.
- `evaluate_policy()` executes the policy deterministically (no noise) and returns the average return across `num_episodes`.
- The demonstration includes plotting (best vs mean fitness over generations) and population diversity analyses.

## Connection Between Theory and Code

- The `ask()/tell()` loop of `SimpleCMA` matches the standard CMA-ES generative and update steps: sample, evaluate, recombine, adapt covariance and step size.
- `GeneticAlgorithm.evolve()` implements selection → crossover → mutation → replacement, which is the canonical GA cycle.
- `SimplePolicy` parameter vector APIs (`get_parameters_as_vector` and `set_parameters_from_vector`) permit black-box optimizers to operate on entire network parameter vectors without dealing with PyTorch gradients.
