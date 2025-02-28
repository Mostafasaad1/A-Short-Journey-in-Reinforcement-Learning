# Chapter 18: Advanced Exploration — Summary

## Theory Summary

Exploration is essential in reinforcement learning, especially when rewards are sparse or require long sequences of coordinated actions. This chapter surveys several advanced exploration families and explains why naive \(\varepsilon\)-greedy or undirected randomness often fails on tasks like MountainCar or River Swim.

Main families covered:

- Noisy Networks: inject parameter noise (learned) into network layers to produce structured, state-dependent exploration. Noise parameters can be learned jointly with the rest of the network.

- Count-based methods: provide intrinsic rewards inversely proportional to state visitation counts (or pseudo-counts derived from density models). Useful to prefer novel states.

- Prediction-based (curiosity) methods: intrinsic rewards derived from prediction error, e.g., random network distillation (RND) or Intrinsic Curiosity Module (ICM). These reward unfamiliar transitions.

- Ensemble/uncertainty methods: bootstrapped DQN or ensemble Q-networks provide uncertainty estimates and support Thompson-sampling-style exploration.

Key takeaways and equations:

- Count bonus (example): intrinsic reward ~ \(1/\sqrt{N(s)}\), where \(N(s)\) is the number of visits to state s.
- Curiosity (ICM) computes intrinsic reward as forward-model prediction error in a learned feature space; inverse model can help focus on actionable variation.
- NoisyNet parameterization: factorized Gaussian noise added to weight parameters with learnable sigma; sample noise less frequently for on-policy algorithms.

Practical observations from the chapter experiments (MountainCar and Atari):

- Noisy networks often find goal states far faster than \(\varepsilon\)-greedy in MountainCar.
- Count-based bonuses (pseudo-counts with hashing) are effective in small state spaces.
- RND / distillation techniques can dramatically speed up discovery in hard-exploration Atari games.
- Bootstrapped DQN and ensembles can provide meaningful uncertainty-based exploration, but are more expensive.

## Code Implementation Breakdown (file: `pytorch_rl_tutorial/chapter_18_advanced_exploration.py`)

This chapter script implements a small suite of exploration techniques and compares them in a synthetic sparse-reward GridWorld (`SparseGridWorld`) to make experiments fast and reproducible. The code includes implementations of ICM, NoisyLinear/NoisyDQN, BootstrappedDQN, and simple count-based exploration.

Important classes and functions:

- `SparseGridWorld`:
  - Small grid-based environment with sparse goal reward (10.0) and small time penalty (-0.01). Useful for fast iteration.

- `ICMNetwork`:
  - Feature extractor (`feature_net`) mapping states to low-dimensional features.
  - Inverse model predicts action from concatenated state and next-state features (cross-entropy if actions are discrete).
  - Forward model predicts next-state feature from current-state feature + action one-hot; MSE forward loss is used as intrinsic reward.
  - `forward(state, next_state, action)` returns inverse_loss, forward_loss, intrinsic_reward.

- `NoisyLinear` and `NoisyDQN`:
  - `NoisyLinear` implements factorized Gaussian noise with learnable mu and sigma. `reset_noise()` samples factorized noise. Noise is applied to weights/biases during training; at eval time deterministic mu is used.
  - `NoisyDQN` composes a feature net with two `NoisyLinear` layers and exposes `reset_noise()`.

- `BootstrappedDQN`:
  - A shared feature extractor with multiple Q-heads (ensemble), exposing a `forward(state, head_idx=None)` API. `get_uncertainty()` computes per-sample variance across head predictions as an uncertainty estimate.

- `CountBasedExplorer`:
  - Simple count-based bonus: discretize states via rounding, maintain counts in a dictionary, and return bonus = c / sqrt(count).

- `ExplorationAgent`:
  - Unified agent that selects and trains using the chosen exploration technique (`icm`, `noisy`, `bootstrapped`, or `count`).
  - Key behaviors per method:
    - `icm`: epsilon-greedy action selection + ICM training; intrinsic reward appended to extrinsic reward scaled by 0.1.
    - `noisy`: NoisyDQN network; actions chosen greedily from noisy network (noise provides exploration). `reset_noise()` is called after training steps.
    - `bootstrapped`: randomly sample a head per episode for Thompson-sampling-style exploration; all heads are trained with their own targets.
    - `count`: epsilon-greedy + count-based bonus applied to rewards.
  - Training pipeline:
    - `select_action(state, training=True)` handles exploration policy per method.
    - `train_step(batch_size)`: samples experiences, computes intrinsic rewards (if ICM), adds bonuses, computes Q-learning loss (per-head for bootstrapped), performs optimizer steps and updates ICM parameters if appropriate.
    - `train_episode()` runs a full episode, calling `select_action`, storing transitions to memory, invoking `train_step()`, and updating epsilon and target networks periodically.

- `compare_exploration_methods(num_episodes)`:
  - Runs experiments across methods, logs success rates and reward statistics every 100 episodes, collects results, and produces plots (`advanced_exploration_comparison.png`).

Implementation notes and engineering details:

- Discretization: `CountBasedExplorer` rounds flattened state values to two decimal places to make hashing practical for continuous observations.
- Intrinsic reward scaling: In ICM the forward-model error is scaled (e.g., 0.1) before adding to extrinsic rewards.
- NoisyNet noise resampling: For on-policy methods, sample noise per batch rather than every forward pass to keep rollout consistency.
- Bootstrapped DQN: Each head is trained on the same sampled batch here (simplified); more advanced versions use per-head masks to control which experiences update which head.
- Vectorization: Many operations use small batch sizes (32) for the synthetic environment to keep training fast.

## Connection Between Theory and Code

- ICM: forward-model error as intrinsic reward is implemented in `ICMNetwork.forward()` and integrated in `ExplorationAgent.train_step()` — the forward loss is used as the scalar intrinsic reward and the inverse loss is included in network updates to ground features in action-predictive components.

- NoisyNet: `NoisyLinear` implements factorized Gaussian noise, and `NoisyDQN.reset_noise()` is used to resample noise appropriately.

- Bootstrapped DQN: ensemble of heads implemented with `BootstrappedDQN`; Thompson-sampling-style exploration achieved by sampling a random head per episode in `ExplorationAgent.select_action()`.

- Count-based: concrete hashing and bonus formula `bonus = c / sqrt(count)` is implemented in `CountBasedExplorer.get_count_bonus()`.

## Diagnostics and "What to Monitor"

- Success rate over time (test episodes), average reward, intrinsic reward magnitude, unique states visited (coverage), and Q-head uncertainty statistics.
- For ICM: forward loss curve and average intrinsic reward — check if intrinsic reward collapses too early.
- For NoisyNet: number of noisy layers and noise sigma values; monitor if exploration collapses (sigma -> 0) prematurely.
- For Bootstrapped DQN: variance across heads and per-head learning curves.
- For counts: size of dictionary (`unique_states`) and average visits per state.

## Suggested Experiments

- Replace the discrete hashing in `CountBasedExplorer` with learned density models (pseudo-counts) or RND (random network distillation) to scale to higher-dimensional observations.
- Combine ICM with a state-embedding learned through auxiliary losses or contrastive methods for better generalization.
- For Bootstrapped DQN, use per-head replay masks or bootstrap masks for more diverse head training.
- Evaluate methods on MountainCar and then scale to Atari (Seaquest/Montezumas Revenge) to compare sample efficiency.

## Files produced by the chapter code

- `pytorch_rl_tutorial/advanced_exploration_comparison.png` — plots comparing methods across success rates and average rewards.

---

*Summary verification: I read `pdf_text/0018_Advanced_Exploration.txt` and `pytorch_rl_tutorial/chapter_18_advanced_exploration.py` and synthesized the above mapping between theory and code.*
