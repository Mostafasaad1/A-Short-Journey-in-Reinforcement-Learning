# Chapter 12: Actor-Critic Methods — A2C and A3C

## Theory Summary (deep dive)

Actor-Critic methods combine policy-based and value-based ideas. Instead of using full-episode Monte Carlo returns like REINFORCE, actor-critic uses a learned value function V_phi(s) (the critic) as a baseline for policy updates (the actor). The canonical objective for the actor is:

\[\n\nJ(\theta) \approx E_{\pi_\theta}[\nabla_\theta \log \pi_\theta(a|s) \; A(s,a)\,],\n\]

where the advantage A(s,a) = Q(s,a) - V_phi(s). The critic is trained to minimize a temporal-difference (TD) loss, e.g. MSE of the TD target:

\[\nL_v(\phi) = E[(r + \gamma V_\phi(s') - V_\phi(s))^2].\n\]

Key variance reduction and bias trade-offs:
- Using V(s) as baseline removes much of the high-variance scale of returns; however, the critic introduces approximation bias. Practically the reduction in variance almost always helps learning speed and stability.
- Bootstrapping (TD) reduces variance but can introduce bias. A2C typically uses n-step returns (reward_steps) to balance variance and bias; A3C extends this to asynchronous, parallel workers.

Generalized Advantage Estimation (GAE):
- GAE computes a weighted sum of k-step TD residuals to create a smoother advantage estimator with a lambda parameter (0 <= lambda <= 1):

\[\n\hat{A}_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l},\quad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)\n\]

- Lambda trades bias (low lambda) for variance (higher lambda).

Entropy regularization:
- Add -beta * H(\pi) to the loss (or +beta * entropy depending on sign convention) to keep the policy stochastic early in training and encourage exploration.

Parallelization approaches:
- Data parallelism: workers collect trajectories and send samples to a central learner that computes gradients and updates weights; updated weights are broadcast back.
- Gradient parallelism: workers compute gradients locally and send gradient tensors to master, which aggregates (sums) gradients and applies an optimizer step. This scales better when gradient computation is the bottleneck.

Practical stabilizers:
- Normalize advantages before applying the policy gradient.
- Clip gradients (L2 norm) to avoid destructive updates.
- Use Adam with tuned epsilon; default eps sometimes destabilizes training in practice for A2C/A3C (book notes using larger eps).
- Vectorized environments (SyncVectorEnv / AsyncVectorEnv) reduce sample correlation and increase throughput.

## Code Implementation Breakdown (mapping to files)

File: `pytorch_rl_tutorial/chapter_12_actor_critic.py`

- ActorCriticNetwork(nn.Module)
  - Shared feature extractor (two Linear+ReLU layers) and two output heads:
    - actor_head: outputs logits for discrete actions
    - critic_head: outputs scalar state value
  - Methods of interest:
    - forward(state) -> (action_logits, state_value)
    - get_action_and_value(state) -> (action, log_prob, entropy, value) — builds a Categorical distribution via logits; samples action and returns log_prob and entropy.
    - evaluate_actions(states, actions) -> (log_probs, values, entropies) — used during update to compute new log-probs and values for the batch.

- A2CAgent
  - select_action(state): runs network.get_action_and_value under torch.no_grad() for interaction and returns action/log_prob/value.
  - compute_gae(rewards, values, dones, next_value, gae_lambda): detailed GAE implementation; iterates backwards and builds advantages and returns.
  - update(...): core update uses:
    - advantages, returns = compute_gae(...)
    - convert to tensors and normalize advantages
    - evaluate_actions to get new_log_probs, new_values, entropies
    - policy_loss = -(new_log_probs * advantages).mean()
    - value_loss = MSE(new_values, returns)
    - entropy_loss = -entropies.mean()
    - total_loss = policy_loss + value_coeff * value_loss + entropy_coeff * entropy_loss
    - backward, clip_grad_norm_, optimizer.step()
  - compute_gae implementation details (line-level logic): handles next_value bootstrap, respects dones flags with next_non_terminal multiplier, inserts advantages and returns in reversed loop.

- Training loop (train_a2c)
  - Collect rollouts of up to update_frequency steps per environment (or using vectorized env, across envs).
  - After collecting batch, compute next_value (0 if done else critic(next_state)), call agent.update() to apply step.
  - Logging every 100 episodes for average reward and loss.

- A3C components
  - A3CWorker: simplified concept worker showing local network sync and local compute of loss. Key methods:
    - sync_with_global(): copy weights from shared global network
    - compute_loss(states, actions, rewards, dones, next_value): creates returns by bootstrapping with next_value, computes advantages = returns - values, then policy_loss, value_loss, entropy_loss, and returns total_loss.
  - The book provides two A3C modes: data parallelism (replace SyncVectorEnv with AsyncVectorEnv to run envs in processes) and gradient parallelism (child processes compute gradients and master aggregates them).

## Line-level walkthrough (important snippets)

I'll call out the most critical blocks and explain why they are written that way.

1) GAE loop (A2CAgent.compute_gae)
- Reverse iterator pattern (for i in reversed(range(len(rewards)))) ensures future rewards are folded into earlier timesteps.
- next_non_terminal = 1.0 - dones[i] protects bootstrap when episode ended inside the batch.
- delta = r_i + gamma * next_value_est * next_non_terminal - values[i]
- gae = delta + gamma * lambda * next_non_terminal * gae
- advantages.insert(0, gae); returns.insert(0, gae + values[i])

Why this matters: using value estimates for bootstrapping reduces variance and GAE mixes multiple step residuals to control bias/variance.

2) Normalizing advantages (Agent.update)
- advantages_tensor = (advantages - mean) / (std + 1e-8)

Why: centers gradients to stable numerical scale; standard best-practice.

3) Detach critic when computing policy loss in A2C (in A2C classic code)
- policy uses advantages computed from returns which are detached from critic updates or uses advantages.detach() to prevent policy loss backprop into value head.

Why: ensure actor and critic losses influence appropriate parameters consistently, preventing double-counting gradients.

4) Two-stage backward to measure policy gradient statistics (from PDF):
- loss_policy.backward(retain_graph=True)
- read p.grad for all params to compute grad norms/variance
- then compute entropy and value losses and call backward() again

Why: The book extracts gradients contributed only by policy loss before adding entropy/value loss so it can log gradient variance specifically for the policy gradient component.

5) Adam eps note (from chapter)
- The author observed that default Adam eps (1e-8) caused instability; increasing eps stabilized training. If you reproduce training and see divergence, try eps=1e-5..1e-3.
