#!/usr/bin/env python3
"""
Chapter 13: The TextWorld Environment

This chapter demonstrates how to train an RL agent to solve text-based games using
a PyTorch-based LSTM/Transformer policy. We'll work with natural language observations
and action spaces to tackle the challenges of partial observability and language grounding.

Key concepts covered:
- Text preprocessing and tokenization for RL
- LSTM/Transformer policies for text environments
- Action masking and command generation
- Attention mechanisms for inventory and location tracking
- Handling sparse rewards in text environments


"""

# Chapter 13: The TextWorld Environment
# Start by installing the required packages:
# !pip install torch textworld gymnasium numpy matplotlib -q

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import numpy as np
import matplotlib.pyplot as plt
from collections import deque, defaultdict
import random
import re
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(
    level=logging.DEBUG,  # or INFO if you want less detail
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Configure matplotlib for non-interactive mode
plt.switch_backend("Agg")
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "PingFang SC", "Arial Unicode MS", "Hiragino Sans GB"]
plt.rcParams["axes.unicode_minus"] = False

# TextWorld Environment Simulation
# Since TextWorld might not be available, we'll create a simplified text-based environment

class SimpleTextWorld:
    """
    A simplified text-based environment that mimics TextWorld mechanics.
    The agent must navigate rooms, collect items, and achieve goals through text commands.
    """
    
    def __init__(self, max_steps=50):
        self.max_steps = max_steps
        self.reset()
        
        # Define vocabulary for actions and objects
        self.action_templates = [
            "go {direction}",
            "take {item}",
            "drop {item}",
            "examine {item}",
            "use {item}",
            "open {container}",
            "look"
        ]
        
        self.directions = ["north", "south", "east", "west"]
        self.items = ["key", "book", "sword", "potion", "coin", "chest"]
        
        # Build action vocabulary
        self.actions = ["look"]
        for direction in self.directions:
            self.actions.append(f"go {direction}")
        for item in self.items:
            self.actions.extend([
                f"take {item}",
                f"drop {item}",
                f"examine {item}",
                f"use {item}",
                f"open {item}"
            ])
        
        self.action_to_idx = {action: idx for idx, action in enumerate(self.actions)}
        self.vocab_size = len(self.actions)
        
    def reset(self):
        """Reset the environment to initial state."""
        self.current_room = "entrance"
        self.inventory = set()
        self.step_count = 0
        self.game_state = {
            "entrance": {"items": {"key"}, "exits": {"north": "hallway"}},
            "hallway": {"items": {"book"}, "exits": {"south": "entrance", "east": "treasure_room"}},
            "treasure_room": {"items": {"chest"}, "exits": {"west": "hallway"}}
        }
        self.goal_achieved = False
        return self._get_observation()
    
    def _get_observation(self):
        """Generate text observation of current state."""
        room_info = self.game_state[self.current_room]
        
        obs = f"You are in the {self.current_room}. "
        
        if room_info["items"]:
            items_str = ", ".join(room_info["items"])
            obs += f"You see: {items_str}. "
        else:
            obs += "The room is empty. "
            
        exits_str = ", ".join(room_info["exits"].keys())
        obs += f"Exits: {exits_str}. "
        
        if self.inventory:
            inv_str = ", ".join(self.inventory)
            obs += f"Inventory: {inv_str}."
        else:
            obs += "Inventory is empty."
            
        return obs
    
    def step(self, action_idx):
        """Execute action and return observation, reward, done."""
        if action_idx >= len(self.actions):
            return self._get_observation(), -0.1, False, {"invalid": True}
            
        action = self.actions[action_idx]
        reward = 0
        done = False
        info = {}
        
        self.step_count += 1
        
        # Parse and execute action
        if action == "look":
            reward = 0  # No penalty for looking
        elif action.startswith("go "):
            direction = action.split()[1]
            room_info = self.game_state[self.current_room]
            if direction in room_info["exits"]:
                self.current_room = room_info["exits"][direction]
                reward = 0.1  # Small reward for successful movement
            else:
                reward = -0.1  # Penalty for invalid movement
        elif action.startswith("take "):
            item = action.split()[1]
            room_info = self.game_state[self.current_room]
            if item in room_info["items"]:
                room_info["items"].remove(item)
                self.inventory.add(item)
                reward = 0.2  # Reward for taking items
            else:
                reward = -0.1  # Penalty for trying to take non-existent item
        elif action.startswith("use "):
            item = action.split()[1]
            if item in self.inventory:
                if item == "key" and self.current_room == "treasure_room" and "chest" in self.game_state["treasure_room"]["items"]:
                    self.goal_achieved = True
                    reward = 10.0  # Large reward for achieving goal
                    done = True
                else:
                    reward = -0.05  # Small penalty for ineffective use
            else:
                reward = -0.1  # Penalty for using item not in inventory
        else:
            reward = -0.05  # Small penalty for other actions
        
        # Check termination conditions
        if self.step_count >= self.max_steps:
            done = True
        
        # Time penalty
        reward -= 0.01
        
        return self._get_observation(), reward, done, info
    
    def get_valid_actions(self):
        """Return mask of valid actions for current state."""
        valid_mask = np.zeros(len(self.actions), dtype=bool)
        
        # Always valid actions
        valid_mask[self.action_to_idx["look"]] = True
        
        # Valid movement actions
        room_info = self.game_state[self.current_room]
        for direction in room_info["exits"]:
            action = f"go {direction}"
            if action in self.action_to_idx:
                valid_mask[self.action_to_idx[action]] = True
        
        # Valid take actions
        for item in room_info["items"]:
            action = f"take {item}"
            if action in self.action_to_idx:
                valid_mask[self.action_to_idx[action]] = True
        
        # Valid use actions (items in inventory)
        for item in self.inventory:
            action = f"use {item}"
            if action in self.action_to_idx:
                valid_mask[self.action_to_idx[action]] = True
        
        return valid_mask


class TextTokenizer:
    """Simple tokenizer for text observations."""
    
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.word_to_idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx_to_word = {0: "<PAD>", 1: "<UNK>"}
        self.word_counts = defaultdict(int)
        
    def build_vocab(self, texts):
        """Build vocabulary from text corpus."""
        # Count words
        for text in texts:
            words = self._tokenize_text(text)
            for word in words:
                self.word_counts[word] += 1
        
        # Add most frequent words to vocabulary
        sorted_words = sorted(self.word_counts.items(), key=lambda x: x[1], reverse=True)
        for word, count in sorted_words[:self.vocab_size - 2]:  # Reserve space for PAD and UNK
            idx = len(self.word_to_idx)
            self.word_to_idx[word] = idx
            self.idx_to_word[idx] = word
    
    def _tokenize_text(self, text):
        """Simple tokenization: lowercase and split on whitespace and punctuation."""
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        return text.split()
    
    def encode(self, text, max_length=50):
        """Encode text to indices."""
        words = self._tokenize_text(text)
        indices = []
        
        for word in words:
            if word in self.word_to_idx:
                indices.append(self.word_to_idx[word])
            else:
                indices.append(self.word_to_idx["<UNK>"])
        
        # Pad or truncate to max_length
        if len(indices) < max_length:
            indices.extend([0] * (max_length - len(indices)))
        else:
            indices = indices[:max_length]
            
        return indices
    
    def decode(self, indices):
        """Decode indices back to text."""
        words = []
        for idx in indices:
            if idx in self.idx_to_word and idx != 0:  # Skip padding
                words.append(self.idx_to_word[idx])
        return " ".join(words)


class LSTMPolicy(nn.Module):
    """LSTM-based policy network for text environments."""
    
    def __init__(self, vocab_size, embedding_dim=64, hidden_dim=128, num_actions=50):
        super(LSTMPolicy, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        
        # Policy and value heads
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions)
        )
        
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, text_indices, valid_actions_mask=None):
        # Embed text
        embedded = self.embedding(text_indices)  # [batch, seq_len, embed_dim]
        
        # LSTM encoding
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Self-attention over sequence
        attended_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Use last timestep for policy and value
        final_hidden = attended_out[:, -1, :]  # [batch, hidden_dim]
        
        # Policy logits
        policy_logits = self.policy_head(final_hidden)
        
        # Apply action masking
        if valid_actions_mask is not None:
            # Set invalid actions to large negative value
            policy_logits = policy_logits.masked_fill(~valid_actions_mask, -1e9)
        
        # Value estimate
        value = self.value_head(final_hidden)
        
        return policy_logits, value


class REINFORCETextAgent:
    """REINFORCE agent for text environments with action masking."""
    
    def __init__(self, env, tokenizer, vocab_size, num_actions, lr=1e-3):
        self.env = env
        self.tokenizer = tokenizer
        self.policy = LSTMPolicy(vocab_size, num_actions=num_actions)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        self.episode_rewards = []
        self.episode_lengths = []

    def select_action(self, observation, training=True):
        """Select action using current policy.

        If training=True, we allow gradients (no torch.no_grad()) so
        log_prob and value remain connected to the graph.
        If training=False (evaluation), we use torch.no_grad().
        """
        # Tokenize observation
        text_indices = torch.tensor([self.tokenizer.encode(observation)], dtype=torch.long)

        # Get valid actions mask
        valid_actions = self.env.get_valid_actions()
        valid_mask = torch.from_numpy(np.array(valid_actions)).bool().unsqueeze(0)  # [1, num_actions]

        if training:
            policy_logits, value = self.policy(text_indices, valid_mask)
        else:
            with torch.no_grad():
                policy_logits, value = self.policy(text_indices, valid_mask)

        # Sample action
        if training:
            action_probs = F.softmax(policy_logits, dim=-1)
            action_dist = Categorical(action_probs)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)    # tensor, requires_grad depends on graph
        else:
            action = torch.argmax(policy_logits, dim=-1)
            log_prob = None

        # Return action as int, log_prob tensor or None, and value tensor (not .item())
        return action.item(), log_prob, value.squeeze(0)  # value shape [1,1] -> [1] or scalar tensor

    def train_episode(self):
        """Train on a single episode using REINFORCE (fixed to keep grads)."""
        observation = self.env.reset()

        states = []
        actions = []
        log_probs = []
        rewards = []
        values = []

        done = False
        step = 0

        while not done:
            # Select action (training=True to keep grads)
            action, log_prob, value = self.select_action(observation, training=True)

            # Store experience
            states.append(observation)
            actions.append(action)

            # log_prob is a tensor; append it (it will be used for policy loss)
            if log_prob is None:
                # should not happen in training mode, but keep safe
                log_probs.append(torch.tensor(0.0, requires_grad=True))
            else:
                log_probs.append(log_prob)

            # value should be a tensor (not .item()); keep as tensor
            values.append(value)

            # Take step (environment steps are non-differentiable)
            observation, reward, done, info = self.env.step(action)
            rewards.append(reward)
            step += 1

        # Calculate returns (numpy or list)
        returns = self._calculate_returns(rewards)
        returns = torch.tensor(returns, dtype=torch.float32)  # shape [T]

        # Stack log_probs and values into tensors
        log_probs = torch.stack(log_probs)            # shape [T]
        values = torch.stack(values).squeeze(-1)      # shape [T] (value head output shape was [1] per step)

        # Compute advantages (use values.detach() for policy loss)
        advantages = returns - values

        # Policy loss: use detached advantages to avoid backprop through return computation
        policy_loss = -(log_probs * advantages.detach()).mean()

        # Value loss: update value head so do NOT detach values
        value_loss = F.mse_loss(values, returns)

        # Total loss
        total_loss = policy_loss + 0.5 * value_loss

        # Backprop and step
        self.optimizer.zero_grad()

        logger.debug(f"total_loss: {total_loss}")
        logger.debug(f"requires_grad: {total_loss.requires_grad}")
        logger.debug(f"grad_fn: {total_loss.grad_fn}")

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Store episode statistics
        episode_reward = sum(rewards)
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(len(rewards))

        return {
            "episode_reward": episode_reward,
            "episode_length": len(rewards),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "goal_achieved": self.env.goal_achieved
        }



    def _calculate_returns(self, rewards, gamma=0.99):
        """Calculate discounted returns."""
        returns = []
        G = 0
        for reward in reversed(rewards):
            G = reward + gamma * G
            returns.insert(0, G)
        return returns


def collect_vocabulary_data(env, num_episodes=100):
    """Collect text observations to build vocabulary."""
    texts = []
    
    for _ in range(num_episodes):
        obs = env.reset()
        texts.append(obs)
        
        done = False
        while not done:
            # Random action
            valid_actions = env.get_valid_actions()
            valid_indices = np.where(valid_actions)[0]
            if len(valid_indices) > 0:
                action = np.random.choice(valid_indices)
            else:
                action = 0
            
            obs, _, done, _ = env.step(action)
            texts.append(obs)
    
    return texts


def train_text_agent(num_episodes=2000):
    """Main training loop for text-based RL agent."""
    print("=== Chapter 13: The TextWorld Environment ===")
    print("Training LSTM-based REINFORCE agent for text-based games...\n")
    
    # Create environment
    env = SimpleTextWorld(max_steps=50)
    print(f"Environment: SimpleTextWorld")
    print(f"Action space: {len(env.actions)} discrete actions")
    print(f"Goal: Use key to open chest in treasure room\n")
    
    # Build vocabulary
    print("Building vocabulary from environment...")
    vocab_texts = collect_vocabulary_data(env, num_episodes=200)
    tokenizer = TextTokenizer(vocab_size=500)
    tokenizer.build_vocab(vocab_texts)
    print(f"Vocabulary size: {len(tokenizer.word_to_idx)}\n")
    
    # Create agent
    agent = REINFORCETextAgent(
        env=env,
        tokenizer=tokenizer,
        vocab_size=len(tokenizer.word_to_idx),
        num_actions=len(env.actions),
        lr=3e-4
    )
    
    # Training statistics
    success_rates = []
    avg_rewards = []
    avg_lengths = []
    
    print("Starting training...")
    for episode in range(num_episodes):
        result = agent.train_episode()
        
        # Log progress
        if (episode + 1) % 100 == 0:
            recent_rewards = agent.episode_rewards[-100:]
            recent_successes = sum(1 for i in range(max(0, len(agent.episode_rewards) - 100), len(agent.episode_rewards)) 
                                 if i < len(agent.episode_rewards) and env.reset() and agent.select_action(env._get_observation(), False)[0] and env.goal_achieved)
            
            avg_reward = np.mean(recent_rewards)
            avg_length = np.mean(agent.episode_lengths[-100:])
            
            # Calculate success rate over recent episodes
            success_count = 0
            for _ in range(20):  # Test 20 episodes
                test_obs = env.reset()
                test_done = False
                test_steps = 0
                while not test_done and test_steps < env.max_steps:
                    test_action, _, _ = agent.select_action(test_obs, training=False)
                    test_obs, _, test_done, _ = env.step(test_action)
                    test_steps += 1
                if env.goal_achieved:
                    success_count += 1
            
            success_rate = success_count / 20
            
            success_rates.append(success_rate)
            avg_rewards.append(avg_reward)
            avg_lengths.append(avg_length)
            
            print(f"Episode {episode+1:4d} | Avg Reward: {avg_reward:6.2f} | "
                  f"Avg Length: {avg_length:5.1f} | Success Rate: {success_rate:5.1%} | "
                  f"Policy Loss: {result['policy_loss']:6.3f}")
    
    print("\nTraining completed!")
    
    # Test final policy
    print("\nTesting final policy...")
    test_successes = 0
    test_episodes = 50
    test_rewards = []
    
    for eps in range(test_episodes):
        obs = env.reset()
        total_reward = 0
        done = False
        steps = 0
        
        print(f"\nTest Episode - Initial state: {obs}")
        
        while not done and steps < env.max_steps:
            action, _, _ = agent.select_action(obs, training=False)
            action_name = env.actions[action]
            obs, reward, done, _ = env.step(action)
            total_reward += reward
            steps += 1
            
            if steps <= 10:  # Show first 10 steps
                print(f"Step {steps}: {action_name} -> Reward: {reward:.2f}")
                print(f"         State: {obs}")
        
        test_rewards.append(total_reward)
        if env.goal_achieved:
            test_successes += 1
            print(f"SUCCESS! Goal achieved in {steps} steps")
        else:
            print(f"Failed - Goal not achieved in {steps} steps")
        
        if eps >= 2:  
            break
    
    final_success_rate = test_successes / test_episodes
    final_avg_reward = np.mean(test_rewards)
    
    print(f"\nFinal Performance:")
    print(f"Success Rate: {final_success_rate:.1%}")
    print(f"Average Reward: {final_avg_reward:.2f}")
    
    # Visualizations
    create_text_training_plots(success_rates, avg_rewards, avg_lengths)
    
    return agent, env, tokenizer


def create_text_training_plots(success_rates, avg_rewards, avg_lengths):
    """Create training progress visualizations."""
    episodes = np.arange(100, len(success_rates) * 100 + 1, 100)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('TextWorld Training Progress', fontsize=16)
    
    # Success rate
    axes[0, 0].plot(episodes, success_rates, 'g-', linewidth=2)
    axes[0, 0].set_title('Success Rate Over Training')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Success Rate')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim(0, 1)
    
    # Average reward
    axes[0, 1].plot(episodes, avg_rewards, 'b-', linewidth=2)
    axes[0, 1].set_title('Average Reward Over Training')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Average Reward')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Episode length
    axes[1, 0].plot(episodes, avg_lengths, 'r-', linewidth=2)
    axes[1, 0].set_title('Average Episode Length')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Steps per Episode')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Learning progression
    axes[1, 1].plot(episodes, success_rates, 'g-', linewidth=2, label='Success Rate')
    axes[1, 1].set_ylabel('Success Rate', color='g')
    axes[1, 1].tick_params(axis='y', labelcolor='g')
    
    ax2 = axes[1, 1].twinx()
    ax2.plot(episodes, avg_rewards, 'b-', linewidth=2, label='Avg Reward')
    ax2.set_ylabel('Average Reward', color='b')
    ax2.tick_params(axis='y', labelcolor='b')
    
    axes[1, 1].set_title('Success Rate vs Reward')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pytorch_rl_tutorial/textworld_training_progress.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\nTraining plots saved as 'textworld_training_progress.png'")


def demonstrate_attention_mechanism(agent, env, tokenizer):
    """Demonstrate how the attention mechanism focuses on different parts of text."""
    print("\n=== Attention Mechanism Analysis ===")
    
    # Test different observations
    test_observations = [
        "You are in the entrance. You see: key. Exits: north. Inventory is empty.",
        "You are in the hallway. You see: book. Exits: south, east. Inventory: key.",
        "You are in the treasure_room. You see: chest. Exits: west. Inventory: key."
    ]
    
    for obs in test_observations:
        print(f"\nObservation: {obs}")
        
        # Tokenize
        text_indices = torch.tensor([tokenizer.encode(obs)], dtype=torch.long)
        valid_actions = env.get_valid_actions()
        valid_mask = torch.from_numpy(np.array(valid_actions)).bool()
        
        # Get policy output
        with torch.no_grad():
            policy_logits, value = agent.policy(text_indices, valid_mask)
            action_probs = F.softmax(policy_logits, dim=-1)
        
        # Show top actions
        top_actions_idx = torch.topk(action_probs, k=3, dim=-1)[1][0]
        print("Top 3 predicted actions:")
        for i, action_idx in enumerate(top_actions_idx):
            action_name = env.actions[action_idx]
            prob = action_probs[0, action_idx].item()
            print(f"  {i+1}. {action_name} (prob: {prob:.3f})")
        
        print(f"Value estimate: {value.item():.3f}")


if __name__ == "__main__":
    print("Chapter 13: The TextWorld Environment")
    print("="*50)
    print("This chapter demonstrates RL in text-based environments.")
    print("We'll train an LSTM-based agent to navigate and solve text games.\n")
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Train the agent
    agent, env, tokenizer = train_text_agent(num_episodes=1000)
    
    # Demonstrate attention mechanism
    demonstrate_attention_mechanism(agent, env, tokenizer)
    
    print("\n" + "="*50)
    print("Key Concepts Demonstrated:")
    print("- Text preprocessing and tokenization for RL")
    print("- LSTM policies with attention mechanisms")
    print("- Action masking for valid command generation")
    print("- Handling sparse rewards in text environments")
    print("- Language grounding and partial observability")
    print("\nText-based RL opens up applications in:")
    print("- Interactive fiction and game playing")
    print("- Natural language interfaces")
    print("- Educational tutoring systems")
    print("- Conversational AI and dialogue systems")
    print("\nNext: Chapter 14 - Web Navigation")
