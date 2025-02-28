#!/usr/bin/env python3
"""
Chapter 20: AlphaGo Zero and MuZero

This chapter implements simplified versions of AlphaGo Zero and MuZero,
demonstrating model-based RL with Monte Carlo Tree Search (MCTS) and
self-play learning without prior knowledge of game rules.

Key concepts covered:
- Monte Carlo Tree Search (MCTS) implementation
- Neural network components: representation, dynamics, prediction
- Self-play training loop
- Value and policy learning from tree search
- Model-based planning without knowing environment dynamics


"""

# Chapter 20: AlphaGo Zero and MuZero
# Start by installing the required packages:
# !pip install torch gymnasium numpy matplotlib tqdm -q

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, deque
import random
import math
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import copy

# Configure matplotlib for non-interactive mode
plt.switch_backend("Agg")
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "PingFang SC", "Arial Unicode MS", "Hiragino Sans GB"]
plt.rcParams["axes.unicode_minus"] = False

# Tic-Tac-Toe Environment
class TicTacToe:
    """Simple Tic-Tac-Toe environment for demonstrating MuZero."""
    
    def __init__(self):
        self.board_size = 3
        self.action_space = 9  # 3x3 grid
        self.reset()
        
    def reset(self):
        """Reset game to initial state."""
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)
        self.current_player = 1  # Player 1 starts
        self.done = False
        self.winner = 0
        return self.get_state()
    
    def get_state(self):
        """Get current state representation."""
        # Create 3-channel representation: empty, player1, player2
        state = np.zeros((3, self.board_size, self.board_size), dtype=np.float32)
        
        state[0] = (self.board == 0).astype(np.float32)  # Empty squares
        state[1] = (self.board == 1).astype(np.float32)  # Player 1
        state[2] = (self.board == 2).astype(np.float32)  # Player 2
        
        return state
    
    def get_valid_actions(self):
        """Get list of valid actions (empty squares)."""
        valid = []
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board[i, j] == 0:
                    valid.append(i * self.board_size + j)
        return valid
    
    def step(self, action):
        """Take action and return new state, reward, done."""
        if self.done:
            return self.get_state(), 0, True, {}
        
        # Convert action to board position
        row, col = action // self.board_size, action % self.board_size
        
        # Check if action is valid
        if self.board[row, col] != 0:
            # Invalid action
            return self.get_state(), -1, True, {"invalid_action": True}
        
        # Make move
        self.board[row, col] = self.current_player
        
        # Check for winner
        winner = self._check_winner()
        reward = 0
        
        if winner != 0:
            self.done = True
            self.winner = winner
            reward = 1 if winner == self.current_player else -1
        elif len(self.get_valid_actions()) == 0:
            # Draw
            self.done = True
            reward = 0
        else:
            # Switch players
            self.current_player = 3 - self.current_player
        
        return self.get_state(), reward, self.done, {"winner": self.winner}
    
    def _check_winner(self):
        """Check if there's a winner."""
        # Check rows
        for row in self.board:
            if row[0] == row[1] == row[2] != 0:
                return row[0]
        
        # Check columns
        for col in range(self.board_size):
            if self.board[0, col] == self.board[1, col] == self.board[2, col] != 0:
                return self.board[0, col]
        
        # Check diagonals
        if self.board[0, 0] == self.board[1, 1] == self.board[2, 2] != 0:
            return self.board[0, 0]
        if self.board[0, 2] == self.board[1, 1] == self.board[2, 0] != 0:
            return self.board[0, 2]
        
        return 0
    
    def render(self):
        """Print board state."""
        symbols = {0: '.', 1: 'X', 2: 'O'}
        for row in self.board:
            print(' '.join(symbols[cell] for cell in row))
        print()


# MCTS Node
class MCTSNode:
    """Node in Monte Carlo Tree Search."""
    
    def __init__(self, state, parent=None, action=None, prior_prob=0.0):
        self.state = state
        self.parent = parent
        self.action = action  # Action that led to this node
        self.prior_prob = prior_prob
        
        self.visit_count = 0
        self.value_sum = 0.0
        self.children = {}
        
        self.is_expanded = False
        
    def is_leaf(self):
        """Check if node is a leaf (not expanded)."""
        return not self.is_expanded
    
    def value(self):
        """Get average value of node."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
    
    def ucb_score(self, c_puct=1.0):
        """Calculate UCB score for action selection."""
        if self.visit_count == 0:
            return float('inf')
        
        exploration = c_puct * self.prior_prob * math.sqrt(self.parent.visit_count) / (1 + self.visit_count)
        return self.value() + exploration
    
    def select_child(self, c_puct=1.0):
        """Select child with highest UCB score."""
        return max(self.children.values(), key=lambda child: child.ucb_score(c_puct))
    
    def expand(self, action_probs):
        """Expand node with children for all possible actions."""
        self.is_expanded = True
        for action, prob in action_probs.items():
            self.children[action] = MCTSNode(
                state=None,  # Will be set when visited
                parent=self,
                action=action,
                prior_prob=prob
            )
    
    def backup(self, value):
        """Backup value through the tree."""
        self.visit_count += 1
        self.value_sum += value
        
        if self.parent is not None:
            self.parent.backup(-value)  # Alternate signs for two-player games


# MuZero Networks
class RepresentationNetwork(nn.Module):
    """Representation function: observation -> hidden state."""
    
    def __init__(self, observation_shape, hidden_dim=64):
        super(RepresentationNetwork, self).__init__()
        
        # For Tic-Tac-Toe: 3x3x3 input
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc = nn.Linear(64 * 3 * 3, hidden_dim)
        
    def forward(self, observation):
        x = F.relu(self.conv1(observation))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        hidden_state = torch.tanh(self.fc(x))
        return hidden_state


class DynamicsNetwork(nn.Module):
    """Dynamics function: hidden_state, action -> next_hidden_state, reward."""
    
    def __init__(self, hidden_dim=64, action_dim=9):
        super(DynamicsNetwork, self).__init__()
        
        self.action_embedding = nn.Embedding(action_dim, 32)
        self.fc1 = nn.Linear(hidden_dim + 32, 128)
        self.fc2 = nn.Linear(128, 128)
        
        # Output heads
        self.next_state_head = nn.Linear(128, hidden_dim)
        self.reward_head = nn.Linear(128, 1)
        
    def forward(self, hidden_state, action):
        action_emb = self.action_embedding(action)
        x = torch.cat([hidden_state, action_emb], dim=-1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        next_hidden_state = torch.tanh(self.next_state_head(x))
        reward = self.reward_head(x)
        
        return next_hidden_state, reward


class PredictionNetwork(nn.Module):
    """Prediction function: hidden_state -> policy, value."""
    
    def __init__(self, hidden_dim=64, action_dim=9):
        super(PredictionNetwork, self).__init__()
        
        self.fc1 = nn.Linear(hidden_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        
        # Output heads
        self.policy_head = nn.Linear(128, action_dim)
        self.value_head = nn.Linear(128, 1)
        
    def forward(self, hidden_state):
        x = F.relu(self.fc1(hidden_state))
        x = F.relu(self.fc2(x))
        
        policy_logits = self.policy_head(x)
        value = torch.tanh(self.value_head(x))
        
        return policy_logits, value


class MuZeroNetwork(nn.Module):
    """Combined MuZero network with all three components."""
    
    def __init__(self, observation_shape, action_dim=9, hidden_dim=64):
        super(MuZeroNetwork, self).__init__()
        
        self.representation = RepresentationNetwork(observation_shape, hidden_dim)
        self.dynamics = DynamicsNetwork(hidden_dim, action_dim)
        self.prediction = PredictionNetwork(hidden_dim, action_dim)
        
    def initial_inference(self, observation):
        """Initial inference from observation."""
        hidden_state = self.representation(observation)
        policy_logits, value = self.prediction(hidden_state)
        return hidden_state, policy_logits, value
    
    def recurrent_inference(self, hidden_state, action):
        """Recurrent inference for planning."""
        next_hidden_state, reward = self.dynamics(hidden_state, action)
        policy_logits, value = self.prediction(next_hidden_state)
        return next_hidden_state, reward, policy_logits, value


# MCTS with MuZero
class MuZeroMCTS:
    """Monte Carlo Tree Search using MuZero network."""
    
    def __init__(self, network, c_puct=1.0, num_simulations=100):
        self.network = network
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        
    def search(self, observation, valid_actions):
        """Run MCTS and return action probabilities."""
        # Get initial hidden state
        with torch.no_grad():
            hidden_state, policy_logits, value = self.network.initial_inference(observation)
        
        # Create root node
        root = MCTSNode(state=hidden_state.squeeze())
        
        # Get initial policy
        policy = F.softmax(policy_logits, dim=-1).squeeze()
        action_probs = {action: policy[action].item() for action in valid_actions}
        
        # Normalize probabilities
        total_prob = sum(action_probs.values())
        if total_prob > 0:
            action_probs = {action: prob / total_prob for action, prob in action_probs.items()}
        else:
            action_probs = {action: 1.0 / len(valid_actions) for action in valid_actions}
        
        root.expand(action_probs)
        
        # Run simulations
        for _ in range(self.num_simulations):
            self._simulate(root, valid_actions)
        
        # Return visit counts as action probabilities
        action_counts = {action: 0 for action in range(9)}
        for action, child in root.children.items():
            action_counts[action] = child.visit_count
        
        total_visits = sum(action_counts.values())
        if total_visits > 0:
            action_probs = {action: count / total_visits for action, count in action_counts.items()}
        else:
            action_probs = {action: 1.0 / len(valid_actions) if action in valid_actions else 0.0 
                           for action in range(9)}
        
        return action_probs, root.value()
    
    def _simulate(self, root, valid_actions):
        """Run one MCTS simulation."""
        path = []
        node = root
        
        # Selection: traverse down the tree
        while not node.is_leaf():
            action = node.select_child(self.c_puct).action
            path.append((node, action))
            node = node.children[action]
        
        # If this is a new leaf, we need to get its hidden state
        if node.state is None and len(path) > 0:
            parent_node, action = path[-1]
            with torch.no_grad():
                action_tensor = torch.tensor([action], dtype=torch.long)
                node.state, _, _, _ = self.network.recurrent_inference(
                    parent_node.state.unsqueeze(0), action_tensor
                )
                node.state = node.state.squeeze()
        
        # Evaluation and Expansion
        with torch.no_grad():
            if node.state is not None:
                policy_logits, value = self.network.prediction(node.state.unsqueeze(0))
                policy = F.softmax(policy_logits, dim=-1).squeeze()
                
                # Create action probabilities for valid actions
                action_probs = {}
                for action in valid_actions:
                    action_probs[action] = policy[action].item()
                
                # Normalize
                total_prob = sum(action_probs.values())
                if total_prob > 0:
                    action_probs = {action: prob / total_prob for action, prob in action_probs.items()}
                else:
                    action_probs = {action: 1.0 / len(valid_actions) for action in valid_actions}
                
                node.expand(action_probs)
                leaf_value = value.item()
            else:
                leaf_value = 0.0
        
        # Backup
        node.backup(leaf_value)


# Training Data
@dataclass
class GameData:
    """Data from a single game."""
    observations: List[np.ndarray]
    actions: List[int]
    rewards: List[float]
    policy_targets: List[Dict[int, float]]
    value_targets: List[float]


# MuZero Training
class MuZeroTrainer:
    """MuZero training with self-play."""
    
    def __init__(self, network, lr=1e-3):
        self.network = network
        self.optimizer = optim.Adam(network.parameters(), lr=lr)
        self.mcts = MuZeroMCTS(network)
        
        self.game_buffer = deque(maxlen=1000)
        self.training_stats = {
            "total_loss": [],
            "policy_loss": [],
            "value_loss": [],
            "reward_loss": [],
            "win_rate": []
        }
    
    def self_play(self, num_games=10):
        """Generate games through self-play."""
        for game_idx in range(num_games):
            game_data = self._play_game()
            self.game_buffer.append(game_data)
            
            if (game_idx + 1) % 5 == 0:
                print(f"  Completed {game_idx + 1}/{num_games} self-play games")
    
    def _play_game(self):
        """Play a single game using MCTS."""
        env = TicTacToe()
        
        observations = []
        actions = []
        rewards = []
        policy_targets = []
        
        state = env.reset()
        done = False
        
        while not done:
            observations.append(state.copy())
            
            # Get valid actions
            valid_actions = env.get_valid_actions()
            
            if len(valid_actions) == 0:
                break
            
            # Run MCTS
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_probs, _ = self.mcts.search(state_tensor, valid_actions)
            
            policy_targets.append(action_probs.copy())
            
            # Sample action based on visit counts with temperature
            valid_probs = [(action, action_probs.get(action, 0.0)) for action in valid_actions]
            actions_list, probs_list = zip(*valid_probs)
            
            # Temperature-based sampling (temperature = 1.0 for exploration)
            probs_array = np.array(probs_list)
            if np.sum(probs_array) > 0:
                probs_array = probs_array / np.sum(probs_array)
                action = np.random.choice(actions_list, p=probs_array)
            else:
                action = np.random.choice(valid_actions)
            
            actions.append(action)
            
            # Take step
            next_state, reward, done, info = env.step(action)
            rewards.append(reward)
            state = next_state
        
        # Calculate value targets (discounted returns)
        value_targets = []
        returns = 0
        for reward in reversed(rewards):
            returns = reward + 0.99 * returns
            value_targets.insert(0, returns)
        
        # Pad value targets to match observations
        while len(value_targets) < len(observations):
            value_targets.append(0.0)
        
        return GameData(
            observations=observations,
            actions=actions,
            rewards=rewards,
            policy_targets=policy_targets,
            value_targets=value_targets
        )
    
    def train(self, num_epochs=10, batch_size=32):
        """Train the network on collected game data."""
        if len(self.game_buffer) == 0:
            return
        
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_reward_loss = 0.0
        num_batches = 0
        
        for epoch in range(num_epochs):
            # Sample batch of games
            batch_games = random.sample(self.game_buffer, min(batch_size, len(self.game_buffer)))
            
            # Prepare batch data
            observations = []
            actions = []
            policy_targets = []
            value_targets = []
            
            for game in batch_games:
                for i in range(len(game.observations)):
                    observations.append(game.observations[i])
                    if i < len(game.actions):
                        actions.append(game.actions[i])
                        policy_targets.append(game.policy_targets[i])
                    else:
                        actions.append(0)  # Dummy action
                        policy_targets.append({j: 0.0 for j in range(9)})
                    value_targets.append(game.value_targets[i])
            
            if len(observations) == 0:
                continue
            
            # Convert to tensors
            obs_tensor = torch.FloatTensor(np.array(observations))
            actions_tensor = torch.LongTensor(actions)
            value_tensor = torch.FloatTensor(value_targets)
            
            # Policy targets
            policy_tensor = torch.zeros(len(observations), 9)
            for i, policy_dict in enumerate(policy_targets):
                for action, prob in policy_dict.items():
                    policy_tensor[i, action] = prob
            
            # Forward pass
            hidden_states, policy_logits, values = self.network.initial_inference(obs_tensor)
            
            # Losses
            policy_loss = F.cross_entropy(policy_logits, policy_tensor)
            value_loss = F.mse_loss(values.squeeze(), value_tensor)
            
            # For simplicity, we'll skip the dynamics loss in this implementation
            reward_loss = torch.tensor(0.0)
            
            loss = policy_loss + value_loss + reward_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_reward_loss += reward_loss.item()
            num_batches += 1
        
        if num_batches > 0:
            avg_loss = total_loss / num_batches
            avg_policy_loss = total_policy_loss / num_batches
            avg_value_loss = total_value_loss / num_batches
            avg_reward_loss = total_reward_loss / num_batches
            
            self.training_stats["total_loss"].append(avg_loss)
            self.training_stats["policy_loss"].append(avg_policy_loss)
            self.training_stats["value_loss"].append(avg_value_loss)
            self.training_stats["reward_loss"].append(avg_reward_loss)
            
            return avg_loss
        
        return 0.0
    
    def evaluate_against_random(self, num_games=50):
        """Evaluate trained agent against random player."""
        wins = 0
        draws = 0
        
        for _ in range(num_games):
            env = TicTacToe()
            state = env.reset()
            done = False
            
            # MuZero plays as player 1, random as player 2
            while not done:
                valid_actions = env.get_valid_actions()
                
                if len(valid_actions) == 0:
                    break
                
                if env.current_player == 1:  # MuZero's turn
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    action_probs, _ = self.mcts.search(state_tensor, valid_actions)
                    
                    # Choose best action
                    best_action = max(valid_actions, key=lambda a: action_probs.get(a, 0.0))
                    action = best_action
                else:  # Random player's turn
                    action = random.choice(valid_actions)
                
                state, reward, done, info = env.step(action)
                
                if done:
                    winner = info.get("winner", 0)
                    if winner == 1:
                        wins += 1
                    elif winner == 0:
                        draws += 1
        
        win_rate = wins / num_games
        draw_rate = draws / num_games
        
        self.training_stats["win_rate"].append(win_rate)
        
        return win_rate, draw_rate


def train_muzero():
    """Main training loop for MuZero on Tic-Tac-Toe."""
    print("=== Chapter 20: AlphaGo Zero and MuZero ===")
    print("Training simplified MuZero on Tic-Tac-Toe...\n")
    
    # Create network and trainer
    observation_shape = (3, 3, 3)  # 3 channels, 3x3 board
    network = MuZeroNetwork(observation_shape, action_dim=9, hidden_dim=64)
    trainer = MuZeroTrainer(network, lr=1e-3)
    
    print("Network architecture:")
    print(f"  Representation: {observation_shape} -> 64D hidden state")
    print(f"  Dynamics: hidden + action -> next_hidden + reward")
    print(f"  Prediction: hidden -> policy(9) + value(1)")
    print(f"  MCTS simulations: {trainer.mcts.num_simulations}\n")
    
    num_iterations = 20
    
    for iteration in range(num_iterations):
        print(f"--- Iteration {iteration + 1}/{num_iterations} ---")
        
        # Self-play
        print("Generating self-play games...")
        trainer.self_play(num_games=10)
        
        # Training
        print("Training network...")
        loss = trainer.train(num_epochs=20, batch_size=16)
        print(f"Training loss: {loss:.4f}")
        
        # Evaluation
        if (iteration + 1) % 5 == 0:
            print("Evaluating against random player...")
            win_rate, draw_rate = trainer.evaluate_against_random(num_games=30)
            print(f"Win rate: {win_rate:.1%}, Draw rate: {draw_rate:.1%}")
        
        print(f"Games in buffer: {len(trainer.game_buffer)}\n")
    
    # Final evaluation
    print("=== Final Evaluation ===")
    final_win_rate, final_draw_rate = trainer.evaluate_against_random(num_games=100)
    print(f"Final performance against random player:")
    print(f"  Win rate: {final_win_rate:.1%}")
    print(f"  Draw rate: {final_draw_rate:.1%}")
    print(f"  Loss rate: {1 - final_win_rate - final_draw_rate:.1%}")
    
    # Create training plots
    create_muzero_plots(trainer)
    
    # Demonstrate game play
    demonstrate_muzero_play(trainer)
    
    return trainer


def create_muzero_plots(trainer):
    """Create training progress plots for MuZero."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('MuZero Training Progress', fontsize=16)
    
    # Training loss
    if trainer.training_stats["total_loss"]:
        axes[0, 0].plot(trainer.training_stats["total_loss"], 'b-', linewidth=2)
        axes[0, 0].set_title('Total Training Loss')
        axes[0, 0].set_xlabel('Training Iteration')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True, alpha=0.3)
    
    # Policy and value losses
    if trainer.training_stats["policy_loss"] and trainer.training_stats["value_loss"]:
        axes[0, 1].plot(trainer.training_stats["policy_loss"], 'r-', linewidth=2, label='Policy Loss')
        axes[0, 1].plot(trainer.training_stats["value_loss"], 'g-', linewidth=2, label='Value Loss')
        axes[0, 1].set_title('Component Losses')
        axes[0, 1].set_xlabel('Training Iteration')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # Win rate over time
    if trainer.training_stats["win_rate"]:
        eval_iterations = np.arange(5, len(trainer.training_stats["win_rate"]) * 5 + 1, 5)
        axes[1, 0].plot(eval_iterations, trainer.training_stats["win_rate"], 'g-', linewidth=2, marker='o')
        axes[1, 0].set_title('Win Rate vs Random Player')
        axes[1, 0].set_xlabel('Training Iteration')
        axes[1, 0].set_ylabel('Win Rate')
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].grid(True, alpha=0.3)
    
    # Learning progress summary
    if trainer.training_stats["total_loss"] and trainer.training_stats["win_rate"]:
        # Normalize losses for comparison
        norm_loss = np.array(trainer.training_stats["total_loss"])
        norm_loss = (norm_loss - norm_loss.min()) / (norm_loss.max() - norm_loss.min()) if norm_loss.max() > norm_loss.min() else norm_loss
        
        axes[1, 1].plot(norm_loss, 'b-', linewidth=2, label='Normalized Loss')
        
        # Plot win rate at evaluation points
        if trainer.training_stats["win_rate"]:
            eval_points = np.arange(4, len(norm_loss), 5)  # Every 5th iteration
            win_rates_at_eval = trainer.training_stats["win_rate"]
            if len(eval_points) == len(win_rates_at_eval):
                axes[1, 1].plot(eval_points, win_rates_at_eval, 'r-', linewidth=2, marker='o', label='Win Rate')
        
        axes[1, 1].set_title('Learning Progress')
        axes[1, 1].set_xlabel('Training Iteration')
        axes[1, 1].set_ylabel('Normalized Value')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pytorch_rl_tutorial/chapter_20_muzero_training_progress.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\nMuZero training plots saved as 'muzero_training_progress.png'")


def demonstrate_muzero_play(trainer):
    """Demonstrate MuZero playing Tic-Tac-Toe."""
    print("\n=== MuZero Game Demonstration ===")
    
    # Play a few example games
    for game_num in range(3):
        print(f"\nGame {game_num + 1}: MuZero (X) vs Random (O)")
        
        env = TicTacToe()
        state = env.reset()
        move_count = 0
        
        print("Initial board:")
        env.render()
        
        while not env.done:
            valid_actions = env.get_valid_actions()
            
            if len(valid_actions) == 0:
                break
            
            if env.current_player == 1:  # MuZero's turn
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                
                # Get action probabilities from MCTS
                with torch.no_grad():
                    action_probs, mcts_value = trainer.mcts.search(state_tensor, valid_actions)
                
                # Show MCTS analysis
                print(f"MuZero's turn (Move {move_count + 1}):")
                print(f"  MCTS value estimate: {mcts_value:.3f}")
                print(f"  Action probabilities:")
                for action in valid_actions:
                    row, col = action // 3, action % 3
                    prob = action_probs.get(action, 0.0)
                    print(f"    Position ({row},{col}): {prob:.3f}")
                
                # Choose best action
                best_action = max(valid_actions, key=lambda a: action_probs.get(a, 0.0))
                action = best_action
                
                row, col = action // 3, action % 3
                print(f"  MuZero chooses: ({row},{col})")
                
            else:  # Random player's turn
                action = random.choice(valid_actions)
                row, col = action // 3, action % 3
                print(f"Random player chooses: ({row},{col})")
            
            # Take step
            state, reward, done, info = env.step(action)
            move_count += 1
            
            print("Board after move:")
            env.render()
            
            if done:
                winner = info.get("winner", 0)
                if winner == 1:
                    print("MuZero wins!")
                elif winner == 2:
                    print("Random player wins!")
                else:
                    print("Draw!")
                break


def analyze_mcts_behavior(trainer):
    """Analyze MCTS search behavior."""
    print("\n=== MCTS Behavior Analysis ===")
    
    # Create test position
    env = TicTacToe()
    
    # Set up an interesting position
    env.board = np.array([
        [1, 0, 2],
        [0, 1, 0],
        [2, 0, 0]
    ])
    env.current_player = 1
    
    state = env.get_state()
    valid_actions = env.get_valid_actions()
    
    print("Test position:")
    env.render()
    print(f"Current player: {env.current_player} (X)")
    print(f"Valid actions: {valid_actions}")
    
    # Run MCTS with different simulation counts
    simulation_counts = [10, 50, 100, 200]
    
    for sim_count in simulation_counts:
        trainer.mcts.num_simulations = sim_count
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_probs, mcts_value = trainer.mcts.search(state_tensor, valid_actions)
        
        print(f"\nMCTS with {sim_count} simulations:")
        print(f"  Value estimate: {mcts_value:.3f}")
        print(f"  Action preferences:")
        
        sorted_actions = sorted(valid_actions, key=lambda a: action_probs.get(a, 0.0), reverse=True)
        for action in sorted_actions:
            row, col = action // 3, action % 3
            prob = action_probs.get(action, 0.0)
            print(f"    Position ({row},{col}): {prob:.3f}")


if __name__ == "__main__":
    print("Chapter 20: AlphaGo Zero and MuZero")
    print("="*50)
    print("This chapter demonstrates model-based RL with MCTS.")
    print("We'll implement a simplified MuZero for Tic-Tac-Toe.\n")
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Train MuZero
    trainer = train_muzero()
    
    # Analyze MCTS behavior
    analyze_mcts_behavior(trainer)
    
    print("\n" + "="*50)
    print("Key Concepts Demonstrated:")
    print("- Monte Carlo Tree Search (MCTS) implementation")
    print("- Neural network components: representation, dynamics, prediction")
    print("- Self-play training without prior game knowledge")
    print("- UCB-based action selection in tree search")
    print("- Value and policy learning from search results")
    print("\nMuZero/AlphaZero principles apply to:")
    print("- Board games (Go, Chess, Shogi)")
    print("- Strategic planning problems")
    print("- Model-based RL in complex domains")
    print("- Any domain where planning is beneficial")
    print("\nNext: Chapter 21 - RL in Discrete Optimization")
