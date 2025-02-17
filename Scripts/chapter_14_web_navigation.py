#!/usr/bin/env python3
"""
Chapter 14: Web Navigation

This chapter demonstrates how to create a PyTorch agent that navigates websites
using DOM state representation. We'll implement an agent that can interact with
web elements through clicks, typing, and focus actions.

Key concepts covered:
- DOM state encoding using CNNs and graph networks
- Multi-modal state representation (visual + structural)
- Action space design for web interaction
- Curriculum learning for complex web tasks
- Sparse reward shaping and task completion metrics


"""

# Chapter 14: Web Navigation
# Start by installing the required packages:
# !pip install torch gym-minigrid selenium numpy matplotlib opencv-python -q

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import numpy as np
import matplotlib.pyplot as plt
from collections import deque, namedtuple
import random
import json
from typing import Dict, List, Tuple, Optional, Any
import cv2
from dataclasses import dataclass

# Configure matplotlib for non-interactive mode
plt.switch_backend("Agg")
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "PingFang SC", "Arial Unicode MS", "Hiragino Sans GB"]
plt.rcParams["axes.unicode_minus"] = False

# Web Environment Simulation
# Since actual web automation requires complex setup, we'll simulate web interactions

@dataclass
class WebElement:
    """Represents a web element with position and attributes."""
    x: int
    y: int
    width: int
    height: int
    tag: str
    text: str
    clickable: bool
    input_type: Optional[str] = None
    required_text: Optional[str] = None


class SimpleWebEnvironment:
    """
    A simplified web environment that simulates common web tasks like
    form filling, navigation, and information extraction.
    """
    
    def __init__(self, screen_width=800, screen_height=600):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.reset()
        
        # Define action space
        self.actions = {
            0: "click",
            1: "type_text", 
            2: "scroll_up",
            3: "scroll_down",
            4: "focus_next",
            5: "submit_form",
            6: "navigate_back"
        }
        
        self.action_space_size = len(self.actions)
        
    def reset(self):
        """Reset environment to initial state."""
        self.current_page = "login"
        self.scroll_position = 0
        self.form_data = {}
        self.focused_element = None
        self.step_count = 0
        self.task_completed = False
        
        # Initialize page elements
        self._setup_pages()
        return self._get_state()
    
    def _setup_pages(self):
        """Setup different web pages with their elements."""
        self.pages = {
            "login": {
                "elements": [
                    WebElement(100, 100, 200, 30, "input", "Username", True, "text", "admin"),
                    WebElement(100, 150, 200, 30, "input", "Password", True, "password", "password123"),
                    WebElement(100, 200, 100, 40, "button", "Login", True),
                    WebElement(250, 200, 100, 40, "button", "Register", True)
                ],
                "goal": "Login with correct credentials"
            },
            "dashboard": {
                "elements": [
                    WebElement(50, 50, 150, 30, "link", "Profile", True),
                    WebElement(220, 50, 150, 30, "link", "Settings", True),
                    WebElement(390, 50, 150, 30, "link", "Logout", True),
                    WebElement(100, 150, 300, 100, "div", "Welcome to Dashboard!", False),
                    WebElement(100, 300, 200, 40, "button", "Complete Task", True)
                ],
                "goal": "Navigate to complete task"
            },
            "task_page": {
                "elements": [
                    WebElement(100, 100, 400, 30, "div", "Task: Fill out the form below", False),
                    WebElement(100, 150, 200, 30, "input", "Name", True, "text", "John Doe"),
                    WebElement(100, 200, 200, 30, "input", "Email", True, "email", "john@example.com"),
                    WebElement(100, 250, 200, 80, "textarea", "Comments", True, "text", "Great service!"),
                    WebElement(100, 350, 100, 40, "button", "Submit", True),
                    WebElement(220, 350, 100, 40, "button", "Cancel", True)
                ],
                "goal": "Fill form and submit"
            }
        }
    
    def _get_state(self):
        """Get current state representation."""
        # Create visual representation (simplified screenshot)
        visual_state = self._create_visual_state()
        
        # Create DOM representation
        dom_state = self._create_dom_state()
        
        return {
            "visual": visual_state,
            "dom": dom_state,
            "page": self.current_page,
            "scroll": self.scroll_position,
            "focused": self.focused_element,
            "form_data": self.form_data.copy()
        }
    
    def _create_visual_state(self):
        """Create simplified visual representation of the page."""
        # Create a simple RGB image representation
        image = np.ones((84, 84, 3), dtype=np.uint8) * 255  # White background
        
        page = self.pages[self.current_page]
        for i, element in enumerate(page["elements"]):
            # Scale coordinates to fit 84x84 image
            x = int(element.x * 84 / self.screen_width)
            y = int(element.y * 84 / self.screen_height)
            w = max(1, int(element.width * 84 / self.screen_width))
            h = max(1, int(element.height * 84 / self.screen_height))
            
            # Different colors for different element types
            if element.tag == "input":
                color = [200, 200, 255]  # Light blue for inputs
            elif element.tag == "button":
                color = [200, 255, 200]  # Light green for buttons
            elif element.tag == "link":
                color = [255, 200, 200]  # Light red for links
            else:
                color = [220, 220, 220]  # Light gray for other elements
            
            # Draw element
            y_end = min(84, y + h)
            x_end = min(84, x + w)
            if y < 84 and x < 84:
                image[y:y_end, x:x_end] = color
        
        # Convert to CHW format for PyTorch
        return np.transpose(image, (2, 0, 1)).astype(np.float32) / 255.0
    
    def _create_dom_state(self):
        """Create DOM tree representation."""
        page = self.pages[self.current_page]
        dom_features = []
        
        for i, element in enumerate(page["elements"]):
            # Create feature vector for each element
            features = [
                element.x / self.screen_width,
                element.y / self.screen_height,
                element.width / self.screen_width,
                element.height / self.screen_height,
                1.0 if element.clickable else 0.0,
                1.0 if element.tag == "input" else 0.0,
                1.0 if element.tag == "button" else 0.0,
                1.0 if element.tag == "link" else 0.0,
                1.0 if i == self.focused_element else 0.0,
                len(element.text) / 100.0  # Normalized text length
            ]
            dom_features.append(features)
        
        # Pad to fixed size (max 10 elements)
        while len(dom_features) < 10:
            dom_features.append([0.0] * 10)
        
        return np.array(dom_features[:10], dtype=np.float32)
    
    def step(self, action, click_x=None, click_y=None, text_input=None):
        """Execute action and return new state, reward, done, info."""
        reward = 0
        done = False
        info = {}
        
        self.step_count += 1
        action_name = self.actions[action]
        
        page = self.pages[self.current_page]
        
        if action_name == "click":
            reward += self._handle_click(click_x, click_y)
        elif action_name == "type_text":
            reward += self._handle_type_text(text_input)
        elif action_name == "scroll_up":
            self.scroll_position = max(0, self.scroll_position - 50)
            reward -= 0.01  # Small penalty for scrolling
        elif action_name == "scroll_down":
            self.scroll_position = min(200, self.scroll_position + 50)
            reward -= 0.01
        elif action_name == "focus_next":
            reward += self._handle_focus_next()
        elif action_name == "submit_form":
            reward += self._handle_submit_form()
        elif action_name == "navigate_back":
            reward += self._handle_navigate_back()
        
        # Check task completion
        if self._check_task_completion():
            reward += 10.0
            done = True
            self.task_completed = True
            info["task_completed"] = True
        
        # Time penalty
        reward -= 0.01
        
        # Check max steps
        if self.step_count >= 50:
            done = True
        
        return self._get_state(), reward, done, info
    
    def _handle_click(self, x, y):
        """Handle click action on coordinates."""
        if x is None or y is None:
            return -0.1
        
        page = self.pages[self.current_page]
        clicked_element = None
        
        # Find clicked element
        for i, element in enumerate(page["elements"]):
            if (element.x <= x <= element.x + element.width and
                element.y <= y <= element.y + element.height and
                element.clickable):
                clicked_element = (i, element)
                break
        
        if clicked_element is None:
            return -0.1  # Penalty for clicking empty space
        
        i, element = clicked_element
        self.focused_element = i
        
        # Handle different element types
        if element.tag == "button":
            if element.text == "Login":
                if self._check_login_credentials():
                    self.current_page = "dashboard"
                    return 2.0  # Reward for successful login
                else:
                    return -1.0  # Penalty for failed login
            elif element.text == "Complete Task":
                self.current_page = "task_page"
                return 1.0
            elif element.text == "Submit":
                return self._handle_submit_form()
            elif element.text == "Register":
                return -0.5  # Not the correct action for this task
        elif element.tag == "link":
            if element.text == "Logout":
                self.current_page = "login"
                self.form_data = {}
                return -2.0  # Penalty for logging out before completing task
        
        return 0.1  # Small reward for valid click
    
    def _handle_type_text(self, text):
        """Handle typing text into focused element."""
        if self.focused_element is None or text is None:
            return -0.1
        
        page = self.pages[self.current_page]
        # element = page["elements"][self.focused_element]
        
        elements = page["elements"]

        if not elements:  # empty page
            return None

        self.focused_element = max(0, min(self.focused_element, len(elements) - 1))
        element = elements[self.focused_element]

        
        if element.tag in ["input", "textarea"]:
            # Store typed text
            element_id = f"{self.current_page}_{self.focused_element}"
            self.form_data[element_id] = text
            
            # Check if correct text was typed
            if element.required_text and text.strip().lower() == element.required_text.lower():
                return 1.0  # Reward for correct input
            elif element.required_text:
                return -0.5  # Penalty for incorrect input
            else:
                return 0.2  # Small reward for typing something
        
        return -0.1
    
    def _handle_focus_next(self):
        """Handle focusing next element."""
        page = self.pages[self.current_page]
        clickable_elements = [i for i, el in enumerate(page["elements"]) if el.clickable]
        
        if not clickable_elements:
            return -0.1
        
        if self.focused_element is None:
            self.focused_element = clickable_elements[0]
        else:
            try:
                current_idx = clickable_elements.index(self.focused_element)
                next_idx = (current_idx + 1) % len(clickable_elements)
                self.focused_element = clickable_elements[next_idx]
            except ValueError:
                self.focused_element = clickable_elements[0]
        
        return 0.05  # Small reward for navigating
    
    def _handle_submit_form(self):
        """Handle form submission."""
        if self.current_page == "task_page":
            # Check if all required fields are filled
            required_fields = 3  # Name, Email, Comments
            filled_fields = len([k for k in self.form_data.keys() if k.startswith("task_page_")])
            
            if filled_fields >= required_fields:
                return 5.0  # Large reward for completing form
            else:
                return -1.0  # Penalty for submitting incomplete form
        
        return -0.1
    
    def _handle_navigate_back(self):
        """Handle navigation back."""
        if self.current_page == "dashboard":
            self.current_page = "login"
            return -1.0  # Penalty for going backwards
        elif self.current_page == "task_page":
            self.current_page = "dashboard"
            return -0.5  # Small penalty for going back
        return -0.1
    
    def _check_login_credentials(self):
        """Check if correct login credentials were entered."""
        username_key = "login_0"
        password_key = "login_1"
        
        username = self.form_data.get(username_key, "").strip().lower()
        password = self.form_data.get(password_key, "").strip()
        
        return username == "admin" and password == "password123"
    
    def _check_task_completion(self):
        """Check if the main task is completed."""
        if self.current_page != "task_page":
            return False
        
        # Check if form is properly filled
        required_form_fields = [f"task_page_{i}" for i in [1, 2, 3]]  # Name, Email, Comments
        filled_correctly = all(field in self.form_data for field in required_form_fields)
        
        return filled_correctly


class MultiModalWebPolicy(nn.Module):
    """Multi-modal policy network combining visual and DOM features."""
    
    def __init__(self, action_dim=7, visual_feature_dim=256, dom_feature_dim=128):
        super(MultiModalWebPolicy, self).__init__()
        
        # Visual processing (CNN for screenshot)
        self.visual_cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, visual_feature_dim),
            nn.ReLU()
        )
        
        # DOM processing (graph-like representation)
        self.dom_encoder = nn.Sequential(
            nn.Linear(10, 32),  # 10 features per element
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )
        
        # Element attention mechanism
        self.attention = nn.MultiheadAttention(32, num_heads=4, batch_first=True)
        
        # Fusion layer
        fusion_dim = visual_feature_dim + dom_feature_dim
        self.fusion = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Policy head
        self.policy_head = nn.Linear(128, action_dim)
        
        # Value head
        self.value_head = nn.Linear(128, 1)
        
    def forward(self, visual_input, dom_input):
        # Process visual input
        visual_features = self.visual_cnn(visual_input)
        
        # Process DOM input
        dom_encoded = self.dom_encoder(dom_input)  # [batch, num_elements, 32]
        dom_attended, _ = self.attention(dom_encoded, dom_encoded, dom_encoded)
        dom_features = dom_attended.mean(dim=1)  # Global average pooling
        
        # Expand dom_features to match visual_features dimension
        dom_expanded = F.adaptive_avg_pool1d(dom_features.unsqueeze(1), visual_features.size(1)).squeeze(1)
        
        # Fuse features
        fused_features = torch.cat([visual_features, dom_expanded], dim=1)
        fused_output = self.fusion(fused_features)
        
        # Get policy logits and value
        policy_logits = self.policy_head(fused_output)
        value = self.value_head(fused_output)
        
        return policy_logits, value


class WebNavigationAgent:
    """Agent for web navigation using multi-modal observations."""
    
    def __init__(self, env, lr=3e-4):
        self.env = env
        self.policy = MultiModalWebPolicy(action_dim=env.action_space_size)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_rate = deque(maxlen=100)
        
    def select_action(self, state, training=True):
        """Select action using current policy."""
        visual_input = torch.FloatTensor(state["visual"]).unsqueeze(0)
        dom_input = torch.FloatTensor(state["dom"]).unsqueeze(0)
        
        with torch.no_grad():
            policy_logits, value = self.policy(visual_input, dom_input)
        
        if training:
            action_probs = F.softmax(policy_logits, dim=-1)
            action_dist = Categorical(action_probs)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
        else:
            action = torch.argmax(policy_logits, dim=-1)
            log_prob = None
        
        return action.item(), log_prob, value.item()
    
    def train_episode(self):
        """Train on a single episode using A2C."""
        state = self.env.reset()
        
        states_visual = []
        states_dom = []
        actions = []
        log_probs = []
        rewards = []
        values = []
        
        done = False
        step = 0
        
        while not done:
            # Select action
            action, log_prob, value = self.select_action(state, training=True)
            
            # Store state
            states_visual.append(state["visual"])
            states_dom.append(state["dom"])
            values.append(value)
            actions.append(action)
            log_probs.append(log_prob)
            
            # Generate action parameters for web interaction
            click_x, click_y, text_input = self._generate_action_params(state, action)
            
            # Take step
            next_state, reward, done, info = self.env.step(action, click_x, click_y, text_input)
            rewards.append(reward)
            
            state = next_state
            step += 1
        
        # Calculate returns
        returns = self._calculate_returns(rewards)
        
        # Convert to tensors
        visual_states = torch.FloatTensor(np.array(states_visual))
        dom_states = torch.FloatTensor(np.array(states_dom))
        log_probs = torch.stack(log_probs)
        returns = torch.FloatTensor(returns)
        values = torch.FloatTensor(values)
        
        # Get final policy output
        policy_logits, pred_values = self.policy(visual_states, dom_states)
        
        # Calculate advantages
        advantages = returns - values
        
        # Policy loss
        policy_loss = -(log_probs * advantages.detach()).mean()
        
        # Value loss
        value_loss = F.mse_loss(pred_values.squeeze(), returns)
        
        # Entropy bonus
        action_probs = F.softmax(policy_logits, dim=-1)
        entropy = -(action_probs * torch.log(action_probs + 1e-8)).sum(dim=-1).mean()
        
        # Total loss
        total_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
        
        # Update policy
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Store episode statistics
        episode_reward = sum(rewards)
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(len(rewards))
        self.success_rate.append(info.get("task_completed", False))
        
        return {
            "episode_reward": episode_reward,
            "episode_length": len(rewards),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
            "task_completed": info.get("task_completed", False)
        }
    
    def _generate_action_params(self, state, action):
        """Generate parameters for web actions (click coordinates, text input)."""
        click_x, click_y, text_input = None, None, None
        
        if action == 0:  # click
            # Simple heuristic: click on focused element or center of screen
            if state["focused"] is not None:
                # Get focused element position
                page = self.env.pages[self.env.current_page]
                # element = page["elements"][state["focused"]]
                
                focused_idx = state["focused"]
                elements = page["elements"]

                if not elements:  # empty page
                    return None

                focused_idx = max(0, min(focused_idx, len(elements) - 1))
                element = elements[focused_idx]

                click_x = element.x + element.width // 2
                click_y = element.y + element.height // 2
            else:
                # Click center of screen
                click_x = self.env.screen_width // 2
                click_y = self.env.screen_height // 2
        
        elif action == 1:  # type_text
            # Generate appropriate text based on current context
            if state["focused"] is not None:
                page = self.env.pages[self.env.current_page]
                # element = page["elements"][state["focused"]]
                
                
                focused_idx = state["focused"]
                elements = page["elements"]

                if not elements:  # empty page
                    return None

                focused_idx = max(0, min(focused_idx, len(elements) - 1))
                element = elements[focused_idx]

                if element.required_text:
                    text_input = element.required_text
                else:
                    text_input = "sample text"
        
        return click_x, click_y, text_input
    
    def _calculate_returns(self, rewards, gamma=0.99):
        """Calculate discounted returns."""
        returns = []
        G = 0
        for reward in reversed(rewards):
            G = reward + gamma * G
            returns.insert(0, G)
        return returns


def train_web_agent(num_episodes=2000):
    """Main training loop for web navigation agent."""
    print("=== Chapter 14: Web Navigation ===")
    print("Training multi-modal agent for web navigation...\n")
    
    # Create environment
    env = SimpleWebEnvironment()
    print(f"Environment: SimpleWebEnvironment")
    print(f"Screen size: {env.screen_width}x{env.screen_height}")
    print(f"Action space: {env.action_space_size} actions")
    print(f"Goal: Login and complete web form task\n")
    
    # Create agent
    agent = WebNavigationAgent(env, lr=3e-4)
    
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
            recent_success_rate = sum(agent.success_rate) / len(agent.success_rate) if agent.success_rate else 0
            
            avg_reward = np.mean(recent_rewards)
            avg_length = np.mean(agent.episode_lengths[-100:])
            
            success_rates.append(recent_success_rate)
            avg_rewards.append(avg_reward)
            avg_lengths.append(avg_length)
            
            print(f"Episode {episode+1:4d} | Avg Reward: {avg_reward:6.2f} | "
                  f"Avg Length: {avg_length:5.1f} | Success Rate: {recent_success_rate:5.1%} | "
                  f"Policy Loss: {result['policy_loss']:6.3f} | Entropy: {result['entropy']:6.3f}")
    
    print("\nTraining completed!")
    
    # Test final policy
    print("\nTesting final policy...")
    test_successes = 0
    test_episodes = 50
    test_rewards = []
    
    for test_ep in range(test_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        steps = 0
        
        if test_ep < 3:  # Show first 3 test episodes in detail
            print(f"\nTest Episode {test_ep + 1}:")
        
        while not done and steps < 50:
            action, _, _ = agent.select_action(state, training=False)
            action_name = env.actions[action]
            
            # Generate action parameters
            click_x, click_y, text_input = agent._generate_action_params(state, action)
            
            state, reward, done, info = env.step(action, click_x, click_y, text_input)
            total_reward += reward
            steps += 1
            
            if test_ep < 3 and steps <= 10:  # Show first 10 steps
                print(f"  Step {steps}: {action_name} on page '{env.current_page}' -> Reward: {reward:.2f}")
        
        test_rewards.append(total_reward)
        if info.get("task_completed", False):
            test_successes += 1
            if test_ep < 3:
                print(f"  SUCCESS! Task completed in {steps} steps")
        else:
            if test_ep < 3:
                print(f"  Failed - Task not completed in {steps} steps")
    
    final_success_rate = test_successes / test_episodes
    final_avg_reward = np.mean(test_rewards)
    
    print(f"\nFinal Performance:")
    print(f"Success Rate: {final_success_rate:.1%}")
    print(f"Average Reward: {final_avg_reward:.2f}")
    
    # Visualizations
    create_web_training_plots(success_rates, avg_rewards, avg_lengths)
    
    return agent, env


def create_web_training_plots(success_rates, avg_rewards, avg_lengths):
    """Create training progress visualizations."""
    episodes = np.arange(100, len(success_rates) * 100 + 1, 100)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Web Navigation Training Progress', fontsize=16)
    
    # Success rate
    axes[0, 0].plot(episodes, success_rates, 'g-', linewidth=2)
    axes[0, 0].set_title('Task Completion Rate')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Success Rate')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim(0, 1)
    
    # Average reward
    axes[0, 1].plot(episodes, avg_rewards, 'b-', linewidth=2)
    axes[0, 1].set_title('Average Reward')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Average Reward')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Episode length
    axes[1, 0].plot(episodes, avg_lengths, 'r-', linewidth=2)
    axes[1, 0].set_title('Average Episode Length')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Steps per Episode')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Learning efficiency
    efficiency = np.array(success_rates) / np.array(avg_lengths)
    axes[1, 1].plot(episodes, efficiency, 'purple', linewidth=2)
    axes[1, 1].set_title('Learning Efficiency (Success/Length)')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Efficiency')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pytorch_rl_tutorial/web_navigation_training.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\nTraining plots saved as 'web_navigation_training.png'")


def demonstrate_multi_modal_features(agent, env):
    """Demonstrate how the agent processes visual and DOM features."""
    print("\n=== Multi-Modal Feature Analysis ===")
    
    # Test on different pages
    test_pages = ["login", "dashboard", "task_page"]
    
    for page in test_pages:
        env.current_page = page
        state = env._get_state()
        
        print(f"\nPage: {page}")
        print(f"Visual state shape: {state['visual'].shape}")
        print(f"DOM state shape: {state['dom'].shape}")
        
        # Get action predictions
        action, _, value = agent.select_action(state, training=False)
        action_name = env.actions[action]
        
        print(f"Predicted action: {action_name}")
        print(f"Value estimate: {value:.3f}")
        
        # Show DOM elements
        page_elements = env.pages[page]["elements"]
        print(f"Page elements:")
        for i, element in enumerate(page_elements):
            print(f"  {i}: {element.tag} - '{element.text}' (clickable: {element.clickable})")


if __name__ == "__main__":
    print("Chapter 14: Web Navigation")
    print("="*50)
    print("This chapter demonstrates RL for web automation tasks.")
    print("We'll train a multi-modal agent to navigate and interact with web pages.\n")
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Train the agent
    agent, env = train_web_agent(num_episodes=1500)
    
    # Demonstrate multi-modal features
    demonstrate_multi_modal_features(agent, env)
    
    print("\n" + "="*50)
    print("Key Concepts Demonstrated:")
    print("- Multi-modal state representation (visual + DOM)")
    print("- CNN for visual processing of web pages")
    print("- Graph neural networks for DOM structure")
    print("- Action parameterization for web interactions")
    print("- Curriculum learning for complex web tasks")
    print("\nWeb navigation RL applications:")
    print("- Automated testing and QA")
    print("- Web scraping and data extraction")
    print("- Accessibility testing")
    print("- User interface optimization")
    print("\nNext: Chapter 15 - Continuous Action Space")
