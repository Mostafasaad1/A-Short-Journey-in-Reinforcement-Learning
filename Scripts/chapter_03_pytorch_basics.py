# Chapter 3: Deep Learning with PyTorch
# Core concepts for Reinforcement Learning

# Install required packages:
# !pip install torch torchvision matplotlib numpy -q

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List

# Configure matplotlib for proper font rendering
plt.switch_backend("Agg")
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "PingFang SC", "Arial Unicode MS", "Hiragino Sans GB"]
plt.rcParams["axes.unicode_minus"] = False

print("Chapter 3: Deep Learning with PyTorch")
print("=" * 50)

# 1. TENSORS - The Foundation of PyTorch
print("\n1. Understanding Tensors")
print("-" * 30)

# Create tensors - the multi-dimensional arrays that store data
# Tensors are similar to NumPy arrays but can run on GPU and support automatic differentiation
data = [[1, 2], [3, 4]]
tensor_from_list = torch.tensor(data, dtype=torch.float32)
print(f"Tensor from list: \n{tensor_from_list}")

# Random tensors - commonly used for weight initialization in neural networks
random_tensor = torch.randn(2, 3)  # Normal distribution with mean=0, std=1
print(f"\nRandom tensor: \n{random_tensor}")

# Zeros and ones - useful for bias initialization and masks
zeros_tensor = torch.zeros(2, 3)
ones_tensor = torch.ones(2, 3)
print(f"\nZeros tensor: \n{zeros_tensor}")
print(f"Ones tensor: \n{ones_tensor}")

# Tensor operations - essential for neural network computations
print("\n2. Tensor Operations")
print("-" * 30)

a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

# Element-wise operations
print(f"a + b = {a + b}")  # Addition
print(f"a * b = {a * b}")  # Element-wise multiplication
print(f"a.dot(b) = {torch.dot(a, b)}")  # Dot product

# Matrix operations - crucial for neural network forward passes
matrix_a = torch.randn(3, 4)
matrix_b = torch.randn(4, 2)
matrix_product = torch.mm(matrix_a, matrix_b)  # Matrix multiplication
print(f"\nMatrix multiplication result shape: {matrix_product.shape}")

# 3. AUTOGRAD - Automatic Differentiation
print("\n3. Autograd - The Heart of PyTorch")
print("-" * 30)

# Enable gradient computation - this is what makes neural networks trainable
x = torch.tensor([2.0], requires_grad=True)  # Input that requires gradients
y = x**2 + 3*x + 1  # A simple function f(x) = x² + 3x + 1

print(f"x = {x.item():.2f}")
print(f"y = f(x) = {y.item():.2f}")

# Compute gradients automatically
y.backward()  # This computes dy/dx
print(f"dy/dx = {x.grad.item():.2f}")  # Should be 2*x + 3 = 2*2 + 3 = 7

# Clear gradients for next computation (important in training loops)
x.grad.zero_()

# 4. NEURAL NETWORK MODULE (nn.Module)
print("\n4. Building Neural Networks with nn.Module")
print("-" * 30)

class SimpleNet(nn.Module):
    """A simple feedforward neural network for function approximation.
    
    This demonstrates the basic structure that will be used for value networks
    and policy networks in reinforcement learning.
    """
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(SimpleNet, self).__init__()
        # Define layers - these are the building blocks of neural networks
        self.fc1 = nn.Linear(input_size, hidden_size)  # First fully connected layer
        self.fc2 = nn.Linear(hidden_size, hidden_size)  # Hidden layer
        self.fc3 = nn.Linear(hidden_size, output_size)  # Output layer
        self.relu = nn.ReLU()  # Activation function for non-linearity
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor after forward propagation
        """
        x = self.relu(self.fc1(x))  # Apply first layer + activation
        x = self.relu(self.fc2(x))  # Apply second layer + activation
        x = self.fc3(x)  # Final layer (no activation for regression)
        return x

# Create a network instance
net = SimpleNet(input_size=1, hidden_size=64, output_size=1)
print(f"Network architecture: \n{net}")

# 5. LOSS FUNCTIONS AND OPTIMIZERS
print("\n5. Loss Functions and Optimizers")
print("-" * 30)

# Loss function - measures how far our predictions are from the target
criterion = nn.MSELoss()  # Mean Squared Error for regression tasks

# Optimizer - updates network parameters to minimize loss
optimizer = optim.Adam(net.parameters(), lr=0.001)  # Adam optimizer

print(f"Loss function: {criterion}")
print(f"Optimizer: {optimizer}")
print(f"Number of trainable parameters: {sum(p.numel() for p in net.parameters() if p.requires_grad)}")

# 6. COMPLETE EXAMPLE: APPROXIMATING sin(x)
print("\n6. Complete Example: Function Approximation")
print("-" * 30)

def generate_data(n_samples: int = 1000) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate training data for sin(x) function approximation.
    
    Args:
        n_samples: Number of data points to generate
        
    Returns:
        Tuple of (input, target) tensors
    """
    x = torch.linspace(0, 2*np.pi, n_samples).unsqueeze(1)  # Input: [0, 2π]
    y = torch.sin(x)  # Target: sin(x)
    return x, y

def train_network(net: nn.Module, x_train: torch.Tensor, y_train: torch.Tensor, 
                 epochs: int = 1000) -> List[float]:
    """Train the neural network to approximate sin(x).
    
    Args:
        net: Neural network to train
        x_train: Training inputs
        y_train: Training targets
        epochs: Number of training epochs
        
    Returns:
        List of loss values during training
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.01)
    
    losses = []
    
    for epoch in range(epochs):
        # Forward pass: compute predictions
        predictions = net(x_train)
        
        # Compute loss: how far are we from the target?
        loss = criterion(predictions, y_train)
        
        # Backward pass: compute gradients
        optimizer.zero_grad()  # Clear previous gradients
        loss.backward()  # Compute new gradients via backpropagation
        
        # Update parameters: this is where learning happens
        optimizer.step()  # Apply gradients to update weights
        
        losses.append(loss.item())
        
        if (epoch + 1) % 200 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")
    
    return losses

# Generate training data
print("Generating training data...")
x_train, y_train = generate_data(1000)
print(f"Training data shape: X={x_train.shape}, Y={y_train.shape}")

# Create and train network
print("\nTraining network to approximate sin(x)...")
net = SimpleNet(input_size=1, hidden_size=64, output_size=1)
losses = train_network(net, x_train, y_train, epochs=1000)

# Generate test data for evaluation
x_test = torch.linspace(0, 2*np.pi, 200).unsqueeze(1)
y_test = torch.sin(x_test)

# Make predictions
with torch.no_grad():  # Disable gradient computation for inference
    predictions = net(x_test)

# 7. VISUALIZATION
print("\n7. Visualizing Results")
print("-" * 30)

# Create plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Training loss over time
ax1.plot(losses)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss (MSE)')
ax1.set_title('Training Loss Over Time')
ax1.grid(True)

# Plot 2: Predictions vs Ground Truth
ax2.plot(x_test.numpy(), y_test.numpy(), 'b-', label='Ground Truth (sin(x))', linewidth=2)
ax2.plot(x_test.numpy(), predictions.numpy(), 'r--', label='Neural Network Prediction', linewidth=2)
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_title('Function Approximation: sin(x)')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig('pytorch_rl_tutorial/chapter_03_results.png', dpi=150, bbox_inches='tight')
print("Results saved to chapter_03_results.png")

# Calculate final error
final_mse = nn.MSELoss()(predictions, y_test).item()
print(f"\nFinal MSE on test data: {final_mse:.6f}")

# 8. CONNECTION TO REINFORCEMENT LEARNING
print("\n8. Connection to Reinforcement Learning")
print("-" * 30)

print("""
The concepts we've covered form the foundation for RL algorithms:

1. TENSORS: Store states, actions, rewards, and Q-values
   - States: environment observations as tensors
   - Actions: discrete indices or continuous values
   - Q-values: action-value estimates for each state-action pair

2. AUTOGRAD: Enables gradient-based policy optimization
   - .backward(): computes gradients for policy/value function updates
   - Essential for Deep Q-Networks (DQN), Policy Gradients, Actor-Critic

3. nn.Module: Foundation for RL network architectures
   - Value Networks: estimate V(s) or Q(s,a)
   - Policy Networks: output action probabilities π(a|s)
   - Actor-Critic: combines both value and policy estimation

4. OPTIMIZERS: Update network parameters based on RL objectives
   - Adam, RMSprop commonly used in RL
   - Learning rates often decay during training

5. LOSS FUNCTIONS: Define RL-specific objectives
   - MSE for value function regression
   - Cross-entropy for policy optimization
   - Custom losses for policy gradients (e.g., REINFORCE)

Next chapters will build on these primitives to create complete RL agents!
""")

print("\nChapter 3 Complete! ✓")
print("Ready to move on to Cross-Entropy Method (Chapter 4)")
