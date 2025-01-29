# Chapter 10: Stocks Trading Using RL
# Applying reinforcement learning to financial markets

# Install required packages:
# !pip install gymnasium torch numpy pandas yfinance matplotlib scikit-learn empyrical -q

import gymnasium as gym
from gymnasium import spaces
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from empyrical import sharpe_ratio
import warnings
import random
warnings.filterwarnings('ignore')

# Configure matplotlib for proper font rendering
plt.switch_backend("Agg")
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "PingFang SC", "Arial Unicode MS", "Hiragino Sans GB"]
plt.rcParams["axes.unicode_minus"] = False

print("Chapter 10: Stocks Trading Using RL")
print("=" * 50)

# 1. INTRODUCTION TO RL IN FINANCE
print("\n1. Reinforcement Learning in Financial Markets")
print("-" * 30)

print("""
RL IN ALGORITHMIC TRADING:

1. PROBLEM FORMULATION:
   - State: Market data (prices, volumes, technical indicators)
   - Action: Buy, Sell, Hold (or continuous position sizing)
   - Reward: Portfolio returns, risk-adjusted metrics
   - Environment: Financial market simulation

2. CHALLENGES:
   - Non-stationarity: Market conditions change over time
   - Partial observability: Limited market information
   - High noise: Random market fluctuations
   - Transaction costs: Spreads, fees, slippage
   - Survivorship bias: Historical data limitations

3. ADVANTAGES OF RL:
   - Adaptive strategies: Learn from changing market conditions
   - Risk management: Incorporate risk constraints
   - Multi-asset: Handle complex portfolios
   - Feature learning: Discover patterns automatically

4. RISK CONSIDERATIONS:
   - Overfitting to historical data
   - Market regime changes
   - Regulatory compliance
   - Capital preservation

WARNING: This is for educational purposes only!
Real trading involves substantial risk of loss.
""")

# 2. DATA COLLECTION AND PREPROCESSING
print("\n2. Financial Data Collection")
print("-" * 30)

class FinancialDataProcessor:
    """Handles downloading and preprocessing of financial data."""
    
    def __init__(self, symbols: List[str], start_date: str, end_date: str):
        """Initialize data processor.
        
        Args:
            symbols: List of stock symbols to download
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
        """
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.raw_data = {}
        self.processed_data = {}
        self.scalers = {}
    
    def download_data(self) -> Dict[str, pd.DataFrame]:
        """Download financial data using yfinance.
        
        Returns:
            Dictionary of DataFrames with stock data
        """
        print(f"Downloading data for {self.symbols}...")
        
        for symbol in self.symbols:
            try:
                # Download data
                ticker = yf.Ticker(symbol)
                data = ticker.history(start=self.start_date, end=self.end_date)
                
                if data.empty:
                    print(f"Warning: No data found for {symbol}")
                    continue
                
                # Clean data
                data = data.dropna()
                self.raw_data[symbol] = data
                print(f"Downloaded {len(data)} days of data for {symbol}")
                
            except Exception as e:
                print(f"Error downloading {symbol}: {e}")
        
        return self.raw_data
    
    def calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators.
        
        Args:
            data: Raw OHLCV data
            
        Returns:
            DataFrame with technical indicators
        """
        df = data.copy()
        
        # Price-based indicators
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Moving averages
        df['SMA_5'] = df['Close'].rolling(window=5).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        
        # Exponential moving averages
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / df['BB_Width']
        
        # Volatility
        df['Volatility'] = df['Returns'].rolling(window=20).std()
        
        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        # Price position within day's range
        df['Daily_Range'] = df['High'] - df['Low']
        df['Close_Position'] = (df['Close'] - df['Low']) / df['Daily_Range']
        
        return df
    
    def prepare_features(self, symbol: str, lookback_window: int = 30) -> np.ndarray:
        """Prepare feature matrix for ML model.
        
        Args:
            symbol: Stock symbol
            lookback_window: Number of days to look back for features
            
        Returns:
            Feature matrix
        """
        if symbol not in self.raw_data:
            raise ValueError(f"No data available for {symbol}")
        
        # Calculate technical indicators
        data = self.calculate_technical_indicators(self.raw_data[symbol])
        
        # Select features
        feature_columns = [
            'Returns', 'SMA_5', 'SMA_20', 'RSI', 'MACD', 'MACD_Histogram',
            'BB_Position', 'Volatility', 'Volume_Ratio', 'Close_Position'
        ]
        
        # Create feature matrix
        features = data[feature_columns].dropna()
        
        # Normalize features
        scaler = StandardScaler()
        normalized_features = scaler.fit_transform(features)
        
        # Store scaler for later use
        self.scalers[symbol] = scaler
        
        # Create sequences for time series
        sequences = []
        for i in range(lookback_window, len(normalized_features)):
            sequences.append(normalized_features[i-lookback_window:i].flatten())
        
        self.processed_data[symbol] = {
            'features': np.array(sequences),
            'prices': data['Close'].iloc[lookback_window:].values,
            'returns': data['Returns'].iloc[lookback_window:].values,
            'dates': data.index[lookback_window:]
        }
        
        return np.array(sequences)

# Download sample data
print("Setting up financial data...")
data_processor = FinancialDataProcessor(
    symbols=['AAPL', 'MSFT', 'GOOGL'],
    start_date='2020-01-01',
    end_date='2023-12-31'
)

# Use synthetic data if yfinance fails
try:
    raw_data = data_processor.download_data()
    if not raw_data:
        raise Exception("No data downloaded")
except:
    print("Using synthetic data for demonstration...")
    # Create synthetic stock data
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    synthetic_data = pd.DataFrame(index=dates)
    
    # Generate realistic stock price movements
    np.random.seed(42)
    initial_price = 100
    returns = np.random.normal(0.0005, 0.02, len(dates))  # Daily returns
    prices = [initial_price]
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    synthetic_data['Close'] = prices
    synthetic_data['Open'] = synthetic_data['Close'] * (1 + np.random.normal(0, 0.001, len(dates)))
    synthetic_data['High'] = synthetic_data[['Open', 'Close']].max(axis=1) * (1 + np.abs(np.random.normal(0, 0.005, len(dates))))
    synthetic_data['Low'] = synthetic_data[['Open', 'Close']].min(axis=1) * (1 - np.abs(np.random.normal(0, 0.005, len(dates))))
    synthetic_data['Volume'] = np.random.lognormal(15, 0.5, len(dates)).astype(int)
    
    data_processor.raw_data['AAPL'] = synthetic_data
    print("Generated synthetic AAPL data for demonstration")

# Process features
features = data_processor.prepare_features('AAPL', lookback_window=20)
print(f"Prepared features shape: {features.shape}")
print(f"Feature dimension per timestep: {features.shape[1] // 20}")

# 3. TRADING ENVIRONMENT
print("\n3. Custom Trading Environment")
print("-" * 30)

class TradingEnvironment(gym.Env):
    """Custom OpenAI Gym environment for stock trading."""
    
    def __init__(self, data_processor: FinancialDataProcessor, symbol: str,
                 initial_balance: float = 10000, transaction_cost: float = 0.001,
                 max_position: float = 1.0, lookback_window: int = 20):
        """Initialize trading environment.
        
        Args:
            data_processor: Processed financial data
            symbol: Stock symbol to trade
            initial_balance: Starting cash balance
            transaction_cost: Transaction cost as fraction of trade value
            max_position: Maximum position size (1.0 = 100% of portfolio)
            lookback_window: Number of days in observation window
        """
        super(TradingEnvironment, self).__init__()
        
        self.data_processor = data_processor
        self.symbol = symbol
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.max_position = max_position
        self.lookback_window = lookback_window
        
        # Get processed data
        if symbol not in data_processor.processed_data:
            data_processor.prepare_features(symbol, lookback_window)
        
        self.data = data_processor.processed_data[symbol]
        self.n_steps = len(self.data['features'])
        
        # Action space: continuous position from -1 (short) to +1 (long)
        self.action_space = spaces.Box(low=-max_position, high=max_position, 
                                     shape=(1,), dtype=np.float32)
        
        # Observation space: features + portfolio state
        feature_dim = self.data['features'].shape[1]
        portfolio_dim = 3  # cash, position, portfolio_value
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                          shape=(feature_dim + portfolio_dim,),
                                          dtype=np.float32)
        
        # Initialize state
        self.reset()
    
    def reset(self, seed=None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state.
        
        Returns:
            Initial observation and info dict
        """
        super().reset(seed=seed)
        
        # Reset portfolio state
        self.current_step = 0
        self.cash = self.initial_balance
        self.position = 0.0  # Current position size (-1 to +1)
        self.shares_held = 0.0
        self.portfolio_value = self.initial_balance
        
        # Trading history
        self.portfolio_history = [self.initial_balance]
        self.action_history = []
        self.trades = []
        
        # Performance metrics
        self.total_trades = 0
        self.winning_trades = 0
        
        return self._get_observation(), {}
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation.
        
        Returns:
            Combined market features and portfolio state
        """
        # Market features
        market_features = self.data['features'][self.current_step]
        
        # Portfolio state (normalized)
        current_price = self.data['prices'][self.current_step]
        portfolio_features = np.array([
            self.cash / self.initial_balance - 1,  # Cash ratio
            self.position,  # Current position
            (self.portfolio_value / self.initial_balance) - 1  # Portfolio return
        ])
        
        return np.concatenate([market_features, portfolio_features])
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one trading step.
        
        Args:
            action: Desired position size (-1 to +1)
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Clip action to valid range
        target_position = np.clip(action[0], -self.max_position, self.max_position)
        
        # Get current price
        current_price = self.data['prices'][self.current_step]
        
        # Calculate position change
        position_change = target_position - self.position
        
        # Execute trade if position changes significantly
        if abs(position_change) > 0.01:  # Minimum trade threshold
            trade_value = abs(position_change) * self.portfolio_value
            transaction_cost = trade_value * self.transaction_cost
            
            # Update position and cash
            self.position = target_position
            self.cash -= transaction_cost
            
            # Record trade
            self.trades.append({
                'step': self.current_step,
                'price': current_price,
                'position_change': position_change,
                'cost': transaction_cost
            })
            self.total_trades += 1
        
        # Calculate portfolio value
        position_value = self.position * self.portfolio_value
        self.shares_held = position_value / current_price if current_price > 0 else 0
        
        # Move to next step
        self.current_step += 1
        
        # Calculate new portfolio value
        if self.current_step < self.n_steps:
            new_price = self.data['prices'][self.current_step]
            new_position_value = self.shares_held * new_price
            cash_equivalent = self.portfolio_value - position_value
            self.portfolio_value = cash_equivalent + new_position_value
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Update history
        self.portfolio_history.append(self.portfolio_value)
        self.action_history.append(target_position)
        
        # Check if episode is done
        terminated = self.current_step >= self.n_steps - 1
        truncated = False
        
        # Get next observation
        if not terminated:
            obs = self._get_observation()
        else:
            obs = np.zeros_like(self._get_observation())
        
        # Info dictionary
        info = {
            'portfolio_value': self.portfolio_value,
            'position': self.position,
            'cash': self.cash,
            'total_trades': self.total_trades
        }
        
        return obs, reward, terminated, truncated, info
    
    def _calculate_reward(self) -> float:
        """Calculate step reward.
        
        Returns:
            Reward value
        """
        if len(self.portfolio_history) < 2:
            return 0.0
        
        # Portfolio return
        portfolio_return = (self.portfolio_history[-1] / self.portfolio_history[-2]) - 1
        
        # Risk-adjusted reward (simplified Sharpe ratio)
        if len(self.portfolio_history) >= 20:
            hist = np.array(self.portfolio_history[-21:], dtype=np.float32)
            returns = hist[1:] / hist[:-1] - 1 if hist.size > 1 else np.array([], dtype=np.float32)
            sharpe = np.mean(returns) / (np.std(returns) + 1e-8)
            return sharpe * 100  # Scale for better learning
        
        # Simple return-based reward for early steps
        return portfolio_return * 100
    
    def get_portfolio_stats(self) -> Dict[str, float]:
        """Calculate portfolio performance statistics.
        
        Returns:
            Dictionary of performance metrics
        """
        if len(self.portfolio_history) < 2:
            return {}
        
        # Calculate returns
        portfolio_values = np.array(self.portfolio_history)
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        # Performance metrics
        total_return = (portfolio_values[-1] / portfolio_values[0]) - 1
        annualized_return = (portfolio_values[-1] / portfolio_values[0]) ** (252 / len(portfolio_values)) - 1
        volatility = np.std(returns) * np.sqrt(252)
        # sharpe_rati = annualized_return / volatility if volatility > 0 else 0
        sharpe_rati = sharpe_ratio(returns, risk_free=0.0, period='daily')
        
        # Drawdown
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - peak) / peak
        max_drawdown = np.min(drawdown)
        
        # Win rate
        positive_returns = returns > 0
        win_rate = np.mean(positive_returns) if len(returns) > 0 else 0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_rati,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': self.total_trades
        }

# Test trading environment
print("Testing trading environment...")
env = TradingEnvironment(data_processor, 'AAPL', initial_balance=10000)
print(f"Action space: {env.action_space}")
print(f"Observation space: {env.observation_space}")

# Test random trading
obs, _ = env.reset()
for _ in range(10):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated:
        break

print(f"Test completed. Final portfolio value: ${info['portfolio_value']:.2f}")

# 4. TRADING AGENT IMPLEMENTATION
print("\n4. Deep RL Trading Agent")
print("-" * 30)

class TradingDQN(nn.Module):
    """Deep Q-Network for trading decisions."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.2):
        super(TradingDQN, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3)  # Buy, Hold, Sell
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class TradingAgent:
    """RL trading agent using DQN."""
    
    def __init__(self, state_dim: int, lr: float = 0.001, gamma: float = 0.99,
                 epsilon: float = 1.0, epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01, buffer_size: int = 10000,
                 batch_size: int = 32, target_update_freq: int = 100):
        """Initialize trading agent."""
        
        self.state_dim = state_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Networks
        self.q_network = TradingDQN(state_dim).to(self.device)
        self.target_network = TradingDQN(state_dim).to(self.device)
        self.update_target_network()
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Replay buffer
        self.replay_buffer = deque(maxlen=buffer_size)
        
        # Tracking
        self.steps = 0
        self.losses = []
        self.episode_rewards = []
    
    def get_action(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        """Get trading action.
        
        Args:
            state: Current market state
            training: Whether in training mode
            
        Returns:
            Action array (position size)
        """
        if training and np.random.random() < self.epsilon:
            # Random action: -1, 0, or 1
            discrete_action = np.random.choice(3)
        else:
            # Greedy action
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
                discrete_action = q_values.argmax().item()
        
        # Convert discrete action to continuous position
        # 0: Sell (-1), 1: Hold (0), 2: Buy (+1)
        position_map = {0: -1.0, 1: 0.0, 2: 1.0}
        position = position_map[discrete_action]
        
        return np.array([position])
    
    def store_experience(self, state: np.ndarray, action: np.ndarray, reward: float,
                        next_state: np.ndarray, done: bool) -> None:
        """Store experience in replay buffer."""
        # Convert continuous action to discrete
        if action[0] < -0.5:
            discrete_action = 0  # Sell
        elif action[0] > 0.5:
            discrete_action = 2  # Buy
        else:
            discrete_action = 1  # Hold
        
        self.replay_buffer.append((state, discrete_action, reward, next_state, done))
    
    def update_target_network(self) -> None:
        """Update target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def train_step(self) -> Optional[float]:
        """Perform training step."""
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample batch
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # Current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Target Q-values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10)
        self.optimizer.step()
        
        # Update target network
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.update_target_network()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()

# 5. TRAINING THE TRADING AGENT
print("\n5. Training Trading Agent")
print("-" * 30)

def train_trading_agent(agent: TradingAgent, env: TradingEnvironment, 
                       n_episodes: int = 200) -> List[Dict]:
    """Train trading agent.
    
    Args:
        agent: Trading agent to train
        env: Trading environment
        n_episodes: Number of training episodes
        
    Returns:
        List of episode statistics
    """
    episode_stats = []
    
    print(f"Training agent for {n_episodes} episodes...")
    
    for episode in range(n_episodes):
        state, _ = env.reset()
        episode_reward = 0
        
        while True:
            # Get action
            action = agent.get_action(state, training=True)
            
            # Take step
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Store experience
            agent.store_experience(state, action, reward, next_state, done)
            
            # Train agent
            loss = agent.train_step()
            if loss is not None:
                agent.losses.append(loss)
            
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        # Record episode statistics
        portfolio_stats = env.get_portfolio_stats()
        episode_stat = {
            'episode': episode,
            'episode_reward': episode_reward,
            'epsilon': agent.epsilon,
            **portfolio_stats
        }
        episode_stats.append(episode_stat)
        agent.episode_rewards.append(episode_reward)
        
        # Logging
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean([s['episode_reward'] for s in episode_stats[-50:]])
            avg_return = np.mean([s.get('total_return', 0) for s in episode_stats[-50:]])
            print(f"Episode {episode+1}: Avg Reward = {avg_reward:.2f}, "
                  f"Avg Return = {avg_return:.3f}, Epsilon = {agent.epsilon:.3f}")
    
    return episode_stats

# Create and train agent
env = TradingEnvironment(data_processor, 'AAPL', initial_balance=10000)
state_dim = env.observation_space.shape[0]

print(f"Creating trading agent with state dimension: {state_dim}")
agent = TradingAgent(state_dim, lr=0.001, epsilon_decay=0.995)

# Train agent
training_stats = train_trading_agent(agent, env, n_episodes=200)

print(f"\nTraining completed!")
final_stats = training_stats[-1] if training_stats else {}
print(f"Final total return: {final_stats.get('total_return', 0):.3f}")
print(f"Final Sharpe ratio: {final_stats.get('sharpe_ratio', 0):.3f}")

# 6. BACKTESTING AND EVALUATION
print("\n6. Backtesting Trained Agent")
print("-" * 30)

def backtest_agent(agent: TradingAgent, env: TradingEnvironment) -> Dict:
    """Backtest trained agent.
    
    Args:
        agent: Trained trading agent
        env: Trading environment
        
    Returns:
        Backtest results
    """
    # Reset environment
    state, _ = env.reset()
    
    # Run episode without training
    while True:
        action = agent.get_action(state, training=False)  # No exploration
        state, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            break
    
    # Get performance statistics
    stats = env.get_portfolio_stats()
    
    # Calculate buy-and-hold benchmark
    initial_price = env.data['prices'][0]
    final_price = env.data['prices'][-1]
    buy_hold_return = (final_price / initial_price) - 1
    
    stats['buy_hold_return'] = buy_hold_return
    stats['excess_return'] = stats['total_return'] - buy_hold_return
    stats['portfolio_history'] = env.portfolio_history
    stats['action_history'] = env.action_history
    stats['trades'] = env.trades
    
    return stats

# Backtest the trained agent
print("Running backtest...")
backtest_results = backtest_agent(agent, env)

print(f"\nBacktest Results:")
print(f"Total Return: {backtest_results['total_return']:.3f} ({backtest_results['total_return']*100:.1f}%)")
print(f"Buy & Hold Return: {backtest_results['buy_hold_return']:.3f} ({backtest_results['buy_hold_return']*100:.1f}%)")
print(f"Excess Return: {backtest_results['excess_return']:.3f} ({backtest_results['excess_return']*100:.1f}%)")
print(f"Annualized Return: {backtest_results['annualized_return']:.3f} ({backtest_results['annualized_return']*100:.1f}%)")
print(f"Volatility: {backtest_results['volatility']:.3f} ({backtest_results['volatility']*100:.1f}%)")
print(f"Sharpe Ratio: {backtest_results['sharpe_ratio']:.3f}")
print(f"Maximum Drawdown: {backtest_results['max_drawdown']:.3f} ({backtest_results['max_drawdown']*100:.1f}%)")
print(f"Win Rate: {backtest_results['win_rate']:.3f} ({backtest_results['win_rate']*100:.1f}%)")
print(f"Total Trades: {backtest_results['total_trades']}")

# 7. VISUALIZATION
print("\n7. Creating Trading Visualizations")
print("-" * 30)

# Create comprehensive trading analysis plots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Portfolio value over time
portfolio_history = backtest_results['portfolio_history']
dates = env.data['dates'][:len(portfolio_history)]
prices = env.data['prices'][:len(portfolio_history)]

# Normalize for comparison
portfolio_normalized = np.array(portfolio_history) / portfolio_history[0]
buy_hold_normalized = prices / prices[0]

ax1.plot(dates, portfolio_normalized, label='RL Agent', linewidth=2, color='blue')
ax1.plot(dates, buy_hold_normalized, label='Buy & Hold', linewidth=2, color='orange')
ax1.set_xlabel('Date')
ax1.set_ylabel('Normalized Value')
ax1.set_title('Portfolio Performance Comparison')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Training progress
if training_stats:
    episodes = [s['episode'] for s in training_stats]
    returns = [s.get('total_return', 0) for s in training_stats]
    
    # Smooth the curve
    window_size = 20
    if len(returns) >= window_size:
        smoothed_returns = np.convolve(returns, np.ones(window_size)/window_size, mode='valid')
        ax2.plot(episodes[window_size-1:], smoothed_returns, linewidth=2)
    else:
        ax2.plot(episodes, returns, linewidth=2)
    
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Total Return')
    ax2.set_title('Training Progress')
    ax2.grid(True, alpha=0.3)

# Plot 3: Action distribution over time
if 'action_history' in backtest_results:
    actions = backtest_results['action_history']
    action_dates = dates[:len(actions)]
    
    # Color-code actions
    colors = ['red' if a < -0.5 else 'gray' if abs(a) <= 0.5 else 'green' for a in actions]
    ax3.scatter(action_dates, actions, c=colors, alpha=0.6, s=10)
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Position')
    ax3.set_title('Trading Actions Over Time')
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax3.grid(True, alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='red', label='Sell'),
                      Patch(facecolor='gray', label='Hold'),
                      Patch(facecolor='green', label='Buy')]
    ax3.legend(handles=legend_elements, loc='upper right')

# Plot 4: Risk-return analysis
if training_stats:
    returns_list = [s.get('total_return', 0) for s in training_stats[-50:]]  # Last 50 episodes
    volatilities = [s.get('volatility', 0) for s in training_stats[-50:]]
    
    # Filter valid data
    valid_data = [(r, v) for r, v in zip(returns_list, volatilities) if v > 0]
    if valid_data:
        returns_valid, vol_valid = zip(*valid_data)
        ax4.scatter(vol_valid, returns_valid, alpha=0.6)
        ax4.set_xlabel('Volatility')
        ax4.set_ylabel('Return')
        ax4.set_title('Risk-Return Profile (Last 50 Episodes)')
        ax4.grid(True, alpha=0.3)
        
        # Add buy-and-hold point
        buy_hold_vol = np.std(np.diff(buy_hold_normalized)) * np.sqrt(252)
        ax4.scatter(buy_hold_vol, backtest_results['buy_hold_return'], 
                   color='orange', s=100, marker='*', label='Buy & Hold')
        ax4.legend()

plt.tight_layout()
plt.savefig('pytorch_rl_tutorial/chapter_10_trading_results.png', dpi=150, bbox_inches='tight')
print("Results saved to chapter_10_trading_results.png")

# 8. RISK ANALYSIS AND DISCUSSION
print("\n8. Risk Analysis and Limitations")
print("-" * 30)

print("""
RL TRADING ANALYSIS:

RESULTS:
- RL Agent vs Buy & Hold comparison shows relative performance
- Sharpe ratio indicates risk-adjusted returns
- Maximum drawdown shows worst-case losses
- Win rate indicates consistency

LIMITATIONS & RISKS:

1. OVERFITTING:
   - Model may memorize historical patterns
   - Poor generalization to new market conditions
   - Need robust validation on out-of-sample data

2. MARKET ASSUMPTIONS:
   - Perfect liquidity assumption
   - No market impact modeling
   - Simplified transaction costs
   - No slippage considerations

3. DATA LIMITATIONS:
   - Historical bias
   - Survivorship bias
   - Limited market regimes in training data
   - Missing fundamental data

4. MODEL LIMITATIONS:
   - Simple reward function
   - Limited state representation
   - No risk management constraints
   - Ignores market microstructure

IMPROVEMENTS FOR PRODUCTION:

1. ROBUST VALIDATION:
   - Walk-forward analysis
   - Multiple market regimes
   - Stress testing
   - Paper trading validation

2. ENHANCED FEATURES:
   - Fundamental data
   - Market sentiment
   - Economic indicators
   - Cross-asset signals

3. RISK MANAGEMENT:
   - Position sizing constraints
   - Stop-loss mechanisms
   - Volatility targeting
   - Portfolio diversification

4. REALISTIC MODELING:
   - Market impact costs
   - Bid-ask spreads
   - Slippage modeling
   - Liquidity constraints

WARNING: This is for educational purposes only!
Real trading involves substantial risk of loss.
Consult financial professionals before trading.
""")

print(f"\nChapter 10 Complete! ✓")
print(f"RL trading system demonstrated with performance analysis")
print(f"Agent achieved {backtest_results['total_return']*100:.1f}% return vs {backtest_results['buy_hold_return']*100:.1f}% buy-and-hold")
print("Ready to explore policy gradient methods (Chapter 11)")
