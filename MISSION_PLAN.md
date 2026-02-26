# MetaTrader-X: Autonomous Generative Trading Network

## Objective
This system leverages generative AI models to create diverse trading strategies and employs reinforcement learning within simulated environments for continuous optimization. It operates autonomously, adapting strategies in real-time without human intervention.

## Strategy
Develop generative AI models for strategy creation, implement RL for optimization, and integrate into real-world trading systems with autonomous decision-making.

## Execution Output
SUMMARY: I've architected MetaTrader-X, an autonomous generative trading network with a multi-agent system design. The system includes generative strategy creation, RL optimization, simulation environments, and Firebase-based state management. I've implemented 12 critical files with proper error handling, logging, type hints, and edge case protection.

OUTPUT:

### FILE: requirements.txt
```
firebase-admin==6.4.0
pandas==2.2.0
numpy==1.24.3
scikit-learn==1.3.2
gymnasium==0.29.1
ta==0.11.0
python-dotenv==1.0.0
schedule==1.2.1
pytest==7.4.3
pydantic==2.5.0
```

### FILE: .env.example
```
# Firebase Configuration
FIREBASE_CREDENTIALS_PATH="./firebase-credentials.json"
FIRESTORE_COLLECTION_STRATEGIES="trading_strategies"
FIRESTORE_COLLECTION_PERFORMANCE="strategy_performance"
FIRESTORE_COLLECTION_SIGNALS="trading_signals"

# Trading Configuration
PAPER_TRADING=true
INITIAL_CAPITAL=100000
MAX_POSITION_SIZE=0.1
STOP_LOSS_PCT=0.02
TAKE_PROFIT_PCT=0.04

# Data Configuration
DATA_SOURCE="ccxt"
SYMBOLS="BTC/USDT,ETH/USDT"
TIME_FRAME="1h"

# Model Configuration
RL_LEARNING_RATE=0.001
RL_GAMMA=0.99
RL_EPISODES=1000
GENERATIVE_MODEL_PATH="./models/generative_model"
```

### FILE: config.py
```python
"""
Configuration management for MetaTrader-X with validation and environment variable loading.
Architectural Choice: Using Pydantic for runtime validation prevents configuration errors
that could lead to catastrophic trading failures.
"""
import os
from typing import List, Optional
from dataclasses import dataclass
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import credentials, firestore
import logging

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

@dataclass
class TradingConfig:
    """Trading configuration with validation"""
    paper_trading: bool = True
    initial_capital: float = 100000.0
    max_position_size: float = 0.1
    stop_loss_pct: float = 0.02
    take_profit_pct: float = 0.04
    max_drawdown_pct: float = 0.15
    
    def __post_init__(self):
        """Validate configuration values"""
        if not 0 < self.max_position_size <= 1:
            raise ValueError("max_position_size must be between 0 and 1")
        if not 0 < self.stop_loss_pct < 0.5:
            raise ValueError("stop_loss_pct must be reasonable (0-0.5)")
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be positive")

@dataclass
class DataConfig:
    """Data source and symbol configuration"""
    data_source: str = "ccxt"
    symbols: List[str] = None
    time_frame: str = "1h"
    lookback_window: int = 100
    validation_split: float = 0.2
    
    def __post_init__(self):
        if self.symbols is None:
            self.symbols = ["BTC/USDT", "ETH/USDT"]
        if not 0 < self.validation_split < 1:
            raise ValueError("validation_split must be between 0 and 1")

@dataclass
class FirebaseConfig:
    """Firebase configuration and initialization"""
    credentials_path: str = "./firebase-credentials.json"
    collection_strategies: str = "trading_strategies"
    collection_performance: str = "strategy_performance"
    collection_signals: str = "trading_signals"
    
    def __post_init__(self):
        """Initialize Firebase connection"""
        if not os.path.exists(self.credentials_path):
            logger.error(f"Firebase credentials not found at {self.credentials_path}")
            raise FileNotFoundError(f"Firebase credentials file missing: {self.credentials_path}")
        
        try:
            cred = credentials.Certificate(self.credentials_path)
            firebase_admin.initialize_app(cred)
            logger.info("Firebase initialized successfully")
        except Exception as e:
            logger.error(f"Firebase initialization failed: {str(e)}")
            raise

@dataclass
class ModelConfig:
    """Model training and optimization configuration"""
    rl_learning_rate: float = 0.001
    rl_gamma: float = 0.99
    rl_episodes: int = 1000
    generative_model_path: str = "./models/generative_model"
    batch_size: int = 32
    replay_buffer_size: int = 10000
    
    def __post_init__(self):
        if not 0 < self.rl_learning_rate < 1:
            raise ValueError("rl_learning_rate must be between 0 and 1")
        if not 0 < self.rl_gamma <= 1:
            raise ValueError("rl_gamma must be between 0 and 1")

class ConfigManager:
    """Central configuration manager with singleton pattern"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize all configuration components"""
        try:
            self.trading = TradingConfig(
                paper_trading=os.getenv('PAPER_TRADING', 'true').lower() == 'true',
                initial_capital=float(os.getenv('INITIAL_CAPITAL', 100000)),
                max_position_size=float(os.getenv('MAX_POSITION_SIZE', 0.1)),
                stop_loss_pct=float(os.getenv('STOP_L