# Systematic Trading System

A modular Python framework for systematic trading following the **Data → Features → Signals → Portfolio → Execution** architecture. This implementation focuses on leveraged ETFs (SQQQ/TQQQ) with comprehensive risk management.

## Architecture Overview

The system is built around five core components that work together in a pipeline:

| Layer | Component | Description |
|-------|-----------|-------------|
| **Data** | `DataHandler` | Fetches historical OHLCV data from Yahoo Finance for single or multiple tickers |
| **Features** | `FeatureEngineer` | Computes technical indicators including Moving Averages, RSI, MACD, Bollinger Bands, ATR, and momentum metrics |
| **Signals** | `SignalGenerator` | Abstract base class with implementations for MA Crossover, RSI Mean Reversion, and Momentum strategies |
| **Portfolio** | `PortfolioManager` | Manages position sizing with volatility targeting, stop-loss, and take-profit rules |
| **Execution** | `ExecutionEngine` | Simulates trade execution with transaction costs and generates performance reports |

## Risk Management Parameters

The `RiskParameters` dataclass provides configurable risk controls:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_position_size` | 25% | Maximum percentage of capital allocated to a single position |
| `stop_loss_pct` | 8% | Stop-loss trigger level (wider for leveraged ETFs) |
| `take_profit_pct` | 15% | Take-profit trigger level |
| `max_drawdown_limit` | 25% | Maximum allowable portfolio drawdown |
| `volatility_lookback` | 20 days | Lookback period for volatility calculation |
| `volatility_target` | 20% | Target annualized volatility for position sizing |
| `max_leverage` | 1.0 | Maximum leverage multiplier |

## Included Strategies

### 1. Moving Average Crossover (`MACrossoverSignal`)
Generates long signals when the short-term MA crosses above the long-term MA. Default parameters use 10-day and 50-day moving averages.

### 2. Momentum Strategy (`MomentumSignal`)
Uses price momentum with an optional volatility filter that reduces position size during high-volatility regimes (>40% annualized).

### 3. RSI Mean Reversion (`RSIMeanReversionSignal`)
Buys when RSI falls below the oversold threshold (30) and exits when RSI rises above the overbought threshold (70).

## Installation

```bash
pip install pandas numpy yfinance matplotlib
```

## Usage

### Basic Usage

```python
from core import TradingSystem, RiskParameters, MACrossoverSignal

# Configure risk parameters
risk_params = RiskParameters(
    max_position_size=0.25,
    stop_loss_pct=0.08,
    take_profit_pct=0.15
)

# Initialize and run the system
system = TradingSystem(
    tickers=["SQQQ", "TQQQ"],
    start_date="2020-01-01",
    end_date="2025-12-31",
    initial_capital=100000.0,
    risk_params=risk_params,
    signal_generator=MACrossoverSignal(short_window=10, long_window=50)
)

performance = system.run()
```

### Running the Full Backtest

```bash
python main.py
```

This will run all three strategies on SQQQ and TQQQ, generate performance reports, and save visualizations.

## Performance Metrics

The system calculates and reports the following metrics:

| Metric | Description |
|--------|-------------|
| **Total Return** | Cumulative return over the backtest period |
| **Annualized Return** | Geometric average annual return |
| **Annualized Volatility** | Standard deviation of returns, annualized |
| **Sharpe Ratio** | Risk-adjusted return (excess return / volatility) |
| **Sortino Ratio** | Downside risk-adjusted return |
| **Calmar Ratio** | Annualized return / maximum drawdown |
| **Max Drawdown** | Largest peak-to-trough decline |
| **Win Rate** | Percentage of profitable trades |

## Output Files

After running the backtest, the following files are generated:

- `backtest_results.csv` - Summary of all strategy/ticker combinations
- `backtest_results.png` - Visualization of equity curves, drawdowns, and rolling Sharpe ratios

## Project Structure

```
systematic_trading/
├── __init__.py
├── core.py          # Core trading system components
├── main.py          # SQQQ/TQQQ backtest runner
├── README.md        # This documentation
├── backtest_results.csv
└── backtest_results.png
```

## Extending the System

### Adding a New Strategy

Create a new class that inherits from `SignalGenerator`:

```python
class MyCustomSignal(SignalGenerator):
    def generate_signals(self, features: pd.DataFrame) -> pd.DataFrame:
        signals = pd.DataFrame(index=features.index)
        signals['signal'] = 0.0
        # Your signal logic here
        signals['positions'] = signals['signal'].diff()
        return signals
```

### Adding New Features

Extend the `FeatureEngineer` class with additional technical indicators:

```python
def add_custom_indicator(self, param: int = 14) -> pd.DataFrame:
    # Your indicator calculation
    self.data['CustomIndicator'] = ...
    return self.data
```

## Disclaimer

This software is for educational and research purposes only. It is not intended as financial advice. Trading leveraged ETFs like SQQQ and TQQQ carries significant risk, including the potential for substantial losses. Always conduct your own research and consider consulting a financial advisor before making investment decisions.

## License

MIT License
