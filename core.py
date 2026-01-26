"""
Systematic Trading System
Architecture: Data -> Features -> Signals -> Portfolio -> Execution

This module implements a modular systematic trading framework with
enhanced risk management capabilities for leveraged ETFs (SQQQ/TQQQ).
"""

import pandas as pd
import numpy as np
import yfinance as yf
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, List


@dataclass
class RiskParameters:
    """
    Risk management parameters for the trading system.
    """
    max_position_size: float = 0.25          # Maximum % of capital per position
    stop_loss_pct: float = 0.05              # Stop loss percentage (5%)
    take_profit_pct: float = 0.10            # Take profit percentage (10%)
    max_drawdown_limit: float = 0.20         # Maximum allowed drawdown (20%)
    volatility_lookback: int = 20            # Lookback period for volatility calculation
    volatility_target: Optional[float] = 0.15  # Target annualized volatility (15%), or None to disable
    max_leverage: float = 1.0                # Maximum leverage allowed
    correlation_threshold: float = 0.7       # Correlation threshold for position sizing


class DataHandler:
    """
    Handles data ingestion from various sources.
    Supports single and multiple tickers.
    """
    def __init__(self, tickers: List[str], start_date: str, end_date: str):
        self.tickers = tickers if isinstance(tickers, list) else [tickers]
        self.start_date = start_date
        self.end_date = end_date
        self.data = {}

    def download_data(self) -> Dict[str, pd.DataFrame]:
        """
        Downloads historical data using yfinance for all tickers.
        Returns a dictionary of DataFrames keyed by ticker.
        """
        print(f"Downloading data for {self.tickers} from {self.start_date} to {self.end_date}...")

        for ticker in self.tickers:
            df = yf.download(ticker, start=self.start_date, end=self.end_date, progress=False)

            # Flatten MultiIndex if it exists
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            self.data[ticker] = df
            print(f"  {ticker}: {len(df)} rows downloaded")

        return self.data


class FeatureEngineer:
    """
    Generates features from raw OHLCV data.
    Includes technical indicators and risk metrics.
    """
    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()

    def add_moving_averages(self, windows: List[int] = [10, 20, 50, 200]) -> pd.DataFrame:
        for window in windows:
            self.data[f'MA_{window}'] = self.data['Close'].rolling(window=window).mean()
        return self.data

    def add_exponential_ma(self, windows: List[int] = [12, 26]) -> pd.DataFrame:
        for window in windows:
            self.data[f'EMA_{window}'] = self.data['Close'].ewm(span=window, adjust=False).mean()
        return self.data

    def add_rsi(self, window: int = 14) -> pd.DataFrame:
        delta = self.data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        self.data['RSI'] = 100 - (100 / (1 + rs))
        return self.data

    def add_macd(self) -> pd.DataFrame:
        ema12 = self.data['Close'].ewm(span=12, adjust=False).mean()
        ema26 = self.data['Close'].ewm(span=26, adjust=False).mean()
        self.data['MACD'] = ema12 - ema26
        self.data['MACD_Signal'] = self.data['MACD'].ewm(span=9, adjust=False).mean()
        self.data['MACD_Hist'] = self.data['MACD'] - self.data['MACD_Signal']
        return self.data

    def add_bollinger_bands(self, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
        rolling_mean = self.data['Close'].rolling(window=window).mean()
        rolling_std = self.data['Close'].rolling(window=window).std()
        self.data['BB_Upper'] = rolling_mean + (rolling_std * num_std)
        self.data['BB_Lower'] = rolling_mean - (rolling_std * num_std)
        self.data['BB_Middle'] = rolling_mean
        self.data['BB_Width'] = (self.data['BB_Upper'] - self.data['BB_Lower']) / self.data['BB_Middle']
        return self.data

    def add_atr(self, window: int = 14) -> pd.DataFrame:
        """Average True Range for volatility-based position sizing."""
        high = self.data['High']
        low = self.data['Low']
        close = self.data['Close']

        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        self.data['ATR'] = tr.rolling(window=window).mean()
        self.data['ATR_Pct'] = self.data['ATR'] / close
        return self.data

    def add_volatility(self, window: int = 20) -> pd.DataFrame:
        self.data['Returns'] = self.data['Close'].pct_change()
        self.data['Volatility'] = self.data['Returns'].rolling(window=window).std()
        self.data['Volatility_Ann'] = self.data['Volatility'] * np.sqrt(252)
        return self.data

    def add_momentum(self, windows: List[int] = [5, 10, 20]) -> pd.DataFrame:
        for window in windows:
            self.data[f'Momentum_{window}'] = self.data['Close'].pct_change(periods=window)
        return self.data

    def get_features(self) -> pd.DataFrame:
        """Generate all features and return cleaned DataFrame."""
        self.add_moving_averages()
        self.add_exponential_ma()
        self.add_rsi()
        self.add_macd()
        self.add_bollinger_bands()
        self.add_atr()
        self.add_volatility()
        self.add_momentum()
        return self.data.dropna()


class SignalGenerator(ABC):
    """Abstract base class for signal generation."""

    @abstractmethod
    def generate_signals(self, features: pd.DataFrame) -> pd.DataFrame:
        pass


class MACrossoverSignal(SignalGenerator):
    """
    Moving Average Crossover Strategy.
    Generates long signals when short MA crosses above long MA.
    """
    def __init__(self, short_window: int = 10, long_window: int = 50):
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self, features: pd.DataFrame) -> pd.DataFrame:
        signals = pd.DataFrame(index=features.index)
        signals['signal'] = 0.0

        short_ma = f'MA_{self.short_window}'
        long_ma = f'MA_{self.long_window}'

        signals['signal'] = np.where(features[short_ma] > features[long_ma], 1.0, 0.0)
        signals['positions'] = signals['signal'].diff()

        return signals


class RSIMeanReversionSignal(SignalGenerator):
    """
    RSI-based Mean Reversion Strategy.
    Buys when RSI is oversold, sells when overbought.
    """
    def __init__(self, oversold: float = 30, overbought: float = 70):
        self.oversold = oversold
        self.overbought = overbought

    def generate_signals(self, features: pd.DataFrame) -> pd.DataFrame:
        signals = pd.DataFrame(index=features.index)
        signals['signal'] = 0.0

        # Long when oversold, exit when overbought
        signals.loc[features['RSI'] < self.oversold, 'signal'] = 1.0
        signals.loc[features['RSI'] > self.overbought, 'signal'] = -1.0

        # Forward fill signals
        signals['signal'] = signals['signal'].replace(0, np.nan).ffill().fillna(0)
        signals['positions'] = signals['signal'].diff()

        return signals


class EnhancedRSIMeanReversionSignal(SignalGenerator):
    """
    Enhanced RSI Mean Reversion Strategy with:
    - Adjusted thresholds (40/60) for more frequent trading
    - MACD confirmation to filter false signals
    - Time-based holding period (5-10 days)
    - Position scaling based on RSI intensity
    """
    def __init__(self, oversold: float = 40, overbought: float = 60,
                 hold_days: int = 7, use_macd_confirmation: bool = True):
        self.oversold = oversold
        self.overbought = overbought
        self.hold_days = hold_days
        self.use_macd_confirmation = use_macd_confirmation

    def generate_signals(self, features: pd.DataFrame) -> pd.DataFrame:
        signals = pd.DataFrame(index=features.index)
        signals['signal'] = 0.0

        # Base RSI signals
        rsi_oversold = features['RSI'] < self.oversold
        rsi_overbought = features['RSI'] > self.overbought

        # MACD confirmation (if enabled)
        if self.use_macd_confirmation and 'MACD_Hist' in features.columns:
            # For long: RSI oversold + MACD below signal line (trending down)
            macd_confirms_oversold = features['MACD'] < features['MACD_Signal']
            entry_signal = rsi_oversold & macd_confirms_oversold

            # For exit: RSI overbought OR MACD crosses above signal (trend reversal)
            exit_signal = rsi_overbought | (features['MACD'] > features['MACD_Signal'])
        else:
            entry_signal = rsi_oversold
            exit_signal = rsi_overbought

        # Generate position signals
        signals['entry'] = entry_signal.astype(int)
        signals['exit'] = exit_signal.astype(int)

        # Implement time-based holding period
        signals['holding'] = 0
        in_trade = False
        entry_idx = None

        for i in range(len(signals)):
            if signals.iloc[i]['entry'] and not in_trade:
                in_trade = True
                entry_idx = i
                signals.iloc[i, signals.columns.get_loc('holding')] = 1
            elif in_trade:
                days_held = i - entry_idx
                if days_held >= self.hold_days or signals.iloc[i]['exit']:
                    in_trade = False
                else:
                    signals.iloc[i, signals.columns.get_loc('holding')] = 1

        # Position scaling based on RSI intensity
        # Stronger signals = larger positions
        rsi_intensity = np.abs(50 - features['RSI']) / 50  # 0 to 1, higher = more extreme
        signals['position_scale'] = rsi_intensity.clip(0.5, 1.0)  # Scale between 0.5 and 1.0

        # Final signal: 1.0 when holding, scaled by RSI intensity
        signals['signal'] = signals['holding'].astype(float) * signals['position_scale']
        signals['positions'] = signals['signal'].diff()

        return signals


class RefinedTickerSpecificSignal(SignalGenerator):
    """
    Refined ticker-specific strategy with different logic for TQQQ vs SQQQ:

    TQQQ (3x Leveraged Nasdaq): Only buy dips when in uptrend
    - Entry: Price > SMA200 (uptrend) AND RSI < 35 (dip)
    - Hold for 7-14 days

    SQQQ (3x Inverse Nasdaq): Buy confirmed crashes only with scalp exits
    - Entry: Price > SMA50 (bounce) AND RSI < 30 (extreme crash)
    - Exit: Within 48 hours (scalp mode) or when RSI > 60
    """
    def __init__(self, tqqq_rsi_threshold: float = 35, sqqq_rsi_threshold: float = 30,
                 scalp_hold_hours: int = 48, normal_hold_days: int = 10):
        self.tqqq_rsi_threshold = tqqq_rsi_threshold
        self.sqqq_rsi_threshold = sqqq_rsi_threshold
        self.scalp_hold_hours = scalp_hold_hours
        self.normal_hold_days = normal_hold_days
        self.ticker = None  # Will be set by TradingSystem

    def set_ticker(self, ticker: str):
        """Set the current ticker being analyzed."""
        self.ticker = ticker

    def generate_signals(self, features: pd.DataFrame, ticker: Optional[str] = None) -> pd.DataFrame:
        """
        Generate signals based on ticker-specific logic.
        """
        if ticker is not None:
            self.ticker = ticker

        if self.ticker is None:
            raise ValueError("Ticker must be set before generating signals")

        signals = pd.DataFrame(index=features.index)
        signals['signal'] = 0.0
        signals['entry'] = False
        signals['exit'] = False

        if self.ticker == "TQQQ":
            # TQQQ: Buy dips in uptrend
            # Requires SMA200 in features
            if 'MA_200' not in features.columns:
                # Fallback: create SMA200 if not present
                features['MA_200'] = features['Close'].rolling(window=200).mean()

            # Entry condition: Price above SMA200 (uptrend) AND RSI below threshold (dip)
            uptrend = features['Close'] > features['MA_200']
            oversold = features['RSI'] < self.tqqq_rsi_threshold
            entry_condition = uptrend & oversold

            # Exit condition: RSI recovers above 65 (bounce complete)
            exit_condition = features['RSI'] > 65

        elif self.ticker == "SQQQ":
            # SQQQ: Buy confirmed crashes with scalp exits
            # Entry: Price above SMA50 AND RSI extremely low (confirmed crash)
            if 'MA_50' not in features.columns:
                features['MA_50'] = features['Close'].rolling(window=50).mean()

            bounce_setup = features['Close'] > features['MA_50']
            crash = features['RSI'] < self.sqqq_rsi_threshold
            entry_condition = bounce_setup & crash

            # Exit condition: RSI recovers above 50 OR 48 hours elapsed (will be handled by holding period)
            exit_condition = features['RSI'] > 50

        else:
            # Fallback for other tickers: simple RSI mean reversion
            entry_condition = features['RSI'] < 35
            exit_condition = features['RSI'] > 65

        # Implement holding logic with entry/exit tracking
        signals['entry'] = entry_condition.astype(int)
        signals['exit'] = exit_condition.astype(int)

        in_trade = False
        entry_idx = None

        hold_periods = (self.scalp_hold_hours // 24) if self.ticker == "SQQQ" else self.normal_hold_days

        for i in range(len(signals)):
            if signals.iloc[i]['entry'] and not in_trade:
                in_trade = True
                entry_idx = i
                signals.iloc[i, signals.columns.get_loc('signal')] = 1.0
            elif in_trade:
                days_held = i - entry_idx
                # Exit if: hold period exceeded OR exit condition met
                if days_held >= hold_periods or signals.iloc[i]['exit']:
                    in_trade = False
                else:
                    signals.iloc[i, signals.columns.get_loc('signal')] = 1.0

        # Calculate position scale based on RSI intensity
        rsi_intensity = np.abs(50 - features['RSI']) / 50
        signals['position_scale'] = rsi_intensity.clip(0.5, 1.0)

        # Apply position scaling
        signals['signal'] = signals['signal'].astype(float) * signals['position_scale']
        signals['positions'] = signals['signal'].diff()

        return signals


class MomentumSignal(SignalGenerator):
    """
    Momentum-based Strategy for leveraged ETFs.
    Uses multiple momentum indicators and volatility filtering.
    """
    def __init__(self, momentum_window: int = 10, volatility_filter: bool = True):
        self.momentum_window = momentum_window
        self.volatility_filter = volatility_filter

    def generate_signals(self, features: pd.DataFrame) -> pd.DataFrame:
        signals = pd.DataFrame(index=features.index)
        signals['signal'] = 0.0

        momentum_col = f'Momentum_{self.momentum_window}'

        # Base signal from momentum
        signals['signal'] = np.where(features[momentum_col] > 0, 1.0, 0.0)

        # Apply volatility filter if enabled
        if self.volatility_filter and 'Volatility_Ann' in features.columns:
            # Reduce position in high volatility environments
            high_vol_mask = features['Volatility_Ann'] > 0.40  # 40% annualized vol
            signals.loc[high_vol_mask, 'signal'] *= 0.5

        signals['positions'] = signals['signal'].diff()

        return signals


class PortfolioManager:
    """
    Manages position sizing and risk with enhanced risk management.
    """
    def __init__(self, initial_capital: float = 100000.0, risk_params: Optional[RiskParameters] = None):
        self.initial_capital = initial_capital
        self.risk_params = risk_params or RiskParameters()

    def calculate_volatility_adjusted_size(self, volatility: float) -> float:
        """
        Calculate position size based on volatility targeting.
        """
        if volatility <= 0 or np.isnan(volatility):
            return 0.0

        target_vol = self.risk_params.volatility_target

        # If volatility_target is None, use max_position_size directly
        if target_vol is None:
            return self.risk_params.max_position_size

        raw_size = target_vol / (volatility * np.sqrt(252))

        # Cap at max position size
        return min(raw_size, self.risk_params.max_position_size)

    def calculate_positions(self, signals: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates position sizes with risk management.
        """
        positions = pd.DataFrame(index=signals.index)

        close_prices = prices['Close']
        if isinstance(close_prices, pd.DataFrame):
            close_prices = close_prices.iloc[:, 0]

        # Get volatility for position sizing
        volatility = prices.get('Volatility', pd.Series(0.02, index=prices.index))
        if isinstance(volatility, pd.DataFrame):
            volatility = volatility.iloc[:, 0]

        # Calculate volatility-adjusted position sizes
        position_sizes = volatility.apply(self.calculate_volatility_adjusted_size)

        # Apply maximum position size constraint
        position_sizes = position_sizes.clip(upper=self.risk_params.max_position_size)

        # Calculate shares
        capital_per_position = self.initial_capital * position_sizes
        positions['shares'] = np.floor(capital_per_position / close_prices) * signals['signal']

        # Store position size for reporting
        positions['position_size_pct'] = position_sizes * signals['signal'].abs()

        return positions

    def apply_stop_loss_take_profit(self, positions: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Apply stop-loss and take-profit rules.
        """
        close_prices = prices['Close']
        if isinstance(close_prices, pd.DataFrame):
            close_prices = close_prices.iloc[:, 0]

        positions = positions.copy()
        entry_price = None

        for i in range(1, len(positions)):
            if positions['shares'].iloc[i-1] == 0 and positions['shares'].iloc[i] != 0:
                # New position opened
                entry_price = close_prices.iloc[i]
            elif entry_price is not None and positions['shares'].iloc[i] != 0:
                current_price = close_prices.iloc[i]
                pnl_pct = (current_price - entry_price) / entry_price

                # Check stop-loss
                if pnl_pct < -self.risk_params.stop_loss_pct:
                    positions.iloc[i:, positions.columns.get_loc('shares')] = 0
                    entry_price = None
                # Check take-profit
                elif pnl_pct > self.risk_params.take_profit_pct:
                    positions.iloc[i:, positions.columns.get_loc('shares')] = 0
                    entry_price = None

        return positions


class ExecutionEngine:
    """
    Simulates execution, accounts for costs, and reports performance.
    """
    def __init__(self, commission: float = 0.001, slippage: float = 0.0005):
        self.commission = commission
        self.slippage = slippage

    def run_backtest(self, positions: pd.DataFrame, prices: pd.DataFrame, initial_capital: float) -> pd.DataFrame:
        """
        Executes trades and calculates portfolio value over time.
        """
        df = pd.DataFrame(index=positions.index)

        close_prices = prices['Close']
        if isinstance(close_prices, pd.DataFrame):
            close_prices = close_prices.iloc[:, 0]

        df['price'] = close_prices
        df['shares'] = positions['shares']
        df['pos_diff'] = df['shares'].diff().fillna(df['shares'].iloc[0])

        # Transaction costs
        df['costs'] = (df['pos_diff'].abs() * df['price'] * (self.commission + self.slippage))

        # Cash tracking
        df['trade_value'] = df['pos_diff'] * df['price']
        df['cash'] = initial_capital - (df['trade_value'] + df['costs']).cumsum()

        # Portfolio value
        df['holdings'] = df['shares'] * df['price']
        df['total'] = df['cash'] + df['holdings']
        df['returns'] = df['total'].pct_change()

        # Calculate drawdown
        df['peak'] = df['total'].cummax()
        df['drawdown'] = (df['total'] - df['peak']) / df['peak']

        return df

    def report_performance(self, results: pd.DataFrame, ticker: str = "") -> Dict:
        """
        Calculates and prints key performance indicators.
        """
        total_return = (results['total'].iloc[-1] / results['total'].iloc[0]) - 1
        num_days = len(results)
        annualized_return = (1 + total_return) ** (252 / num_days) - 1

        # Volatility and Sharpe
        daily_vol = results['returns'].std()
        annualized_vol = daily_vol * np.sqrt(252)
        risk_free_rate = 0.05  # Assume 5% risk-free rate
        sharpe_ratio = (annualized_return - risk_free_rate) / annualized_vol if annualized_vol != 0 else 0

        # Sortino Ratio (downside deviation)
        downside_returns = results['returns'][results['returns'] < 0]
        downside_vol = downside_returns.std() * np.sqrt(252)
        sortino_ratio = (annualized_return - risk_free_rate) / downside_vol if downside_vol != 0 else 0

        # Drawdown metrics
        max_drawdown = results['drawdown'].min()

        # Win rate
        trades = results['pos_diff'] != 0
        winning_trades = (results.loc[trades, 'returns'] > 0).sum()
        total_trades = trades.sum()
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        # Calmar Ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0

        print(f"\n{'='*50}")
        print(f"Performance Report: {ticker}")
        print(f"{'='*50}")
        print(f"Period: {results.index[0].strftime('%Y-%m-%d')} to {results.index[-1].strftime('%Y-%m-%d')}")
        print(f"Trading Days: {num_days}")
        print(f"-"*50)
        print(f"Total Return:      {total_return:>10.2%}")
        print(f"Annualized Return: {annualized_return:>10.2%}")
        print(f"Annualized Vol:    {annualized_vol:>10.2%}")
        print(f"-"*50)
        print(f"Sharpe Ratio:      {sharpe_ratio:>10.2f}")
        print(f"Sortino Ratio:     {sortino_ratio:>10.2f}")
        print(f"Calmar Ratio:      {calmar_ratio:>10.2f}")
        print(f"-"*50)
        print(f"Max Drawdown:      {max_drawdown:>10.2%}")
        print(f"Win Rate:          {win_rate:>10.2%}")
        print(f"Total Trades:      {total_trades:>10.0f}")
        print(f"{'='*50}")

        return {
            "ticker": ticker,
            "total_return": total_return,
            "annualized_return": annualized_return,
            "annualized_vol": annualized_vol,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "calmar_ratio": calmar_ratio,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "total_trades": total_trades
        }


class TradingSystem:
    """
    Orchestrates the entire trading pipeline.
    """
    def __init__(self,
                 tickers: List[str],
                 start_date: str,
                 end_date: str,
                 initial_capital: float = 100000.0,
                 risk_params: Optional[RiskParameters] = None,
                 signal_generator: Optional[SignalGenerator] = None):

        self.tickers = tickers if isinstance(tickers, list) else [tickers]
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.risk_params = risk_params or RiskParameters()
        self.signal_generator = signal_generator or MACrossoverSignal()

        self.data_handler = DataHandler(self.tickers, start_date, end_date)
        self.portfolio_manager = PortfolioManager(initial_capital, self.risk_params)
        self.execution_engine = ExecutionEngine()

        self.results = {}
        self.performance = {}

    def run(self) -> Dict:
        """
        Execute the full trading pipeline for all tickers.
        """
        # 1. Data Layer
        raw_data = self.data_handler.download_data()

        for ticker in self.tickers:
            print(f"\nProcessing {ticker}...")

            if raw_data[ticker].empty:
                print(f"  No data for {ticker}. Skipping.")
                continue

            # 2. Feature Layer
            engineer = FeatureEngineer(raw_data[ticker])
            features = engineer.get_features()

            # 3. Signal Layer
            # Pass ticker to signal generator if it supports it (for ticker-specific strategies)
            if hasattr(self.signal_generator, 'set_ticker'):
                self.signal_generator.set_ticker(ticker)
                signals = self.signal_generator.generate_signals(features, ticker=ticker)
            else:
                signals = self.signal_generator.generate_signals(features)

            # 4. Portfolio Layer
            positions = self.portfolio_manager.calculate_positions(signals, features)

            # Apply stop-loss and take-profit
            positions = self.portfolio_manager.apply_stop_loss_take_profit(positions, features)

            # 5. Execution Layer
            results = self.execution_engine.run_backtest(
                positions, features, self.initial_capital
            )

            # Store results
            self.results[ticker] = results
            self.performance[ticker] = self.execution_engine.report_performance(results, ticker)

        return self.performance

    def get_combined_results(self) -> pd.DataFrame:
        """
        Combine results from all tickers into a summary DataFrame.
        """
        return pd.DataFrame(self.performance).T
