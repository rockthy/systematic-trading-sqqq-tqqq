"""
Systematic Trading System - SQQQ/TQQQ Backtest
Demonstrates the full trading pipeline with risk management.
"""

import matplotlib.pyplot as plt
import pandas as pd
from core import (
    TradingSystem, 
    RiskParameters, 
    MACrossoverSignal, 
    MomentumSignal,
    RSIMeanReversionSignal
)


def run_sqqq_tqqq_backtest():
    """
    Run backtest on SQQQ and TQQQ with enhanced risk management.
    """
    # Configuration
    tickers = ["SQQQ", "TQQQ"]
    start_date = "2020-01-01"
    end_date = "2025-12-31"
    initial_capital = 100000.0
    
    # Risk Management Parameters
    risk_params = RiskParameters(
        max_position_size=0.25,        # Max 25% of capital per position
        stop_loss_pct=0.08,            # 8% stop loss (wider for leveraged ETFs)
        take_profit_pct=0.15,          # 15% take profit
        max_drawdown_limit=0.25,       # 25% max drawdown limit
        volatility_lookback=20,        # 20-day volatility lookback
        volatility_target=0.20,        # 20% target volatility
        max_leverage=1.0,              # No additional leverage
        correlation_threshold=0.7
    )
    
    print("="*60)
    print("SYSTEMATIC TRADING SYSTEM - SQQQ/TQQQ BACKTEST")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Tickers: {tickers}")
    print(f"  Period: {start_date} to {end_date}")
    print(f"  Initial Capital: ${initial_capital:,.2f}")
    print(f"\nRisk Parameters:")
    print(f"  Max Position Size: {risk_params.max_position_size:.0%}")
    print(f"  Stop Loss: {risk_params.stop_loss_pct:.0%}")
    print(f"  Take Profit: {risk_params.take_profit_pct:.0%}")
    print(f"  Max Drawdown Limit: {risk_params.max_drawdown_limit:.0%}")
    print(f"  Volatility Target: {risk_params.volatility_target:.0%}")
    
    # Strategy 1: Moving Average Crossover
    print("\n" + "="*60)
    print("STRATEGY 1: Moving Average Crossover (10/50)")
    print("="*60)
    
    ma_signal = MACrossoverSignal(short_window=10, long_window=50)
    system_ma = TradingSystem(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        risk_params=risk_params,
        signal_generator=ma_signal
    )
    
    performance_ma = system_ma.run()
    
    # Strategy 2: Momentum Strategy
    print("\n" + "="*60)
    print("STRATEGY 2: Momentum (10-day) with Volatility Filter")
    print("="*60)
    
    momentum_signal = MomentumSignal(momentum_window=10, volatility_filter=True)
    system_momentum = TradingSystem(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        risk_params=risk_params,
        signal_generator=momentum_signal
    )
    
    performance_momentum = system_momentum.run()
    
    # Strategy 3: RSI Mean Reversion
    print("\n" + "="*60)
    print("STRATEGY 3: RSI Mean Reversion (30/70)")
    print("="*60)
    
    rsi_signal = RSIMeanReversionSignal(oversold=30, overbought=70)
    system_rsi = TradingSystem(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        risk_params=risk_params,
        signal_generator=rsi_signal
    )
    
    performance_rsi = system_rsi.run()
    
    # Create comparison summary
    print("\n" + "="*60)
    print("STRATEGY COMPARISON SUMMARY")
    print("="*60)
    
    summary_data = []
    for ticker in tickers:
        summary_data.append({
            'Ticker': ticker,
            'Strategy': 'MA Crossover',
            'Total Return': performance_ma.get(ticker, {}).get('total_return', 0),
            'Sharpe Ratio': performance_ma.get(ticker, {}).get('sharpe_ratio', 0),
            'Max Drawdown': performance_ma.get(ticker, {}).get('max_drawdown', 0),
            'Win Rate': performance_ma.get(ticker, {}).get('win_rate', 0)
        })
        summary_data.append({
            'Ticker': ticker,
            'Strategy': 'Momentum',
            'Total Return': performance_momentum.get(ticker, {}).get('total_return', 0),
            'Sharpe Ratio': performance_momentum.get(ticker, {}).get('sharpe_ratio', 0),
            'Max Drawdown': performance_momentum.get(ticker, {}).get('max_drawdown', 0),
            'Win Rate': performance_momentum.get(ticker, {}).get('win_rate', 0)
        })
        summary_data.append({
            'Ticker': ticker,
            'Strategy': 'RSI Mean Reversion',
            'Total Return': performance_rsi.get(ticker, {}).get('total_return', 0),
            'Sharpe Ratio': performance_rsi.get(ticker, {}).get('sharpe_ratio', 0),
            'Max Drawdown': performance_rsi.get(ticker, {}).get('max_drawdown', 0),
            'Win Rate': performance_rsi.get(ticker, {}).get('win_rate', 0)
        })
    
    summary_df = pd.DataFrame(summary_data)
    print("\n")
    print(summary_df.to_string(index=False, formatters={
        'Total Return': '{:.2%}'.format,
        'Sharpe Ratio': '{:.2f}'.format,
        'Max Drawdown': '{:.2%}'.format,
        'Win Rate': '{:.2%}'.format
    }))
    
    # Save summary to CSV
    summary_df.to_csv('/home/ubuntu/systematic_trading/backtest_results.csv', index=False)
    print("\nResults saved to backtest_results.csv")
    
    # Create visualization
    create_visualizations(system_ma, system_momentum, system_rsi, tickers)
    
    return {
        'ma_crossover': performance_ma,
        'momentum': performance_momentum,
        'rsi_mean_reversion': performance_rsi,
        'summary': summary_df
    }


def create_visualizations(system_ma, system_momentum, system_rsi, tickers):
    """
    Create performance visualization charts.
    """
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle('SQQQ/TQQQ Systematic Trading Backtest Results', fontsize=14, fontweight='bold')
    
    colors = {'SQQQ': '#e74c3c', 'TQQQ': '#27ae60'}
    
    for idx, ticker in enumerate(tickers):
        # Equity curves for MA Crossover
        ax1 = axes[0, idx]
        if ticker in system_ma.results:
            results = system_ma.results[ticker]
            ax1.plot(results.index, results['total'], label='MA Crossover', color='blue', alpha=0.8)
        if ticker in system_momentum.results:
            results = system_momentum.results[ticker]
            ax1.plot(results.index, results['total'], label='Momentum', color='orange', alpha=0.8)
        if ticker in system_rsi.results:
            results = system_rsi.results[ticker]
            ax1.plot(results.index, results['total'], label='RSI', color='green', alpha=0.8)
        
        ax1.set_title(f'{ticker} - Portfolio Value')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=100000, color='gray', linestyle='--', alpha=0.5)
        
        # Drawdown for all strategies
        ax2 = axes[1, idx]
        if ticker in system_ma.results:
            results = system_ma.results[ticker]
            ax2.fill_between(results.index, results['drawdown'] * 100, 0, 
                           label='MA Crossover', color='blue', alpha=0.3)
        if ticker in system_momentum.results:
            results = system_momentum.results[ticker]
            ax2.fill_between(results.index, results['drawdown'] * 100, 0, 
                           label='Momentum', color='orange', alpha=0.3)
        if ticker in system_rsi.results:
            results = system_rsi.results[ticker]
            ax2.fill_between(results.index, results['drawdown'] * 100, 0, 
                           label='RSI', color='green', alpha=0.3)
        
        ax2.set_title(f'{ticker} - Drawdown')
        ax2.set_ylabel('Drawdown (%)')
        ax2.legend(loc='lower left')
        ax2.grid(True, alpha=0.3)
        
        # Rolling Sharpe Ratio (252-day)
        ax3 = axes[2, idx]
        if ticker in system_ma.results:
            results = system_ma.results[ticker]
            rolling_sharpe = (results['returns'].rolling(252).mean() / 
                            results['returns'].rolling(252).std()) * np.sqrt(252)
            ax3.plot(results.index, rolling_sharpe, label='MA Crossover', color='blue', alpha=0.8)
        if ticker in system_momentum.results:
            results = system_momentum.results[ticker]
            rolling_sharpe = (results['returns'].rolling(252).mean() / 
                            results['returns'].rolling(252).std()) * np.sqrt(252)
            ax3.plot(results.index, rolling_sharpe, label='Momentum', color='orange', alpha=0.8)
        if ticker in system_rsi.results:
            results = system_rsi.results[ticker]
            rolling_sharpe = (results['returns'].rolling(252).mean() / 
                            results['returns'].rolling(252).std()) * np.sqrt(252)
            ax3.plot(results.index, rolling_sharpe, label='RSI', color='green', alpha=0.8)
        
        ax3.set_title(f'{ticker} - Rolling Sharpe Ratio (252-day)')
        ax3.set_ylabel('Sharpe Ratio')
        ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax3.legend(loc='upper left')
        ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/systematic_trading/backtest_results.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved to backtest_results.png")


# Need to import numpy for visualization
import numpy as np

if __name__ == "__main__":
    results = run_sqqq_tqqq_backtest()
