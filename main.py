"""
Systematic Trading System - Multi-Ticker Backtest
Demonstrates the full trading pipeline with risk management for SQQQ, TQQQ, and UNH.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime
from core import (
    TradingSystem,
    RiskParameters,
    MACrossoverSignal,
    MomentumSignal,
    RSIMeanReversionSignal,
    EnhancedRSIMeanReversionSignal,
    RefinedTickerSpecificSignal
)


def run_multi_ticker_backtest(tickers=["SQQQ", "TQQQ", "UNH"]):
    """
    Run backtest on specified tickers with enhanced risk management.
    """
    # Configuration
    start_date = "2020-01-01"
    end_date = datetime.now().strftime("%Y-%m-%d")
    initial_capital = 100000.0

    # Optimized Risk Management Parameters
    risk_params = RiskParameters(
        max_position_size=0.80,        # Max 80% of capital per position
        stop_loss_pct=0.15,            # 15% stop loss
        take_profit_pct=0.40,          # 40% take profit
        max_drawdown_limit=0.30,       # 30% max drawdown limit
        volatility_lookback=20,        # 20-day volatility lookback
        volatility_target=None,        # Disable volatility targeting for direct exposure
        max_leverage=1.0,              # No additional leverage
        correlation_threshold=0.7
    )

    print("="*60)
    print("SYSTEMATIC TRADING SYSTEM - MULTI-TICKER BACKTEST")
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
    
    # Define strategies to run
    strategies = [
        ("MA Crossover", MACrossoverSignal(short_window=10, long_window=50)),
        ("Momentum", MomentumSignal(momentum_window=10, volatility_filter=True)),
        ("RSI (30/70)", RSIMeanReversionSignal(oversold=30, overbought=70)),
        ("Enhanced RSI", EnhancedRSIMeanReversionSignal(oversold=40, overbought=60, hold_days=7)),
        ("Ticker-Specific", RefinedTickerSpecificSignal())
    ]

    all_results = {}
    summary_data = []

    for strat_name, signal_gen in strategies:
        print("\n" + "="*60)
        print(f"STRATEGY: {strat_name}")
        print("="*60)
        
        system = TradingSystem(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            risk_params=risk_params,
            signal_generator=signal_gen
        )
        
        performance = system.run()
        all_results[strat_name] = system
        
        for ticker in tickers:
            if ticker in performance:
                perf = performance[ticker]
                summary_data.append({
                    'Ticker': ticker,
                    'Strategy': strat_name,
                    'Total Return': perf.get('total_return', 0),
                    'Sharpe Ratio': perf.get('sharpe_ratio', 0),
                    'Max Drawdown': perf.get('max_drawdown', 0),
                    'Win Rate': perf.get('win_rate', 0)
                })

    # Create comparison summary
    summary_df = pd.DataFrame(summary_data)
    print("\n" + "="*60)
    print("STRATEGY COMPARISON SUMMARY")
    print("="*60)
    print("\n")
    print(summary_df.to_string(index=False, formatters={
        'Total Return': '{:.2%}'.format,
        'Sharpe Ratio': '{:.2f}'.format,
        'Max Drawdown': '{:.2%}'.format,
        'Win Rate': '{:.2%}'.format
    }))

    # Save summary to CSV
    summary_df.to_csv('backtest_results.csv', index=False)
    print("\nResults saved to backtest_results.csv")

    # Create visualization
    create_visualizations(all_results, tickers)

    return all_results, summary_df


def create_visualizations(all_results, tickers):
    """
    Create performance visualization charts for all strategies and tickers.
    """
    num_tickers = len(tickers)
    fig, axes = plt.subplots(3, num_tickers, figsize=(6 * num_tickers, 15))
    fig.suptitle('Systematic Trading Backtest Results', fontsize=16, fontweight='bold')

    # Handle single ticker case for axes indexing
    if num_tickers == 1:
        axes = axes.reshape(3, 1)

    for idx, ticker in enumerate(tickers):
        # 1. Equity Curves
        ax1 = axes[0, idx]
        for strat_name, system in all_results.items():
            if ticker in system.results:
                results = system.results[ticker]
                ax1.plot(results.index, results['total'], label=strat_name, alpha=0.8)
        
        ax1.set_title(f'{ticker} - Portfolio Value')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend(loc='upper left', fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=100000, color='gray', linestyle='--', alpha=0.5)

        # 2. Drawdowns
        ax2 = axes[1, idx]
        for strat_name, system in all_results.items():
            if ticker in system.results:
                results = system.results[ticker]
                ax2.plot(results.index, results['drawdown'] * 100, label=strat_name, alpha=0.6)
        
        ax2.set_title(f'{ticker} - Drawdown (%)')
        ax2.set_ylabel('Drawdown (%)')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.2)

        # 3. Rolling Sharpe Ratio (252-day)
        ax3 = axes[2, idx]
        for strat_name, system in all_results.items():
            if ticker in system.results:
                results = system.results[ticker]
                if len(results) > 252:
                    rolling_sharpe = (results['returns'].rolling(252).mean() / 
                                    results['returns'].rolling(252).std()) * np.sqrt(252)
                    ax3.plot(results.index, rolling_sharpe, label=strat_name, alpha=0.8)
        
        ax3.set_title(f'{ticker} - Rolling Sharpe (252d)')
        ax3.set_ylabel('Sharpe Ratio')
        ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax3.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('backtest_results.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved to backtest_results.png")


if __name__ == "__main__":
    # Tickers: SQQQ, TQQQ, UNH, META (Meta Platforms), NVO (Novo Nordisk)
    target_tickers = ["SQQQ", "TQQQ", "UNH", "META", "NVO"]
    run_multi_ticker_backtest(target_tickers)
