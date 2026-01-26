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
    RSIMeanReversionSignal,
    EnhancedRSIMeanReversionSignal,
    RefinedTickerSpecificSignal
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

    # Optimized Risk Management Parameters
    risk_params = RiskParameters(
        max_position_size=0.80,        # Increase to 80% to actually use your capital
        stop_loss_pct=0.20,            # Loosen to 20% (3x ETFs need room for noise)
        take_profit_pct=0.50,          # Let winners run further
        max_drawdown_limit=0.40,       # Accept higher drawdown for higher reward
        volatility_lookback=20,        # 20-day volatility lookback
        volatility_target=None,        # Remove the vol-target constraint for now
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
    volatility_str = f"{risk_params.volatility_target:.0%}" if risk_params.volatility_target is not None else "Disabled"
    print(f"  Volatility Target: {volatility_str}")

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

    # Strategy 4: Enhanced RSI Mean Reversion (Improved)
    print("\n" + "="*60)
    print("STRATEGY 4: Enhanced RSI Mean Reversion (40/60 + MACD + Time-based)")
    print("="*60)

    enhanced_rsi_signal = EnhancedRSIMeanReversionSignal(
        oversold=40,
        overbought=60,
        hold_days=7,
        use_macd_confirmation=True
    )
    system_enhanced_rsi = TradingSystem(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        risk_params=risk_params,
        signal_generator=enhanced_rsi_signal
    )

    performance_enhanced_rsi = system_enhanced_rsi.run()

    # Strategy 5: Refined Ticker-Specific (TQQQ uptrend dips + SQQQ crash scalps)
    print("\n" + "="*60)
    print("STRATEGY 5: Refined Ticker-Specific (TQQQ: Uptrend Dips + SQQQ: Crash Scalps)")
    print("="*60)

    refined_signal = RefinedTickerSpecificSignal(
        tqqq_rsi_threshold=35,
        sqqq_rsi_threshold=30,
        scalp_hold_hours=48,
        normal_hold_days=10
    )
    system_refined = TradingSystem(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        risk_params=risk_params,
        signal_generator=refined_signal
    )

    performance_refined = system_refined.run()

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
        summary_data.append({
            'Ticker': ticker,
            'Strategy': 'Enhanced RSI Mean Reversion',
            'Total Return': performance_enhanced_rsi.get(ticker, {}).get('total_return', 0),
            'Sharpe Ratio': performance_enhanced_rsi.get(ticker, {}).get('sharpe_ratio', 0),
            'Max Drawdown': performance_enhanced_rsi.get(ticker, {}).get('max_drawdown', 0),
            'Win Rate': performance_enhanced_rsi.get(ticker, {}).get('win_rate', 0)
        })
        summary_data.append({
            'Ticker': ticker,
            'Strategy': 'Refined Ticker-Specific',
            'Total Return': performance_refined.get(ticker, {}).get('total_return', 0),
            'Sharpe Ratio': performance_refined.get(ticker, {}).get('sharpe_ratio', 0),
            'Max Drawdown': performance_refined.get(ticker, {}).get('max_drawdown', 0),
            'Win Rate': performance_refined.get(ticker, {}).get('win_rate', 0)
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
    summary_df.to_csv('backtest_results.csv', index=False)
    print("\nResults saved to backtest_results.csv")

    # Create visualization
    create_visualizations(system_ma, system_momentum, system_rsi, system_enhanced_rsi, system_refined, tickers)

    return {
        'ma_crossover': performance_ma,
        'momentum': performance_momentum,
        'rsi_mean_reversion': performance_rsi,
        'enhanced_rsi_mean_reversion': performance_enhanced_rsi,
        'refined_ticker_specific': performance_refined,
        'summary': summary_df
    }


def create_visualizations(system_ma, system_momentum, system_rsi, system_enhanced_rsi, system_refined, tickers):
    """
    Create performance visualization charts for all strategies.
    """
    fig, axes = plt.subplots(4, 2, figsize=(16, 16))
    fig.suptitle('SQQQ/TQQQ Systematic Trading Backtest Results', fontsize=14, fontweight='bold')

    colors = {'SQQQ': '#e74c3c', 'TQQQ': '#27ae60'}

    for idx, ticker in enumerate(tickers):
        # Equity curves for all strategies
        ax1 = axes[0, idx]
        if ticker in system_ma.results:
            results = system_ma.results[ticker]
            ax1.plot(results.index, results['total'], label='MA Crossover', color='blue', alpha=0.8)
        if ticker in system_momentum.results:
            results = system_momentum.results[ticker]
            ax1.plot(results.index, results['total'], label='Momentum', color='orange', alpha=0.8)
        if ticker in system_rsi.results:
            results = system_rsi.results[ticker]
            ax1.plot(results.index, results['total'], label='RSI (30/70)', color='green', alpha=0.8)
        if ticker in system_enhanced_rsi.results:
            results = system_enhanced_rsi.results[ticker]
            ax1.plot(results.index, results['total'], label='Enhanced RSI (40/60)', color='red', alpha=0.8, linewidth=2)
        if ticker in system_refined.results:
            results = system_refined.results[ticker]
            ax1.plot(results.index, results['total'], label='Refined Ticker-Specific', color='purple', alpha=0.8, linewidth=2)

        ax1.set_title(f'{ticker} - Portfolio Value')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend(loc='upper left', fontsize=8)
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
                           label='RSI (30/70)', color='green', alpha=0.3)
        if ticker in system_enhanced_rsi.results:
            results = system_enhanced_rsi.results[ticker]
            ax2.fill_between(results.index, results['drawdown'] * 100, 0,
                           label='Enhanced RSI (40/60)', color='red', alpha=0.3)
        if ticker in system_refined.results:
            results = system_refined.results[ticker]
            ax2.fill_between(results.index, results['drawdown'] * 100, 0,
                           label='Refined Ticker-Specific', color='purple', alpha=0.3)

        ax2.set_title(f'{ticker} - Drawdown')
        ax2.set_ylabel('Drawdown (%)')
        ax2.legend(loc='lower left', fontsize=8)
        ax2.grid(True, alpha=0.3)

        # Rolling Sharpe Ratio (252-day) - Row 3
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
            ax3.plot(results.index, rolling_sharpe, label='RSI (30/70)', color='green', alpha=0.8)
        if ticker in system_enhanced_rsi.results:
            results = system_enhanced_rsi.results[ticker]
            rolling_sharpe = (results['returns'].rolling(252).mean() /
                            results['returns'].rolling(252).std()) * np.sqrt(252)
            ax3.plot(results.index, rolling_sharpe, label='Enhanced RSI (40/60)', color='red', alpha=0.8, linewidth=2)
        if ticker in system_refined.results:
            results = system_refined.results[ticker]
            rolling_sharpe = (results['returns'].rolling(252).mean() /
                            results['returns'].rolling(252).std()) * np.sqrt(252)
            ax3.plot(results.index, rolling_sharpe, label='Refined Ticker-Specific', color='purple', alpha=0.8, linewidth=2)

        ax3.set_title(f'{ticker} - Rolling Sharpe Ratio (252-day)')
        ax3.set_ylabel('Sharpe Ratio')
        ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax3.legend(loc='upper left', fontsize=8)
        ax3.grid(True, alpha=0.3)

        # All strategies cumulative returns - Row 4
        ax4 = axes[3, idx]
        if ticker in system_ma.results:
            results = system_ma.results[ticker]
            cumulative_returns = (1 + results['returns']).cumprod() - 1
            ax4.plot(results.index, cumulative_returns * 100, label='MA Crossover', color='blue', alpha=0.8)
        if ticker in system_momentum.results:
            results = system_momentum.results[ticker]
            cumulative_returns = (1 + results['returns']).cumprod() - 1
            ax4.plot(results.index, cumulative_returns * 100, label='Momentum', color='orange', alpha=0.8)
        if ticker in system_rsi.results:
            results = system_rsi.results[ticker]
            cumulative_returns = (1 + results['returns']).cumprod() - 1
            ax4.plot(results.index, cumulative_returns * 100, label='RSI (30/70)', color='green', alpha=0.8)
        if ticker in system_enhanced_rsi.results:
            results = system_enhanced_rsi.results[ticker]
            cumulative_returns = (1 + results['returns']).cumprod() - 1
            ax4.plot(results.index, cumulative_returns * 100, label='Enhanced RSI (40/60)', color='red', alpha=0.8, linewidth=2)
        if ticker in system_refined.results:
            results = system_refined.results[ticker]
            cumulative_returns = (1 + results['returns']).cumprod() - 1
            ax4.plot(results.index, cumulative_returns * 100, label='Refined Ticker-Specific', color='purple', alpha=0.8, linewidth=2)

        ax4.set_title(f'{ticker} - Cumulative Returns')
        ax4.set_ylabel('Cumulative Return (%)')
        ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax4.legend(loc='upper left', fontsize=8)
        ax4.grid(True, alpha=0.3)
        ax4.set_title(f'{ticker} - RSI Strategies: Cumulative Returns')
        ax4.set_ylabel('Cumulative Return (%)')
        ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax4.legend(loc='upper left')
        ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('backtest_results.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved to backtest_results.png")


# Need to import numpy for visualization
import numpy as np

if __name__ == "__main__":
    results = run_sqqq_tqqq_backtest()
