import sys
import os

# Ensure project root is on sys.path when running from the scripts/ folder
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core import (
    TradingSystem,
    RiskParameters,
    MACrossoverSignal,
    MomentumSignal,
    RSIMeanReversionSignal,
    EnhancedRSIMeanReversionSignal,
    RefinedTickerSpecificSignal,
)
from datetime import datetime


def main():
    start_date = "2020-01-01"
    end_date = datetime.now().strftime("%Y-%m-%d")
    initial_capital = 100000.0

    risk_params = RiskParameters(
        max_position_size=0.80,
        stop_loss_pct=0.15,
        take_profit_pct=0.40,
        max_drawdown_limit=0.30,
        volatility_lookback=20,
        volatility_target=None,
        max_leverage=1.0,
        correlation_threshold=0.7
    )

    tickers = ["SQQQ", "TQQQ", "UNH", "META", "NVO", "MNT.TO"]

    strategies = [
        ("MA Crossover", MACrossoverSignal(short_window=10, long_window=50)),
        ("Momentum", MomentumSignal(momentum_window=10, volatility_filter=True)),
        ("RSI (30/70)", RSIMeanReversionSignal(oversold=30, overbought=70)),
        ("Enhanced RSI", EnhancedRSIMeanReversionSignal(oversold=40, overbought=60, hold_days=7)),
        ("Ticker-Specific", RefinedTickerSpecificSignal())
    ]

    all_trades = []

    for strat_name, signal_gen in strategies:
        system = TradingSystem(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            risk_params=risk_params,
            signal_generator=signal_gen
        )

        _ = system.run()

        for ticker, results in system.results.items():
            if results is None or results.empty:
                continue

            trades = results[results['pos_diff'] != 0]
            for idx, row in trades.iterrows():
                action = 'Buy' if row['pos_diff'] > 0 else 'Sell'
                all_trades.append({
                    'date': idx,
                    'strategy': strat_name,
                    'ticker': ticker,
                    'action': action,
                    'shares_change': int(row['pos_diff']),
                    'price': float(row['price'])
                })

    if not all_trades:
        print('No trades executed in backtests')
        return

    # Find the latest trade by date
    latest = max(all_trades, key=lambda x: x['date'])
    print('LATEST_TRADE')
    print(f"Date: {latest['date'].strftime('%Y-%m-%d')}")
    print(f"Strategy: {latest['strategy']}")
    print(f"Ticker: {latest['ticker']}")
    print(f"Action: {latest['action']}")
    print(f"Shares change: {latest['shares_change']}")
    print(f"Price: {latest['price']:.2f}")


if __name__ == '__main__':
    main()
