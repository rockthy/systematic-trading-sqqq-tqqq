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
from datetime import datetime, timedelta
import yfinance as yf


def main():
    start_date = "2020-01-01"
    # Use inclusive end date for yfinance (end is exclusive), so add one day
    end_date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    initial_capital = 100000.0

    risk_params = RiskParameters(volatility_target=None)

    tickers = ["SQQQ", "TQQQ", "UNH", "META", "NVO", "MNT.TO"]

    strategies = [
        ("MA Crossover", MACrossoverSignal(short_window=10, long_window=50)),
        ("Momentum", MomentumSignal(momentum_window=10, volatility_filter=True)),
        ("RSI (30/70)", RSIMeanReversionSignal(oversold=30, overbought=70)),
        ("Enhanced RSI", EnhancedRSIMeanReversionSignal(oversold=40, overbought=60, hold_days=7)),
        ("Ticker-Specific", RefinedTickerSpecificSignal())
    ]

    mnt_trades = []

    # Quick check: report latest available raw data date for MNT.TO
    try:
        raw = yf.download('MNT.TO', start=start_date, end=end_date, progress=False)
        if not raw.empty:
            last_date = raw.index[-1].strftime('%Y-%m-%d')
            print(f"Latest available raw data for MNT.TO: {last_date}")
        else:
            print("No raw data returned for MNT.TO from yfinance.")
    except Exception as e:
        print(f"Failed to fetch raw data for MNT.TO: {e}")

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

        results = system.results.get('MNT.TO')
        if results is None or results.empty:
            continue

        trades = results[results['pos_diff'] != 0]
        for idx, row in trades.iterrows():
            action = 'Buy' if row['pos_diff'] > 0 else 'Sell'
            mnt_trades.append({
                'date': idx,
                'strategy': strat_name,
                'action': action,
                'shares_change': int(row['pos_diff']),
                'price': float(row['price'])
            })

    if not mnt_trades:
        print('No trades executed for MNT.TO')
        return

    latest = max(mnt_trades, key=lambda x: x['date'])
    print('LATEST_MNT_TRADE')
    print(f"Date: {latest['date'].strftime('%Y-%m-%d')}")
    print(f"Strategy: {latest['strategy']}")
    print(f"Action: {latest['action']}")
    print(f"Shares change: {latest['shares_change']}")
    print(f"Price: {latest['price']:.2f}")


if __name__ == '__main__':
    main()
