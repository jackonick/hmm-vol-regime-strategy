# HMM Volatility Regime Strategy

Uses a Hidden Markov Model to detect VIX volatility regimes and adjust SPY exposure monthly. Prioritizes drawdown control over raw returns.

## Results (2023-2026)
- Total Return: 22.81%
- Sharpe Ratio: 1.26
- Max Drawdown: -5.24%

## Setup

1. Install dependencies:
```
pip install -r requirements.txt
```

2. Set your Polygon.io API key:

Windows (PowerShell):
```
$env:POLYGON_API_KEY = "your_key_here"
```

Mac/Linux:
```
export POLYGON_API_KEY="your_key_here"
```

3. Run:
```
python stockstrat.py
```

## Output

- Prints total return, Sharpe ratio, and max drawdown
- Saves a 3-panel chart (`hmm_backtest.png`) showing equity curve vs SPY, drawdowns, and rolling Sharpe ratio
