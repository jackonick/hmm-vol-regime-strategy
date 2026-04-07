from sklearn.ensemble import RandomForestClassifier
from hmmlearn.hmm import GaussianHMM
from polygon import RESTClient
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np

import os
client = RESTClient(api_key=os.environ.get("POLYGON_API_KEY"))

#download stock data
def datadownload (ticker):
    aggs = []
    for a in client.list_aggs(
        ticker = ticker,
        multiplier=1,
        timespan="day",
        from_="2023-01-01",
        to="2026-04-06",
        limit=50000
    ):
        aggs.append(a)

    df = pd.DataFrame([{
        "timestamp": a.timestamp,
        "open": a.open,
        "high": a.high,
        "low": a.low,
        "close": a.close,
        "volume": a.volume
    } for a in aggs])

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.set_index("timestamp")
    
    
    return df


#import vix data
VIX = yf.download("^VIX")
close = VIX["Close"]["^VIX"]
daily_returns = close.pct_change()
rolling_vol = daily_returns.rolling(window=20).std()
spydat = datadownload("SPY")

HMMDF = pd.DataFrame({"close": close, "daily returns": daily_returns, "rolling vol": rolling_vol}).dropna()

months = HMMDF.index.to_period("M").unique()
months = months[months >= "2023-01"]

signals = {}

for month in months:
    month_start = month.to_timestamp()
    train_data = HMMDF.loc[:month_start].iloc[:-1]  # everything before this month
    current_month = HMMDF.loc[HMMDF.index.to_period("M") == month]  # this month's data
    
    model = GaussianHMM(n_components=3)
    model.fit(train_data[["close", "daily returns", "rolling vol"]].to_numpy())
    meanvix = model.means_[:, 0]
    order = np.argsort(meanvix) 
    regime_map = {order[0]: 1, order[1]: 0.5, order[2]: 0}
    regime = model.predict(train_data.iloc[[-1]][["close", "daily returns", "rolling vol"]].to_numpy())
    
    monthly_signal = regime_map[regime[0]]
    for date in current_month.index:
        signals[date] = monthly_signal
    
    
signal = pd.Series(signals).shift(1)
spydat.index = spydat.index.normalize()
spyret = spydat["close"].pct_change()
strategy_returns = signal * spyret - 0.001 * (signal.diff() != 0)
strategy_returns = strategy_returns.loc["2023-01-01":]
strategy_returns = strategy_returns.dropna()


#calculators
cumulative_returns = (1 + strategy_returns).cumprod()
total_return = (cumulative_returns.iloc[-1] - 1) * 100
sharpe = (strategy_returns.mean() / strategy_returns.std()) * np.sqrt(252)
running_max = cumulative_returns.cummax()
drawdown = (cumulative_returns - running_max) / running_max
max_drawdown = drawdown.min() * 100
rolling_sharpe = (strategy_returns.rolling(window=50).mean() / strategy_returns.rolling(window=50).std()) * np.sqrt(252)
spy_cumu_ret = (1 + spyret.loc["2023-01-01":]).cumprod()

#matplot
fig, axes = plt.subplots(3)
axes[0].plot(cumulative_returns, label="cumulative_returns")
axes[0].plot(spy_cumu_ret, label="SPY returns")
axes[1].plot(drawdown, label="drawdown")
axes[2].plot(rolling_sharpe, label="rolling_sharpe")

axes[0].set_title("strategy returns")
axes[1].set_title("drawdown")
axes[2].set_title("rolling sharpe ratio")

axes[0].legend()
axes[1].legend()
axes[2].legend()

fig.tight_layout()
plt.savefig("hmm_backtest.png")

print(f"Total Return: {total_return:.2f}% | Sharpe: {sharpe:.2f} | Max Drawdown: {max_drawdown:.2f}%")