"""Backtest the earnings alpha signal on the held-out test set.

Simulates a long/short strategy based on model predictions and computes
cumulative returns, Sharpe ratio, win rate, and generates performance plots.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PLOTS_DIR = PROJECT_ROOT / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def run_backtest(target="target_5d", return_col="return_5d"):
    """Backtest: go long when model predicts Up, go short when Down."""
    pred_path = PROJECT_ROOT / "data" / "processed" / f"test_predictions_{target}.csv"
    df = pd.read_csv(pred_path)
    df["earnings_date"] = pd.to_datetime(df["earnings_date"])

    print(f"Backtesting {target} signal on {len(df)} test trades...")

    # Strategy return: long if prediction=1 (Up), short if prediction=0 (Down)
    df["strategy_return"] = np.where(
        df["prediction"] == 1,
        df[return_col],       # long: earn the return
        -df[return_col],      # short: earn the inverse
    )

    # Cumulative returns
    df["cum_strategy"] = (1 + df["strategy_return"]).cumprod()
    df["cum_buyhold"] = (1 + df[return_col]).cumprod()

    # Metrics
    total_return = df["cum_strategy"].iloc[-1] - 1
    buyhold_return = df["cum_buyhold"].iloc[-1] - 1
    n_trades = len(df)
    wins = (df["strategy_return"] > 0).sum()
    win_rate = wins / n_trades

    # Annualized Sharpe (assume ~4 earnings per year per stock)
    mean_ret = df["strategy_return"].mean()
    std_ret = df["strategy_return"].std()
    sharpe = (mean_ret / std_ret) * np.sqrt(4 * 10) if std_ret > 0 else 0  # ~40 trades/year across 10 tickers

    print(f"\n{'=' * 50}")
    print(f"  BACKTEST RESULTS — {target}")
    print(f"{'=' * 50}")
    print(f"  Total trades:       {n_trades}")
    print(f"  Win rate:           {win_rate:.1%}")
    print(f"  Strategy return:    {total_return:+.1%}")
    print(f"  Buy & hold return:  {buyhold_return:+.1%}")
    print(f"  Excess return:      {total_return - buyhold_return:+.1%}")
    print(f"  Sharpe ratio:       {sharpe:.2f}")
    print(f"  Mean trade return:  {mean_ret:+.2%}")
    print(f"  Std trade return:   {std_ret:.2%}")
    print(f"  Best trade:         {df['strategy_return'].max():+.2%}")
    print(f"  Worst trade:        {df['strategy_return'].min():+.2%}")
    print(f"{'=' * 50}\n")

    # --- Plot 1: Cumulative Returns ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Earnings Alpha Engine — Backtest ({target})", fontsize=14, fontweight="bold")

    ax = axes[0, 0]
    ax.plot(range(len(df)), df["cum_strategy"], "b-o", markersize=4, label="Strategy")
    ax.plot(range(len(df)), df["cum_buyhold"], "r--o", markersize=4, label="Buy & Hold", alpha=0.7)
    ax.axhline(1, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Trade #")
    ax.set_ylabel("Cumulative Return")
    ax.set_title("Cumulative Returns")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Plot 2: Per-trade returns ---
    ax = axes[0, 1]
    colors = ["green" if r > 0 else "red" for r in df["strategy_return"]]
    ax.bar(range(len(df)), df["strategy_return"] * 100, color=colors, alpha=0.7)
    ax.axhline(0, color="gray", linestyle="-", alpha=0.5)
    ax.set_xlabel("Trade #")
    ax.set_ylabel("Return (%)")
    ax.set_title(f"Per-Trade Returns (Win Rate: {win_rate:.0%})")
    ax.grid(True, alpha=0.3)

    # --- Plot 3: Prediction confidence vs actual return ---
    ax = axes[1, 0]
    scatter = ax.scatter(
        df["pred_prob"],
        df[return_col] * 100,
        c=df["prediction"],
        cmap="RdYlGn",
        edgecolors="black",
        linewidth=0.5,
        alpha=0.8,
        s=60,
    )
    ax.axhline(0, color="gray", linestyle=":", alpha=0.5)
    ax.axvline(0.5, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Predicted Probability (Up)")
    ax.set_ylabel(f"Actual {return_col} (%)")
    ax.set_title("Confidence vs Actual Return")
    ax.grid(True, alpha=0.3)

    # --- Plot 4: Return by ticker ---
    ax = axes[1, 1]
    ticker_returns = df.groupby("ticker")["strategy_return"].sum() * 100
    colors_ticker = ["green" if r > 0 else "red" for r in ticker_returns]
    ticker_returns.plot(kind="bar", ax=ax, color=colors_ticker, alpha=0.7)
    ax.axhline(0, color="gray", linestyle="-", alpha=0.5)
    ax.set_ylabel("Total Return (%)")
    ax.set_title("Return by Ticker")
    ax.grid(True, alpha=0.3)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

    plt.tight_layout()
    plot_path = PLOTS_DIR / f"backtest_{target}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {plot_path.relative_to(PROJECT_ROOT)}")

    return {
        "total_return": float(total_return),
        "buyhold_return": float(buyhold_return),
        "excess_return": float(total_return - buyhold_return),
        "sharpe": float(sharpe),
        "win_rate": float(win_rate),
        "n_trades": n_trades,
    }


if __name__ == "__main__":
    run_backtest("target_5d", "return_5d")
    print()
    run_backtest("target_1d", "return_1d")
