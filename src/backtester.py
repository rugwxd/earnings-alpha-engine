"""Backtest the earnings alpha signal on walk-forward out-of-sample predictions.

Simulates a long/short strategy, computes Sharpe ratio, win rate,
cumulative returns, and bootstrap confidence intervals.
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

RISK_FREE_RATE = 0.04   # approximate annualized risk-free rate (T-bills)
TRADES_PER_YEAR = 40     # ~4 earnings/year x 10 tickers


def _bootstrap_ci(returns, n_boot=5000, ci=0.95):
    """Bootstrap confidence interval for mean return."""
    rng = np.random.RandomState(42)
    means = []
    for _ in range(n_boot):
        sample = rng.choice(returns, size=len(returns), replace=True)
        means.append(np.mean(sample))
    means = np.sort(means)
    lo = means[int((1 - ci) / 2 * n_boot)]
    hi = means[int((1 + ci) / 2 * n_boot)]
    return float(lo), float(hi)


def run_backtest(target="target_5d", return_col="return_5d"):
    """Backtest: go long when model predicts Up, go short when Down."""
    pred_path = PROJECT_ROOT / "data" / "processed" / f"oos_predictions_{target}.csv"
    df = pd.read_csv(pred_path)
    df["earnings_date"] = pd.to_datetime(df["earnings_date"])
    df = df.sort_values("earnings_date").reset_index(drop=True)

    print(f"Backtesting {target} signal on {len(df)} OOS trades...")

    # Strategy return: long if prediction=1, short if prediction=0
    df["strategy_return"] = np.where(
        df["prediction"] == 1,
        df[return_col],
        -df[return_col],
    )

    # Cumulative returns
    df["cum_strategy"] = (1 + df["strategy_return"]).cumprod()
    df["cum_buyhold"] = (1 + df[return_col]).cumprod()

    # Core metrics
    total_return = df["cum_strategy"].iloc[-1] - 1
    buyhold_return = df["cum_buyhold"].iloc[-1] - 1
    n_trades = len(df)
    wins = (df["strategy_return"] > 0).sum()
    win_rate = wins / n_trades
    mean_ret = df["strategy_return"].mean()
    std_ret = df["strategy_return"].std()

    # Annualized Sharpe ratio (proper calculation)
    # Per-trade excess return over risk-free rate per trade
    rf_per_trade = RISK_FREE_RATE / TRADES_PER_YEAR
    excess_mean = mean_ret - rf_per_trade
    sharpe = (excess_mean / std_ret) * np.sqrt(TRADES_PER_YEAR) if std_ret > 0 else 0

    # Bootstrap 95% CI on mean trade return
    ci_lo, ci_hi = _bootstrap_ci(df["strategy_return"].values)

    # Max drawdown
    cum = df["cum_strategy"].values
    running_max = np.maximum.accumulate(cum)
    drawdowns = (cum - running_max) / running_max
    max_drawdown = drawdowns.min()

    print(f"\n{'=' * 55}")
    print(f"  WALK-FORWARD BACKTEST — {target}")
    print(f"{'=' * 55}")
    print(f"  Total OOS trades:     {n_trades}")
    print(f"  Win rate:             {win_rate:.1%}")
    print(f"  Strategy return:      {total_return:+.1%}")
    print(f"  Buy & hold return:    {buyhold_return:+.1%}")
    print(f"  Excess return:        {total_return - buyhold_return:+.1%}")
    print(f"  Annualized Sharpe:    {sharpe:.2f}")
    print(f"  Max drawdown:         {max_drawdown:.1%}")
    print(f"  Mean trade return:    {mean_ret:+.2%}")
    print(f"  Std trade return:     {std_ret:.2%}")
    print(f"  95% CI (mean return): [{ci_lo:+.2%}, {ci_hi:+.2%}]")
    print(f"  Best trade:           {df['strategy_return'].max():+.2%}")
    print(f"  Worst trade:          {df['strategy_return'].min():+.2%}")

    # Statistical significance: is 0 inside the CI?
    sig = "YES" if (ci_lo > 0 or ci_hi < 0) else "NO"
    print(f"  Statistically significant (95%): {sig}")
    print(f"{'=' * 55}\n")

    # ---- Plots ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Earnings Alpha Engine — Walk-Forward Backtest ({target})",
                 fontsize=14, fontweight="bold")

    # Plot 1: Cumulative Returns
    ax = axes[0, 0]
    ax.plot(df["earnings_date"], df["cum_strategy"], "b-o", markersize=4, label="Strategy")
    ax.plot(df["earnings_date"], df["cum_buyhold"], "r--o", markersize=4, label="Buy & Hold", alpha=0.7)
    ax.axhline(1, color="gray", linestyle=":", alpha=0.5)
    ax.set_ylabel("Cumulative Return")
    ax.set_title("Cumulative Returns (OOS)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="x", rotation=45)

    # Plot 2: Per-trade returns
    ax = axes[0, 1]
    colors = ["green" if r > 0 else "red" for r in df["strategy_return"]]
    ax.bar(range(len(df)), df["strategy_return"] * 100, color=colors, alpha=0.7)
    ax.axhline(0, color="gray", linestyle="-", alpha=0.5)
    ax.set_xlabel("Trade #")
    ax.set_ylabel("Return (%)")
    ax.set_title(f"Per-Trade Returns (Win Rate: {win_rate:.0%})")
    ax.grid(True, alpha=0.3)

    # Plot 3: Prediction confidence vs actual return
    ax = axes[1, 0]
    ax.scatter(
        df["pred_prob"], df[return_col] * 100,
        c=df["prediction"], cmap="RdYlGn",
        edgecolors="black", linewidth=0.5, alpha=0.8, s=60,
    )
    ax.axhline(0, color="gray", linestyle=":", alpha=0.5)
    ax.axvline(0.5, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Predicted Probability (Up)")
    ax.set_ylabel(f"Actual {return_col} (%)")
    ax.set_title("Confidence vs Actual Return")
    ax.grid(True, alpha=0.3)

    # Plot 4: Drawdown
    ax = axes[1, 1]
    ax.fill_between(range(len(drawdowns)), drawdowns * 100, 0, color="red", alpha=0.3)
    ax.plot(range(len(drawdowns)), drawdowns * 100, "r-", linewidth=1)
    ax.set_xlabel("Trade #")
    ax.set_ylabel("Drawdown (%)")
    ax.set_title(f"Drawdown (Max: {max_drawdown:.1%})")
    ax.grid(True, alpha=0.3)

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
        "max_drawdown": float(max_drawdown),
        "n_trades": n_trades,
        "ci_95_lo": float(ci_lo),
        "ci_95_hi": float(ci_hi),
        "statistically_significant": sig == "YES",
    }


if __name__ == "__main__":
    run_backtest("target_5d", "return_5d")
    print()
    run_backtest("target_1d", "return_1d")
