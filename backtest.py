"""
backtest.py — Vectorized backtest engine for crypto stat-arb pairs.
Computes PnL, Sharpe, drawdown, trade statistics, and alpha t-stat vs benchmark.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BacktestConfig:
    capital: float = 100_000
    transaction_cost_bps: float = 20.0  # 10bps per leg — crypto market standard
    slippage_bps: float = 3.0           # 1.5bps per leg (conservative)
    max_drawdown_limit: float = 0.25    # halt at 25% drawdown


@dataclass
class BacktestResult:
    pair: tuple
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    num_trades: int
    win_rate: float
    avg_trade_return: float
    calmar_ratio: float
    equity_curve: pd.Series
    daily_pnl: pd.Series
    signals_df: pd.DataFrame
    # Alpha t-stat vs benchmark (None if no benchmark provided)
    alpha_tstat: Optional[float] = None
    alpha_pvalue: Optional[float] = None
    beta_to_benchmark: Optional[float] = None


def compute_alpha_tstat(
    strategy_returns: pd.Series,
    benchmark_prices: pd.Series,
) -> tuple:
    """
    Regress strategy bar returns against benchmark bar returns.
    Returns (alpha_tstat, alpha_pvalue, beta).

    The alpha t-stat is the t-statistic on the intercept of:
        r_strategy = alpha + beta * r_benchmark + epsilon

    A high |alpha_tstat| (e.g. > 2) with positive alpha indicates the strategy
    is generating returns not explained by benchmark exposure — i.e. genuine alpha.
    This is the standard way to assess a strategy alongside its benchmark correlation.

    Args:
        strategy_returns: Bar-level PnL / capital (fractional returns series)
        benchmark_prices: Raw price series for the benchmark asset (e.g. BTC close prices)
    """
    bm_ret = benchmark_prices.pct_change().dropna()
    common = strategy_returns.index.intersection(bm_ret.index)
    if len(common) < 30:
        return None, None, None

    y = strategy_returns.loc[common].fillna(0)
    x = bm_ret.loc[common].fillna(0)

    X = sm.add_constant(x)
    try:
        model = sm.OLS(y, X).fit()
        alpha_tstat = float(model.tvalues.iloc[0])
        alpha_pvalue = float(model.pvalues.iloc[0])
        beta = float(model.params.iloc[1]) if len(model.params) > 1 else None
        return alpha_tstat, alpha_pvalue, beta
    except Exception:
        return None, None, None


def _apply_kill_switch(daily_pnl: pd.Series, config: BacktestConfig):
    equity = config.capital + daily_pnl.cumsum()
    roll_max = equity.cummax()
    drawdown = (equity - roll_max) / roll_max
    dd_breach = drawdown < -config.max_drawdown_limit
    if dd_breach.any():
        first_breach = dd_breach.idxmax()
        daily_pnl = daily_pnl.copy()
        daily_pnl.loc[first_breach:] = 0.0
    return daily_pnl


def run_backtest(
    price_a: pd.Series,
    price_b: pd.Series,
    signals_df: pd.DataFrame,
    sizing_df: pd.DataFrame,
    hedge_ratio: float,
    asset_a: str,
    asset_b: str,
    config: BacktestConfig = None,
    benchmark_prices: Optional[pd.Series] = None,
) -> BacktestResult:
    """
    Vectorized backtest. PnL computed on actual price returns, not spread.

    Long spread: long A at weight w_a, short B at weight w_b
    Daily PnL = w_a * ret_a + w_b * ret_b - transaction costs on trade days

    Args:
        benchmark_prices: Optional price series (e.g. BTC close) used to compute the
                          alpha t-stat. Pass prices["BTC"] from the caller.
    """
    if config is None:
        config = BacktestConfig()

    common = price_a.index.intersection(price_b.index).intersection(signals_df.index)
    pa = price_a.loc[common]
    pb = price_b.loc[common]
    pos = signals_df.loc[common, "position"]
    trade_sig = signals_df.loc[common, "trade_signal"]
    pos_a_usd = sizing_df.loc[common, "pos_a_usd"].fillna(0)
    pos_b_usd = sizing_df.loc[common, "pos_b_usd"].fillna(0)

    ret_a = pa.pct_change().fillna(0)
    ret_b = pb.pct_change().fillna(0)

    pnl_a = pos_a_usd.shift(1).fillna(0) * ret_a
    pnl_b = pos_b_usd.shift(1).fillna(0) * ret_b
    daily_pnl = pnl_a + pnl_b

    trade_days = trade_sig.abs() > 0
    total_cost_bps = (config.transaction_cost_bps + config.slippage_bps) / 10_000
    gross_exposure = pos_a_usd.abs() + pos_b_usd.abs()
    costs = trade_days * gross_exposure * total_cost_bps
    daily_pnl -= costs

    n_bars = len(daily_pnl)
    bars_per_year = 365
    if n_bars > 0:
        median_gap_hours = pd.Series(daily_pnl.index).diff().dropna().median().total_seconds() / 3600
        bars_per_year = 365 * 24 / max(median_gap_hours, 1)

    daily_pnl = _apply_kill_switch(daily_pnl, config)

    equity = config.capital + daily_pnl.cumsum()
    roll_max = equity.cummax()
    drawdown = (equity - roll_max) / roll_max
    max_dd = float(drawdown.min())

    total_ret = float((equity.iloc[-1] - config.capital) / config.capital)
    ann_ret = (1 + total_ret) ** (bars_per_year / n_bars) - 1
    bar_ret_series = daily_pnl / config.capital
    sharpe = float(bar_ret_series.mean() / (bar_ret_series.std() + 1e-9) * np.sqrt(bars_per_year))
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else np.nan

    pos_change = (pos != 0) & (pos.shift(1).fillna(0) == 0)
    num_trades = int(pos_change.sum())

    trade_pnls = []
    in_trade = False
    trade_pnl = 0.0
    for i in range(len(pos)):
        if pos.iloc[i] != 0:
            trade_pnl += daily_pnl.iloc[i]
            in_trade = True
        elif in_trade:
            trade_pnls.append(trade_pnl)
            trade_pnl = 0.0
            in_trade = False
    if in_trade:
        trade_pnls.append(trade_pnl)

    win_rate = float(np.mean([p > 0 for p in trade_pnls])) if trade_pnls else 0.0
    avg_trade = float(np.mean(trade_pnls)) if trade_pnls else 0.0

    alpha_tstat, alpha_pvalue, beta_to_bm = None, None, None
    if benchmark_prices is not None:
        alpha_tstat, alpha_pvalue, beta_to_bm = compute_alpha_tstat(bar_ret_series, benchmark_prices)

    return BacktestResult(
        pair=(asset_a, asset_b),
        total_return=round(total_ret, 4),
        annualized_return=round(ann_ret, 4),
        sharpe_ratio=round(sharpe, 3),
        max_drawdown=round(max_dd, 4),
        num_trades=num_trades,
        win_rate=round(win_rate, 3),
        avg_trade_return=round(avg_trade, 2),
        calmar_ratio=round(calmar, 3) if not np.isnan(calmar) else None,
        equity_curve=equity,
        daily_pnl=daily_pnl,
        signals_df=signals_df.loc[common],
        alpha_tstat=round(alpha_tstat, 3) if alpha_tstat is not None else None,
        alpha_pvalue=round(alpha_pvalue, 4) if alpha_pvalue is not None else None,
        beta_to_benchmark=round(beta_to_bm, 4) if beta_to_bm is not None else None,
    )


def run_backtest_walkforward(
    price_a: pd.Series,
    price_b: pd.Series,
    signals_df: pd.DataFrame,
    hedge_ratio: float,
    asset_a: str,
    asset_b: str,
    config: BacktestConfig = None,
    sizing_cfg=None,
    rebalance_freq: str = "QS",
    benchmark_prices: Optional[pd.Series] = None,
) -> BacktestResult:
    """
    Walk-forward backtest with quarterly Kelly weight reestimation.

    At the END of each quarter, we compute Kelly weights using all data UP TO that point.
    Those weights are then applied FORWARD to the next quarter — never to the same period
    they were estimated from.

    This eliminates the in-sample weight optimisation bias:
      "Standing at June 30, compute weights using data up to June 30,
       then use those weights July, Aug, Sept. On Sept 30, recompute."

    The first forward window is always flat (no exposure) until we have at least
    min_vol_lookback bars of history to estimate Kelly from.

    Args:
        rebalance_freq: Pandas offset alias. "QS" = quarterly (Jan/Apr/Jul/Oct starts).
        sizing_cfg:     SizingConfig instance. Defaults to SizingConfig() if None.
        benchmark_prices: Optional benchmark for alpha t-stat.
    """
    from sizing import kelly_size, SizingConfig

    if config is None:
        config = BacktestConfig()
    if sizing_cfg is None:
        sizing_cfg = SizingConfig()

    common = price_a.index.intersection(price_b.index).intersection(signals_df.index)
    pa = price_a.loc[common]
    pb = price_b.loc[common]
    sig = signals_df.loc[common]

    # Quarter-start boundaries over the full date range
    rebalance_dates = pd.date_range(
        start=common[0].normalize(),
        end=common[-1].normalize(),
        freq=rebalance_freq,
        tz=common.tz,
    )
    rebalance_dates = rebalance_dates[rebalance_dates > common[0]]
    # Sentinel is one day past the last bar so the window_mask (< rb_date) includes it
    rebalance_dates = rebalance_dates.append(pd.DatetimeIndex([common[-1] + pd.Timedelta(days=1)]))

    all_sizing_chunks = []
    prev_date = common[0]

    for rb_date in rebalance_dates:
        train_mask = sig.index < rb_date
        window_mask = (sig.index >= prev_date) & (sig.index < rb_date)
        window_sig = sig.loc[window_mask]

        if window_sig.empty:
            prev_date = rb_date
            continue

        if train_mask.sum() < sizing_cfg.min_vol_lookback:
            # Insufficient history — stay flat for this window
            zero_sizing = pd.DataFrame({
                "pos_a_usd": 0.0,
                "pos_b_usd": 0.0,
                "dollar_exposure": 0.0,
                "kelly_f": 0.0,
                "rolling_vol": np.nan,
            }, index=window_sig.index)
            all_sizing_chunks.append(zero_sizing)
            prev_date = rb_date
            continue

        train_sizing = kelly_size(
            sig.loc[train_mask],
            pa.loc[train_mask],
            pb.loc[train_mask],
            hedge_ratio,
            sizing_cfg,
        )
        # Trailing-60-day median of kelly_f rather than the last point — a single
        # noisy day at quarter-end was deactivating the whole next quarter.
        recent_kelly = train_sizing["kelly_f"].iloc[-60:] if not train_sizing.empty else pd.Series([0.0])
        target_kelly_f = float(recent_kelly.median())
        target_kelly_f = max(0.0, min(target_kelly_f, sizing_cfg.max_position_pct))

        dollar_exposure = target_kelly_f * sizing_cfg.capital
        pos = window_sig["position"]

        window_sizing = pd.DataFrame({
            "pos_a_usd": (dollar_exposure * pos).round(2),
            "pos_b_usd": (-dollar_exposure * hedge_ratio * pos).round(2),
            "dollar_exposure": dollar_exposure,
            "kelly_f": target_kelly_f,
            "rolling_vol": np.nan,
        }, index=window_sig.index)
        all_sizing_chunks.append(window_sizing)
        prev_date = rb_date

    if not all_sizing_chunks:
        raise RuntimeError("Walk-forward sizing produced no output — check data length.")

    wf_sizing = pd.concat(all_sizing_chunks).sort_index()
    wf_sizing = wf_sizing[~wf_sizing.index.duplicated(keep="first")]

    return run_backtest(
        pa, pb, sig, wf_sizing,
        hedge_ratio, asset_a, asset_b, config,
        benchmark_prices=benchmark_prices,
    )


def print_results(result: BacktestResult, label: str = "") -> None:
    pair_str = f"{result.pair[0]}/{result.pair[1]}"
    title = f"BACKTEST RESULTS  —  {pair_str}" + (f"  [{label}]" if label else "")
    print(f"\n{'='*60}")
    print(title)
    print(f"{'='*60}")
    print(f"  Total return:        {result.total_return*100:>8.2f}%")
    print(f"  Annualized return:   {result.annualized_return*100:>8.2f}%")
    print(f"  Sharpe ratio:        {result.sharpe_ratio:>8.3f}")
    print(f"  Max drawdown:        {result.max_drawdown*100:>8.2f}%")
    print(f"  Calmar ratio:        {str(result.calmar_ratio):>8}")
    print(f"  Number of trades:    {result.num_trades:>8}")
    print(f"  Win rate:            {result.win_rate*100:>8.1f}%")
    print(f"  Avg trade PnL (USD): {result.avg_trade_return:>8.2f}")
    if result.alpha_tstat is not None:
        sig_flag = "***" if abs(result.alpha_tstat) > 2.58 else ("**" if abs(result.alpha_tstat) > 1.96 else ("*" if abs(result.alpha_tstat) > 1.645 else ""))
        print(f"  Alpha t-stat (vs BTC):{result.alpha_tstat:>7.3f}  {sig_flag}")
        print(f"  Alpha p-value:       {result.alpha_pvalue:>8.4f}")
        print(f"  Beta to benchmark:   {result.beta_to_benchmark:>8.4f}")
    print(f"{'='*60}")
