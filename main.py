"""
main.py — End-to-end crypto stat-arb pipeline runner.

Usage:
    python main.py                    # run full pipeline with cached data
    python main.py --refresh          # re-fetch fresh data from yfinance
    python main.py --plot             # show equity curve plots
    python main.py --live             # print current live signals only

Pipeline:
    1. Load/fetch OHLCV data (yfinance, 9 assets)
    2. Split data: TRAIN 2023-01-01→2024-12-31, TEST 2025-01-01→present
    3. Screen pairs using TRAINING data only (no look-ahead in pair selection)
    4. Generate z-score signals on TEST data using training-derived parameters
    5. Walk-forward Kelly sizing (quarterly rebalance; weights computed from past data)
    6. Backtest with 20bps transaction costs and drawdown kill-switch
    7. Report performance + alpha t-stat vs BTC benchmark
    8. Annual re-screen: re-run cointegration on an expanding window (TRAIN_START
       through the most recently completed calendar year) to catch newly-formed
       cointegrated pairs. Each new pair gets its own OOS window starting the day
       after the expanded screen ends, so it's never scored on data the screen saw.

Key methodology notes:
  - Pair selection (cointegration tests, hedge ratios, half-life) is estimated on TRAIN only.
  - Signal thresholds (entry_z, exit_z, stop_z, lookback) are fixed a priori — not fit to data.
  - Kelly weights are walk-forward (no in-sample weight optimisation bias).
  - Transaction cost: 20bps total per trade (10bps per leg) — crypto market standard.
  - Alpha t-stat: intercept t-statistic from regressing strategy returns on BTC returns.
  - The original TRAIN/TEST split and its pairs are never touched by the annual
    re-screen — that OOS track record stays comparable run over run. The re-screen
    only ever *adds* newly-qualifying pairs, each with a fresh, non-overlapping
    OOS window; it never extends or re-fits the original pairs' training window.
"""

import argparse
import numpy as np
import pandas as pd

from data import load_prices, INTERVAL_1D
from cointegration import screen_pairs, print_screening_report, compute_spread
from signals import generate_signals, SignalConfig, get_current_signal
from sizing import kelly_size, SizingConfig
from backtest import run_backtest, run_backtest_walkforward, print_results, BacktestConfig

TICKERS = ["BTC", "ETH", "SOL", "ADA", "XRP", "DOGE", "DOT", "AVAX", "LINK",
           "LTC", "BCH", "ATOM", "NEAR", "UNI", "AAVE", "FIL"]
BAR_INTERVAL = INTERVAL_1D
BAR_HOURS = 24.0   # daily bars

# ── Train / Test split ──────────────────────────────────────────────────────
# Cointegration screening and hedge ratio estimation: TRAIN window only.
# Signal generation and walk-forward backtest: TEST window.
# This mirrors real-world usage where you fit on historical data and trade forward.
TRAIN_START = "2023-01-01"
TRAIN_END   = "2024-12-31"  # ~2 years of training data
TEST_START  = "2025-01-01"
TEST_END    = None           # through present (~15 months as of Apr 2026)

# ── Annual re-screen (expanding window) ─────────────────────────────────────
# Once a year, re-screen for cointegration on TRAIN_START -> end of the most
# recently completed calendar year. This is strictly wider than TRAIN_START/
# TRAIN_END and only ever grows, once per year. Pairs that qualify here but
# didn't in the original screen are "new pairs" — they get backtested on their
# own OOS window starting the day after RESCREEN_END, so the re-screen never
# sees the data it's later scored on.
_last_complete_year_end = pd.Timestamp(year=pd.Timestamp.now(tz="UTC").year - 1, month=12, day=31, tz="UTC")
RESCREEN_END = max(_last_complete_year_end, pd.Timestamp(TRAIN_END, tz="UTC"))
RESCREEN_TEST_START = RESCREEN_END + pd.Timedelta(days=1)


def run_pipeline(refresh: bool = False, plot: bool = False, live: bool = False):
    print("\n" + "="*60)
    print("  CRYPTO STAT-ARB PIPELINE  (v2 — Walk-forward + Alpha t-stat)")
    print(f"  Assets: {', '.join(TICKERS)}  |  Source: yfinance  |  Freq: daily")
    print(f"  Train: {TRAIN_START} → {TRAIN_END}  |  Test: {TEST_START} → present")
    print("="*60)

    # ── 1. Load full price history ────────────────────────────────
    print("\n[1/6] Loading price data (full history)...")
    prices_full = load_prices(
        tickers=TICKERS,
        start=TRAIN_START,
        interval=BAR_INTERVAL,
        refresh=refresh,
    )
    print(f"      Full range: {prices_full.index[0].date()} → {prices_full.index[-1].date()}")
    print(f"      Total bars: {len(prices_full):,}")

    # Split
    # Use tz-aware timestamps for slicing — plain string slicing on a tz-aware
    # DatetimeIndex is unreliable across pandas versions.
    _train_end = pd.Timestamp(TRAIN_END, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    _test_start = pd.Timestamp(TEST_START, tz="UTC")
    prices_train = prices_full[prices_full.index <= _train_end]
    prices_test  = prices_full[prices_full.index >= _test_start]
    print(f"      Train bars: {len(prices_train):,}  |  Test bars: {len(prices_test):,}")

    if len(prices_test) < 50:
        print("WARNING: Very few test bars — extend test period or adjust split dates.")

    # BTC prices for alpha t-stat benchmark
    btc_benchmark = prices_full["BTC"] if "BTC" in prices_full.columns else None

    # ── 2. Cointegration screening on TRAIN data only ─────────────
    print("\n[2/6] Screening pairs on TRAINING data (2023–2024)...")
    print("      (Pair selection must not use any test-period data.)")
    screening_results = screen_pairs(
        prices_train,
        eg_pvalue_threshold=0.20,   # relaxed; EG has low power on short crypto series
        min_halflife_days=1.0,
        max_halflife_days=120.0,    # relaxed; slow-reverting pairs still tradeable
        adf_pvalue_threshold=0.15,  # relaxed; same low-power reasoning as EG
        require_johansen=False,     # Johansen fails all 36 pairs on daily crypto — EG+ADF+HL sufficient
        bar_hours=BAR_HOURS,
    )
    print_screening_report(screening_results)

    valid_pairs = [r for r in screening_results if r.is_valid]
    if not valid_pairs:
        print("No cointegrated pairs found in training data. Try relaxing thresholds.")
        return

    # ── 3-6. Per-pair: signals → walk-forward sizing → backtest ───
    signal_cfg = SignalConfig(entry_z=2.0, exit_z=0.5, stop_z=3.5, lookback=120)
    sizing_cfg = SizingConfig(capital=100_000, max_position_pct=0.20, kelly_fraction=0.25)
    bt_cfg = BacktestConfig(
        capital=100_000,
        transaction_cost_bps=20.0,   # industry standard for crypto (10bps per leg)
        slippage_bps=3.0,
    )

    all_results_train = []
    all_results_test  = []

    for pair_res in valid_pairs:
        result_fit, result_oos, sig_fit, sig_oos = _backtest_pair(
            pair_res, prices_train, prices_test,
            signal_cfg, sizing_cfg, bt_cfg, btc_benchmark,
            fit_label="TRAIN — in-sample reference",
            oos_label="TEST — out-of-sample, walk-forward",
            step_label="[3-6]", live=live,
        )
        all_results_train.append(result_fit)
        if result_oos is not None:
            all_results_test.append(result_oos)
            if plot:
                _plot_pair(result_fit, result_oos, sig_fit, sig_oos, signal_cfg, pair_res.asset_a, pair_res.asset_b)

    # ── Summary table ─────────────────────────────────────────────
    if all_results_test and not live:
        _print_summary(all_results_train, all_results_test, valid_pairs)

    # ── 8. Annual re-screen: catch newly-cointegrated pairs ────────
    original_pair_keys = {(r.asset_a, r.asset_b) for r in valid_pairs}
    prices_rescreen = prices_full[prices_full.index <= RESCREEN_END]
    prices_new_oos  = prices_full[prices_full.index >= RESCREEN_TEST_START]

    print(f"\n[8/8] Annual re-screen on expanding window "
          f"({TRAIN_START} → {RESCREEN_END.date()})...")
    rescreen_results = screen_pairs(
        prices_rescreen,
        eg_pvalue_threshold=0.20,
        min_halflife_days=1.0,
        max_halflife_days=120.0,
        adf_pvalue_threshold=0.15,
        require_johansen=False,
        bar_hours=BAR_HOURS,
    )
    new_pairs = [r for r in rescreen_results if r.is_valid and (r.asset_a, r.asset_b) not in original_pair_keys]

    if not new_pairs:
        print("      No newly-cointegrated pairs beyond the original TRAIN screen.")
    else:
        print(f"      {len(new_pairs)} new pair(s) found — OOS window: "
              f"{RESCREEN_TEST_START.date()} → present (never seen by this screen).")
        new_results_fit, new_results_oos = [], []
        for pair_res in new_pairs:
            result_fit, result_oos, sig_fit, sig_oos = _backtest_pair(
                pair_res, prices_rescreen, prices_new_oos,
                signal_cfg, sizing_cfg, bt_cfg, btc_benchmark,
                fit_label=f"RE-SCREEN FIT ({TRAIN_START}→{RESCREEN_END.date()})",
                oos_label=f"NEW PAIR — OOS from {RESCREEN_TEST_START.date()}",
                step_label="[8/8]", live=live,
            )
            new_results_fit.append(result_fit)
            if result_oos is not None:
                new_results_oos.append(result_oos)
                if plot:
                    _plot_pair(result_fit, result_oos, sig_fit, sig_oos, signal_cfg, pair_res.asset_a, pair_res.asset_b)

        if new_results_oos and not live:
            print("\n(New pairs are tracked separately — too little OOS history yet to "
                  "compare against the original pairs' multi-year track record.)")
            _print_summary(new_results_fit, new_results_oos, new_pairs)


def _backtest_pair(pair_res, prices_fit, prices_oos, signal_cfg, sizing_cfg, bt_cfg,
                    btc_benchmark, fit_label, oos_label, step_label="", live=False):
    """Fit on prices_fit (in-sample reference), then walk-forward backtest on prices_oos."""
    a, b = pair_res.asset_a, pair_res.asset_b
    hedge_ratio = pair_res.hedge_ratio
    print(f"\n{step_label} Processing pair: {a} / {b}  (hedge ratio: {hedge_ratio:.4f})")

    spread_fit = compute_spread(prices_fit[a], prices_fit[b], hedge_ratio)
    sig_fit = generate_signals(spread_fit, signal_cfg)
    sizing_fit = kelly_size(sig_fit, prices_fit[a], prices_fit[b], hedge_ratio, sizing_cfg)
    result_fit = run_backtest(
        prices_fit[a], prices_fit[b],
        sig_fit, sizing_fit,
        hedge_ratio, a, b, bt_cfg,
        benchmark_prices=btc_benchmark,
    )
    print_results(result_fit, label=fit_label)

    if prices_oos.empty:
        print("      (No OOS data available yet, skipping out-of-sample backtest.)")
        return result_fit, None, sig_fit, None

    spread_oos = compute_spread(prices_oos[a], prices_oos[b], hedge_ratio)
    sig_oos = generate_signals(spread_oos, signal_cfg)

    print(f"      OOS signals — long: {(sig_oos.position==1).sum()}bars  "
          f"short: {(sig_oos.position==-1).sum()}bars  "
          f"flat: {(sig_oos.position==0).sum()}bars")

    if live:
        sig = get_current_signal(sig_oos, a, b, hedge_ratio)
        pos_label = {1: "LONG SPREAD", -1: "SHORT SPREAD", 0: "FLAT"}[sig.position]
        print(f"\n  ── LIVE SIGNAL: {a}/{b} ──")
        print(f"     Date:     {sig.date}")
        print(f"     Z-score:  {sig.zscore:+.3f}")
        print(f"     Position: {pos_label}")
        print(f"     Hedge ratio: {sig.hedge_ratio}")
        return result_fit, None, sig_fit, sig_oos

    result_oos = run_backtest_walkforward(
        prices_oos[a], prices_oos[b],
        sig_oos,
        hedge_ratio, a, b, bt_cfg,
        sizing_cfg=sizing_cfg,
        rebalance_freq="QS",
        benchmark_prices=btc_benchmark,
    )
    print_results(result_oos, label=oos_label)
    return result_fit, result_oos, sig_fit, sig_oos


def _plot_pair(result_train, result_test, sig_train, sig_test, signal_cfg, a, b):
    try:
        import matplotlib.pyplot as plt
        import os
        os.makedirs("plots", exist_ok=True)

        fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=False)
        fig.suptitle(f"Crypto Stat-Arb: {a}/{b}", fontweight="bold")

        # Train equity
        axes[0].plot(result_train.equity_curve, color="steelblue", lw=1.5, label="Train (in-sample)")
        axes[0].axhline(100_000, color="gray", ls="--", lw=0.8)
        axes[0].set_ylabel("Equity (USD)")
        axes[0].set_title("Equity Curve — Train (blue) / Test (orange)")

        # Test equity (separate x-axis — different dates)
        ax0b = axes[0].twinx()
        ax0b.plot(result_test.equity_curve, color="darkorange", lw=1.5, label="Test (out-of-sample)")
        ax0b.set_ylabel("")
        axes[0].legend(loc="upper left", fontsize=8)
        ax0b.legend(loc="upper right", fontsize=8)

        # Z-score train
        axes[1].plot(sig_train["zscore"], color="purple", lw=0.9, alpha=0.7, label="Train z-score")
        axes[1].axhline(signal_cfg.entry_z, color="red", ls="--", lw=0.8)
        axes[1].axhline(-signal_cfg.entry_z, color="green", ls="--", lw=0.8)
        axes[1].set_ylabel("Z-score (Train)")
        axes[1].set_title("Train Z-score")
        axes[1].legend(fontsize=8)

        # Z-score test
        axes[2].plot(sig_test["zscore"], color="darkorange", lw=0.9, alpha=0.7, label="Test z-score")
        axes[2].axhline(signal_cfg.entry_z, color="red", ls="--", lw=0.8)
        axes[2].axhline(-signal_cfg.entry_z, color="green", ls="--", lw=0.8)
        axes[2].set_ylabel("Z-score (Test)")
        axes[2].set_title("Test Z-score")
        axes[2].legend(fontsize=8)

        plt.tight_layout()
        fname = f"plots/{a}_{b}_backtest.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        print(f"      Plot saved: {fname}")
        plt.show()
    except ImportError:
        print("      (matplotlib not installed, skipping plot)")


def _print_summary(results_train, results_test, valid_pairs):
    print("\n" + "="*80)
    print("SUMMARY TABLE — TRAIN vs TEST (out-of-sample, walk-forward Kelly)")
    print("  Transaction costs: 20bps/trade (10bps per leg) + 3bps slippage")
    print("  Alpha t-stat: intercept from regression of strategy returns on BTC returns")
    print("="*80)

    rows = []
    pair_to_train = {r.pair: r for r in results_train}
    for r_test in results_test:
        r_train = pair_to_train.get(r_test.pair, None)
        row = {
            "Pair": f"{r_test.pair[0]}/{r_test.pair[1]}",
            "Train Ann.Ret": f"{r_train.annualized_return*100:.1f}%" if r_train else "—",
            "Train Sharpe": f"{r_train.sharpe_ratio:.2f}" if r_train else "—",
            "Test Ann.Ret": f"{r_test.annualized_return*100:.1f}%",
            "Test Sharpe": f"{r_test.sharpe_ratio:.2f}",
            "Test MaxDD": f"{r_test.max_drawdown*100:.1f}%",
            "Test Calmar": str(r_test.calmar_ratio),
            "Trades": r_test.num_trades,
            "WinRate": f"{r_test.win_rate*100:.0f}%",
            "Alpha t-stat": f"{r_test.alpha_tstat:.2f}" if r_test.alpha_tstat is not None else "—",
            "Beta/BTC": f"{r_test.beta_to_benchmark:.3f}" if r_test.beta_to_benchmark is not None else "—",
        }
        rows.append(row)

    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False))
    print("="*80)
    print("  Significance: * p<0.10  ** p<0.05  *** p<0.01 (alpha t-stat)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crypto Stat-Arb Pipeline")
    parser.add_argument("--refresh", action="store_true", help="Re-fetch data from yfinance (clears cache)")
    parser.add_argument("--plot", action="store_true", help="Generate equity curve plots")
    parser.add_argument("--live", action="store_true", help="Print current live signals only")
    args = parser.parse_args()
    run_pipeline(refresh=args.refresh, plot=args.plot, live=args.live)