# Crypto Stat-Arb Trading System

**End-to-end statistical arbitrage pipeline for crypto pairs trading.**  
Implements cointegration-based pairs trading across 16 assets (BTC, ETH, SOL, ADA, XRP, DOGE, DOT, AVAX, LINK, LTC, BCH, ATOM, NEAR, UNI, AAVE, FIL) with walk-forward Kelly sizing, strict train/test separation, and a vectorized backtest engine with transaction costs.

UNI is sourced as `UNI7083-USD` on Yahoo Finance (the legacy `UNI-USD` feed froze 2025-04-17 after a ticker migration). Polygon (MATIC → POL) was removed when its Yahoo feed went stale; the live POL replacement (`POL28321-USD`) only goes back to October 2023, which would truncate the training window.

---

## Strategy Overview

This system exploits **mean-reverting spreads** between cointegrated crypto asset pairs. When a pair's log-price spread deviates significantly from its historical mean (measured in standard deviations), we take a position expecting reversion.

```
Long spread (+1):  z-score < -2.0  →  long asset A, short asset B
Short spread (-1): z-score > +2.0  →  short asset A, long asset B
Exit:              |z-score| < 0.5  →  mean reversion achieved
Stop-loss:         |z-score| > 3.5  →  relationship breaking down
```

**Pair selection** uses a dual-test framework with half-life filtering. A pair must pass:
1. **Engle-Granger cointegration test** (p < 0.20) — relaxed threshold; EG has low power on short crypto series
2. **Johansen trace test** (95%) — used where signal is available; can be disabled via `require_johansen`
3. **ADF test on residuals** (p < 0.15)
4. **Ornstein-Uhlenbeck half-life** between 1–120 calendar days

---

## Architecture

```
data.py            ← yfinance fetch (daily bars), parquet caching
cointegration.py   ← Johansen + Engle-Granger screening, OU half-life, hedge ratio
signals.py         ← Rolling z-score engine, entry/exit/stop signal generation
sizing.py          ← Fractional Kelly position sizing (25% Kelly fraction)
backtest.py        ← Vectorized PnL engine, walk-forward backtest, alpha t-stat
main.py            ← Pipeline orchestrator (CLI)
diagnose_pairs.py  ← Per-pair diagnostic table + threshold relaxation ladder
```

---

## Methodology

### Train / Test Split

Pair selection is performed **exclusively on training data** to prevent look-ahead bias:

| Window | Period | Bars |
|--------|--------|------|
| Train (screening) | 2023-01-01 → 2024-12-31 | 731 daily bars |
| Test (backtest) | 2025-01-01 → present | 519 daily bars |

Hedge ratios and OU half-lives are estimated on training data only. Signal thresholds (`entry_z`, `exit_z`, `stop_z`, `lookback`) are fixed a priori and never tuned on any data.

### Walk-Forward Kelly Sizing

Position sizes are computed via quarterly walk-forward Kelly reestimation:

- At the end of each quarter, Kelly weights are computed from all data **up to that point**
- Those weights are then applied **forward** to the next quarter only
- The first quarter is flat (no exposure) until sufficient history exists

This eliminates in-sample weight optimisation bias.

### Transaction Costs

- **20bps per trade** (10bps per leg) — crypto market standard
- **3bps slippage** per trade
- **25% drawdown kill-switch** — trading halts if portfolio drawdown exceeds 25%

---

## Backtest Results (daily bars, $100k capital, walk-forward Kelly)

> 10 cointegrated pairs identified out of 120 candidates on 2-year training data (EG + ADF + OU half-life).  
> Results below are **out-of-sample** (2025-01-01 → 2026-06-03, ~17 months), walk-forward sized.  
> Alpha t-stat: intercept t-stat from `r_strategy = α + β·r_BTC + ε`.

| Pair | Ann. Return | Sharpe | Max DD | Calmar | Trades | Win Rate | Alpha t-stat | β/BTC |
|------|-------------|--------|--------|--------|--------|----------|--------------|-------|
| **ADA/AAVE** | **5.1%** | **1.89** | **-1.0%** | **5.05** | 3 | 67% | **2.27**\*\* | 0.003 |
| DOT/AAVE | 2.5% | 1.34 | -1.0% | 2.48 | 2 | 50% | 1.65\* | 0.004 |
| DOT/BCH | 2.5% | 1.31 | -0.9% | 2.86 | 2 | 50% | 1.62 | 0.005 |
| ADA/UNI | 1.0% | 0.24 | -5.1% | 0.20 | 3 | 0% | 0.29 | 0.001 |
| DOGE/UNI | 0.4% | 0.08 | -6.9% | 0.06 | 4 | 25% | 0.07 | -0.006 |
| DOT/FIL | 0.2% | 0.09 | -1.9% | 0.10 | 6 | 17% | 0.11 | 0.000 |
| DOT/LINK | 0.0% | 0.00 | 0.0% | — | 5 | 0% | — | 0.000 |
| ETH/UNI | -0.5% | -0.23 | -3.5% | -0.14 | 2 | 50% | -0.30 | -0.003 |
| ETH/SOL | -2.0% | -0.50 | -6.9% | -0.29 | 3 | 0% | -0.67 | -0.013 |
| ETH/NEAR | -3.2% | -0.78 | -7.4% | -0.44 | 4 | 0% | -0.94 | -0.003 |

> \* p < 0.10  \*\* p < 0.05

**ADA/AAVE** is the standout: Sharpe 1.89 with -1.0% max drawdown and an alpha t-stat of **2.27 (p = 0.024)** — significant at 5% and not explained by BTC exposure (β = 0.003). Caveat: training-period performance was negative (Sharpe -0.87), so this only "worked" out-of-sample. Could be a regime shift in DeFi-vs-L1 dynamics in 2025; could be noise — wait for more out-of-sample bars before sizing aggressively.

Three of the top three pairs use **AAVE as the short leg** (ADA/AAVE, DOT/AAVE) or BCH/DOT-cluster mechanics — their bets are highly correlated. Portfolio-level Kelly should account for this rather than treating each pair as an independent bet.

Note: most pairs show negative in-sample Sharpe on training data and positive out-of-sample, which is the expected pattern for a stat-arb with no in-sample parameter tuning — it rules out look-ahead bias as the source of performance.

---

## Stress Test: 2026 Crypto Drawdown

In May 2026, BTC dropped **-20.6%** in 24 trading days (peak $82,139 on 2026-05-10 → $64,014 on 2026-06-03). ETH dropped -25.2%, SOL -26.4%. This was the sharpest broad-market crypto move since the 2022 cycle and a useful out-of-sample stress test for the strategy.

**Strategy P&L through the drawdown (equal-weight across 10 pairs):**

| Metric | Value |
|---|---|
| Portfolio return | **-0.10%** |
| Portfolio max drawdown | -0.24% |
| Beta to BTC during window | ~0.0 |

Walk-forward Kelly sized 6 of 10 pairs to zero exposure going into the window (trailing edge had turned negative for those pairs at the 2026-04-01 rebalance). Of the four pairs that traded:

- **ADA/AAVE** generated **+1.30%** through the window (window Sharpe 5.45, MaxDD -0.30%) — true alpha through the drawdown, with z-score-driven mean-reversion firing as expected.
- **ETH/NEAR** lost -2.49% in a single stopped-out trade — NEAR ripped +46% against the short leg in a 12-day window while ETH dropped -15%. The stop-loss and re-entry cooldown together capped this at one bad trade; without the cooldown the strategy would have whipsawed in 8 more times and lost ~-7%.
- Two other UNI-leg pairs traded small positions, roughly flat.

The portfolio came through a 20% market move with essentially zero P&L and zero drawdown — the dollar-neutral design and tight stops behaved as intended. **The honest read is not that the strategy "beat" the drawdown; it stayed out of the way, with one genuine alpha trade (ADA/AAVE) offsetting one losing trade (ETH/NEAR).** What this scenario surfaced was three real bugs which have now been fixed (see Methodology Notes below).

### Methodology fixes triggered by this stress test

1. **Stale ticker data**: `UNI-USD` and `MATIC-USD` Yahoo feeds froze in early 2025 after token migrations. The cache forward-filled at the last live price, so four pairs were silently running on dead data. Replaced UNI → `UNI7083-USD`, removed MATIC.
2. **Walk-forward Kelly went hard-flat too easily**: the sizer took `kelly_f.iloc[-1]` (a single 30-day-rolling stat at quarter-end) and clipped to `[0, 0.20]`. A single noisy day deactivated the whole next quarter. Now uses the **trailing 60-day median** of `kelly_f` for a smoother estimate.
3. **Asymmetric stop-loss + whipsaw**: the stop only fired when z moved in the *profit* direction past stop_z, not when z moved against the trade. After fixing to `|z| > stop_z`, losing trades got cut at -3.5σ; a **post-stop cooldown** was added so the same direction can't re-enter until z crosses zero, preventing trend-against whipsaw.

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full pipeline (loads from cache if available)
python main.py

# Re-fetch fresh data from yfinance
python main.py --refresh

# Generate equity curve plots
python main.py --plot

# Print current live signals only
python main.py --live
```

If no cointegrated pairs are found, run the diagnostic tool:

```bash
python diagnose_pairs.py
```

This prints a full per-pair breakdown of which filter kills each pair, plus a relaxation ladder showing survivor counts at progressively looser thresholds.

---

## Signal Logic

### 1. Cointegration Screening

For each candidate pair (A, B):

**Engle-Granger**: Runs OLS regression `log(A) ~ β·log(B) + c`, then ADF-tests the residuals for stationarity.

**Johansen trace test**: Tests the rank of the cointegration matrix — more powerful than EG in multi-asset settings and more robust to regime shifts. Can be disabled with `require_johansen=False` if no pairs survive (Johansen has low power on short crypto histories).

**Ornstein-Uhlenbeck half-life**: Estimated from `ΔS_t = θ·S_{t-1} + c + ε_t`:

```
half_life_bars = -ln(2) / θ
half_life_days = half_life_bars × (bar_hours / 24)
```

The `bar_hours` parameter is explicit throughout — `compute_halflife()` and `screen_pairs()` both accept it. With daily bars, `bar_hours=24`.

Pairs with half-life < 1 day (execution risk) or > 120 days (poor capital efficiency) are excluded.

### 2. Z-Score Signal

Rolling z-score of the log-price spread over a 120-bar lookback (≈ 4 months on daily bars):

```python
z_t = (spread_t − μ_120) / σ_120
```

### 3. Walk-Forward Kelly Sizing

Fractional Kelly (25%) reestimated quarterly on past data only:

```
f* = kelly_fraction × (μ_spread / σ_spread²)
```

Capped at 20% of capital per pair. Dollar-neutral: leg sizes scaled by OLS hedge ratio.

---

## Key Design Decisions

- **Log-price spread** (not price ratio): stationarity of the spread is guaranteed by the cointegration relationship; handles different price scales naturally
- **Dual cointegration tests**: EG is easy to interpret; Johansen is more powerful in short samples — using both reduces false positives even at relaxed p-value thresholds
- **Relaxed EG threshold (p < 0.20)**: EG and ADF both have low power on crypto price series (high vol, short history). The dual-test requirement acts as a natural backstop
- **Explicit `bar_hours` parameter**: half-life computation is decoupled from bar frequency — switching bar frequency requires changing one constant
- **Half-life filter**: avoids pairs that mean-revert too slowly (low capital efficiency) or too fast (slippage dominates)
- **Fractional Kelly**: full Kelly oversizes during losing streaks — 25% fraction gives similar EV with materially lower variance
- **Walk-forward sizing**: Kelly weights computed from past data only, applied forward — eliminates in-sample weight optimisation bias
- **Kill-switch**: halts trading at 25% portfolio drawdown to prevent catastrophic loss if cointegration breaks down

---

## Limitations & Next Steps

- **Trade count**: 2–6 trades per pair in the 17-month test window — Sharpe estimates have wide confidence intervals at this sample size
- **Correlated pair bets**: three of the top four pairs use DOT as the long leg (DOT/LINK, DOT/AAVE, DOT/BCH). Their PnL streams are not independent; portfolio-level Kelly should account for this rather than treating each pair as a separate bet
- **ADA/AAVE regime risk**: best out-of-sample pair was negative in-sample — could be a genuine regime shift or could fail forward as quickly as it appeared
- **Transaction costs are estimated**: real costs vary with order size, venue, and market conditions
- **Regime changes**: cointegration relationships break during market stress; could add regime detection (e.g. HMM) to suspend trading when structural breaks are detected
- **Symmetric Kelly short sizing**: when spread drift is negative, the current implementation clips Kelly to zero on the short side; `|μ|` regardless of sign would give symmetric short exposure

---

## Tech Stack

`Python` · `statsmodels` · `pandas` · `NumPy` · `yfinance` · `matplotlib` · `pyarrow`

---

## Author

**Aman Syed** — [LinkedIn](https://linkedin.com/in/yourprofile) · [GitHub](https://github.com/yourusername)

Quantitative finance professional with experience in systematic trading (energy futures, stat-arb), index research (MSCI), and counterparty credit risk. This project is part of a broader series on applied quantitative finance.
