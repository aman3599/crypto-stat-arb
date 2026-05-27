# Crypto Stat-Arb Trading System

**End-to-end statistical arbitrage pipeline for crypto pairs trading.**  
Implements cointegration-based pairs trading across 17 assets (BTC, ETH, SOL, ADA, XRP, DOGE, DOT, AVAX, LINK, LTC, BCH, MATIC, ATOM, NEAR, UNI, AAVE, FIL) with walk-forward Kelly sizing, strict train/test separation, and a vectorized backtest engine with transaction costs.

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
| Test (backtest) | 2025-01-01 → present | 511 daily bars |

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

> 11 cointegrated pairs identified out of 136 candidates on 2-year training data (EG + ADF + OU half-life). 5 also pass the Johansen trace test.  
> Results below are **out-of-sample** (2025-01-01 → 2026-05-26, ~17 months), walk-forward sized.  
> Alpha t-stat: intercept t-stat from `r_strategy = α + β·r_BTC + ε`.

| Pair | Ann. Return | Sharpe | Max DD | Calmar | Trades | Win Rate | Alpha t-stat | β/BTC |
|------|-------------|--------|--------|--------|--------|----------|--------------|-------|
| **ADA/AAVE** | **4.9%** | **1.85** | **-1.0%** | **4.86** | 3 | 67% | **2.20**\*\* | 0.004 |
| **DOT/LINK** | **3.6%** | **1.37** | **-1.5%** | **2.41** | 5 | 40% | 1.66\* | 0.009 |
| DOT/AAVE | 2.8% | 1.35 | -1.1% | 2.52 | 2 | 50% | 1.62 | 0.005 |
| DOT/BCH | 2.8% | 1.32 | -1.0% | 2.90 | 2 | 50% | 1.58 | 0.006 |
| UNI/FIL | 8.6% | 0.81 | -4.2% | 2.04 | 5 | 80% | 1.02 | 0.059 |
| ETH/NEAR | 2.1% | 0.51 | -3.5% | 0.60 | 4 | 25% | 0.74 | 0.040 |
| MATIC/ATOM | 2.1% | 0.28 | -11.8% | 0.18 | 2 | 50% | 0.36 | 0.027 |
| ETH/SOL | 0.6% | 0.13 | -4.5% | 0.14 | 3 | 33% | 0.26 | 0.052 |
| UNI/AAVE | 0.0% | 0.00 | 0.0% | — | 1 | 0% | — | 0.000 |
| DOT/FIL | -0.4% | -0.40 | -1.2% | -0.33 | 6 | 0% | -0.48 | -0.001 |
| LTC/UNI | -5.0% | -0.98 | -9.5% | -0.53 | 3 | 0% | -1.25 | 0.055 |

> \* p < 0.10  \*\* p < 0.05

**ADA/AAVE** is the new standout: Sharpe 1.85 with -1.0% max drawdown and an alpha t-stat of 2.20 (p = 0.028) — significant at 5% and not explained by BTC exposure (β = 0.004). Caveat: training-period performance was negative (Sharpe -0.89), so this only "worked" out-of-sample. Could be a regime shift in DeFi-vs-L1 dynamics in 2025; could be noise — wait for more out-of-sample bars before sizing aggressively.

**DOT/LINK** continues to hold up (Sharpe 1.37, alpha t-stat 1.66). Three of the top four pairs use DOT as the long leg, so their bets are highly correlated — don't naively stack capital across them.

Note: most pairs show negative in-sample Sharpe on training data and positive out-of-sample, which is the expected pattern for a stat-arb with no in-sample parameter tuning — it rules out look-ahead bias as the source of performance.

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
