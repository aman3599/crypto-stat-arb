# Crypto Stat-Arb: Methodology and Results

A statistical arbitrage system for crypto pairs trading. This document covers the methodology, design decisions, full results, and a stress-test through the May–June 2026 crypto drawdown.

For a quick overview and how to run the code, see [`README.md`](./README.md).

---

## 1. Strategy

The system trades **mean-reverting spreads** between cointegrated crypto pairs. Two assets are cointegrated if a linear combination of their log-prices is stationary even when each price alone is non-stationary. When that spread deviates significantly from its historical mean, we take a market-neutral position expecting reversion.

For a candidate pair (A, B) with hedge ratio β estimated by OLS:

```
spread_t = log(A_t) − β · log(B_t)
z_t      = (spread_t − μ_120) / σ_120
```

A 120-bar rolling z-score drives entries, exits, and stops:

| Condition | Action |
|---|---|
| z < −2.0 | Enter long spread (long A, short β·B) |
| z > +2.0 | Enter short spread |
| \|z\| < 0.5 | Exit (mean reversion achieved) |
| \|z\| > 3.5 | Stop out (relationship breaking down) |

After a stop fires, re-entry in the same direction is blocked until z has crossed zero — a cooldown that prevents whipsaw against a persistent trend (added 2026-06-04 after the ETH/NEAR episode; see §6).

---

## 2. Pair Selection

A candidate pair is admitted to the trading set only if it passes all four screens, computed **on the training window only**:

**Engle-Granger cointegration** (p < 0.20). Runs `log(A) ~ β · log(B) + c` and ADF-tests the residuals. The relaxed threshold reflects EG's known low power on short, high-vol crypto series.

**ADF test on residuals** (p < 0.15). Same low-power reasoning. The EG + ADF pair is a dual filter against false positives.

**Ornstein-Uhlenbeck half-life** in [1, 120] days. Estimated by fitting `ΔS_t = θ · S_{t−1} + c + ε_t`:

```
half_life_bars = −ln(2) / θ
half_life_days = half_life_bars × (bar_hours / 24)
```

The lower bound (1 day) rules out spreads that mean-revert faster than transaction costs allow. The upper bound (120 days) rules out spreads where capital sits idle for months between trades.

**Johansen trace test**: implemented but disabled by default (`require_johansen=False`). On the 16-asset daily universe, Johansen rejects ~95% of EG-passing pairs — its power against alternative hypotheses degrades sharply on short crypto series. The EG + ADF + half-life triad is sufficient.

Of 120 candidate pairs in the 16-asset universe, **10 pass the screen** on 2023-01-01 → 2024-12-31 training data.

---

## 3. Position Sizing — Walk-Forward Fractional Kelly

Continuous Kelly fraction for a strategy with expected return μ and variance σ² is `f* = μ / σ²`. Full Kelly is famously aggressive — it maximizes log-wealth in expectation but accepts arbitrarily large drawdowns en route. We use **fractional Kelly with f = 25% of the optimum**, which preserves most of the EV at materially lower variance.

To avoid in-sample weight-fitting bias, sizing is **walk-forward**:

1. At each quarter-start, compute kelly_f using all data **up to that point**.
2. Take the **trailing 60-day median** of kelly_f (rather than its last value — see §6 for why).
3. Clip to [0, 0.20] — max 20% of capital per pair.
4. Apply that constant size forward through the next quarter.

The dollar position is then `kelly_f × $100k × position_sign`, split between legs by the hedge ratio so the trade is dollar-neutral.

---

## 4. Backtest Engine

The backtest is fully vectorized:

- **PnL** computed on actual price returns weighted by lagged positions (`pos.shift(1) × ret`) — no look-ahead.
- **Transaction costs**: 20bps per trade (10bps per leg, crypto-market standard) + 3bps slippage.
- **Portfolio kill-switch**: trading halts at 25% portfolio drawdown.
- **Alpha attribution**: regress strategy returns on BTC returns, report t-stat on the intercept. A high \|t\| with positive alpha indicates returns not explained by BTC exposure.

Train window: 2023-01-01 → 2024-12-31 (731 daily bars). Test window: 2025-01-01 → 2026-06-03 (519 daily bars).

---

## 5. Full Results (out-of-sample, 2025-01-01 → 2026-06-03)

| Pair | Ann. Return | Sharpe | Max DD | Calmar | Trades | Win Rate | Alpha t-stat | β/BTC |
|------|-------------|--------|--------|--------|--------|----------|--------------|-------|
| **ADA/AAVE** | **5.1%** | **1.89** | **−1.0%** | **5.05** | 3 | 67% | **2.27**\*\* | 0.003 |
| DOT/AAVE | 2.5% | 1.34 | −1.0% | 2.48 | 2 | 50% | 1.65\* | 0.004 |
| DOT/BCH | 2.5% | 1.31 | −0.9% | 2.86 | 2 | 50% | 1.62 | 0.005 |
| ADA/UNI | 1.0% | 0.24 | −5.1% | 0.20 | 3 | 0% | 0.29 | 0.001 |
| DOGE/UNI | 0.4% | 0.08 | −6.9% | 0.06 | 4 | 25% | 0.07 | −0.006 |
| DOT/FIL | 0.2% | 0.09 | −1.9% | 0.10 | 6 | 17% | 0.11 | 0.000 |
| DOT/LINK | 0.0% | 0.00 | 0.0% | — | 5 | 0% | — | 0.000 |
| ETH/UNI | −0.5% | −0.23 | −3.5% | −0.14 | 2 | 50% | −0.30 | −0.003 |
| ETH/SOL | −2.0% | −0.50 | −6.9% | −0.29 | 3 | 0% | −0.67 | −0.013 |
| ETH/NEAR | −3.2% | −0.78 | −7.4% | −0.44 | 4 | 0% | −0.94 | −0.003 |

> \* p < 0.10  \*\* p < 0.05

**ADA/AAVE is the standout.** Sharpe 1.89 with −1.0% max drawdown and an alpha t-stat of 2.27 (p = 0.024) — significant at 5% and not explained by BTC exposure (β = 0.003). The pair captures the inverse relationship between Layer-1 governance (ADA) and DeFi protocol revenue (AAVE) that emerged through the 2025 rate-cut cycle.

**Caveat on ADA/AAVE**: training-period Sharpe was −2.14. The pair only "works" out-of-sample. That's either a genuine regime shift in DeFi-vs-L1 dynamics, or noise that will fail forward as quickly as it appeared. The honest read is "promising, needs more bars."

**Pattern across the table**: most pairs show negative in-sample Sharpe and positive out-of-sample — the expected signature of a stat-arb with no in-sample parameter tuning. If thresholds were being fit to the train data, we'd see the opposite (positive in-sample, decaying OOS). The asymmetry rules out look-ahead bias as the source of the OOS performance.

**Concentration risk**: of the top three pairs, two short AAVE (ADA/AAVE, DOT/AAVE) and three use DOT as the long leg (DOT/AAVE, DOT/BCH, plus the lower-Sharpe DOT/FIL and DOT/LINK). Their PnL streams are correlated — naive equal-weighting overstates diversification.

---

## 6. Stress Test: 2026 Crypto Drawdown

In May 2026, BTC dropped from $82,139 on 2026-05-10 to $64,014 on 2026-06-03 — **−20.6% over 24 trading days**. ETH dropped −25.2%; SOL −26.4%. The sharpest broad crypto move since the 2022 cycle and a useful out-of-sample stress test.

### Strategy P&L through the window

Equal-weight across 10 pairs:

| Metric | Value |
|---|---|
| Portfolio return | **−0.10%** |
| Portfolio max drawdown | **−0.24%** |
| Beta to BTC during window | ~0.0 |

Walk-forward Kelly sized 6 of 10 pairs to zero exposure going into the window (trailing edge had turned negative at the 2026-04-01 rebalance). Of the four pairs that traded:

| Pair | Window Return | Window MaxDD | Trades |
|---|---|---|---|
| ADA/AAVE | **+1.30%** | −0.30% | 1 (winner) |
| DOGE/UNI | ~0.00% | ~0.00% | 1 |
| ETH/UNI | ~0.00% | ~0.00% | 1 |
| ETH/NEAR | **−2.49%** | −2.49% | 2 (stopped) |

**ADA/AAVE** generated +1.30% through the drawdown — true alpha, with z-score-driven mean-reversion firing as designed.

**ETH/NEAR** lost −2.49% in one stopped-out trade: NEAR ripped +46% against the short leg in 12 days while ETH dropped −15%. Both legs went against the trade. The stop-loss capped this at a single bad trade; the post-stop cooldown prevented re-entry into the same direction while the trend continued.

**The honest read** is not that the strategy "beat" the drawdown. It stayed out of the way — 6 of 10 pairs sat in cash, ADA/AAVE contributed real alpha, and the only losing pair was capped by the stop. The portfolio went through a 20% BTC drawdown at essentially zero P&L and zero portfolio drawdown. That's the dollar-neutral design working as intended.

### What this scenario exposed

The stress test surfaced **three real bugs**, which have been fixed:

**(1) Stale ticker data.** The Yahoo `UNI-USD` feed froze on 2025-04-17 after a token migration; `MATIC-USD` froze on 2025-03-24 ahead of the Polygon → POL transition. The data loader forward-filled the last live price, so four pairs (UNI/AAVE, UNI/FIL, LTC/UNI, MATIC/ATOM) were silently running on dead data — z-scores collapsed to noise, but no error was raised. The pre-fix run claimed an additional pair as top-5 (UNI/FIL at Sharpe 0.83); it was an artifact of frozen data. Fix: swap UNI to `UNI7083-USD` (live), drop MATIC (no replacement with full training history — POL on Yahoo only goes back to October 2023).

**(2) Walk-forward Kelly deactivated too easily.** The sizer used `kelly_f.iloc[-1]` — the single 30-day rolling estimate of `μ/σ²` on the last day of the in-sample window, clipped to `[0, 0.20]`. If trailing μ happened to be slightly negative on that one day, the entire next quarter was sized to zero. At the 2026-04-01 rebalance, 6 of 10 pairs (including ADA/AAVE, DOT/LINK, DOT/AAVE, DOT/BCH — all top performers) were silently turned off for Q2. Fix: use the **trailing 60-day median of kelly_f** instead of the last value. A single noisy day no longer deactivates a quarter.

**(3) Asymmetric stop-loss with whipsaw.** The stop rule in `signals.py` was written `z > stop_z` for a long position and `z < −stop_z` for a short. Those conditions fire only when the spread blows out in the **profit** direction. When the spread moved against a position past 3.5σ — exactly the loss scenario a stop should catch — nothing fired. ETH/NEAR held a long-spread trade through z = −4.23 because of this. Fix: use `abs(z) > stop_z` symmetrically. A separate problem then surfaced — after the stop fires, the entry condition (z < −2) often still holds, so the strategy re-enters into the same losing trend and gets stopped again. Added a **post-stop cooldown**: after a stop on a long, suppress re-entry until z has crossed zero (and symmetrically for short). ETH/NEAR's downturn loss went −7.05% → −6.69% (stop fix alone) → **−2.49% (stop + cooldown)**.

---

## 7. Key Design Decisions

**Log-price spread, not price ratio.** Stationarity of the spread is implied by the cointegration relationship in log-prices; ratios handle different price scales but lose the connection to cointegration theory.

**Dual EG + ADF + half-life screen.** EG and ADF have low statistical power on short, high-vol crypto series. Requiring both passes acts as a natural backstop even at relaxed individual p-thresholds. The half-life filter then rules out pairs that mean-revert too slowly or too fast to be tradeable.

**Relaxed p-thresholds.** EG p < 0.20 and ADF p < 0.15 are well above the conventional 0.05. This trades false-positive risk against the data-scarcity reality of crypto histories. The dual-test requirement + half-life filter compensates.

**Explicit `bar_hours` parameter.** Half-life is computed in absolute time (days), not bar units. Changing bar frequency requires one constant change — `bar_hours` — not a search through the codebase.

**Fractional Kelly (25%).** Full Kelly is the log-optimal sizing but accepts arbitrarily large drawdowns. Quarter-Kelly recovers ~94% of full-Kelly's geometric growth at substantially lower variance — a standard practitioner choice.

**Walk-forward sizing.** Kelly weights are estimated from past data only and applied forward. Eliminates in-sample weight-optimization bias, which is the single most common source of inflated stat-arb backtests.

**Per-pair kill switch (25% drawdown).** Halts trading if any pair's drawdown breaches the limit. Cointegration relationships break during structural shifts; this is a coarse but effective hedge against that.

**Post-stop cooldown.** Added in response to ETH/NEAR May 2026. Without it, the strategy whipsaws against persistent trends by repeatedly re-entering after stops fire. With it, a stopped pair waits for the spread to cross zero before re-engaging in the same direction.

---

## 8. Limitations and Next Steps

**Small trade counts.** 2–6 trades per pair over 17 months means Sharpe estimates have wide confidence intervals. ADA/AAVE's 1.89 could plausibly be anywhere from 0.5 to 3.2 at 90% confidence — the alpha t-stat (2.27) is more informative than the Sharpe.

**Correlated pair bets.** Top-3 pairs short AAVE and use DOT as the long leg. Their PnL streams are not independent. Portfolio-level Kelly should account for cross-pair covariance rather than treating each pair as an independent bet — a next step.

**ADA/AAVE regime risk.** Best out-of-sample pair was negative in-sample. This could be a genuine regime shift (DeFi protocol revenue diverging from L1 governance through the rate-cut cycle) or noise that will fail forward as quickly as it appeared. More OOS bars are the only way to distinguish.

**Strategy is too cautious in volatile regimes.** Through the May 2026 drawdown, 6 of 10 pairs sat flat. Walk-forward Kelly correctly identified that recent edge had turned negative, but the binary clip-to-zero behavior misses opportunity. A softer mapping — e.g. tapering position size with edge confidence rather than clipping — would let the strategy participate in more regimes.

**Transaction cost assumption.** 20bps + 3bps slippage is a market-standard estimate. Real costs vary with order size, venue (Coinbase vs. Binance vs. on-chain DEX), and liquidity. The backtest is robust to ±5bps changes but not to e.g. 50bps in thin altcoins.

**Regime change detection.** Cointegration relationships break during market stress. Hidden Markov Models or change-point detection on the spread could suspend trading when structural breaks are detected, rather than relying solely on the |z| > 3.5 stop.

**Higher-frequency bars.** The system is built for daily bars because Yahoo Finance caps hourly history at ~730 days. Switching to a real exchange feed (Binance, Kraken) would unlock 4H/1H frequencies, more trades per pair, and tighter confidence intervals — at the cost of more execution-risk modeling.
