# Crypto Stat-Arb

End-to-end statistical arbitrage system for crypto pairs trading. Cointegration-screened pairs traded with walk-forward fractional Kelly sizing, strict train/test separation, and a vectorized backtest engine with transaction costs.

For full methodology and design rationale, see [**paper.md**](./paper.md).

---

## Strategy

The system trades **mean-reverting spreads** between cointegrated crypto pairs. When the rolling z-score of a pair's log-price spread deviates significantly from zero, we take a dollar-neutral position expecting reversion.

```
spread_t = log(A_t) − β · log(B_t)
z_t      = (spread_t − μ_120) / σ_120
```

| Condition | Action |
|---|---|
| z < −2.0 | Enter long spread (long A, short β·B) |
| z > +2.0 | Enter short spread |
| \|z\| < 0.5 | Exit (mean reversion achieved) |
| \|z\| > 3.5 | Stop out (relationship breaking down) |

After a stop fires, re-entry in the same direction is suppressed until z crosses zero — a cooldown that prevents whipsaw against a persistent trend.

**Pair screening** (training window only): Engle-Granger cointegration (p < 0.20) + ADF on residuals (p < 0.15) + Ornstein-Uhlenbeck half-life in [1, 120] days. Of 120 candidate pairs in the 16-asset universe, **10 pass**.

**Sizing**: fractional Kelly at 25% of optimum, recomputed quarterly on past data only (walk-forward), clipped at 20% of capital per pair, applied dollar-neutrally across the two legs.

---

## Asset Universe

16 assets — large-caps + DeFi + L1 alts where Yahoo Finance provides full daily history from 2023-01-01:

`BTC` `ETH` `SOL` `ADA` `XRP` `DOGE` `DOT` `AVAX` `LINK` `LTC` `BCH` `ATOM` `NEAR` `UNI` `AAVE` `FIL`

UNI is sourced as `UNI7083-USD` (the legacy `UNI-USD` feed froze 2025-04-17 after a token migration). MATIC was removed when the Yahoo feed went stale and POL — its live replacement — only goes back to October 2023, which would truncate the training window.

---

## Results (out-of-sample, 2025-01-01 → 2026-06-03)

Test window: 519 daily bars (~17 months). $100k capital, walk-forward Kelly. Transaction costs: 20bps per trade + 3bps slippage. Alpha t-stat: intercept of `r_strategy = α + β·r_BTC + ε`.

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

**ADA/AAVE** is the standout: Sharpe 1.89 with −1.0% max drawdown and an alpha t-stat of **2.27 (p = 0.024)** — significant at 5% and not explained by BTC exposure (β = 0.003). Caveat: training-period Sharpe was −2.14, so this only "worked" out-of-sample. Could be a genuine regime shift in DeFi-vs-L1 dynamics through the 2025 rate-cut cycle; could be noise. Wait for more out-of-sample bars before sizing aggressively.

**Pattern across the table**: most pairs show negative in-sample Sharpe and positive out-of-sample — the expected signature of a stat-arb with no in-sample parameter tuning. If thresholds were being fit to train data, we'd see the opposite. The asymmetry rules out look-ahead bias as the source of OOS performance.

**Concentration**: top-3 pairs short AAVE in 2 of 3 cases and use DOT as the long leg in others — their PnL streams are correlated. Naive equal-weighting overstates diversification.

---

## Stress Test: 2026 Crypto Drawdown

In May 2026, BTC dropped **−20.6% over 24 trading days** (peak $82,139 on 2026-05-10 → $64,014 on 2026-06-03). ETH dropped −25.2%; SOL −26.4%. The sharpest broad crypto move since the 2022 cycle and a useful out-of-sample stress test.

**Strategy P&L through the window (equal-weight across 10 pairs):**

| Metric | Value |
|---|---|
| Portfolio return | **−0.10%** |
| Portfolio max drawdown | **−0.24%** |
| Beta to BTC during window | ~0.0 |

Walk-forward Kelly sized 6 of 10 pairs to zero going into the window (trailing edge had turned negative at the 2026-04-01 rebalance). Of the four pairs that traded:

- **ADA/AAVE: +1.30%** through the drawdown (window Sharpe 5.45, MaxDD −0.30%) — true alpha, with z-score-driven mean reversion firing as designed.
- **ETH/NEAR: −2.49%** in a single stopped-out trade — NEAR ripped +46% against the short leg in 12 days while ETH dropped −15%. The stop-loss + post-stop cooldown capped this at one bad trade; without the cooldown the strategy would have whipsawed in 8 more times and lost ~−7%.
- Two UNI-leg pairs traded small positions, roughly flat.

The portfolio went through a 20% BTC drawdown at essentially zero P&L and zero portfolio drawdown — the dollar-neutral design working as intended. The honest read is not that the strategy "beat" the drawdown; it stayed out of the way, with one genuine alpha trade (ADA/AAVE) offsetting one losing trade (ETH/NEAR).

The stress test surfaced three real bugs which have been fixed (see [paper.md §6](./paper.md#6-stress-test-2026-crypto-drawdown)):

1. **Stale ticker data** — UNI-USD and MATIC-USD froze in early 2025; the cache forward-filled stale prices, so four pairs were silently running on dead data.
2. **Walk-forward Kelly deactivated too easily** — a single noisy day at quarter-end was clipping the next quarter's exposure to zero.
3. **Asymmetric stop-loss with whipsaw** — the stop only fired in the profit direction; the fix added a symmetric stop and a post-stop cooldown.

---

## Architecture

```
data.py            yfinance fetch (daily bars), parquet caching
cointegration.py   Engle-Granger + Johansen screening, OU half-life, hedge ratios
signals.py         Rolling z-score engine, entry/exit/stop with post-stop cooldown
sizing.py          Fractional Kelly (25%) position sizing
backtest.py        Vectorized PnL engine, walk-forward backtest, alpha t-stat
main.py            Pipeline orchestrator (CLI)
diagnose_pairs.py  Per-pair diagnostic + threshold relaxation ladder
```

---

## Quick start

```bash
pip install -r requirements.txt

python main.py             # run pipeline with cached data
python main.py --refresh   # re-fetch data from yfinance
python main.py --plot      # generate equity curve plots
python main.py --live      # print current live signals only

python diagnose_pairs.py   # per-pair screening breakdown
```

---

## Tech stack

`Python` · `statsmodels` · `pandas` · `NumPy` · `yfinance` · `matplotlib` · `pyarrow`

---

## Author

**Aman Syed** — [LinkedIn](https://linkedin.com/in/yourprofile) · [GitHub](https://github.com/yourusername)

Quantitative finance professional with experience in systematic trading (energy futures, stat-arb), index research (MSCI), and counterparty credit risk. Part of a broader series on applied quantitative finance.
