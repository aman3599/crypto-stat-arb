# Crypto Stat-Arb

End-to-end statistical arbitrage system for crypto pairs trading. Cointegration-screened pairs from a 16-asset universe (BTC, ETH, SOL, ADA, XRP, DOGE, DOT, AVAX, LINK, LTC, BCH, ATOM, NEAR, UNI, AAVE, FIL), walk-forward Kelly sizing, strict train/test separation, vectorized backtest with transaction costs.

For full methodology, results, and the May 2026 stress test, see [**paper.md**](./paper.md).

---

## Headline result

Out-of-sample, 2025-01-01 → 2026-06-03 (519 daily bars, $100k capital, walk-forward fractional Kelly):

| Pair | Sharpe | Max DD | Alpha t-stat (vs BTC) | β/BTC |
|---|---|---|---|---|
| **ADA/AAVE** | **1.89** | −1.0% | **2.27** (p = 0.024) | 0.003 |
| DOT/AAVE | 1.34 | −1.0% | 1.65 | 0.004 |
| DOT/BCH | 1.31 | −0.9% | 1.62 | 0.005 |

ADA/AAVE alpha t-stat is significant at the 5% level. β/BTC ~0 across the top pairs — the alpha is not BTC exposure in disguise.

**Stress test (BTC −20.6% drawdown, 2026-05-10 → 2026-06-03):** Equal-weight portfolio returned −0.10% with −0.24% max drawdown. ADA/AAVE generated +1.30% through the drawdown; ETH/NEAR was the only meaningful loser at −2.49%, capped by a stop-loss + cooldown that prevented whipsaw. See [paper.md §6](./paper.md#6-stress-test-2026-crypto-drawdown).

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

## Tech stack

`Python` · `statsmodels` · `pandas` · `NumPy` · `yfinance` · `matplotlib` · `pyarrow`

---

## Author

**Aman Syed** — [LinkedIn](https://linkedin.com/in/yourprofile) · [GitHub](https://github.com/yourusername)

Quantitative finance professional with experience in systematic trading (energy futures, stat-arb), index research (MSCI), and counterparty credit risk. Part of a broader series on applied quantitative finance.
