"""
signals.py — Z-score signal engine for crypto stat-arb.
Computes rolling z-score of the spread and generates entry/exit signals.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional


@dataclass
class SignalConfig:
    entry_z: float = 2.0        # z-score threshold to enter
    exit_z: float = 0.5         # z-score to exit (mean reversion)
    stop_z: float = 3.5         # stop-loss z-score
    lookback: int = 30          # rolling window for z-score (days)


@dataclass
class Signal:
    date: pd.Timestamp
    pair: tuple[str, str]
    zscore: float
    spread: float
    position: int               # +1 long spread, -1 short spread, 0 flat
    hedge_ratio: float


def compute_zscore(spread: pd.Series, lookback: int = 30) -> pd.Series:
    """Rolling z-score of the spread."""
    mu = spread.rolling(lookback).mean()
    sigma = spread.rolling(lookback).std()
    return (spread - mu) / sigma


def generate_signals(
    spread: pd.Series,
    config: SignalConfig = SignalConfig(),
) -> pd.DataFrame:
    """
    Generate long/short/flat signals from spread z-score.

    Position logic:
      - Long spread (+1):  z < -entry_z  (spread too low → expect reversion up)
      - Short spread (-1): z > +entry_z  (spread too high → expect reversion down)
      - Exit:              |z| < exit_z
      - Stop:              |z| > stop_z  (flip or exit)

    Returns DataFrame with columns: zscore, spread, position, trade_signal
    """
    zscore = compute_zscore(spread, config.lookback)

    positions = pd.Series(0, index=spread.index, dtype=int)
    current_pos = 0
    # Cooldown: after a stop-out we suppress re-entry in the SAME direction
    # until z has crossed back through zero. Prevents whipsaw against a
    # persistent trend (e.g. NEAR ripping vs ETH May–June 2026).
    blocked_long = False
    blocked_short = False

    for i in range(config.lookback, len(zscore)):
        z = zscore.iloc[i]
        if np.isnan(z):
            continue

        if blocked_long and z >= 0:
            blocked_long = False
        if blocked_short and z <= 0:
            blocked_short = False

        if current_pos == 0:
            if z < -config.entry_z and not blocked_long:
                current_pos = 1
            elif z > config.entry_z and not blocked_short:
                current_pos = -1

        elif current_pos == 1:
            if abs(z) < config.exit_z:
                current_pos = 0
            elif abs(z) > config.stop_z:
                current_pos = 0
                blocked_long = True

        elif current_pos == -1:
            if abs(z) < config.exit_z:
                current_pos = 0
            elif abs(z) > config.stop_z:
                current_pos = 0
                blocked_short = True

        positions.iloc[i] = current_pos

    signals_df = pd.DataFrame({
        "spread": spread,
        "zscore": zscore,
        "position": positions,
        "trade_signal": positions.diff().fillna(0).astype(int)
    })
    return signals_df


def get_current_signal(
    signals_df: pd.DataFrame,
    asset_a: str,
    asset_b: str,
    hedge_ratio: float,
) -> Signal:
    """Return the most recent signal for live monitoring."""
    last = signals_df.iloc[-1]
    return Signal(
        date=signals_df.index[-1],
        pair=(asset_a, asset_b),
        zscore=round(last["zscore"], 3),
        spread=round(last["spread"], 6),
        position=int(last["position"]),
        hedge_ratio=hedge_ratio,
    )
