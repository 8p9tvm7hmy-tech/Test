"""Synthetic market used when TSETMC is unreachable (or for trying the dashboard).

Symbols are clearly fake ("نمونه۱", "نمونه۲", ...) so demo output can never be
mistaken for real advice.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis import session_date
from data import finalize_market
from sectors import SECTORS


def demo_market(n_symbols: int = 150, days: int = 500, seed: int = 7):
    rng = np.random.default_rng(seed)
    sectors = list(SECTORS)
    # Tehran trading calendar: Saturday–Wednesday, ending at the current session.
    cal = pd.date_range(end=session_date(), periods=days * 2, freq="D")
    dates = cal[~cal.weekday.isin([3, 4])][-days:]
    rows, histories = [], {}
    for i in range(n_symbols):
        code = f"DEMO{i:04d}"
        drift = rng.normal(0.0012, 0.0012)
        vol = rng.uniform(0.012, 0.035)
        rets = rng.normal(drift, vol, days)
        # Iranian daily price limit is roughly ±5%.
        rets = np.clip(rets, -0.05, 0.05)
        close = rng.uniform(800, 40000) * np.exp(np.cumsum(rets))
        open_ = close * np.exp(rng.normal(0, vol / 3, days))
        high = np.maximum(open_, close) * (1 + rng.uniform(0, vol, days))
        low = np.minimum(open_, close) * (1 - rng.uniform(0, vol, days))
        base_vol = rng.uniform(2e5, 2e7)
        volume = base_vol * rng.lognormal(0, 0.5, days) * (1 + 8 * np.abs(rets))
        hist = pd.DataFrame({
            "open": open_, "high": high, "low": low, "close": close, "last": close,
            "volume": volume, "value": volume * close, "count": volume / 5000,
        }, index=dates)
        hist["yesterday"] = hist["close"].shift().fillna(hist["open"])
        histories[code] = hist

        last_row, y = hist.iloc[-1], hist["yesterday"].iloc[-1]
        shares = rng.uniform(5e8, 5e10)
        eps = close[-1] / rng.uniform(3, 25) if rng.random() > 0.12 else -rng.uniform(10, 200)
        buy_ci, sell_ci = rng.integers(50, 3000, 2)
        tot = last_row["volume"]
        ind_buy, ind_sell = tot * rng.uniform(0.3, 1.0), tot * rng.uniform(0.3, 1.0)
        rows.append({
            "ins_code": code, "symbol": f"نمونه{i + 1}", "name": f"شرکت نمونه {i + 1}",
            "sector_code": sectors[i % len(sectors)],
            "first": last_row["open"], "close": last_row["close"], "last": last_row["close"],
            "yesterday": y, "low": last_row["low"], "high": last_row["high"],
            "count": last_row["count"], "volume": tot, "value": last_row["value"],
            "eps": eps, "shares": shares,
            "max_allowed": y * 1.05, "min_allowed": y * 0.95,
            "buy_count_i": buy_ci, "buy_count_n": rng.integers(0, 10),
            "buy_vol_i": ind_buy, "buy_vol_n": tot - ind_buy if tot > ind_buy else 0,
            "sell_count_i": sell_ci, "sell_count_n": rng.integers(0, 10),
            "sell_vol_i": ind_sell, "sell_vol_n": tot - ind_sell if tot > ind_sell else 0,
        })
    return finalize_market(pd.DataFrame(rows), filter_symbols=False), histories


def demo_flows(histories: dict, seed: int = 11) -> dict:
    """Synthetic individual money flow for the demo symbols (never written to memory)."""
    rng = np.random.default_rng(seed)
    out = {}
    for code, h in histories.items():
        ret = h["close"].pct_change().fillna(0).to_numpy()
        n = len(h)
        # Flow leans with the same day's move, like the real market — no look-ahead.
        tilt = np.clip(1 + 8 * ret + rng.normal(0, 0.15, n), 0.2, 2.0)
        vol = h["volume"].to_numpy()
        out[code] = pd.DataFrame({
            "buy_count_i": rng.integers(50, 3000, n).astype(float),
            "sell_count_i": rng.integers(50, 3000, n).astype(float),
            "buy_vol_i": vol * rng.uniform(0.4, 0.9, n) * tilt,
            "sell_vol_i": vol * rng.uniform(0.4, 0.9, n) / tilt,
        }, index=h.index)
    return out
