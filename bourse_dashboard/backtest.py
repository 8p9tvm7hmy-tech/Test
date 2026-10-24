"""Backtest of the swing (نوسان‌گیری) signals on past daily data.

score_series mirrors analysis.swing_analysis: the technical rules always, plus the
money-flow and memory rules on days where the stored individual money flow
(memory) exists. Without memory only the technical part is tested.

Rules per symbol, one position at a time:
  * signal: score >= min_score at a day's close
  * entry:  next day's open (skipped if that day is a locked queue, high == low)
  * stop / target: as in the live analysis (ATR and 10-day low; target = k × risk)
  * exit:   stop first if both are touched the same day (conservative), gaps fill
            at the open, otherwise at the close after max_hold days
  * fees:   round-trip fee subtracted from every trade
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import indicators as ind


@dataclass
class BTParams:
    min_score: float = 65
    max_hold: int = 10            # روز معاملاتی
    target_r: float = 1.5         # هدف = چند برابر ریسک (۱.۵ = هدف ۱، ۳ = هدف ۲)
    fee_round_trip: float = 1.25  # ٪
    min_avg_value: float = 5e9
    lookback: int = 250           # فقط سیگنال‌های این تعداد روز اخیر


def score_series(df: pd.DataFrame, min_avg_value: float = 0) -> pd.Series:
    """Swing score for every day (df must contain indicators, optionally flow features)."""
    c, r = df["close"], df["rsi"]
    hist = df["macd_hist"]
    s = pd.Series(45.0, index=df.index)
    s += np.where(c > df["ema20"], 7, -7)
    s += np.where(df["ema20"] > df["ema50"], 7, -5)
    s += np.where(df["sma200"].isna(), 0, np.where(c > df["sma200"], 6, -4))

    rsi_pts = np.select(
        [(r >= 50) & (r < 68), (r >= 68) & (r < 75), r >= 75, r < 30],
        [8, 2, -8, np.where(c > df["high"].shift(), 5, -4)], 0)
    s += rsi_pts

    prev = hist.shift()
    s += np.where((hist > 0) & (hist > prev), 6, np.where((hist < 0) & (hist < prev), -6, 0))
    recent_nonpos = (hist.shift(1) <= 0) | (hist.shift(2) <= 0) | (hist.shift(3) <= 0)
    s += np.where(recent_nonpos & (hist > 0), 6, 0)

    vol_ratio = df["volume"] / df["vol_avg20"].replace(0, np.nan)
    day_chg = np.where(df["yesterday"] > 0, c / df["yesterday"] - 1, 0)
    s += np.where(vol_ratio >= 1.5, np.where(day_chg > 0, 8, -8), 0)
    s += np.where(c >= df["high"].shift().rolling(20).max(), 6, 0)
    s += np.where((c > df["bb_up"]) & (r < 75), 3, np.where((c < df["bb_low"]) & (r >= 30), -3, 0))
    s += np.where(df["value"].rolling(20).mean() < min_avg_value, -5, 0)
    if "bp" in df.columns:
        bp, ni = df["bp"], df["net_ind"]
        s += np.select([bp >= 1.5, bp >= 1.2, bp < 0.8], [8, 4, -6], 0)
        s += np.where(ni > 0, 3, np.where(ni < 0, -3, 0))
        mem = df["flow_days"] >= 5
        s += np.where(mem & (df["in_streak"] >= 3), 4, 0)
        s += np.where(mem & (df["out_streak"] >= 3), -4, 0)
        s += np.where(mem & (df["net5"] > 0) & (df["bp5"] >= 1.2), 3, 0)
    s = s.clip(0, 100)
    s.iloc[:59] = np.nan  # live analysis needs at least 60 days
    return s


OPEN = "open"  # trade_at result for a position still open at the end of the data


def prepare(df: pd.DataFrame, min_avg_value: float = 0) -> dict:
    """Numpy arrays used by trade_at (computed once per symbol)."""
    return {
        "score": score_series(df, min_avg_value).to_numpy(),
        **{k: df[k].to_numpy() for k in ("open", "high", "low", "close", "atr")},
        "low10": df["low"].rolling(10).min().to_numpy(),
        "dates": df.index,
    }


def trade_at(a: dict, i: int, max_hold: int, target_r: float):
    """Trade for a signal at day i's close: None (no fill), OPEN, or (e, j, entry, exit, reason)."""
    o, h, l, c, atr = a["open"], a["high"], a["low"], a["close"], a["atr"]
    n, e = len(c), i + 1
    if e >= n or np.isnan(atr[i]) or h[e] == l[e]:   # no next day, or a locked queue that won't fill
        return None
    stop = max(c[i] - 2 * atr[i], a["low10"][i])
    if stop >= c[i]:
        stop = c[i] - 1.5 * atr[i]
    entry = o[e]
    if entry <= stop:                                  # gapped below the stop
        return None
    target = entry + target_r * (c[i] - stop)
    for j in range(e, n):
        if l[j] <= stop:
            return e, j, entry, (min(o[j], stop) if j > e else stop), "حد ضرر"
        if h[j] >= target:
            return e, j, entry, (max(o[j], target) if j > e else target), "هدف"
        if j - e + 1 >= max_hold:
            return e, j, entry, c[j], "پایان مهلت"
    return OPEN


def simulate(df: pd.DataFrame, p: BTParams, symbol: str = "") -> list[dict]:
    if len(df) < 80:
        return []
    a = prepare(df, p.min_avg_value)
    score, dates = a["score"], a["dates"]
    trades, i = [], max(60, len(df) - p.lookback)
    while i < len(df) - 1:
        t = trade_at(a, i, p.max_hold, p.target_r) if score[i] >= p.min_score else None
        if t is None:
            i += 1
            continue
        if t == OPEN:
            break
        e, j, entry, exit_px, reason = t
        trades.append({
            "symbol": symbol, "signal_date": dates[i], "entry_date": dates[e], "exit_date": dates[j],
            "score": score[i], "entry": entry, "exit": exit_px, "days": j - e + 1,
            "reason": reason, "return_pct": (exit_px / entry - 1) * 100 - p.fee_round_trip,
        })
        i = j + 1
    return trades


def run(frames: dict, names: dict, p: BTParams, progress=None) -> pd.DataFrame:
    """frames: ins_code -> history with indicators. names: ins_code -> symbol."""
    trades, total = [], len(frames)
    for k, (code, df) in enumerate(frames.items(), 1):
        if "atr" not in df:
            df = ind.add_all(df)
        trades += simulate(df, p, names.get(code, code))
        if progress:
            progress(k, total)
    cols = ["symbol", "signal_date", "entry_date", "exit_date", "score", "entry", "exit",
            "days", "reason", "return_pct"]
    return pd.DataFrame(trades, columns=cols).sort_values("exit_date").reset_index(drop=True)


def summary(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {"trades": 0}
    r = trades["return_pct"]
    wins, losses = r[r > 0], r[r <= 0]
    return {
        "trades": len(r),
        "win_rate": len(wins) / len(r) * 100,
        "avg_return": r.mean(),
        "median_return": r.median(),
        "avg_win": wins.mean() if len(wins) else 0.0,
        "avg_loss": losses.mean() if len(losses) else 0.0,
        "profit_factor": wins.sum() / -losses.sum() if losses.sum() < 0 else np.inf,
        "avg_days": trades["days"].mean(),
    }


def by_score_bucket(trades: pd.DataFrame) -> pd.DataFrame:
    """Does a higher score lead to better trades?"""
    if trades.empty:
        return pd.DataFrame()
    bins = [0, 65, 70, 75, 80, 85, 90, 101]
    labels = ["<65", "65-70", "70-75", "75-80", "80-85", "85-90", "90+"]
    g = trades.groupby(pd.cut(trades["score"], bins, labels=labels, right=False), observed=True)
    out = g["return_pct"].agg(trades="count", avg_return="mean",
                              win_rate=lambda x: (x > 0).mean() * 100)
    return out.reset_index().rename(columns={"score": "bucket"})


def portfolio(trades: pd.DataFrame, position_pct: float = 20) -> pd.DataFrame:
    """Trades a real account could have taken: at most 100/position_pct positions at once.

    Signals on the same day are taken highest score first.
    """
    if trades.empty:
        return trades
    slots = max(1, int(100 // position_pct))
    open_exits, taken = [], []
    for idx, t in trades.sort_values(["entry_date", "score"], ascending=[True, False]).iterrows():
        open_exits = [d for d in open_exits if d >= t["entry_date"]]
        if len(open_exits) < slots:
            open_exits.append(t["exit_date"])
            taken.append(idx)
    return trades.loc[taken].sort_values("exit_date")


def equity_curve(trades: pd.DataFrame, position_pct: float = 20) -> pd.Series:
    """Growth of 100 for the portfolio() trades, each sized at position_pct% of the account."""
    taken = portfolio(trades, position_pct)
    if taken.empty:
        return pd.Series(dtype=float)
    growth = 1 + taken["return_pct"] / 100 * position_pct / 100
    return pd.Series(100 * growth.cumprod().to_numpy(), index=taken["exit_date"])


def max_drawdown(curve: pd.Series) -> float:
    if curve.empty:
        return 0.0
    return float((curve / curve.cummax() - 1).min() * 100)
