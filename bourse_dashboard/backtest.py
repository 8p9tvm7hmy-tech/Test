"""Backtest of the swing (نوسان‌گیری) signals on past daily data.

The live score also uses today's individual/institutional money flow, which
TSETMC doesn't provide historically, so the backtest uses the technical part of
the score only (technical_score_series mirrors analysis.swing_analysis).

Rules per symbol, one position at a time:
  * signal: technical score >= min_score at a day's close
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


def technical_score_series(df: pd.DataFrame, min_avg_value: float = 0) -> pd.Series:
    """Technical part of the swing score for every day (df must contain indicators)."""
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
    s = s.clip(0, 100)
    s.iloc[:59] = np.nan  # live analysis needs at least 60 days
    return s


def simulate(df: pd.DataFrame, p: BTParams, symbol: str = "") -> list[dict]:
    if len(df) < 80:
        return []
    score = technical_score_series(df, p.min_avg_value).to_numpy()
    o, h, l, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
    atr = df["atr"].to_numpy()
    low10 = df["low"].rolling(10).min().to_numpy()
    dates = df.index
    start = max(60, len(df) - p.lookback)
    trades, i = [], start
    while i < len(df) - 1:
        if not (score[i] >= p.min_score) or np.isnan(atr[i]):
            i += 1
            continue
        e = i + 1
        if h[e] == l[e]:          # locked queue: an order would not fill
            i += 1
            continue
        stop = max(c[i] - 2 * atr[i], low10[i])
        if stop >= c[i]:
            stop = c[i] - 1.5 * atr[i]
        entry = o[e]
        target = entry + p.target_r * (c[i] - stop)
        if entry <= stop:         # gapped below the stop: skip the trade
            i += 1
            continue
        exit_px, reason, j = None, "", e
        while j < len(df):
            if l[j] <= stop:
                exit_px, reason = (min(o[j], stop) if j > e else stop), "حد ضرر"
                break
            if h[j] >= target:
                exit_px, reason = (max(o[j], target) if j > e else target), "هدف"
                break
            if j - e + 1 >= p.max_hold:
                exit_px, reason = c[j], "پایان مهلت"
                break
            j += 1
        if exit_px is None:       # still open at the end of the data
            break
        ret = (exit_px / entry - 1) * 100 - p.fee_round_trip
        trades.append({
            "symbol": symbol, "signal_date": dates[i], "entry_date": dates[e], "exit_date": dates[j],
            "score": score[i], "entry": entry, "exit": exit_px, "days": j - e + 1,
            "reason": reason, "return_pct": ret,
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
