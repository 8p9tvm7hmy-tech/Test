"""Weekly auto-tuning of the swing strategy, checked out-of-sample.

1. Every combination of (minimum score, holding days, target multiple) is
   backtested on the training window (the older part of the recent history).
2. The best combination on training is then measured on the test window (the
   most recent weeks), which the search never saw.
3. It replaces the defaults only if, on the test window, it is profitable after
   fees and beats the defaults; otherwise the defaults stay. Either way the
   result is stored in memory.

Each signal is the day a symbol's score crosses up through the threshold;
entry, stop, target and exit follow backtest.trade_at exactly.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

import backtest as bt
import memory

GRID_SCORE = [55, 60, 65, 70, 75, 80, 85]
GRID_HOLD = [5, 10, 15]
GRID_TARGET = [1.0, 1.5, 2.0, 3.0]
DEFAULT = {"min_score": 65, "max_hold": 10, "target_r": 1.5}
MIN_TRAIN_TRADES = 50
MIN_TEST_TRADES = 15
KEY = "tuning"


def outcomes(frames: dict, fee: float, min_avg_value: float, lookback: int) -> pd.DataFrame:
    """Return of every threshold-crossing signal for every (hold, target) pair.

    Columns: date (signal day), score, prev (previous day's score), hold, target, ret.
    """
    rows = []
    lo = min(GRID_SCORE)
    for df in frames.values():
        if len(df) < 80:
            continue
        a = bt.prepare(df, min_avg_value)
        score = a["score"]
        prev = np.concatenate([[np.nan], score[:-1]])
        start = max(60, len(df) - lookback)
        # Days where at least one grid threshold is crossed upward.
        cand = [i for i in range(start, len(df) - 1)
                if score[i] >= lo and any(prev[i] < m <= score[i] or np.isnan(prev[i]) and m <= score[i]
                                          for m in GRID_SCORE)]
        for hold in GRID_HOLD:
            for tr in GRID_TARGET:
                for i in cand:
                    t = bt.trade_at(a, i, hold, tr)
                    if t is None or t == bt.OPEN:
                        continue
                    _, j, entry, exit_px, _ = t
                    rows.append((a["dates"][i], score[i], prev[i], hold, tr,
                                 (exit_px / entry - 1) * 100 - fee, a["dates"][j]))
    return pd.DataFrame(rows, columns=["date", "score", "prev", "hold", "target", "ret", "exit_date"])


def _stats(r: pd.Series) -> dict:
    return {"trades": int(len(r)), "avg": float(r.mean()) if len(r) else None,
            "win": float((r > 0).mean() * 100) if len(r) else None}


def tune(frames: dict, fee: float = 1.25, min_avg_value: float = 5e9,
         lookback: int = 500, test_days: int = 60, store: bool = True) -> dict:
    """store=False keeps the result out of memory (used for demo data)."""
    out = outcomes(frames, fee, min_avg_value, lookback)
    if out.empty:
        result = {"date": datetime.now().strftime("%Y-%m-%d"), "adopted": False, "params": DEFAULT,
                  "reason": "داده کافی برای تنظیم وجود نداشت؛ پارامترهای پیش‌فرض استفاده می‌شوند.", "grid": []}
        if store:
            memory.put(KEY, result)
        return result
    calendar = sorted(out["date"].unique())
    split = calendar[-test_days] if len(calendar) > test_days else calendar[len(calendar) // 2]

    grid = []
    for (hold, tr), g in out.groupby(["hold", "target"]):
        for m in GRID_SCORE:
            crossed = g[(g["score"] >= m) & ~(g["prev"] >= m)]
            # Train trades must also have closed before the test window starts.
            train = crossed[(crossed["date"] < split) & (crossed["exit_date"] < split)]["ret"]
            test = crossed[crossed["date"] >= split]["ret"]
            tr_s, te_s = _stats(train), _stats(test)
            grid.append({"min_score": m, "max_hold": int(hold), "target_r": float(tr),
                         "train_trades": tr_s["trades"], "train_avg": tr_s["avg"], "train_win": tr_s["win"],
                         "test_trades": te_s["trades"], "test_avg": te_s["avg"], "test_win": te_s["win"]})
    gdf = pd.DataFrame(grid)
    default = gdf[(gdf["min_score"] == DEFAULT["min_score"]) & (gdf["max_hold"] == DEFAULT["max_hold"])
                  & (gdf["target_r"] == DEFAULT["target_r"])].iloc[0].to_dict()
    eligible = gdf[gdf["train_trades"] >= MIN_TRAIN_TRADES].dropna(subset=["train_avg"])
    if eligible.empty:
        best, adopted = default, False
        reason = "هیچ ترکیبی معامله کافی در دوره آموزش نداشت؛ پارامترهای پیش‌فرض حفظ شد."
    else:
        best = eligible.sort_values("train_avg", ascending=False).iloc[0].to_dict()
        ok_trades = best["test_trades"] >= MIN_TEST_TRADES
        beats = best["test_avg"] is not None and default["test_avg"] is not None \
            and not np.isnan(best["test_avg"]) and best["test_avg"] >= default["test_avg"]
        profitable = beats and best["test_avg"] > 0
        adopted = bool(ok_trades and profitable)
        if adopted:
            reason = "بهترین ترکیب دوره آموزش در دوره آزمون (که در جستجو دیده نشده بود) هم از پیش‌فرض بهتر بود؛ اعمال شد."
        elif not ok_trades:
            reason = "بهترین ترکیب در دوره آزمون معامله کافی نداشت؛ پارامترهای پیش‌فرض حفظ شد."
        elif beats:
            reason = "بهترین ترکیب در دوره آزمون از پیش‌فرض بهتر بود ولی پس از کارمزد سود نداشت؛ پیش‌فرض حفظ شد."
        else:
            reason = "بهترین ترکیب دوره آموزش در دوره آزمون از پیش‌فرض بدتر بود (احتمال بیش‌برازش)؛ پیش‌فرض حفظ شد."
    params = {k: best[k] if adopted else DEFAULT[k] for k in DEFAULT}
    params = {"min_score": int(params["min_score"]), "max_hold": int(params["max_hold"]),
              "target_r": float(params["target_r"])}
    result = {
        "date": datetime.now().strftime("%Y-%m-%d"), "adopted": adopted, "reason": reason, "params": params,
        "best": best, "default": default, "split": str(pd.Timestamp(split).date()),
        "first": str(pd.Timestamp(calendar[0]).date()), "last": str(pd.Timestamp(calendar[-1]).date()),
        "grid": gdf.replace({np.nan: None}).to_dict("records"),
    }
    if store:
        memory.put(KEY, result)
    return result


def latest() -> dict | None:
    return memory.get(KEY)


def current_params() -> dict:
    """Parameters in use: the adopted tuning result, or the defaults."""
    r = latest()
    return dict(r["params"]) if r and r.get("params") else dict(DEFAULT)


def is_stale(days: int = 7) -> bool:
    r = latest()
    if not r:
        return True
    return (datetime.now() - datetime.strptime(r["date"], "%Y-%m-%d")).days >= days
