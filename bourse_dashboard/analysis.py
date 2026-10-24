"""Scoring logic.

* swing_* : short-term technical + money-flow view (نوسان‌گیری)
* long_*  : fundamental/valuation view for long-term holding (سرمایه‌گذاری بلندمدت)

Scores are 0-100 rules of thumb, not predictions.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import indicators as ind

TEHRAN_TZ = "Asia/Tehran"


@dataclass
class Params:
    min_avg_value: float = 5e9          # ریال؛ حداقل میانگین ارزش معاملات ۲۰ روزه
    risk_free: float = 30.0             # ٪ سالانه، نرخ بدون ریسک (سود سپرده/اوراق)
    fee_round_trip: float = 1.25        # ٪ کارمزد خرید + فروش + مالیات
    capital: float = 1e9                # ریال
    risk_per_trade: float = 1.0         # ٪ از سرمایه در هر معامله


def merge_today(hist: pd.DataFrame, row: pd.Series) -> pd.DataFrame:
    """Append today's market-watch row when the daily history doesn't have it yet."""
    if hist.empty or not (row.get("volume", 0) > 0):
        return hist
    today = pd.Timestamp.now(tz=TEHRAN_TZ).tz_localize(None).normalize()
    if hist.index[-1] >= today:
        return hist
    new = pd.DataFrame([{
        "open": row["first"] or row["yesterday"], "high": row["high"], "low": row["low"],
        "close": row["close"], "last": row["last"], "yesterday": row["yesterday"],
        "volume": row["volume"], "value": row["value"], "count": row["count"],
    }], index=[today])
    return pd.concat([hist, new])


def _signal(score: float) -> str:
    if score >= 80:
        return "خرید قوی"
    if score >= 65:
        return "خرید"
    if score >= 50:
        return "خنثی"
    if score >= 35:
        return "احتیاط"
    return "دوری/فروش"


# ------------------------------------------------------------------ swing
def swing_analysis(df: pd.DataFrame, row: pd.Series, p: Params) -> dict | None:
    """df must already contain indicators (indicators.add_all)."""
    if len(df) < 60:
        return None
    t, y = df.iloc[-1], df.iloc[-2]
    score, why = 45.0, []

    def add(points, text):
        nonlocal score
        score += points
        why.append(("+" if points > 0 else "−") + f" {text}")

    # Trend
    add(7, "قیمت بالای EMA20") if t.close > t.ema20 else add(-7, "قیمت زیر EMA20")
    add(7, "EMA20 بالای EMA50 (روند صعودی)") if t.ema20 > t.ema50 else add(-5, "EMA20 زیر EMA50")
    if not np.isnan(t.sma200):
        add(6, "بالای میانگین ۲۰۰ روزه") if t.close > t.sma200 else add(-4, "زیر میانگین ۲۰۰ روزه")

    # Momentum
    r = t.rsi
    if 50 <= r < 68:
        add(8, f"RSI سالم و صعودی ({r:.0f})")
    elif 68 <= r < 75:
        add(2, f"RSI نزدیک اشباع خرید ({r:.0f})")
    elif r >= 75:
        add(-8, f"اشباع خرید، RSI={r:.0f}")
    elif r < 30:
        if t.close > y.high:
            add(5, f"برگشت از اشباع فروش (RSI={r:.0f})")
        else:
            add(-4, f"اشباع فروش بدون نشانه برگشت (RSI={r:.0f})")

    hist3 = df["macd_hist"].iloc[-4:]
    if t.macd_hist > 0 and t.macd_hist > y.macd_hist:
        add(6, "هیستوگرام MACD مثبت و رو به رشد")
    elif t.macd_hist < 0 and t.macd_hist < y.macd_hist:
        add(-6, "هیستوگرام MACD منفی و رو به افت")
    if (hist3.iloc[:-1] <= 0).any() and t.macd_hist > 0:
        add(6, "کراس صعودی MACD در ۳ روز اخیر")

    # Volume & breakout
    vol_ratio = t.volume / t.vol_avg20 if t.vol_avg20 else np.nan
    day_chg = (t.close / t.yesterday - 1) * 100 if t.yesterday else 0
    if vol_ratio >= 1.5:
        add(8, f"حجم {vol_ratio:.1f} برابر میانگین با رشد قیمت") if day_chg > 0 else \
            add(-8, f"حجم {vol_ratio:.1f} برابر میانگین با افت قیمت")
    prior_high20 = df["high"].iloc[-21:-1].max()
    if t.close >= prior_high20:
        add(6, "شکست سقف ۲۰ روزه")
    if t.close > t.bb_up and r < 75:
        add(3, "خروج از باند بالای بولینگر (قدرت روند)")
    elif t.close < t.bb_low and r >= 30:
        add(-3, "زیر باند پایین بولینگر")

    # Money flow (today, from market watch)
    bp = row.get("buyer_power", np.nan)
    if not np.isnan(bp):
        if bp >= 1.5:
            add(8, f"قدرت خریدار حقیقی بالا ({bp:.2f})")
        elif bp >= 1.2:
            add(4, f"قدرت خریدار حقیقی مثبت ({bp:.2f})")
        elif bp < 0.8:
            add(-6, f"قدرت فروشنده غالب ({bp:.2f})")
    ni = row.get("net_individual", np.nan)
    if not np.isnan(ni) and ni != 0:
        add(3, "ورود پول حقیقی") if ni > 0 else add(-3, "خروج پول حقیقی")
    if row.get("sell_queue"):
        add(-10, "صف فروش")
    if row.get("buy_queue"):
        why.append("⚠ صف خرید؛ خرید ممکن است انجام نشود")

    avg_value = df["value"].iloc[-20:].mean()
    if avg_value < p.min_avg_value:
        add(-5, "نقدشوندگی پایین")

    score = float(np.clip(score, 0, 100))
    close, atr = float(t.close), float(t.atr)
    stop = max(close - 2 * atr, float(df["low"].iloc[-10:].min()))
    if stop >= close:
        stop = close - 1.5 * atr
    risk = close - stop
    t1, t2 = close + 1.5 * risk, close + 3 * risk
    shares = int(min(p.capital * p.risk_per_trade / 100 / risk, p.capital / close)) if risk > 0 else 0
    return {
        "score": round(score, 1), "signal": _signal(score), "reasons": why,
        "close": close, "change_pct": day_chg, "rsi": r, "vol_ratio": vol_ratio,
        "buyer_power": bp, "atr_pct": atr / close * 100,
        "entry": close, "stop": stop, "target1": t1, "target2": t2,
        "stop_pct": (stop / close - 1) * 100,
        "t1_net_pct": (t1 / close - 1) * 100 - p.fee_round_trip,
        "shares": shares, "position_value": shares * close,
        "avg_value20": avg_value, "resistance60": float(df["high"].iloc[-60:].max()),
    }


# -------------------------------------------------------------- long term
def history_metrics(df: pd.DataFrame) -> dict:
    c = df["close"]
    rets = c.pct_change().dropna()
    year = c.iloc[-250:]
    sma200 = ind.sma(c, 200).iloc[-1] if len(c) >= 200 else np.nan
    return {
        "ret_1y": (c.iloc[-1] / c.iloc[-250] - 1) * 100 if len(c) >= 250 else np.nan,
        "ret_6m": (c.iloc[-1] / c.iloc[-125] - 1) * 100 if len(c) >= 125 else np.nan,
        "vol_1y": rets.iloc[-250:].std() * np.sqrt(240) * 100 if len(rets) > 60 else np.nan,
        "drawdown": (c.iloc[-1] / year.max() - 1) * 100,
        "above_sma200": bool(c.iloc[-1] > sma200) if not np.isnan(sma200) else np.nan,
        "avg_value60": df["value"].iloc[-60:].mean(),
        "days": len(c),
    }


def long_term_table(market: pd.DataFrame, metrics: pd.DataFrame, p: Params) -> pd.DataFrame:
    df = market.merge(metrics, left_on="ins_code", right_index=True, how="inner")
    pos = df[df["pe"] > 0]
    market_pe = pos["pe"].median()
    sec = pos.groupby("sector")["pe"].agg(["median", "count"])
    sec_pe = sec["median"].where(sec["count"] >= 3, market_pe)
    df["sector_pe"] = df["sector"].map(sec_pe).fillna(market_pe)
    df["pe_rel"] = df["pe"] / df["sector_pe"]
    df["earnings_yield"] = np.where(df["pe"] > 0, 100 / df["pe"], np.nan)
    df["fair_price"] = np.where(df["eps"] > 0, df["eps"] * df["sector_pe"], np.nan)
    df["upside"] = (df["fair_price"] / df["close"] - 1) * 100

    def rank(s, higher_better=True):
        return s.rank(pct=True, ascending=higher_better).fillna(0.3)

    value = 0.6 * rank(df["earnings_yield"]) + 0.4 * rank(df["pe_rel"], higher_better=False)
    trend = 0.6 * rank(df["ret_1y"]) + 0.4 * df["above_sma200"].map({True: 1.0, False: 0.0}).fillna(0.3)
    stability = rank(df["vol_1y"], higher_better=False)
    size = 0.5 * rank(df["market_cap"]) + 0.5 * rank(df["avg_value60"])
    discount = rank(df["drawdown"], higher_better=False)
    score = 100 * (0.35 * value + 0.25 * trend + 0.15 * stability + 0.15 * size + 0.10 * discount)
    score = score + np.where(df["earnings_yield"] > p.risk_free, 5, 0)
    score = np.where(df["eps"] > 0, score, np.minimum(score, 30))
    score = np.where(df["avg_value60"] < p.min_avg_value, np.minimum(score, 45), score)
    df["lt_score"] = np.clip(score, 0, 100).round(1)
    df["lt_signal"] = df["lt_score"].map(_signal)
    return df.sort_values("lt_score", ascending=False)


def long_term_reasons(r: pd.Series, p: Params) -> list[str]:
    why = []
    if not r["eps"] > 0:
        return ["− شرکت زیان‌ده است یا EPS گزارش نشده"]
    why.append(f"{'+' if r['pe_rel'] < 1 else '−'} P/E برابر {r['pe']:.1f} در برابر میانه گروه {r['sector_pe']:.1f}")
    ey = r["earnings_yield"]
    why.append(f"{'+' if ey > p.risk_free else '−'} بازده سود {ey:.0f}٪ در برابر نرخ بدون ریسک {p.risk_free:.0f}٪")
    if not np.isnan(r["upside"]):
        why.append(f"{'+' if r['upside'] > 0 else '−'} فاصله تا ارزش نسبی (EPS × P/E گروه): {r['upside']:.0f}٪")
    if not np.isnan(r["ret_1y"]):
        why.append(f"{'+' if r['ret_1y'] > 0 else '−'} بازده یک‌ساله {r['ret_1y']:.0f}٪")
    if r["above_sma200"] is True:
        why.append("+ روند بلندمدت صعودی (بالای میانگین ۲۰۰ روزه)")
    elif r["above_sma200"] is False:
        why.append("− زیر میانگین ۲۰۰ روزه")
    if r["avg_value60"] < p.min_avg_value:
        why.append("− نقدشوندگی پایین")
    return why


# ----------------------------------------------------------------- scan
def scan(market: pd.DataFrame, histories: dict, p: Params):
    """Run both analyses on every symbol that has a history."""
    swings, metrics, frames = [], {}, {}
    rows = market.set_index("ins_code")
    for code, hist in histories.items():
        if code not in rows.index or hist.empty:
            continue
        row = rows.loc[code]
        h = ind.add_all(merge_today(hist, row))
        frames[code] = h
        metrics[code] = history_metrics(h)
        s = swing_analysis(h, row, p)
        if s:
            swings.append({"ins_code": code, "symbol": row["symbol"], "name": row["name"],
                           "sector": row["sector"], "buy_queue": row["buy_queue"], **s})
    swing_df = pd.DataFrame(swings)
    if not swing_df.empty:
        swing_df = swing_df.sort_values("score", ascending=False)
    lt_df = long_term_table(market, pd.DataFrame(metrics).T.infer_objects(), p) if metrics else pd.DataFrame()
    return swing_df, lt_df, frames
