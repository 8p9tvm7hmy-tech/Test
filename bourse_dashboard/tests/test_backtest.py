import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import analysis as an  # noqa: E402
import backtest as bt  # noqa: E402
import demo  # noqa: E402
import indicators as ind  # noqa: E402


def test_score_series_matches_live_analysis():
    """The backtest must test the same rules the dashboard uses (minus money flow)."""
    _, hist = demo.demo_market(n_symbols=15, days=320)
    p = an.Params()
    no_flow = pd.Series({"buyer_power": np.nan, "net_individual": np.nan,
                         "buy_queue": False, "sell_queue": False})
    for df in hist.values():
        d = ind.add_all(df)
        s = bt.technical_score_series(d, p.min_avg_value)
        for k in (60, 200, len(d)):
            assert s.iloc[k - 1] == pytest.approx(an.swing_analysis(d.iloc[:k], no_flow, p)["score"])


def frame(n=100):
    idx = pd.bdate_range("2024-01-01", periods=n)
    c = np.full(n, 1000.0)
    df = pd.DataFrame({"open": c, "high": c + 10, "low": c - 10, "close": c, "yesterday": c,
                       "volume": 1e6, "value": 1e10}, index=idx)
    # Lows before the signal set the 10-day-low stop (990); later lows stay above it.
    df.iloc[70:, df.columns.get_loc("low")] = 995
    return df


def run_one(df, monkeypatch, signal_day, **kw):
    """Force a single signal on signal_day and simulate with fixed ATR."""
    df = ind.add_all(df)
    df["atr"] = 20.0
    score = pd.Series(0.0, index=df.index)
    score.iloc[signal_day] = 90
    monkeypatch.setattr(bt, "technical_score_series", lambda d, m=0: score)
    p = bt.BTParams(min_score=80, fee_round_trip=1.0, lookback=1000, **kw)
    return bt.simulate(df, p, "X")


def test_target_hit(monkeypatch):
    df = frame()
    df.iloc[71, df.columns.get_loc("high")] = 1100   # day after entry rallies
    t = run_one(df, monkeypatch, 69)
    # stop = max(1000 - 40, low10 = 990) = 990 → risk 10 → target = 1000 + 15
    assert t[0]["reason"] == "هدف" and t[0]["exit"] == pytest.approx(1015)
    assert t[0]["return_pct"] == pytest.approx(1.5 - 1.0)


def test_stop_wins_when_both_touched(monkeypatch):
    df = frame()
    df.iloc[71, df.columns.get_loc("high")] = 1100
    df.iloc[71, df.columns.get_loc("low")] = 900
    t = run_one(df, monkeypatch, 69)
    assert t[0]["reason"] == "حد ضرر" and t[0]["exit"] == pytest.approx(990)


def test_gap_down_fills_at_open(monkeypatch):
    df = frame()
    for col, v in (("open", 950), ("high", 960), ("low", 940), ("close", 950)):
        df.iloc[72, df.columns.get_loc(col)] = v
    t = run_one(df, monkeypatch, 69)
    assert t[0]["exit"] == pytest.approx(950)


def test_time_exit_and_queue_skip(monkeypatch):
    df = frame()
    t = run_one(df, monkeypatch, 69, max_hold=5)
    assert t[0]["reason"] == "پایان مهلت" and t[0]["days"] == 5
    df = frame()
    df.iloc[70, df.columns.get_loc("high")] = 1000
    df.iloc[70, df.columns.get_loc("low")] = 1000     # locked queue on entry day
    assert run_one(df, monkeypatch, 69) == []


def test_run_and_summary_on_demo():
    market, hist = demo.demo_market(n_symbols=20, days=300)
    trades = bt.run(hist, dict(zip(market["ins_code"], market["symbol"])), bt.BTParams(min_avg_value=0))
    s = bt.summary(trades)
    assert s["trades"] > 0 and 0 <= s["win_rate"] <= 100
    assert (trades["exit_date"] >= trades["entry_date"]).all()
    assert bt.max_drawdown(bt.equity_curve(trades)) <= 0
    assert bt.by_score_bucket(trades)["trades"].sum() == len(trades)


def test_portfolio_limits_concurrent_positions():
    d = pd.to_datetime
    trades = pd.DataFrame({
        "entry_date": d(["2024-01-01"] * 3 + ["2024-01-10"]),
        "exit_date": d(["2024-01-05"] * 3 + ["2024-01-12"]),
        "score": [70, 90, 80, 75], "return_pct": [1.0, 2.0, 3.0, 4.0],
    })
    taken = bt.portfolio(trades, position_pct=50)       # 2 slots
    assert sorted(taken["score"]) == [75, 80, 90]        # lowest score on the crowded day skipped
    assert bt.equity_curve(trades, 50).iloc[-1] == pytest.approx(100 * 1.01 * 1.015 * 1.02)
