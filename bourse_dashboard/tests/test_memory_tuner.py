import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("BOURSE_HOME", str(tmp_path))
    return tmp_path


import alerts  # noqa: E402
import analysis as an  # noqa: E402
import demo  # noqa: E402
import memory  # noqa: E402
import tuner  # noqa: E402

T = pd.Timestamp


def test_session_date():
    assert an.session_date(T("2026-10-24 10:00")) == T("2026-10-24")   # Saturday, open
    assert an.session_date(T("2026-10-24 08:00")) == T("2026-10-21")   # Saturday before open → Wednesday
    assert an.session_date(T("2026-10-29 11:00")) == T("2026-10-28")   # Thursday → Wednesday
    assert an.session_date(T("2026-10-30 11:00")) == T("2026-10-28")   # Friday → Wednesday


def flow_rows(code, dates, buy=9.0):
    return pd.DataFrame({"date": dates, "ins_code": code, "buy_count_i": 10.0, "sell_count_i": 10.0,
                         "buy_vol_i": buy, "sell_vol_i": 5.0})


def test_flow_roundtrip_and_replace():
    memory.save_flow(flow_rows("A", ["2026-10-20", "2026-10-21"]))
    memory.save_flow(flow_rows("A", ["2026-10-21"], buy=1.0))      # same day again → replaced
    f = memory.load_flows(["A"])["A"]
    assert list(f.index.strftime("%Y-%m-%d")) == ["2026-10-20", "2026-10-21"]
    assert f["buy_vol_i"].tolist() == [9.0, 1.0]
    assert memory.stats()["symbols"] == 1 and memory.load_flows([]) == {}


def test_needs_backfill(monkeypatch):
    monkeypatch.setattr(memory, "session_date", lambda: T("2026-10-24"))
    memory.save_flow(flow_rows("FULL", ["2026-10-20", "2026-10-21"]))
    memory.save_flow(flow_rows("SNAP", ["2026-10-21"]))               # only a daily snapshot
    memory.save_flow(flow_rows("OLD", ["2026-09-01", "2026-09-02"]))  # long gap
    assert memory.needs_backfill(["FULL", "SNAP", "OLD", "NEW"]) == ["SNAP", "OLD", "NEW"]


def test_backfill_and_refresh(monkeypatch):
    monkeypatch.setattr(memory, "session_date", lambda: T("2026-10-24"))
    calls = []

    def fake_history(code):
        calls.append(code)
        if code == "BAD":
            raise RuntimeError("down")
        return flow_rows(code, ["2026-10-20", "2026-10-21"]).drop(columns="ins_code")

    monkeypatch.setattr(memory.data, "fetch_flow_history", fake_history)
    market = pd.DataFrame({"ins_code": ["A", "BAD"], "volume": [5.0, 5.0], "buy_count_i": 10.0,
                           "sell_count_i": 10.0, "buy_vol_i": 7.0, "sell_vol_i": 5.0})
    flows = memory.refresh(market, ["A", "BAD"])
    assert sorted(calls) == ["A", "BAD"]
    assert list(flows["A"].index.strftime("%Y-%m-%d")) == ["2026-10-20", "2026-10-21", "2026-10-24"]
    assert len(flows["BAD"]) == 1                 # today's snapshot kept even though backfill failed
    calls.clear()
    memory.refresh(market, ["A"])
    assert calls == []                            # already backfilled


def test_scores_and_kv():
    market, hist = demo.demo_market(n_symbols=10, days=300)
    swing, lt, _ = an.scan(market, hist, an.Params())
    assert memory.save_scores(swing, lt, T("2026-10-24")) == len(set(swing["ins_code"]) | set(lt["ins_code"]))
    code = swing["ins_code"].iloc[0]
    s = memory.load_scores(code)
    assert s["swing_score"].iloc[0] == pytest.approx(swing["score"].iloc[0])
    memory.put("x", {"a": 1})
    assert memory.get("x") == {"a": 1} and memory.get("missing", 5) == 5


def test_tune_stores_and_respects_out_of_sample(monkeypatch):
    # Synthetic outcomes: score>=80 is great in training but bad in the test window.
    dates = pd.date_range("2025-01-01", periods=200)
    rows = []
    for k, d in enumerate(dates):
        for hold in tuner.GRID_HOLD:
            for tr in tuner.GRID_TARGET:
                late = k >= 140
                rows.append((d, 90, 0, hold, tr, -3.0 if late else 3.0, d))
                rows.append((d, 66, 0, hold, tr, 0.5, d))
    fake = pd.DataFrame(rows, columns=["date", "score", "prev", "hold", "target", "ret", "exit_date"])
    monkeypatch.setattr(tuner, "outcomes", lambda *a, **k: fake)
    r = tuner.tune({}, test_days=60)
    assert r["best"]["min_score"] >= 70 and not r["adopted"]
    assert tuner.current_params() == tuner.DEFAULT and not tuner.is_stale()

    # When the best combination also wins out of sample it is adopted.
    fake.loc[fake["score"] == 90, "ret"] = 3.0
    r = tuner.tune({}, test_days=60)
    assert r["adopted"] and tuner.current_params()["min_score"] == r["best"]["min_score"]

    # Better than the default but still losing after fees out of sample → not adopted.
    fake.loc[fake["score"] == 66, "ret"] = -2.0
    fake.loc[(fake["score"] == 90) & (fake["date"] >= dates[140]), "ret"] = -0.5
    r = tuner.tune({}, test_days=60)
    assert not r["adopted"] and "کارمزد" in r["reason"]


def test_tune_on_demo_does_not_store_when_asked():
    market, hist = demo.demo_market(n_symbols=20, days=320)
    _, _, frames = an.scan(market, hist, an.Params(), demo.demo_flows(hist))
    r = tuner.tune(frames, min_avg_value=0, store=False)
    assert r["grid"] and tuner.latest() is None


def test_alert_threshold():
    cfg = {"use_tuned": True, "alert_min_score": 80}
    adopted = {"adopted": True, "params": {"min_score": 70}}
    assert alerts.alert_threshold(cfg, adopted) == 70
    assert alerts.alert_threshold(cfg, {"adopted": False, "params": {"min_score": 65}}) == 80
    assert alerts.alert_threshold({**cfg, "use_tuned": False}, adopted) == 80
