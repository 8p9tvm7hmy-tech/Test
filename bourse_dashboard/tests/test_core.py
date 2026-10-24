import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import analysis as an  # noqa: E402
import data  # noqa: E402
import demo  # noqa: E402
import indicators as ind  # noqa: E402


def test_rsi_bounds_and_trend():
    up = pd.Series(np.arange(1, 60, dtype=float))
    assert ind.rsi(up).iloc[-1] == 100
    down = pd.Series(np.arange(60, 1, -1, dtype=float))
    assert ind.rsi(down).iloc[-1] < 1


def test_adjust_prices_removes_capital_increase_gap():
    # Close 1000, then a 100% capital increase halves "yesterday" to 500.
    df = pd.DataFrame({
        "open": [1000, 1000, 500], "high": [1000, 1000, 510], "low": [1000, 1000, 490],
        "close": [1000.0, 1000.0, 505.0], "last": [1000, 1000, 505],
        "yesterday": [1000, 1000, 500], "volume": [10.0, 10.0, 20.0],
    }, index=pd.date_range("2024-01-01", periods=3))
    out = data.adjust_prices(df)
    assert out["close"].tolist() == pytest.approx([500, 500, 505])
    assert out["volume"].tolist() == pytest.approx([20, 20, 20])


def test_symbol_filter():
    assert data.is_tradeable_share("فولاد")
    assert not data.is_tradeable_share("فولادح")      # rights issue
    assert not data.is_tradeable_share("فولاد2")      # block market duplicate


def test_parse_old_market_watch(monkeypatch):
    row = ",".join(["123", "IRO1", "خودرو", "ایران خودرو", "0", "2000", "2050", "2100", "500",
                    "1000000", "2050000000", "1950", "2100", "2000", "300", "1", "0", "1", "34",
                    "2100", "1900", "5000000", "300"])
    other = row.replace(",300", ",400")  # not a share → dropped
    text = f"header@x@{row};{other}@limits"

    class R:
        status_code, content = 200, b"x"
        def __init__(self, t): self.text = t

    monkeypatch.setattr(data, "_get", lambda url, **k: R(text))
    df = data._market_watch_old()
    assert len(df) == 1 and df.iloc[0]["symbol"] == "خودرو"
    final = data.finalize_market(df.assign(**{c: np.nan for c in data.CLIENT_COLUMNS}))
    r = final.iloc[0]
    assert r["pe"] == pytest.approx(2050 / 300)
    assert r["buy_queue"] and r["sector"] == "خودرو و قطعات"


def test_parse_new_history(monkeypatch):
    payload = {"closingPriceDaily": [
        {"dEven": 20240102, "priceFirst": 100, "priceMax": 110, "priceMin": 95, "pClosing": 105,
         "pDrCotVal": 106, "priceYesterday": 100, "qTotTran5J": 1000, "qTotCap": 105000, "zTotTran": 9},
        {"dEven": 20240101, "priceFirst": 98, "priceMax": 101, "priceMin": 97, "pClosing": 100,
         "pDrCotVal": 100, "priceYesterday": 98, "qTotTran5J": 500, "qTotCap": 50000, "zTotTran": 4},
    ]}

    class R:
        status_code, content = 200, b"x"
        def json(self): return payload

    monkeypatch.setattr(data, "_get", lambda url, **k: R())
    h = data.clean_history(data._history_new("1"))
    assert list(h.index.strftime("%Y-%m-%d")) == ["2024-01-01", "2024-01-02"]
    assert h["close"].iloc[-1] == 105


def test_scan_on_demo_market():
    market, hist = demo.demo_market(n_symbols=40, days=300)
    swing, lt, frames = an.scan(market, hist, an.Params())
    assert len(frames) == 40 and not swing.empty and not lt.empty
    assert swing["score"].between(0, 100).all() and lt["lt_score"].between(0, 100).all()
    assert (swing["stop"] < swing["entry"]).all() and (swing["target1"] > swing["entry"]).all()
    # Loss-making companies are capped.
    assert (lt.loc[lt["eps"] <= 0, "lt_score"] <= 30).all()
