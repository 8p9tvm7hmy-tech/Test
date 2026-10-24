import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("BOURSE_HOME", str(tmp_path))
    return tmp_path


import alerts  # noqa: E402
import analysis as an  # noqa: E402
import codal  # noqa: E402
import config  # noqa: E402
import demo  # noqa: E402


def cell(r, c, v, span=1):
    return {"rowSequence": r, "columnSequence": c, "value": v, "colSpan": span}


def monthly_report_html():
    cells = [
        cell(1, 1, "نام محصول"), cell(1, 2, "از ابتدای سال مالی تا پایان ماه قبل", 3),
        cell(1, 5, "دوره يك ماهه منتهی به ۱۴۰۳/۰۶/۳۱", 3),
        cell(2, 2, "تعداد"), cell(2, 3, "نرخ"), cell(2, 4, "مبلغ فروش"),
        cell(2, 5, "تعداد"), cell(2, 6, "نرخ"), cell(2, 7, "مبلغ فروش (میلیون ریال)"),
        cell(3, 1, "ورق"), cell(3, 4, "۵۰۰"), cell(3, 7, "۱۲۰"),
        cell(4, 1, "جمع"), cell(4, 4, "۱,۰۰۰"), cell(4, 7, "۲۵۰,۵۰۰"),
    ]
    ds = {"sheets": [{"tables": [{"cells": cells}]}]}
    return f"<script>var datasource = {json.dumps(ds, ensure_ascii=False)};\nvar x=1;</script>"


def test_parse_monthly_total():
    ds = codal.extract_datasource(monthly_report_html())
    assert codal.parse_monthly_total(ds) == 250500


def test_parse_monthly_total_unknown_layout_returns_none():
    assert codal.parse_monthly_total({"sheets": [{"tables": [{"cells": [cell(1, 1, "x")]}]}]}) is None


def test_period_and_digits():
    assert codal.period_from_title("گزارش فعالیت ماهانه دوره ۱ ماهه منتهی به ۱۴۰۳/۰۶/۳۱") == "1403/06/31"
    assert codal._num("(۱,۲۰۰)") == -1200


def test_monthly_sales_from_letters(monkeypatch):
    letters = [
        {"Title": "گزارش فعالیت ماهانه دوره ۱ ماهه منتهی به ۱۴۰۳/۰۶/۳۱", "TracingNo": 2, "Url": "/r?2"},
        {"Title": "گزارش فعالیت ماهانه دوره ۱ ماهه منتهی به ۱۴۰۳/۰۵/۳۱", "TracingNo": 1, "Url": "/r?1"},
    ]
    monkeypatch.setattr(codal, "search_letters", lambda *a, **k: letters)

    class R:
        text = monthly_report_html()

    monkeypatch.setattr(codal, "_get", lambda *a, **k: R())
    df = codal.monthly_sales("فولاد")
    assert list(df["period"]) == ["1403/05/31", "1403/06/31"]
    assert codal.sales_growth(df)["mom"] == 0


def test_latest_letters_flags_important(monkeypatch):
    monkeypatch.setattr(codal, "search_letters", lambda *a, **k: [
        {"Title": "آگهی دعوت به مجمع عمومی", "PublishDateTime": "۱۴۰۳/۰۷/۰۱", "Url": "Reports/a", "TracingNo": 5},
        {"Title": "اطلاعیه عادی", "PublishDateTime": "۱۴۰۳/۰۷/۰۲", "Url": "", "TracingNo": 6},
    ])
    df = codal.latest_letters("فولاد")
    assert df["important"].tolist() == [True, False]
    assert df["url"].iloc[0] == "https://codal.ir/Reports/a" and df["date"].iloc[0] == "1403/07/01"


def test_config_roundtrip(home):
    cfg = config.load()
    assert cfg["alert_min_score"] == 80
    cfg["telegram_chat_id"] = "42"
    config.save(cfg)
    assert config.load()["telegram_chat_id"] == "42"
    assert (home / "settings.json").exists()


def test_market_hours():
    tz = alerts.TEHRAN
    assert alerts.market_open(datetime(2026, 10, 24, 10, 0, tzinfo=tz))      # Saturday
    assert not alerts.market_open(datetime(2026, 10, 29, 10, 0, tzinfo=tz))  # Thursday
    assert not alerts.market_open(datetime(2026, 10, 24, 14, 0, tzinfo=tz))


def test_run_once_sends_each_symbol_once_per_day(monkeypatch):
    sent = []
    monkeypatch.setattr(alerts, "send", lambda cfg, text: sent.append(text))
    market, hist = demo.demo_market(n_symbols=40, days=300)
    cfg = {**config.load(), "alert_min_score": 60, "min_avg_value": 0, "alert_codal_events": False}

    n1 = alerts.run_once(cfg, market=market, histories=hist, log=lambda *_: None)
    n2 = alerts.run_once(cfg, market=market, histories=hist, log=lambda *_: None)
    assert n1 > 1 and n2 == 0
    assert any("بلندمدت" in t for t in sent) and any("نوسان‌گیری" in t for t in sent)


def test_swing_message_escapes_html():
    swing, _, _ = an.scan(*demo.demo_market(n_symbols=10, days=300), an.Params())
    r = swing.iloc[0].copy()
    r["symbol"] = "<b>x"
    assert "&lt;b&gt;x" in alerts.swing_message(r)
