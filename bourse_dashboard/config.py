"""User settings and data folder.

The installed app lives in a read-only folder (e.g. Program Files), so the cache,
settings and alert history go to a per-user folder instead.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

APP_NAME = "BourseDashboard"

DEFAULTS = {
    "telegram_token": "",
    "telegram_chat_id": "",
    "telegram_proxy": "",          # e.g. socks5h://127.0.0.1:1080 or http://127.0.0.1:8080
    "alerts_enabled": False,
    "alert_min_score": 80,
    "alert_interval_min": 15,
    "alert_long_term": True,       # one daily message with the top long-term picks
    "alert_codal_events": True,    # important Codal announcements for alerted symbols
    "alert_outside_hours": False,
    "min_avg_value": 5e9,
    "max_symbols": 0,
}


def data_dir() -> Path:
    if os.environ.get("BOURSE_HOME"):
        d = Path(os.environ["BOURSE_HOME"])
    elif sys.platform == "win32":
        d = Path(os.environ.get("APPDATA", Path.home())) / APP_NAME
    elif sys.platform == "darwin":
        d = Path.home() / "Library" / "Application Support" / APP_NAME
    else:
        d = Path.home() / ".bourse_dashboard"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _path() -> Path:
    return data_dir() / "settings.json"


def load() -> dict:
    try:
        saved = json.loads(_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        saved = {}
    return {**DEFAULTS, **{k: v for k, v in saved.items() if k in DEFAULTS}}


def save(cfg: dict) -> None:
    clean = {k: cfg.get(k, v) for k, v in DEFAULTS.items()}
    tmp = _path().with_suffix(".tmp")
    tmp.write_text(json.dumps(clean, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(_path())
