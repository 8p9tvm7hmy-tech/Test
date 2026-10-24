"""Telegram alerts.

Sends a message when a symbol reaches a strong short-term buy signal, plus one
daily list of top long-term picks. Each symbol is alerted at most once a day.

Can run inside the dashboard (background thread) or on its own:
    python alerts.py            # watch during market hours
    python alerts.py --once     # one scan now
"""
from __future__ import annotations

import argparse
import html
import json
import threading
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import requests

import analysis as an
import config
import data

TEHRAN = ZoneInfo("Asia/Tehran")
API = "https://api.telegram.org/bot{token}/{method}"


class TelegramError(RuntimeError):
    pass


# ------------------------------------------------------------------ telegram
def _call(cfg: dict, method: str, **payload):
    token = (cfg.get("telegram_token") or "").strip()
    if not token:
        raise TelegramError("توکن ربات وارد نشده است")
    proxy = (cfg.get("telegram_proxy") or "").strip()
    proxies = {"http": proxy, "https": proxy} if proxy else None
    try:
        r = requests.post(API.format(token=token, method=method), json=payload,
                          proxies=proxies, timeout=20)
        body = r.json()
    except (requests.RequestException, ValueError) as e:
        raise TelegramError(f"اتصال به تلگرام ممکن نشد: {e}") from e
    if not body.get("ok"):
        raise TelegramError(f"خطای تلگرام: {body.get('description', r.status_code)}")
    return body["result"]


def send(cfg: dict, text: str) -> None:
    chat = str(cfg.get("telegram_chat_id") or "").strip()
    if not chat:
        raise TelegramError("شناسه چت (chat id) وارد نشده است")
    # Telegram rejects messages over 4096 characters.
    for i in range(0, len(text), 4000):
        _call(cfg, "sendMessage", chat_id=chat, text=text[i:i + 4000],
              parse_mode="HTML", disable_web_page_preview=True)


def find_chat_ids(cfg: dict) -> list[tuple[str, str]]:
    """Chats that recently messaged the bot: [(chat_id, name)]."""
    seen = {}
    for u in _call(cfg, "getUpdates"):
        msg = u.get("message") or u.get("channel_post") or {}
        chat = msg.get("chat") or {}
        if "id" in chat:
            name = chat.get("title") or " ".join(
                x for x in (chat.get("first_name"), chat.get("last_name")) if x) or chat.get("username", "")
            seen[str(chat["id"])] = name
    return list(seen.items())


# ------------------------------------------------------------------ messages
def fmt(x, d=0):
    return f"{x:,.{d}f}"


def swing_message(r: pd.Series) -> str:
    reasons = "\n".join(html.escape(x) for x in r["reasons"][:5])
    return (
        f"⚡ <b>سیگنال نوسان‌گیری: {html.escape(r['symbol'])}</b> ({html.escape(r['sector'])})\n"
        f"امتیاز: <b>{r['score']:.0f}</b> · {r['signal']}\n"
        f"قیمت: {fmt(r['close'])} ({r['change_pct']:+.2f}٪)\n"
        f"🎯 هدف ۱: {fmt(r['target1'])} · هدف ۲: {fmt(r['target2'])}\n"
        f"🛑 حد ضرر: {fmt(r['stop'])} ({r['stop_pct']:.1f}٪)\n"
        f"تعداد پیشنهادی: {fmt(r['shares'])} سهم\n\n{reasons}"
    )


def long_term_message(lt: pd.DataFrame, n: int = 10) -> str:
    top = lt[lt["eps"] > 0].head(n)
    lines = [f"{i}. <b>{html.escape(r.symbol)}</b> · امتیاز {r.lt_score:.0f} · P/E {r.pe:.1f}"
             f" · پتانسیل {r.upside:.0f}٪" for i, r in enumerate(top.itertuples(), 1)]
    return "🏦 <b>برترین گزینه‌های بلندمدت امروز</b>\n" + "\n".join(lines)


DISCLAIMER = "\n\n<i>مشاوره مالی نیست؛ پیش از خرید بررسی کنید.</i>"


# --------------------------------------------------------------------- state
def _state_path():
    return config.data_dir() / "alerts_state.json"


def _load_state() -> dict:
    today = datetime.now(TEHRAN).strftime("%Y-%m-%d")
    try:
        s = json.loads(_state_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        s = {}
    if s.get("date") != today:
        s = {"date": today, "swing": [], "long_term_sent": False, "codal": []}
    return s


def _save_state(s: dict) -> None:
    _state_path().write_text(json.dumps(s, ensure_ascii=False), encoding="utf-8")


def market_open(now: datetime | None = None) -> bool:
    """Tehran market: Saturday–Wednesday, roughly 09:00–12:30 (plus a few minutes for closing prices)."""
    now = now or datetime.now(TEHRAN)
    # Python weekday: Mon=0 … Sat=5, Sun=6. Thursday (3) and Friday (4) are closed.
    if now.weekday() in (3, 4):
        return False
    minutes = now.hour * 60 + now.minute
    return 9 * 60 <= minutes <= 12 * 60 + 45


# ---------------------------------------------------------------------- scan
def pick_new_alerts(swing: pd.DataFrame, cfg: dict, already: set) -> pd.DataFrame:
    if swing.empty:
        return swing
    s = swing[(swing["score"] >= cfg["alert_min_score"]) & (~swing["buy_queue"])
              & (swing["avg_value20"] >= cfg["min_avg_value"])]
    return s[~s["ins_code"].isin(already)]


def codal_events(symbols) -> list[str]:
    import codal
    out = []
    for sym in symbols:
        try:
            letters = codal.latest_letters(sym, limit=10)
        except Exception:  # noqa: BLE001 - Codal being down must not stop alerts
            continue
        for r in letters[letters["important"]].itertuples():
            out.append((sym, r.title, r.url, r.tracing_no))
    return out


def run_once(cfg: dict | None = None, market=None, histories=None, log=print) -> int:
    """One scan; returns the number of messages sent."""
    cfg = cfg or config.load()
    if market is None:
        market, histories = data.load_market_and_histories(int(cfg.get("max_symbols") or 0))
    params = an.Params(min_avg_value=cfg["min_avg_value"])
    swing, lt, _ = an.scan(market, histories, params)
    state, sent = _load_state(), 0

    new = pick_new_alerts(swing, cfg, set(state["swing"]))
    for _, r in new.iterrows():
        send(cfg, swing_message(r) + DISCLAIMER)
        state["swing"].append(r["ins_code"])
        sent += 1
        _save_state(state)

    if cfg.get("alert_codal_events") and not new.empty:
        for sym, title, url, tracing in codal_events(new["symbol"]):
            if tracing in state["codal"]:
                continue
            send(cfg, f"📄 <b>کدال · {html.escape(sym)}</b>\n{html.escape(title)}\n{url}")
            state["codal"].append(tracing)
            sent += 1
        _save_state(state)

    if cfg.get("alert_long_term") and not state["long_term_sent"] and not lt.empty:
        send(cfg, long_term_message(lt) + DISCLAIMER)
        state["long_term_sent"] = True
        sent += 1
        _save_state(state)
    log(f"{datetime.now(TEHRAN):%H:%M} scan done: {len(swing)} symbols, {sent} messages")
    return sent


class Watcher:
    """Background thread that scans periodically while alerts are enabled."""

    def __init__(self):
        self._thread = None
        self._stop = threading.Event()
        self.last_run = None
        self.last_error = None
        self.last_sent = 0

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self):
        if self.running:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="bourse-alerts")
        self._thread.start()

    def stop(self):
        self._stop.set()

    def _loop(self):
        while not self._stop.is_set():
            cfg = config.load()
            if not cfg["alerts_enabled"]:
                break
            if cfg["alert_outside_hours"] or market_open():
                try:
                    self.last_sent = run_once(cfg, log=lambda *_: None)
                    self.last_error = None
                except Exception as e:  # noqa: BLE001 - keep watching after a failure
                    self.last_error = str(e)
                self.last_run = datetime.now(TEHRAN)
            self._stop.wait(max(1, int(cfg["alert_interval_min"])) * 60)


def main():
    ap = argparse.ArgumentParser(description="Telegram alerts for the Tehran Stock Exchange")
    ap.add_argument("--once", action="store_true", help="run one scan and exit")
    args = ap.parse_args()
    if args.once:
        run_once()
        return
    while True:
        cfg = config.load()
        if cfg["alert_outside_hours"] or market_open():
            try:
                run_once(cfg)
            except Exception as e:  # noqa: BLE001
                print("error:", e)
        time.sleep(max(1, int(cfg["alert_interval_min"])) * 60)


if __name__ == "__main__":
    main()
