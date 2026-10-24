"""Long-term memory: a local SQLite database in the user's data folder.

* flow:   daily individual (حقیقی) money flow per symbol. Backfilled from TSETMC's
          history on first run, then extended every session from the market watch.
* scores: the swing and long-term score of every symbol at each session, so the
          app can show how a symbol's condition evolved.
* kv:     small JSON values, e.g. the latest auto-tuning result.
"""
from __future__ import annotations

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import closing

import pandas as pd

import config
import data
from analysis import FLOW_COLS, session_date

SCHEMA = """
CREATE TABLE IF NOT EXISTS flow (
    date TEXT NOT NULL, ins_code TEXT NOT NULL,
    buy_count_i REAL, sell_count_i REAL, buy_vol_i REAL, sell_vol_i REAL,
    PRIMARY KEY (date, ins_code));
CREATE INDEX IF NOT EXISTS flow_code ON flow (ins_code, date);
CREATE TABLE IF NOT EXISTS scores (
    date TEXT NOT NULL, ins_code TEXT NOT NULL, symbol TEXT,
    swing_score REAL, lt_score REAL, close REAL,
    PRIMARY KEY (date, ins_code));
CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT);
"""


def _db() -> sqlite3.Connection:
    con = sqlite3.connect(config.data_dir() / "memory.db", timeout=30)
    con.execute("PRAGMA journal_mode=WAL")
    con.executescript(SCHEMA)
    return con


# ---------------------------------------------------------------------- flow
def save_flow(rows: pd.DataFrame) -> int:
    """rows: columns date (YYYY-MM-DD), ins_code and FLOW_COLS. Existing days are replaced."""
    rows = rows.dropna(subset=["buy_vol_i", "sell_vol_i"])
    if rows.empty:
        return 0
    recs = rows[["date", "ins_code", *FLOW_COLS]].itertuples(index=False, name=None)
    with closing(_db()) as con, con:
        con.executemany(f"INSERT OR REPLACE INTO flow VALUES ({','.join('?' * (2 + len(FLOW_COLS)))})", recs)
    return len(rows)


def save_market_flow(market: pd.DataFrame, day: pd.Timestamp | None = None) -> int:
    """Store the current session's money flow for every traded symbol."""
    day = (day or session_date()).strftime("%Y-%m-%d")
    traded = market[market["volume"] > 0]
    return save_flow(traded.assign(date=day)[["date", "ins_code", *FLOW_COLS]])


def last_flow_dates() -> dict:
    with closing(_db()) as con:
        return dict(con.execute("SELECT ins_code, MAX(date) FROM flow GROUP BY ins_code").fetchall())


def first_flow_dates() -> dict:
    with closing(_db()) as con:
        return dict(con.execute("SELECT ins_code, MIN(date) FROM flow GROUP BY ins_code").fetchall())


def needs_backfill(codes, max_gap_days: int = 5) -> list:
    """Symbols with no history in memory, or a gap the daily snapshots can't fill."""
    first, last = first_flow_dates(), last_flow_dates()
    today = session_date()
    out = []
    for c in codes:
        # A single snapshot day means the history was never downloaded.
        if c not in last or first[c] == last[c] or (today - pd.Timestamp(last[c])).days > max_gap_days:
            out.append(c)
    return out


def backfill(codes, workers: int = 6, progress=None) -> int:
    """Download the money-flow history of symbols that need it. Returns how many succeeded."""
    todo = needs_backfill(codes)
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(data.fetch_flow_history, c): c for c in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                save_flow(fut.result().assign(ins_code=futures[fut]))
                done += 1
            except Exception:  # noqa: BLE001 - one symbol must not stop the rest
                pass
            if progress:
                progress(i, len(todo))
    return done


def load_flows(codes=None) -> dict:
    """ins_code -> DataFrame (index: date) of stored money flow."""
    q, args = "SELECT * FROM flow", ()
    if codes is not None:
        codes = list(codes)
        if not codes:
            return {}
        q += f" WHERE ins_code IN ({','.join('?' * len(codes))})"
        args = codes
    with closing(_db()) as con:
        df = pd.read_sql_query(q, con, params=args)
    if df.empty:
        return {}
    df["date"] = pd.to_datetime(df["date"])
    return {c: g.drop(columns="ins_code").set_index("date").sort_index() for c, g in df.groupby("ins_code")}


# -------------------------------------------------------------------- scores
def save_scores(swing: pd.DataFrame, longterm: pd.DataFrame, day: pd.Timestamp | None = None) -> int:
    day = (day or session_date()).strftime("%Y-%m-%d")
    s = swing[["ins_code", "symbol", "score", "close"]] if not swing.empty else \
        pd.DataFrame(columns=["ins_code", "symbol", "score", "close"])
    lt = longterm[["ins_code", "lt_score"]] if not longterm.empty else pd.DataFrame(columns=["ins_code", "lt_score"])
    rows = s.merge(lt, on="ins_code", how="outer")
    if rows.empty:
        return 0
    recs = [(day, r.ins_code, r.symbol, r.score, r.lt_score, r.close) for r in rows.itertuples()]
    with closing(_db()) as con, con:
        con.executemany("INSERT OR REPLACE INTO scores VALUES (?,?,?,?,?,?)", recs)
    return len(recs)


def load_scores(ins_code: str) -> pd.DataFrame:
    with closing(_db()) as con:
        df = pd.read_sql_query("SELECT date, swing_score, lt_score, close FROM scores WHERE ins_code=? ORDER BY date",
                               con, params=(ins_code,))
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")


# ------------------------------------------------------------------------ kv
def get(key: str, default=None):
    with closing(_db()) as con:
        row = con.execute("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
    return json.loads(row[0]) if row else default


def put(key: str, value) -> None:
    with closing(_db()) as con, con:
        con.execute("INSERT OR REPLACE INTO kv VALUES (?,?)", (key, json.dumps(value, ensure_ascii=False, default=str)))


def stats() -> dict:
    with closing(_db()) as con:
        f = con.execute("SELECT COUNT(DISTINCT ins_code), COUNT(DISTINCT date), MIN(date), MAX(date) FROM flow").fetchone()
        s = con.execute("SELECT COUNT(DISTINCT date) FROM scores").fetchone()
    return {"symbols": f[0], "flow_days": f[1], "first": f[2], "last": f[3], "score_days": s[0]}


def refresh(market: pd.DataFrame, codes, progress=None) -> dict:
    """Daily update: store today's flow, backfill missing history, return flows for codes."""
    save_market_flow(market)
    backfill(codes, progress=progress)
    return load_flows(codes)
