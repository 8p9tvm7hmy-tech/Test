"""Data access for the Tehran Stock Exchange (TSETMC).

Two TSETMC interfaces are supported, newest first:
  * cdn.tsetmc.com/api/...         (JSON, current site)
  * old.tsetmc.com/tsev2/data/...  (legacy text format)

Everything is normalised into two shapes:
  * market watch: one row per symbol for today
  * history: daily OHLCV for one symbol, adjusted for capital increases/dividends
"""
from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests

import config
from sectors import sector_name

NEW_API = "https://cdn.tsetmc.com/api"
OLD_API = "http://old.tsetmc.com/tsev2/data"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.tsetmc.com/",
}
CACHE_DIR = config.data_dir() / "cache"

# Instrument type codes (yval) that are ordinary shares: 300 بورس, 303 فرابورس, 309 پایه.
STOCK_TYPES = {"300", "303", "309"}

MARKET_COLUMNS = [
    "ins_code", "symbol", "name", "sector_code", "first", "close", "last", "yesterday",
    "low", "high", "count", "volume", "value", "eps", "shares", "max_allowed", "min_allowed",
]
CLIENT_COLUMNS = [
    "buy_count_i", "buy_count_n", "buy_vol_i", "buy_vol_n",
    "sell_count_i", "sell_count_n", "sell_vol_i", "sell_vol_n",
]


class DataError(RuntimeError):
    pass


_session = requests.Session()
_session.headers.update(HEADERS)


def _get(url: str, retries: int = 3, timeout: int = 20) -> requests.Response:
    last = None
    for attempt in range(retries):
        try:
            r = _session.get(url, timeout=timeout)
            if r.status_code == 200 and r.content:
                return r
            last = DataError(f"HTTP {r.status_code} from {url}")
        except requests.RequestException as e:
            last = e
        time.sleep(1.5 * (attempt + 1))
    raise DataError(f"دریافت داده ناموفق بود: {url} ({last})")


def _num(v):
    try:
        if isinstance(v, str):
            v = v.replace(",", "").strip()
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def _pick(d: dict, *keys):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def is_tradeable_share(symbol: str) -> bool:
    """Drop rights issues (…ح), block/odd-lot duplicates (symbols ending in digits)."""
    s = (symbol or "").strip()
    if not s or re.search(r"\d$", s):
        return False
    return not (len(s) > 2 and s.endswith("ح"))


# ---------------------------------------------------------------- market watch
def _market_watch_new() -> pd.DataFrame:
    url = (f"{NEW_API}/ClosingPrice/GetMarketWatch?market=0&industrialGroup="
           "&paperTypes%5B0%5D=1&paperTypes%5B1%5D=2&showTraded=false&withBestLimits=false")
    rows = _get(url).json().get("marketwatch") or []
    out = []
    for r in rows:
        eps = _pick(r, "eps", "estimatedEPS", "epsValue")
        if isinstance(eps, dict):
            eps = _pick(eps, "estimatedEPS", "epsValue")
        out.append({
            "ins_code": str(_pick(r, "insCode", "insID")),
            "symbol": (_pick(r, "lVal18AFC", "symbol") or "").strip(),
            "name": (_pick(r, "lVal30", "name") or "").strip(),
            "sector_code": str(_pick(r, "cSecVal", "sectorCode", "lSecVal") or "").strip(),
            "first": _num(_pick(r, "priceFirst", "pf")),
            "close": _num(_pick(r, "pClosing", "pcl")),
            "last": _num(_pick(r, "pDrCotVal", "pdv")),
            "yesterday": _num(_pick(r, "priceYesterday", "py")),
            "low": _num(_pick(r, "priceMin", "pmn")),
            "high": _num(_pick(r, "priceMax", "pmx")),
            "count": _num(_pick(r, "zTotTran", "tno")),
            "volume": _num(_pick(r, "qTotTran5J", "tvol")),
            "value": _num(_pick(r, "qTotCap", "tval")),
            "eps": _num(eps),
            "shares": _num(_pick(r, "zTitad", "z")),
            "max_allowed": _num(_pick(r, "psGelStaMax", "tmax")),
            "min_allowed": _num(_pick(r, "psGelStaMin", "tmin")),
        })
    df = pd.DataFrame(out, columns=MARKET_COLUMNS)
    if df.empty or df["close"].isna().all():
        raise DataError("پاسخ دیده‌بان بازار (API جدید) خالی یا ناشناخته بود")
    return df


def _market_watch_old() -> pd.DataFrame:
    text = _get(f"{OLD_API}/MarketWatchInit.aspx?h=0&r=0").text
    parts = text.split("@")
    if len(parts) < 3:
        raise DataError("قالب پاسخ دیده‌بان بازار (API قدیمی) ناشناخته بود")
    out = []
    for row in parts[2].split(";"):
        f = row.split(",")
        if len(f) < 23 or f[22] not in STOCK_TYPES:
            continue
        out.append({
            "ins_code": f[0], "symbol": f[2].strip(), "name": f[3].strip(),
            "sector_code": f[18].strip(),
            "first": _num(f[5]), "close": _num(f[6]), "last": _num(f[7]),
            "count": _num(f[8]), "volume": _num(f[9]), "value": _num(f[10]),
            "low": _num(f[11]), "high": _num(f[12]), "yesterday": _num(f[13]),
            "eps": _num(f[14]), "max_allowed": _num(f[19]), "min_allowed": _num(f[20]),
            "shares": _num(f[21]),
        })
    df = pd.DataFrame(out, columns=MARKET_COLUMNS)
    if df.empty:
        raise DataError("دیده‌بان بازار (API قدیمی) خالی بود")
    return df


def _client_types_new() -> pd.DataFrame:
    rows = _get(f"{NEW_API}/ClientType/GetClientTypeAll").json().get("clientTypeAllDto") or []
    keys = {
        "buy_count_i": "buy_CountI", "buy_count_n": "buy_CountN",
        "buy_vol_i": "buy_I_Volume", "buy_vol_n": "buy_N_Volume",
        "sell_count_i": "sell_CountI", "sell_count_n": "sell_CountN",
        "sell_vol_i": "sell_I_Volume", "sell_vol_n": "sell_N_Volume",
    }
    out = [{"ins_code": str(r.get("insCode")), **{k: _num(r.get(v)) for k, v in keys.items()}}
           for r in rows]
    if not out:
        raise DataError("داده حقیقی/حقوقی خالی بود")
    return pd.DataFrame(out)


def _client_types_old() -> pd.DataFrame:
    text = _get(f"{OLD_API}/ClientTypeAll.aspx").text
    out = []
    for row in text.split(";"):
        f = row.split(",")
        if len(f) >= 9:
            out.append({"ins_code": f[0], **dict(zip(CLIENT_COLUMNS, map(_num, f[1:9])))})
    if not out:
        raise DataError("داده حقیقی/حقوقی (API قدیمی) خالی بود")
    return pd.DataFrame(out)


def fetch_market_watch() -> pd.DataFrame:
    """Today's snapshot for all ordinary shares, with individual/institutional flows."""
    errors = []
    df = None
    for fn in (_market_watch_new, _market_watch_old):
        try:
            df = fn()
            break
        except Exception as e:  # noqa: BLE001 - try the next source
            errors.append(str(e))
    if df is None:
        raise DataError(" | ".join(errors))

    clients = None
    for fn in (_client_types_new, _client_types_old):
        try:
            clients = fn()
            break
        except Exception as e:  # noqa: BLE001
            errors.append(str(e))
    if clients is not None:
        df = df.merge(clients, on="ins_code", how="left")
    else:
        for c in CLIENT_COLUMNS:
            df[c] = np.nan
    return finalize_market(df)


def finalize_market(df: pd.DataFrame, filter_symbols: bool = True) -> pd.DataFrame:
    if filter_symbols:
        df = df[df["symbol"].map(is_tradeable_share)]
    df = df.copy()
    df = df.drop_duplicates("ins_code")
    df["sector"] = df["sector_code"].map(sector_name)
    df["change_pct"] = (df["close"] / df["yesterday"] - 1) * 100
    df["last_change_pct"] = (df["last"] / df["yesterday"] - 1) * 100
    df["market_cap"] = df["shares"] * df["close"]
    df["pe"] = np.where(df["eps"] > 0, df["close"] / df["eps"], np.nan)
    # Per-capita individual buy vs sell ("قدرت خریدار حقیقی").
    buy_pc = df["buy_vol_i"] * df["close"] / df["buy_count_i"].replace(0, np.nan)
    sell_pc = df["sell_vol_i"] * df["close"] / df["sell_count_i"].replace(0, np.nan)
    df["buyer_power"] = buy_pc / sell_pc
    df["net_individual"] = (df["buy_vol_i"] - df["sell_vol_i"]) * df["close"]
    tol = 1e-9
    df["buy_queue"] = (df["max_allowed"] > 0) & (df["last"] >= df["max_allowed"] - tol)
    df["sell_queue"] = (df["min_allowed"] > 0) & (df["last"] <= df["min_allowed"] + tol)
    return df.reset_index(drop=True)


# --------------------------------------------------------------------- history
def _history_new(ins_code: str) -> pd.DataFrame:
    rows = _get(f"{NEW_API}/ClosingPrice/GetClosingPriceDailyList/{ins_code}/0").json()
    rows = rows.get("closingPriceDaily") or []
    return pd.DataFrame([{
        "date": str(r.get("dEven")),
        "open": _num(r.get("priceFirst")), "high": _num(r.get("priceMax")),
        "low": _num(r.get("priceMin")), "close": _num(r.get("pClosing")),
        "last": _num(r.get("pDrCotVal")), "yesterday": _num(r.get("priceYesterday")),
        "volume": _num(r.get("qTotTran5J")), "value": _num(r.get("qTotCap")),
        "count": _num(r.get("zTotTran")),
    } for r in rows])


def _history_old(ins_code: str) -> pd.DataFrame:
    text = _get(f"{OLD_API}/InstTradeHistory.aspx?i={ins_code}&Top=999999&A=0").text
    out = []
    for row in text.split(";"):
        f = row.split("@")
        if len(f) >= 10:
            out.append({
                "date": f[0], "high": _num(f[1]), "low": _num(f[2]), "close": _num(f[3]),
                "last": _num(f[4]), "open": _num(f[5]), "yesterday": _num(f[6]),
                "value": _num(f[7]), "volume": _num(f[8]), "count": _num(f[9]),
            })
    return pd.DataFrame(out)


def clean_history(raw: pd.DataFrame, adjust: bool = True) -> pd.DataFrame:
    if raw.empty:
        return raw
    df = raw.copy()
    df["date"] = pd.to_datetime(df["date"].astype(str).str[:8], format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["date", "close"]).sort_values("date").drop_duplicates("date")
    df = df[df["volume"] > 0].set_index("date")
    # Days with no first price report 0; fall back to yesterday's close.
    df["open"] = df["open"].where(df["open"] > 0, df["yesterday"])
    df["high"] = df[["high", "open", "close"]].max(axis=1)
    df["low"] = df[["low", "open", "close"]].where(lambda x: x > 0).min(axis=1)
    if adjust:
        df = adjust_prices(df)
    return df


def adjust_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Back-adjust prices for capital increases and dividends.

    After such an event TSETMC sets the next day's "yesterday" price below the
    previous close; the ratio is applied to all earlier prices.
    """
    prev_close = df["close"].shift()
    ratio = (df["yesterday"] / prev_close).where(lambda r: (r > 0.1) & (r < 0.995), 1.0)
    ratio = ratio.fillna(1.0)
    # Factor for day t = product of ratios on all *later* days.
    factor = ratio[::-1].cumprod()[::-1].shift(-1).fillna(1.0)
    out = df.copy()
    for c in ("open", "high", "low", "close", "last", "yesterday"):
        out[c] = out[c] * factor
    out["volume"] = out["volume"] / factor
    return out


def fetch_history(ins_code: str, max_age_hours: float = 6, use_cache: bool = True) -> pd.DataFrame:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{ins_code}.csv"
    if use_cache and path.exists() and time.time() - path.stat().st_mtime < max_age_hours * 3600:
        raw = pd.read_csv(path, dtype={"date": str})
    else:
        raw = pd.DataFrame()
        for fn in (_history_new, _history_old):
            try:
                raw = fn(ins_code)
                if not raw.empty:
                    break
            except Exception:  # noqa: BLE001 - try the next source
                continue
        if raw.empty:
            raise DataError(f"سابقه معاملات {ins_code} دریافت نشد")
        raw.to_csv(path, index=False)
    return clean_history(raw)


def fetch_histories(ins_codes, workers: int = 6, progress=None, **kw) -> dict:
    """Fetch many histories in parallel. progress(done, total) is called as they finish."""
    codes = list(ins_codes)
    result, total = {}, len(codes)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(fetch_history, c, **kw): c for c in codes}
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                result[futures[fut]] = fut.result()
            except Exception:  # noqa: BLE001 - a missing symbol must not stop the scan
                pass
            if progress:
                progress(i, total)
    return result


def load_market_and_histories(max_symbols: int = 0, progress=None):
    """Market watch plus daily histories for every traded share (most traded first)."""
    market = fetch_market_watch()
    universe = market[market["volume"] > 0].sort_values("value", ascending=False)
    if max_symbols:
        universe = universe.head(max_symbols)
    return market, fetch_histories(universe["ins_code"], progress=progress)
