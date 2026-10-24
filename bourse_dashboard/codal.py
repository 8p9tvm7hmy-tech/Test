"""Codal (codal.ir) announcements and monthly sales.

* latest_letters(symbol): recent announcements with links (search.codal.ir JSON API)
* monthly_sales(symbol):  sales totals from the "گزارش فعالیت ماهانه" reports

Monthly reports are HTML pages that embed their tables as a JSON `datasource`
variable; the parser looks for the total row ("جمع") under the one-month
period column. Report layouts vary between industries, so a report that can't
be read is skipped rather than guessed.
"""
from __future__ import annotations

import json
import re
import time

import pandas as pd
import requests

import config

SEARCH_URL = "https://search.codal.ir/api/search/v2/q"
SITE = "https://codal.ir"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://codal.ir/",
}
MONTHLY_ACTIVITY = 58  # LetterType: گزارش فعالیت ماهانه

# Announcement titles worth an alert.
IMPORTANT_WORDS = ("افزایش سرمایه", "مجمع", "تقسیم سود", "توقف", "بازگشایی", "تعدیل",
                   "صورت‌های مالی", "صورتهای مالی", "افشای اطلاعات بااهمیت", "عرضه")

_FA_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")
_session = requests.Session()
_session.headers.update(HEADERS)


class CodalError(RuntimeError):
    pass


def to_latin(s) -> str:
    return str(s or "").translate(_FA_DIGITS)


def normalize(s) -> str:
    """Unify Arabic/Persian letters and spacing so text matching is reliable."""
    s = str(s or "").replace("ي", "ی").replace("ك", "ک").replace("‌", " ")
    return re.sub(r"\s+", " ", s).strip()


def _get(url, params=None, retries=3, timeout=20):
    last = None
    for attempt in range(retries):
        try:
            r = _session.get(url, params=params, timeout=timeout)
            if r.status_code == 200:
                return r
            last = f"HTTP {r.status_code}"
        except requests.RequestException as e:
            last = e
        time.sleep(1.5 * (attempt + 1))
    raise CodalError(f"دسترسی به کدال ممکن نشد ({last})")


# ------------------------------------------------------------- announcements
def search_letters(symbol: str, letter_type: int = -1, page: int = 1) -> list[dict]:
    params = {
        "Symbol": symbol, "LetterType": letter_type, "PageNumber": page,
        "Audited": "true", "NotAudited": "true", "Mains": "true", "Childs": "true",
        "Consolidatable": "true", "NotConsolidatable": "true", "IsNotAudited": "false",
        "AuditorRef": -1, "Category": -1, "CompanyState": -1, "CompanyType": -1,
        "Length": -1, "Publisher": "false", "TracingNo": -1, "search": "true",
    }
    return _get(SEARCH_URL, params).json().get("Letters") or []


def _abs(url: str) -> str:
    if not url:
        return ""
    return url if url.startswith("http") else f"{SITE}/{url.lstrip('/')}"


def latest_letters(symbol: str, limit: int = 20) -> pd.DataFrame:
    rows = []
    for l in search_letters(symbol)[:limit]:
        title = normalize(l.get("Title"))
        rows.append({
            "date": to_latin(l.get("PublishDateTime") or l.get("SentDateTime")),
            "title": title,
            "code": to_latin(l.get("LetterCode")),
            "url": _abs(l.get("Url")),
            "pdf": _abs(l.get("PdfUrl")),
            "important": any(normalize(w) in title for w in IMPORTANT_WORDS),
            "tracing_no": l.get("TracingNo"),
        })
    return pd.DataFrame(rows, columns=["date", "title", "code", "url", "pdf", "important", "tracing_no"])


# ------------------------------------------------------------ monthly sales
def _num(v):
    s = to_latin(v).replace(",", "").replace("٬", "").strip()
    neg = s.startswith("(") and s.endswith(")")
    s = s.strip("()")
    try:
        x = float(s)
    except ValueError:
        return None
    return -x if neg else x


def extract_datasource(html: str) -> dict | None:
    m = re.search(r"var\s+datasource\s*=\s*(\{.*?\});\s*(?:\n|var|</script>)", html, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except ValueError:
        return None


def _grid(table: dict) -> dict:
    """(row, col) -> cell, with colSpan cells repeated across their span."""
    grid = {}
    for c in table.get("cells") or []:
        r, col = c.get("rowSequence"), c.get("columnSequence")
        if r is None or col is None:
            continue
        for k in range(int(c.get("colSpan") or 1)):
            grid.setdefault((int(r), int(col) + k), c)
    return grid


def parse_monthly_total(ds: dict) -> float | None:
    """Total sales for the one-month period (میلیون ریال), or None if not found."""
    for sheet in ds.get("sheets") or []:
        for table in sheet.get("tables") or []:
            grid = _grid(table)
            if not grid:
                continue
            text = {k: normalize(c.get("value")) for k, c in grid.items()}
            rows = sorted({r for r, _ in grid})
            cols = sorted({c for _, c in grid})
            # Columns under a "یک ماهه" header whose own header mentions an amount.
            month_cols = {c for (r, c), t in text.items() if "یک ماهه" in t or "1 ماهه" in to_latin(t)}
            amount_cols = [c for c in month_cols
                           if any(("مبلغ" in text.get((r, c), "") or "درآمد" in text.get((r, c), ""))
                                  for r in rows)]
            if not amount_cols:
                continue
            col = max(amount_cols)  # amount is the last sub-column (after count and rate)
            total_rows = [r for r in rows
                          if any(text.get((r, c), "").startswith("جمع") for c in cols[:3])]
            for r in reversed(total_rows):
                v = _num(grid.get((r, col), {}).get("value"))
                if v is not None:
                    return v
    return None


def period_from_title(title: str) -> str:
    m = re.search(r"(\d{4}/\d{1,2}/\d{1,2})", to_latin(title))
    return m.group(1) if m else ""


def monthly_sales(symbol: str, months: int = 12) -> pd.DataFrame:
    """Monthly sales from the latest monthly-activity reports (newest last)."""
    cache = config.data_dir() / "codal"
    cache.mkdir(exist_ok=True)
    out = []
    letters = search_letters(symbol, MONTHLY_ACTIVITY)
    seen = set()
    for l in letters:
        title = normalize(l.get("Title"))
        period = period_from_title(title)
        if not period or period in seen or "اصلاحیه" in title:
            continue
        seen.add(period)
        path = cache / f"{l.get('TracingNo')}.json"
        if path.exists():
            value = json.loads(path.read_text()).get("value")
        else:
            try:
                ds = extract_datasource(_get(_abs(l.get("Url"))).text)
            except CodalError:
                continue
            value = parse_monthly_total(ds) if ds else None
            path.write_text(json.dumps({"value": value}))
        if value is not None:
            out.append({"period": period, "sales": value, "title": title, "url": _abs(l.get("Url"))})
        if len(seen) >= months:
            break
    df = pd.DataFrame(out, columns=["period", "sales", "title", "url"])
    return df.sort_values("period").reset_index(drop=True)


def sales_growth(df: pd.DataFrame) -> dict:
    """Recent 3-month sales vs the 3 months before (%), and month-over-month."""
    if len(df) < 2:
        return {"mom": None, "q_growth": None}
    s = df["sales"].astype(float)
    mom = (s.iloc[-1] / s.iloc[-2] - 1) * 100 if s.iloc[-2] else None
    q = None
    if len(s) >= 6 and s.iloc[-6:-3].sum():
        q = (s.iloc[-3:].sum() / s.iloc[-6:-3].sum() - 1) * 100
    return {"mom": mom, "q_growth": q}
