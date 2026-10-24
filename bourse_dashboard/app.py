"""داشبورد تحلیل بورس تهران — run with:  streamlit run app.py"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import alerts
import analysis as an
import codal
import config
import data
import demo
import indicators as ind

st.set_page_config(page_title="داشبورد بورس تهران", page_icon="📈", layout="wide")

st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Vazirmatn:wght@400;600;800&display=swap" rel="stylesheet">
<style>
html, body, [class*="css"], .stMarkdown, .stText, .stMetric, button, input, label, p, h1, h2, h3, h4, li {
  font-family: 'Vazirmatn', Tahoma, sans-serif !important;
}
.main .block-container, section[data-testid="stSidebar"] { direction: rtl; text-align: right; }
[data-testid="stMetricValue"] { direction: ltr; text-align: right; }
.card { border: 1px solid rgba(128,128,128,.3); border-radius: 12px; padding: 14px 18px; margin-bottom: 10px; }
.card h4 { margin: 0 0 6px 0; }
.big { font-size: 2rem; font-weight: 800; }
.pos { color: #16a34a; } .neg { color: #dc2626; } .neu { color: #ca8a04; }
.reason { margin: 2px 0; font-size: .92rem; }
</style>
""", unsafe_allow_html=True)

SIGNAL_COLORS = {"خرید قوی": "#15803d", "خرید": "#22c55e", "خنثی": "#eab308",
                 "احتیاط": "#f97316", "دوری/فروش": "#dc2626"}
BILLION = 1e9

# ------------------------------------------------------------------ sidebar
cfg = config.load()

with st.sidebar:
    st.header("⚙️ تنظیمات")
    source = st.radio("منبع داده", ["TSETMC (زنده)", "داده نمونه (آزمایشی)"])
    min_value_b = st.number_input("حداقل میانگین ارزش معاملات روزانه (میلیارد ریال)",
                                  0.0, 10000.0, float(cfg["min_avg_value"]) / 1e9, step=1.0)
    max_symbols = st.number_input("حداکثر تعداد نماد برای بررسی (۰ = همه)", 0, 2000,
                                  int(cfg["max_symbols"]), step=50,
                                  help="نمادها بر اساس ارزش معاملات امروز مرتب می‌شوند.")
    # The alert watcher reads the same universe settings.
    if (min_value_b * 1e9, int(max_symbols)) != (cfg["min_avg_value"], cfg["max_symbols"]):
        cfg.update(min_avg_value=min_value_b * 1e9, max_symbols=int(max_symbols))
        config.save(cfg)
    st.subheader("سرمایه و ریسک")
    capital_m = st.number_input("سرمایه (میلیون تومان)", 1.0, 1e7, 100.0, step=10.0)
    risk_pct = st.slider("ریسک هر معامله (٪ سرمایه)", 0.25, 5.0, 1.0, 0.25)
    fee = st.number_input("کارمزد رفت‌وبرگشت + مالیات (٪)", 0.0, 5.0, 1.25, step=0.05)
    rf = st.number_input("نرخ بدون ریسک سالانه (٪)", 0.0, 100.0, 30.0, step=1.0,
                         help="برای مقایسه بازده سود (E/P) سهم با سود سپرده یا اوراق.")
    refresh = st.button("🔄 دریافت دوباره داده‌ها", width="stretch")
    st.caption("سابقه قیمت‌ها ۶ ساعت ذخیره می‌شود؛ اجرای اول برای همه نمادها چند دقیقه طول می‌کشد.")

params = an.Params(min_avg_value=min_value_b * BILLION, risk_free=rf, fee_round_trip=fee,
                   capital=capital_m * 1e6 * 10, risk_per_trade=risk_pct)


# --------------------------------------------------------------------- data
def load(source: str, max_symbols: int):
    if source.startswith("داده نمونه"):
        return (*demo.demo_market(), "نمونه")
    bar = st.progress(0.0, text="دریافت سابقه قیمت نمادها…")
    market, histories = data.load_market_and_histories(
        max_symbols, progress=lambda d, t: bar.progress(d / t, text=f"دریافت سابقه قیمت: {d} از {t}"))
    bar.empty()
    return market, histories, "TSETMC"


key = (source, int(max_symbols))
if refresh or st.session_state.get("key") != key:
    try:
        with st.spinner("در حال دریافت داده‌ها…"):
            st.session_state["raw"] = load(source, int(max_symbols))
            st.session_state["key"] = key
            st.session_state["loaded_at"] = time.strftime("%H:%M")
    except data.DataError as e:
        st.error(f"اتصال به TSETMC برقرار نشد: {e}")
        st.info("اگر خارج از ایران هستید یا اینترنت محدود است، سایت TSETMC ممکن است در دسترس نباشد. "
                "برای دیدن داشبورد «داده نمونه» را انتخاب کنید.")
        st.stop()

market, histories, origin = st.session_state["raw"]
swing, longterm, frames = an.scan(market, histories, params)

st.title("📈 داشبورد تحلیل بورس تهران")
st.caption(f"منبع: {origin} · آخرین دریافت: {st.session_state['loaded_at']} · "
           f"{len(market)} نماد در دیده‌بان، {len(frames)} نماد با سابقه قیمت")
if origin == "نمونه":
    st.warning("داده‌ها ساختگی هستند و فقط برای آشنایی با داشبورد است.")
st.info("⚠️ این ابزار مشاوره مالی نیست؛ امتیازها بر پایه قواعد ساده تکنیکال و بنیادی‌اند و آینده را "
        "تضمین نمی‌کنند. پیش از هر معامله گزارش‌های کدال و اخبار نماد را بررسی کنید.")


def fmt(x, digits=0):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:,.{digits}f}"


def badge(signal):
    return f"<span style='color:{SIGNAL_COLORS.get(signal, '#888')};font-weight:700'>{signal}</span>"


@st.cache_resource
def get_watcher():
    return alerts.Watcher()


watcher = get_watcher()
if cfg["alerts_enabled"] and not watcher.running:
    watcher.start()


@st.cache_data(ttl=3600, show_spinner=False)
def codal_letters(symbol):
    return codal.latest_letters(symbol)


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def codal_sales(symbol):
    return codal.monthly_sales(symbol)


tabs = st.tabs(["📊 نمای بازار", "⚡ نوسان‌گیری (کوتاه‌مدت)", "🏦 سرمایه‌گذاری (بلندمدت)",
                "🔍 تحلیل نماد", "🧮 مدیریت ریسک", "🔔 هشدار تلگرام", "ℹ️ راهنما"])

# ---------------------------------------------------------------- overview
with tabs[0]:
    traded = market[market["volume"] > 0]
    up, down = (traded["change_pct"] > 0).sum(), (traded["change_pct"] < 0).sum()
    c = st.columns(6)
    c[0].metric("نمادهای مثبت", f"{up:,}")
    c[1].metric("نمادهای منفی", f"{down:,}")
    c[2].metric("ارزش معاملات (همت)", fmt(traded["value"].sum() / 1e13, 2))
    c[3].metric("ورود پول حقیقی (میلیارد تومان)", fmt(traded["net_individual"].sum() / 1e10, 0))
    c[4].metric("صف خرید", f"{int(traded['buy_queue'].sum()):,}")
    c[5].metric("صف فروش", f"{int(traded['sell_queue'].sum()):,}")

    tm = traded.copy()
    tm["size"] = tm["market_cap"].where(tm["market_cap"] > 0, tm["value"])
    tm = tm[tm["size"] > 0]
    if not tm.empty:
        fig = px.treemap(tm, path=[px.Constant("بازار"), "sector", "symbol"], values="size",
                         color="change_pct", color_continuous_scale="RdYlGn",
                         range_color=(-5, 5), custom_data=["change_pct", "name"])
        fig.update_traces(hovertemplate="<b>%{label}</b><br>تغییر: %{customdata[0]:.2f}٪<extra></extra>")
        fig.update_layout(margin=dict(t=30, l=0, r=0, b=0), height=520, coloraxis_colorbar_title="تغییر٪",
                          title="نقشه بازار (اندازه: ارزش بازار، رنگ: درصد تغییر قیمت پایانی)")
        st.plotly_chart(fig, width="stretch")

    left, right = st.columns(2)
    flow = traded.groupby("sector")["net_individual"].sum().div(1e10).sort_values()
    flow = pd.concat([flow.head(8), flow.tail(8)]).drop_duplicates()
    fig = go.Figure(go.Bar(x=flow.values, y=flow.index, orientation="h",
                           marker_color=np.where(flow.values >= 0, "#16a34a", "#dc2626")))
    fig.update_layout(title="ورود/خروج پول حقیقی به گروه‌ها (میلیارد تومان)", height=460,
                      margin=dict(l=10, r=10, t=40, b=10))
    left.plotly_chart(fig, width="stretch")

    cols = {"symbol": "نماد", "sector": "گروه", "close": "قیمت پایانی", "change_pct": "تغییر٪",
            "value": "ارزش (میلیارد ریال)", "buyer_power": "قدرت خریدار"}
    top = traded.assign(value=traded["value"] / BILLION)
    right.markdown("**بیشترین ارزش معاملات**")
    right.dataframe(top.nlargest(12, "value")[list(cols)].rename(columns=cols),
                    hide_index=True, width="stretch",
                    column_config={"تغییر٪": st.column_config.NumberColumn(format="%.2f"),
                                   "قدرت خریدار": st.column_config.NumberColumn(format="%.2f"),
                                   "ارزش (میلیارد ریال)": st.column_config.NumberColumn(format="%.1f"),
                                   "قیمت پایانی": st.column_config.NumberColumn(format="localized")})

# ------------------------------------------------------------------- swing
with tabs[1]:
    st.subheader("⚡ فرصت‌های نوسان‌گیری")
    st.caption("ترکیب روند (EMA)، مومنتوم (RSI، MACD)، حجم، شکست مقاومت و جریان پول حقیقی امروز. "
               "حد ضرر بر اساس ATR و کف ۱۰ روزه، هدف‌ها ۱.۵ و ۳ برابر ریسک.")
    if swing.empty:
        st.warning("هیچ نمادی سابقه کافی (۶۰ روز) نداشت.")
    else:
        f1, f2, f3, f4 = st.columns([1, 2, 2, 1])
        min_score = f1.slider("حداقل امتیاز", 0, 100, 65)
        sigs = f2.multiselect("سیگنال", list(SIGNAL_COLORS), default=["خرید قوی", "خرید"])
        secs = f3.multiselect("گروه صنعت", sorted(swing["sector"].unique()))
        no_queue = f4.checkbox("حذف صف خرید", True)
        v = swing[(swing["score"] >= min_score) & (swing["avg_value20"] >= params.min_avg_value)]
        if sigs:
            v = v[v["signal"].isin(sigs)]
        if secs:
            v = v[v["sector"].isin(secs)]
        if no_queue:
            v = v[~v["buy_queue"]]
        v = v.round({c: 0 for c in ("close", "stop", "target1", "target2")})
        st.markdown(f"**{len(v)} نماد** با این فیلترها")
        cols = {"symbol": "نماد", "sector": "گروه", "score": "امتیاز", "signal": "سیگنال",
                "close": "قیمت", "change_pct": "تغییر امروز٪", "rsi": "RSI", "vol_ratio": "نسبت حجم",
                "buyer_power": "قدرت خریدار", "atr_pct": "نوسان روزانه٪", "stop": "حد ضرر",
                "target1": "هدف ۱", "target2": "هدف ۲", "t1_net_pct": "سود خالص هدف۱٪",
                "shares": "تعداد سهم پیشنهادی"}
        num = st.column_config.NumberColumn
        st.dataframe(v[list(cols)].rename(columns=cols), hide_index=True, width="stretch",
                     height=520, column_config={
                         "امتیاز": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f"),
                         **{k: num(format="%.2f") for k in ("تغییر امروز٪", "نسبت حجم", "قدرت خریدار",
                                                            "نوسان روزانه٪", "سود خالص هدف۱٪")},
                         "RSI": num(format="%.0f"),
                         **{k: num(format="localized") for k in ("قیمت", "حد ضرر", "هدف ۱", "هدف ۲",
                                                                  "تعداد سهم پیشنهادی")},
                     })
        dist = swing["signal"].value_counts().reindex(list(SIGNAL_COLORS)).fillna(0)
        fig = go.Figure(go.Bar(x=dist.index, y=dist.values, marker_color=list(SIGNAL_COLORS.values())))
        fig.update_layout(title="پراکندگی سیگنال‌های کوتاه‌مدت در کل بازار", height=300,
                          margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, width="stretch")

# --------------------------------------------------------------- long term
with tabs[2]:
    st.subheader("🏦 گزینه‌های سرمایه‌گذاری بلندمدت")
    st.caption("ارزندگی (P/E نسبت به گروه، بازده سود)، روند یک‌ساله، ثبات قیمت، اندازه و نقدشوندگی. "
               "ارزش نسبی = EPS × میانه P/E گروه.")
    if longterm.empty:
        st.warning("داده کافی برای تحلیل بلندمدت نیست.")
    else:
        f1, f2, f3 = st.columns([1, 1, 2])
        min_lt = f1.slider("حداقل امتیاز بلندمدت", 0, 100, 60)
        only_profit = f2.checkbox("فقط شرکت‌های سودده", True)
        secs = f3.multiselect("گروه صنعت ", sorted(longterm["sector"].unique()))
        v = longterm[longterm["lt_score"] >= min_lt]
        if only_profit:
            v = v[v["eps"] > 0]
        if secs:
            v = v[v["sector"].isin(secs)]
        v = v.assign(market_cap_b=v["market_cap"] / 1e13).round({"close": 0, "eps": 0, "fair_price": 0})
        cols = {"symbol": "نماد", "sector": "گروه", "lt_score": "امتیاز", "lt_signal": "سیگنال",
                "close": "قیمت", "eps": "EPS", "pe": "P/E", "sector_pe": "P/E گروه",
                "earnings_yield": "بازده سود٪", "fair_price": "ارزش نسبی", "upside": "پتانسیل٪",
                "ret_1y": "بازده ۱ ساله٪", "vol_1y": "نوسان سالانه٪", "drawdown": "فاصله از سقف٪",
                "market_cap_b": "ارزش بازار (همت)"}
        num = st.column_config.NumberColumn
        st.markdown(f"**{len(v)} نماد** با این فیلترها")
        st.dataframe(v[list(cols)].rename(columns=cols), hide_index=True, width="stretch",
                     height=520, column_config={
                         "امتیاز": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f"),
                         **{k: num(format="%.1f") for k in ("P/E", "P/E گروه", "بازده سود٪", "پتانسیل٪",
                                                            "بازده ۱ ساله٪", "نوسان سالانه٪",
                                                            "فاصله از سقف٪", "ارزش بازار (همت)")},
                         **{k: num(format="localized") for k in ("قیمت", "EPS", "ارزش نسبی")},
                     })
        st.markdown("**رشد فروش از کدال** (گزارش‌های فعالیت ماهانه)")
        g1, g2 = st.columns([1, 3])
        n_codal = g1.number_input("تعداد نماد برتر", 5, 100, 20, step=5)
        if g2.button("📥 دریافت فروش ماهانه از کدال برای نمادهای برتر جدول بالا"):
            rows, bar = [], st.progress(0.0)
            for i, r in enumerate(v.head(int(n_codal)).itertuples(), 1):
                try:
                    sales = codal_sales(r.symbol)
                    g = codal.sales_growth(sales)
                    rows.append({"نماد": r.symbol, "ماه‌های خوانده‌شده": len(sales),
                                 "آخرین ماه": sales["period"].iloc[-1] if len(sales) else "—",
                                 "رشد ماه به ماه٪": g["mom"], "رشد ۳ ماه اخیر٪": g["q_growth"]})
                except codal.CodalError as e:
                    st.error(str(e))
                    break
                bar.progress(i / int(n_codal), text=f"کدال: {r.symbol}")
            bar.empty()
            st.session_state["codal_growth"] = pd.DataFrame(rows)
        if "codal_growth" in st.session_state and not st.session_state["codal_growth"].empty:
            st.dataframe(st.session_state["codal_growth"], hide_index=True, width="stretch",
                         column_config={k: st.column_config.NumberColumn(format="%.1f")
                                        for k in ("رشد ماه به ماه٪", "رشد ۳ ماه اخیر٪")})
            st.caption("رشد ۳ ماه اخیر = جمع فروش ۳ ماه آخر نسبت به ۳ ماه قبل از آن.")

        sc = v[(v["pe"] > 0) & (v["pe"] < 60)].dropna(subset=["ret_1y"])
        if not sc.empty:
            fig = px.scatter(sc, x="pe", y="ret_1y", color="lt_score", hover_name="symbol",
                             size=sc["market_cap"].clip(lower=1).pow(0.5), size_max=35,
                             color_continuous_scale="RdYlGn",
                             labels={"pe": "P/E", "ret_1y": "بازده یک‌ساله٪", "lt_score": "امتیاز"})
            fig.update_layout(title="ارزندگی در برابر روند (گوشه چپ-بالا: ارزان و در روند صعودی)",
                              height=460, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, width="stretch")

# ------------------------------------------------------------------ symbol
with tabs[3]:
    if not frames:
        st.warning("سابقه قیمتی برای نمایش وجود ندارد.")
    else:
        mk = market.set_index("ins_code")
        order = list(swing["ins_code"]) if not swing.empty else []
        order += [c for c in frames if c not in order]
        labels = {c: f"{mk.loc[c, 'symbol']} — {mk.loc[c, 'name']}" for c in order}
        code = st.selectbox("انتخاب نماد", order, format_func=labels.get)
        h = frames[code].iloc[-250:]
        srow = swing.set_index("ins_code").loc[code] if code in set(swing.get("ins_code", [])) else None
        lrow = longterm.set_index("ins_code").loc[code] if code in set(longterm.get("ins_code", [])) else None

        a, b = st.columns(2)
        with a:
            if srow is not None:
                reasons = "".join(f"<div class='reason'>{r}</div>" for r in srow["reasons"])
                a.markdown(f"""<div class='card'><h4>⚡ نوسان‌گیری: {badge(srow['signal'])}</h4>
                <div class='big'>{srow['score']:.0f}<small>/100</small></div>
                ورود ≈ {fmt(srow['entry'])} · حد ضرر {fmt(srow['stop'])} ({srow['stop_pct']:.1f}٪)<br>
                هدف ۱: {fmt(srow['target1'])} · هدف ۲: {fmt(srow['target2'])} ·
                سود خالص هدف ۱: {srow['t1_net_pct']:.1f}٪<br>
                حجم پیشنهادی: {fmt(srow['shares'])} سهم (≈ {fmt(srow['position_value'] / 1e7, 1)} میلیون تومان)
                <hr style='margin:8px 0'>{reasons}</div>""", unsafe_allow_html=True)
        with b:
            if lrow is not None:
                reasons = "".join(f"<div class='reason'>{r}</div>" for r in an.long_term_reasons(lrow, params))
                b.markdown(f"""<div class='card'><h4>🏦 بلندمدت: {badge(lrow['lt_signal'])}</h4>
                <div class='big'>{lrow['lt_score']:.0f}<small>/100</small></div>
                P/E: {fmt(lrow['pe'], 1)} · EPS: {fmt(lrow['eps'])} · ارزش نسبی: {fmt(lrow['fair_price'])}<br>
                ارزش بازار: {fmt(lrow['market_cap'] / 1e13, 2)} همت · بازده یک‌ساله: {fmt(lrow['ret_1y'], 0)}٪
                <hr style='margin:8px 0'>{reasons}</div>""", unsafe_allow_html=True)

        fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                            row_heights=[0.55, 0.15, 0.15, 0.15])
        fig.add_trace(go.Candlestick(x=h.index, open=h["open"], high=h["high"], low=h["low"],
                                     close=h["close"], name="قیمت",
                                     increasing_line_color="#16a34a", decreasing_line_color="#dc2626"), 1, 1)
        for col, color, name in (("ema20", "#3b82f6", "EMA20"), ("ema50", "#f59e0b", "EMA50"),
                                 ("sma200", "#a855f7", "SMA200")):
            fig.add_trace(go.Scatter(x=h.index, y=h[col], line=dict(color=color, width=1.4), name=name), 1, 1)
        fig.add_trace(go.Scatter(x=h.index, y=h["bb_up"], line=dict(color="gray", width=0.8, dash="dot"),
                                 name="بولینگر", showlegend=False), 1, 1)
        fig.add_trace(go.Scatter(x=h.index, y=h["bb_low"], line=dict(color="gray", width=0.8, dash="dot"),
                                 fill="tonexty", fillcolor="rgba(128,128,128,.07)", name="بولینگر"), 1, 1)
        if srow is not None:
            for y, color, txt in ((srow["stop"], "#dc2626", "حد ضرر"), (srow["target1"], "#16a34a", "هدف ۱"),
                                  (srow["target2"], "#15803d", "هدف ۲")):
                fig.add_hline(y=y, line=dict(color=color, dash="dash", width=1), row=1, col=1,
                              annotation_text=txt, annotation_position="top left")
        fig.add_trace(go.Bar(x=h.index, y=h["volume"], name="حجم",
                             marker_color=np.where(h["close"] >= h["open"], "#86efac", "#fca5a5")), 2, 1)
        fig.add_trace(go.Scatter(x=h.index, y=h["vol_avg20"], line=dict(color="#555", width=1),
                                 name="میانگین حجم"), 2, 1)
        fig.add_trace(go.Scatter(x=h.index, y=h["rsi"], line=dict(color="#0ea5e9"), name="RSI"), 3, 1)
        for lvl in (30, 70):
            fig.add_hline(y=lvl, line=dict(color="gray", dash="dot", width=1), row=3, col=1)
        fig.add_trace(go.Bar(x=h.index, y=h["macd_hist"], name="هیستوگرام MACD",
                             marker_color=np.where(h["macd_hist"] >= 0, "#16a34a", "#dc2626")), 4, 1)
        fig.add_trace(go.Scatter(x=h.index, y=h["macd"], line=dict(color="#3b82f6", width=1), name="MACD"), 4, 1)
        fig.add_trace(go.Scatter(x=h.index, y=h["macd_signal"], line=dict(color="#f59e0b", width=1),
                                 name="سیگنال"), 4, 1)
        fig.update_layout(height=820, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=30, b=10),
                          legend=dict(orientation="h", y=1.02), hovermode="x unified")
        fig.update_xaxes(rangebreaks=[dict(bounds=["thu", "sat"])] if origin == "TSETMC" else [])
        st.plotly_chart(fig, width="stretch")

        st.subheader("📄 کدال")
        sym = mk.loc[code, "symbol"]
        if origin == "نمونه":
            st.caption("برای نمادهای نمونه اطلاعات کدال وجود ندارد.")
        else:
            c1, c2 = st.columns([3, 2])
            try:
                letters = codal_letters(sym)
                c1.markdown("**آخرین اطلاعیه‌ها**")
                c1.dataframe(letters.assign(important=letters["important"].map({True: "⭐", False: ""}))
                             [["important", "date", "title", "url"]],
                             hide_index=True, width="stretch", height=380, column_config={
                                 "important": st.column_config.TextColumn("", width="small"),
                                 "date": "تاریخ انتشار", "title": "عنوان",
                                 "url": st.column_config.LinkColumn("لینک", display_text="مشاهده")})
            except codal.CodalError as e:
                c1.warning(str(e))
            try:
                sales = codal_sales(sym)
                if sales.empty:
                    c2.info("فروش ماهانه‌ای از گزارش‌های این نماد خوانده نشد (قالب گزارش برخی صنایع، "
                            "مثل بانک‌ها و سرمایه‌گذاری‌ها، متفاوت است).")
                else:
                    g = codal.sales_growth(sales)
                    figs = go.Figure(go.Bar(x=sales["period"], y=sales["sales"] / 1e3, marker_color="#3b82f6"))
                    figs.update_layout(title="فروش ماهانه (میلیارد ریال)", height=300,
                                       margin=dict(l=10, r=10, t=40, b=10))
                    c2.plotly_chart(figs, width="stretch")
                    c2.markdown(f"رشد ماه به ماه: **{fmt(g['mom'], 1)}٪** · "
                                f"رشد ۳ ماه اخیر: **{fmt(g['q_growth'], 1)}٪**")
            except codal.CodalError as e:
                c2.warning(str(e))

# -------------------------------------------------------------------- risk
with tabs[4]:
    st.subheader("🧮 هدف سود روزانه و مدیریت ریسک")
    a, b = st.columns(2)
    with a:
        target = st.number_input("هدف سود خالص روزانه (٪ از سرمایه)", 0.1, 5.0, 0.5, step=0.1)
        capital_t = params.capital / 10
        st.markdown(f"""<div class='card'>
        سرمایه: <b>{fmt(capital_t / 1e6)}</b> میلیون تومان<br>
        هدف روزانه: <b>{fmt(capital_t * target / 100 / 1e6, 2)}</b> میلیون تومان<br>
        حرکت لازم قیمت با کل سرمایه در یک معامله: <b>{target + fee:.2f}٪</b> (شامل کارمزد {fee}٪)<br>
        معادل ماهانه (۲۰ روز معاملاتی، مرکب): <b>{((1 + target / 100) ** 20 - 1) * 100:.1f}٪</b>
        </div>""", unsafe_allow_html=True)
        if not swing.empty:
            med_atr = swing["atr_pct"].median()
            st.caption(f"میانه نوسان روزانه (ATR) نمادها: {med_atr:.1f}٪. "
                       "هدف روزانه‌ای که به هر معامله حرکتی بیش از ATR تحمیل کند واقع‌بینانه نیست.")
        st.markdown("""
**قواعد پیشنهادی برای نوسان‌گیری:**
- برای هر معامله **پیش از ورود** حد ضرر تعیین کنید و به آن پایبند باشید.
- کارمزد رفت‌وبرگشت حدود ۱.۲۵٪ است؛ معاملات با هدف کمتر از ۲٪ معمولاً ارزش ندارند.
- دامنه نوسان روزانه محدود است؛ هدف‌ها اغلب در چند روز محقق می‌شوند، نه یک روز.
- حداکثر ۳ تا ۵ نماد هم‌زمان و از گروه‌های متفاوت.
- بخش بلندمدت سبد را از بخش نوسان‌گیری جدا نگه دارید.
""")
    with b:
        st.markdown("**ماشین‌حساب حجم معامله**")
        entry = st.number_input("قیمت ورود (ریال)", 1.0, 1e7, 10000.0, step=10.0)
        stop = st.number_input("حد ضرر (ریال)", 1.0, 1e7, 9500.0, step=10.0)
        if stop < entry:
            risk_amount = params.capital * params.risk_per_trade / 100
            shares = int(min(risk_amount / (entry - stop), params.capital / entry))
            st.markdown(f"""<div class='card'>
            حداکثر زیان مجاز: <b>{fmt(risk_amount / 1e7, 2)}</b> میلیون تومان ({params.risk_per_trade}٪)<br>
            تعداد سهم: <b>{fmt(shares)}</b><br>
            ارزش خرید: <b>{fmt(shares * entry / 1e7, 2)}</b> میلیون تومان
            ({shares * entry / params.capital * 100:.0f}٪ سرمایه)<br>
            هدف با نسبت ۱:۲ → <b>{fmt(entry + 2 * (entry - stop))}</b> ریال
            </div>""", unsafe_allow_html=True)
        else:
            st.error("حد ضرر باید کمتر از قیمت ورود باشد.")

# ---------------------------------------------------------------- telegram
with tabs[5]:
    st.subheader("🔔 هشدار تلگرام")
    with st.expander("راهنمای راه‌اندازی (یک بار)", expanded=not cfg["telegram_token"]):
        st.markdown("""
1. در تلگرام به **@BotFather** پیام دهید، دستور `/newbot` را بفرستید و یک نام بدهید.
2. توکنی که BotFather می‌دهد (شبیه `123456:ABC-...`) را در کادر زیر وارد کنید.
3. ربات خودتان را در تلگرام باز کنید و دکمه **Start** را بزنید (یا هر پیامی بفرستید).
4. دکمه «پیدا کردن شناسه چت» را بزنید و سپس «ارسال پیام آزمایشی».

اگر تلگرام روی اینترنت شما فیلتر است، آدرس پروکسی (مثلاً `socks5h://127.0.0.1:1080` یا
`http://127.0.0.1:10809`) فیلتر‌شکن خود را وارد کنید تا فقط پیام‌های تلگرام از آن عبور کنند.
""")
    a, b = st.columns(2)
    with a:
        token = st.text_input("توکن ربات", cfg["telegram_token"], type="password")
        chat_id = st.text_input("شناسه چت (chat id)", cfg["telegram_chat_id"])
        proxy = st.text_input("پروکسی (اختیاری)", cfg["telegram_proxy"])
    with b:
        min_alert = st.slider("حداقل امتیاز نوسان‌گیری برای هشدار", 50, 100, int(cfg["alert_min_score"]))
        interval = st.number_input("فاصله بررسی (دقیقه)", 5, 240, int(cfg["alert_interval_min"]), step=5)
        lt_daily = st.checkbox("ارسال روزانه برترین‌های بلندمدت", cfg["alert_long_term"])
        codal_ev = st.checkbox("ارسال اطلاعیه‌های مهم کدال برای نمادهای هشدار داده‌شده", cfg["alert_codal_events"])
        outside = st.checkbox("بررسی خارج از ساعت بازار هم انجام شود", cfg["alert_outside_hours"])
    new_cfg = {**cfg, "telegram_token": token.strip(), "telegram_chat_id": chat_id.strip(),
               "telegram_proxy": proxy.strip(), "alert_min_score": min_alert,
               "alert_interval_min": int(interval), "alert_long_term": lt_daily,
               "alert_codal_events": codal_ev, "alert_outside_hours": outside}

    c = st.columns(4)
    if c[0].button("💾 ذخیره تنظیمات", width="stretch"):
        config.save(new_cfg)
        st.success("ذخیره شد.")
    if c[1].button("🔎 پیدا کردن شناسه چت", width="stretch"):
        try:
            chats = alerts.find_chat_ids(new_cfg)
            if chats:
                new_cfg["telegram_chat_id"] = chats[-1][0]
                config.save(new_cfg)
                st.success("پیدا شد و ذخیره شد: " + "، ".join(f"{n} ({i})" for i, n in chats))
                st.rerun()
            else:
                st.warning("پیامی پیدا نشد. ابتدا در تلگرام به ربات خود پیام بدهید و دوباره امتحان کنید.")
        except alerts.TelegramError as e:
            st.error(str(e))
    if c[2].button("✉️ ارسال پیام آزمایشی", width="stretch"):
        try:
            alerts.send(new_cfg, "✅ اتصال داشبورد بورس به تلگرام برقرار است.")
            st.success("پیام ارسال شد.")
        except alerts.TelegramError as e:
            st.error(str(e))
    if c[3].button("🚀 ارسال هشدارهای همین حالا", width="stretch"):
        try:
            n = alerts.run_once(new_cfg, market=market, histories=histories, log=lambda *_: None)
            st.success(f"{n} پیام ارسال شد (نمادهایی که امروز قبلاً اعلام شده‌اند تکرار نمی‌شوند).")
        except alerts.TelegramError as e:
            st.error(str(e))

    st.divider()
    st.markdown("**هشدار خودکار**")
    st.caption("تا وقتی برنامه باز است (حتی اگر مرورگر بسته باشد)، در ساعات بازار (شنبه تا چهارشنبه "
               "۹:۰۰ تا ۱۲:۴۵) هر چند دقیقه بازار بررسی و نمادهای جدید ارسال می‌شوند. هر نماد روزی یک بار.")
    on = st.toggle("فعال", cfg["alerts_enabled"])
    if on != cfg["alerts_enabled"]:
        new_cfg["alerts_enabled"] = on
        config.save(new_cfg)
        watcher.start() if on else watcher.stop()
        st.rerun()
    status = "🟢 در حال اجرا" if watcher.running else "⚪ متوقف"
    last = watcher.last_run.strftime("%H:%M") if watcher.last_run else "—"
    st.markdown(f"وضعیت: **{status}** · آخرین بررسی: {last} · پیام‌های آخرین بررسی: {watcher.last_sent}")
    if watcher.last_error:
        st.error(f"خطای آخرین بررسی: {watcher.last_error}")
    if origin == "نمونه":
        st.caption("هشدار خودکار همیشه از داده زنده TSETMC استفاده می‌کند، نه داده نمونه.")

# -------------------------------------------------------------------- help
with tabs[6]:
    st.markdown(f"""
### امتیاز نوسان‌گیری (۰ تا ۱۰۰)
| عامل | منطق |
|---|---|
| روند | قیمت بالای EMA20، چیدمان EMA20 > EMA50، بالای SMA200 |
| مومنتوم | RSI بین ۵۰ تا ۶۸ مثبت، بالای ۷۵ منفی؛ هیستوگرام و کراس MACD |
| حجم | حجم بیش از ۱.۵ برابر میانگین ۲۰ روزه همراه جهت قیمت |
| شکست | عبور از سقف ۲۰ روزه و باند بالای بولینگر |
| پول حقیقی | قدرت خریدار حقیقی (سرانه خرید ÷ سرانه فروش) و خالص ورود پول |
| فیلترها | صف فروش و نقدشوندگی پایین امتیاز را کم می‌کنند |

### امتیاز بلندمدت (۰ تا ۱۰۰)
رتبه نسبی هر نماد در کل بازار: ارزندگی ۳۵٪ (بازده سود و P/E نسبت به گروه)، روند ۲۵٪، ثبات ۱۵٪،
اندازه و نقدشوندگی ۱۵٪، فاصله از سقف ۱۰٪. شرکت‌های زیان‌ده حداکثر ۳۰ امتیاز می‌گیرند.

### محدودیت‌ها
- EPS از TSETMC گرفته می‌شود (پیش‌بینی یا آخرین گزارش شرکت) و ممکن است قدیمی باشد؛ برای تحلیل دقیق
  بنیادی صورت‌های مالی را در [کدال](https://codal.ir) ببینید.
- قیمت‌ها برای افزایش سرمایه و سود نقدی تعدیل می‌شوند؛ تعدیل تقریبی است.
- نخستین دریافت همه نمادها چند دقیقه طول می‌کشد؛ سابقه‌ها ۶ ساعت ذخیره می‌شوند.
- تنظیمات، کش و سابقه هشدارها در این پوشه نگه داشته می‌شوند: `{config.data_dir()}`
- فروش ماهانه از «گزارش فعالیت ماهانه» کدال خوانده می‌شود؛ قالب گزارش بانک‌ها، بیمه‌ها و شرکت‌های
  سرمایه‌گذاری متفاوت است و ممکن است خوانده نشود.
""")
