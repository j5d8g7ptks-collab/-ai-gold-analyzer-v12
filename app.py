import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

from engine_v12_fast import prepare, train_model, predict_latest, backtest

st.set_page_config(page_title="AI Gold Analyzer V12", layout="wide")

st.title("📈 AI GOLD ANALYZER V12")
st.caption("XAUUSD real market data / research & paper-trading dashboard — no real orders")

c1, c2, c3 = st.columns(3)

with c1:
    interval = st.selectbox("الفريم", ["5m", "15m", "1h"], index=0)

with c2:
    period = st.selectbox("الفترة", ["5d", "1mo", "3mo"], index=0)

with c3:
    refresh = st.button("🔄 تحديث بيانات الذهب")


@st.cache_data(ttl=60, show_spinner=False)
def load_gold(interval_value, period_value):
    df = yf.download(
        "XAUUSD=X",
        period=period_value,
        interval=interval_value,
        auto_adjust=False,
        progress=False,
        threads=False,
        timeout=15,
        multi_level_index=False,
    )

    if df is None or df.empty:
        raise ValueError("لم تصل بيانات XAUUSD من مزود البيانات.")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            c[0] if isinstance(c, tuple) else c
            for c in df.columns
        ]

    df.columns = [str(c).strip().lower() for c in df.columns]

    needed = ["open", "high", "low", "close"]
    missing = [c for c in needed if c not in df.columns]

    if missing:
        raise ValueError("بيانات ناقصة: " + ", ".join(missing))

    df = df[needed].copy()
    df = df.dropna()
    df["time"] = df.index

    return df


if refresh:
    load_gold.clear()

try:
    df = load_gold(interval, period)

    st.success(
        f"تم جلب {len(df):,} شمعة حقيقية لـ XAUUSD من Yahoo Finance."
    )
    st.caption(f"آخر شمعة: {df['time'].iloc[-1]}")

    prepared = prepare(df)

    features = [
        "f_trend",
        "f_ema_spread",
        "f_rsi",
        "f_adx",
        "f_bos",
        "f_choch",
        "f_fvg",
        "f_sweep",
        "f_ob",
        "f_sr",
    ]

    model, calibration, test = train_model(
        prepared, features
    )

    pred = predict_latest(
        model,
        calibration,
        prepared,
        features
    )

    stats, trades = backtest(
        prepared,
        model,
        calibration,
        features
    )

    st.subheader("الإشارة الحالية")

    m1, m2, m3, m4 = st.columns(4)

    m1.metric("Signal", pred["signal"])
    m2.metric("Confidence", f'{pred["confidence"]:.1f}%')
    m3.metric("Entry", f'{pred["entry"]:.2f}')
    m4.metric("Trend", pred["state"])

    p1, p2, p3 = st.columns(3)

    p1.metric("Stop Loss", f'{pred["sl"]:.2f}')
    p2.metric("Take Profit", f'{pred["tp"]:.2f}')
    p3.metric("Model Score", f'{pred["score"] * 100:.1f}%')

    st.subheader("اختبار تاريخي على نفس البيانات")

    b1, b2, b3, b4 = st.columns(4)

    b1.metric("Win Rate", f'{stats["win_rate"]:.1f}%')
    b2.metric("Trades", str(stats["trades"]))
    b3.metric("Profit Factor", f'{stats["profit_factor"]:.2f}')
    b4.metric("Max Drawdown (R)", f'{stats["max_dd"]:.2f}')

    st.subheader("آخر الشموع")
    st.dataframe(df.tail(20), use_container_width=True)

    st.warning(
        "تنبيه: هذه أداة بحث وتجربة فقط. بيانات Yahoo قد تكون متأخرة أو تختلف عن وسيطك، "
        "والإشارة ليست ضماناً للربح. لا تستخدمها لأوامر حقيقية دون مصدر بيانات وتنفيذ موثوقين."
    )

except Exception as e:
    st.error("تعذر تحميل/تحليل بيانات الذهب.")
    st.code(str(e))
    st.info("إذا رجع خطأ، أرسل لي صورة الخطأ فقط وأنا أصلحه لك.")
