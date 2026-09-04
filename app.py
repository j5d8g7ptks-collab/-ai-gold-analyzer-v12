import streamlit as st
import pandas as pd
import numpy as np
import requests

from engine_v12_fast import prepare, train_model, predict_latest, backtest

st.set_page_config(page_title="AI Gold Analyzer V12", layout="wide")

st.title("📈 AI GOLD ANALYZER V12")
st.caption("XAU/USD spot reference data / research & paper-trading dashboard — no real orders")

c1, c2, c3 = st.columns(3)

with c1:
    interval = st.selectbox("الفريم", ["5m", "15m", "1h"], index=0)

with c2:
    period = st.selectbox("الفترة", ["5d", "1mo", "3mo"], index=0)

with c3:
    refresh = st.button("🔄 تحديث بيانات الذهب")


@st.cache_data(ttl=30, show_spinner=False)
def load_gold(interval_value, period_value):
    """Load XAU/USD reference spot prices from XAUS and build candles."""
    del period_value  # XAUS anonymous intraday endpoint is limited to 48 hours.

    interval_map = {
        "5m": "5min",
        "15m": "15min",
        "1h": "1h",
    }
    rule = interval_map[interval_value]

    url = "https://xaus.com/api/v1/intraday"
    response = requests.get(
        url,
        params={"symbol": "xau", "hours": 48},
        headers={"User-Agent": "AI-Gold-Analyzer-V12/1.0"},
        timeout=20,
    )
    response.raise_for_status()
    payload = response.json()

    # XAUS documents the intraday series under the top-level "points" key.
    # Keep compatibility with older/alternate response shapes too.
    points = (
        payload.get("points")
        or payload.get("data")
        or payload.get("prices")
        or []
    )
    if not points:
        state = payload.get("data_state", {})
        detail = payload.get("error") or state.get("status") or "no points"
        raise ValueError(f"XAUS returned no intraday XAU/USD points ({detail}).")

    rows = []
    for item in points:
        if not isinstance(item, dict):
            continue
        ts = item.get("t")
        price = item.get("p")
        if ts is None or price is None:
            continue
        try:
            # XAUS numeric timestamps are Unix seconds. Handle milliseconds too
            # so the parser remains robust if the API response format changes.
            if isinstance(ts, (int, float, np.integer, np.floating)):
                unit = "ms" if abs(float(ts)) >= 1_000_000_000_000 else "s"
                dt = pd.to_datetime(ts, unit=unit, utc=True)
            else:
                dt = pd.to_datetime(ts, utc=True)
            rows.append((dt, float(price)))
        except (TypeError, ValueError, OverflowError):
            continue

    if not rows:
        raise ValueError("XAUS returned invalid price points.")

    raw = pd.DataFrame(rows, columns=["time", "price"])
    raw = raw.dropna().drop_duplicates("time").sort_values("time")
    raw = raw.set_index("time")

    if len(raw) < 20:
        raise ValueError("XAUS returned too few intraday points.")

    ohlc = raw["price"].resample(rule).ohlc().dropna()
    if ohlc.empty:
        raise ValueError("Could not build candles from XAUS prices.")

    ohlc["volume"] = 0.0
    ohlc["time"] = ohlc.index
    ohlc.attrs["source"] = "XAUS XAU/USD spot"
    ohlc.attrs["source_note"] = (
        "Reference XAU/USD mid-market prices aggregated from XAUS intraday points; "
        "not broker Bid/Ask execution prices."
    )
    return ohlc


if refresh:
    load_gold.clear()

try:
    df = load_gold(interval, period)

    st.success(
        f"تم جلب {len(df):,} شمعة من XAU/USD spot عبر XAUS."
    )
    st.caption(f"آخر شمعة: {df['time'].iloc[-1]}")
    st.caption(df.attrs.get("source_note", "XAUS reference spot data."))

    if interval == "1h" and len(df) < 100:
        st.warning(
            "الفريم 1h غير كافٍ حالياً للمحرك V12 لأن XAUS يوفر 48 ساعة فقط "
            "من البيانات اللحظية في هذا المسار. استخدم 5m أو 15m."
        )
        st.stop()

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

    model, calibration, test = train_model(prepared, features)

    pred = predict_latest(
        model,
        calibration,
        prepared,
        features,
    )

    stats, trades = backtest(
        prepared,
        model,
        calibration,
        features,
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

    st.info(
        "مصدر السعر: XAUS XAU/USD spot reference. الأسعار مرجعية mid-market "
        "وليست Bid/Ask لوسيط MT5، والشموع مبنية من نقاط لحظية مجمعة."
    )
    st.warning(
        "تنبيه: هذه أداة بحث وتجربة فقط وليست منصة تنفيذ. لا تستخدم الإشارة وحدها "
        "لأوامر حقيقية، ويجب مطابقة السعر مع وسيطك قبل أي قرار تداول."
    )

except requests.RequestException as e:
    st.error("تعذر الاتصال بمصدر XAU/USD.")
    st.code(str(e))
    st.info("أرسل لي صورة الخطأ إذا ظهر لك هذا التنبيه.")
except Exception as e:
    st.error("تعذر تحميل/تحليل بيانات الذهب.")
    st.code(str(e))
    st.info("أرسل لي صورة الخطأ فقط وأنا أصلحه لك.")

# XAUS note: reference market data, not broker execution quotes.
