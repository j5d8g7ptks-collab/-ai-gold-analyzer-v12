import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.ensemble import RandomForestClassifier

from engine_v12_fast import prepare, train_model

st.set_page_config(page_title="AI Gold Analyzer V13", layout="wide")

st.title("📈 AI GOLD ANALYZER V13")
st.caption("Multi-Timeframe XAUUSD research dashboard — H1 direction + M15 context + M1 precision entry")

c1, c2 = st.columns(2)
with c1:
    threshold = st.selectbox("قوة الفرصة", ["65%", "70%", "75%"], index=0)
with c2:
    refresh = st.button("🔄 تحديث الذهب")

THRESHOLD = float(threshold.rstrip("%")) / 100.0
RR = 2.0
HORIZON_M1 = 40
FEATURES = [
    "f_trend", "f_ema_spread", "f_rsi", "f_adx", "f_bos",
    "f_choch", "f_fvg", "f_sweep", "f_ob", "f_sr"
]

@st.cache_data(ttl=30, show_spinner=False)
def load_bars(interval, limit=1000):
    r = requests.get(
        "https://biquote.io/api/XAUUSD/ohlc",
        params={"interval": interval, "limit": min(int(limit), 1000)},
        headers={"User-Agent": "AI-Gold-Analyzer-V13/1.0"},
        timeout=20,
    )
    r.raise_for_status()
    data = r.json()
    bars = data.get("bars") or []
    if not bars:
        raise ValueError(f"BiQuote returned no {interval} XAUUSD bars.")

    # Build the frame only from the OHLC fields we need. This avoids pandas
    # duplicate-column issues that can make pd.to_numeric receive a DataFrame.
    rows = []
    for b in bars:
        if not isinstance(b, dict):
            continue
        rows.append({
            "time": b.get("openTime", b.get("timestamp", b.get("time"))),
            "open": b.get("open"),
            "high": b.get("high"),
            "low": b.get("low"),
            "close": b.get("close"),
            "volume": b.get("tickVolume", b.get("volume", 0)),
            "isOpen": b.get("isOpen", False),
        })

    df = pd.DataFrame.from_records(rows)
    need = ["time", "open", "high", "low", "close"]
    missing = [x for x in need if x not in df.columns]
    if missing:
        raise ValueError(f"Missing OHLC fields: {missing}")

    # BiQuote returns ISO timestamps such as 2026-02-24T13:00:00Z.
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")

    # Each conversion is guaranteed to receive a single Series.
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = (
        df.dropna(subset=need)
          .drop_duplicates(subset=["time"])
          .sort_values("time")
    )

    # Do not feed the still-open candle to the model.
    if "isOpen" in df.columns:
        df = df[df["isOpen"].fillna(False) == False]

    if len(df) < 120:
        raise ValueError(f"BiQuote returned only {len(df)} closed {interval} bars.")

    return df.set_index("time")


def trend_state(df):
    x = df.copy()
    x["ema20"] = x["close"].ewm(span=20, adjust=False).mean()
    x["ema50"] = x["close"].ewm(span=50, adjust=False).mean()
    if len(x) < 50:
        return "Mixed"
    e20, e50 = float(x["ema20"].iloc[-1]), float(x["ema50"].iloc[-1])
    slope = float(x["ema20"].iloc[-1] - x["ema20"].iloc[-6])
    if e20 > e50 and slope > 0:
        return "Bullish"
    if e20 < e50 and slope < 0:
        return "Bearish"
    return "Mixed"

def add_context(m1, m15, h1):
    x = m1.copy()
    x["h1_trend"] = h1["trend"].reindex(x.index, method="ffill")
    x["m15_trend"] = m15["trend"].reindex(x.index, method="ffill")
    return x

def evaluate_signal_rows(df, model, threshold, rr=2.0, horizon=40):
    d = df.dropna(subset=FEATURES + ["atr"]).copy()
    if len(d) < 120:
        return pd.DataFrame()

    cut = max(1, int(len(d) * 0.70))
    end = max(cut, len(d) - horizon)
    if end <= cut:
        return pd.DataFrame()

    segment = d.iloc[cut:end]
    probs = model.predict_proba(segment[FEATURES])
    classes = list(model.classes_)
    if 1 not in classes:
        return pd.DataFrame()
    probs = probs[:, classes.index(1)]

    rows = []
    for k, p in enumerate(probs):
        i = cut + k
        row = d.iloc[i]
        h1 = row.get("h1_trend", "Mixed")
        m15 = row.get("m15_trend", "Mixed")

        sig = "BUY" if p >= threshold else ("SELL" if p <= 1.0 - threshold else "WAIT")
        if sig == "BUY" and h1 == "Bearish":
            sig = "WAIT"
        if sig == "SELL" and h1 == "Bullish":
            sig = "WAIT"
        if sig == "BUY" and float(row["ema20"]) <= float(row["ema50"]):
            sig = "WAIT"
        if sig == "SELL" and float(row["ema20"]) >= float(row["ema50"]):
            sig = "WAIT"
        if sig == "BUY" and float(row["adx"]) < 18:
            sig = "WAIT"
        if sig == "SELL" and float(row["adx"]) < 18:
            sig = "WAIT"

        if sig == "WAIT":
            continue

        entry = float(row["close"])
        risk = max(float(row["atr"]) * 1.2, entry * 0.0005)
        sl = entry - risk if sig == "BUY" else entry + risk
        tp = entry + risk * rr if sig == "BUY" else entry - risk * rr

        future = d.iloc[i + 1:min(i + 1 + horizon, len(d))]
        result = "OPEN"
        for _, bar in future.iterrows():
            if sig == "BUY":
                hit_sl = float(bar["low"]) <= sl
                hit_tp = float(bar["high"]) >= tp
            else:
                hit_sl = float(bar["high"]) >= sl
                hit_tp = float(bar["low"]) <= tp

            if hit_sl and hit_tp:
                result = "LOSS"  # conservative when both occur in one candle
                break
            if hit_sl:
                result = "LOSS"
                break
            if hit_tp:
                result = "WIN"
                break

        if result == "OPEN":
            continue

        rows.append({
            "وقت الإشارة": row.name,
            "الإشارة": sig,
            "الثقة": round(max(float(p), 1 - float(p)) * 100, 1),
            "الدخول": round(entry, 2),
            "SL": round(sl, 2),
            "TP": round(tp, 2),
            "الاتجاه H1": h1,
            "السياق M15": m15,
            "النتيجة": result,
        })

    return pd.DataFrame(rows)

if refresh:
    load_bars.clear()

try:
    m1 = load_bars("1m", 1000)
    m5 = load_bars("5m", 1000)
    m15 = load_bars("15m", 1000)
    h1 = load_bars("1h", 1000)

    st.success(
        f"XAUUSD عبر BiQuote/MT5 feed: M1={len(m1):,} | M5={len(m5):,} | "
        f"M15={len(m15):,} | H1={len(h1):,}"
    )
    st.caption("المصدر: BiQuote XAUUSD OHLC من تغذية MT5، ويدعم M1/M5/M15/H1 بدون API key.")

    # Higher-timeframe direction/context
    h1_ctx = pd.DataFrame(index=h1.index)
    h1_ctx["trend"] = [
        "Bullish" if (h1["close"].ewm(span=20, adjust=False).mean().iloc[i] >
                      h1["close"].ewm(span=50, adjust=False).mean().iloc[i])
        else "Bearish"
        for i in range(len(h1))
    ]
    m15_ctx = pd.DataFrame(index=m15.index)
    m15_ctx["trend"] = [
        "Bullish" if (m15["close"].ewm(span=20, adjust=False).mean().iloc[i] >
                      m15["close"].ewm(span=50, adjust=False).mean().iloc[i])
        else "Bearish"
        for i in range(len(m15))
    ]

    prepared = prepare(m1)
    if len(prepared.dropna(subset=FEATURES + ["target"])) < 100:
        raise ValueError("Not enough valid M1 rows to train the model.")

    # Current model uses all available M1 history; historical score uses a 70/30 split.
    model, calibration, test = train_model(prepared, FEATURES)

    valid = prepared.dropna(subset=FEATURES).copy()
    last = valid.iloc[-1]
    p = float(model.predict_proba(last[FEATURES].to_frame().T)[0, list(model.classes_).index(1)])

    h1_direction = trend_state(h1)
    m15_direction = trend_state(m15)
    m1_direction = (
        "Bullish" if last["ema20"] > last["ema50"] else
        "Bearish" if last["ema20"] < last["ema50"] else "Mixed"
    )

    signal = "BUY" if p >= THRESHOLD else ("SELL" if p <= 1 - THRESHOLD else "WAIT")
    if signal == "BUY" and h1_direction == "Bearish":
        signal = "WAIT"
    if signal == "SELL" and h1_direction == "Bullish":
        signal = "WAIT"
    if signal == "BUY" and m1_direction != "Bullish":
        signal = "WAIT"
    if signal == "SELL" and m1_direction != "Bearish":
        signal = "WAIT"
    if signal in ("BUY", "SELL") and float(last["adx"]) < 18:
        signal = "WAIT"

    entry = float(last["close"])
    atr = float(last["atr"])
    risk = max(atr * 1.2, entry * 0.0005)
    if signal == "BUY":
        sl, tp = entry - risk, entry + RR * risk
    elif signal == "SELL":
        sl, tp = entry + risk, entry - RR * risk
    else:
        sl, tp = entry - risk, entry + risk

    st.subheader("🎯 الفرصة الحالية")
    a, b, c, d = st.columns(4)
    a.metric("Signal", signal)
    b.metric("Confidence", f"{max(p,1-p)*100:.1f}%")
    c.metric("Entry M1", f"{entry:.2f}")
    d.metric("وقت الإشارة", last.name.strftime("%Y-%m-%d %H:%M:%S UTC"))

    e, f, g = st.columns(3)
    e.metric("H1 الاتجاه", h1_direction)
    f.metric("M15 السياق", m15_direction)
    g.metric("M1 الاتجاه", m1_direction)

    x1, x2, x3 = st.columns(3)
    x1.metric("Stop Loss", f"{sl:.2f}")
    x2.metric("Take Profit", f"{tp:.2f}")
    x3.metric("ADX M1", f"{float(last['adx']):.1f}")

    if signal == "WAIT":
        st.warning("⚪ لا توجد فرصة واضحة الآن. لن نقترح صفقة حتى تتجاوز الثقة الحد المطلوب وتكون مع اتجاه H1 وM1.")
    else:
        st.success(f"🟢 فرصة {signal} واضحة — دخول دقيق من M1، مع اتجاه H1.")

    # Attach higher-timeframe context for historical signal scoring.
    bt = prepared.copy()
    bt["h1_trend"] = h1_ctx["trend"].reindex(bt.index, method="ffill")
    bt["m15_trend"] = m15_ctx["trend"].reindex(bt.index, method="ffill")

    # Train a historical model on the first 70% only, then score the unseen 30%.
    clean = bt.dropna(subset=FEATURES + ["target"]).copy()
    split = int(len(clean) * 0.70)
    hist_train = clean.iloc[:split]
    hist_test = clean.iloc[split:]
    if len(hist_train) < 100:
        raise ValueError("Not enough historical training rows.")
    hist_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=7,
        min_samples_leaf=10,
        random_state=42,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )
    hist_model.fit(hist_train[FEATURES], hist_train["target"].astype(int))

    hist_df = evaluate_signal_rows(bt, hist_model, THRESHOLD, RR, HORIZON_M1)
    wins = int((hist_df["النتيجة"] == "WIN").sum()) if not hist_df.empty else 0
    losses = int((hist_df["النتيجة"] == "LOSS").sum()) if not hist_df.empty else 0
    total = wins + losses
    winrate = (wins / total * 100) if total else 0.0

    st.subheader("📊 سجل الصفقات المقترحة")
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("عدد الصفقات", str(total))
    q2.metric("صح ✅", str(wins))
    q3.metric("غلط ❌", str(losses))
    q4.metric("نسبة الصح", f"{winrate:.1f}%")

    if not hist_df.empty:
        st.dataframe(hist_df.tail(20), use_container_width=True)
    else:
        st.info("لم تظهر فرص واضحة كافية في فترة الاختبار حسب الشروط الحالية.")

    st.subheader("🕒 آخر تحديث")
    st.caption(f"آخر شمعة M1: {m1.index[-1].strftime('%Y-%m-%d %H:%M:%S UTC')}")
    st.caption(f"السعر الحالي المرجعي من آخر شمعة M1: {entry:.2f}")

    st.warning(
        "تنبيه: هذا نظام بحث وPaper Trading. بيانات BiQuote موصوفة بأنها من تغذية MT5، "
        "لكنها ليست بالضرورة نفس وسيطك/السيرفر؛ طابق Bid/Ask مع MT5 قبل أي تداول حقيقي."
    )

except requests.RequestException as e:
    st.error("تعذر الاتصال بمصدر XAUUSD.")
    st.code(str(e))
except Exception as e:
    st.error("تعذر تحميل/تحليل بيانات الذهب.")
    st.code(str(e))
