import streamlit as st
import pandas as pd
import numpy as np
from   engine_v12_fixed import prepare, train_model, predict_latest, backtest, monte_carlo

st.set_page_config(page_title="AI Gold Analyzer V12", page_icon="📈", layout="wide")
st.title("📈 AI GOLD ANALYZER V12")
st.caption("XAUUSD research / paper-trading dashboard — no real orders")

uploaded = st.file_uploader("ارفع بيانات XAUUSD بصيغة CSV", type="csv")

if uploaded:
    df = pd.read_csv(uploaded)
    df.columns = [x.lower().strip() for x in df.columns]
else:
    # Synthetic data so the project opens without a data source.
    rng = np.random.default_rng(42)
    n = 6000
    ret = rng.normal(0, 0.0012, n)
    close = 2400*np.exp(np.cumsum(ret))
    high = close*(1+rng.uniform(0,.0018,n))
    low = close*(1-rng.uniform(0,.0018,n))
    open_ = np.r_[close[0], close[:-1]]
    df = pd.DataFrame({
        "time": pd.date_range("2025-01-01", periods=n, freq="5min"),
        "open": open_, "high": high, "low": low, "close": close,
        "volume": rng.integers(100, 10000, n)
    })
    st.info("يتم استخدام بيانات تجريبية. ارفع CSV للحصول على اختبار حقيقي.")

df = df.dropna(subset=["open","high","low","close"]).copy()
df = prepare(df)

feature_cols = [c for c in df.columns if c.startswith("f_")]
model, calibration, test = train_model(df, feature_cols)

latest = predict_latest(model, calibration, df, feature_cols)
stats, trades = backtest(df, model, calibration, feature_cols)
mc = monte_carlo(trades["r"].values if len(trades) else np.array([]))

c = st.columns(6)
c[0].metric("Signal", latest["signal"])
c[1].metric("Confidence", f'{latest["confidence"]:.1f}%')
c[2].metric("Win Rate", f'{stats["win_rate"]:.1f}%')
c[3].metric("Trades", stats["trades"])
c[4].metric("Profit Factor", f'{stats["profit_factor"]:.2f}')
c[5].metric("Max DD", f'{stats["max_dd"]:.2f} R')

st.subheader("🎯 آخر إشارة")
a,b,c = st.columns(3)
a.metric("Entry", f'{latest["entry"]:.2f}')
b.metric("Stop Loss", f'{latest["sl"]:.2f}')
c.metric("Take Profit", f'{latest["tp"]:.2f}')

st.write("**Market state:**", latest["state"])
st.write("**Model score:**", f'{latest["score"]:.3f}')

st.subheader("📊 السعر والمؤشرات")
st.line_chart(df.tail(500)[["close","ema20","ema50","ema200"]])

st.subheader("🧪 Out-of-sample test")
st.write({
    "test_rows": len(test),
    "calibration_samples": len(calibration),
    "Monte Carlo median final R": round(mc.get("median_final_r",0),2),
    "Monte Carlo 5% final R": round(mc.get("p05_final_r",0),2),
    "Monte Carlo 95% final R": round(mc.get("p95_final_r",0),2),
})

st.subheader("آخر الصفقات")
st.dataframe(trades.tail(50), use_container_width=True)

st.warning("هذه أداة بحث وتعليم وليست توصية مالية. لا تستخدم النتائج أو Confidence كضمان للربح.")
