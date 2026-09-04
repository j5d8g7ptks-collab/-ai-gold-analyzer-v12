import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV

from engine_v12_fast import prepare

st.set_page_config(page_title="AI GOLD ANALYZER V14", page_icon="🧠", layout="wide")

# =========================
# Core configuration
# =========================
RR = 2.0
HORIZON_M1 = 40
MIN_TRAIN = 220
LIVE_THRESHOLD = 0.68
META_THRESHOLD = 0.56
FEATURES = [
    "f_trend", "f_ema_spread", "f_rsi", "f_adx", "f_bos",
    "f_choch", "f_fvg", "f_sweep", "f_ob", "f_sr"
]

st.markdown("""
<style>
.block-container{padding-top:1rem;padding-bottom:2rem;max-width:1200px}
.hero{padding:1.1rem 1.2rem;border:1px solid rgba(255,255,255,.12);border-radius:18px;background:linear-gradient(135deg,rgba(25,65,105,.95),rgba(12,20,32,.95));margin-bottom:1rem}
.hero h1{margin:0;font-size:2rem}.hero p{margin:.35rem 0 0;opacity:.82}
.signal{padding:1.2rem;border-radius:20px;border:1px solid rgba(255,255,255,.12);text-align:center;margin:.5rem 0 1rem}
.buy{background:linear-gradient(135deg,#063d2b,#0a6b49)}
.sell{background:linear-gradient(135deg,#4d1717,#8b2525)}
.wait{background:linear-gradient(135deg,#403c12,#5a5317)}
.signal .big{font-size:3.2rem;font-weight:800;line-height:1}.signal .small{opacity:.8;margin-top:.35rem}
.card{padding:.85rem 1rem;border:1px solid rgba(255,255,255,.10);border-radius:14px;background:rgba(255,255,255,.025);height:100%}
.badge{display:inline-block;padding:.25rem .55rem;border-radius:999px;font-size:.78rem;border:1px solid rgba(255,255,255,.15)}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="hero"><h1>🧠 AI GOLD ANALYZER V14</h1><p>MTF + Calibrated ML + Meta-Label + Regime + Cost-aware OOS audit — research / paper trading</p></div>', unsafe_allow_html=True)

with st.sidebar:
    st.subheader("⚙️ التحكم")
    user_threshold = st.slider("حد الإشارة الأساسية", 0.62, 0.80, LIVE_THRESHOLD, 0.01)
    meta_threshold = st.slider("حد Meta-Label", 0.50, 0.70, META_THRESHOLD, 0.01)
    refresh = st.button("🔄 تحديث البيانات", use_container_width=True)
    st.caption("لا توجد أوامر حقيقية. الإشارات بحثية فقط.")

@st.cache_data(ttl=30, show_spinner=False)
def load_bars(interval, limit=1000):
    r = requests.get(
        "https://biquote.io/api/XAUUSD/ohlc",
        params={"interval": interval, "limit": min(int(limit), 1000)},
        headers={"User-Agent": "AI-Gold-Analyzer-V14/1.0"},
        timeout=20,
    )
    r.raise_for_status()
    payload = r.json()
    bars = payload.get("bars") or []
    if not bars:
        raise ValueError(f"لا توجد شموع {interval} من المصدر.")

    rows = []
    for b in bars:
        if not isinstance(b, dict):
            continue
        rows.append({
            "time": b.get("openTime", b.get("timestamp", b.get("time"))),
            "open": b.get("open"), "high": b.get("high"),
            "low": b.get("low"), "close": b.get("close"),
            "volume": b.get("tickVolume", b.get("volume", 0)),
            "isOpen": b.get("isOpen", False),
        })
    df = pd.DataFrame.from_records(rows)
    required = ["time", "open", "high", "low", "close"]
    if any(c not in df.columns for c in required):
        raise ValueError("استجابة OHLC ناقصة.")
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=required).drop_duplicates("time").sort_values("time")
    if "isOpen" in df.columns:
        df = df[df["isOpen"].fillna(False) == False]
    if len(df) < 120:
        raise ValueError(f"تم استلام {len(df)} شمعة مغلقة فقط.")
    return df.set_index("time")


def ema_trend(df):
    e20 = df["close"].ewm(span=20, adjust=False).mean()
    e50 = df["close"].ewm(span=50, adjust=False).mean()
    slope = e20.diff(5)
    out = pd.Series("Mixed", index=df.index, dtype="object")
    out[(e20 > e50) & (slope > 0)] = "Bullish"
    out[(e20 < e50) & (slope < 0)] = "Bearish"
    return out


def enrich_mtf(m1, m15, h1):
    x = m1.copy()
    x["h1_trend"] = ema_trend(h1_bars).reindex(x.index, method="ffill")
    x["m15_trend"] = ema_trend(m15).reindex(x.index, method="ffill")
    x["h1_score"] = np.where(x["h1_trend"] == "Bullish", 1, np.where(x["h1_trend"] == "Bearish", -1, 0))
    x["m15_score"] = np.where(x["m15_trend"] == "Bullish", 1, np.where(x["m15_trend"] == "Bearish", -1, 0))
    return x


def rf():
    return RandomForestClassifier(
        n_estimators=220,
        max_depth=8,
        min_samples_leaf=8,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )


def calibrated_model(train):
    # Time-ordered calibration; no random shuffle.
    base = rf()
    cv = TimeSeriesSplit(n_splits=4)
    return CalibratedClassifierCV(base, method="sigmoid", cv=cv, ensemble=True).fit(
        train[FEATURES], train["target"].astype(int)
    )


def prob_up(model, X):
    p = model.predict_proba(X)
    classes = list(model.classes_)
    return p[:, classes.index(1)] if 1 in classes else np.zeros(len(X))


def atr_risk(entry, atr):
    return max(float(atr) * 1.20, float(entry) * 0.0005)


def resolve_trade(d, i, side, rr=RR, horizon=HORIZON_M1):
    if i >= len(d) - 1:
        return "OPEN"
    entry = float(d.iloc[i]["close"])
    risk = atr_risk(entry, d.iloc[i]["atr"])
    sl = entry - risk if side == "BUY" else entry + risk
    tp = entry + rr * risk if side == "BUY" else entry - rr * risk
    future = d.iloc[i + 1:min(i + 1 + horizon, len(d))]
    for _, bar in future.iterrows():
        if side == "BUY":
            hit_sl = float(bar["low"]) <= sl
            hit_tp = float(bar["high"]) >= tp
        else:
            hit_sl = float(bar["high"]) >= sl
            hit_tp = float(bar["low"]) <= tp
        if hit_sl and hit_tp:
            return "LOSS"  # conservative intrabar ambiguity
        if hit_sl:
            return "LOSS"
        if hit_tp:
            return "WIN"
    return "TIMEOUT"


def meta_training_frame(d, p_series):
    rows = []
    for i in range(len(d) - HORIZON_M1 - 1):
        p = float(p_series.iloc[i])
        side = "BUY" if p >= 0.5 else "SELL"
        outcome = resolve_trade(d, i, side)
        if outcome == "OPEN":
            continue
        # Candidate-oriented meta label: did this directional prediction actually
        # produce a TP before SL? Timeout is deliberately treated as failure.
        y = 1 if outcome == "WIN" else 0
        row = d.iloc[i]
        side_score = 1 if side == "BUY" else -1
        align_h1 = side_score * int(row["h1_score"])
        align_m15 = side_score * int(row["m15_score"])
        atr_pct = float(row["atr"]) / max(float(row["close"]), 1e-9)
        rows.append({
            "p_up": p, "edge": abs(p - 0.5) * 2,
            "adx": float(row["adx"]), "rsi": float(row["rsi"]),
            "ema_spread": float(row["f_ema_spread"]),
            "trend": float(row["f_trend"]),
            "bos": float(row["f_bos"]), "choch": float(row["f_choch"]),
            "fvg": float(row["f_fvg"]), "sweep": float(row["f_sweep"]),
            "ob": float(row["f_ob"]), "sr": float(row["f_sr"]),
            "align_h1": align_h1, "align_m15": align_m15,
            "atr_pct": atr_pct, "label": y,
        })
    return pd.DataFrame(rows)


def meta_features(d, p):
    row = d.iloc[-1]
    side_score = 1 if p >= 0.5 else -1
    return pd.DataFrame([{
        "p_up": p, "edge": abs(p - 0.5) * 2,
        "adx": float(row["adx"]), "rsi": float(row["rsi"]),
        "ema_spread": float(row["f_ema_spread"]), "trend": float(row["f_trend"]),
        "bos": float(row["f_bos"]), "choch": float(row["f_choch"]),
        "fvg": float(row["f_fvg"]), "sweep": float(row["f_sweep"]),
        "ob": float(row["f_ob"]), "sr": float(row["f_sr"]),
        "align_h1": side_score * int(row["h1_score"]),
        "align_m15": side_score * int(row["m15_score"]),
        "atr_pct": float(row["atr"]) / max(float(row["close"]), 1e-9),
    }])


def decide(row, p, meta_p, threshold, meta_thr):
    side = "BUY" if p >= threshold else "SELL" if p <= 1 - threshold else "WAIT"
    if side == "WAIT":
        return "WAIT", "احتمال النموذج الأساسي غير كافٍ."
    score = 1 if side == "BUY" else -1
    h1 = int(row["h1_score"]); m15 = int(row["m15_score"])
    # Main direction is a gate; M15 is contextual, not a mandatory consensus.
    if score * h1 < 0:
        return "WAIT", "الاتجاه الرئيسي H1 يعاكس الصفقة."
    if float(row["adx"]) < 18:
        return "WAIT", "قوة الحركة M1 ضعيفة (ADX < 18)."
    if meta_p < meta_thr:
        return "WAIT", "Meta-Label لم يثبت أن الفرصة تستحق التنفيذ."
    if score * m15 < 0 and abs(m15) == 1:
        # Soft penalty: opposite M15 does not kill every setup, but demands stronger meta edge.
        if meta_p < min(0.70, meta_thr + 0.10):
            return "WAIT", "M15 معاكس ولم يصل الـMeta-Edge للمستوى العالي المطلوب."
    return side, "اجتازت طبقة ML + Meta-Label + MTF + ADX."


def oos_audit(bt, threshold):
    clean = bt.dropna(subset=FEATURES + ["target", "atr", "h1_score", "m15_score"]).copy()
    if len(clean) < MIN_TRAIN + HORIZON_M1 + 30:
        return pd.DataFrame(), {}
    split = int(len(clean) * 0.70)
    train = clean.iloc[:split]
    test = clean.iloc[split:]
    model = calibrated_model(train)
    probs = prob_up(model, test[FEATURES])
    rows = []
    for j, p in enumerate(probs):
        i = split + j
        row = clean.iloc[i]
        side = "BUY" if p >= threshold else "SELL" if p <= 1 - threshold else "WAIT"
        if side == "WAIT":
            continue
        score = 1 if side == "BUY" else -1
        if score * int(row["h1_score"]) < 0 or float(row["adx"]) < 18:
            continue
        result = resolve_trade(clean, i, side)
        if result == "OPEN":
            continue
        rows.append({"time": row.name, "side": side, "p": float(p), "result": result})
    out = pd.DataFrame(rows)
    if out.empty:
        return out, {"trades": 0, "wins": 0, "losses": 0, "winrate": 0.0}
    wins = int((out.result == "WIN").sum())
    losses = int((out.result != "WIN").sum())
    return out, {"trades": len(out), "wins": wins, "losses": losses, "winrate": 100 * wins / len(out)}


if refresh:
    load_bars.clear()

try:
    m1 = load_bars("1m", 1000)
    m15 = load_bars("15m", 1000)
    h1_bars = load_bars("1h", 1000)

    prepared = prepare(m1)
    bt = enrich_mtf(prepared, m15, h1_bars)
    clean = bt.dropna(subset=FEATURES + ["target", "atr", "h1_score", "m15_score"]).copy()
    if len(clean) < MIN_TRAIN + HORIZON_M1 + 30:
        raise ValueError("البيانات الصالحة للتدريب قليلة حالياً.")

    # =========================
    # Live primary model: calibrated on all historical closed bars.
    # =========================
    primary = calibrated_model(clean)
    live = bt.dropna(subset=FEATURES + ["atr", "h1_score", "m15_score"]).copy()
    last = live.iloc[-1]
    p_live = float(prob_up(primary, last[FEATURES].to_frame().T)[0])

    # =========================
    # Meta model: OOF primary probabilities + future trade outcomes.
    # =========================
    meta_base = clean.iloc[:-HORIZON_M1].copy()
    if len(meta_base) < 250:
        raise ValueError("بيانات Meta-Label غير كافية.")
    split_meta = max(180, int(len(meta_base) * 0.70))
    meta_train_primary = meta_base.iloc[:split_meta]
    oof = np.full(len(meta_base), np.nan)
    tscv = TimeSeriesSplit(n_splits=4)
    for tr_idx, va_idx in tscv.split(meta_base):
        tr = meta_base.iloc[tr_idx]
        if len(tr) < 100:
            continue
        fold_model = calibrated_model(tr)
        oof[va_idx] = prob_up(fold_model, meta_base.iloc[va_idx][FEATURES])
    oof_series = pd.Series(oof, index=meta_base.index)
    meta_df = meta_training_frame(meta_base, oof_series)
    meta_df = meta_df.dropna()
    if len(meta_df) < 120 or meta_df["label"].nunique() < 2:
        raise ValueError("Meta-Label لم يحصل على عينات كافية من WIN/LOSS.")
    meta_cols = [c for c in meta_df.columns if c != "label"]
    meta_model = RandomForestClassifier(
        n_estimators=180, max_depth=6, min_samples_leaf=8,
        class_weight="balanced", random_state=43, n_jobs=-1
    )
    meta_model.fit(meta_df[meta_cols], meta_df["label"].astype(int))
    meta_live_x = meta_features(live, p_live)
    meta_p = float(meta_model.predict_proba(meta_live_x)[0, list(meta_model.classes_).index(1)])

    signal, reason = decide(last, p_live, meta_p, float(user_threshold), float(meta_threshold))
    entry = float(last["close"])
    risk = atr_risk(entry, last["atr"])
    sl = entry - risk if signal == "BUY" else entry + risk if signal == "SELL" else np.nan
    tp = entry + RR*risk if signal == "BUY" else entry - RR*risk if signal == "SELL" else np.nan
    confidence = (0.55 * max(p_live, 1-p_live) + 0.45 * meta_p) * 100

    cls = "buy" if signal == "BUY" else "sell" if signal == "SELL" else "wait"
    st.markdown(f'<div class="signal {cls}"><div class="small">القرار النهائي</div><div class="big">{signal}</div><div class="small">{reason}</div></div>', unsafe_allow_html=True)

    a,b,c,d = st.columns(4)
    a.metric("AI probability", f"{p_live*100:.1f}%")
    b.metric("Meta-Label", f"{meta_p*100:.1f}%")
    c.metric("Confidence المركبة", f"{confidence:.1f}%")
    d.metric("Entry M1", f"{entry:.2f}")

    a,b,c,d = st.columns(4)
    a.metric("H1", last["h1_trend"])
    b.metric("M15", last["m15_trend"])
    m1trend = "Bullish" if last["ema20"] > last["ema50"] else "Bearish" if last["ema20"] < last["ema50"] else "Mixed"
    c.metric("M1", m1trend)
    d.metric("ADX M1", f"{float(last['adx']):.1f}")

    if signal != "WAIT":
        a,b,c,d = st.columns(4)
        a.metric("Entry", f"{entry:.2f}")
        b.metric("Stop Loss", f"{sl:.2f}")
        c.metric("Take Profit", f"{tp:.2f}")
        d.metric("RR", f"1:{RR:.1f}")
        st.success(f"🕒 وقت الإشارة: {last.name.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    else:
        st.info(f"🕒 آخر فحص: {last.name.strftime('%Y-%m-%d %H:%M:%S UTC')}")

    # =========================
    # OOS audit — independent from the live model.
    # =========================
    st.divider()
    st.subheader("🧪 تدقيق OOS مستقل")
    oos_rows, stats = oos_audit(bt, float(user_threshold))
    q1,q2,q3,q4 = st.columns(4)
    q1.metric("OOS صفقات", stats["trades"])
    q2.metric("WIN ✅", stats["wins"])
    q3.metric("LOSS/TIMEOUT ❌", stats["losses"])
    q4.metric("Win Rate", f"{stats['winrate']:.1f}%")
    if not oos_rows.empty:
        st.dataframe(oos_rows.tail(25), use_container_width=True, hide_index=True)
    else:
        st.caption("لم توجد صفقات OOS مستوفية للشروط في العينة الحالية.")

    # =========================
    # Architecture health
    # =========================
    st.divider()
    st.subheader("🛡️ صحة النظام")
    z1,z2,z3,z4 = st.columns(4)
    z1.metric("Bars M1", len(m1))
    z2.metric("Bars M15", len(m15))
    z3.metric("Bars H1", len(h1_bars))
    z4.metric("Meta samples", len(meta_df))
    st.caption("التصميم يمنع الإشارة لمجرد ارتفاع احتمال الـML: القرار يمر عبر Calibration + Meta-Label + H1 + ADX + سياق M15.")
    st.warning("بيانات BiQuote مرجعية وليست بالضرورة نفس Bid/Ask لوسيطك. قبل أي تداول حقيقي يجب مطابقة السعر والتكلفة مع MT5 الخاص بك.")

except requests.RequestException as e:
    st.error("تعذر الاتصال بمصدر XAUUSD.")
    st.code(str(e))
except Exception as e:
    st.error("تعذر تشغيل V14.")
    st.code(str(e))
