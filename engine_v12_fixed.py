import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def prepare(df):
    x = df.copy()
    x.columns = [str(c).strip().lower() for c in x.columns]

    required = {"open", "high", "low", "close"}
    missing = required - set(x.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    c, h, l = x["close"], x["high"], x["low"]

    x["ema20"] = c.ewm(span=20, adjust=False).mean()
    x["ema50"] = c.ewm(span=50, adjust=False).mean()
    x["ema200"] = c.ewm(span=200, adjust=False).mean()

    d = c.diff()
    gain = d.clip(lower=0).rolling(14).mean()
    loss = (-d.clip(upper=0)).rolling(14).mean().replace(0, np.nan)
    x["rsi"] = 100 - 100 / (1 + gain / loss)

    tr = pd.concat(
        [(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()],
        axis=1,
    ).max(axis=1)
    x["atr"] = tr.rolling(14).mean()

    plus = (h - h.shift()).clip(lower=0)
    minus = (l.shift() - l).clip(lower=0)
    atr = x["atr"].replace(0, np.nan)
    pdi = 100 * plus.rolling(14).sum() / atr.rolling(14).sum()
    mdi = 100 * minus.rolling(14).sum() / atr.rolling(14).sum()
    x["adx"] = (
        100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    ).rolling(14).mean()

    hh = h.shift(1).rolling(20).max()
    ll = l.shift(1).rolling(20).min()

    x["bos_up"] = (c > hh).astype(int)
    x["bos_down"] = (c < ll).astype(int)
    x["choch"] = (
        (x["bos_up"].diff().fillna(0) > 0).astype(int)
        - (x["bos_down"].diff().fillna(0) > 0).astype(int)
    )

    x["fvg_up"] = (l > h.shift(2)).astype(int)
    x["fvg_down"] = (h < l.shift(2)).astype(int)

    x["sweep_low"] = ((l < ll) & (c > ll)).astype(int)
    x["sweep_high"] = ((h > hh) & (c < hh)).astype(int)

    x["ob_bull"] = (
        (c.shift(1) < x["open"].shift(1))
        & (c > h.shift(1))
    ).astype(int)
    x["ob_bear"] = (
        (c.shift(1) > x["open"].shift(1))
        & (c < l.shift(1))
    ).astype(int)

    atr_safe = x["atr"].replace(0, np.nan)
    x["sr_dist_high"] = (hh - c) / atr_safe
    x["sr_dist_low"] = (c - ll) / atr_safe

    x["f_trend"] = (c - x["ema50"]) / atr_safe
    x["f_ema_spread"] = (x["ema20"] - x["ema50"]) / atr_safe
    x["f_rsi"] = (x["rsi"] - 50) / 50
    x["f_adx"] = x["adx"] / 100
    x["f_bos"] = x["bos_up"] - x["bos_down"]
    x["f_choch"] = x["choch"]
    x["f_fvg"] = x["fvg_up"] - x["fvg_down"]
    x["f_sweep"] = x["sweep_low"] - x["sweep_high"]
    x["f_ob"] = x["ob_bull"] - x["ob_bear"]
    x["f_sr"] = x["sr_dist_low"] - x["sr_dist_high"]

    future = c.shift(-8)
    x["target"] = np.where(
        future > c, 1,
        np.where(future < c, 0, np.nan)
    )

    return x.replace([np.inf, -np.inf], np.nan)


def train_model(df, features):
    d = df.dropna(subset=features + ["target"]).copy()

    if len(d) < 100:
        raise ValueError("Not enough valid rows to train the model.")

    cut = max(1, int(len(d) * 0.70))
    if cut >= len(d):
        cut = len(d) - 1

    train = d.iloc[:cut]
    test = d.iloc[cut:]

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=8,
        random_state=42,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )
    model.fit(train[features], train["target"].astype(int))

    # Kept for API compatibility with the Streamlit app.
    calibration = train.iloc[max(0, int(len(train) * 0.80)):].copy()
    return model, calibration, test


def predict_latest(model, calibration, df, features):
    valid = df.dropna(subset=features)
    if valid.empty:
        raise ValueError("No valid row available for prediction.")

    row = valid.iloc[-1]
    proba = model.predict_proba(row[features].to_frame().T)[0]
    classes = list(model.classes_)
    p = float(proba[classes.index(1)]) if 1 in classes else 0.0

    signal = "BUY" if p >= 0.60 else ("SELL" if p <= 0.40 else "WAIT")
    confidence = (p if p >= 0.50 else 1 - p) * 100

    entry = float(row["close"])
    atr = float(row["atr"]) if np.isfinite(row["atr"]) else 0.0
    risk = max(atr * 1.2, entry * 0.0005)

    if signal == "BUY":
        sl, tp = entry - risk, entry + 2 * risk
    elif signal == "SELL":
        sl, tp = entry + risk, entry - 2 * risk
    else:
        sl, tp = entry - risk, entry + risk

    state = (
        "Bullish"
        if row["ema20"] > row["ema50"] > row["ema200"]
        else "Bearish"
        if row["ema20"] < row["ema50"] < row["ema200"]
        else "Mixed"
    )

    return {
        "signal": signal,
        "confidence": confidence,
        "entry": entry,
        "sl": sl,
        "tp": tp,
        "state": state,
        "score": p,
    }


def backtest(df, model, calibration, features, rr=2.0, horizon=40):
    d = df.dropna(subset=features + ["atr"]).copy()
    start = max(250, int(len(d) * 0.70))
    rows = []

    for i in range(start, max(start, len(d) - horizon)):
        row = d.iloc[i]
        proba = model.predict_proba(row[features].to_frame().T)[0]
        classes = list(model.classes_)
        p = float(proba[classes.index(1)]) if 1 in classes else 0.0

        sig = "BUY" if p >= 0.60 else ("SELL" if p <= 0.40 else "WAIT")
        if sig == "WAIT":
            continue

        entry = float(row["close"])
        risk = max(float(row["atr"]) * 1.2, entry * 0.0005)

        sl = entry - risk if sig == "BUY" else entry + risk
        tp = entry + risk * rr if sig == "BUY" else entry - risk * rr

        result = 0.0
        for j in range(i + 1, min(i + 1 + horizon, len(d))):
            bar = d.iloc[j]
            if sig == "BUY":
                if bar["low"] <= sl:
                    result = -1.0
                    break
                if bar["high"] >= tp:
                    result = float(rr)
                    break
            else:
                if bar["high"] >= sl:
                    result = -1.0
                    break
                if bar["low"] <= tp:
                    result = float(rr)
                    break

        rows.append({
            "time": row.get("time", i),
            "signal": sig,
            "confidence": max(p, 1 - p) * 100,
            "entry": entry,
            "sl": sl,
            "tp": tp,
            "r": result,
        })

    trades = pd.DataFrame(rows)

    if trades.empty:
        return {
            "win_rate": 0.0,
            "trades": 0,
            "profit_factor": 0.0,
            "max_dd": 0.0,
            "expectancy": 0.0,
        }, trades

    equity = trades["r"].cumsum()
    drawdown = equity - equity.cummax()

    wins = trades.loc[trades["r"] > 0, "r"].sum()
    losses = abs(trades.loc[trades["r"] < 0, "r"].sum())
    profit_factor = wins / losses if losses else float("inf")

    return {
        "win_rate": float((trades["r"] > 0).mean() * 100),
        "trades": int(len(trades)),
        "profit_factor": float(profit_factor),
        "max_dd": float(drawdown.min()),
        "expectancy": float(trades["r"].mean()),
    }, trades


def monte_carlo(rs, sims=1000):
    if len(rs) == 0:
        return {}

    rng = np.random.default_rng(123)
    finals = []

    for _ in range(int(sims)):
        sample = rng.choice(np.asarray(rs), size=len(rs), replace=True)
        finals.append(float(sample.sum()))

    return {
        "median_final_r": float(np.percentile(finals, 50)),
        "p05_final_r": float(np.percentile(finals, 5)),
        "p95_final_r": float(np.percentile(finals, 95)),
    }
