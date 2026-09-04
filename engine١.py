import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

def prepare(df):
    x=df.copy()
    c,h,l=x.close,x.high,x.low
    x["ema20"]=c.ewm(span=20,adjust=False).mean()
    x["ema50"]=c.ewm(span=50,adjust=False).mean()
    x["ema200"]=c.ewm(span=200,adjust=False).mean()
    d=c.diff()
    gain=d.clip(lower=0).rolling(14).mean()
    loss=(-d.clip(upper=0)).rolling(14).mean().replace(0,np.nan)
    x["rsi"]=100-100/(1+gain/loss)
    tr=pd.concat([(h-l),(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    x["atr"]=tr.rolling(14).mean()
    plus=(h-h.shift()).clip(lower=0)
    minus=(l.shift()-l).clip(lower=0)
    atr=x["atr"].replace(0,np.nan)
    pdi=100*plus.rolling(14).sum()/atr.rolling(14).sum()
    mdi=100*minus.rolling(14).sum()/atr.rolling(14).sum()
    x["adx"]=(100*(pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan)).rolling(14).mean()
    hh=x.high.shift(1).rolling(20).max()
    ll=x.low.shift(1).rolling(20).min()
    x["bos_up"]=(c>hh).astype(int)
    x["bos_down"]=(c<ll).astype(int)
    x["choch"]=(x.bos_up.diff().fillna(0)>0).astype(int)-(x.bos_down.diff().fillna(0)>0).astype(int)
    x["fvg_up"]=(l>h.shift(2)).astype(int)
    x["fvg_down"]=(h<l.shift(2)).astype(int)
    x["sweep_low"]=((l<ll)&(c>ll)).astype(int)
    x["sweep_high"]=((h>hh)&(c<hh)).astype(int)
    x["ob_bull"]=((c.shift(1)<x.open.shift(1))&(c>h.shift(1))).astype(int)
    x["ob_bear"]=((c.shift(1)>x.open.shift(1))&(c<l.shift(1))).astype(int)
    x["sr_dist_high"]=(hh-c)/x["atr"]
    x["sr_dist_low"]=(c-ll)/x["atr"]

    # ML features
    x["f_trend"]=(c-x.ema50)/x.atr
    x["f_ema_spread"]=(x.ema20-x.ema50)/x.atr
    x["f_rsi"]=(x.rsi-50)/50
    x["f_adx"]=x.adx/100
    x["f_bos"]=x.bos_up-x.bos_down
    x["f_choch"]=x.choch
    x["f_fvg"]=x.fvg_up-x.fvg_down
    x["f_sweep"]=x.sweep_low-x.sweep_high
    x["f_ob"]=x.ob_bull-x.ob_bear
    x["f_sr"]=x.sr_dist_low-x.sr_dist_high
    # Target: next 8 bars direction after costs are ignored for the ML label.
    future=c.shift(-8)
    x["target"]=np.where(future>c,1,np.where(future<c,0,np.nan))
    return x.replace([np.inf,-np.inf],np.nan)

def train_model(df, features):
    d=df.dropna(subset=features+["target"]).copy()
    cut=int(len(d)*0.7)
    train=d.iloc[:cut]; test=d.iloc[cut:]
    base=RandomForestClassifier(n_estimators=300,max_depth=8,min_samples_leaf=8,
                                random_state=42,class_weight="balanced_subsample")
    base.fit(train[features],train.target.astype(int))
    # Calibrate using the held-out calibration slice of the training period.
    cal_start=int(len(train)*0.8)
    cal=CalibratedClassifierCV(base,method="sigmoid",cv="prefit")
    cal.fit(train.iloc[cal_start:][features],train.iloc[cal_start:].target.astype(int))
    return cal, train.iloc[cal_start:].copy(), test

def predict_latest(model, calibration, df, features):
    row=df.dropna(subset=features).iloc[-1]
    p=float(model.predict_proba(row[features].to_frame().T)[0,1])
    signal="BUY" if p>=0.60 else ("SELL" if p<=0.40 else "WAIT")
    conf=(p if p>=.5 else 1-p)*100
    entry=float(row.close); risk=float(row.atr*1.2)
    if signal=="BUY": sl,tp=entry-risk,entry+2*risk
    elif signal=="SELL": sl,tp=entry+risk,entry-2*risk
    else: sl,tp=entry-risk,entry+risk
    state="Bullish" if row.ema20>row.ema50>row.ema200 else ("Bearish" if row.ema20<row.ema50<row.ema200 else "Mixed")
    return dict(signal=signal,confidence=conf,entry=entry,sl=sl,tp=tp,state=state,score=p)

def backtest(df, model, calibration, features, rr=2.0, horizon=40):
    d=df.dropna(subset=features+["atr"]).copy()
    start=max(250,int(len(d)*0.7))
    rows=[]
    for i in range(start,len(d)-horizon):
        row=d.iloc[i]
        p=float(model.predict_proba(row[features].to_frame().T)[0,1])
        sig="BUY" if p>=.60 else ("SELL" if p<=.40 else "WAIT")
        if sig=="WAIT": continue
        entry=float(row.close); risk=float(row.atr*1.2)
        sl=entry-risk if sig=="BUY" else entry+risk
        tp=entry+risk*rr if sig=="BUY" else entry-risk*rr
        r=0
        for j in range(i+1,min(i+1+horizon,len(d))):
            b=d.iloc[j]
            if sig=="BUY":
                if b.low<=sl: r=-1; break
                if b.high>=tp: r=rr; break
            else:
                if b.high>=sl: r=-1; break
                if b.low<=tp: r=rr; break
        rows.append({"time":row.get("time",i),"signal":sig,"confidence":max(p,1-p)*100,
                     "entry":entry,"sl":sl,"tp":tp,"r":r})
    t=pd.DataFrame(rows)
    if t.empty: return dict(win_rate=0,trades=0,profit_factor=0,max_dd=0,expectancy=0),t
    equity=t.r.cumsum(); dd=(equity-equity.cummax()).min()
    wins=t.loc[t.r>0,"r"].sum(); losses=abs(t.loc[t.r<0,"r"].sum())
    pf=wins/losses if losses else float("inf")
    return dict(win_rate=(t.r>0).mean()*100,trades=len(t),profit_factor=pf,
                max_dd=float(dd),expectancy=t.r.mean()),t

def monte_carlo(rs, sims=1000):
    if len(rs)==0: return {}
    rng=np.random.default_rng(123)
    finals=[]
    for _ in range(sims):
        sample=rng.choice(rs,size=len(rs),replace=True)
        finals.append(sample.sum())
    return {"median_final_r":float(np.percentile(finals,50)),
            "p05_final_r":float(np.percentile(finals,5)),
            "p95_final_r":float(np.percentile(finals,95))}
