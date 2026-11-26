"""Fintech inference, separate from UK retail models."""
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from functools import lru_cache
from fintech_features import snapshot
ROOT=Path(__file__).resolve().parent/'models'/'fintech'
@lru_cache(maxsize=1)
def bundle():
    return joblib.load(ROOT/'bundle.joblib')
def score_features(d):
    b=bundle(); x=d[b['features']].astype(float)
    if not np.isfinite(x.to_numpy()).all(): raise ValueError('Nonfinite derived features')
    p=np.clip(b['model'].predict_proba(x)[:,1],1e-6,1-1e-6)
    p=b['calibrator'].predict_proba(np.log(p/(1-p)).reshape(-1,1))[:,1]
    v={'last30':x.volume_30.to_numpy(),'last60':x.volume_60.to_numpy()/2,'last90':x.volume_90.to_numpy()/3,'last180':x.volume_180.to_numpy()/6}
    for name,m in b['turnover_models'].items():
        q=m.predict(x);v[name]=np.maximum(0,np.expm1(q) if name=='hist_log' else q)
    v['rolling_blend']=(v['last30']+v['last90'])/2
    for k in ('hist_direct','hist_log'):
        for w in (.25,.5,.75):v[k+'_'+str(w)]=w*v[k]+(1-w)*v['rolling_blend']
    out=pd.DataFrame({'customer_id':d.customer_id.astype(str),'risk':p,'flagged':p>=b['threshold'],'turnover_cop':v[b['turnover_choice']],'segment':b['segment'].predict(np.log1p(x.clip(lower=0)))+1,'recency':x.recency.to_numpy(),'last30_cop':x.volume_30.to_numpy()})
    return {'customers':out.round(6).to_dict('records'),'threshold':b['threshold'],'currency':'COP','horizon_days':30}
def default_payload():
    result=score_features(pd.read_csv(ROOT/'default.csv',dtype={'customer_id':str}))
    result['as_of']='2023-12-29'
    return result
def score_transactions(raw):
    needed=['customer_id','date','amount','type']
    if not set(needed)<=set(raw):raise ValueError('Required columns: customer_id, date, amount, type')
    if not 1<=len(raw)<=50000:raise ValueError('Supply 1–50,000 transaction rows.')
    raw=raw[needed].copy()
    if raw.isna().any().any():raise ValueError('Missing transaction fields.')
    raw.customer_id=raw.customer_id.astype(str)
    if not raw.customer_id.str.fullmatch(r'[A-Za-z0-9_-]{1,80}').all():raise ValueError('Use anonymous customer references.')
    raw.date=pd.to_datetime(raw.date,errors='raise').dt.normalize()
    raw.amount=pd.to_numeric(raw.amount,errors='raise')
    if not np.isfinite(raw.amount).all() or raw.amount.abs().gt(1e12).any():raise ValueError('Invalid COP amounts.')
    raw=raw[raw.amount!=0].copy()
    if raw.empty:raise ValueError('No nonzero transactions.')
    raw['value']=raw.amount.abs();raw['type']=raw.type.astype(str)
    if raw.customer_id.nunique()>3000:raise ValueError('Limit to 3,000 customers.')
    if (raw.date.max()-raw.date.min()).days<179:raise ValueError('Supply at least 180 calendar days of observation history across the portfolio.')
    raw=raw.sort_values('date')
    cut=raw.date.max()+pd.Timedelta(days=1)
    d=snapshot(raw,cut,False)
    result=score_features(d);result['as_of']=str(cut.date())
    return result
