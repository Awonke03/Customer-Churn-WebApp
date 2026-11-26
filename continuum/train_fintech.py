"""COFINFAD challenger. Real dated transactions only; never use full-year customer aggregates as predictors.
Outputs are isolated from the production UK model until reviewed. Amounts are COP transaction
turnover, not bank revenue. Dataset: Munoz Guerrero, Ceballos & Trejos Rojas (2025),
https://doi.org/10.17632/mhb4zn3258.1 (CC BY 4.0); author's Hugging Face mirror.
"""
from pathlib import Path
import hashlib, json, os, urllib.request
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss, log_loss, mean_absolute_error, precision_score, recall_score, f1_score
from sklearn.cluster import MiniBatchKMeans
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'challenger_artifacts'
OUT.mkdir(exist_ok=True)
URL='https://huggingface.co/datasets/luisdavidtrejosrojas/cofinfad/resolve/f7b6a9f45bd75fba9f791238dee85c5028ef5b19/transactions_data.csv'
H=30
def fetch():
    path=OUT/'transactions.csv'
    if not path.exists():
        urllib.request.urlretrieve(URL,path)
    raw=pd.read_csv(path,usecols=['customer_id','date','amount','type'],dtype={'customer_id':str})
    n=len(raw)
    raw['date']=pd.to_datetime(raw.date,errors='coerce')
    raw['amount']=pd.to_numeric(raw.amount,errors='coerce')
    raw=raw.dropna(subset=['customer_id','date','amount'])
    raw=raw[np.isfinite(raw.amount)&(raw.amount!=0)].copy()
    raw['value']=raw.amount.abs()
    raw['type']=raw.type.fillna('unknown').astype(str)
    raw=raw.sort_values('date')
    assert len(raw)>1_000_000 and raw.customer_id.nunique()>10_000, 'Unexpected data coverage'
    print('Data quality',n,len(raw),raw.date.min(),raw.date.max(),flush=True)
    return raw, {'raw_rows':n,'eligible_rows':len(raw),'customers':raw.customer_id.nunique(),'first_date':str(raw.date.min()),'last_date':str(raw.date.max()),'sha256':hashlib.file_digest(path.open('rb'),'sha256').hexdigest()}
def snapshot(raw,day,labeled=True):
    cut=pd.Timestamp(day)
    past=raw[raw.date<cut]
    recent=past[past.date>=cut-pd.Timedelta(days=180)].copy()
    g=recent.groupby('customer_id')
    x=g.agg(last=('date','max'),count_180=('value','size'),volume_180=('value','sum'),mean_180=('value','mean'),std_180=('value','std'),max_180=('value','max'),min_180=('value','min'),types_180=('type','nunique'),active_days_180=('date','nunique'))
    x['recency']=(cut-x.pop('last')).dt.days
    x['observed_tenure']=(cut-past.groupby('customer_id').date.min()).dt.days
    recent['weekend']=(recent.date.dt.dayofweek>=5).astype(float)
    recent['month_end']=(recent.date.dt.day>=25).astype(float)
    recent['negative']=(recent.amount<0).astype(float)
    recent['gap']=recent.groupby('customer_id').date.diff().dt.days
    g=recent.groupby('customer_id')
    for k in ['weekend','month_end','negative','gap']:
        x[k+'_mean']=g[k].mean()
    x['gap_std']=g.gap.std()
    for w in (7,14,30,60,90):
        q=recent[recent.date>=cut-pd.Timedelta(days=w)].groupby('customer_id')
        a=q.agg(**{f'count_{w}':('value','size'),f'volume_{w}':('value','sum'),f'mean_{w}':('value','mean'),f'std_{w}':('value','std'),f'active_days_{w}':('date','nunique'),f'types_{w}':('type','nunique')})
        x=x.join(a)
    x=x.fillna(0)
    for w in (7,14,30,60,90):
        x[f'count_share_{w}']=x[f'count_{w}']/x.count_180
        x[f'volume_share_{w}']=x[f'volume_{w}']/x.volume_180
    x['count_acceleration_30']=x.count_30/(1+(x.count_90-x.count_30)/2)
    x['volume_acceleration_30']=x.volume_30/(1+(x.volume_90-x.volume_30)/2)
    x['relative_amount_volatility']=x.std_180/(1+x.mean_180)
    x['transactions_per_active_day']=x.count_180/x.active_days_180
    x['as_of']=str(cut.date())
    if labeled:
        assert cut+pd.Timedelta(days=H-1)<=raw.date.max().normalize()
        f=raw[(raw.date>=cut)&(raw.date<cut+pd.Timedelta(days=H))].groupby('customer_id').value.agg(['size','sum'])
        x['future_count']=f['size'].reindex(x.index,fill_value=0)
        x['future_volume']=f['sum'].reindex(x.index,fill_value=0)
        x['inactive']=(x.future_count==0).astype(int)
    return x.reset_index()
def group_id(cid):
    return int(hashlib.sha256(str(cid).encode()).hexdigest()[:8],16)%10
def metrics(y,p):
    return dict(pr_auc=float(average_precision_score(y,p)),roc_auc=float(roc_auc_score(y,p)),brier=float(brier_score_loss(y,p)),log_loss=float(log_loss(y,p)),precision=float(precision_score(y,p>=.5,zero_division=0)),recall=float(recall_score(y,p>=.5,zero_division=0)),f1=float(f1_score(y,p>=.5,zero_division=0)),prevalence=float(np.mean(y)))
def logit(p):
    p=np.clip(p,1e-6,1-1e-6)
    return np.log(p/(1-p)).reshape(-1,1)
def main():
    raw,quality=fetch()
    train=pd.concat([snapshot(raw,d) for d in ['2023-07-04','2023-08-03']],ignore_index=True)
    val=snapshot(raw,'2023-09-02')
    cal=snapshot(raw,'2023-10-02')
    test=snapshot(raw,'2023-11-01')
    # Fixed customer-disjoint partitions and strictly non-overlapping label windows.
    for d in (train,val,cal,test): d['group']=d.customer_id.map(group_id)
    train=train[train.group<7];val=val[val.group==7];cal=cal[cal.group==8];test=test[test.group==9]
    features=[c for c in train if c not in ['customer_id','as_of','future_count','future_volume','inactive','group']]
    def X(d): return d[features].astype(float)
    assert len(features)>=45 and all(d.inactive.nunique()==2 for d in (train,val,cal,test))
    assert not set(train.customer_id)&set(test.customer_id)
    candidates={'logistic':make_pipeline(StandardScaler(),LogisticRegression(max_iter=800)),
        'extra_trees':ExtraTreesClassifier(n_estimators=180,min_samples_leaf=15,max_depth=15,n_jobs=2,random_state=42),
        'random_forest':RandomForestClassifier(n_estimators=120,min_samples_leaf=20,max_depth=12,n_jobs=2,random_state=42),
        'hist_gradient':HistGradientBoostingClassifier(max_iter=160,max_leaf_nodes=31,l2_regularization=3,random_state=42),
        'mlp':make_pipeline(StandardScaler(),MLPClassifier(hidden_layer_sizes=(64,32),max_iter=80,early_stopping=True,random_state=42))}
    for leaves in (15,31,63):
        candidates[f'lightgbm_{leaves}']=LGBMClassifier(n_estimators=250,learning_rate=.04,num_leaves=leaves,min_child_samples=50,n_jobs=2,verbosity=-1,random_state=42)
    for depth in (3,6):
        candidates[f'xgboost_{depth}']=XGBClassifier(n_estimators=220,max_depth=depth,learning_rate=.04,n_jobs=2,random_state=42)
        candidates[f'catboost_{depth}']=CatBoostClassifier(iterations=220,depth=depth,learning_rate=.04,thread_count=2,verbose=False,random_seed=42)
    scores={}
    for name,m in candidates.items():
        m.fit(X(train),train.inactive)
        scores[name]=metrics(val.inactive,m.predict_proba(X(val))[:,1])
        print(name,scores[name],flush=True)
    winner=max(scores,key=lambda n:scores[n]['pr_auc'])
    model=candidates[winner]
    calibrator=LogisticRegression().fit(logit(model.predict_proba(X(cal))[:,1]),cal.inactive)
    prob=lambda d:calibrator.predict_proba(logit(model.predict_proba(X(d))[:,1]))[:,1]
    p=prob(test); result=metrics(test.inactive,p)
    baseline=metrics(test.inactive,np.full(len(test),train.inactive.mean()))
    # Conditional transaction turnover; no observed fee or margin labels.
    act=train[train.inactive==0]
    regs={'hist':HistGradientBoostingRegressor(loss='absolute_error',max_iter=150,l2_regularization=5,random_state=42),'lgbm':LGBMRegressor(objective='mae',n_estimators=220,learning_rate=.05,n_jobs=2,verbosity=-1,random_state=42)}
    rev_scores={}
    for name,m in regs.items():
        m.fit(X(act),np.log1p(act.future_volume))
        rev_scores[name]=float(mean_absolute_error(val.future_volume,(1-model.predict_proba(X(val))[:,1])*np.maximum(0,np.expm1(m.predict(X(val))))))
    rname=min(rev_scores,key=rev_scores.get);reg=regs[rname]
    conditional=np.maximum(0,np.expm1(reg.predict(X(test))))
    mae=float(mean_absolute_error(test.future_volume,(1-p)*conditional))
    naive=float(mean_absolute_error(test.future_volume,test.volume_30))
    # Stress checks: neutralise each feature family using train medians, not test data.
    stress={}
    for name,cols in {'short_windows':[c for c in features if c.endswith(('_7','_14','_30'))],'amounts':[c for c in features if c.startswith(('volume','mean','std','max','min'))],'timing':[c for c in features if c.startswith(('gap','recency','weekend','month_end'))]}.items():
        z=test.copy();z[cols]=train[cols].median()
        stress[name]=metrics(test.inactive,prob(z))
    subgroups={}
    for name,mask in {'lower_activity':test.count_180<=train.count_180.median(),'higher_activity':test.count_180>train.count_180.median(),'recent':test.recency<=30,'longer_inactive':test.recency>30}.items():
        if mask.sum()>50 and test.loc[mask,'inactive'].nunique()==2: subgroups[name]={'n':int(mask.sum()),**metrics(test.loc[mask,'inactive'],p[mask.to_numpy()])}
    rng=np.random.default_rng(42); boot=[]
    y=test.inactive.to_numpy()
    for _ in range(200):
        ix=rng.integers(0,len(y),len(y))
        if np.unique(y[ix]).size==2: boot.append(average_precision_score(y[ix],p[ix]))
    seg=make_pipeline(StandardScaler(),MiniBatchKMeans(n_clusters=5,n_init=5,random_state=42)).fit(np.log1p(X(train).clip(lower=0)))
    report={'dataset':'COFINFAD Colombia (not South Africa)','source':'https://doi.org/10.17632/mhb4zn3258.1','license':'CC BY 4.0','currency':'COP','target':'No observed nonzero transaction in next 30 days; not account closure','data_quality':quality,'features':features,'n_features':len(features),'partition_counts':{k:len(v) for k,v in [('train',train),('validation',val),('calibration',cal),('test',test)]},'validation':scores,'winner':winner,'test':result,'baseline':baseline,'pr_auc_bootstrap_95':np.quantile(boot,[.025,.975]).tolist(),'stress':stress,'behavioural_subgroups':subgroups,'conditional_turnover_model':rname,'turnover_validation_mae':rev_scores,'test_turnover_mae_cop':mae,'naive_turnover_mae_cop':naive,'release_gate':bool(result['pr_auc']>baseline['pr_auc'] and result['brier']<baseline['brier'] and mae<naive),'limitations':'One Colombian fintech in 2023; transaction turnover is not bank revenue. No causal retention effect, demographic fairness or South African validation established.'}
    (OUT/'report.json').write_text(json.dumps(report,indent=2))
    joblib.dump({'model':model,'calibrator':calibrator,'turnover':reg,'segment':seg,'features':features,'medians':train[features].median().to_dict(),'horizon_days':H,'currency':'COP'},OUT/'bundle.joblib',compress=3)
    print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__': main()
