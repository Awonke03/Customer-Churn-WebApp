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
from fintech_features import snapshot
def group_id(cid):
    return int(hashlib.sha256(str(cid).encode()).hexdigest()[:8],16)%10
def metrics(y,p,threshold=.5):
    return dict(pr_auc=float(average_precision_score(y,p)),roc_auc=float(roc_auc_score(y,p)),brier=float(brier_score_loss(y,p)),log_loss=float(log_loss(y,p)),precision=float(precision_score(y,p>=threshold,zero_division=0)),recall=float(recall_score(y,p>=threshold,zero_division=0)),f1=float(f1_score(y,p>=threshold,zero_division=0)),prevalence=float(np.mean(y)))
def logit(p):
    p=np.clip(p,1e-6,1-1e-6)
    return np.log(p/(1-p)).reshape(-1,1)
def main():
    raw,quality=fetch()
    train=pd.concat([snapshot(raw,d) for d in ['2023-07-04','2023-08-03']],ignore_index=True)
    val=snapshot(raw,'2023-09-02')
    cal=snapshot(raw,'2023-10-02')
    test=snapshot(raw,'2023-11-30')
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
    best=max(scores,key=lambda n:scores[n]['pr_auc'])
    # Native sklearn inference keeps the Vercel bundle small.
    serving=[n for n in scores if n in ('logistic','extra_trees','random_forest','hist_gradient','mlp')]
    winner=max(serving,key=lambda n:scores[n]['pr_auc'])
    model=candidates[winner]
    calibrator=LogisticRegression().fit(logit(model.predict_proba(X(cal))[:,1]),cal.inactive)
    prob=lambda d:calibrator.predict_proba(logit(model.predict_proba(X(d))[:,1]))[:,1]
    p_cal=prob(cal)
    threshold=float(max(np.linspace(.03,.60,115),key=lambda t:f1_score(cal.inactive,p_cal>=t,zero_division=0)))
    p=prob(test); result=metrics(test.inactive,p,threshold)
    baseline=metrics(test.inactive,np.full(len(test),train.inactive.mean()))
    # Select a point forecast for absolute error; do not call it expected revenue.
    # Include strong rolling-history baselines and direct supervised forecasts.
    regs={'hist_direct':HistGradientBoostingRegressor(loss='absolute_error',max_iter=200,max_leaf_nodes=31,l2_regularization=3,random_state=42),
          'hist_log':HistGradientBoostingRegressor(loss='absolute_error',max_iter=200,max_leaf_nodes=15,l2_regularization=5,random_state=42)}
    for name,m in regs.items():
        m.fit(X(train),np.log1p(train.future_volume) if name=='hist_log' else train.future_volume)
    def predictions(d):
        v={'last30':d.volume_30.to_numpy(),'last60':d.volume_60.to_numpy()/2,'last90':d.volume_90.to_numpy()/3,'last180':d.volume_180.to_numpy()/6}
        for name,m in regs.items():
            q=m.predict(X(d));v[name]=np.maximum(0,np.expm1(q) if name=='hist_log' else q)
        v['rolling_blend']=(v['last30']+v['last90'])/2
        for k in ('hist_direct','hist_log'):
            for w in (.25,.5,.75):
                v[k+'_'+str(w)]=w*v[k]+(1-w)*v['rolling_blend']
        return v
    rev_scores={name:float(mean_absolute_error(val.future_volume,v)) for name,v in predictions(val).items()}
    rname=min(rev_scores,key=rev_scores.get)
    mae=float(mean_absolute_error(test.future_volume,predictions(test)[rname]))
    naive=float(mean_absolute_error(test.future_volume,test.volume_30))
    # Stress checks: neutralise each feature family using train medians, not test data.
    stress={}
    for name,cols in {'short_windows':[c for c in features if c.endswith(('_7','_14','_30'))],'amounts':[c for c in features if c.startswith(('volume','mean','std','max','min'))],'timing':[c for c in features if c.startswith(('gap','recency','weekend','month_end'))]}.items():
        z=test.copy();z.loc[:,cols]=np.broadcast_to(train[cols].median().to_numpy(),(len(z),len(cols)))
        stress[name]=metrics(test.inactive,prob(z),threshold)
    subgroups={}
    for name,mask in {'lower_activity':test.count_180<=train.count_180.median(),'higher_activity':test.count_180>train.count_180.median(),'recent':test.recency<=30,'longer_inactive':test.recency>30}.items():
        if mask.sum()>50 and test.loc[mask,'inactive'].nunique()==2: subgroups[name]={'n':int(mask.sum()),**metrics(test.loc[mask,'inactive'],p[mask.to_numpy()],threshold)}
    rng=np.random.default_rng(42); boot=[]
    y=test.inactive.to_numpy()
    for _ in range(200):
        ix=rng.integers(0,len(y),len(y))
        if np.unique(y[ix]).size==2: boot.append(average_precision_score(y[ix],p[ix]))
    seg=make_pipeline(StandardScaler(),MiniBatchKMeans(n_clusters=5,n_init=5,random_state=42)).fit(np.log1p(X(train).clip(lower=0)))
    report={'dataset':'COFINFAD Colombia (not South Africa)','source':'https://doi.org/10.17632/mhb4zn3258.1','license':'CC BY 4.0','currency':'COP','target':'No observed nonzero transaction in next 30 days; not account closure','data_quality':quality,'features':features,'n_features':len(features),'partition_counts':{k:len(v) for k,v in [('train',train),('validation',val),('calibration',cal),('test',test)]},'validation':scores,'winner':winner,'best_unrestricted':best,'threshold':threshold,'test_date':'2023-11-30','test':result,'baseline':baseline,'pr_auc_bootstrap_95':np.quantile(boot,[.025,.975]).tolist(),'stress':stress,'behavioural_subgroups':subgroups,'turnover_model':rname,'turnover_validation_mae':rev_scores,'test_turnover_mae_cop':mae,'naive_turnover_mae_cop':naive,'release_gate':bool(result['pr_auc']>baseline['pr_auc'] and result['brier']<baseline['brier'] and result['recall']>=.35 and result['precision']>result['prevalence'] and mae<=naive and scores[best]['pr_auc']-scores[winner]['pr_auc']<=.02),'limitations':'One Colombian fintech in 2023; transaction turnover is not bank revenue. No causal retention effect, demographic fairness or South African validation established.'}
    (OUT/'report.json').write_text(json.dumps(report,indent=2))
    joblib.dump({'model':model,'calibrator':calibrator,'turnover_models':regs,'turnover_choice':rname,'threshold':threshold,'segment':seg,'features':features,'medians':train[features].median().to_dict(),'horizon_days':H,'currency':'COP'},OUT/'bundle.joblib',compress=3)
    # Only anonymised features, no raw transaction histories.
    default=snapshot(raw,'2023-12-29',False).sample(n=1000,random_state=42)
    default['customer_id']=default.customer_id.map(lambda x:'F-'+hashlib.sha256(('loyaltics-fintech-'+str(x)).encode()).hexdigest()[:12])
    default[['customer_id']+features].to_csv(OUT/'default.csv',index=False,float_format='%.10g')
    print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__': main()
