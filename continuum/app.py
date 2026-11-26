from pathlib import Path
import io, json, os, threading
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify, send_from_directory, Response
ROOT=Path(__file__).resolve().parent
app=Flask(__name__,static_folder='public/static', static_url_path='/static')
app.config['MAX_CONTENT_LENGTH']=2*1024*1024
FEATURES=['recency','tenure','orders_30','orders_90','orders_180','spend_30','spend_90','spend_180','avg_order_180']
BUNDLE=None
REPORT=None
DEMO=None
LOAD_LOCK=threading.Lock()
def load():
    global BUNDLE,REPORT,DEMO
    if BUNDLE is None:
        with LOAD_LOCK:
            if BUNDLE is None:
                bundle=joblib.load(ROOT/'models/bundle.joblib')
                REPORT=json.loads((ROOT/'models/report.json').read_text())
                DEMO=pd.read_csv(ROOT/'models/demo.csv',dtype={'customer_id':str})
                BUNDLE=bundle
    return BUNDLE
def validate(d):
    if len(d)<1 or len(d)>3000: raise ValueError('Upload 1–3,000 customer records.')
    if not set(['customer_id']+FEATURES)<=set(d): raise ValueError('Missing required columns. Use the CSV template.')
    d=d[['customer_id']+FEATURES].copy()
    d.customer_id=d.customer_id.astype(str)
    if d.customer_id.duplicated().any() or d.customer_id.str.len().gt(80).any() or d.customer_id.str.strip().eq('').any() or d.customer_id.str.lower().eq('nan').any(): raise ValueError('Use unique, nonempty anonymous customer IDs.')
    for c in FEATURES: d[c]=pd.to_numeric(d[c],errors='raise')
    x=d[FEATURES].to_numpy(float)
    if not np.isfinite(x).all() or (x<0).any() or (x>1e10).any(): raise ValueError('Features must be finite and nonnegative.')
    if ((d.recency>180)|(d.recency>d.tenure)|(d.orders_180<1)|(d.spend_180<=0)|(d.orders_30>d.orders_90)|(d.orders_90>d.orders_180)|(d.spend_30>d.spend_90+.01)|(d.spend_90>d.spend_180+.01)).any(): raise ValueError('Inconsistent purchase windows, tenure or active-customer eligibility.')
    return d
def probability(x,b):
    p=np.clip(b['model'].predict_proba(x)[:,1],1e-6,1-1e-6)
    return b['calibrator'].predict_proba(np.log(p/(1-p)).reshape(-1,1))[:,1]
def score(d, explain=False):
    b=load(); d=validate(d); x=d[FEATURES].astype(float); p=probability(x,b)
    conditional=np.maximum(0,np.expm1(b['revenue'].predict(x)))
    d['risk']=p; d['potential_value']=conditional;d['expected_revenue']=(1-p)*conditional;d['revenue_at_risk']=p*conditional
    d['segment']=b['segment'].predict(np.log1p(x))+1
    d['tier']=np.select([p>=.7,p>=.4],['High','Watch'],default='Low')
    d['driver']=''
    # Local sensitivity: replace one feature with the training median. Not SHAP or causal attribution.
    if explain:
        effects=[]
        for c in FEATURES:
            changed=x.copy();changed[c]=b['medians'][c]
            effects.append(p-probability(changed,b))
        a=np.array(effects).T; idx=a.argmax(axis=1)
        d['driver']=[FEATURES[i] if a[j,i]>0 else 'No positive median-replacement effect' for j,i in enumerate(idx)]
    return d
@app.get('/')
def home():return send_from_directory(ROOT/'public','index.html')
@app.get('/template.csv')
def static_template():return send_from_directory(ROOT/'public','template.csv',as_attachment=True,download_name='loyaltics_template.csv')
@app.get('/health')
def health():
    try:load();return jsonify(ready=True)
    except (OSError,ValueError):return jsonify(ready=False),503
@app.get('/api/demo')
def demo():
    load()
    return jsonify(as_of=REPORT['demo_date'],currency='GBP',source='Historical retail example',customers=score(DEMO,True).round(5).to_dict('records'))
@app.post('/api/score')
def upload():
    try:
        file=request.files.get('file')
        if not file:raise ValueError('Select a CSV file.')
        d=pd.read_csv(file,dtype={'customer_id':str})
        response=jsonify(as_of=None,currency='GBP',source='Imported customer features',customers=score(d,True).round(5).to_dict('records'))
        if len(response.get_data())>4_000_000:return jsonify(error='Scored output is too large. Split the CSV into smaller batches.'),413
        return response
    except (ValueError,TypeError,pd.errors.ParserError) as e:return jsonify(error=str(e)),400
@app.get('/api/template')
def template():
    return Response('customer_id,'+','.join(FEATURES)+'\nexample-1,18,240,1,3,6,50,240,600,100\n',mimetype='text/csv',headers={'Content-Disposition':'attachment; filename=loyaltics_template.csv'})
@app.errorhandler(413)
def too_large(e):return jsonify(error='File too large. Maximum request size is 2 MB.'),413
@app.after_request
def security(resp):
    resp.headers['X-Content-Type-Options']='nosniff';resp.headers['Referrer-Policy']='no-referrer'
    resp.headers['Content-Security-Policy']="default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; object-src 'none'; frame-ancestors 'none'; base-uri 'self'"
    if request.path.startswith('/api/'):resp.headers['Cache-Control']='no-store'
    return resp
