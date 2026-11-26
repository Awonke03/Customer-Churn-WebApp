import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import app,load,validate
import pandas as pd
import io
def test_model_and_demo():
    load()
    c=app.test_client()
    assert c.get('/health').status_code==200
    d=c.get('/api/demo').get_json()
    assert len(d['customers'])>100 and d['currency']=='ZAR'
    for r in d['customers']:
        assert 0<=r['risk']<=1 and r['potential_value']>=0
        assert abs(r['expected_revenue']+r['revenue_at_risk']-r['potential_value'])<.001
def test_import():
    c=app.test_client()
    template=c.get('/api/template').data
    assert c.post('/api/score',data={'file':(io.BytesIO(template),'example.csv')}).status_code==200
    assert c.post('/api/score',data={'file':(io.BytesIO(b'bad\n1'),'bad.csv')}).status_code==400
def test_ui():
    c=app.test_client()
    assert b'Loyaltics' in c.get('/').data
    assert c.get('/static/app.js').status_code==200

def test_static_and_limits():
    c=app.test_client()
    assert c.get('/template.csv').status_code==200
    assert c.get('/').status_code==200
    assert c.get('/static/icon.svg').status_code==200
    template=c.get('/api/template').data.decode().splitlines()
    records=[template[0]]+[template[1].replace('example-1','C-'+str(i)) for i in range(3001)]
    assert c.post('/api/score',data={'file':(io.BytesIO(('\n'.join(records)).encode()),'too-many.csv')}).status_code==400
    assert c.post('/api/score',data={'file':(io.BytesIO(b'x'*(2*1024*1024+1)),'large.csv')}).status_code==413
    assert len(c.get('/api/demo').data)<4_000_000

def test_currency_roundtrip_and_preload():
    from app import DEMO, score, load, ZAR_PER_GBP, demo_payload
    import app as module
    load()
    d=module.DEMO.iloc[:1].copy()
    gbp=score(d,True).iloc[0]
    for k in module.MONEY_FEATURES: d[k]*=ZAR_PER_GBP
    c=app.test_client()
    res=c.post('/api/score',data={'file':(io.BytesIO(d.to_csv(index=False).encode()),'zar.csv')}).get_json()
    row=res['customers'][0]
    assert res['currency']=='ZAR'
    assert abs(row['risk']-gbp.risk)<.00001
    assert abs(row['potential_value']-gbp.potential_value*ZAR_PER_GBP)<.00001
    snapshot=c.get('/demo.json')
    assert snapshot.status_code==200
    assert snapshot.get_json()==demo_payload()
