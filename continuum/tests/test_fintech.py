import io,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import app
def test_fintech_production():
    c=app.test_client()
    r=c.get('/fintech-evaluation.json').get_json()
    assert r['release_gate'] and r['n_features']==59 and r['test']['recall']>=.35
    d=c.get('/api/fintech/default').get_json()
    assert len(d['customers'])==1000 and d['horizon_days']==30
    assert all(0<=x['risk']<=1 and x['turnover_cop']>=0 and x['flagged']==(x['risk']>=d['threshold']) for x in d['customers'])
    raw=b'customer_id,date,amount,type\nF-1,2023-01-01,10000,transfer\nF-1,2023-07-01,20000,transfer\n'
    res=c.post('/api/fintech/score',data={'file':(io.BytesIO(raw),'transactions.csv'),'single':'1'})
    assert res.status_code==200 and len(res.get_json()['customers'])==1
    assert c.get('/fintech-template.csv').status_code==200
    assert c.post('/api/fintech/score',data={'file':(io.BytesIO(b'wrong\n1'),'bad.csv')}).status_code==400
