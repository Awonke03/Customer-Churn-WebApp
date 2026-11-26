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
    assert len(d['customers'])>100 and d['currency']=='GBP'
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
    assert b'Continuum' in c.get('/').data
    assert c.get('/static/app.js').status_code==200
