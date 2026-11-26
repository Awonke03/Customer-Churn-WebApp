"""Export historical scored example for CDN delivery. Inference only, never trains."""
from pathlib import Path
import json
from app import demo_payload
p=Path(__file__).resolve().parent/'public'/'demo.json'
d=demo_payload()
assert d['currency']=='ZAR' and len(d['customers'])==2772
p.write_text(json.dumps(d,ensure_ascii=False,separators=(',',':'),allow_nan=False),encoding='utf-8')
print(f'{len(d["customers"])} pre-scored records; {p.stat().st_size:,} bytes')
