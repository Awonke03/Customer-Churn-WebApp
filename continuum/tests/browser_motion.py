import os,sys,time,subprocess,socket,urllib.request
from pathlib import Path
from playwright.sync_api import sync_playwright,expect
root=Path(__file__).resolve().parents[1]
with socket.socket() as sock:
    sock.bind(('127.0.0.1',0))
    port=sock.getsockname()[1]
url=f'http://127.0.0.1:{port}'
proc=subprocess.Popen([sys.executable,'-m','flask','--app','app','run','--port',str(port)],cwd=root,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
try:
    for _ in range(80):
        try:
            urllib.request.urlopen(url,timeout=1)
            break
        except Exception:
            time.sleep(.15)
    else:
        raise RuntimeError('Local application unavailable')
    with sync_playwright() as p:
        browser=p.chromium.launch()
        page=browser.new_page(viewport={'width':1440,'height':900})
        errors=[];requests=[]
        page.on('pageerror',lambda e:errors.append(str(e)))
        page.on('request',lambda req:requests.append(req.url) if '/api/' in req.url else None)
        page.goto(url+'/#campaign')
        expect(page.locator('#overview')).to_be_visible()
        expect(page.locator('#status')).to_have_text('1,000 customers',timeout=90000)
        expect(page.locator('#kpis .kpi strong').first).to_have_text('1,000')
        expect(page.locator('#kpis .kpi strong').nth(2)).to_contain_text('R')
        expect(page.locator('#source')).to_contain_text('COFINFAD')
        assert page.locator('#kpis .kpi').first.evaluate('(el)=>getComputedStyle(el).borderRadius')=='16px'
        expect(page.locator('#risk-bars svg[role="img"]')).to_be_visible()
        page.locator('[data-view="segments"]').click()
        expect(page.locator('#segment-donut svg[role="img"]')).to_be_visible()
        expect(page.locator('#segment-scatter svg[role="img"]')).to_be_visible()
        page.locator('#segment-table th').first.hover()
        expect(page.locator('#column-tooltip')).to_be_visible()
        page.locator('[data-view="customers"]').click()
        expect(page.locator('#customer-table tbody tr')).to_have_count(10)
        page.locator('#next').click()
        expect(page.locator('#pages')).to_contain_text('11–20')
        page.reload()
        expect(page.locator('#overview')).to_be_visible()
        expect(page.locator('#status')).to_have_text('1,000 customers',timeout=90000)
        page.locator('[data-view="campaign"]').click()
        expect(page.locator('#budget-donut svg[role="img"]')).to_be_visible()
        page.locator('#single').click()
        expect(page.locator('#score-dialog')).to_be_visible()
        page.locator('#file').set_input_files({'name':'transactions.csv','mimeType':'text/csv','buffer':b'customer_id,date,amount,type\\nF-1,2023-01-01,10000,transfer\\nF-1,2023-07-01,20000,transfer\\n'.replace(b'\\n',b'\n')})
        with page.expect_response(lambda r:r.url.endswith('/api/fintech/score') and r.ok):
            page.locator('#score-submit').click()
        expect(page.locator('#customers')).to_be_visible()
        expect(page.locator('#customer-table tbody tr')).to_have_count(1)
        page.locator('#demo').click()
        expect(page.locator('#status')).to_have_text('1,000 customers',timeout=90000)
        for width in [320,390,768,1024,1440]:
            page.set_viewport_size({'width':width,'height':900})
            for v in ['overview','customers','segments','campaign','fintech']:
                page.locator('[data-view="'+v+'"]').click()
                expect(page.locator('#'+v)).to_be_visible()
                page.wait_for_timeout(450)
                assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1'),(width,v)
        expect(page.locator('#fintech-metrics .kpi')).to_have_count(8)
        assert not errors,errors
        browser.close()
finally:
    proc.terminate()
    proc.wait(timeout=10)
