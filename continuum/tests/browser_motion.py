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
        page.goto(url)
        expect(page.locator('#status')).to_have_text('2,772 customers',timeout=90000)
        expect(page.locator('#kpis .kpi strong').first).to_have_text('2,772')
        expect(page.locator('#kpis .kpi strong').nth(2)).to_contain_text('R')
        expect(page.locator('nav')).to_have_class('ink-ready')
        assert not requests,requests
        before=page.locator('.nav-ink').evaluate('(e)=>getComputedStyle(e).transform')
        page.locator('[data-view="campaign"]').click()
        expect(page.locator('#campaign')).to_be_visible()
        expect(page.locator('[data-view="campaign"]')).to_have_attribute('aria-current','page')
        page.wait_for_timeout(500)
        after=page.locator('.nav-ink').evaluate('(e)=>getComputedStyle(e).transform')
        assert before!=after
        assert not requests
        page.locator('#single').click()
        expect(page.locator('#score-dialog')).to_be_visible()
        page.keyboard.press('Escape')
        expect(page.locator('#score-dialog')).to_be_hidden()
        page.locator('#single').click()
        with page.expect_response(lambda r:r.url.endswith('/api/score') and r.ok):
            page.locator('#score-submit').click()
        expect(page.locator('#score-dialog')).to_be_hidden()
        expect(page.locator('#customers')).to_be_visible()
        expect(page.locator('#customer-table tbody tr')).to_have_count(1)
        page.locator('[data-view="overview"]').click()
        expect(page.locator('#kpis .kpi strong').first).to_have_text('1')
        page.locator('#demo').click()
        expect(page.locator('#status')).to_have_text('2,772 customers',timeout=90000)
        expect(page.locator('#kpis .kpi strong').first).to_have_text('2,772')
        assert len(requests)==1,requests
        for width in [320,390,768,1024,1440]:
            page.set_viewport_size({'width':width,'height':900})
            for v in ['overview','customers','segments','campaign']:
                page.locator('[data-view="'+v+'"]').click()
                expect(page.locator('#'+v)).to_be_visible()
                page.wait_for_timeout(450)
                assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1'),(width,v)
        assert len(requests)==1,requests
        page.emulate_media(reduced_motion='reduce')
        page.locator('[data-view="overview"]').click()
        expect(page.locator('#overview')).to_be_visible()
        assert page.locator('#overview').evaluate('(e)=>e.getAnimations().length')==0
        assert not errors,errors
        browser.close()
finally:
    proc.terminate()
    proc.wait(timeout=10)
