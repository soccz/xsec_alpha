#!/usr/bin/env python3
"""Verify the two-page site against real encrypted data and browser-only states.

Requires Selenium, Chrome, a cached chromedriver, BeautifulSoup, and Node.
Without --url, a temporary loopback server serves output/site-preview.
No fixture is written into an export or an experiment ledger.
"""

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from threading import Thread
import time
from urllib.parse import urljoin

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.dashboard_export import PIN_DEFAULT  # noqa: E402


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *_args):
        pass


def check_source(site):
    for relative in ('index.html', 'dashboard/index.html'):
        path = 'projects/xsec-alpha/' + relative
        soup = BeautifulSoup((site / path).read_text(), 'html.parser')
        ids = [node['id'] for node in soup.select('[id]')]
        assert len(ids) == len(set(ids)), f'Duplicate IDs: {relative}'
        for script in soup.find_all('script', src=False):
            if script.get('type') in (None, 'module', 'text/javascript'):
                subprocess.run(['node', '--check'], input=script.get_text(), text=True,
                               capture_output=True, check=True)
        original = subprocess.check_output([
            'git', '-C', str(ROOT / 'output/dashboard_publication/git'), 'show',
            'refs/remotes/origin/main:' + path,
        ], text=True)
        previous = BeautifulSoup(original, 'html.parser')
        for node in previous.select('section[id], article[id]'):
            assert soup.find(id=node['id']), f'Lost historical anchor: {node["id"]}'
    print('JavaScript syntax, unique IDs, historical section/article anchors: PASS')


def check_width(driver, width):
    assert driver.execute_script('return document.documentElement.scrollWidth') <= width + 2
    for element in driver.find_elements(By.CSS_SELECTOR, '.page-switch a, .project-brand, .ops-facts dd, .coin-tile h3'):
        assert driver.execute_script('return arguments[0].scrollWidth <= arguments[0].clientWidth + 2', element), element.tag_name


def assert_canvas(driver, canvas_id):
    assert driver.execute_script('''
        const c = document.getElementById(arguments[0]);
        const p = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
        let n = 0; for(let i=3;i<p.length;i+=4) if(p[i]>0) n++;
        return c.width > 200 && c.height > 80 && n > 150;
    ''', canvas_id), canvas_id


def check_browser(base, output):
    drivers = list((Path.home() / '.cache/selenium/chromedriver').glob('**/chromedriver'))
    executable = shutil.which('chromedriver') or (str(max(drivers, key=lambda p: p.stat().st_mtime)) if drivers else None)
    if not executable:
        raise RuntimeError('Install a Chrome-compatible chromedriver first')
    with tempfile.TemporaryDirectory(dir=output, prefix='chrome-') as profile:
        options = webdriver.ChromeOptions()
        options.binary_location = shutil.which('google-chrome') or shutil.which('chromium')
        for flag in ('--headless=new', '--no-sandbox', '--disable-dev-shm-usage', '--user-data-dir=' + profile):
            options.add_argument(flag)
        options.set_capability('goog:loggingPrefs', {'browser': 'ALL'})
        driver = webdriver.Chrome(service=Service(executable), options=options)
        try:
            wait = WebDriverWait(driver, 45)
            driver.set_script_timeout(45)
            for width, height in ((1440, 1000), (390, 844), (360, 740)):
                driver.execute_cdp_cmd('Emulation.setDeviceMetricsOverride', {
                    'width': width, 'height': height, 'deviceScaleFactor': 1, 'mobile': width < 500,
                })
                driver.get(urljoin(base, 'projects/xsec-alpha/'))
                wait.until(lambda d: len(d.find_elements(By.CSS_SELECTOR, '.chapter-group')) == 2)
                assert driver.find_element(By.CSS_SELECTOR, '.page-switch [aria-current]').text == '개발일지'
                assert '60회로 검증한다' not in driver.find_element(By.CSS_SELECTOR, '.hero').text
                check_width(driver, width)
                assert_canvas(driver, 'trialInterval')
                driver.save_screenshot(str(output / f'story-{width}.png'))
                driver.execute_script("location.hash='post-trial-result'")
                wait.until(lambda d: d.find_element(By.ID, 'post-trial-result').is_displayed())
                assert driver.execute_script("return document.querySelector('#post-trial').closest('details').open")
                driver.execute_script("location.hash='operating-closeout'")
                wait.until(lambda d: d.find_element(By.ID, 'operating-closeout').is_displayed())
                driver.get(urljoin(base, 'projects/xsec-alpha/dashboard/'))
                field = wait.until(lambda d: d.find_element(By.ID, 'pinInput'))
                assert not driver.find_elements(By.CSS_SELECTOR, '#operatorCoinTiles .coin-tile')
                assert driver.find_element(By.CSS_SELECTOR, '.page-switch [aria-current]').text == '운영 대시보드'
                driver.save_screenshot(str(output / f'locked-{width}.png'))
                field.send_keys(PIN_DEFAULT)
                driver.find_element(By.ID, 'pinSubmit').click()
                wait.until(lambda d: d.execute_script("return document.body.dataset.loaded === 'true'"))
                assert not driver.execute_script('return document.body.dataset.loadError')
                wait.until(lambda d: d.find_elements(By.CSS_SELECTOR, '#operatorCoinTiles .coin-tile'))
                driver.execute_script('window.scrollTo(0,0)')
                time.sleep(0.3)
                check_width(driver, width)
                position = driver.execute_script("return document.getElementById('operator-report').getBoundingClientRect().top")
                assert position < height - 50, ('Candidates below first viewport', width, position)
                tile_top = driver.execute_script("return document.querySelector('.coin-tile').getBoundingClientRect().top")
                assert tile_top < height - 70, ('Candidate tiles below first viewport', width, tile_top)
                assert len(driver.find_elements(By.CSS_SELECTOR, '.evidence-group')) == 6
                assert not driver.find_elements(By.CSS_SELECTOR, '.evidence-group[open]')
                assert_canvas(driver, 'pilotDifferenceChart')
                driver.save_screenshot(str(output / f'dashboard-{width}.png'))
                driver.find_element(By.CSS_SELECTOR, '.hero a[href="#experiment-overview"]').click()
                time.sleep(0.4)
                driver.save_screenshot(str(output / f'experiment-{width}.png'))
                for anchor in ('forecast-audit', 'rotation-pilot', 'prospective-experiment', 'cumulative', 'ic', 'contract'):
                    driver.execute_script('location.hash = arguments[0]', anchor)
                    wait.until(lambda d: d.find_element(By.ID, anchor).is_displayed())
                    check_width(driver, width)
                assert_canvas(driver, 'icChart')
                driver.find_element(By.ID, 'navToggle').click()
                assert driver.find_element(By.ID, 'navToggle').get_attribute('aria-expanded') == 'true'
                driver.find_element(By.CSS_SELECTOR, '#navDrawer a[href="#operator-report"]').click()
                wait.until(lambda d: d.find_element(By.ID, 'navToggle').get_attribute('aria-expanded') == 'false')
                assert not driver.execute_script("return document.querySelector('#navDrawer').inert === false")
                print(f'Viewport {width}: routing, protected data, first-viewport candidates, charts, disclosures PASS')

            # Exercise missing, delayed, failed, and negative results only in memory.
            assert driver.execute_async_script('''
                const done=arguments[arguments.length-1];
                fetchJson('data/summary.json').then(p=>decryptPayload(p,arguments[0])).then(s=>{window.fixtureSource=s;done(true)}).catch(()=>done(false));
            ''', PIN_DEFAULT)
            driver.execute_script('renderOverview({}); renderOperatorReport({});')
            assert '자료 없음' in driver.find_element(By.ID, 'operatorCoinTiles').text
            assert driver.find_element(By.ID, 'pilotProgressValue').text == '— / —'
            assert '전환 적용 기록 없음' in driver.find_element(By.ID, 'pilotExercise').text
            assert '미집계' in driver.find_element(By.ID, 'regimeOverview').text
            driver.execute_script('''
                const s=structuredClone(window.fixtureSource);
                s.asof='2020-01-01T00:00:00Z';
                s.operator_report.telegram={state:'pending'};
                s.gates.short={status:'FREEZE',reason:'fixture'};
                s.rotation_pilot.decision.counts={matured:1,pending:1,missed:1,invalid:1,exercised:0,exercised_dates:0};
                s.rotation_pilot.recent_windows=[{signal_at:s.asof,status:'matured',paired_pp:-1},{signal_at:'2020-01-01T06:00:00Z',status:'missed',paired_pp:null}];
                renderOverview(s); renderOperatorReport(s);
                document.getElementById('opsWarningDetail').open = true;
            ''')
            assert '게시 자료 지연' in driver.find_element(By.ID, 'opsAlerts').text
            assert '동결' in driver.find_element(By.ID, 'opsFacts').text
            assert '누락 1' in driver.find_element(By.ID, 'pilotCoverageLegend').text
            assert driver.execute_script("return Chart.getChart('pilotDifferenceChart').data.datasets[0].data") == [-1, None]
            driver.execute_script("location.hash='experiment-overview'")
            time.sleep(0.3)
            driver.save_screenshot(str(output / 'browser-only-failure.png'))
            driver.execute_script('renderOverview(window.fixtureSource); renderOperatorReport(window.fixtureSource); delete window.fixtureSource;')
            errors = [r for r in driver.get_log('browser') if r['level'] == 'SEVERE' and r.get('source') == 'javascript']
            assert not errors, errors
            print('Browser-only empty/stale/FREEZE/missing/negative states and nonzero canvas pixels: PASS')
            driver.execute_cdp_cmd('Network.enable', {})
            for blocked, expected_error in ((['*chart.umd.min.js*'], False), (['*/data/history.json*'], True)):
                driver.execute_cdp_cmd('Network.setBlockedURLs', {'urls': blocked})
                driver.get(urljoin(base, 'projects/xsec-alpha/dashboard/'))
                wait.until(lambda d: d.find_element(By.ID, 'pinInput')).send_keys(PIN_DEFAULT)
                driver.find_element(By.ID, 'pinSubmit').click()
                wait.until(lambda d: d.find_elements(By.CSS_SELECTOR, '#operatorCoinTiles .coin-tile'))
                if expected_error:
                    wait.until(lambda d: d.execute_script("return document.body.dataset.loadError === 'true'"))
                    assert '일부 데이터 로드 실패' in driver.find_element(By.ID, 'heroMeta').text
                else:
                    wait.until(lambda d: d.execute_script("return document.body.dataset.loaded === 'true'"))
                    assert '차트 로드 실패' in driver.find_element(By.ID, 'pilotChartNote').text
                assert driver.find_element(By.ID, 'opsFacts').text
            driver.execute_cdp_cmd('Network.setBlockedURLs', {'urls': []})
            print('Chart CDN and historical payload outages preserve current report and warnings: PASS')
        finally:
            driver.quit()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', help='Public site origin, e.g. https://soccz.github.io/')
    parser.add_argument('--output', default=str(ROOT / 'output/site-ux-verification'))
    args = parser.parse_args()
    site = ROOT / 'output/site-preview'
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    check_source(site)
    if args.url:
        check_browser(args.url, output)
    else:
        server = ThreadingHTTPServer(('127.0.0.1', 0), partial(QuietHandler, directory=str(site)))
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            check_browser(f'http://127.0.0.1:{server.server_port}/', output)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
    print(json.dumps({'verified': True, 'screenshots': str(output)}))


if __name__ == '__main__':
    main()
