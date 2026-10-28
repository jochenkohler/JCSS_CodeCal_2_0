"""
Screenshots of the app for the manual (docs/manual/figures/).

    streamlit run app.py --server.headless true --server.port 8765 &
    python docs/manual/make_screenshots.py

Needs playwright with chromium (pip install playwright && playwright install chromium).
"""

import asyncio
import os
import sys

from playwright.async_api import async_playwright

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
URL = sys.argv[1] if len(sys.argv) > 1 else 'http://localhost:8765/'
os.makedirs(OUT, exist_ok=True)


async def shot(pg, name, full=True):
    await pg.wait_for_timeout(800)
    await pg.screenshot(path=os.path.join(OUT, name), full_page=full)
    print('  ', name)


async def wait_idle(pg, ms=1500):
    """Wait until Streamlit shows no 'Running' status."""
    for _ in range(600):
        await pg.wait_for_timeout(250)
        running = await pg.locator('[data-testid="stStatusWidget"]').count()
        if not running:
            break
    await pg.wait_for_timeout(ms)


async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch()
        pg = await b.new_page(viewport={'width': 1440, 'height': 900})
        await pg.goto(URL, wait_until='networkidle')
        await wait_idle(pg, 3000)

        # ---- learning mode -------------------------------------------------------
        await shot(pg, 'learning_step1.png')
        steps = ['2 · FORM playground', '3 · One design situation', '4 · The sweep (status quo)',
                 '5 · Calibration', '6 · Verification']
        await pg.get_by_text(steps[0], exact=True).click(); await wait_idle(pg)
        await pg.get_by_text('g = c − Θ·Q  (product)', exact=True).click(); await wait_idle(pg)
        await shot(pg, 'learning_step2.png')
        await pg.get_by_text(steps[1], exact=True).click(); await wait_idle(pg)
        await shot(pg, 'learning_step3.png')
        await pg.get_by_text(steps[2], exact=True).click(); await wait_idle(pg)
        await pg.get_by_role('button', name='Run the sweep').click(); await wait_idle(pg, 3000)
        await shot(pg, 'learning_step4.png')
        await pg.get_by_text(steps[3], exact=True).click(); await wait_idle(pg)
        await pg.get_by_text('quick (1 × 5)', exact=True).click(); await wait_idle(pg)
        await pg.get_by_role('button', name='Calibrate').click(); await wait_idle(pg, 4000)
        await shot(pg, 'learning_step5.png')
        await pg.get_by_text(steps[4], exact=True).click(); await wait_idle(pg)
        await pg.get_by_role('button', name='Verify').click(); await wait_idle(pg, 4000)
        await shot(pg, 'learning_step6.png')

        # ---- lab mode ------------------------------------------------------------
        await pg.get_by_text('Lab mode', exact=True).click(); await wait_idle(pg, 4000)
        await shot(pg, 'lab_model.png')
        await pg.get_by_role('tab', name='Status quo').click(); await wait_idle(pg)
        await pg.get_by_role('button', name='Run status quo').click(); await wait_idle(pg, 4000)
        await pg.get_by_placeholder('e.g. "status quo, 8.12"').fill('status quo 8.12')
        await pg.get_by_role('button', name='Save snapshot').click(); await wait_idle(pg)
        await shot(pg, 'lab_status_quo.png')
        await pg.get_by_role('tab', name='Calibration').click(); await wait_idle(pg)
        await pg.get_by_role('button', name='Calibrate').click(); await wait_idle(pg, 5000)
        await pg.get_by_placeholder('e.g. "calibration, 8.12"').fill('calibrated 8.12')
        await pg.get_by_role('button', name='Save snapshot').click(); await wait_idle(pg)
        await shot(pg, 'lab_calibration.png')
        await pg.get_by_role('tab', name='Compare').click(); await wait_idle(pg)
        await shot(pg, 'lab_compare.png')
        await pg.get_by_role('tab', name='Sensitivity').click(); await wait_idle(pg)
        await pg.get_by_text('quick (1 × 5)', exact=True).click(); await wait_idle(pg)
        await pg.get_by_role('button', name='Run sensitivity study').click(); await wait_idle(pg, 5000)
        await shot(pg, 'lab_sensitivity.png')
        await pg.get_by_role('tab', name='Verification').click(); await wait_idle(pg)
        await pg.get_by_role('button', name='Verify').click(); await wait_idle(pg, 5000)
        await shot(pg, 'lab_verification.png')
        await b.close()


asyncio.run(main())
