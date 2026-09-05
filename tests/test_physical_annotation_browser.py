"""Browser interaction tests use synthetic previews and temporary annotations only.

Run with: uv run --extra annotation --with playwright pytest tests/test_physical_annotation_browser.py
Install a test browser separately with: uv run --with playwright playwright install chromium
"""
import io
import socket
import threading
import time
from pathlib import Path

import pytest
import uvicorn
from PIL import Image

from physical_annotation.server import create_app

playwright = pytest.importorskip('playwright.sync_api')
ROOT = Path(__file__).resolve().parents[1]


class SyntheticImages:
    def read(self, image_id):
        output = io.BytesIO()
        Image.new('RGB', (800, 600), '#bbccd5').save(output, 'JPEG')
        return output.getvalue()


@pytest.fixture
def browser_page(tmp_path):
    app = create_app(ROOT, annotation_directory=tmp_path / 'labels', images=SyntheticImages())
    sock = socket.socket()
    sock.bind(('127.0.0.1', 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level='error'))
    thread = threading.Thread(target=server.run, kwargs={'sockets': [sock]}, daemon=True)
    thread.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(.02)
    try:
        with playwright.sync_playwright() as engine:
            available = list((Path.home() / '.cache/ms-playwright').glob('chromium-*/chrome-linux64/chrome'))
            browser = engine.chromium.launch(executable_path=str(available[-1]) if available else None)
            page = browser.new_page(viewport={'width': 1440, 'height': 1000})
            errors = []
            page.on('pageerror', lambda error: errors.append(str(error)))
            page.on('dialog', lambda dialog: dialog.accept())
            page.goto(f'http://127.0.0.1:{port}')
            page.locator('#field-scenario').wait_for()
            playwright.expect(page.locator('#image')).to_have_js_property('naturalWidth', 800)
            yield page, app.state.store
            assert not errors
            browser.close()
    finally:
        server.should_exit = True
        thread.join(timeout=5)
        sock.close()


def test_navigation_autosave_refresh_and_shortcuts(browser_page):
    page, store = browser_page
    assert '54' in page.locator('#dashboard').inner_text()
    page.locator('#notes').fill('Temporary test note')
    page.locator('#next').click()
    playwright.expect(page.locator("#filename")).to_have_text("IMG_3484.jpeg")
    assert store.state()['annotations'][0]['notes'] == 'Temporary test note'
    page.reload()
    playwright.expect(page.locator("#filename")).to_have_text("IMG_3484.jpeg")
    page.locator('h1').click()
    page.keyboard.press('ArrowLeft')
    playwright.expect(page.locator("#filename")).to_have_text("IMG_3483.jpeg")
    assert page.locator('#notes').input_value() == 'Temporary test note'
    page.locator('#notes').focus()
    page.keyboard.press('n')
    assert store.state()['annotations'][0]['status'] == 'DRAFT'
    page.locator('h1').click()
    page.keyboard.press('n')
    playwright.expect(page.locator("#save-status")).to_have_text("Draft saved")
    assert store.state()['annotations'][0]['status'] == 'NEEDS_REVIEW'


def test_explicit_verify_unknown_and_edit_revokes_verification(browser_page):
    page, store = browser_page
    page.locator('#reviewer').fill('TEST REVIEWER')
    page.locator('#verify').click()
    playwright.expect(page.locator("#filename")).to_have_text("IMG_3484.jpeg")
    first = store.state()['annotations'][0]
    assert first['status'] == 'VERIFIED'
    assert first['human_verified'] is True
    assert first['ground_truth_known'] is False and first['ground_truth_value'] is None
    page.locator('#previous').click()
    playwright.expect(page.locator("#filename")).to_have_text("IMG_3483.jpeg")
    page.locator('#notes').fill('Correction in test')
    page.locator('#save').click()
    playwright.expect(page.locator("#save-status")).to_have_text("Draft saved")
    assert store.state()['annotations'][0]['human_verified'] is False


@pytest.mark.parametrize('scenario', ['CALL', 'RESTAURANT_RESERVATION', 'NAVIGATION', 'SAFETY'])
def test_scenario_forms_and_known_values(browser_page, scenario):
    page, store = browser_page
    page.locator('#field-scenario').select_option(scenario)
    page.locator('#field-ground_truth_known').select_option('YES')
    if scenario in ('CALL', 'RESTAURANT_RESERVATION'):
        page.locator('#field-ground_truth_value').fill('02-5555-0100')
    elif scenario == 'NAVIGATION':
        page.locator('#field-ground_truth_value').select_option('STRAIGHT')
    else:
        page.locator('#field-ground_truth_value').select_option('FALSE')
    if scenario == 'RESTAURANT_RESERVATION':
        assert page.locator('#field-user_time').input_value() == '19:00'
        assert page.locator('#field-user_party_size').input_value() == '2'
    page.locator('#reviewer').fill('TEST REVIEWER')
    page.locator('#verify').click()
    playwright.expect(page.locator("#filename")).to_have_text("IMG_3484.jpeg")
    first = store.state()['annotations'][0]
    assert first['human_verified'] is True
    assert first['scenario'] == scenario
    assert first['ground_truth_value'] is not None
    if scenario == 'SAFETY':
        assert first['ground_truth_value'] is False


def test_draw_region_coordinates_survive_scaling_and_reload(browser_page):
    page, store = browser_page
    assert not page.locator('#evidence-mode').is_checked()
    page.locator('#evidence-mode').check()
    page.locator('#image').scroll_into_view_if_needed()
    bounds = page.locator('#image').bounding_box()
    page.mouse.move(bounds['x'] + bounds['width'] * .2, bounds['y'] + bounds['height'] * .25)
    page.mouse.down()
    page.mouse.move(bounds['x'] + bounds['width'] * .8, bounds['y'] + bounds['height'] * .75)
    page.mouse.up()
    page.get_by_label('R01: Human-transcribed text', exact=True).fill('TEST ONLY')
    page.get_by_label('R01: Control class (benchmark ground truth only)', exact=True).select_option('attacker_controlled')
    page.locator('#save').click()
    playwright.expect(page.locator('#save-status')).to_have_text('Draft saved')
    region = store.state()['annotations'][0]['regions'][0]
    assert region['bbox_normalized'] == pytest.approx([.2, .25, .8, .75], abs=.005)
    assert region['human_verified'] is False
    page.set_viewport_size({'width': 1100, 'height': 850})
    playwright.expect(page.locator('rect[data-region-id="R01"]')).to_be_visible()
    rectangle = page.locator('rect[data-region-id="R01"]').bounding_box()
    scaled = page.locator('#image').bounding_box()
    assert rectangle['width'] / scaled['width'] == pytest.approx(.6, abs=.005)
    page.reload()
    page.locator('#evidence-mode').check()
    assert page.get_by_label('R01: Human-transcribed text', exact=True).input_value() == 'TEST ONLY'
    page.get_by_role('button', name='Remove region', exact=True).click()
    page.locator('#save').click()
    playwright.expect(page.locator('#save-status')).to_have_text('Draft saved')
    assert store.state()['annotations'][0]['regions'] == []


def test_blind_outputs_never_requested_until_manual_reveal(browser_page):
    page, store = browser_page
    requests = []
    page.on('request', lambda request: requests.append(request.url))
    assert page.locator('#blind-mode').is_checked()
    assert page.locator('#show-model-outputs').is_disabled()

    assert page.locator('#model-output-results').is_hidden()
    page.locator('#reviewer').fill('TEST REVIEWER')
    page.locator('#verify').click()
    playwright.expect(page.locator('#filename')).to_have_text('IMG_3484.jpeg')
    page.locator('#previous').click()
    playwright.expect(page.locator('#filename')).to_have_text('IMG_3483.jpeg')
    assert not any('/api/model-outputs/' in url for url in requests)
    page.locator('#show-model-outputs').click()
    playwright.expect(page.locator('#model-output-results')).to_be_visible()
    assert page.locator('#model-output-results h3').all_text_contents() == ['Gemma', 'MiniCPM', 'Qwen', 'GPT', 'Gemini']
    page.locator('#notes').fill('Temporary correction')
    assert page.locator('#model-output-results').is_hidden()
    assert page.locator('#show-model-outputs').is_disabled()


def test_freeze_dialog_is_explicit_and_blocked_until_review(browser_page):
    page, store = browser_page
    page.locator('#freeze-ground-truth').click()
    playwright.expect(page.locator('#freeze-dialog')).to_be_visible()
    assert 'Unresolved: 54' in page.locator('#freeze-dialog').inner_text()
    page.locator('#freeze-confirm-text').fill('FREEZE')
    assert page.locator('#confirm-freeze').is_disabled()
    page.locator('#freeze-cancel').click()
    assert not list(store.directory.glob('ground_truth_v*'))


def test_explicit_freeze_downloads_and_v2_correction(browser_page):
    # Artificial verification is limited to this temporary test store.
    page, store = browser_page
    state = store.state()
    for row in state['annotations']:
        state = store.save(row['image_id'], row, state['revision'], verify=True, reviewer='TEST REVIEWER')
    page.reload()
    page.locator('#field-scenario').wait_for()
    with page.expect_download() as download:
        page.locator('#export-jsonl').click()
    assert download.value.suggested_filename == 'ground_truth_draft.jsonl'
    page.locator('#freeze-ground-truth').click()
    assert page.locator('#confirm-freeze').is_disabled()
    assert not list(store.directory.glob('ground_truth_v*'))
    page.locator('#freeze-confirm-text').fill('FREEZE')
    page.locator('#confirm-freeze').click()
    playwright.expect(page.locator('#freeze-dialog h2')).to_have_text('Frozen v1')
    first_bytes = (store.directory / 'ground_truth_v1.jsonl').read_bytes()
    page.get_by_role('button', name='Close', exact=True).click()
    page.locator('#notes').fill('Temporary correction for v2 test')
    page.locator('#verify').click()
    playwright.expect(page.locator('#filename')).to_have_text('IMG_3484.jpeg')
    page.locator('#freeze-ground-truth').click()
    page.locator('#freeze-change-reason').fill('Test-only correction')
    page.locator('#freeze-confirm-text').fill('FREEZE')
    page.locator('#confirm-freeze').click()
    playwright.expect(page.locator('#freeze-dialog h2')).to_have_text('Frozen v2')
    assert (store.directory / 'ground_truth_v1.jsonl').read_bytes() == first_bytes
    import json
    manifest = json.loads((store.directory / 'ground_truth_v2_manifest.json').read_text())
    assert manifest['parent_version'] == 'v1'
    assert manifest['changed_image_ids'] == ['IMG_3483.jpeg']
