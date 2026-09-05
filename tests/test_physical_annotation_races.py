"""Regression tests for delayed autosaves during navigation and verification.

All server writes use the synthetic-image fixture's temporary annotation store.
"""

import io

import pytest
from PIL import Image

from test_physical_annotation_browser import browser_page, playwright


def delay_next_annotation_write(page):
    page.evaluate("""() => {
        const original = window.fetch;
        let pending = true;
        window.fetch = async (...args) => {
            if (pending && String(args[0]).includes('/api/annotations/')) {
                pending = false;
                document.body.dataset.writeDelayed = 'yes';
                await new Promise(resolve => { window.releaseAnnotationWrite = resolve; });
            }
            return original(...args);
        };
    }""")


def wait_for_delayed_write(page):
    playwright.expect(page.locator('body')).to_have_attribute('data-write-delayed', 'yes')


def test_navigation_locks_review_controls_while_snapshot_is_saving(browser_page):
    page, store = browser_page
    delay_next_annotation_write(page)
    page.locator('#notes').fill('Snapshot before navigation')
    page.locator('#next').click()
    wait_for_delayed_write(page)
    assert page.locator('.workspace').evaluate('(node) => node.inert') is True
    playwright.expect(page.locator('#verify')).to_be_disabled()
    playwright.expect(page.locator('#previous')).to_be_disabled()
    playwright.expect(page.locator('#next')).to_be_disabled()
    # A shortcut during navigation must not verify the image being left or the
    # image that has not loaded yet. Disabled controls ignore synthesized clicks.
    page.keyboard.press('v')
    page.evaluate('() => window.releaseAnnotationWrite()')
    playwright.expect(page.locator('#filename')).to_have_text('IMG_3484.jpeg')
    state = store.state()
    assert state['annotations'][0]['notes'] == 'Snapshot before navigation'
    assert state['annotations'][0]['status'] == 'DRAFT'
    assert state['annotations'][1]['status'] == 'UNREVIEWED'
    assert page.locator('.workspace').evaluate('(node) => node.inert') is False


def test_navigation_flushes_edit_made_after_inflight_autosave_snapshot(browser_page):
    page, store = browser_page
    delay_next_annotation_write(page)
    page.locator('#notes').fill('First autosave snapshot')
    page.locator('#save').click()
    wait_for_delayed_write(page)
    # Ordinary autosave allows continued typing. Navigation must flush the most
    # recent generation rather than dropping it when the older request returns.
    page.locator('#notes').fill('Newer edit while autosave was in flight')
    page.locator('#next').click()
    page.evaluate('() => window.releaseAnnotationWrite()')
    playwright.expect(page.locator('#filename')).to_have_text('IMG_3484.jpeg')
    state = store.state()
    assert state['annotations'][0]['notes'] == 'Newer edit while autosave was in flight'
    assert state['annotations'][1]['status'] == 'UNREVIEWED'
    assert state['revision'] == 2


def test_verify_confirms_one_image_while_autosave_is_inflight(browser_page):
    page, store = browser_page
    delay_next_annotation_write(page)
    page.locator('#reviewer').fill('TEST REVIEWER')
    page.locator('#notes').fill('Only first image has been reviewed')
    page.locator('#save').click()
    wait_for_delayed_write(page)
    page.locator('#verify').click()
    assert page.locator('.workspace').evaluate('(node) => node.inert') is True
    playwright.expect(page.locator('#next')).to_be_disabled()
    page.keyboard.press('ArrowRight')
    page.evaluate('() => window.releaseAnnotationWrite()')
    playwright.expect(page.locator('#filename')).to_have_text('IMG_3484.jpeg')
    state = store.state()
    assert state['annotations'][0]['status'] == 'VERIFIED'
    assert state['annotations'][0]['reviewer'] == 'TEST REVIEWER'
    assert all(row['status'] == 'UNREVIEWED' for row in state['annotations'][1:])


def test_evidence_coordinates_use_exif_oriented_display_not_raw_jpeg_dimensions(browser_page):
    page, store = browser_page
    output = io.BytesIO()
    source = Image.new('RGB', (800, 600), '#bbccd5')
    orientation = source.getexif()
    orientation[274] = 6
    source.save(output, 'JPEG', exif=orientation)
    original_bytes = output.getvalue()
    page.route('**/api/images/*', lambda route: route.fulfill(body=original_bytes, content_type='image/jpeg'))
    page.reload()
    page.locator('#evidence-mode').check()
    bounds = page.locator('#image').bounding_box()
    assert bounds['width'] / bounds['height'] == pytest.approx(.75, abs=.005)
    page.mouse.move(bounds['x'] + bounds['width'] * .1, bounds['y'] + bounds['height'] * .2)
    page.mouse.down()
    page.mouse.move(bounds['x'] + bounds['width'] * .7, bounds['y'] + bounds['height'] * .9)
    page.mouse.up()
    page.locator('#save').click()
    playwright.expect(page.locator('#save-status')).to_have_text('Draft saved')
    row = store.state()['annotations'][0]
    assert row['bbox_coordinate_space'] == 'EXIF_ORIENTED_NORMALIZED'
    assert row['regions'][0]['bbox_normalized'] == pytest.approx([.1, .2, .7, .9], abs=.005)
    page.set_viewport_size({'width': 1100, 'height': 850})
    playwright.expect(page.locator('rect[data-region-id="R01"]')).to_be_visible()
    # Measure both elements in one browser callback so independent layout reads
    # cannot observe different resize frames or a replaced SVG child handle.
    ratios = page.locator('#image-stage').evaluate("""stage => {
        const rectangle = stage.querySelector('rect[data-region-id="R01"]').getBoundingClientRect();
        const displayedImage = stage.querySelector('img').getBoundingClientRect();
        return {width: rectangle.width / displayedImage.width,
                height: rectangle.height / displayedImage.height};
    }""")
    assert ratios['width'] == pytest.approx(.6, abs=.005)
    assert ratios['height'] == pytest.approx(.7, abs=.005)
