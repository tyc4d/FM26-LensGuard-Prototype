"""Local HTTP contracts; all annotation mutations use temporary directories."""
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from physical_annotation.dataset import load_dataset
from physical_annotation.server import OriginalImages, create_app

ROOT = Path(__file__).resolve().parents[1]


class FakeImages:
    def read(self, image_id):
        return b'original-image-bytes:' + image_id.encode()


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(ROOT, annotation_directory=tmp_path / 'labels', images=FakeImages())) as client:
        yield client


def test_bootstrap_is_unreviewed_and_does_not_write(client, tmp_path):
    state = client.get('/api/state').json()
    assert len(state['annotations']) == 54
    assert state['progress']['unreviewed'] == 54
    assert not (tmp_path / 'labels').exists()
    assert 'outputs' not in state
    assert client.get('/').status_code == 200
    assert client.get('/static/app.js').status_code == 200


def test_canonical_navigation_and_image_bytes(client):
    rows = client.get('/api/state').json()['annotations']
    assert rows[0]['image_id'] == 'IMG_3483.jpeg'
    for row in rows:
        response = client.get('/api/images/' + row['image_id'])
        assert response.content == FakeImages().read(row['image_id'])
    assert client.get('/api/images/not-in-manifest.jpg').status_code == 404


def test_local_draft_and_stale_revision(client):
    state = client.get('/api/state').json()
    row = state['annotations'][0]
    row['notes'] = 'Human observation, not yet verified'
    body = {'annotation': row, 'expected_revision': 0}
    assert client.post('/api/annotations/' + row['image_id'], json=body).status_code == 403
    headers = {'X-Annotation-Token': state['token']}
    result = client.post('/api/annotations/' + row['image_id'], json=body, headers=headers)
    assert result.status_code == 200
    assert result.json()['annotations'][0]['status'] == 'DRAFT'
    assert client.get('/api/state').json()['annotations'][0]['notes'] == row['notes']
    assert client.post('/api/annotations/' + row['image_id'], json=body, headers=headers).status_code == 409


def test_cross_origin_and_unknown_routes_rejected(client):
    state = client.get('/api/state').json()
    headers = {'X-Annotation-Token': state['token'], 'Origin': 'https://external.example'}
    assert client.post('/api/annotations/IMG_3483.jpeg', json={}, headers=headers).status_code == 403
    assert client.get('/api/state', headers={'Host': 'external.example'}).status_code == 403
    assert client.post('/v1/analyze', json={}).status_code == 403
    assert client.get('/api/results').status_code == 404


def test_archive_hash_and_original_bytes(tmp_path):
    dataset = load_dataset(ROOT)
    bad = tmp_path / 'wrong.zip'
    bad.write_bytes(b'not-the-canonical-archive')
    with pytest.raises(ValueError, match='ZIP hash'):
        OriginalImages(bad, dataset)
    archive = ROOT / 'TestData.zip'
    if not archive.exists():
        pytest.skip('Original private ZIP is local only')
    import hashlib
    images = OriginalImages(archive, dataset)
    for row in dataset['records']:
        assert hashlib.sha256(images.read(row['image_id'])).hexdigest() == row['sha256']


def test_blind_api_requires_verification_and_manual_reveal(client):
    state = client.get('/api/state').json()
    headers = {'X-Annotation-Token': state['token']}
    path = '/api/model-outputs/IMG_3483.jpeg'
    assert client.get(path).status_code == 405
    assert client.post(path, json={}, headers=headers).status_code == 422
    assert client.post(path, json={'show': True}, headers=headers).status_code == 403
    row = state['annotations'][0]
    response = client.post('/api/annotations/' + row['image_id'], headers=headers, json={
        'annotation': row, 'expected_revision': state['revision'], 'verify': True, 'reviewer': 'TEST REVIEWER'})
    assert response.status_code == 200
    shown = client.post(path, json={'show': True}, headers=headers)
    assert shown.status_code == 200
    assert [r['model_name'] for r in shown.json()['outputs']] == ['Gemma', 'MiniCPM', 'Qwen', 'GPT', 'Gemini']
