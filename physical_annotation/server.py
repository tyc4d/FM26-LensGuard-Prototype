"""Loopback-only human review transport. No model, OCR, or action-model imports."""
from __future__ import annotations

import hashlib
import secrets
import zipfile
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from .dataset import load_dataset
from .model_outputs import outputs_for_image
from .storage import AnnotationStore, RevisionConflict

ROOT = Path(__file__).resolve().parents[1]
STATIC = Path(__file__).parent / 'static'


class OriginalImages:
    """Read original archive bytes by canonical ID; never extract or rewrite them."""

    def __init__(self, archive, dataset):
        self.archive = Path(archive)
        self.records = {r['image_id']: r for r in dataset['records']}
        digest = hashlib.sha256()
        with self.archive.open('rb') as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(block)
        if digest.hexdigest() != dataset['archive_sha256']:
            raise ValueError('Dataset ZIP hash differs from the canonical input manifest')

    def read(self, image_id):
        row = self.records[image_id]
        with zipfile.ZipFile(self.archive) as archive:
            data = archive.read(row['archive_member'])
        if hashlib.sha256(data).hexdigest() != row['sha256']:
            raise ValueError('Original image hash mismatch')
        return data


def create_app(root=ROOT, *, archive=None, annotation_directory=None, images=None):
    root = Path(root)
    dataset = load_dataset(root)
    store = AnnotationStore(root, dataset, directory=annotation_directory)
    images = images if images is not None else OriginalImages(archive or root / 'TestData.zip', dataset)
    token = secrets.token_urlsafe(32)
    app = FastAPI(title='Physical Ground Truth — Human Review', docs_url=None, redoc_url=None,
                  openapi_url=None)
    app.state.store = store

    @app.middleware('http')
    async def local_requests(request: Request, call_next):
        host = request.url.hostname
        if host not in {'localhost', '127.0.0.1', 'testserver'}:
            return JSONResponse({'detail': 'Use localhost'}, status_code=403)
        if request.method not in {'GET', 'HEAD'}:
            if request.headers.get('x-annotation-token') != token:
                return JSONResponse({'detail': 'Refresh this local annotation page'}, status_code=403)
            if request.headers.get('origin') not in {None, str(request.base_url).rstrip('/')}:
                return JSONResponse({'detail': 'Cross-origin write rejected'}, status_code=403)
        response = await call_next(request)
        response.headers['Cache-Control'] = 'no-store'
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Content-Security-Policy'] = (
            "default-src 'self'; img-src 'self' blob:; style-src 'self' 'unsafe-inline'; "
            "script-src 'self'; connect-src 'self'; frame-ancestors 'none'"
        )
        return response

    @app.exception_handler(ValueError)
    async def invalid(_request, exc):
        return JSONResponse({'detail': str(exc)}, status_code=422)

    @app.exception_handler(FileExistsError)
    async def conflict(_request, exc):
        return JSONResponse({'detail': str(exc)}, status_code=409)

    app.add_exception_handler(RevisionConflict, conflict)

    @app.exception_handler(PermissionError)
    async def forbidden(_request, exc):
        return JSONResponse({'detail': str(exc)}, status_code=403)

    @app.get('/')
    def index():
        return FileResponse(STATIC / 'index.html')

    @app.get('/api/state')
    def state():
        return {**store.state(), 'token': token, 'archive_sha256': dataset['archive_sha256']}

    @app.get('/api/images/{image_id}')
    def image(image_id: str):
        if image_id not in {r['image_id'] for r in dataset['records']}:
            raise HTTPException(404, 'Unknown canonical image ID')
        return Response(images.read(image_id), media_type='image/jpeg')

    @app.post('/api/annotations/{image_id}')
    def save(image_id: str, body: dict):
        if image_id not in {r['image_id'] for r in dataset['records']}:
            raise HTTPException(404, 'Unknown canonical image ID')
        if not isinstance(body.get('annotation'), dict) or 'expected_revision' not in body:
            raise ValueError('annotation and expected_revision are required')
        return store.save(image_id, body['annotation'], body['expected_revision'],
                          verify=body.get('verify', False), reviewer=body.get('reviewer'),
                          confirm_attacker_match=body.get('confirm_attacker_match', False))

    @app.post('/api/model-outputs/{image_id}')
    def model_outputs(image_id: str, body: dict):
        if body.get('show') is not True:
            raise ValueError('Model outputs require an explicit Show model outputs request')
        annotation = next((a for a in store.state()['annotations'] if a['image_id'] == image_id), None)
        if annotation is None:
            raise HTTPException(404, 'Unknown canonical image ID')
        return outputs_for_image(root, image_id, annotation, blind_mode=body.get('blind_mode', True))

    app.mount('/static', StaticFiles(directory=STATIC), name='static')
    return app
