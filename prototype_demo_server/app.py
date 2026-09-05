"""Loopback multipart service. Uploaded bytes exist only for one request."""
import asyncio
import io
import logging
import tempfile
from contextlib import asynccontextmanager
from time import perf_counter
from uuid import uuid4
from typing import Any, Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from .policy import ActionValidationError, authorize, proposal
from .runtime import LocalRuntime, SPEC

MAX_BYTES = 10 * 1024 * 1024


class AnalyzeResponse(BaseModel):
    contract_version: Literal['lensguard-demo-v1'] = 'lensguard-demo-v1'
    request_id: str
    model: dict[str, Any]
    input: dict[str, Any]
    output: dict[str, Any]
    provenance: dict[str, Any] | None
    policy: dict[str, Any] | None
    timing: dict[str, float]


def create_app(runtime=None):
    runtime = runtime if runtime is not None else LocalRuntime()
    spec = getattr(runtime, 'spec', SPEC)
    lock = asyncio.Lock()
    state = {'status': 'unloaded', 'error': None, 'warmed': False}

    @asynccontextmanager
    async def lifespan(app):
        yield
        runtime.close()

    app = FastAPI(title='LensGuard Prototype Demo Runtime', lifespan=lifespan)

    @app.get('/health')
    def health():
        return {'status': state['status'], 'model_loaded': runtime.loaded,
                'model_profile': spec['family_alias'], 'device': 'cuda', 'phase': 'real',
                'error': state['error'], 'provenance': 'transport_only', 'policy': 'deterministic',
                'gpu': getattr(runtime, 'gpu', None)}

    async def analyze(image, user_request, scenario_id, mode):
        if mode != 'action_only':
            raise HTTPException(422, 'Only the frozen action_only mode is supported.')
        started = perf_counter()
        data = await image.read(MAX_BYTES + 1)
        await image.close()
        if not data or len(data) > MAX_BYTES:
            raise HTTPException(413, 'Image must be nonempty and at most 10 MiB.')
        try:
            with Image.open(io.BytesIO(data)) as decoded:
                if decoded.format not in ('JPEG', 'PNG') or decoded.width * decoded.height > 16_000_000:
                    raise ValueError('Use JPEG/PNG with at most 16 megapixels.')
                decoded.verify()
        except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
            raise HTTPException(422, 'Invalid image. Use JPEG/PNG with at most 16 megapixels.') from exc
        if lock.locked():
            raise HTTPException(409, 'Model loading or inference already in progress. Try again after it completes.')
        async with lock:
            state.update(status='processing' if runtime.loaded else 'loading', error=None)
            try:
                with tempfile.NamedTemporaryFile(suffix='.image') as file:
                    file.write(data)
                    file.flush()
                    # shield ensures cancellation cannot release the model lock while
                    # its worker still generates or delete its temporary input early.
                    task = asyncio.create_task(asyncio.to_thread(runtime.infer, file.name, user_request))
                    try:
                        inferred = await asyncio.shield(task)
                    except asyncio.CancelledError:
                        await task
                        raise
                action = inferred['parsed_action']
                policy_started = perf_counter()
                policy_error = None
                validation_error = None
                try:
                    policy = authorize(action, user_request) if action else None
                except ActionValidationError as exc:
                    policy = None
                    validation_error = str(exc)
                except Exception:
                    logging.exception('Deterministic policy unavailable')
                    policy = None
                    policy_error = 'Policy unavailable; automatic execution must be withheld.'
                policy_ms = (perf_counter() - policy_started) * 1000
                state['status'] = 'ready'
                return AnalyzeResponse(request_id=f'infer_{uuid4().hex}',
                    model={'profile': spec['family_alias'], 'model_id': spec['model_id'], 'revision': spec['revision']},
                    input={'user_request': user_request, 'image_received': True, 'scenario_id': scenario_id},
                    output={'raw_text': inferred['raw_text'], 'parsed': action is not None,
                            'proposed_action': proposal(action) if action and validation_error is None else None,
                            'native_action': action, 'candidate_action': inferred.get('candidate_action'),
                            'validation_error': validation_error, 'policy_error': policy_error,
                            'diagnostics': inferred['diagnostics'], 'metadata': inferred.get('metadata', {})},
                    provenance={'kind': 'transport_only', 'semantic_grounding': 'unavailable',
                                'lineage': ['image_upload', 'local_vlm', 'proposed_action'],
                                'delegated': policy['delegated'] if policy else False},
                    policy=policy, timing={**inferred['timing'], 'policy_ms': policy_ms, 'total_ms': (perf_counter() - started) * 1000})
            except Exception as exc:
                logging.exception('Local runtime failed')
                detail = str(exc)
                code = 'CUDA_OOM' if 'out of memory' in detail.lower() else 'RUNTIME_UNAVAILABLE'
                state.update(status='error', error=f'{code}: {detail[:300]}')
                raise HTTPException(503, state['error']) from exc

    @app.post('/v1/analyze', response_model=AnalyzeResponse)
    async def endpoint(image: UploadFile = File(...), user_request: str = Form(..., min_length=1, max_length=4000), scenario_id: str | None = Form(None), mode: str = Form('action_only')):
        if not user_request.strip():
            raise HTTPException(422, 'User request must not be blank.')
        return await analyze(image, user_request, scenario_id, mode)

    @app.post('/warmup')
    async def warmup():
        if state['warmed']:
            return {'status': 'already_warmed'}
        image = io.BytesIO()
        Image.new('RGB', (320, 240), 'white').save(image, format='PNG')
        image.seek(0)
        response = await analyze(UploadFile(filename='warmup.png', file=image), 'Describe no action; return NONE.', None, 'action_only')
        state['warmed'] = True
        return {'status': 'warmed', 'parsed': response.output['parsed'], 'timing': response.timing}

    return app
