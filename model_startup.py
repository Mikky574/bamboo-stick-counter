"""Explicit server startup preparation; importing the web app loads no models."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import threading
import time

_LOCK = threading.Lock()
_STATUS = dict(state='not_started', models={})


def snapshot():
    with _LOCK:
        return deepcopy(_STATUS)


def prepare_models():
    """Finish both warmups before serving the first image request.

    A missing/busy GPU must not prevent simple CPU counting from being served.
    Failed preparations are recorded, and existing request-time resource/error
    handling remains responsible for later requests.
    """
    from auto_counter import prewarm_model
    from scene_classifier import prewarm_scene_model
    started = time.perf_counter()
    with _LOCK:
        if _STATUS['state'] == 'preparing':
            raise RuntimeError('模型启动预热已在进行。')
        _STATUS.clear()
        _STATUS.update(state='preparing', models={})
    jobs = {'counting': prewarm_model, 'scene': prewarm_scene_model}
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {name: pool.submit(prepare) for name, prepare in jobs.items()}
        for name, future in futures.items():
            try:
                result = future.result()
                if result.get('ready') is not True:
                    raise RuntimeError('模型未完成预热。')
                row = dict(state='ready', **result)
            except Exception as error:
                row = dict(state='unavailable', ready=False, error=str(error)[:700])
            with _LOCK:
                _STATUS['models'][name] = row
    with _LOCK:
        ready = all(row['ready'] for row in _STATUS['models'].values())
        _STATUS.update(state='ready' if ready else 'degraded',
                       elapsed_seconds=round(time.perf_counter()-started, 3))
        return deepcopy(_STATUS)


def shutdown_models():
    """Close this service's sessions when its server stops."""
    from auto_counter import release_model
    from scene_classifier import release_scene_model
    errors = []
    for release in (release_model, release_scene_model):
        try:
            release()
        except RuntimeError as error:
            errors.append(str(error))
    return errors
