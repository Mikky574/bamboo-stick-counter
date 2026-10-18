"""Experimental CPU TinyCLIP scene hints, isolated from counting environments.

classify_scene accepts OpenCV BGR pixels only and never returns an object count.
Its text similarities and rejection thresholds are not calibrated probabilities.
The smart counter can use accepted scenes as routing hints; the chosen counting
method must still validate its own predictions. An accepted scene is not proof
that the photo is countable, in scope, or correctly classified.
"""
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import traceback

from inference_process import ResidentWorker
from supervised_scene import active_profile
from runtime_capacity import SCENE_IDLE_TIMEOUT_SECONDS

ROOT = Path(__file__).resolve().parent
MODEL_ID = 'wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M'
ASSETS = ROOT/'models/scene_classifier/tinyclip_vit_8m16_text3m_yfcc15m'
PYTHON = ROOT/'.venv-countgdpp/Scripts/python.exe'
SCENES = {
    'sharp': (
        'a close-up photo of the pointed tips of a dense bundle of bamboo skewers',
        'a bundle of sharpened wooden toothpicks with many pointed tips'),
    'flat_end': (
        'front view of a bundle of wooden dowels showing flat circular end faces',
        'looking straight at the round flat ends of packed bamboo sticks'),
    'laid_rods': (
        'whole straight wooden rods laid flat side by side on a table',
        'full length bamboo sticks laid out separately in parallel, viewed from the side'),
    'chopsticks': (
        'several pairs of wooden chopsticks lying flat on a table',
        'full length chopsticks arranged side by side on a plain surface'),
    'cotton': (
        'top view of densely packed white cotton swab heads in a box',
        'a container filled with white cotton buds seen from above'),
    'rice': (
        'scattered individual uncooked rice grains on a plain background',
        'a pile of small white rice grains'),
    'mung': (
        'green mung beans on a plate or in a bowl',
        'small green mung bean seeds scattered on a plain surface'),
    'coins': (
        'metal coins scattered on a table',
        'top view of round metal coins laid flat'),
    'other': (
        'an unrelated photo with no countable objects',
        'a landscape, person, document or household scene'),
}
# Chopsticks and plain rods use the same scene route. Distinguishing these names
# is optional; distinguishing their side view from end-on bundles is essential.
FAMILIES = {name: ('laid_rods' if name == 'chopsticks' else name) for name in SCENES}
MIN_SIMILARITY = .22
MIN_MARGIN = .025
_RUNTIME = None
_LOCK = threading.Lock()


class _CPUWorker(ResidentWorker):
    def snapshot(self):
        result = super().snapshot()
        result['cache_device'] = 'cpu' if result['model_resident'] else None
        return result


_WORKER = _CPUWorker(idle_timeout_seconds=SCENE_IDLE_TIMEOUT_SECONDS)
SCENE_REQUEST_TIMEOUT_SECONDS = 30


def readiness():
    profile = active_profile()
    if profile:
        required = [PYTHON, profile['checkpoint_path']]
        return dict(installed=all(p.is_file() for p in required),
                    missing=[str(p.relative_to(ROOT)) for p in required if not p.is_file()],
                    model_id=profile['model_id'],device='cpu',runtime='isolated_cpu_session',
                    supports=list(SCENES),experimental=True,**_WORKER.snapshot())
    required = [PYTHON, *(ASSETS/name for name in (
        'model.safetensors', 'config.json', 'preprocessor_config.json', 'tokenizer.json'))]
    return dict(installed=all(p.is_file() for p in required),
                missing=[str(p.relative_to(ROOT)) for p in required if not p.is_file()],
                model_id=MODEL_ID, device='cpu', runtime='isolated_cpu_session',
                supports=list(SCENES), experimental=True, **_WORKER.snapshot())


def release_scene_model():
    if not _LOCK.acquire(blocking=False):
        raise RuntimeError('场景识别正在运行，完成后才能释放。')
    try:
        released = _WORKER.release()
    finally:
        _LOCK.release()
    return dict(released=released, **readiness())


def _worker_environment():
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='-1', PYTORCH_NVML_BASED_CUDA_CHECK='0',
               PYTHONIOENCODING='utf-8', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
               TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1',
               TRANSFORMERS_OFFLINE='1')
    return env


def prewarm_scene_model():
    """Load and warm the CPU scene model in the same session as classification."""
    started = time.perf_counter()
    if not readiness()['installed']:
        raise RuntimeError('场景分类模型资产尚未准备好。')
    if not _LOCK.acquire(blocking=False):
        raise RuntimeError('本项目已有场景识别任务正在运行。')
    try:
        output = ROOT/'outputs'
        output.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='scene_warmup_', dir=output) as temporary:
            result = Path(temporary)/'result.json'
            try:
                reused = _WORKER.request(
                    [str(PYTHON), str(Path(__file__).resolve()), '--serve'],
                    dict(action='warmup', output=str(result.resolve())),
                    cwd=ROOT, env=_worker_environment(),
                    log_path=output/'scene_classifier.log', timeout=SCENE_REQUEST_TIMEOUT_SECONDS,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            except RuntimeError as error:
                raise RuntimeError('场景模型预热未完成：' + str(error)) from error
            try:
                payload = json.loads(result.read_text(encoding='utf-8'))
                if payload.get('ready') is not True or payload.get('device') != 'cpu':
                    raise RuntimeError('场景模型预热返回格式无效。')
            except Exception:
                _WORKER.release()
                raise
            payload['meta'] = dict(cache_reused=reused, worker_exited=False,
                                  total_elapsed_seconds=round(time.perf_counter()-started, 4))
    finally:
        _LOCK.release()
    return payload


def classify_scene(image):
    """Return a scene hint from BGR uint8 pixels; never consult sample metadata."""
    import cv2
    import numpy as np
    started = time.perf_counter()
    if (not isinstance(image, np.ndarray) or image.dtype != np.uint8 or
            image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 16):
        raise ValueError('场景识别需要有效的 BGR uint8 图片。')
    if not readiness()['installed']:
        raise RuntimeError('场景分类模型资产尚未准备好。')
    if not _LOCK.acquire(blocking=False):
        raise RuntimeError('本项目已有场景识别任务正在运行。')
    try:
        output = ROOT/'outputs'
        output.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='scene_', dir=output) as temporary:
            folder = Path(temporary)
            source, result = folder/'image.png', folder/'result.json'
            ok, encoded = cv2.imencode('.png', image)
            if not ok:
                raise ValueError('场景图片编码失败。')
            encoded.tofile(source)
            try:
                reused = _WORKER.request(
                    [str(PYTHON), str(Path(__file__).resolve()), '--serve'],
                    dict(image=str(source.resolve()), output=str(result.resolve())),
                    cwd=ROOT, env=_worker_environment(), log_path=output/'scene_classifier.log', timeout=SCENE_REQUEST_TIMEOUT_SECONDS,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            except RuntimeError as error:
                raise RuntimeError('场景识别未完成：' + str(error)) from error
            try:
                payload = json.loads(result.read_text(encoding='utf-8'))
                if not isinstance(payload.get('meta'), dict):
                    raise RuntimeError('场景分类返回格式无效。')
            except Exception:
                _WORKER.release()
                raise
            payload['meta'].update(cache_reused=reused, worker_exited=False)
    finally:
        _LOCK.release()
    payload['meta']['total_elapsed_seconds'] = round(time.perf_counter()-started, 4)
    return payload


def _load_runtime():
    global _RUNTIME
    if _RUNTIME is not None:
        return _RUNTIME, 0.
    started = time.perf_counter()
    import torch
    torch.set_num_threads(4)
    profile = active_profile()
    if profile:
        from supervised_scene import load_runtime
        _RUNTIME = load_runtime(profile)
        return _RUNTIME, time.perf_counter()-started
    from transformers import CLIPModel, CLIPProcessor
    provenance_file = ASSETS/'provenance.json'
    if provenance_file.is_file():
        provenance = json.loads(provenance_file.read_text(encoding='utf-8'))
        wanted = provenance.get('files', {}).get('model.safetensors', {}).get('sha256')
        if wanted:
            with (ASSETS/'model.safetensors').open('rb') as stream:
                if hashlib.file_digest(stream, 'sha256').hexdigest() != wanted:
                    raise RuntimeError('TinyCLIP 权重校验失败。')
    model, loading = CLIPModel.from_pretrained(
        str(ASSETS), local_files_only=True, use_safetensors=True,
        torch_dtype=torch.float32, output_loading_info=True)
    if loading.get('missing_keys') or loading.get('mismatched_keys'):
        raise RuntimeError('TinyCLIP 预训练参数未完整加载。')
    model.eval().to('cpu')
    processor = CLIPProcessor.from_pretrained(str(ASSETS), local_files_only=True)
    names = list(SCENES)
    phrases = [text for name in names for text in SCENES[name]]
    text_inputs = processor(text=phrases, return_tensors='pt', padding=True, truncation=True,
                            max_length=model.config.text_config.max_position_embeddings)
    with torch.inference_mode():
        features = model.get_text_features(**text_inputs)
        features = features/features.norm(dim=-1, keepdim=True)
        features = features.reshape(len(names), 2, -1).mean(dim=1)
        features = features/features.norm(dim=-1, keepdim=True)
    statistics = dict(total_parameters=sum(p.numel() for p in model.parameters()),
                      vision_parameters=sum(p.numel() for p in model.vision_model.parameters()),
                      text_parameters=sum(p.numel() for p in model.text_model.parameters()),
                      text_token_embedding_parameters=model.text_model.embeddings.token_embedding.weight.numel(),
                      text_token_embedding_shape=list(model.text_model.embeddings.token_embedding.weight.shape),
                      embedding_dimension=int(features.shape[1]))
    _RUNTIME = dict(model=model, processor=processor, names=names,
                    text_features=features, statistics=statistics)
    return _RUNTIME, time.perf_counter()-started


def _warmup_local():
    """Exercise a synthetic image without creating a user-facing scene result."""
    started = time.perf_counter()
    runtime, load_seconds = _load_runtime()
    import torch
    from PIL import Image
    image = Image.new('RGB', (224, 224), color=(127, 127, 127))
    inference_started = time.perf_counter()
    with torch.inference_mode():
        if runtime.get('kind') == 'supervised':
            from supervised_scene import preprocess
            runtime['model'](preprocess(image)[None])
        else:
            inputs = runtime['processor'](images=[image, image], return_tensors='pt')
            runtime['model'].get_image_features(**inputs)
    return dict(ready=True, device='cpu', model_load_seconds=round(load_seconds, 4),
                warmup_seconds=round(time.perf_counter()-inference_started, 4),
                elapsed_seconds=round(time.perf_counter()-started, 4))


def _classify_local(image_path):
    started = time.perf_counter()
    runtime, load_seconds = _load_runtime()
    if runtime.get('kind') == 'supervised':
        from supervised_scene import classify_local
        result = classify_local(image_path,runtime)
        result['meta']['model_load_seconds'] = round(load_seconds,4)
        return result
    import torch
    import numpy as np
    from PIL import Image, ImageOps
    image = Image.open(image_path).convert('RGB')
    # Test native central crop alongside an uncropped square-padded view. A
    # disagreement is evidence to abstain, rather than silently losing rod ends.
    pixels = np.asarray(image)
    border = np.concatenate((pixels[0], pixels[-1], pixels[:, 0], pixels[:, -1]))
    fill = tuple(int(v) for v in np.median(border, axis=0))
    square = ImageOps.pad(image, (max(image.size),)*2, color=fill)
    inference_started = time.perf_counter()
    inputs = runtime['processor'](images=[image, square], return_tensors='pt')
    with torch.inference_mode():
        features = runtime['model'].get_image_features(**inputs)
        features = features/features.norm(dim=-1, keepdim=True)
        similarities = (features @ runtime['text_features'].T).cpu().numpy()
    averaged = similarities.mean(axis=0)
    order = np.argsort(-averaged)
    candidates = [dict(scene=runtime['names'][i], score=round(float(averaged[i]), 5)) for i in order]
    family_scores = {}
    for row in candidates:
        family = FAMILIES[row['scene']]
        family_scores[family] = max(family_scores.get(family, -1.), row['score'])
    families = sorted(family_scores, key=family_scores.get, reverse=True)
    margin = family_scores[families[0]]-family_scores[families[1]]
    view_winners = [runtime['names'][int(np.argmax(row))] for row in similarities]
    consistent_views = FAMILIES[view_winners[0]] == FAMILIES[view_winners[1]]
    best = candidates[0]
    accepted = (best['scene'] != 'other' and best['score'] >= MIN_SIMILARITY and
                margin >= MIN_MARGIN and consistent_views)
    return dict(scene=best['scene'] if accepted else 'unknown', accepted=bool(accepted),
                suggested_scene=best['scene'], family=FAMILIES[best['scene']] if accepted else None,
                candidates=candidates, margin=round(float(margin), 5),
                meta=dict(framework='TinyCLIP', model_id=MODEL_ID, device='cpu',
                          experimental=True, routing_hint_only=True,
                          open_set_rejection_calibrated=False,
                          score_type='uncalibrated_cosine_similarity',
                          minimum_similarity=MIN_SIMILARITY, minimum_family_margin=MIN_MARGIN,
                          view_winners=view_winners, consistent_views=bool(consistent_views),
                          scene_prompts=SCENES, parameter_statistics=runtime['statistics'],
                          trained_on_project_data=False, model_load_seconds=round(load_seconds, 4),
                          image_inference_seconds=round(time.perf_counter()-inference_started, 4),
                          elapsed_seconds=round(time.perf_counter()-started, 4),
                          limitation='Scene routing hint only; no localization/counting or calibrated confidence.'))


def _serve():
    for line in sys.stdin:
        try:
            request = json.loads(line)
            with contextlib.redirect_stdout(sys.stderr):
                if request.get('action') == 'warmup':
                    result = _warmup_local()
                else:
                    result = _classify_local(request['image'])
                Path(request['output']).write_text(json.dumps(result, ensure_ascii=False), encoding='utf-8')
            reply = dict(ok=True)
        except Exception as error:
            traceback.print_exc(file=sys.stderr)
            reply = dict(ok=False, error=str(error))
        print(json.dumps(reply, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image', nargs='?')
    parser.add_argument('--serve', action='store_true')
    parser.add_argument('--output')
    args = parser.parse_args()
    if args.serve:
        _serve()
    else:
        import cv2
        import numpy as np
        if not args.image:
            parser.error('image is required unless --serve is used')
        result = classify_scene(cv2.imdecode(np.fromfile(args.image, np.uint8), cv2.IMREAD_COLOR))
        if args.output:
            Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(result, ensure_ascii=False))
        release_scene_model()
