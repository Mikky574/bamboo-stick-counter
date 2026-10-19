"""Offline CPU diagnostics. Isolated probes never start the web/GPU service."""
import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import uuid


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def cpu_environment():
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='-1', PYTORCH_NVML_BASED_CUDA_CHECK='0',
               OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', PYTHONIOENCODING='utf-8',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    return env


def inference_status(result):
    meta = result.get('meta', {})
    if meta.get('device') != 'cpu':
        raise RuntimeError('Expected actual CPU inference metadata, got another/missing device.')
    if meta.get('diagnosis') == 'no_initial_exemplars':
        return 'executed_no_initial_exemplars'
    return 'executed_with_candidates' if meta.get('prediction_valid') else 'executed_no_valid_count'


def worker(args, root, out):
    phase = 'starting_python'
    details = dict(python=sys.executable, version=sys.version, prefix=sys.prefix,
                   base_prefix=sys.base_prefix, cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))

    def progress(name):
        nonlocal phase
        phase = name
        write_json(out / 'progress.json', dict(stage=name, pid=os.getpid()))
        print('[stage] ' + name, flush=True)

    result_file = out / (args.worker + '_result.json')
    try:
        sys.path.insert(0, str(root))
        progress('import_torch')
        import torch
        torch.set_num_threads(4)
        details.update(torch_version=torch.__version__, torch_path=torch.__file__,
                       torch_cuda_build=torch.version.cuda, threads=torch.get_num_threads())
        cfg = Path(sys.prefix) / 'pyvenv.cfg'
        details['pyvenv_cfg'] = cfg.read_text(encoding='utf-8') if cfg.is_file() else None
        details['torch_outside_environment'] = not Path(torch.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
        if args.worker == 'environment':
            progress('cpu_tensor_and_torchvision_ops')
            import torchvision
            from torchvision.ops import nms
            x = torch.ones((4, 4), device='cpu')
            assert float((x @ x).sum()) == 64.
            boxes = torch.tensor([[0., 0., 10., 10.], [0., 0., 10., 10.], [30., 30., 40., 40.]])
            assert len(nms(boxes, torch.tensor([.9, .8, .7]), .5)) == 2
            import numpy
            import cv2
            import PIL
            import transformers
            import timm
            details.update(torchvision_version=torchvision.__version__, torchvision_path=torchvision.__file__,
                           numpy_version=numpy.__version__, opencv_version=cv2.__version__,
                           pillow_version=PIL.__version__, transformers_version=transformers.__version__,
                           timm_version=timm.__version__, cpu_tensor_device=str(x.device), cpu_nms_passed=True)
            payload = dict(ok=True, status='cpu_environment_passed', details=details)
        else:
            source = Path(args.image).resolve() if args.image else None
            if source:
                progress('decode_input_image')
                from PIL import Image
                with Image.open(source) as image:
                    details['image'] = dict(path=str(source), size=list(image.size), format=image.format,
                                            exif_orientation=image.getexif().get(274))
                    image.convert('RGB').load()
            if args.check_scene and source:
                progress('cpu_scene_classification')
                from scene_classifier import _classify_local
                details['scene_classification'] = _classify_local(source)
            progress('load_countgd_weights_on_cpu')
            import countgdpp_worker as counter
            started = time.perf_counter()
            runtime, load_seconds = counter.load_runtime(torch, 'cpu')
            devices = sorted({str(p.device) for p in runtime['model'].parameters()})
            if devices != ['cpu']:
                raise RuntimeError('Model parameters are not all on CPU: ' + repr(devices))
            details.update(checkpoint_load_seconds=round(load_seconds, 3), parameter_devices=devices,
                           numeric_policy=runtime.get('numeric_policy'))
            if source:
                progress('cpu_image_forward_pass')
                prediction = counter.infer(source, category=args.category, device='cpu', text=args.text)
                write_json(out / 'prediction.json', prediction)
                status = inference_status(prediction)
                details['prediction_summary'] = dict(count=prediction.get('count'), meta=prediction.get('meta'))
            else:
                progress('cpu_synthetic_warmup_two_forward_passes')
                details['warmup'] = counter.warmup_runtime('cpu')
                status = 'cpu_model_warmup_passed'
            details['model_probe_seconds'] = round(time.perf_counter()-started, 3)
            payload = dict(ok=True, status=status, details=details)
        progress('completed')
        write_json(result_file, payload)
        return 0
    except Exception as error:
        traceback.print_exc()
        write_json(result_file, dict(ok=False, status='failed', failed_stage=phase,
            error_type=type(error).__name__, error=str(error), details=details))
        return 1


def stop_owned_probe(process):
    if process.poll() is None:
        if os.name == 'nt':
            taskkill = Path(os.environ.get('SystemRoot', r'C:\Windows'))/'System32/taskkill.exe'
            stopped = subprocess.run([str(taskkill), '/PID', str(process.pid), '/T', '/F'],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10,
                creationflags=subprocess.CREATE_NO_WINDOW)
            if stopped.returncode and process.poll() is None:
                raise RuntimeError('Failed to terminate the owned diagnostic process tree.')
        else:
            process.kill()
        process.wait(timeout=10)


def run_logged(command, root, log_path, timeout, progress_path=None):
    """Own exactly this child. Capture launch failure, timeout, and full stderr."""
    started = time.perf_counter()
    process = None
    result = None
    with log_path.open('w', encoding='utf-8') as log:
        try:
            process = subprocess.Popen(command, cwd=root, env=cpu_environment(),
                stdout=log, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            last_notice = started
            while True:
                try:
                    code = process.wait(timeout=min(2., max(.01, timeout-(time.perf_counter()-started))))
                    result = dict(returncode=code, timed_out=False, seconds=round(time.perf_counter()-started, 3),
                                  log=str(log_path), launched=True, pid=process.pid)
                    break
                except subprocess.TimeoutExpired:
                    elapsed = time.perf_counter()-started
                    if elapsed >= timeout:
                        result = dict(returncode=None, timed_out=True, seconds=round(elapsed, 3),
                                      log=str(log_path), launched=True, pid=process.pid)
                        break
                    if time.perf_counter()-last_notice >= 10:
                        phase = 'working'
                        try:
                            phase = json.loads(progress_path.read_text(encoding='utf-8'))['stage'] if progress_path else phase
                        except (OSError, ValueError, KeyError):
                            pass
                        print(f'  running: {phase}, {elapsed:.0f}s; log={log_path}', flush=True)
                        last_notice = time.perf_counter()
        except OSError as error:
            result = dict(launched=process is not None, error=str(error), returncode=None,
                          timed_out=False, log=str(log_path))
        finally:
            if process is not None and process.poll() is None:
                try:
                    stop_owned_probe(process)
                except Exception as error:
                    if result is not None:result['cleanup_error'] = str(error)
                    print('CLEANUP ERROR: ' + str(error), flush=True)
            if process is not None and result is not None:result['returncode'] = process.poll()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--python', type=Path, help='Exact model interpreter; defaults to current interpreter.')
    parser.add_argument('--image', help='Optional original image; implies --run-model.')
    parser.add_argument('--category', default='mung', choices=['mung','rice','cotton','bamboo','bamboo_sharp','rods','chopsticks','coins','insects','automatic'])
    parser.add_argument('--text', help='Optional research text; omitted preserves the original preset.')
    parser.add_argument('--run-model', action='store_true')
    parser.add_argument('--check-scene', action='store_true', help='Also test CPU scene classifier on the image.')
    parser.add_argument('--timeout', type=float, default=600)
    parser.add_argument('--out-dir', type=Path)
    parser.add_argument('--worker', choices=['environment','model'], help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or not 0 < args.timeout <= 3600:
        parser.error('--timeout must be finite and in (0, 3600].')
    root = args.project_root.resolve()
    if args.worker:
        if not args.out_dir:parser.error('Internal worker requires --out-dir.')
        out = args.out_dir.resolve()
    else:
        base = args.out_dir.resolve() if args.out_dir else root/'outputs/cpu_diagnostics'
        out = base/(datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + str(os.getpid()) + '_' + uuid.uuid4().hex[:8])
    out.mkdir(parents=True, exist_ok=bool(args.worker))
    if args.worker:
        return worker(args, root, out)
    python = args.python.resolve() if args.python else Path(sys.executable).resolve()
    print('Project: ' + str(root), flush=True)
    print('Model Python: ' + str(python), flush=True)
    print('Report folder: ' + str(out), flush=True)
    print('GPU is hidden for probes. No server, package install, or model download is started.', flush=True)
    script = str(Path(__file__).resolve())
    common = [str(python), '-u', script, '--project-root', str(root), '--out-dir', str(out)]
    report = dict(project=str(root), interpreter=str(python), gpu_hidden=True,
                  service_started=False, steps={}, full_smart_route_executed=False)
    print('[1/3] Python, imports, actual CPU tensor and NMS operators', flush=True)
    env_step = run_logged(common+['--worker','environment'], root, out/'environment.log', min(args.timeout, 120), out/'progress.json')
    report['steps']['environment'] = env_step
    for name in ['environment']:
        result_path = out/(name+'_result.json')
        if result_path.is_file():report['steps'][name]['result'] = json.loads(result_path.read_text(encoding='utf-8'))
    environment_result = env_step.get('result', {})
    if environment_result.get('ok'):
        details = environment_result['details']
        print('  Torch: ' + str(details['torch_version']) + ' at ' + details['torch_path'], flush=True)
        print('  CPU tensor / torchvision NMS: passed', flush=True)
        if details.get('torch_outside_environment'):
            print('  NOTE: Torch is inherited from outside this virtual environment; install dependencies independently after migration.', flush=True)
    print('[2/3] Required model hashes (offline verification only)', flush=True)
    report['steps']['assets'] = run_logged([sys.executable, str(root/'tools/fetch_models.py'), '--verify-only'],
        root, out/'assets.log', min(args.timeout, 120))
    environment_ok = (env_step.get('result', {}).get('ok', False) and env_step.get('returncode') == 0
                      and not env_step.get('timed_out') and not env_step.get('cleanup_error'))
    assets_ok = (report['steps']['assets'].get('returncode') == 0
                 and not report['steps']['assets'].get('timed_out'))
    report['status'] = 'environment_failed' if not environment_ok else ('assets_failed' if not assets_ok else 'environment_and_assets_passed_model_not_requested')
    if (args.run_model or args.image) and environment_ok and assets_ok:
        print('[3/3] Actual CPU model loading and forward pass', flush=True)
        command = common+['--worker','model','--category',args.category]
        if args.image:command += ['--image',str(Path(args.image).resolve())]
        if args.text is not None:command += ['--text',args.text]
        if args.check_scene:command += ['--check-scene']
        write_json(out/'progress.json', dict(stage='launching_model_probe'))
        step = run_logged(command, root, out/'model.log', args.timeout, out/'progress.json')
        report['steps']['model'] = step
        result_path = out/'model_result.json'
        if result_path.is_file():step['result'] = json.loads(result_path.read_text(encoding='utf-8'))
        report['status'] = ('model_cleanup_failed' if step.get('cleanup_error') else
            'model_timeout' if step.get('timed_out') else
            step.get('result', {}).get('status', 'model_launch_failed_or_crashed') if step.get('returncode') == 0 else
            'model_failed')
    else:
        print('[3/3] Model not run; add --run-model or --image after prerequisite checks pass.', flush=True)
    write_json(out/'diagnostics.json', report)
    print('STATUS: ' + report['status'], flush=True)
    print('REPORT: ' + str(out/'diagnostics.json'), flush=True)
    for stage in ['environment', 'model']:
        result = report['steps'].get(stage, {}).get('result', {})
        if result.get('failed_stage'):print('FAILED STAGE: ' + result['failed_stage'], flush=True)
        if result.get('error'):print('ERROR: ' + result['error'], flush=True)
    if not environment_ok or not assets_ok:return 1
    model = report['steps'].get('model')
    return 0 if model is None or (model.get('result', {}).get('ok') and model.get('returncode') == 0
                                 and not model.get('timed_out') and not model.get('cleanup_error')) else 1


if __name__ == '__main__':
    raise SystemExit(main())
