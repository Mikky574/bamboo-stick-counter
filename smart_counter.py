"""Image-only counting: conservative geometry gate, then automatic text probing.

No filenames, sample catalogue, saved points or user category are consulted.
The supported semantic candidates are bounded; this is not an arbitrary-object
counter and ambiguous images deliberately return no numeric count.
"""
import os
import shutil
import subprocess
import time

import cv2
import numpy as np

from runtime_capacity import COUNTGDPP_MIN_FREE_MIB as MIN_GPU_FREE_MIB


def inference_device():
    """Choose an interactive neural runtime; never silently start slow CPU work.

    CPU geometry remains available independently of this check. Explicit
    research API/CLI calls can still request the CPU neural counter.
    """
    from auto_counter import readiness
    if readiness().get('model_resident'):
        return 'cuda'
    executable = shutil.which('nvidia-smi')
    if executable:
        try:
            result = subprocess.run(
                [executable, '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=4,
                text=True, creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            # The worker uses CUDA device 0; do not choose a different GPU's memory.
            free_memory = int(result.stdout.strip().splitlines()[0])
            if result.returncode == 0 and free_memory >= MIN_GPU_FREE_MIB:
                return 'cuda'
        except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
            pass
    return None


def no_count(image, diagnosis, message):
    return dict(count=None, points=[], width=image.shape[1], height=image.shape[0],
                meta=dict(framework='ImageOnlyCounter', automatic_category=True,
                          prediction_valid=False, diagnosis=diagnosis, user_message=message))


def scene_hint(image):
    """A small CPU classifier chooses a candidate route, never the count."""
    from scene_classifier import classify_scene,readiness
    if not readiness()['installed']:
        return None
    # A missing optional classifier permits category probing; a failed or busy
    # installed worker must surface its error instead of launching more work.
    return classify_scene(image)


def _has_rod_geometry_prior(hint):
    """Allow one strong geometric route for rod-like scenes or a bounded conflict."""
    if not hint:
        return False
    if hint.get('accepted'):
        return hint.get('scene') in {'flat_end', 'sharp', 'laid_rods', 'chopsticks'}
    candidates = hint.get('candidates') or []
    if len(candidates) < 2:
        return False
    leading = {row.get('scene') for row in candidates[:2]}
    # A disagreement between stick views is not evidence of an unrelated
    # object. Keep classification, then let complete pixel tracks resolve the
    # counting view; the strict detector may still abstain.
    return len(leading) == 2 and leading <= {'flat_end', 'sharp', 'laid_rods', 'chopsticks'}


def _carded_rod_detection(image):
    from carded_rod_counter import detect_carded_rods
    return detect_carded_rods(image)


def _strict_rod_geometry(image):
    """Only complete terminal-track evidence can resolve a rod-like scene.

    Legacy silhouettes and seam counts were not designed to distinguish a
    bundle's end faces from whole rods. They must not bypass scene rejection.
    """
    geometry = _carded_rod_detection(image)
    if geometry is None:
        return None
    points, meta = geometry
    if (meta.get('framework') != 'CardedRodTerminalTracks' or
            meta.get('routing_evidence') != 'complete_individual_terminal_tracks' or
            not meta.get('prediction_valid') or not points or
            any(len(point.get('segment', [])) != 2 for point in points)):
        return None
    return points, meta


def review_sharp_predictions(result):
    """Remove clear shaft/truncated-box candidates, without inventing apexes."""
    meta = result['meta']
    boxes = [p['box'] for p in result['points'] if 'box' in p]
    aspects = [max(b[2]-b[0], b[3]-b[1])/max(1., min(b[2]-b[0], b[3]-b[1])) for b in boxes]
    if aspects and float(np.median(aspects)) < 1.5:
        meta.update(category='bamboo', detected_category='bamboo_flat',
                    subtype_review='round_boxes_not_pointed_regions')
        return
    before = len(result['points']);retained = []
    width,height = result['width'],result['height']
    rejected = {'truncated_region':0,'shaft_instead_of_tip':0}
    for point in result['points']:
        box = point.get('box')
        if box:
            bw,bh = box[2]-box[0],box[3]-box[1]
            if min(box[:2]) <= 1 or box[2] >= width-1 or box[3] >= height-1:
                rejected['truncated_region'] += 1
                continue
            if max(bw,bh)/max(1.,min(bw,bh)) > 8:
                rejected['shaft_instead_of_tip'] += 1
                continue
        retained.append(point)
    result.update(points=retained,count=len(retained))
    meta.update(raw_prediction_count=before,sharp_rejected=rejected,
                prediction_valid=bool(retained),count_kind='sharp_candidates',
                localization='tip_region_box_center_not_exact_apex',
                user_message='这是可见尖头区域的候选数，仍可能漏检或重复；标记位于尖头区域，尚不能当作精确顶点或整捆准确数量。')
    if not retained:
        result['count'] = None
        meta['diagnosis'] = 'no_verified_sharp_regions'


def detect_smart(image):
    started = time.perf_counter()
    from particle_gate import detect_separated
    preview = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)
    if float(preview.reshape(-1, 3).std(axis=0).max()) < 2.:
        result = no_count(image, 'no_visible_targets', '没有找到清晰的计数目标，请上传有实物的照片。')
    else:
        separated = detect_separated(image)
        if separated is not None:
            points, meta = separated
            result = dict(count=len(points), points=points, width=image.shape[1],
                          height=image.shape[0], meta=meta)
            result['meta']['user_message'] = '已按清晰分开的目标计数，请结合编号核对。遮挡的物体无法从照片中数出。'
        else:
            from auto_counter import detect_auto
            hint = scene_hint(image)
            routes = dict(sharp='bamboo_sharp',flat_end='bamboo',laid_rods='rods',
                          chopsticks='chopsticks',cotton='cotton',rice='rice',mung='mung',coins='coins')
            category = routes.get(hint.get('scene')) if hint and hint.get('accepted') else None
            if _has_rod_geometry_prior(hint):
                rods = _strict_rod_geometry(image)
                if rods is not None:
                    points, rod_meta = rods
                    result = dict(count=len(points), points=points,
                                  width=image.shape[1], height=image.shape[0], meta=rod_meta)
                    result['meta'].update(scene_classification=hint,
                        routing='scene_then_strict_terminal_tracks',
                        automatic_category=True,
                        total_elapsed_seconds=round(time.perf_counter()-started, 3))
                    return result
            if hint and not hint.get('accepted') and hint.get('suggested_scene') in {'laid_rods','chopsticks'}:
                result=no_count(image,'uncertain_whole_rod_view',
                    '暂时无法确定这是平铺整根还是成捆端头，不能可靠给出根数。请将整根完整拍入，并减少相互遮挡。')
                result['meta'].update(scene_classification=hint,routing='ambiguous_rod_view_abstained',
                    total_elapsed_seconds=round(time.perf_counter()-started,3))
                return result
            if category in {'rods','chopsticks'}:
                from laid_rod_counter import detect_laid_rods
                rods = detect_laid_rods(image)
                if rods is None:
                    result = no_count(image,'unresolved_individual_rods',
                        '已识别为平铺长条，但贴合、交叉或颜色对比不足，暂时无法分清每一根。请留出间隙并完整拍入画面。')
                else:
                    points,rod_meta = rods
                    result = dict(count=len(points),points=points,width=image.shape[1],
                                  height=image.shape[0],meta=rod_meta)
                    result['meta'].setdefault('user_message', '按单根计数，每条轴线对应一根；请核对相贴处和木纹，避免把一双当成一根。')
                result['meta'].update(scene_classification=hint,routing='scene_then_rod_sections',
                    automatic_category=True,total_elapsed_seconds=round(time.perf_counter()-started,3))
                return result
            device = inference_device()
            if device != 'cuda':
                result = no_count(image, 'gpu_resources_unavailable',
                    '当前GPU不可用或可用显存不足，深度计数暂不能运行；请等待其他GPU任务释放资源后再试。')
                result['meta'].update(scene_classification=hint,
                    routing='neural_runtime_unavailable', retryable=True,
                    required_free_gpu_mib=MIN_GPU_FREE_MIB,
                    total_elapsed_seconds=round(time.perf_counter()-started, 3))
                return result
            result = detect_auto(image, category or 'automatic', device, keep_warm=device == 'cuda')
            meta = result['meta']
            meta['scene_classification'] = hint
            meta['routing'] = 'cpu_scene_hint_then_counting' if category else 'automatic_category_after_geometry_gate'
            if not meta.get('prediction_valid', bool(result['points'])):
                result['count'] = None
                meta.setdefault('user_message', '本次没有找到可靠目标，请换一张目标清晰、遮挡更少的照片。')
            elif meta.get('category') in {'mung', 'coins'}:
                boxes = [p['box'] for p in result['points'] if 'box' in p]
                ratios = [max(b[2]-b[0], b[3]-b[1])/max(1., min(b[2]-b[0], b[3]-b[1])) for b in boxes]
                # Round seeds/coins from the supported overhead scenes must not
                # silently accept elongated touching grains as "beans".
                if ratios and float(np.median(ratios)) > 2.:
                    meta.update(prediction_valid=False, diagnosis='target_shape_mismatch',
                                user_message='目标轮廓不明确或互相接触，暂时无法可靠计数。请分开摆放并正对拍摄。')
                    meta['rejected_prediction_count'] = result['count']
                    result['count'] = None
                    result['points'] = []
            elif meta.get('category') == 'bamboo_sharp':
                review_sharp_predictions(result)
            if meta.get('category') == 'mung' and meta.get('prediction_valid'):
                meta.update(count_kind='visible_candidates',
                    user_message='仅统计照片中检出的可见绿豆，仍可能漏检或重复；容器内或其他豆子下被遮挡的数量无法从这张照片得出。')
            if meta.get('category') in {'rods','chopsticks'} and meta.get('prediction_valid'):
                meta.update(count_kind='rod_candidates',
                    user_message='这是整根长条的模型候选数；相贴成对或交叉摆放时仍可能合并、重复，尚不能当作准确根数。')
            meta.setdefault('user_message', '已自动识别重复目标；密集区域可能漏计或重复，请结合编号核对。')
    result['meta'].update(automatic_category=True,
                          total_elapsed_seconds=round(time.perf_counter()-started, 3))
    return result
