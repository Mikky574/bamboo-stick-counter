"""Local GeCo2 pretrained inference, using the author's center-response head.

Code/weights: https://github.com/jerpelhan/GECO2 (AAAI 2026).
The default returns center peaks without box NMS or SAM mask refinement.
No training on the user's image is performed.
"""
import os
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
_lock = threading.Lock()
_model = None
_cache = None


def load_model():
    global _model
    if _model is not None:
        return _model
    import torch
    sys.path.insert(0, str(ROOT / 'vendor' / 'geco2'))
    os.environ['GECO2_FULL_CHECKPOINT'] = '1'
    os.environ['GECO2_POINTS_ONLY'] = '1'
    from models.counter_infer import build_model
    from utils.arg_parser import get_argparser
    checkpoint = ROOT / 'models' / 'CNTQG_multitrain_ca44.pth'
    if not checkpoint.is_file():
        raise RuntimeError('缺少 GeCo2 权重，请先运行模型下载脚本。')
    args = get_argparser().parse_args([])
    args.zero_shot = True  # matches the author's exemplar inference demo
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)['model']
    state = {k.removeprefix('module.'): v for k, v in state.items()}
    model = build_model(args)
    # The checkpoint includes auxiliary training heads unused during inference.
    state = {k: v for k, v in state.items()
             if not k.startswith(('class_embed_aux.', 'bbox_embed_aux.'))}
    model.load_state_dict(state, strict=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.return_masks = False
    model.to(device).eval()
    _model = (model, device)
    return _model


def validate_exemplars(boxes, width, height):
    if not isinstance(boxes, list) or not 1 <= len(boxes) <= 8:
        raise ValueError('请框选 1–8 个端面作为示例。')
    result = []
    for box in boxes:
        if len(box) != 4:
            raise ValueError('示例框格式错误。')
        x1, y1, x2, y2 = map(float, box)
        if not all(np.isfinite([x1, y1, x2, y2])):
            raise ValueError('示例框坐标无效。')
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError('示例框必须位于图片内。')
        if min(x2-x1, y2-y1) < 4:
            raise ValueError('示例框过小，请框住完整端面。')
        result.append([x1, y1, x2, y2])
    return result


def detect_model(image, exemplars, threshold=0.33, nms_iou=None):
    global _cache
    import hashlib
    import torch
    from torchvision import transforms as T
    from torchvision.ops import nms

    height, width = image.shape[:2]
    boxes = validate_exemplars(exemplars, width, height)
    threshold = float(threshold)
    if not .125 <= threshold <= .95:
        raise ValueError('模型筛选阈值必须在 0.125–0.95 之间。')
    if nms_iou is not None and not .1 <= float(nms_iou) <= .95:
        raise ValueError('NMS IoU 必须在 0.1–0.95 之间。')
    key = (hashlib.sha256(image.tobytes()).hexdigest(), tuple(map(tuple, boxes)))
    started = time.perf_counter()
    with _lock:
        model, device = load_model()
        cached = _cache is not None and _cache[0] == key
        if not cached:
            from utils.data import resize_and_pad
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float().to(device) / 255
            tensor = T.Normalize([.485, .456, .406], [.229, .224, .225])(tensor)
            box_tensor = torch.tensor(boxes, dtype=torch.float32, device=device)
            tensor, box_tensor, scale = resize_and_pad(tensor, box_tensor, size=1024.)
            with torch.inference_mode():
                outputs, refs, response, _, _ = model(tensor[None], box_tensor[None])
            scores = outputs[0]['box_v'].flatten().detach().cpu()
            pred_boxes = outputs[0]['pred_boxes'].reshape(-1, 4).detach().cpu()
            # Author's ref_points are heatmap row/column locations (y, x).
            centers = refs[0].T.flip(-1).detach().cpu().float()
            centers *= 1024. / response.shape[-1] / scale
            pred_boxes *= 1024. / scale
            _cache = (key, scores, pred_boxes, centers)
        _, scores, pred_boxes, centers = _cache
        peak = float(scores.max()) if len(scores) else 0.
        selected = torch.where(
            (scores > max(0., peak * threshold)) &
            (centers[:, 0] >= 0) & (centers[:, 0] < width) &
            (centers[:, 1] >= 0) & (centers[:, 1] < height)
        )[0]
        before_nms = len(selected)
        if nms_iou is not None and before_nms:
            selected = selected[nms(pred_boxes[selected], scores[selected], float(nms_iou))]
        points = []
        for index in selected.tolist():
            x, y = centers[index].tolist()
            b = pred_boxes[index].tolist()
            radius = max(4., min(b[2]-b[0], b[3]-b[1])/2)
            points.append(dict(x=round(x, 2), y=round(y, 2), r=round(radius, 2),
                               score=round(float(scores[index])/peak, 4),
                               box=[round(v, 2) for v in b]))
        points.sort(key=lambda p: (round(p['y']/24), p['x']))
        return points, dict(framework='GeCo2', mode='center_peaks' if nms_iou is None else 'box_nms',
                            candidate_count=len(scores), before_nms=before_nms,
                            nms_removed=before_nms-len(points), threshold=threshold,
                            nms_iou=nms_iou, cached=cached, device=str(device),
                            elapsed_seconds=round(time.perf_counter()-started, 3),
                            refinement='disabled', score_type='relative_center_response')
