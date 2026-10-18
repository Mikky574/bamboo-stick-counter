"""Shared preprocessing and inference for the supervised scene experiment.

Model libraries are imported only inside the isolated worker/training process.
An explicit active profile is required before this model affects the web app.
"""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
CLASSES=('flat_end','sharp','laid_rods','cotton','rice','mung','coins','other')
ACTIVE_PROFILE=ROOT/'models/scene_classifier/active_profile.json'
MEAN=(.485,.456,.406)
STD=(.229,.224,.225)


def active_profile():
    if not ACTIVE_PROFILE.is_file():return None
    profile=json.loads(ACTIVE_PROFILE.read_text(encoding='utf-8'))
    if profile.get('architecture')!='convnext_small':return None
    path=(ROOT/profile['checkpoint']).resolve()
    if not path.is_relative_to(ROOT) or not path.is_file():return None
    return {**profile,'checkpoint_path':path}


def preprocess(image,augment=False):
    import random
    import numpy as np
    from PIL import Image,ImageOps
    from torchvision.transforms import functional as F
    image=image.convert('RGB')
    image.thumbnail((1200,1200))
    pixels=np.asarray(image)
    border=np.concatenate((pixels[0],pixels[-1],pixels[:,0],pixels[:,-1]))
    fill=tuple(int(v) for v in np.median(border,axis=0))
    if augment:
        if random.random()<.5:image=ImageOps.mirror(image)
        angle=random.choice((0,90,180,270))+random.uniform(-10,10)
        image=image.rotate(angle,resample=Image.Resampling.BILINEAR,expand=True,fillcolor=fill)
        image=F.adjust_brightness(image,random.uniform(.85,1.15))
        image=F.adjust_contrast(image,random.uniform(.85,1.15))
        image=F.adjust_saturation(image,random.uniform(.85,1.15))
    image=ImageOps.pad(image,(224,224),method=Image.Resampling.BILINEAR,color=fill)
    return F.normalize(F.to_tensor(image),MEAN,STD)


def load_runtime(profile):
    import hashlib
    import torch
    from torchvision.models import convnext_small
    path=profile['checkpoint_path']
    if profile.get('sha256'):
        with path.open('rb') as stream:
            if hashlib.file_digest(stream,'sha256').hexdigest()!=profile['sha256']:
                raise RuntimeError('Supervised scene checkpoint checksum mismatch')
    payload=torch.load(path,map_location='cpu',weights_only=True)
    classes=tuple(payload['classes'])
    if classes!=CLASSES:raise RuntimeError('Supervised scene classes differ from routing contract')
    model=convnext_small(weights=None,num_classes=len(classes))
    model.load_state_dict(payload['model'],strict=True)
    model.eval().to('cpu')
    return dict(kind='supervised',model=model,classes=classes,
                calibration=payload['calibration'],profile=profile)


def classify_local(image_path,runtime):
    import time
    import torch
    from PIL import Image
    started=time.perf_counter()
    image=Image.open(image_path).convert('RGB')
    inputs=preprocess(image)[None]
    calibration=runtime['calibration']
    with torch.inference_mode():
        logits=runtime['model'](inputs)[0]
        probabilities=(logits/calibration.get('temperature',1.)).softmax(-1)
    order=probabilities.argsort(descending=True).tolist()
    candidates=[dict(scene=runtime['classes'][i],score=round(float(probabilities[i]),5)) for i in order]
    best,runner=candidates[:2]
    score=float(probabilities[order[0]])
    margin=score-float(probabilities[order[1]])
    accepted=(best['scene']!='other' and score>=calibration['min_score']
              and margin>=calibration['min_margin'])
    return dict(scene=best['scene'] if accepted else 'unknown',accepted=accepted,
        suggested_scene=best['scene'],family=best['scene'] if accepted else None,
        candidates=candidates,margin=round(margin,5),meta=dict(
            framework='ConvNeXt-Small',model_id=runtime['profile']['model_id'],device='cpu',
            routing_hint_only=True,trained_on_project_data=True,
            score_type='validation_temperature_scaled_softmax_not_guaranteed_probability',
            minimum_similarity=calibration['min_score'],minimum_family_margin=calibration['min_margin'],
            open_set_rejection_calibrated=False,experimental=True,
            checkpoint_sha256=runtime['profile'].get('sha256'),
            elapsed_seconds=round(time.perf_counter()-started,4),
            limitation='Supervised scene routing only; small source-group split, no count guarantee.'))
