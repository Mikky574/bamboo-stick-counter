"""Compare two fixed classifiers on sealed legacy and fresh source cohorts."""
import argparse
import hashlib
import json
from pathlib import Path
import time
from unittest.mock import patch

import torch
import scene_classifier as tiny
from supervised_scene import load_runtime,classify_local
from compare_scene_models import dump,summarize,contact_sheet

ROOT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--baseline',type=Path,default=ROOT/'models/scene_classifier/convnext_small_v1/scene_convnext_small.pt')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    data=json.loads(args.manifest.read_text('utf-8'));test=[r for r in data['samples'] if r['split'] in {'test','heldout_test'}]
    runtimes={}
    for name,path in [('v1',args.baseline),('v2',args.checkpoint)]:
        profile=dict(checkpoint_path=path,model_id=f'project/ConvNeXt-Small-scene-{name}',sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        runtimes[name]=load_runtime(profile)
    # Explicitly pin the old text-prompt baseline, regardless of app profile.
    with patch.object(tiny,'active_profile',return_value=None):tiny._RUNTIME=None;tiny._load_runtime()
    rows=[]
    for sample in test:
        path=Path(sample.get('image') or sample.get('path'))
        if not path.is_absolute():path=ROOT/path
        row=dict(id=sample.get('id'),image=str(path),expected=sample['label'],
                 cohort=sample.get('test_cohort','legacy_regression'),group=sample['source_group'])
        for name in ('tinyclip','v1','v2'):
            start=time.perf_counter()
            result=tiny._classify_local(path) if name=='tinyclip' else classify_local(path,runtimes[name])
            predicted=result['suggested_scene'];predicted='laid_rods' if predicted=='chopsticks' else predicted
            row[name]=dict(predicted=predicted,accepted=result['accepted'],scene=result['scene'],
                           scores=result['candidates'],seconds=round(time.perf_counter()-start,4))
        rows.append(row)
        print(json.dumps(dict(image=path.name,cohort=row['cohort'],expected=row['expected'],v1=row['v1']['predicted'],v2=row['v2']['predicted'])),flush=True)
    report={cohort:{name:summarize(part,name) for name in ('tinyclip','v1','v2')}
            for cohort in sorted({r['cohort'] for r in rows})
            if (part:=[r for r in rows if r['cohort']==cohort])}
    report['all_heldout']={name:summarize(rows,name) for name in ('tinyclip','v1','v2')}
    dump(args.output/'comparison_predictions.json',rows);dump(args.output/'comparison_report.json',report)
    # Reuse the established image renderer; the displayed comparison is v1/v2.
    display=[{**r,'tinyclip':r['v1'],'convnext':r['v2']} for r in rows]
    contact_sheet(display,args.output/'v1_v2_contact_sheet.jpg')
    selected=[r for r in display if r['cohort']=='fresh_v2']
    if selected:contact_sheet(selected,args.output/'fresh_v2_contact_sheet.jpg')
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
