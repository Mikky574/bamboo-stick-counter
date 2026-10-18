"""Offline held-out classification audit; labels never enter either predictor."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFont,ImageOps

import scene_classifier as tiny
from supervised_scene import CLASSES,load_runtime,classify_local

ROOT=Path(__file__).resolve().parent
NAMES=dict(flat_end='平端',sharp='尖头',laid_rods='平铺杆/筷子',cotton='棉头',rice='生米',
           mung='绿豆',coins='硬币',other='其他',unknown='不确定')


def dump(path,value):
    path.write_bytes(json.dumps(value,ensure_ascii=False,indent=2).encode('utf-8'))


def summarize(rows,key):
    correct=sum(row[key]['predicted']==row['expected'] for row in rows)
    accepted=[row for row in rows if row[key]['accepted']]
    per_class={label:dict(n=sum(r['expected']==label for r in rows),
        correct=sum(r['expected']==label and r[key]['predicted']==label for r in rows),
        accepted=sum(r['expected']==label and r[key]['accepted'] for r in rows),
        accepted_correct=sum(r['expected']==label and r[key]['accepted'] and r[key]['predicted']==label for r in rows))
        for label in CLASSES}
    return dict(n=len(rows),top1_correct=correct,top1_accuracy=correct/len(rows),
        accepted=len(accepted),accepted_correct=sum(r[key]['predicted']==r['expected'] for r in accepted),
        accepted_precision=sum(r[key]['predicted']==r['expected'] for r in accepted)/len(accepted) if accepted else None,
        known_class_coverage=sum(r[key]['accepted'] for r in rows if r['expected']!='other')/max(1,sum(r['expected']!='other' for r in rows)),
        per_class=per_class)


def contact_sheet(rows,path):
    font_path=Path('C:/Windows/Fonts/msyh.ttc')
    font=ImageFont.truetype(str(font_path),17)
    columns=4;width,height=340,330
    canvas=Image.new('RGB',(columns*width,((len(rows)+columns-1)//columns)*height),'#f2f3f4')
    draw=ImageDraw.Draw(canvas)
    for i,row in enumerate(rows):
        x=(i%columns)*width;y=(i//columns)*height
        image=Image.open(row['image']).convert('RGB')
        image=ImageOps.contain(image,(width-16,230))
        canvas.paste(image,(x+(width-image.width)//2,y+(230-image.height)//2))
        draw.text((x+8,y+235),f"{i+1}. 实际：{NAMES[row['expected']]}",font=font,fill='#222222')
        for offset,key,title in ((260,'tinyclip','旧模型'),(285,'convnext','新模型')):
            p=row[key];text=NAMES[p['predicted']] if p['accepted'] else '不确定'
            color='#18713b' if p['accepted'] and p['predicted']==row['expected'] else '#9a5800' if not p['accepted'] else '#b92b2b'
            draw.text((x+8,y+offset),f"{title}：{text}",font=font,fill=color)
    canvas.save(path,quality=90)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    loaded=json.loads(args.manifest.read_text(encoding='utf-8'))
    samples=loaded['samples'] if isinstance(loaded,dict) else loaded
    test=[r for r in samples if r.get('split') in {'test','heldout_test'}]
    profile=dict(checkpoint_path=args.checkpoint,model_id='project/ConvNeXt-Small-scene-v1',
                 sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest())
    runtime=load_runtime(profile)
    rows=[]
    for row in test:
        path=Path(row.get('image') or row.get('path'))
        if not path.is_absolute():path=ROOT/path
        entry=dict(id=row.get('id'),image=str(path),expected=row['label'],group=row.get('group') or row.get('source_group'))
        # Both functions receive the original image path only, with no labels.
        for key,fn in [('tinyclip',tiny._classify_local),('convnext',lambda image:classify_local(image,runtime))]:
            start=time.perf_counter();result=fn(path)
            predicted=result['suggested_scene']
            if predicted=='chopsticks':predicted='laid_rods'
            entry[key]=dict(predicted=predicted,accepted=result['accepted'],scene=result['scene'],
                scores=result['candidates'],seconds=round(time.perf_counter()-start,4))
        rows.append(entry)
        print(json.dumps(dict(image=path.name,expected=entry['expected'],
                              tinyclip=entry['tinyclip']['predicted'],convnext=entry['convnext']['predicted'])),flush=True)
    dump(args.output/'comparison_predictions.json',rows)
    report=dict(tinyclip=summarize(rows,'tinyclip'),convnext=summarize(rows,'convnext'),
        checkpoint_sha256=profile['sha256'],
        caveat='Small, source-group separated held-out photos. No counting-accuracy measurement; no checkpoint or threshold selection on these results.')
    dump(args.output/'comparison_report.json',report)
    contact_sheet(rows,args.output/'comparison_contact_sheet.jpg')
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
