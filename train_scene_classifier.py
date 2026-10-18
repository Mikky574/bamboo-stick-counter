"""GPU transfer learning with source-group splits and no held-out selection."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import time

os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('MKL_NUM_THREADS','4')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader,WeightedRandomSampler
from torchvision.models import convnext_small
from torchvision.ops import StochasticDepth

from supervised_scene import CLASSES,preprocess

ROOT=Path(__file__).resolve().parent


def dump(path,value):
    path.write_bytes(json.dumps(value,ensure_ascii=False,indent=2).encode('utf-8'))


def data_path(row):
    path=Path(row.get('image') or row.get('path') or row['file'])
    return path if path.is_absolute() else ROOT/path


class SceneDataset(Dataset):
    def __init__(self,rows,augment=False):
        self.rows=rows;self.augment=augment;self.images=[]
        for row in rows:
            image=Image.open(data_path(row)).convert('RGB');image.thumbnail((1200,1200))
            self.images.append(image.copy())
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):
        row=self.rows[index]
        return preprocess(self.images[index].copy(),self.augment),CLASSES.index(row['label']),index


def metrics(logits,labels):
    predictions=logits.argmax(1)
    per_class={}
    for i,name in enumerate(CLASSES):
        selected=labels==i;n=int(selected.sum())
        per_class[name]=dict(n=n,correct=int((predictions[selected]==i).sum()),
            accuracy=float((predictions[selected]==i).float().mean()) if n else None)
    represented=[v['accuracy'] for v in per_class.values() if v['n']]
    return dict(n=len(labels),accuracy=float((predictions==labels).float().mean()),
        macro_recall=float(np.mean(represented)),per_class=per_class,
        represented_classes=len(represented))


def evaluate(model,loader,device):
    model.eval();scores=[];truth=[]
    with torch.inference_mode():
        for images,labels,_ in loader:
            with torch.autocast(device_type='cuda',dtype=torch.float16):
                logits=model(images.to(device))
            scores.append(logits.float().cpu());truth.append(labels)
    return torch.cat(scores),torch.cat(truth)


def calibration(logits,labels):
    temperatures=(.5,.75,1.,1.5,2.,3.,4.)
    temperature=min(temperatures,key=lambda t:float(nn.functional.cross_entropy(logits/t,labels)))
    probabilities=(logits/temperature).softmax(1);top,indices=probabilities.topk(2,dim=1)
    foreground=indices[:,0]!=CLASSES.index('other');correct=indices[:,0]==labels
    choices=[]
    # Predeclared conservative floors; a tiny validation set must not enable
    # accepting every softmax winner just because its observed accuracy is high.
    for threshold in (.55,.65,.75,.85,.9,.95):
        for margin in (.1,.2,.3):
            accepted=foreground & (top[:,0]>=threshold) & (top[:,0]-top[:,1]>=margin)
            n=int(accepted.sum());precision=float(correct[accepted].float().mean()) if n else 0.
            if n and precision>=.9:choices.append((n,precision,-threshold,-margin,threshold,margin))
    if choices:
        best=max(choices);threshold,margin=best[-2:]
    else:threshold,margin=1.01,1.01
    return dict(temperature=temperature,min_score=threshold,min_margin=margin,
        selection='validation_only_maximum_coverage_with_observed_precision_at_least_0.9',
        predeclared_minimum_score=.55,predeclared_minimum_margin=.1,
        calibration_samples=len(labels),not_a_probability_guarantee=True)


def routed_metrics(logits,labels,settings):
    probabilities=(logits/settings['temperature']).softmax(1)
    top,indices=probabilities.topk(2,1)
    accepted=((indices[:,0]!=CLASSES.index('other')) & (top[:,0]>=settings['min_score'])
              & (top[:,0]-top[:,1]>=settings['min_margin']))
    correct=indices[:,0]==labels
    return dict(n=len(labels),accepted=int(accepted.sum()),rejected=int((~accepted).sum()),
        accepted_correct=int((accepted & correct).sum()),
        accepted_precision=float(correct[accepted].float().mean()) if accepted.any() else None,
        known_class_coverage=float(accepted[labels!=7].float().mean()) if (labels!=7).any() else None)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--head-epochs',type=int,default=18);p.add_argument('--finetune-epochs',type=int,default=12)
    p.add_argument('--batch-size',type=int,default=8);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--memory-gib',type=float,default=3.);p.add_argument('--step-delay',type=float,default=.02)
    args=p.parse_args()
    if args.head_epochs<1 or args.finetune_epochs<1 or args.batch_size<1:
        p.error('Both stages require at least one epoch and a positive batch size')
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    if not torch.cuda.is_available():raise RuntimeError('This training run requires the authorized GPU')
    free,total=torch.cuda.mem_get_info()
    if free<(args.memory_gib+1)*1024**3:raise RuntimeError('Insufficient free VRAM for configured cap plus reserve')
    torch.cuda.set_per_process_memory_fraction(min(.95,args.memory_gib*1024**3/total),0)
    torch.backends.cudnn.benchmark=False
    device=torch.device('cuda:0')
    loaded=json.loads(args.manifest.read_text(encoding='utf-8'))
    rows=loaded['samples'] if isinstance(loaded,dict) else loaded
    rows=[row for row in rows if row.get('split') in {'train','val','test','heldout_test'}]
    for row in rows:
        if row['label'] not in CLASSES:raise ValueError('Unrecognized scene label')
        if not data_path(row).is_file():raise FileNotFoundError(data_path(row))
    train=[r for r in rows if r['split']=='train'];val=[r for r in rows if r['split']=='val']
    test=[r for r in rows if r['split'] in {'test','heldout_test'}]
    counts=Counter(r['label'] for r in train)
    if set(CLASSES)-set(counts):raise ValueError('Training classes missing: '+str(set(CLASSES)-set(counts)))
    if not val or not test:raise ValueError('Source-separated validation and held-out test are required')
    groups={split:{r.get('group') or r.get('source_group') for r in part} for split,part in [('train',train),('val',val),('test',test)]}
    if any(None in values for values in groups.values()):raise ValueError('Every image requires a source group')
    if groups['train']&groups['val'] or groups['train']&groups['test'] or groups['val']&groups['test']:
        raise ValueError('Source group leakage across splits')
    hashes={}
    for row in rows:
        digest=hashlib.sha256(data_path(row).read_bytes()).hexdigest()
        split='test' if row['split']=='heldout_test' else row['split']
        if digest in hashes and hashes[digest]!=split:raise ValueError('Exact duplicate leaks across splits')
        hashes[digest]=split
    dump(out/'input_manifest.json',rows)
    config=dict(vars(args));config={k:str(v) if isinstance(v,Path) else v for k,v in config.items()}
    config.update(architecture='convnext_small',classes=CLASSES,training_counts=dict(counts),
        val_counts=dict(Counter(r['label'] for r in val)),test_counts=dict(Counter(r['label'] for r in test)),
        manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        pretrained='IMAGENET1K_V1',full_image_square_pad=True,device_name=torch.cuda.get_device_name(),
        vram_free_at_start_gib=free/1024**3)
    dump(out/'config.json',config)
    dataset=SceneDataset(train,True)
    sampler=WeightedRandomSampler([1/counts[r['label']] for r in train],len(train),replacement=True,
        generator=torch.Generator().manual_seed(args.seed))
    loader=DataLoader(dataset,batch_size=args.batch_size,sampler=sampler,num_workers=0)
    val_loader=DataLoader(SceneDataset(val),batch_size=args.batch_size,num_workers=0)
    test_loader=DataLoader(SceneDataset(test),batch_size=args.batch_size,num_workers=0)
    model=convnext_small(weights=None)
    pretrained=ROOT/'models/scene_classifier/convnext_small_v1/convnext_small_imagenet1k.pth'
    state=torch.load(pretrained,map_location='cpu',weights_only=True)
    model.load_state_dict(state,strict=True);del state
    model.classifier[2]=nn.Linear(768,len(CLASSES));model.to(device)
    for param in model.parameters():param.requires_grad_(False)
    for param in model.classifier.parameters():param.requires_grad_(True)
    history=[];best_key=(-1.,-1.);best_state=None;best_epoch=None;started=time.perf_counter()
    scaler=torch.amp.GradScaler('cuda')
    for phase,epochs in [('head',args.head_epochs),('last_stage',args.finetune_epochs)]:
        if phase=='last_stage':
            model.load_state_dict(best_state)
            for param in model.features[-1].parameters():param.requires_grad_(True)
            for module in model.features[-1].modules():
                if isinstance(module,StochasticDepth):module.p=min(module.p,.1)
            parameters=[dict(params=model.features[-1].parameters(),lr=1e-5),
                        dict(params=model.classifier.parameters(),lr=3e-4)]
        else:parameters=[dict(params=model.classifier.parameters(),lr=3e-3)]
        optimizer=torch.optim.AdamW(parameters,weight_decay=1e-3)
        schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=max(1,epochs),eta_min=1e-6)
        for epoch in range(1,epochs+1):
            epoch_started=time.perf_counter();model.eval();model.classifier.train()
            if phase=='last_stage':model.features[-1].train()
            total_loss=0.;seen=0
            for images,labels,_ in loader:
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type='cuda',dtype=torch.float16):
                    logits=model(images.to(device));loss=nn.functional.cross_entropy(logits,labels.to(device),label_smoothing=.05)
                scaler.scale(loss).backward();scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad),1.)
                scaler.step(optimizer);scaler.update()
                total_loss+=float(loss)*len(labels);seen+=len(labels)
                if args.step_delay:time.sleep(args.step_delay)
            schedule.step()
            logits,labels=evaluate(model,val_loader,device);score=metrics(logits,labels)
            key=(score['macro_recall'],-float(nn.functional.cross_entropy(logits,labels)))
            if key>best_key:
                best_key=key;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                best_epoch=dict(phase=phase,epoch=epoch)
                torch.save(dict(model=best_state,classes=CLASSES,best_epoch=best_epoch),out/'best_training.pt')
            record=dict(phase=phase,epoch=epoch,train_loss=total_loss/max(1,seen),
                validation=score,epoch_seconds=round(time.perf_counter()-epoch_started,3),
                gpu_max_allocated_gib=round(torch.cuda.max_memory_allocated()/1024**3,3),
                gpu_max_reserved_gib=round(torch.cuda.max_memory_reserved()/1024**3,3))
            history.append(record);dump(out/'history.json',history)
            print(json.dumps({k:v for k,v in record.items() if k!='validation'}|{'val_macro':score['macro_recall'],'val_accuracy':score['accuracy']}),flush=True)
    # Only after model and threshold selection on validation, open the held-out test.
    model.load_state_dict(best_state);val_logits,val_labels=evaluate(model,val_loader,device)
    settings=calibration(val_logits,val_labels)
    dump(out/'validation_logits.json',dict(logits=val_logits.tolist(),labels=val_labels.tolist(),
        images=[str(data_path(r)) for r in val],routing=routed_metrics(val_logits,val_labels,settings)))
    payload=dict(model=best_state,classes=CLASSES,calibration=settings,architecture='convnext_small',
        best_epoch=best_epoch,training_config=config)
    torch.save(payload,out/'scene_convnext_small.pt')
    test_logits,test_labels=evaluate(model,test_loader,device)
    report=dict(best_epoch=best_epoch,validation=metrics(val_logits,val_labels),
        calibration=settings,validation_routing=routed_metrics(val_logits,val_labels,settings),
        test=metrics(test_logits,test_labels),
        test_routing=routed_metrics(test_logits,test_labels,settings),
        elapsed_seconds=round(time.perf_counter()-started,3),
        total_parameters=sum(p.numel() for p in model.parameters()),
        trainable_last_phase=sum(p.numel() for p in model.parameters() if p.requires_grad),
        gpu_max_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
        gpu_max_reserved_gib=torch.cuda.max_memory_reserved()/1024**3,
        caveat='Small source-group split; test not used for checkpoint or threshold selection; no general accuracy guarantee.')
    dump(out/'report.json',report)
    probabilities=(test_logits/settings['temperature']).softmax(1)
    predictions=[]
    for row,truth,values in zip(test,test_labels.tolist(),probabilities.tolist()):
        order=np.argsort(values)[::-1];best=CLASSES[int(order[0])];margin=values[order[0]]-values[order[1]]
        accepted=best!='other' and values[order[0]]>=settings['min_score'] and margin>=settings['min_margin']
        predictions.append(dict(id=row.get('id'),image=str(data_path(row)),expected=CLASSES[truth],
            predicted=best,scene=best if accepted else 'unknown',accepted=bool(accepted),
            scores={label:round(v,6) for label,v in zip(CLASSES,values)}))
    dump(out/'test_predictions.json',predictions)
    print(json.dumps(dict(complete=True,report=report)),flush=True)


if __name__=='__main__':main()
