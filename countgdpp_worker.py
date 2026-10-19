"""CountGD++ text/self-exemplar inference, optionally serving a GPU session.

No image-specific hand-drawn prompts or saved sample coordinates are read here.
"""
import argparse,contextlib,json,os,runpy,sys,time,traceback
from pathlib import Path
from runtime_capacity import COUNTGDPP_MIN_FREE_BYTES, COUNTGDPP_MIN_FREE_MIB

ROOT=Path(__file__).resolve().parent
VENDOR=ROOT/'vendor/countgdpp'
ASSETS=ROOT/'models/countgdpp'
PROMPTS={'bamboo':'bamboo stick end', 'cotton':'cotton swab tip', 'rice':'rice grain', 'mung':'mung bean', 'coins':'coin',
 'bamboo_sharp':'bamboo skewer tip','chopsticks':'chopstick','rods':'wooden stick','insects':'insect'}
PROBE_CATEGORIES=('bamboo','cotton','rice','mung','coins','bamboo_sharp','insects')
_RUNTIME=None


def rice_fallback_prompt(requested_category, text_override, initial_count):
 """Only an explicit rice route with zero seeds may try the brown-rice alias."""
 return ('brown rice grain' if requested_category=='rice' and
         text_override is None and initial_count==0 else None)


def choose_category(candidates,independent=False):
 """Compare distinct families; two bamboo-end prompts must not veto each other."""
 families={}
 for row in candidates:
  family='bamboo' if row['category']=='bamboo_sharp' else row['category']
  if family not in families or row['evidence']>families[family]['evidence']:families[family]=row
 ordered=sorted(families.values(),key=lambda c:c['evidence'],reverse=True)
 best,runner=ordered[:2];margin=best['evidence']-runner['evidence']
 competing=runner['support']>=3 and runner['evidence']>=.30
 if independent:
  accepted=(best.get('seed_support',0)>=3 and best['evidence']>=.18 and margin>=.07
            and best['evidence']>=runner['evidence']*1.5 and not competing)
 else:
  accepted=(best['support']>=3 and best['evidence']>=.30 and margin>=.07
            and best['evidence']>=runner['evidence']*1.25 and not competing)
 return dict(category=best['category'] if accepted else None,candidates=candidates,
  margin=round(margin,5),accepted=accepted,competing_targets=competing,
  score_type='uncalibrated_mean_token_evidence',
  method='independent_phrase_confirmation' if independent else 'multi_phrase_image_probe')


def probe_category(runtime,nested,torch,device):
 """Compare supported target phrases using this image, never sample metadata."""
 from torchvision.ops import box_convert,nms
 caption=' . '.join(PROMPTS[c] for c in PROBE_CATEGORIES)+' .'
 model=runtime['model']
 with torch.inference_mode():
  output=model(nested,nested,[torch.empty((0,4),device=device)],[],[],captions=[caption])
  token=output['token'][0]
  probabilities=output['pred_logits'][0].sigmoid()
  boxes=output['pred_boxes'][0]
  areas=boxes[:,2]*boxes[:,3]
  usable=(areas>0.000005)&(areas<.45)
  xyxy=box_convert(boxes,'cxcywh','xyxy').clamp(0,1)
  candidates=[];offset=0
  for category in PROBE_CATEGORIES:
   phrase=PROMPTS[category]
   first=token.char_to_token(offset);last=token.char_to_token(offset+len(phrase)-1)
   if first is None or last is None:raise RuntimeError('Automatic target phrase could not be tokenized')
   # A whole phrase must agree; a common word such as "end" must not win alone.
   scores=probabilities[:,first:last+1].mean(-1)
   indices=torch.where(usable & (scores>.12))[0]
   keep=nms(xyxy[indices],scores[indices],.5) if len(indices) else indices
   ranked=scores[indices[keep]].sort(descending=True).values
   top=ranked[:3]
   candidates.append(dict(category=category,phrase=phrase,
    evidence=round(float(top.mean()) if len(top) else 0.,5),
    peak=round(float(ranked[0]) if len(ranked) else 0.,5),
    support=int((ranked>.20).sum())))
   offset+=len(phrase)+3
 selection=choose_category(candidates)
 if selection['accepted'] or selection['competing_targets']:return selection
 # Dense small objects can be suppressed by the combined caption. Confirm each
 # phrase independently before rejecting the image, preserving cross-class checks.
 independent=[]
 with torch.inference_mode():
  for category in PROBE_CATEGORIES:
   phrase=PROMPTS[category]
   output=model(nested,nested,[torch.empty((0,4),device=device)],[],[],captions=[phrase+' .'])
   token=output['token'][0];probabilities=output['pred_logits'][0].sigmoid()
   first=token.char_to_token(0);last=token.char_to_token(len(phrase)-1)
   values=probabilities[:,first:last+1];scores=values.mean(-1);seeds=values.max(-1).values
   boxes=output['pred_boxes'][0];areas=boxes[:,2]*boxes[:,3]
   indices=torch.where((areas>.000005)&(areas<.45)&(seeds>.15))[0]
   xyxy=box_convert(boxes,'cxcywh','xyxy').clamp(0,1)
   keep=nms(xyxy[indices],seeds[indices],.5) if len(indices) else indices
   indices=indices[keep];ranked=indices[torch.argsort(seeds[indices],descending=True)]
   top=ranked[:3]
   independent.append(dict(category=category,phrase=phrase,
    evidence=round(float(scores[top].mean()) if len(top) else 0.,5),
    peak=round(float(scores[top].max()) if len(top) else 0.,5),
    support=int((scores[indices]>.20).sum()),seed_support=int((seeds[indices]>.23).sum())))
 confirmed=choose_category(independent,independent=True)
 confirmed['initial_probe']=selection
 return confirmed

def load_runtime(torch,device):
 global _RUNTIME
 if _RUNTIME is None and device=='cuda':
  if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
  if torch.cuda.mem_get_info()[0]<COUNTGDPP_MIN_FREE_BYTES:
   raise RuntimeError(f'GPU free memory below {COUNTGDPP_MIN_FREE_MIB} MiB; wait for other tasks to release memory')
 # Keep CUDA convolution/matmul in full FP32. TF32 can alter the top-ranked
 # pseudo-exemplars and amplify a small first-pass difference during refinement.
 # This improves cross-device agreement; it does not promise bitwise equality.
 torch.backends.cuda.matmul.allow_tf32=False
 torch.backends.cudnn.allow_tf32=False
 torch.backends.cudnn.benchmark=False
 torch.backends.cudnn.deterministic=True
 if _RUNTIME is not None:
  if _RUNTIME['device']!=device:raise RuntimeError('Worker device cannot change within a session')
  return _RUNTIME,0.0
 started=time.perf_counter()
 os.environ['COUNTGDPP_FULL_CHECKPOINT']='1'
 os.environ['HF_HUB_OFFLINE']='1'
 os.environ['TRANSFORMERS_OFFLINE']='1'
 os.environ['TOKENIZERS_PARALLELISM']='false'
 sys.path.insert(0,str(VENDOR))
 from models.GroundingDINO.groundingdino_app import build_groundingdino
 from util.misc import nested_tensor_from_tensor_list
 import datasets.transforms_app as T
 config={k:v for k,v in runpy.run_path(str(VENDOR/'cfg_app.py')).items() if not k.startswith('__')}
 config.update(device=device,text_encoder_type=str(ASSETS/'bert-base-uncased'))
 args=argparse.Namespace(**config)
 model,_,_=build_groundingdino(args)
 with torch.serialization.safe_globals([argparse.Namespace]):
  checkpoint=torch.load(ASSETS/'countgd_plusplus.pth',map_location='cpu',weights_only=True)
 # The official demo does not instantiate this checkpoint module; its forward call
 # is also commented out in upstream groundingdino.py. No inference parameters skipped.
 state={k:v for k,v in checkpoint['model'].items() if not k.startswith('feature_map_encoder.')}
 model.load_state_dict(state,strict=True)
 del checkpoint,state
 model.to(device).eval()
 transform=T.Compose([T.RandomResize([800],max_size=1333),T.ToTensor(),T.Normalize([.485,.456,.406],[.229,.224,.225])])
 if device=='cuda':torch.cuda.synchronize()
 _RUNTIME=dict(model=model,transform=transform,nested_tensor=nested_tensor_from_tensor_list,device=device,
  numeric_policy=dict(name='full_fp32',cuda_matmul_tf32=False,cudnn_tf32=False,
                      cudnn_benchmark=False,cudnn_deterministic=True))
 return _RUNTIME,time.perf_counter()-started

def infer(image_path,category='bamboo',device='auto',threshold=.23,text=None,
          allow_category_recovery=False,exemplar_points=None,negative_exemplar_points=None):
 """A failed system route may reselect once; explicit text is never replaced.

 Keep the first pass in its own scope so its tensors are released before the
 retry. Both passes reuse the same model, device and threshold. A classifier
 hint is not evidence that a new target actually exists in this image.
 """
 if not isinstance(allow_category_recovery,bool):
  raise ValueError('Category recovery must be a boolean')
 if negative_exemplar_points is not None and exemplar_points is None:
  raise ValueError('使用反例前，请至少选择一个要数的正例。')
 if exemplar_points is not None:
  if text is not None:raise ValueError('User point examples cannot be combined with a text override')
  if negative_exemplar_points is not None:
   return infer_point_examples(image_path,exemplar_points,device,threshold,
                               negative_exemplar_points=negative_exemplar_points)
  return infer_point_examples(image_path,exemplar_points,device,threshold)
 result=_infer_once(image_path,category,device,threshold,text)
 meta=result['meta']
 if not (allow_category_recovery and text is None and category!='automatic' and
         meta.get('prediction_valid') is False and
         meta.get('diagnosis') in {'no_initial_exemplars','no_final_detections'}):
  return result
 retry=_infer_once(image_path,'automatic',meta.get('device',device),threshold,None)
 retry_meta=retry['meta']
 selected=retry_meta.get('detected_category') or retry_meta.get('category')
 accepted=bool(retry_meta.get('prediction_valid') and retry.get('points') and
               selected and selected!=category)
 recovery=dict(attempted=True,original_category=category,
  original_diagnosis=meta.get('diagnosis'),accepted=accepted,
  category_selection=retry_meta.get('category_selection'))
 target=retry if accepted else result
 target['meta']['category_recovery']=recovery
 target['meta']['elapsed_seconds']=round(
  meta.get('elapsed_seconds',0)+retry_meta.get('elapsed_seconds',0),3)
 # The accepted retry reports a reused runtime, but the request may have loaded it.
 target['meta']['model_load_seconds']=round(
  meta.get('model_load_seconds',0)+retry_meta.get('model_load_seconds',0),3)
 target['meta']['model_reused']=target['meta']['model_load_seconds']==0.
 if accepted:target['meta']['automatic_category']=True
 else:target['count']=None
 return target


def positive_negative_scores(output):
 """Split injected visual-token labels as in the vendored official demo.

 The first period terminates positive label 0; all following tokens belong to
 the negative label. Never allow a strong negative score to become a detection.
 """
 import torch
 probabilities=output['pred_logits'][0].sigmoid()
 token_ids=output['input_ids'][0]
 separators=torch.where(token_ids==1012)[0]
 if not len(separators):raise RuntimeError('正反例标签分隔信息缺失，无法可靠计数。')
 split=int(separators[0])
 if (split<1 or len(token_ids)>probabilities.shape[-1] or
     not bool((token_ids[1:split]==1008).any()) or
     not bool((token_ids[split+1:]==1008).any())):
  raise RuntimeError('正反例视觉标签缺失，无法可靠计数。')
 # Use only present tokens; padded logits outside input_ids carry no label.
 return (probabilities[:,:split+1].max(-1).values,
         probabilities[:,split+1:len(token_ids)].max(-1).values)


def infer_point_examples(image_path,points,device='auto',threshold=.23,
                         negative_exemplar_points=None):
 """Condition on user visual boxes directly, without automatic pseudo-examples.

 The generic caption intentionally avoids the previous scene label overriding
 the selected visual target (e.g. stick shafts competing with end faces).
 """
 started=time.perf_counter()
 from PIL import Image
 from point_prompts import validate_exemplar_points,validate_negative_exemplar_points,exemplar_boxes
 image=Image.open(image_path).convert('RGB');width,height=image.size
 points=validate_exemplar_points(points,width,height)
 boxes=exemplar_boxes(points,width,height)
 negative_boxes=[]
 if negative_exemplar_points is not None:
  negative_exemplar_points=validate_negative_exemplar_points(
   negative_exemplar_points,points,width,height)
  negative_boxes=exemplar_boxes(negative_exemplar_points,width,height)
 import torch
 import numpy as np
 from torchvision.ops import box_convert,nms
 torch.set_num_threads(4)
 if device not in {'auto','cpu','cuda'}:raise ValueError('Unknown counting device')
 if device=='auto':
  device='cuda' if torch.cuda.is_available() and torch.cuda.mem_get_info()[0]>=COUNTGDPP_MIN_FREE_BYTES else 'cpu'
 torch.manual_seed(42);np.random.seed(42)
 runtime,load_seconds=load_runtime(torch,device)
 phase=time.perf_counter()
 tensor,_=runtime['transform'](image,{'exemplars':torch.empty((0,4))})
 tensor=tensor.to(device);nested=runtime['nested_tensor'](tensor[None])
 reference=torch.tensor(boxes,dtype=torch.float32,device=device)*torch.tensor(
  [tensor.shape[-1]/width,tensor.shape[-2]/height]*2,device=device)
 negative_reference=None
 if negative_boxes:
  negative_reference=torch.tensor(negative_boxes,dtype=torch.float32,device=device)*torch.tensor(
   [tensor.shape[-1]/width,tensor.shape[-2]/height]*2,device=device)
 if device=='cuda':torch.cuda.synchronize()
 preprocess_seconds=time.perf_counter()-phase
 phase=time.perf_counter()
 with torch.inference_mode():
  if negative_reference is None:
   output=runtime['model'](nested,nested,[reference],[],[],captions=['object .'])
   scores=output['pred_logits'][0].sigmoid().max(-1).values
   selected=torch.where(scores>threshold)[0]
   negative_rejected=0
  else:
   # One native positive/negative forward, with one negative class containing
   # the current image's negative boxes. Empty text leaves label 1 visual-only.
   output=runtime['model'](nested,nested,[reference],[nested],[[negative_reference]],
                           captions=['object . .'])
   scores,negative_scores=positive_negative_scores(output)
   positive_mask=scores>threshold
   negative_rejected=int((positive_mask & (scores<=negative_scores)).sum())
   selected=torch.where(positive_mask & (scores>negative_scores))[0]
  xyxy=box_convert(output['pred_boxes'][0,selected],'cxcywh','xyxy').clamp(0,1)
  keep=nms(xyxy,scores[selected],.7) if len(selected) else selected
  raw_count=int(len(selected));selected=selected[keep]
  xyxy=xyxy[keep]*torch.tensor([width,height]*2,device=device)
  coordinates=output['pred_boxes'][0,selected,:2]*torch.tensor([width,height],device=device)
  rows=list(zip(coordinates.cpu().tolist(),xyxy.cpu().tolist(),scores[selected].cpu().tolist()))
 if device=='cuda':torch.cuda.synchronize()
 inference_seconds=time.perf_counter()-phase
 predictions=[]
 for (x,y),box,score in rows:
  if not (0<=x<width and 0<=y<height) or min(box[2]-box[0],box[3]-box[1])<=0:continue
  predictions.append(dict(x=round(x,2),y=round(y,2),r=round(max(2.,min(box[2]-box[0],box[3]-box[1])/2),2),
   box=[round(v,2) for v in box],score=round(score,4)))
 predictions.sort(key=lambda p:(round(p['y']/24),p['x']))
 result=dict(count=len(predictions) if predictions else None,points=predictions,
  width=width,height=height,exemplars=boxes,
  meta=dict(framework='CountGD++',mode='user_point_visual_exemplars',category='user_examples',
   detected_category='user_examples',automatic_category=False,manual_exemplars=True,
   reference_source='current_request_user_points',exemplar_points=points,
   pseudo_exemplar_count=0,user_exemplar_count=len(points),prompt='object',
   prompt_source='generic_caption_with_user_visual_examples',threshold=threshold,
   prediction_valid=bool(predictions),diagnosis='detections_returned' if predictions else 'no_final_detections',
   device=device,numeric_policy=runtime.get('numeric_policy'),
   elapsed_seconds=round(time.perf_counter()-started,3),model_load_seconds=round(load_seconds,3),
   model_reused=load_seconds==0.,preprocess_seconds=round(preprocess_seconds,3),
   refine_pass_seconds=round(inference_seconds,3),process_lifetime='one_request',
   raw_prediction_count=raw_count,final_nms=True,final_nms_iou=.7,
   count_kind='exemplar_candidates',count_is_total=False,review_required=True,
   score_type='uncalibrated_token_score',
   user_message='已根据你点选的目标示例重新检测整张图片；示例点本身不会直接加到数量里。仍需核对漏检和重复；若范围含相邻目标或杆身，请调整示例大小后重试。'))
 if negative_boxes:
  result['negative_exemplars']=negative_boxes
  result['meta'].update(negative_exemplar_points=negative_exemplar_points,
   user_negative_exemplar_count=len(negative_exemplar_points),
   negative_exemplar_method='native_negative_visual_tokens',
   prompt_source='generic_positive_caption_with_negative_visual_examples',
   visual_caption='object . .',positive_negative_filter='positive_above_threshold_and_negative',
   negative_rejected_prediction_count=negative_rejected,
   user_message='已根据正例寻找相似目标，并参考反例排除不想统计的对象；不会按选点数量加减。结果仍可能漏检或误删，请核对。')
 return result


def _infer_once(image_path,category='bamboo',device='auto',threshold=.23,text=None):
 started=time.perf_counter()
 requested_category=category
 import torch
 import numpy as np
 from PIL import Image
 torch.set_num_threads(4)
 if category not in {*PROMPTS,'automatic'}:raise ValueError('Unknown counting category')
 if device=='auto':
  device='cuda' if torch.cuda.is_available() and torch.cuda.mem_get_info()[0]>=COUNTGDPP_MIN_FREE_BYTES else 'cpu'
 torch.manual_seed(42);np.random.seed(42)
 runtime,load_seconds=load_runtime(torch,device)
 model=runtime['model'];transform=runtime['transform']
 def sync():
  if device=='cuda':torch.cuda.synchronize()
 pre_started=time.perf_counter()
 image=Image.open(image_path).convert('RGB');width,height=image.size
 tensor,_=transform(image,{'exemplars':torch.empty((0,4))})
 tensor=tensor.to(device);nested=runtime['nested_tensor'](tensor[None]);sync()
 preprocess_seconds=time.perf_counter()-pre_started
 selection=None;selection_seconds=0.
 if category=='automatic':
  phase_started=time.perf_counter();selection=probe_category(runtime,nested,torch,device);sync()
  selection_seconds=time.perf_counter()-phase_started
  if not selection['accepted']:
   return dict(count=None,points=[],width=width,height=height,exemplars=[],
    meta=dict(framework='CountGD++',category=None,detected_category=None,
     automatic_category=True,category_selection=selection,prediction_valid=False,
     diagnosis='ambiguous_or_unsupported_target',device=device,
     numeric_policy=runtime.get('numeric_policy'),
     elapsed_seconds=round(time.perf_counter()-started,3),
     category_selection_seconds=round(selection_seconds,3),
     model_load_seconds=round(load_seconds,3),manual_exemplars=False,
     user_message='未能确定要计数的目标。请使用同一种物体、目标清晰、背景简单的照片。'))
  category=selection['category']
 prompt=(text or PROMPTS[category]).strip().lower();caption=prompt+' .'
 def run(boxes):return model(nested,nested,[boxes],[],[],captions=[caption])
 def scores(output):
  # As in the author's pseudo-exemplar helper, restrict initial scores to target text tokens.
  tok=output['token'][0];first=tok.char_to_token(0);last=tok.char_to_token(len(prompt)-1)
  if first is None or last is None:raise RuntimeError('Target phrase could not be tokenized')
  return output['pred_logits'][0].sigmoid()[:,first:last+1].max(-1).values
 fallback=None;fallback_selected=None
 with torch.inference_mode():
  from torchvision.ops import box_convert,nms
  phase_started=time.perf_counter()
  initial=run(torch.empty((0,4),device=device))
  initial_scores=scores(initial)
  default_initial_count=int((initial_scores>threshold).sum())
  alternative=rice_fallback_prompt(requested_category,text,default_initial_count)
  if alternative:
   original_prompt=prompt
   prompt=alternative;caption=prompt+' .'
   proposal=run(torch.empty((0,4),device=device))
   proposal_scores=scores(proposal)
   proposal_indices=torch.where(proposal_scores>threshold)[0]
   proposal_xyxy=box_convert(proposal['pred_boxes'][0,proposal_indices],'cxcywh','xyxy').clamp(0,1)
   keep=nms(proposal_xyxy,proposal_scores[proposal_indices],.5) if len(proposal_indices) else proposal_indices
   distinct=proposal_indices[keep]
   fallback=dict(original_prompt=original_prompt,original_initial_count=default_initial_count,
    alternative_prompt=alternative,alternative_initial_count=int(len(proposal_indices)),
    distinct_seed_support=int(len(distinct)),minimum_distinct_seeds=3,
    threshold_unchanged=True,accepted=len(distinct)>=3)
   if fallback['accepted']:
    initial=proposal;initial_scores=proposal_scores
    fallback_selected=distinct[:3]
   else:
    prompt=original_prompt;caption=prompt+' .'
   del proposal,proposal_scores
  sync();initial_seconds=time.perf_counter()-phase_started
  if fallback_selected is not None:selected=fallback_selected
  else:
   selected=torch.where(initial_scores>threshold)[0]
   selected=selected[torch.argsort(initial_scores[selected],descending=True)[:3]]
  boxes=initial['pred_boxes'][0,selected]
  # Suppress only duplicate pseudo-prompts; final counting retains author's default no NMS.
  from torchvision.ops import box_convert,nms
  normalized_xyxy=box_convert(boxes,'cxcywh','xyxy').clamp(0,1)
  if len(selected):
   keep=nms(normalized_xyxy,initial_scores[selected],.5)
   normalized_xyxy=normalized_xyxy[keep]
  pseudo=normalized_xyxy*torch.tensor([tensor.shape[-1],tensor.shape[-2]]*2,device=device)
  original_pseudo=(normalized_xyxy*torch.tensor([width,height]*2,device=device)).cpu().tolist()
  initial_count=int((initial_scores>threshold).sum())
  del initial
  refine_seconds=0.0
  if len(pseudo):
   phase_started=time.perf_counter()
   final=run(pseudo)
   final_scores=final['pred_logits'][0].sigmoid().max(-1).values
   indices=torch.where(final_scores>threshold)[0]
   final_boxes=final['pred_boxes'][0,indices].cpu()
   final_scores=final_scores[indices].cpu()
   sync();refine_seconds=time.perf_counter()-phase_started
  else:
   final_boxes=torch.empty((0,4));final_scores=torch.empty((0,))
  points=[]
  for box,score in zip(final_boxes.tolist(),final_scores.tolist()):
   cx,cy,bw,bh=box
   if not(0<=cx<1 and 0<=cy<1):continue
   points.append(dict(x=round(cx*width,2),y=round(cy*height,2),r=round(max(2,min(bw*width,bh*height)/2),2),score=round(score,4),box=[round(max(0,(cx-bw/2)*width),2),round(max(0,(cy-bh/2)*height),2),round(min(width,(cx+bw/2)*width),2),round(min(height,(cy+bh/2)*height),2)]))
 end_review=None
 if category=='bamboo' and text is None:
  from end_face_review import mixed_shaft_evidence,suppress_uniform_end_centers
  end_review=mixed_shaft_evidence(points,original_pseudo)
  if end_review:
   # The text pass supplied compact face examples, but the word "stick" also
   # admitted long shaft boxes. Reconfirm faces with the SAME image examples.
   phase_started=time.perf_counter()
   with torch.inference_mode():
    refined=model(nested,nested,[pseudo],[],[],captions=['circular wooden end face .'])
    refined_scores=refined['pred_logits'][0].sigmoid().max(-1).values
    selected=torch.where(refined_scores>threshold)[0]
    raw_boxes=refined['pred_boxes'][0,selected].cpu().tolist()
    raw_scores=refined_scores[selected].cpu().tolist()
   confirmed=[]
   for (cx,cy,bw,bh),score in zip(raw_boxes,raw_scores):
    if not(0<=cx<1 and 0<=cy<1):continue
    confirmed.append(dict(x=round(cx*width,2),y=round(cy*height,2),
     r=round(max(2,min(bw*width,bh*height)/2),2),score=round(score,4),
     box=[round(max(0,(cx-bw/2)*width),2),round(max(0,(cy-bh/2)*height),2),
          round(min(width,(cx+bw/2)*width),2),round(min(height,(cy+bh/2)*height),2)]))
   end_review.update(refinement_prompt='circular wooden end face',
                     raw_refined_count=len(confirmed),threshold_unchanged=True)
   if confirmed:
    points,center_review=suppress_uniform_end_centers(confirmed)
    end_review['center_review']=center_review
   else:
    # Do not revert to a mixed face/shaft count when confirmation is empty.
    points=[]
   sync();refine_seconds+=time.perf_counter()-phase_started
 points.sort(key=lambda p:(round(p['y']/24),p['x']))
 result=dict(count=len(points),points=points,width=width,height=height,exemplars=original_pseudo,
  meta=dict(framework='CountGD++',mode='text_then_pseudo_exemplars',category=category,prompt=prompt,
   detected_category=category,automatic_category=selection is not None,category_selection=selection,
   category_selection_seconds=round(selection_seconds,3),prediction_valid=bool(points),
   prompt_source=('system_preset_fallback' if fallback and fallback['accepted'] else
                  ('system_preset' if text is None else 'text_override')),
   prompt_fallback=fallback,manual_exemplars=False,
   initial_count=initial_count,pseudo_exemplar_count=len(original_pseudo),threshold=threshold,
   device=device,numeric_policy=runtime.get('numeric_policy'),
   elapsed_seconds=round(time.perf_counter()-started,3),process_lifetime='one_request',
   model_load_seconds=round(load_seconds,3),model_reused=load_seconds==0.0,
   preprocess_seconds=round(preprocess_seconds,3),initial_pass_seconds=round(initial_seconds,3),
   refine_pass_seconds=round(refine_seconds,3),
   diagnosis='no_initial_exemplars' if not len(pseudo) else ('no_final_detections' if not len(points) else 'detections_returned'),
   score_type='uncalibrated_token_score',refinement='no_synthetic_exemplars_or_superresolution',
   input_short_side=800,final_nms=bool(end_review and end_review.get('center_review',{}).get('applied'))))
 if end_review:
  result['meta'].update(end_face_review=end_review,count_kind='flat_end_candidates',
   count_is_total=False,review_required=True,
   user_message='已针对端面重新检测，减少杆身混入；这是可见端面的候选数。近邻合并假设端面大小接近，遮挡、明显透视或尺寸混杂时仍需核对；可点选目标示例重新计数。')
 if fallback and fallback['accepted'] and points:
  result['meta'].update(count_kind='grain_candidates',count_is_total=False,review_required=True,
   user_message='这是可见谷粒和断粒的候选数量，包含图边截断部分；密集相贴处可能漏检或重复，请结合标记核对，不能据此确定完整谷粒总数。')
 if category=='insects':
  result['meta'].update(count_kind='insect_candidates',count_is_total=False,review_required=True,
   user_message='这是可见昆虫的候选数。小个体、相互遮挡和裁边处可能漏检，翅膀也可能被重复计数，请逐个核对标记。')
 return result

def warmup_runtime(device):
 """Load weights and exercise both forward paths, without a user image/count."""
 import torch
 from PIL import Image
 started=time.perf_counter()
 torch.set_num_threads(4)
 runtime,load_seconds=load_runtime(torch,device)
 tensor,_=runtime['transform'](Image.new('RGB',(800,800),(128,128,128)),
                                {'exemplars':torch.empty((0,4))})
 tensor=tensor.to(device);nested=runtime['nested_tensor'](tensor[None])
 with torch.inference_mode():
  for boxes in (torch.empty((0,4),device=device),
                torch.tensor([[300.,300.,350.,350.]],device=device)):
   output=runtime['model'](nested,nested,[boxes],[],[],captions=['rice grain .'])
   del output
 if device=='cuda':torch.cuda.synchronize()
 meta=dict(device=device,numeric_policy=runtime.get('numeric_policy'),model_load_seconds=round(load_seconds,3),
           elapsed_seconds=round(time.perf_counter()-started,3),warmup_forward_passes=2)
 if device=='cuda':
  meta.update(allocated_mib=round(torch.cuda.memory_allocated()/1024**2,1),
              reserved_mib=round(torch.cuda.memory_reserved()/1024**2,1))
 return dict(ready=True,meta=meta)


def serve(device):
 # All library progress goes to stderr; stdout is reserved for one JSON ack per job.
 for line in sys.stdin:
  try:
   request=json.loads(line)
   with contextlib.redirect_stdout(sys.stderr):
    if request.get('action')=='warmup':result=warmup_runtime(device)
    else:
     result=infer(request['image'],request.get('category','bamboo'),device,
                  allow_category_recovery=request.get('allow_category_recovery',False),
                  exemplar_points=request.get('exemplar_points'),
                  negative_exemplar_points=request.get('negative_exemplar_points'))
     result['meta']['process_lifetime']='gpu_session'
    Path(request['output']).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
   reply=dict(ok=True)
  except Exception as error:
   traceback.print_exc(file=sys.stderr);reply=dict(ok=False,error=str(error))
  print(json.dumps(reply,ensure_ascii=False),flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('image',nargs='?');p.add_argument('--category',choices=[*PROMPTS,'automatic'],default='bamboo')
 p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto');p.add_argument('--threshold',type=float,default=.23)
 p.add_argument('--text');p.add_argument('--output');p.add_argument('--serve',action='store_true')
 p.add_argument('--exemplar-points-file')
 p.add_argument('--negative-exemplar-points-file')
 p.add_argument('--recover-category',action='store_true');a=p.parse_args()
 if a.serve:return serve(a.device)
 if not a.image or not a.output:p.error('image and --output are required unless --serve is used')
 if not .05<=a.threshold<=.95:p.error('Threshold must be in [0.05,0.95]')
 examples=json.loads(Path(a.exemplar_points_file).read_text(encoding='utf-8')) if a.exemplar_points_file else None
 if a.exemplar_points_file and examples is None:p.error('Example points file must contain a nonempty array')
 negatives=json.loads(Path(a.negative_exemplar_points_file).read_text(encoding='utf-8')) if a.negative_exemplar_points_file else None
 if a.negative_exemplar_points_file and negatives is None:p.error('Negative example points file must contain a nonempty array')
 result=infer(a.image,a.category,a.device,a.threshold,a.text,a.recover_category,examples,negatives)
 out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps(dict(count=result['count'],meta=result['meta']),ensure_ascii=False),flush=True)
if __name__=='__main__':main()
