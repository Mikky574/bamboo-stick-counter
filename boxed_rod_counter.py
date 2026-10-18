"""Restricted pixel-only terminal silhouette counting in regular open rod boxes.

The scene classifier must have established laid rods. Packaging can merge the
whole foreground; this routine counts separately visible terminal silhouettes
across consecutive transverse sections. It never doubles a number of pairs and
never adds upper and lower fragments of the same longitudinal track.
"""
import time
import cv2
import numpy as np

PARAMETERS = dict(maximum_processing_side=1200, minimum_instances=4,
    minimum_terminal_support_fraction=.02, saturation_thresholds=(35,45,55),
    minimum_visible_value=45, dark_thresholds=(85,110),
    maximum_terminal_width_height_fraction=.035, minimum_frame_length_fraction=.38,
    maximum_frame_edges=32, maximum_frame_pairs=128,
    maximum_processing_seconds=1.0)

def _budget_expired(deadline):
    return deadline is not None and time.perf_counter() >= deadline

def _runs(mask):
    delta=np.diff(np.r_[False,mask,False].astype(np.int8))
    return [(int(a),int(b)) for a,b in zip(np.where(delta==1)[0],np.where(delta==-1)[0])]

def _frames(gray):
    h,w=gray.shape
    edges=cv2.Canny(gray,60,160)
    lines=cv2.HoughLinesP(edges,1,np.pi/720,threshold=max(30,int(h*.13)),
        minLineLength=h*.38,maxLineGap=h*.025)
    result=[]
    for x1,y1,x2,y2 in ([] if lines is None else lines[:,0]):
        if abs(y2-y1)<h*.38 or abs(x2-x1)>abs(y2-y1)*.2:continue
        slope=(x2-x1)/(y2-y1)
        result.append((float(slope),float(x1-slope*y1),int(min(y1,y2)),int(max(y1,y2))))
    return result

def _frame_pairs(frames,gray,deadline=None):
    h,w=gray.shape
    # Dense straight-line texture produces many Hough edges. Pairing all of
    # them and scanning every image row grows quadratically; those scenes are
    # outside this restricted box method. Reject the full scene rather than
    # truncate the edges, which could silently retain only some real rods.
    if len(frames)>PARAMETERS['maximum_frame_edges']:return []
    pairs=[]
    for left in frames:
        for right in frames:
            if _budget_expired(deadline):return []
            if abs(left[0]-right[0])>.17:continue
            if right[1]-left[1]<w*.15:continue
            crossbars=[]
            for y in range(0,h,2):
                if y%32==0 and _budget_expired(deadline):return []
                a=int(round(left[0]*y+left[1]));b=int(round(right[0]*y+right[1]))
                if not 0<=a<b<w:continue
                if np.mean(gray[max(0,y-1):min(h,y+2),a:b]<110)>.80:crossbars.append(y)
            if crossbars and crossbars[-1]-crossbars[0]>=h*.4:
                pairs.append((left,right,crossbars[0],crossbars[-1]))
                if len(pairs)>PARAMETERS['maximum_frame_pairs']:return []
    return pairs

def _row_groups(mask,y,h,w,frame_pairs,gray_row):
    all_runs=_runs(mask)
    runs=[(a,b) for a,b in all_runs if 2<=b-a<=h*.035]
    if len(runs)<4:return []
    groups=[]
    for reference in sorted(set(b-a for a,b in runs)):
        selected=[(a,b) for a,b in runs if reference*.65<=b-a<=reference*1.55]
        chunks=[]
        for interval in selected:
            if not chunks or interval[0]-chunks[-1][-1][1]>reference*8:chunks.append([])
            chunks[-1].append(interval)
        for chunk in chunks:
            if not 4<=len(chunk)<=40:continue
            centers=np.array([(a+b-1)/2 for a,b in chunk]); widths=np.array([b-a for a,b in chunk])
            med=float(np.median(widths)); span=chunk[-1][1]-chunk[0][0]
            if span<w*.15 or np.std(widths)/med>.40:continue
            # Count the whole foreground cross section. A merged wide pair may
            # not be dropped by the similar-width filter while the other eight
            # rods are accepted as a subset.
            if any(b-a>=med*.6 and a<chunk[-1][1] and b>chunk[0][0]
                   and (a,b) not in chunk for a,b in all_runs):continue
            gaps=[(chunk[j][1],chunk[j+1][0]) for j in range(len(chunk)-1)]
            gap_pixels=np.concatenate([gray_row[a:b] for a,b in gaps if b>a])
            if len(gap_pixels) and np.median(gap_pixels)>140:
                # On a light liner, black rods are foreground too. A colored-only
                # mask may not silently omit a dark stroke in an apparent gap.
                if any(d-c>=max(2,med*.6) for a,b in gaps for c,d in _runs(gray_row[a:b]<110)):
                    continue
            # Both sides must have long, near-parallel package edges, outside
            # the visible terminal strokes. Text on an adjacent lid has no
            # such enclosing frame and is not an instance candidate.
            enclosed=False
            for left,right,top,bottom in frame_pairs:
                if not top<y<bottom:continue
                ld=chunk[0][0]-(left[0]*y+left[1])
                rd=(right[0]*y+right[1])-chunk[-1][1]
                if med*.8<ld<span*.35 and med*.8<rd<span*.35:
                    enclosed=True;break
            if not enclosed:continue
            if any(len(g['centers'])==len(centers) and np.max(np.abs(g['centers']-centers))<1 for g in groups):continue
            groups.append(dict(y=y,centers=centers,widths=widths,median_width=med))
    return groups

def _detect_boxed_rods_upright(image,deadline=None):
    """Return (points, metadata) for supported terminal tracks, else None.

    Only near-frontal, regular boxed layouts with separately visible slender
    tips are supported. This is a development-scene method, not a model score.
    """
    if not isinstance(image,np.ndarray) or image.dtype!=np.uint8 or image.ndim!=3 or image.shape[2]!=3:
        raise ValueError('Expected uint8 HxWx3 BGR pixels')
    started=time.perf_counter();oh,ow=image.shape[:2]
    if min(oh,ow)<80:return None
    scale=min(1.,1200/max(oh,ow))
    if scale<1:image=cv2.resize(image,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA)
    h,w=image.shape[:2];hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    frames=_frames(gray)
    if len(frames)<2:return None
    if _budget_expired(deadline):return None
    frame_pairs=_frame_pairs(frames,gray,deadline)
    if not frame_pairs:return None
    candidates=[]
    # Independent color thresholds must agree. A dark-foreground variant also
    # includes black rods on a light liner; it fails safely on a solid black bed.
    for saturation in PARAMETERS['saturation_thresholds']:
        colored=(hsv[:,:,1]>saturation)&(hsv[:,:,2]>45)
        for dark in (None,85,110):
            mask=colored if dark is None else colored|(hsv[:,:,2]<dark)
            row_groups=[]
            for y in range(int(h*.12),int(h*.94),2):
                if _budget_expired(deadline):return None
                row_groups.extend(_row_groups(mask[y],y,h,w,frame_pairs,gray[y]))
            if not row_groups:continue
            for number in sorted({len(g['centers']) for g in row_groups},reverse=True):
                if _budget_expired(deadline):return None
                available=[g for g in row_groups if len(g['centers'])==number]
                # Link only adjacent rows with the same left-to-right tracks.
                tracks=[]
                for g in available:
                    if _budget_expired(deadline):return None
                    compatible=[t for t in tracks if g['y']-t[-1]['y']<=4 and
                        g['y']>t[-1]['y'] and np.max(np.abs(g['centers']-t[-1]['centers']))<=max(2,t[-1]['median_width']*.4)]
                    if compatible:max(compatible,key=len).append(g)
                    else:tracks.append([g])
                for track in tracks:
                    span=track[-1]['y']-track[0]['y']
                    if span<max(12,h*.02) or len(track)<7:continue
                    yy=np.array([g['y'] for g in track]);xx=np.stack([g['centers'] for g in track]);widths=np.stack([g['widths'] for g in track])
                    coefs=[np.polyfit(yy,xx[:,j],1) for j in range(number)]
                    medwidth=float(np.median(widths))
                    if max(np.max(np.abs(xx[:,j]-np.polyval(c,yy))) for j,c in enumerate(coefs))>max(1.6,medwidth*.2):continue
                    if max(abs(c[0]) for c in coefs)>.25:continue
                    if max(c[0] for c in coefs)-min(c[0] for c in coefs)>.22:continue
                    # Terminal contour evidence is mandatory: most individual
                    # strokes must narrow towards one shared end. Constant-width
                    # paint/woodgrain inside a wide slab cannot establish rods.
                    quarter=max(2,len(track)//4)
                    start_width=np.median(widths[:quarter],axis=0)
                    end_width=np.median(widths[-quarter:],axis=0)
                    taper_forward=np.mean(start_width>end_width*1.12)
                    taper_reverse=np.mean(end_width>start_width*1.12)
                    if max(taper_forward,taper_reverse)<.65:continue
                    candidates.append(dict(number=number,track=track,coefs=coefs,
                        saturation=saturation,dark=dark,span=span,median_width=medwidth))
    if not candidates:return None
    if _budget_expired(deadline):return None
    maximum=max(c['number'] for c in candidates)
    bests=[c for c in candidates if c['number']==maximum]
    def compatible(a,b):
        if a['number']!=b['number']:return False
        ya=max(a['track'][0]['y'],b['track'][0]['y'])
        yb=min(a['track'][-1]['y'],b['track'][-1]['y'])
        # Agreement is spatial, not just equal cardinality. Two different text
        # patches or two separated row bands cannot vote for one another.
        if yb-ya<max(10,min(a['span'],b['span'])*.5):return False
        y=(ya+yb)/2
        xa=np.array([np.polyval(coef,y) for coef in a['coefs']])
        xb=np.array([np.polyval(coef,y) for coef in b['coefs']])
        if np.max(np.abs(xa-xb))>max(1.25,min(a['median_width'],b['median_width'])*.3):return False
        return max(a['median_width'],b['median_width'])/max(1,min(a['median_width'],b['median_width']))<1.6
    agreed=[]
    for candidate in bests:
        if _budget_expired(deadline):return None
        support=[other for other in bests if compatible(candidate,other)]
        if len({c['saturation'] for c in support})>=2:agreed.append((candidate,support))
    if not agreed:return None
    best,support=max(agreed,key=lambda item:(item[0]['number'],item[0]['span']))
    maximum=best['number']
    track=best['track'];y0,y1=track[0]['y'],track[-1]['y']
    points=[]
    for j,coef in enumerate(best['coefs']):
        segment=[[round(float(np.polyval(coef,y))/scale,2),round(y/scale,2)] for y in (y0,y1)]
        points.append(dict(x=round(np.mean([p[0] for p in segment]),2),y=round((y0+y1)/(2*scale),2),
            r=3.,segment=segment,terminal_width=round(float(np.median([g['widths'][j] for g in track]))/scale,2)))
    meta=dict(framework='BoxedRodTerminalTracks',device='cpu',category='laid_rods',
        detected_category='laid_rods',prediction_valid=True,diagnosis='detections_returned',
        count_unit='individual_whole_rod_not_pair',mode='framed_exposed_terminal_tracks',
        count_kind='visible_rod_candidates',review_required=True,count_is_total=False,user_message='按纸带外清晰可见的单根筷子末端计数；同一根上下两段只计一次，请核对遮挡处。',
        manual_exemplars=False,elapsed_seconds=round(time.perf_counter()-started,4),
        segment_definition='Only the visible terminal section of one rod; packaging-hidden shaft is not drawn.',
        evidence=dict(consecutive_sections=len(track),terminal_y_range=[round(y0/scale,2),round(y1/scale,2)],
            independently_agreeing_saturation_thresholds=sorted({c['saturation'] for c in support}),
            enclosing_long_frame_edges=len(frames),median_terminal_width=round(best['median_width']/scale,2),
            directly_observed_individual_strokes=maximum,pair_multiplier_used=False),
        parameters=PARAMETERS.copy(),
        limitation='Restricted near-frontal open-box layouts with visible separated terminal strokes. Hidden, crossing, cropped, or unresolved tips are rejected. Development-scene results are not an independent accuracy estimate.')
    return points,meta


def detect_boxed_rods(image):
    """Count supported visible terminal tracks in quarter-turned box photos."""
    if not isinstance(image,np.ndarray) or image.dtype!=np.uint8 or image.ndim!=3 or image.shape[2]!=3:
        raise ValueError('Expected uint8 HxWx3 BGR pixels')
    started=time.perf_counter()
    deadline=started+PARAMETERS['maximum_processing_seconds']
    height,width=image.shape[:2]
    for turns in range(4):
        if _budget_expired(deadline):return None
        turned=image if turns==0 else np.ascontiguousarray(np.rot90(image,turns))
        result=_detect_boxed_rods_upright(turned,deadline)
        if result is None:continue
        points,meta=result
        def original(point):
            x,y=point
            if turns==0:values=(x,y)
            elif turns==1:values=(width-1-y,x)
            elif turns==2:values=(width-1-x,height-1-y)
            else:values=(y,height-1-x)
            return [round(float(v),2) for v in values]
        for point in points:
            point['x'],point['y']=original([point['x'],point['y']])
            point['segment']=[original(p) for p in point['segment']]
        meta.update(input_quarter_turn_for_processing=turns,elapsed_seconds=round(time.perf_counter()-started,4))
        return points,meta
    return None
