"""Restricted, pixel-only counting of exposed rods on a colored display card.

Separate longitudinal highlight/shadow ridges, taper and terminal closure are
required. The printed sleeve and the two sides of it are never counted twice.
This supplies visible candidates, not an inferred inventory behind packaging.
"""
import time
import cv2
import numpy as np
from scipy.signal import find_peaks

PARAMETERS = dict(maximum_processing_seconds=.95, maximum_processing_side=900,
    saturation_thresholds=(40, 60), shadow_prominences=(25, 35),
    minimum_track_length_fraction=.15, minimum_track_coverage=.58,
    minimum_taper_ratio=1.12, minimum_tapered_fraction=.70)


def _cards(image):
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    cards=[]
    for saturation in PARAMETERS['saturation_thresholds']:
        mask=np.uint8((hsv[:,:,1]>saturation)&(hsv[:,:,2]>50))*255
        mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
        contours=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]
        for contour in sorted(contours,key=cv2.contourArea,reverse=True)[:3]:
            area=cv2.contourArea(contour)
            if not .12*image.shape[0]*image.shape[1]<area<.90*image.shape[0]*image.shape[1]:continue
            polygon=cv2.approxPolyDP(contour,cv2.arcLength(contour,True)*.02,True)
            if len(polygon)!=4 or not cv2.isContourConvex(polygon):continue
            q=polygon[:,0].astype(np.float32)
            if abs(cv2.contourArea(q))/max(area,1)<.90:continue
            # The contour order supplies orientation without a fixed image angle.
            lengths=np.linalg.norm(q-np.roll(q,-1,axis=0),axis=1)
            if min(lengths)/max(lengths)>.76:continue
            if lengths[0]>lengths[1]:q=np.roll(q,-1,axis=0)
            if cv2.contourArea(q,oriented=True)<0:q=q[[1,0,3,2]]
            width=int(round((np.linalg.norm(q[0]-q[1])+np.linalg.norm(q[2]-q[3]))/2))
            height=int(round((np.linalg.norm(q[0]-q[3])+np.linalg.norm(q[1]-q[2]))/2))
            if not 100<=width<=height or height<200:continue
            # A reflected contour is fine; both ends are checked below.
            if any(np.max(np.linalg.norm(q-old[0],axis=1))<5 for old in cards):continue
            cards.append((q,width,height,saturation))
    return cards


def _sections(luminance,prominence,deadline):
    h,w=luminance.shape
    groups=[]
    for y in range(max(5,int(h*.05)),h-5,2):
        if time.perf_counter()>=deadline:return []
        row=luminance[y]
        shadows,shadow_properties=find_peaks(-row,prominence=prominence,distance=max(4,int(w*.015)))
        highlights,_=find_peaks(row,prominence=8,distance=max(3,int(w*.012)))
        strokes=[]
        unresolved=False
        for index,x in enumerate(shadows):
            if row[x]>175:continue
            left=highlights[(highlights<x-w*.016)&(highlights>=x-w*.08)]
            left=left[row[left]>135]
            if not len(left):
                # Strong unmatched shadows may be an unresolved/cropped rod;
                # do not silently count only its better separated neighbours.
                # Fainter boundary/engraving texture is not a rod claim.
                if row[x]<130 and shadow_properties['prominences'][index]>=75:
                    unresolved=True
                    break
                continue
            edge=int(left[-1])
            strokes.append((float(x),float(edge)))
        if unresolved or not 2<=len(strokes)<=40:continue
        xx=np.array([p[0] for p in strokes]);ee=np.array([p[1] for p in strokes])
        widths=xx-ee
        if np.std(widths)/np.median(widths)>.42:continue
        if xx[-1]-ee[0]<w*.45:continue
        group=dict(y=y,x=xx,left=ee,width=widths)
        options=[g for g in groups if len(g[-1]['x'])==len(xx) and
            0<y-g[-1]['y']<=max(12,int(h*.028)) and
            np.max(np.abs(xx-g[-1]['x']))<=max(4,w*.025)]
        if options:max(options,key=len).append(group)
        else:groups.append([group])
    return groups


def _upright(image,deadline):
    h,w=image.shape[:2]
    raw=cv2.cvtColor(image,cv2.COLOR_BGR2LAB)[:,:,0]
    luminance=cv2.GaussianBlur(raw,(5,7),1.35).astype(np.float32)
    candidates=[]
    for prominence in PARAMETERS['shadow_prominences']:
        for group in _sections(luminance,prominence,deadline):
            first,last=group[0]['y'],group[-1]['y'];span=last-first
            if span<h*PARAMETERS['minimum_track_length_fraction'] or len(group)*2/span<PARAMETERS['minimum_track_coverage']:continue
            # A terminal band must reach the card's end, not stop at lettering
            # or the middle retaining band.
            if last<h*.91:continue
            ys=np.array([g['y'] for g in group]);xs=np.stack([g['x'] for g in group]);left=np.stack([g['left'] for g in group]);widths=xs-left
            quarter=max(4,len(group)//4)
            initial=np.median(widths[:quarter],axis=0);final=np.median(widths[-quarter:],axis=0)
            if np.mean(initial>final*PARAMETERS['minimum_taper_ratio'])<PARAMETERS['minimum_tapered_fraction']:continue
            # The shaft is a separate bright ridge followed by a dark side;
            # a flat painted stroke has no such longitudinal bright ridge.
            coefs=[np.polyfit(ys,(xs[:,j]+left[:,j])/2,1) for j in range(xs.shape[1])]
            medwidth=float(np.median(widths))
            if max(np.median(np.abs((xs[:,j]+left[:,j])/2-np.polyval(c,ys))) for j,c in enumerate(coefs))>max(1.5,medwidth*.18):continue
            if max(abs(c[0]) for c in coefs)>.2:continue
            # Each center must independently terminate in a dark cap followed
            # by a brighter background. A continuous board with drawn lengthwise
            # lines cannot borrow the end of a neighbouring rod.
            caps=[]
            for j,c in enumerate(coefs):
                axis_x=np.rint(np.polyval(c,np.arange(h))).astype(int)
                if np.any((axis_x<2)|(axis_x>=w-2)):break
                profile=np.mean(np.stack([luminance[np.arange(h),axis_x+offset] for offset in (-1,0,1)]),axis=0)
                start=max(last-8,int(h*.88));end=h-2
                closures=[y for y in range(start,end) if profile[y]<160 and
                    np.max(profile[y+1:min(h,y+14)])-profile[y]>=24]
                if not closures:break
                cap=min(closures,key=lambda y:float(profile[y]))
                caps.append(cap)
            if len(caps)!=len(coefs):continue
            candidates.append(dict(group=group,coefs=coefs,width=medwidth,prominence=prominence,caps=caps,span=span))
    if not candidates:return None
    agreed=[]
    for a in candidates:
        others=[]
        for b in candidates:
            if len(a['coefs'])!=len(b['coefs']):continue
            y=max(a['group'][0]['y'],b['group'][0]['y'])
            overlap=min(a['group'][-1]['y'],b['group'][-1]['y'])-y
            if overlap<min(a['span'],b['span'])*.6:continue
            if max(abs(np.polyval(x,y)-np.polyval(z,y)) for x,z in zip(a['coefs'],b['coefs']))>max(2,a['width']*.3):continue
            others.append(b)
        if len({b['prominence'] for b in others})>=2:agreed.append(a)
    if not agreed:return None
    return max(agreed,key=lambda a:(len(a['coefs']),a['span']))


def detect_carded_rods(image):
    """Return separately observed visible rod candidates, or reject with None."""
    if not isinstance(image,np.ndarray) or image.dtype!=np.uint8 or image.ndim!=3 or image.shape[2]!=3:
        raise ValueError('Expected uint8 HxWx3 BGR pixels')
    started=time.perf_counter();deadline=started+PARAMETERS['maximum_processing_seconds']
    oh,ow=image.shape[:2]
    scale=min(1.,PARAMETERS['maximum_processing_side']/max(oh,ow))
    small=image if scale==1 else cv2.resize(image,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA)
    for quad,width,height,saturation in _cards(small):
        if time.perf_counter()>=deadline:return None
        transform=cv2.getPerspectiveTransform(quad,np.float32([[0,0],[width-1,0],[width-1,height-1],[0,height-1]]))
        warped=cv2.warpPerspective(small,transform,(width,height))
        for turns,mirror in ((0,False),(1,False),(0,True),(1,True)):
            if time.perf_counter()>=deadline:return None
            upright=warped if turns==0 else cv2.rotate(warped,cv2.ROTATE_180)
            if mirror:upright=cv2.flip(upright,1)
            result=_upright(upright,deadline)
            if result is None:continue
            inv=np.linalg.inv(transform)
            def original(x,y):
                if mirror:x=width-1-x
                if turns:x,y=width-1-x,height-1-y
                p=inv@np.array([x,y,1.]);p=p[:2]/p[2]/scale
                return [round(float(p[0]),2),round(float(p[1]),2)]
            y0=result['group'][0]['y'];y1=result['group'][-1]['y']
            points=[]
            for j,coef in enumerate(result['coefs']):
                segment=[original(np.polyval(coef,y),y) for y in (y0,min(y1,result['caps'][j]))]
                center=original(np.polyval(coef,(y0+y1)/2),(y0+y1)/2)
                points.append(dict(x=center[0],y=center[1],r=3.,segment=segment))
            return points,dict(framework='CardedRodTerminalTracks',device='cpu',category='laid_rods',detected_category='laid_rods',prediction_valid=True,
                routing_evidence='complete_individual_terminal_tracks',diagnosis='visible_end_candidates',
                count_kind='visible_rod_candidates',review_required=True,count_is_total=False,pair_multiplier=False,
                manual_exemplars=False,count_unit='individual_visible_rod',
                mode='rectified_card_exposed_terminal_tracks',elapsed_seconds=round(time.perf_counter()-started,4),
                user_message='按纸带外各自可见的单根筷子轨迹计数；上下两段只计一次，请核对贴合处。',
                evidence=dict(directly_observed_individual_strokes=len(points),pair_multiplier_used=False,
                    independent_terminal_caps=len(result['caps']),consecutive_sections=len(result['group']),
                    card_saturation_threshold=saturation,independently_agreeing_shadow_prominences=list(PARAMETERS['shadow_prominences'])),
                parameters=PARAMETERS.copy(),limitation='Only regular colored display cards with individually visible tapered shafts, highlights, shadows and terminal caps. Counts are visible candidates; unseen inventory is not inferred.')
    return None
