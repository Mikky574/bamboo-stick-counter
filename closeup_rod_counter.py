"""Restricted visible rod end-face candidates; image pixels only, CPU.

This is not whole-rod or hidden-total counting. It requires compact contrasting
terminal faces, a common long-axis direction, and brown shaft evidence preceding
each face. Printed patches midway down a continuing shaft are rejected. Boxed,
all-wood, crossing and non-contrasting end faces remain unsupported.
"""
import time
import cv2
import numpy as np


def _read_regions(mask, minimum_area):
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype('uint8'))
    regions=[]
    for k in range(1,n):
        if stats[k,4] < minimum_area: continue
        y,x=np.where(lab==k)
        short,long=sorted(cv2.minAreaRect(np.column_stack([x,y]).astype('float32'))[1])
        if long/max(short,1)>6: continue
        distance=cv2.distanceTransform((lab==k).astype('uint8'),cv2.DIST_L2,5)
        radius=float(distance.max())
        if radius<3: continue
        levels=[]
        for fraction in (.25,.35,.45,.55,.65):
            _,_,ss,cc=cv2.connectedComponentsWithStats((distance>fraction*radius).astype('uint8'))
            cores=[(s,c) for s,c in zip(ss[1:],cc[1:]) if s[4]>max(10,minimum_area*.15)]
            levels.append(cores)
        # Never silently drop a compact face whose split is unstable: doing so
        # can report a confident subtotal after JPEG holes split just one core.
        if len(set(len(c) for c in levels))!=1 or not levels[0] or len(levels[0])>8:
            return None
        reference=levels[2]
        for level in levels:
            distances=np.linalg.norm(np.array([c for _,c in level])[:,None,:]-np.array([c for _,c in reference])[None,:,:],axis=2)
            matches=np.argmin(distances,axis=1)
            # Matching must be one-to-one, so two cores cannot track the same
            # reference while another reference silently loses support.
            if len(set(matches.tolist()))!=len(reference) or np.max(distances[np.arange(len(level)),matches])>radius*.65:
                return None
        for _,center in reference:
            regions.append(dict(center=center,radius=radius,core_support_levels=len(levels)))
    return regions


def detect_closeup_rods(image):
    """Return candidate points/meta, or None; never infer obscured total rods."""
    started=time.perf_counter()
    if not isinstance(image,np.ndarray) or image.dtype!=np.uint8 or image.ndim!=3 or image.shape[2]!=3:
        raise ValueError('Expected uint8 BGR pixels')
    original_h,original_w=image.shape[:2]
    scale=min(1.,1400/max(original_h,original_w))
    if scale<1:image=cv2.resize(image,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA)
    h,w=image.shape[:2]
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV);H,S,V=cv2.split(hsv)
    lab=cv2.cvtColor(image,cv2.COLOR_BGR2LAB)
    # Warm brown shafts need a clearly contrasting terminal face. Both paint
    # and dark/cool faces are supported; no stored sample templates are used.
    face=(((H>150)&(S>70)&(V>80))|((lab[:,:,2]<134)&(V<150)&(S<100))).astype('uint8')
    regions=_read_regions(face,h*w*.00015)
    if not regions or len(regions)>32:return None
    warm=((H>=2)&(H<=40)&(S>65)&(V>65)&(V<246))
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    lines=cv2.HoughLinesP(cv2.Canny(gray,40,110),1,np.pi/180,50,minLineLength=max(45,min(h,w)*.18),maxLineGap=15)
    if lines is None:return None
    vectors=[];angles=[];weights=[]
    for line in lines[:,0]:
        a=line[:2].astype(float);b=line[2:].astype(float);v=b-a;length=float(np.linalg.norm(v));v/=length
        angle=np.arctan2(v[1],v[0])%np.pi
        vectors.append((a,b,v,length,angle));angles.append(angle);weights.append(length)
    bins=np.linspace(0,np.pi,37)
    hist,_=np.histogram(angles,bins,weights=weights)
    dominant=(bins[int(np.argmax(hist))]+bins[int(np.argmax(hist))+1])/2
    def angle_distance(a,b):return abs((a-b+np.pi/2)%np.pi-np.pi/2)
    candidates=[]
    for region in regions:
        c=region['center'];r=region['radius'];options=[]
        for a,b,v,length,angle in vectors:
            if angle_distance(angle,dominant)>np.deg2rad(20) or length<4*r:continue
            cross=abs(v[0]*(c-a)[1]-v[1]*(c-a)[0])
            if cross>1.7*r or min(np.linalg.norm(c-a),np.linalg.norm(c-b))>12*r:continue
            v=v.copy()
            if np.dot(c-(a+b)/2,v)<0:v=-v
            def support(sign,lo,hi,n):
                pos=c+sign*np.linspace(lo*r,hi*r,n)[:,None]*v
                valid=(pos[:,0]>=0)&(pos[:,0]<w)&(pos[:,1]>=0)&(pos[:,1]<h)
                if valid.mean()<.7:return None
                pos=pos[valid].astype(int)
                return float(warm[pos[:,1],pos[:,0]].mean())
            before=support(-1,2.5,10,30);after=support(1,1.3,2.7,16)
            if before is None or after is None:continue
            if before<.85 or after>.72 or before-after<.27:continue
            options.append((before-after,before,after,v))
        if not options:continue
        _,before,after,direction=max(options,key=lambda v:v[0])
        region.update(direction=direction,shaft_support=before,post_end_shaft_support=after)
        candidates.append(region)
    # Accept only the entire eligible compact-face set, not a convenient subset.
    # An unsupported nearby face signals unresolved separation or a false patch.
    if not candidates or len(candidates)!=len(regions):return None
    if len(candidates)>1:
        rs=np.array([x['radius'] for x in candidates])
        if rs.max()/rs.min()>2:return None
        # Different perspective angles invalidate direct projections; use common
        # axis to establish one terminal band rather than two ends per shaft.
        common=np.array([np.cos(dominant),np.sin(dominant)])
        p=np.array([np.dot(x['center'],common) for x in candidates])
        if np.ptp(p)>np.median(rs)*6:return None
    candidates.sort(key=lambda c:(float(c['center'][0]),float(c['center'][1])))
    points=[]
    for c in candidates:
        center=c['center'];end=center-c['direction']*c['radius']*6
        points.append(dict(x=round(float(center[0])/scale,2),y=round(float(center[1])/scale,2),r=round(c['radius']*.4/scale,2),
            segment=[[round(float(v)/scale,2) for v in end],[round(float(v)/scale,2) for v in center]]))
    points.sort(key=lambda p:(p['x'],p['y']))
    return points,dict(framework='CloseupRodTerminalFaces',device='cpu',category='laid_rods',detected_category='laid_rods',
        prediction_valid=True,diagnosis='visible_end_candidates',count_kind='visible_rod_candidates',count_unit='visible_terminal_face_candidate',
        count_is_total=False,review_required=True,manual_exemplars=False,mode='compact_terminal_face_and_shaft_support',
        user_message='按清晰可见的端面统计筷子候选；上端或部分杆身未完整入镜，不能据此判断整盒或遮挡部分的总数量，请核对标记。',
        elapsed_seconds=round(time.perf_counter()-started,4),processing_scale=scale,
        candidate_evidence=[dict(shaft_support=round(c['shaft_support'],3),post_end_shaft_support=round(c['post_end_shaft_support'],3),
            core_support_levels=c['core_support_levels'],radius=round(c['radius']/scale,2)) for c in candidates],
        limitation='Restricted contrasting terminal faces backed by brown shafts; candidate visible rods only. Cropped or hidden ends and shafts do not establish a full bundle total. Paint patches, wooden trays, shadows and touching faces can remain ambiguous.')
