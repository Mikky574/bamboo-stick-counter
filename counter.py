"""Local end-face counter. No pretrained semantic model is used."""
import cv2
import numpy as np
from scipy import ndimage as ndi

def detect(image, diameter=24, polarity='bright', threshold=0, sensitivity=0.45):
    h,w=image.shape[:2]
    scale=min(1,1600/max(h,w)); im=cv2.resize(image,None,fx=scale,fy=scale) if scale<1 else image
    d=max(4,float(diameter)*scale)
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    gray=cv2.GaussianBlur(gray,(3,3),0)
    flag=cv2.THRESH_BINARY_INV if polarity=='dark' else cv2.THRESH_BINARY
    _,mask=cv2.threshold(gray,int(threshold),255,flag | (cv2.THRESH_OTSU if not threshold else 0))
    mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    distance=ndi.distance_transform_edt(mask)
    peaks=(distance==ndi.maximum_filter(distance,size=max(3,int(d*0.65)))) & (distance>=d*float(sensitivity)/2)
    labels,n=ndi.label(peaks)
    candidates=[]
    for sl in ndi.find_objects(labels):
        if sl is None: continue
        region=distance[sl]; y,x=np.unravel_index(np.argmax(region),region.shape)
        y+=sl[0].start; x+=sl[1].start; radius=float(distance[y,x])
        if radius>d*1.1: continue
        candidates.append((x,y,radius))
    candidates.sort(key=lambda p:-p[2]); kept=[]
    for x,y,r in candidates:
        if all((x-a)**2+(y-b)**2 >= (d*0.55)**2 for a,b,_ in kept): kept.append((x,y,r))
    kept.sort(key=lambda p:(round(p[1]/max(d,1)),p[0]))
    return [{'x':round(x/scale,2),'y':round(y/scale,2),'r':round(r/scale,2)} for x,y,r in kept],mask

def annotate(image,points):
    out=image.copy()
    for i,p in enumerate(points,1):
        x,y=int(p['x']),int(p['y']); r=max(4,int(p.get('r',8)))
        cv2.circle(out,(x,y),r,(0,220,90),2)
        cv2.putText(out,str(i),(x+3,y-3),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,0,0),3,cv2.LINE_AA)
        cv2.putText(out,str(i),(x+3,y-3),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1,cv2.LINE_AA)
    return out
