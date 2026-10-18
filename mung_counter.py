"""Color baseline for separated green mung beans, never a pile-total estimator."""
import time
import cv2
import numpy as np

def detect_mung(image):
 started=time.perf_counter();height,width=image.shape[:2];scale=min(1.,1800/max(width,height))
 small=cv2.resize(image,None,fx=scale,fy=scale) if scale<1 else image
 hsv=cv2.cvtColor(small,cv2.COLOR_BGR2HSV)
 # One fixed color rule for the entire batch, not per-image tuned counts.
 mask=cv2.inRange(hsv,np.array([20,40,35]),np.array([95,255,255]))
 mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))
 contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
 filled=np.zeros_like(mask);cv2.drawContours(filled,contours,-1,255,cv2.FILLED)
 n,labels,stats,centers=cv2.connectedComponentsWithStats(filled,8)
 points=[];min_area=max(12,filled.size*.00003)
 for i in range(1,n):
  x,y,w,h,area=stats[i]
  if area<min_area or area>filled.size*.25 or x==0 or y==0 or x+w>=filled.shape[1] or y+h>=filled.shape[0]:continue
  cx,cy=centers[i];points.append(dict(x=round(float(cx/scale),2),y=round(float(cy/scale),2),r=round(float(min(w,h)/2/scale),2),area=round(float(area/scale**2),2)))
 points.sort(key=lambda p:(round(p['y']/20),p['x']))
 return points,dict(framework='MungColor',mode='green_color_components',diagnosis='detections_returned' if points else 'no_final_detections',elapsed_seconds=round(time.perf_counter()-started,4),
  foreground_fraction=round(float(np.mean(filled>0)),4),
  limitation='Separated green beans on contrasting background only; touching beans merge; border fragments excluded; yellow/green backgrounds may fail.')
