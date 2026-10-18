"""Pixel-only regressions for paint/printing invariance of separated rods."""
import base64
from pathlib import Path
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from app import app
from laid_rod_counter import detect_laid_rods
from rod_silhouette import detect_rod_silhouettes

ROOT=Path(__file__).resolve().parent


def rods(count=7,paint=False):
    image=np.full((700,1000,3),246,np.uint8)
    for index in range(count):
        x=500 if count==1 else 120+index*125
        shape=np.array([[x-10,110],[x+10,110],[x+8,590],[x-8,590]],np.int32)
        cv2.fillPoly(image,[shape],(110,176,222))
        if paint:
            cv2.line(image,(x-5,130),(x-5,575),(30,30,235),2)
            cv2.line(image,(x+4,130),(x+4,575),(25,25,25),2)
            for y in (200,350,500):
                cv2.circle(image,(x,y),8,(20,20,20),-1)
                cv2.circle(image,(x,y),3,(240,240,240),-1)
            for y in range(145,570,33):
                cv2.line(image,(x-8,y),(x+8,y),(45,90,210),2)
    return image


class RodSurfaceChecks(unittest.TestCase):
    def test_surface_lines_and_icons_do_not_change_instance_count(self):
        for count in (1,7):
            for paint in (False,True):
                for turns in (0,1):
                    with self.subTest(count=count,paint=paint,turns=turns):
                        image=np.ascontiguousarray(np.rot90(rods(count,paint),turns))
                        points,meta=detect_laid_rods(image)
                        self.assertEqual(len(points),count)
                        self.assertFalse(meta['texture_seams_used'])
                        self.assertEqual(meta['framework'],'RodSilhouetteConsensus')

    def test_user_decorated_review_photo_keeps_ten_rods_through_image_only_api(self):
        source=ROOT/'samples/flat_rods_eval_20261018/truth/chopsticks_striped.review.jpg'
        image=cv2.imdecode(np.fromfile(source,np.uint8),cv2.IMREAD_COLOR)
        encoded=cv2.imencode('.png',image)[1]
        with patch('smart_counter.scene_hint',return_value=dict(accepted=True,scene='laid_rods')):
            with patch('auto_counter.detect_auto',side_effect=AssertionError('No neural counter needed')):
                with app.test_client() as client:
                    r=client.post('/api/count',json={'image':'data:image/png;base64,'+base64.b64encode(encoded).decode()})
        self.assertEqual(r.status_code,200)
        result=r.get_json();self.assertEqual(result['count'],10)
        self.assertEqual(result['meta']['framework'],'RodSilhouetteConsensus')
        self.assertEqual(len(result['points']),10)

    def test_crossed_and_cut_off_shapes_do_not_pass_silhouette_gate(self):
        crossed=np.full((700,1000,3),246,np.uint8)
        cv2.line(crossed,(250,150),(750,550),(110,176,222),20)
        cv2.line(crossed,(750,150),(250,550),(110,176,222),20)
        self.assertIsNone(detect_rod_silhouettes(crossed))
        cropped=np.full_like(crossed,246)
        cv2.rectangle(cropped,(460,0),(480,590),(110,176,222),-1)
        self.assertIsNone(detect_rod_silhouettes(cropped))

    def test_touching_pairs_require_more_than_a_single_outer_silhouette(self):
        image=np.full((700,1000,3),246,np.uint8)
        cv2.rectangle(image,(460,110),(501,590),(110,176,222),-1)
        self.assertIsNone(detect_rod_silhouettes(image))

    def test_large_images_preserve_count_and_return_original_coordinates(self):
        image=rods(7,True)
        large=cv2.resize(image,None,fx=3,fy=3)
        points,meta=detect_laid_rods(large)
        self.assertEqual(len(points),7);self.assertLess(meta['processing_scale'],1)
        reference=detect_laid_rods(image)[0]
        for a,b in zip(points,reference):
            self.assertAlmostEqual(a['x']/3,b['x'],delta=2)
            self.assertAlmostEqual(a['y']/3,b['y'],delta=2)
            for x,y in a['segment']:
                self.assertTrue(0<=x<large.shape[1] and 0<=y<large.shape[0])


if __name__=='__main__':unittest.main(verbosity=2)
