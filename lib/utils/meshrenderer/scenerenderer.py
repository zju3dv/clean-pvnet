# -*- coding: utf-8 -*-
import os
import glob

import math
import numpy as np
from write_xml import *
import auto_pose.meshrenderer.meshrenderer as mr
import auto_pose.meshrenderer.meshrenderer_phong as mr_phong
import cv2

from pysixd import view_sampler
from pysixd import transform

class SceneRenderer(object):

    def __init__(self, 
        models_cad_files, 
        vertex_tmp_store_folder, 
        vertex_scale,
        width,
        height,
        K,
        augmenters,
        vocdevkit_path,
        min_num_objects_per_scene,
        max_num_objects_per_scene,
        near_plane=10,
        far_plane=2000,
        min_n_views=1000,
        radius=650,
        obj_ids=None,
        model_type='reconst'):

        self._models_cad_files = models_cad_files
        self._width = width
        self._height = height
        self._radius = radius
        self._K = K
        self._augmenters = augmenters
        self._min_num_objects_per_scene = min_num_objects_per_scene
        self._max_num_objects_per_scene = max_num_objects_per_scene
        self._near_plane = near_plane
        self._far_plane = far_plane
        self.obj_ids = np.array(obj_ids)


        # pascal_imgs_path = os.path.join(vocdevkit_path, 'VOC2012/JPEGImages')
        self._voc_imgs = glob.glob( os.path.join(vocdevkit_path , '*.jpg' )) + glob.glob( os.path.join(vocdevkit_path, '*.png') )
        print len(self._voc_imgs)
        if model_type == 'reconst':
            self._renderer = mr_phong.Renderer(
                self._models_cad_files, 
                1, 
                vertex_tmp_store_folder=vertex_tmp_store_folder,
                vertex_scale=vertex_scale
            )
        elif model_type == 'cad':
            self._renderer = mr.Renderer(
                self._models_cad_files, 
                1, 
                vertex_tmp_store_folder=vertex_tmp_store_folder,
                vertex_scale=vertex_scale
            )
        else:
            print 'unknown model_type, ', model_type
            exit()

        azimuth_range =  (0, 2 * math.pi)
        elev_range =  (-0.5 * math.pi, 0.5 * math.pi)
        self.all_views, _ = view_sampler.sample_views(min_n_views, radius, azimuth_range, elev_range)    


    def render(self):
        if self._min_num_objects_per_scene == self._max_num_objects_per_scene:
            N =  self._min_num_objects_per_scene
        else:
            N = np.random.randint(
                self._min_num_objects_per_scene, 
                self._max_num_objects_per_scene
                )
        views = np.random.choice(self.all_views, N)
        obj_is = np.random.choice(len(self._models_cad_files), N)


        ts = []
        ts_norm = []
        Rs = []

        for v in views:
            success = False
            while not success:

                tz = np.random.triangular(self._radius-self._radius/3,self._radius,self._radius+self._radius/3)

                tx = np.random.uniform(-0.35 * tz * self._width / self._K[0,0], 0.35 * tz * self._width / self._K[0,0])
                ty = np.random.uniform(-0.35 * tz * self._height / self._K[1,1], 0.35 * tz * self._height / self._K[1,1])
                
                t = np.array([tx, ty, tz])
                R = transform.random_rotation_matrix()[:3,:3]
                t_norm = t/np.linalg.norm(t)

                if len(ts_norm) > 0 and np.any(np.dot(np.array(ts_norm),t_norm.reshape(3,1)) > 0.99):
                    success = False
                    print 'fail'
                else:
                    ts_norm.append(t_norm)
                    ts.append( t )
                    Rs.append( R )
                    success = True


        bgr, depth, bbs = self._renderer.render_many(
            obj_is, 
            self._width, 
            self._height, 
            self._K.copy(), 
            Rs, 
            ts, 
            self._near_plane, 
            self._far_plane,
            random_light=True 
        )


        rand_voc = cv2.imread( self._voc_imgs[np.random.randint( len(self._voc_imgs) )] )
        rand_voc = cv2.resize(rand_voc, (self._width, self._height))
        rand_voc = rand_voc.astype(np.float32) / 255.
        # print bgr.max()
        bgr = bgr.astype(np.float32) / 255.

        depth_three_chan = np.dstack((depth,)*3)
        bgr = rand_voc*(depth_three_chan==0.0).astype(np.uint8) + bgr*(depth_three_chan>0).astype(np.uint8)

        obj_info = []
        for (x, y, w, h), obj_id in zip(bbs, self.obj_ids[np.array(obj_is)]):
            xmin = np.minimum(x, x+w)
            xmax = np.maximum(x, x+w)
            ymin = np.minimum(y, y+h)
            ymax = np.maximum(y, y+h)
            obj_info.append({'id': obj_id, 'bb': [int(xmin), int(ymin), int(xmax), int(ymax)]})

        bgr = (bgr*255.0).astype(np.uint8)

        if self._augmenters != None:
            bgr = self._augmenters.augment_image(bgr)

        return bgr, obj_info