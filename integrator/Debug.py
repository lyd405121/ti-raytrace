import sys
import os
sys.path.append("accel")
sys.path.append("brdf")
sys.path.append("texture")


import taichi as ti
import time
import math
import numpy as np
import Camera as Camera
import Scene as Scene
import UtilsFunc as UF
import SceneData as SCD
import Texture as TX
import Disney as Disney
import Glass as Glass
import taichi_glsl as ts


@ti.data_oriented
class Debug:
    def __init__(self, imgSizeX, imgSizeY, cam, scene, stack_size):
        self.imgSizeX   = imgSizeX
        self.imgSizeY   = imgSizeY
        self.rgb_film   = ti.Vector.field(3, dtype=ti.f32)
        self.hdr        = ti.Vector.field(3, dtype=ti.f32)
        self.stack      = ti.field(dtype=ti.i32)
        self.cam        = cam
        self.scene      = scene
        self.stack_size = stack_size
    
    def setup_data_cpu(self):
        ti.root.dense(ti.ij, [self.imgSizeX, self.imgSizeY] ).place(self.rgb_film)
        ti.root.dense(ti.ij, [self.imgSizeX, self.imgSizeY] ).place(self.hdr)
        ti.root.dense(ti.ijk, [self.imgSizeX, self.imgSizeY, self.stack_size] ).place(self.stack)

    
    def setup_data_gpu(self):
        # do nothing
        self.imgSizeX = self.imgSizeX

    @ti.kernel
    def render(self):
        cam  = self.cam 
        scene = self.scene
        for i,j in self.rgb_film:

            origin          = cam.get_ray_origin()
            direction       = cam.get_ray_direction(i,j)
            radiance        = ti.Vector([0.0, 0.0, 0.0])

 
            t,pos,gnormal,normal,tex,prim_id  = scene.closet_hit(origin,direction, self.stack, i,j, self.stack_size)
            fnormal                   = UF.faceforward(normal, -direction, gnormal)
            mat_id                    = UF.get_prim_mindex(scene.primitive, prim_id)
            mat_color                 = UF.get_material_color(scene.material, mat_id)
            mat_type                  = UF.get_material_type(scene.material, mat_id)
            
            if t < UF.INF_VALUE:
                #radiance       = (fnormal + ti.Vector([1.0,1.0,1.0])) * 0.5
                #radiance       = (normal  + ti.Vector([1.0,1.0,1.0])) * 0.5 
                #radiance       = (gnormal + ti.Vector([1.0,1.0,1.0])) * 0.5
                radiance       = UF.get_material_color(scene.material, mat_id)

            self.hdr[i,j] = radiance