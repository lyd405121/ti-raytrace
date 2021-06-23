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

MAX_DEPTH  = 15
@ti.data_oriented
class PathTrace:
    def __init__(self, imgSizeX, imgSizeY, cam, scene, stack_size):
        self.imgSizeX   = imgSizeX
        self.imgSizeY   = imgSizeY
        self.rgb_film   = ti.Vector.field(3, dtype=ti.f32)
        self.hdr        = ti.Vector.field(3, dtype=ti.f32)
        self.stack      = ti.field(dtype=ti.i32)
        self.cam        = cam
        self.scene      = scene
        self.stack_size = stack_size
    @ti.pyfunc
    def setup_data_cpu(self):
        ti.root.dense(ti.ij, [self.imgSizeX, self.imgSizeY] ).place(self.rgb_film)
        ti.root.dense(ti.ij, [self.imgSizeX, self.imgSizeY] ).place(self.hdr)
        ti.root.dense(ti.ijk, [self.imgSizeX, self.imgSizeY, self.stack_size] ).place(self.stack)

    @ti.pyfunc
    def setup_data_gpu(self):
        # do nothing
        self.imgSizeX = self.imgSizeX

    @ti.kernel
    def render(self):
        cam, scene  = ti.static(self.cam, self.scene)
        for i,j in self.rgb_film:

            next_origin     = cam.get_ray_origin()
            next_dir        = cam.get_ray_direction(i,j)
            depth           = 0
            light_pdf       = 1.0
            brdf_pdf        = 1.0
            perfect_spec    = 1
            f_or_b          = 1.0
            brdf            = 1.0
            throughout      = ti.Vector([1.0, 1.0, 1.0])
            radiance        = ti.Vector([0.0, 0.0, 0.0])
            while(depth < MAX_DEPTH):
                origin    = next_origin 
                direction = next_dir
 
                t,pos,gnormal,normal,tex,prim_id  = scene.closet_hit(origin,direction, self.stack, i,j, self.stack_size)
                fnormal                   = UF.faceforward(normal, -direction, gnormal)
                mat_id                    = UF.get_prim_mindex(scene.primitive, prim_id)
                mat_color                 = UF.get_material_color(scene.material, mat_id)
                mat_type                  = UF.get_material_type(scene.material, mat_id)
                
                if t < UF.INF_VALUE:
                    if mat_type == SCD.MAT_LIGHT:
                        fCosTheta = abs(direction.dot(gnormal))

                        if perfect_spec == 1:
                            radiance += throughout * mat_color 
                        else:
                            area      = scene.get_prim_area(prim_id)*scene.light_count
                            light_pdf = (t * t) / (area * fCosTheta)
                            radiance += UF.powerHeuristic(brdf_pdf, light_pdf)   * throughout * mat_color
                        break
                    else:
                        

                        
                        reflect_color  = UF.srgb_to_lrgb(UF.get_material_color(scene.material, mat_id))

                        #btdf
                        if mat_type == SCD.MAT_GLASS:
                            perfect_spec       = 1
                            next_dir,f_or_b    = Glass.sample(direction, normal, t,  scene.material, mat_id)
                            brdf, brdf_pdf     = Glass.evaluate_pdf(normal,  next_dir, -direction,scene.material, mat_id)
                            
                        else:
                            #Disney
                            perfect_spec = 0

                            
                            #direct lighting 
                            light_pos,light_normal, light_dir,light_emission,light_dist,light_prim_id,light_choice_pdf,light_dir_pdf = scene.sample_li( pos)
                            NdotL_surface          = fnormal.dot(light_dir)
                            NdotL_light            = light_normal.dot(light_dir)
                            if (NdotL_surface < 0.0) & (NdotL_light > 0.0):
                                shadow_t,shadow_prim    = scene.closet_hit_shadow(light_pos, light_dir, self.stack, i,j, self.stack_size)
                                if shadow_prim == prim_id:
                                    brdf,brdf_pdf          = Disney.evaluate_pdf(fnormal, -direction, -light_dir,  scene.material, mat_id)
                                    light_pdf              = light_dist * light_dist * light_choice_pdf/NdotL_light
                                    if (brdf_pdf > 0.0 ):
                                        radiance              += UF.powerHeuristic(light_pdf, brdf_pdf )/ max(0.0001, light_pdf) * light_emission * throughout*reflect_color*brdf*abs(NdotL_surface)
                                        #print(light_choice_pdf, brdf_pdf, light_emission , throughout,reflect_color,brdf,NdotL_surface)
                            
                            next_dir, f_or_b               = Disney.sample(direction, fnormal,scene.material, mat_id)
                            brdf,brdf_pdf                  = Disney.evaluate_pdf(fnormal,  -direction, next_dir,  scene.material, mat_id)
                            brdf                          *= abs(normal.dot(next_dir))
                        next_origin         = UF.offset_ray(pos , ts.sign(f_or_b)*fnormal)
                        
                        if brdf_pdf > 0.0:
                            if f_or_b < 0.0:
                                extinction      = UF.get_material_extinction(scene.material, mat_id)
                                R               = ti.exp(-t / extinction) 
                                if ti.random() >= R:
                                    break
                            throughout     *=  brdf  / brdf_pdf  * reflect_color 
                            depth     += 1
                        else:
                            break
                else:
                    dis = ti.sqrt(direction.x*direction.x + direction.z*direction.z)
                    tx  = (ts.atan(direction.z , direction.x) + 3.1415926) / 3.1415926 / 2.0
                    ty  = (ts.atan(direction.y , dis))        / 3.1415926 + 0.5
                    radiance += UF.srgb_to_lrgb( scene.env.texture2D(tx, ty)) * throughout * scene.env_power
                    break
                
            frame = float(cam.frame_gpu[0])
            coff  = 1.0 / (frame + 1.0)
            self.hdr[i,j] = radiance * coff  + self.hdr[i,j]  * (1.0 - coff)