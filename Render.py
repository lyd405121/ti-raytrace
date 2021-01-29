import sys
import os
sys.path.append("accel")
sys.path.append("brdf")
sys.path.append("texture")
import taichi as ti
import time
import math
import numpy as np
import Canvas as Canvas
import Bvh as Bvh
import UtilsFunc as UF
import SceneData as SCD
import Texture as TX
import Disney as Disney
import Glass as Glass
import taichi_glsl as ts

ti.init(arch=ti.gpu)
MAX_DEPTH  = 10
imgSizeX   = 512
imgSizeY   = 512
gui        = ti.GUI('sesph', res=(imgSizeX, imgSizeY))
rt_canvas  = Canvas.Canvas(imgSizeX, imgSizeY)
bvh        = Bvh.Bvh()
disney     = Disney.Disney()
glass      = Glass.Glass()
env        = TX.Texture()

@ti.func
def powerHeuristic( a,  b):
    t = a* a
    return t / (b*b + t)

@ti.func
def  offset_ray(p,  n):
    int_scale   = 256.0
    float_scale = 1.0 / 2048.0
    origin      = 1.0 / 256.0
    ret         = ti.Vector([0.0, 0.0, 0.0])

    for k in ti.static(range(3)):
        i_of = int(int_scale * n[k])
        i_p  = ti.bit_cast(p[k], ti.i32)
        if p[k] < 0.0:
            i_p = i_p - i_of
        else:
            i_p = i_p + i_of
        f_p = ti.bit_cast(i_p, ti.f32)

        
        if abs(p[k]) < origin:
            ret[k] = p[k] + float_scale * n[k]
        else:
            ret[k] = f_p
    return ret

@ti.func
def faceforward(n,  i,  nref):
    return ts.sign(i.dot(nref)) * n


@ti.kernel
def render():
    for i,j in rt_canvas.hdr:

        next_origin     = rt_canvas.get_ray_origin()
        next_dir        = rt_canvas.get_ray_direction(i,j)
        depth           = 0
        pdf             = 1.0
        perfect_spec    = 0
        f_or_b          = 1.0
        brdf            = ti.Vector([1.0, 1.0, 1.0])
        throughout      = ti.Vector([1.0, 1.0, 1.0])
        radiance        = ti.Vector([0.0, 0.0, 0.0])
        
        while(depth < MAX_DEPTH):
            origin    = next_origin 
            direction = next_dir
            
            t,pos,normal,tex,prim_id  = bvh.closet_hit(origin,direction, rt_canvas.stack, i,j, Canvas.STACK_SIZE)
            fnormal                   = faceforward(normal, -direction, normal)
            mat_id        = UF.get_prim_mindex(bvh.primitive, prim_id)
            mat_emission  = UF.get_material_emission(bvh.material, mat_id)
            mat_type      = UF.get_material_type(bvh.material, mat_id)
            
            if t < UF.INF_VALUE:
                if mat_emission.sum() > Bvh.IS_LIGHT:
                    fCosTheta = direction.dot(normal)
                    if fCosTheta < 0.0:
                        area = bvh.get_prim_area(prim_id)
                        if (perfect_spec ==1) | (depth == 0):
                            radiance += mat_emission 
                            #radiance = radiance
                        else:
                            lightPdf = (t * t) / (area * fCosTheta)
                            radiance += powerHeuristic(pdf, lightPdf) * throughout * mat_emission
                    break
                else:
                    next_origin    = offset_ray(pos , fnormal)

                    #btdf
                    if mat_type == 1:
                        perfect_spec = 1
                        next_dir,f_or_b    = glass.sample(direction, normal, t, i, j, bvh.material, mat_id)
                        brdf,pdf  = glass.evaluate(normal,  next_dir, -direction,bvh.material, mat_id)
                    else:
                        #disney
                        perfect_spec = 0

                        #direct lighting 
                        light_prim_id          = bvh.get_random_light_prim_index()
                        light_pos,light_normal = bvh.get_prim_random_point_normal(light_prim_id)

                        light_dir              = light_pos - next_origin 
                        light_dist             = light_dir.norm()
                        light_dir              = light_dir / light_dist
                        NdotL_light            = -light_normal.dot(light_dir)
                        NdotL_surface          = normal.dot(light_dir)

                        ####shadow
                        if (NdotL_light > 0.0) & (NdotL_surface > 0.0):
                            shadow_prim      = bvh.closet_hit_shadow(next_origin, light_dir, rt_canvas.stack, i,j, Canvas.STACK_SIZE)
                            shadow_mat       = UF.get_prim_mindex(bvh.primitive, shadow_prim)
                            shadow_emission  = UF.get_material_emission(bvh.material, shadow_mat)
                            if shadow_emission.sum() > Bvh.IS_LIGHT:
                                brdf,pdf = disney.evaluate(normal, -direction, light_dir,  bvh.material, mat_id)
                                if pdf > 0.0:
                                    light_emission         = UF.get_material_emission(bvh.material, UF.get_prim_mindex(bvh.primitive, light_prim_id))
                                    light_area             = bvh.get_prim_area(light_prim_id)
                                    lightPdf  = light_dist * light_dist / (light_area * NdotL_light) 
                                    radiance += powerHeuristic(lightPdf, pdf ) * light_emission / max(0.001, lightPdf)  *throughout*brdf

                        next_dir, f_or_b    = disney.sample(direction, normal, i, j, bvh.material, mat_id)
                        brdf,pdf            = disney.evaluate(normal,  next_dir, -direction, bvh.material, mat_id)

                    next_origin         = offset_ray(pos , ts.sign(f_or_b)*fnormal)

                    if pdf > 0.0:
                        throughout    *= brdf / pdf * abs(f_or_b)
                        depth     += 1
                    else:
                        break
                    

                    #debug code , color map 
                    #nl = -normal.dot(direction)
                    #nv = -normal.dot(direction)
                    #radiance = ts.reflect(-direction, normal) 
                    #radiance = (next_dir + ti.Vector([1.0, 1.0, 1.0]))*0.5
                    #radiance  = brdf
                    #if depth == 0:
                    #    radiance  = ti.Vector([1.0, 0.0, 0.0])
                    #if depth == 1:
                    #    radiance  = ti.Vector([0.0, 1.0, 0.0])
                    #if depth == 2:
                    #    radiance  = ti.Vector([0.0, 0.0, 1.0])
                    #break

            else:
                dis = ti.sqrt(direction.x*direction.x + direction.z*direction.z)
                tx  = (ts.atan(direction.z , direction.x) + 3.1415926) / 3.1415926 / 2.0
                ty  = (ts.atan(direction.y , dis))        / 3.1415926 + 0.5
                radiance += env.teture2D(tx, ty) * throughout
                break
        
        frame = float(rt_canvas.frame_gpu[0])
        rt_canvas.hdr[i,j] = (radiance + rt_canvas.hdr[i,j] * frame)/ (frame + 1.0)
        #rt_canvas.hdr[i,j] = radiance

def build_cornell():
    bvh.add_obj('cornell_box.obj')

    shape           = SCD.Shape()
    shape.type      = SCD.SHPAE_SPHERE
    shape.pos       = [255.0, 400.0, 255.0]
    shape.setRadius(50.0)
    
    mat   = SCD.Material()
    mat.setEmission([2.0, 2.0, 2.0])
    mat.setColor([1.0, 1.0, 0.0])
    #bvh.add_shape(shape, mat)


def set_trans():
    matTrans = SCD.Material()
    matTrans.type = 1
    matTrans.setIor(1.4)
    matTrans.setColor([1.0, 1.0, 1.0])
    bvh.modify_mat(0, matTrans)

def set_metal():
    matMetal = SCD.Material()
    matMetal.setColor([1.0, 1.0, 1.0])
    matMetal.setMetal(1.0)
    matMetal.setRough(0.0)
    bvh.modify_mat(0, matMetal)

def build_test():
    #bvh.add_obj('model/mc.obj')
    bvh.add_obj('model/sphere.obj')
    #bvh.add_obj('model/box.obj')
    #bvh.add_obj('model/cylinder.obj')
    #bvh.add_obj('model/Teapot.obj')
    centre = bvh.maxboundarynp+bvh.minboundarynp
    size   = bvh.maxboundarynp-bvh.minboundarynp
    
    shape           = SCD.Shape()
    shape.type      = SCD.SHPAE_SPHERE
    shape.pos       = [centre[0,0], centre[0,1] + size[0,1], centre[0,2]]
    shape.setRadius(0.3)
    
    mat   = SCD.Material()
    mat.setEmission([10.0, 10.0, 10.0])
    mat.setColor([1.0, 1.0, 0.0])
    bvh.add_shape(shape, mat)

    #set_trans()
    #set_metal()
    



def build_scene():
    build_test()
    #build_cornell()

    bvh.setup_data_cpu()
    env.load_image("image/env.png")
    

    bvh.setup_data_gpu()
    env.setup_data_gpu()

def centre_cam():
    centre = bvh.maxboundarynp+bvh.minboundarynp
    size   = bvh.maxboundarynp-bvh.minboundarynp
    rt_canvas.scale = math.sqrt(size[0,0]*size[0,0] + size[0,1]*size[0,1] + size[0,2]*size[0,2]) 

    rt_canvas.set_target(centre[0,0]*0.5, centre[0,1]*0.5, centre[0,2]*0.5)
    #rt_canvas.yaw = rt_canvas.yaw + 0.01 
    #rt_canvas.pitch = 0.9
    rt_canvas.update_cam()



build_scene()
centre_cam()

while gui.running:
    #print("********frame start*********")
    #if rt_canvas.frame == 0:
    #    centre_cam()
    #    rt_canvas.frame +=1

    render()
    rt_canvas.tone_map_host(0.5, 1.0/2.2)
    gui.set_image(rt_canvas.img.to_numpy())
    gui.show()
    rt_canvas.update_frame()

    #print("********frame end*********")
    #sys.exit()


