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
import BDPT_Vertex as Vertex
import taichi_glsl as ts

STOP_DEPTH      = 10000
MAX_DEPTH       = 5
EYE_MAX_DEPTH   = MAX_DEPTH +2 
LIGHT_MAX_DEPTH = MAX_DEPTH +1

VERTEX_NONE     = 0
VERTEX_LIGHT    = 1 
VERTEX_LENS     = 2 
VERTEX_SURFACE  = 3


@ti.data_oriented
class BDPT:
    def __init__(self, imgSizeX, imgSizeY, cam, scene, stack_size):
        self.imgSizeX    = imgSizeX
        self.imgSizeY    = imgSizeY
 
        self.rgb_film    = ti.Vector.field(3, dtype=ti.f32)
        self.hdr         = ti.Vector.field(3, dtype=ti.f32)
        self.radiance    = ti.Vector.field(3, dtype=ti.f32)
        self.stack       = ti.field(dtype=ti.i32)

        self.cam         = cam
        self.scene       = scene

        self.light       = Vertex.Vertex(imgSizeX, imgSizeY, LIGHT_MAX_DEPTH)
        self.eye         = Vertex.Vertex(imgSizeX, imgSizeY, EYE_MAX_DEPTH)
        self.sample      = Vertex.Vertex(imgSizeX, imgSizeY, 1)

        self.ltemp       = Vertex.Vertex(imgSizeX, imgSizeY, 1)
        self.etemp       = Vertex.Vertex(imgSizeX, imgSizeY, 1)
        self.lminustemp  = Vertex.Vertex(imgSizeX, imgSizeY, 1)
        self.eminustemp  = Vertex.Vertex(imgSizeX, imgSizeY, 1)

        self.stack_size = stack_size

    @ti.pyfunc
    def setup_data_cpu(self):
        ti.root.dense(ti.ij,  [self.imgSizeX, self.imgSizeY] ).place(self.rgb_film)
        ti.root.dense(ti.ij,  [self.imgSizeX, self.imgSizeY] ).place(self.radiance)
        ti.root.dense(ti.ij,  [self.imgSizeX, self.imgSizeY] ).place(self.hdr)
        ti.root.dense(ti.ijk, [self.imgSizeX, self.imgSizeY, self.stack_size] ).place(self.stack)


        self.light.setup_data_cpu()
        self.eye.setup_data_cpu()
        self.sample.setup_data_cpu()
        self.ltemp.setup_data_cpu()
        self.etemp.setup_data_cpu()
        self.lminustemp.setup_data_cpu()
        self.eminustemp.setup_data_cpu()

    @ti.pyfunc
    def setup_data_gpu(self):
        # do nothing
        self.imgSizeX = self.imgSizeX
        self.light.setup_data_gpu()
        self.eye.setup_data_gpu()
        self.sample.setup_data_gpu()
        self.ltemp.setup_data_gpu()
        self.etemp.setup_data_gpu()
        self.lminustemp.setup_data_gpu()
        self.eminustemp.setup_data_gpu()




    @ti.func
    def remap0(self, f):
        if f == 0.0:
            f = 1.0
        return f



    @ti.func
    def light_origin_pdf(self, prim_id):
        pdfPos            = 1.0 / self.scene.get_prim_area(prim_id)
        pdfChoice         = 1.0 / float(self.scene.light_count)
        return pdfPos*pdfChoice


    @ti.func
    def eye_path(self, i, j):
        eye,scene,cam             = ti.static(self.eye, self.scene, self.cam)
        #eye,scene            = ti.static(self.eye, self.scene)
        origin                    = self.cam.get_ray_origin()
        dir                       = self.cam.get_ray_direction(i,j)

        eye.pos[i, j, 0]       = origin
        eye.normal[i, j, 0]    = dir
        eye.beta[i, j, 0]      = ti.Vector([1.0,1.0,1.0])
        eye.fpdf[i, j, 0]      = 1.0
        eye.type[i, j, 0]      = VERTEX_LENS

        pre_depth                 = 0
        depth                     = 1
        #pdfFwd                    = abs(dir.dot(cam.get_optical_axis()))
        pdfFwd                    = 1.0
        pdfRev                    = 0.0
        beta                      = ti.Vector([1.0,1.0,1.0])

        
        while(depth < EYE_MAX_DEPTH):
            t,pos,gnormal,normal,tex,prim_id  = scene.closet_hit(origin, dir, self.stack, i,j, self.stack_size)
            if t < UF.INF_VALUE:
                fnormal                           = UF.faceforward(normal, -dir, gnormal)
                mat_id                            = UF.get_prim_mindex(scene.primitive, prim_id)
                mat_color                         = UF.get_material_color(scene.material, mat_id)
                mat_type                          = UF.get_material_type(scene.material, mat_id)

                to                                = pos  - origin
                dist                              = max(ts.length(to), 0.01)
                inv_dist2                         = 1.0 / (dist*dist)
                to                                = to / dist
                
                eye.pos[i, j, depth]           = pos
                eye.normal[i, j, depth]        = normal
                eye.snormal[i, j, depth]       = fnormal
                eye.wo[i, j, depth]            = dir
                eye.rpdf[i, j, depth]          = 0.0
                eye.prim [i, j, depth]         = prim_id
                eye.mat[i, j, depth]           = mat_id
                eye.fpdf[i, j, depth]          = pdfFwd * abs(to.dot(eye.normal[i, j, pre_depth])) * inv_dist2
                if mat_type == SCD.MAT_LIGHT:
                    eye.beta[i, j, depth]  = beta * mat_color * abs(normal.dot(dir) ) 
                    eye.type[i, j, depth]  = VERTEX_LIGHT
                    depth+=1
                    break
                else:              
                    eye.beta[i, j, depth]      = beta* abs(dir.dot(normal))
                    eye.type[i, j, depth]      = VERTEX_SURFACE
                    
                
                next_dir                          = dir
                reflect_color                     = UF.srgb_to_lrgb(mat_color)
                brdf                              = 0.0
                f_or_b                            = 1.0
                if mat_type == SCD.MAT_GLASS:
                    next_dir,f_or_b    = Glass.sample(dir, normal, t,  scene.material, mat_id)
                    brdf,pdfFwd        = Glass.evaluate_pdf(normal, -dir, next_dir,  scene.material, mat_id)
                    eye.delta[i, j, depth] = 1
                else:
                    next_dir, f_or_b   = Disney.sample(dir, fnormal,scene.material, mat_id)
                    brdf,pdfFwd        = Disney.evaluate_pdf(fnormal,  -dir, next_dir, scene.material, mat_id)
                    eye.delta[i, j, depth] = 0

                if pdfFwd > 0.0:
                    
                    if mat_type == SCD.MAT_GLASS:
                        pdfRev             = 0.0
                        pdfFwd             = 0.0
                        beta              *=  brdf * reflect_color
                    else:
                        beta              *=  brdf * reflect_color  * abs(normal.dot(next_dir)) / pdfFwd
                        pdfRev         = Disney.pdf(fnormal,  next_dir, -dir,   scene.material, mat_id)
                    eye.rpdf[i, j, pre_depth]  = pdfRev * abs(to.dot(eye.normal[i, j, depth])) * inv_dist2

                    if f_or_b < 0.0:
                        extinction      = UF.get_material_extinction(scene.material, mat_id)
                        R               = ti.exp(-t / extinction) 
                        if ti.random() >= R:
                            break

                    depth     += 1
                    pre_depth += 1
                    origin     = UF.offset_ray(pos , ts.sign(f_or_b)*fnormal)
                    dir        = next_dir


                else:
                    break
            else:
                break
        return depth

    @ti.func
    def light_path(self, i, j):
        light,scene               = ti.static(self.light, self.scene)
        light_pos,light_normal, light_dir,light_emission,light_prim,light_choice_pdf,light_dir_pdf= scene.sample_light()

        light_pdf                             = light_choice_pdf
        light.pos[i, j, 0]                    = light_pos
        light.normal[i, j, 0]                 = light_normal
        light.beta[i, j, 0]                   = light_emission / light_pdf
        light.fpdf[i, j, 0]                   = light_pdf
        light.rpdf[i, j, 0]                   = 0.0
        light.wo[i, j, 0]                     = light_dir
        light.type[i, j, 0]                   = VERTEX_LIGHT

        pre_depth                             = 0
        depth                                 = 1
        pdfFwd                                = light_dir_pdf
        pdfRev                                = 0.0
        beta                                  = light_emission/light_pdf *abs(light_normal.dot(light_dir))
        origin                                = light_pos
        dir                                   = light_dir


        while(depth < LIGHT_MAX_DEPTH):
            t,pos,gnormal,normal,tex,prim_id          = scene.closet_hit(origin, dir, self.stack, i,j, self.stack_size)
            fnormal                           = UF.faceforward(normal, -dir, gnormal)
            mat_id                            = UF.get_prim_mindex(scene.primitive, prim_id)
            mat_color                         = UF.get_material_color(scene.material, mat_id)
            mat_type                          = UF.get_material_type(scene.material, mat_id)

            if t < UF.INF_VALUE:
                
                if mat_type == SCD.MAT_LIGHT:
                    break
                else:              
                    light.pos[i, j, depth]       = pos
                    light.normal[i, j, depth]    = normal
                    light.snormal[i, j, depth]   = fnormal
                    light.beta[i, j, depth]      = beta * abs(dir.dot(normal))
                    light.wo[i, j, depth]        = dir
                    light.fpdf[i, j, depth]      = pdfFwd
                    light.rpdf[i, j, depth]      = 0.0
                    light.type[i, j, depth]      = VERTEX_SURFACE
                    light.prim[i, j, depth]      = prim_id
                    light.mat[i, j, depth]       = mat_id
                to                                = pos  - light.pos[i, j, pre_depth] 
                dist                              = ts.length(to)
                inv_dist2                         = 1.0 / (dist*dist)
                to                                = to / dist
                light.fpdf[i, j, depth]          *= abs(to.dot(light.normal[i, j, pre_depth])) * inv_dist2

                next_dir                          = dir
                reflect_color                     = UF.srgb_to_lrgb(mat_color)
                brdf                              = 0.0
                f_or_b                            = 1.0
                if mat_type == SCD.MAT_GLASS:
                    next_dir,f_or_b    = Glass.sample(dir, normal, t,  scene.material, mat_id)
                    brdf,pdfFwd        = Glass.evaluate_pdf(normal,  -dir, next_dir,    scene.material, mat_id)
                    light.delta[i, j, depth] = 1
                else:
                    next_dir, f_or_b   = Disney.sample(dir, fnormal,scene.material, mat_id)
                    brdf,pdfFwd        = Disney.evaluate_pdf(fnormal, -dir, next_dir,     scene.material, mat_id)
                    light.delta[i, j, depth] = 0

                if pdfFwd > 0.0:
                   
                    if mat_type == SCD.MAT_GLASS:
                        pdfRev             = 0.0
                        pdfFwd             = 0.0
                        beta              *=  brdf * reflect_color
                    else:
                        beta              *=  brdf * reflect_color  * abs(normal.dot(next_dir)) /pdfFwd
                        pdfRev             = Disney.pdf(fnormal, next_dir,  -dir,     scene.material, mat_id)
                        

                    light.rpdf[i, j, pre_depth]  = pdfRev * abs(to.dot(light.normal[i, j, depth])) * inv_dist2

                    if f_or_b < 0.0:
                        extinction      = UF.get_material_extinction(scene.material, mat_id)
                        R               = ti.exp(-t / extinction) 
                        if ti.random() >= R:
                            break
                    
                    origin     = UF.offset_ray(pos , ts.sign(f_or_b)*fnormal)
                    dir        = next_dir

                    depth     += 1
                    pre_depth += 1
                else:
                    break
            else:
                break
        return depth






    @ti.func
    def mis_weight(self, i, j, e, l):
        sample,eye,light,scene,cam               = ti.static(self.sample,self.eye,self.light, self.scene,self.cam)
        weight_sum = 0.0

        #if ((l == 0)) & (i==273) &(j == 151):
        #if (l != 0) | (e >1):
        if (l+e !=2):
            #store origin data


            if l>0:
                self.ltemp.copy(0, light,i, j, l-1)
            if e>0:
                self.etemp.copy(0, eye,  i, j, e-1)
            if l>1:
                self.lminustemp.copy(0, light,i, j, l-2)
            if e>1:
                self.eminustemp.copy(0, eye,i, j, e-2)

            #change the path value
            #as we sampled light and camera vertex in some path, -> l= 1, e=1
            #so we should change some end point
            if l==1:
                light.copy(0, sample, i, j, 0) 
            elif e == 1:
                eye.copy(0, sample, i, j, 0) 

            if l > 0:
                light.delta[i,j,l-1] = 0
            if e > 0:
                eye.delta[i,j,e-1] = 0               

       
            if e > 0:
                if l == 0:
                    pdfPos            = 1.0 / scene.get_prim_area(eye.prim[i,j,e-1])
                    pdfChoice         = 1.0 / float(scene.light_count)
                    eye.rpdf[i,j,e-1] = pdfPos * pdfChoice

                elif l == 1:
                    if  eye.type[i,j,e-1] == VERTEX_SURFACE:
                        to   = eye.pos[i,j,e-1] - light.pos[i,j, 0]
                        dist = ts.length(to)
                        to   = to / dist

                        pdfDir = UF.CosineHemisphere_pdf(abs(to.dot(light.normal[i,j,0])))
                        LdotN  = abs(to.dot(light.normal[i,j,0]) )
                        eye.rpdf[i,j,e-1] =    pdfDir*LdotN / (dist * dist)

                    else:
                        eye.rpdf[i,j,e-1] = 1.0
                        print(11)

                else:
                    wi = light.pos[i,j,l-2] - light.pos[i,j,l-1]
                    wo = eye.pos[i,j,e-1] - light.pos[i,j,l-1]
                    dist = ts.length(wo)
                    wi  = wi.normalized()
                    wo  = wo.normalized()
                    
                    pdf = 1.0
                    mat_id = light.mat[i,j,l-1]
                    if mat_id == SCD.MAT_DISNEY:
                        pdf    = Disney.pdf(light.snormal[i, j, l-1], wi, wo,   scene.material, mat_id)

                    eye.rpdf[i,j,e-1] = pdf *  abs(light.normal[i, j, l-1].dot(wo))  / (dist * dist)
            if l > 0:
                if e>1:
                    if  eye.type[i,j,e-1] == VERTEX_SURFACE:
                        wi = eye.pos[i,j,e-2] - eye.pos[i,j,e-1]
                        wo = light.pos[i,j,l-1] - eye.pos[i,j,e-1]
                        dist = ts.length(wo)
                        wi  = wi.normalized()
                        wo  = wo.normalized()

                        pdf = 1.0
                        mat_id = eye.mat[i,j,e-1]
                        if mat_id == SCD.MAT_DISNEY:
                            pdf    = Disney.pdf(eye.snormal[i, j, e-1], wi, wo,   scene.material, mat_id)
                        light.rpdf[i,j,l-1] = pdf *abs(eye.normal[i, j, e-1].dot(wo)) / (dist * dist)
                    else:
                        light.rpdf[i,j,l-1] = 1.0
                        print(22)
                else:
                    to   = eye.pos[i,j, 0] - light.pos[i,j, l-1]
                    dist = ts.length(to)
                    to   = to / dist

                    normal = cam.get_optical_axis()
                    #pdfDir = to.dot(normal)
                    LdotN  = to.dot(normal)                    
                    light.rpdf[i,j,l-1] = LdotN / (dist * dist) 

            if e > 1:
                if l == 0:
                    to   = eye.pos[i,j,e-2] - eye.pos[i,j,e-1]
                    dist = ts.length(to)
                    to   = to / dist

                    pdfDir = UF.CosineHemisphere_pdf(abs(to.dot(eye.normal[i,j,e-1])))
                    LdotN  = to.dot(eye.normal[i,j,e-1])
                    eye.rpdf[i,j,e-2] = abs(pdfDir*LdotN )/ (dist*dist) 
                else:
                    if  eye.type[i,j,e-1] == VERTEX_SURFACE:
                        wi = light.pos[i,j,l-1] - eye.pos[i,j,e-1]
                        wo = eye.pos[i,j,e-2] - eye.pos[i,j,e-1]
                        dist = ts.length(wo)
                        wi  = wi.normalized()
                        wo  = wo.normalized()


                        mat_id = eye.mat[i,j,e-1]
                        pdf    = Disney.pdf(eye.snormal[i, j, e-1], wi, wo,   scene.material, mat_id)
                        eye.rpdf[i,j,e-2] = pdf / (dist * dist)
                        if eye.type[i,j,e-2] == VERTEX_SURFACE:
                            eye.rpdf[i,j,e-2] *= abs(eye.normal[i, j, e-1].dot(wo))                    
                    else:
                        eye.rpdf[i,j,e-2] = 1.0
                        print(33)
                    
            if l > 1:
                if eye.type[i,j,e-1] != VERTEX_LIGHT:
                    wi = eye.pos[i,j,e-1]   - light.pos[i,j,l-1]
                    wo = light.pos[i,j,l-2] - light.pos[i,j,l-1]
                    dist = ts.length(wo)
                    wi  = wi.normalized()
                    wo  = wo.normalized()

                    pdf = 1.0
                    mat_id = light.mat[i,j,l-1]
                    if mat_id == SCD.MAT_DISNEY:
                        pdf    = Disney.pdf(light.normal[i, j, l-1], wi, wo,   scene.material, mat_id)
                    light.rpdf[i,j,l-2] = pdf  / (dist * dist)  
                    if light.type[i,j,l-2] == VERTEX_SURFACE:
                        light.rpdf[i,j,l-2] *= abs(light.normal[i, j, l-1].dot(wo))                                         
                else :
                    light.rpdf[i,j,l-2] = 1.0
                    print(44)
 


            #print(eye.rpdf[i,j,e], eye.rpdf[i,j,e-1])
            #do mis calculate
            
            
            weight = 1.0
            k = e-1
            while k > 0:
                weight *= self.remap0(eye.rpdf[i,j,k]) / self.remap0(eye.fpdf[i,j,k]) 
                #print(i, j, k, e, l, weight, eye.fpdf[i,j,k], eye.rpdf[i,j,k])
                if (eye.delta[i,j,k] == 0) & (eye.delta[i,j,k-1] == 0):
                    weight_sum += weight
                k -=1
            
            
            weight = 1.0
            k = l-1
            while k >= 0:
                weight *= self.remap0(light.rpdf[i,j,k]) / self.remap0(light.fpdf[i,j,k]) 
                #weight *= 1.0 / self.remap0(light.fpdf[i,j,k]) 
                if (k ==0):
                    if (light.delta[i,j,k] == 0) :
                        weight_sum += weight
                else:
                    if (light.delta[i,j,k] == 0) & (light.delta[i,j,k-1]==0) :
                        weight_sum += weight
                k -=1   
            

            #give back the origin data
            light.copy(l-1, self.ltemp, i, j, 0)
            eye.copy(e-1,  self.etemp, i, j, 0)
            if l>0:
                light.copy(l-2,self.lminustemp, i, j, 0)
            if e>0:
                eye.copy(e-2,  self.eminustemp, i, j, 0)
        #return 1.0
        return 1.0 / (1.0 + weight_sum)

    @ti.func
    def connect_path(self, i, j, e, l):
        scene                     = ti.static(self.scene)
        cam                       = ti.static(self.cam)
        eye                       = ti.static(self.eye)
        light                     = ti.static(self.light)
        sample                    = ti.static(self.sample)

        #new sample vertex
        radiance  = ti.Vector([0.0,0.0,0.0])
        new_pos   = ti.Vector([i, j])
        misweight = 1.0
        if (l == 0) :
            radiance = radiance
            
            if eye.type[i, j, e-1] == VERTEX_LIGHT:
                radiance = eye.beta[i, j, e-1] 
            
        elif e == 1:
            radiance = radiance

            
            prim                  = light.prim[i, j, l-1]
            surface               = light.pos[i, j, l-1]
            new_pos, wi           = cam.get_image_point(surface)
            origin                = cam.get_ray_origin()

            mat_id       = light.mat[i, j, l-1]
            snormal      = light.snormal[i ,j, l-1]
            NdotL        = wi.dot(snormal)
            if (new_pos.x >= 0) & (light.delta[i,j,l-1] !=  1) & (NdotL < 0.0) & (light.type[i,j,l-1] ==  VERTEX_SURFACE):
                t, hit_prim  = scene.closet_hit_shadow(origin, wi, self.stack, i,j, self.stack_size)
                if (hit_prim == prim):
                    brdf,pdf         = Disney.evaluate_pdf(snormal,  -light.wo[i ,j, l-1], -wi, scene.material, mat_id)
                    if pdf > 0.0:
                        G                     = abs(NdotL )/(t*t)
                        radiance              = G  * light.beta[i ,j, l-1] *UF.srgb_to_lrgb(UF.get_material_color(scene.material, mat_id)) *brdf/pdf
                        sample.pos[i,j,0]     = cam.get_ray_origin()
                        sample.wo[i,j,0]      = wi
                        sample.type[i,j,0]    = VERTEX_LENS
                        sample.fpdf[i,j,0]    = 1.0
            
            
        elif l == 1:
            radiance = radiance

            
            prim               = eye.prim[i, j, e-1]
            surface            = UF.offset_ray(eye.pos[i, j, e-1] , eye.snormal[i, j, e-1])
            mat_id             = eye.mat[i, j, e-1]

            if (eye.delta[i,j,e-1] !=  1) :
                #light_pos,light_normal,light_emission,light_area,light_prim = scene.sample_light()
                light_pos,light_normal, wi,light_emission,light_dist,light_prim,light_choice_pdf,light_dir_pdf = scene.sample_li( surface)


                #wi                     = (surface - light_pos).normalized()
                NdotLl                 = wi.dot(light_normal)
                NdotLe                 = wi.dot(eye.snormal[i, j, e-1])
                
                t, shadow_prim = scene.closet_hit_shadow(surface, -wi, self.stack, i,j, self.stack_size)
                if (shadow_prim == light_prim) & (t > UF.EPS):
                    light_pdf             = light_choice_pdf 
                    wo                    = eye.wo[i ,j, e-1]
                    brdf,pdf              = Disney.evaluate_pdf(eye.snormal[i ,j, e-1],  -wo, -wi,   scene.material, mat_id)
                    #dir_pdf               = UF.CosineHemisphere_pdf(abs(NdotLl))
                    
                    if pdf > 0.0:
                        G                     = abs(NdotLe*NdotLl)/ (t*t)
                        radiance              = G * eye.beta[i ,j, e-1]*brdf/pdf *UF.srgb_to_lrgb(UF.get_material_color(scene.material, mat_id))*light_emission/light_pdf
                    #print(G , eye.beta[i ,j, e-1] , light_emission/light_pdf,  1.0, 1.0, brdf,pdf )

                    sample.pos[i,j,0]     = light_pos
                    sample.wo[i,j,0]      = wi
                    sample.type[i,j,0]    = VERTEX_LIGHT
                    sample.fpdf[i,j,0]    = light_pdf
                    sample.prim[i,j,0]    = light_prim
                    sample.normal[i,j,0]  = light_normal
                    sample.snormal[i,j,0]  = light_normal
            
        else:
            radiance = radiance
            
            if  (light.delta[i,j,l-1] !=  1) &  (eye.delta[i,j,e-1] !=  1) & (eye.type[i ,j, e-1] == VERTEX_SURFACE) & (light.type[i ,j, l-1] == VERTEX_SURFACE):
                primE        = eye.prim[i, j, e-1]
                primL        = light.prim[i, j, l-1]
                mat_idE      = eye.mat[i, j, e-1]
                mat_idL      = light.mat[i, j, l-1]

                surfaceE     = eye.pos[i, j, e-1]
                surfaceL     = light.pos[i, j, l-1]
                dir          = surfaceE - surfaceL
                dist         = ts.length(dir)
                dir          = dir / dist

                NdotLl = dir.dot(light.snormal[i, j, l-1])
                NdotLe = dir.dot(eye.snormal[i, j, e-1])

                t, shadow_prim    = scene.closet_hit_shadow(surfaceL, dir, self.stack, i,j, self.stack_size)
                if (shadow_prim  == primE) & (t > UF.EPS):
                    brdfL,lpdf   = Disney.evaluate_pdf(light.snormal[i, j, l-1],   -light.wo[i, j, l-1], dir,        scene.material, mat_idL)
                    brdfE,epdf   = Disney.evaluate_pdf(eye.snormal[i, j, e-1],     -eye.wo[i, j, e-1],-dir,    scene.material, mat_idE)
                    if (brdfL>0.0) & (brdfE >0.0):
                        G            = abs(NdotLe * NdotLl )  / (dist*dist)
                        radiance      =  G * eye.beta[i ,j, e-1] * light.beta[i ,j, l-1]  *brdfL/lpdf *brdfE/epdf * \
                            UF.srgb_to_lrgb(UF.get_material_color(scene.material, mat_idE)) * UF.srgb_to_lrgb(UF.get_material_color(scene.material, mat_idL))
                        #print(G , eye.beta[i ,j, e-1] , light.beta[i ,j, l-1]  ,brdfL,lpdf ,brdfE,epdf )
                        #print(epdf,lpdf)
            
        if (radiance[0] > 0.0) & (radiance[1] > 0.0) & (radiance[2] > 0.0):
            misweight = self.mis_weight(i, j, e, l)
        return radiance * misweight, new_pos


    @ti.kernel
    def render(self):
        for i,j in self.radiance:
            self.radiance[i,j] = ti.Vector([0.0, 0.0, 0.0])

            e = 0
            while e < EYE_MAX_DEPTH:
                self.eye.beta[i, j, e] = ti.Vector([0.0, 0.0, 0.0])
                self.eye.type[i, j, e] = VERTEX_NONE
                self.eye.fpdf[i, j, e] = 0.0
                self.eye.rpdf[i, j, e] = 0.0
                e +=1

            l = 0
            while l < LIGHT_MAX_DEPTH:
                self.light.beta[i, j, l] = ti.Vector([0.0, 0.0, 0.0])
                self.light.type[i, j, l] = VERTEX_NONE
                self.light.fpdf[i, j, l] = 0.0
                self.light.rpdf[i, j, l] = 0.0
                l +=1


        for i,j in self.radiance:
            eye_depth   = self.eye_path(i,j)
            light_depth = self.light_path(i,j)
            
            e = 1
            while e <= eye_depth:
                l = 0
                while l <= light_depth:
                    depth = l+e-2
                    if ( ((l==1)&(e==1)) | (depth<0) | (depth>MAX_DEPTH)):
                        l += 1
                        continue
                    
                    
                    '''
                    r_path, eye_new_pos  = self.connect_path(i,j,e,l)
                    self.radiance[i, j]        += r_path
                    '''

                    
                    r_path, eye_new_pos  = self.connect_path(i,j,e,l)
                    if e == 1:
                        self.radiance[eye_new_pos] += r_path 
                    else:
                        self.radiance[i, j]        += r_path
                    
                    l += 1
                e += 1

        for i,j in self.hdr:
            frame    = float(self.cam.frame_gpu[0])
            coff     = 1.0 / (frame + 1.0)
            self.hdr[i,j] = self.radiance[i, j] * coff  + self.hdr[i,j]  * (1.0 - coff)
            #if (i<=5) & (j>345) &(j <350) &(self.hdr[i,j].sum()> 0.0):
            #    print("color",i,j,self.hdr[i,j])
