import sys
import os
sys.path.append("accel")
sys.path.append("brdf")
sys.path.append("texture")
sys.path.append("spectrum")
sys.path.append("sky")
import taichi as ti
import time
import math
import numpy as np
import Camera as Camera
import Scene as Scene
import UtilsFunc as UF
import SceneData as SCD
import Texture as TX
import Rgb2Spec as RGB2SPEC
import Disney as Disney
import Glass as Glass
import Spectrum as Spec
import taichi_glsl as ts
import HeroSample as Hero
import Sky


MAX_DEPTH  = 10


@ti.data_oriented
class PathTrace:
    def __init__(self, imgSizeX, imgSizeY, cam, scene, stack_size):
        self.imgSizeX   = imgSizeX
        self.imgSizeY   = imgSizeY
        self.lambda_min   = 10000
        self.lambda_max   = 0
        self.lambda_range = 0
        self.size         = 0

        self.rgb_film   = ti.Vector.field(3, dtype=ti.f32)
        self.hdr        = ti.Vector.field(3, dtype=ti.f32)
        self.sensor     = ti.Vector.field(3, dtype=ti.f32)
        self.stack      = ti.field(dtype=ti.i32)

        self.d65        = Spec.Spectrum()
        self.white      = Spec.Spectrum()
        self.red        = Spec.Spectrum()
        self.green      = Spec.Spectrum()
        self.rgb2spec   = RGB2SPEC.Rgb2Spec()
        self.sky        = Sky.Sky(3.0, 0.5, 0.17)
        self.cam        = cam
        self.scene      = scene
        self.stack_size = stack_size

    
    def setup_data_cpu(self):
        Data   = []
        for line in open("spectrum/ciexyz31_1.csv", "r"):
            values = line.split(',', 4)
            x      = float(values[1])
            y      = float(values[2])
            z      = float(values[3])
            Lambda = float(values[0])
            Data.append(x)
            Data.append(y)
            Data.append(z)
            if self.size == 0:
                self.lambda_min = Lambda
            self.lambda_max = Lambda
            self.size += 1
        
        self.lambda_range = (self.lambda_max - self.lambda_min) / (self.size -1)
        self.data_np  = np.zeros(shape=(self.size, 3), dtype=np.float32)
        for i in range(self.size):
            for j in range(3):
                self.data_np[i, j] = Data[3*i+j]

        ti.root.dense(ti.i, (self.size) ).place(self.sensor)
        ti.root.dense(ti.ij, [self.imgSizeX, self.imgSizeY] ).place(self.rgb_film)
        ti.root.dense(ti.ij, [self.imgSizeX, self.imgSizeY] ).place(self.hdr)
        ti.root.dense(ti.ijk, [self.imgSizeX, self.imgSizeY, self.stack_size] ).place(self.stack)

        self.rgb2spec.load_table("spectrum/spec_table")
        self.d65.load_table("spectrum/Illuminantd65.csv")
        self.red.load_table( "spectrum/red-spec.csv")
        self.green.load_table("spectrum/green-spec.csv")
        self.white.load_table("spectrum/white-spec.csv")

    
    def setup_data_gpu(self):
        self.sensor.from_numpy(self.data_np)
        self.rgb2spec.setup_data_gpu()
        self.d65.setup_data_gpu()
        self.red.setup_data_gpu()
        self.green.setup_data_gpu()
        self.white.setup_data_gpu()
        self.sky.setup_data_gpu()

        self.normalize_spec(self.d65)

    
    def normalize_spec(self, spec):
        self.cal_white_point(spec)
        
        white_point_np = spec.white_point.to_numpy()
        coff           = 1.0 / float(white_point_np[0,1])
        spec.scale(coff)
        white_point_np = white_point_np*coff
        #print(white_point_np)

    @ti.func
    def emission_to_rad(self, emission, Lambda):
        scale = ts.length(emission)
        ret  = ti.Vector([0.0,0.0,0.0,0.0])
        if scale > 0.0:
            tint = emission / scale
            ret  = Hero.srgb_to_spec(self.rgb2spec, tint , Lambda)
        return ret * scale

    @ti.func
    def get_spec_power(self, scene:ti.template(), mat_id, Lambda):
        mat_type = UF.get_material_type(scene.material, mat_id)
        mat_tex  = UF.get_material_tex(scene.material,  mat_id)

        ret = ti.Vector([0.0, 0.0, 0.0, 0.0])
        if mat_type == SCD.MAT_SPECTRAL:
            if mat_tex == 0:
                ret =  Hero.sample(self.white, Lambda)
            if mat_tex == 1:
                ret =  Hero.sample(self.red, Lambda)
            if mat_tex == 2:
                ret =  Hero.sample(self.green, Lambda)
        else:
            mat_color    = UF.get_material_color(scene.material, mat_id)
            ret          = Hero.srgb_to_spec(self.rgb2spec, mat_color , Lambda)
        return ret


    @ti.func
    def sample(self, Lambda):
        ret = ti.Vector([0.0,0.0,0.0])
        if  (Lambda>= self.lambda_min) & (Lambda <= self.lambda_max):
            offset = Lambda - self.lambda_min
            idx    = int (offset / self.lambda_range)
            w      = ts.fract(offset)
            ret    = ts.mix(self.sensor[idx], self.sensor[idx+1], w)
        return ret

    @ti.func
    def AddSplat(self, spec, i, j, Lambda0, coff):
        #Lambda0 = 448.107513
        #spec    = ti.Vector([0.0151173174,0.0137298098, 0.0148762511,0.0121297557])
        x_flux,y_flux,z_flux = Hero.sample_xyz(self, Lambda0)
        x_flux              *= spec
        y_flux              *= spec
        z_flux              *= spec

        #kind of monte carlo
        x_integrate    = x_flux * (self.lambda_max - self.lambda_min) / Hero.SAMPLE_WAVELENGTHS
        y_integrate    = y_flux * (self.lambda_max - self.lambda_min) / Hero.SAMPLE_WAVELENGTHS
        z_integrate    = z_flux * (self.lambda_max - self.lambda_min) / Hero.SAMPLE_WAVELENGTHS

        xyz            = ti.Vector([x_integrate.sum(), y_integrate.sum(), z_integrate.sum()])
        lrgb           = UF.xyz_to_srgb@xyz
        
        self.hdr[i,j]  = ts.mix(self.hdr[i,j], lrgb, coff)
        #print(x_flux,y_flux,z_flux)

    #@ti.kernel
    #def tone_map_xyz(self, exposure:ti.f32, spec:ti.template()):
    #    for i,j in self.hdr:
    #        self.img[i,j] = UF.lrgb_to_srgb(UF.tone_ACES(spec.xyz_to_lrgb[0]@self.hdr[i,j]* exposure) )
    #        #self.img[i,j] = UF.lrgb_to_srgb(UF.tone_ACES(self.xyz_to_lrgb@self.hdr[i,j]* exposure) )

    @ti.kernel
    def cal_white_point(self, spec:ti.template()):
        for i in self.sensor :
            Lambda = self.lambda_min + float(i)*self.lambda_range
            
            h      = float(self.lambda_max - self.lambda_min) / float(self.size - 1)
            weight = 3.0 / 8.0 * h
            if (i ==0) | (i == self.size-1):
                weight = weight
            elif ((i-1)%3 == 2):
                weight = weight * 2.0
            else:
                weight = weight * 3.0
            spec.white_point[0] += self.sensor[i] * spec.sample(Lambda) * weight

    @ti.kernel
    def render(self):
        cam  = self.cam 
        scene = self.scene
        for i,j in self.rgb_film:

            next_origin     = cam.get_ray_origin()
            next_dir        = cam.get_ray_direction(i,j)
            Lambda          = Hero.LAMBDA_MIN + Hero.LAMBDA_STEP * ti.random()

            depth           = 0
            light_pdf       = 1.0
            brdf_pdf        = 1.0
            f_or_b          = 1.0
            brdf            = 1.0
            throughout      = ti.Vector([1.0, 1.0, 1.0, 1.0])
            radiance        = ti.Vector([0.0, 0.0, 0.0, 0.0])
            
            while(depth < MAX_DEPTH):
                origin    = next_origin 
                direction = next_dir

                #t,pos,normal,tex,prim_id          = scene.closet_hit(origin,direction, self.stack, i,j, self.stack_size)
                t,pos,gnormal,normal,tex,prim_id  = scene.closet_hit(origin,direction, self.stack, i,j, self.stack_size)
                fnormal                   = UF.faceforward(normal, -direction, gnormal)
                mat_id                    = UF.get_prim_mindex(scene.primitive, prim_id)
                mat_color                 = UF.get_material_color(scene.material, mat_id)
                mat_type                  = UF.get_material_type(scene.material, mat_id)
                light_rad                 = Hero.sample(self.d65, Lambda)
                light_tint                = self.emission_to_rad(mat_color, Lambda)
                perfect_spec              = 1
                if t < UF.INF_VALUE:
                    if mat_type == SCD.MAT_LIGHT:
                        fCosTheta  = direction.dot(normal)
                        
                        if fCosTheta < 0.0:
                            area = scene.get_prim_area(prim_id)
                            light_pdf = (t * t) / (area * fCosTheta)

                            if perfect_spec == 1 :
                                radiance += throughout  * light_rad*light_tint
                            else:
                                radiance += UF.powerHeuristic(brdf_pdf, light_pdf) * throughout  * light_rad*light_tint
                        break
                    else:
                        next_origin    = UF.offset_ray(pos , fnormal)

                        reflect_spec   = self.get_spec_power(scene, mat_id, Lambda)

                        #mat_type = 1
                        #btdf
                        if mat_type == SCD.MAT_GLASS:
                            perfect_spec       = 1
                            index,rnd_lanmda   = Hero.get_rnd_hero(Lambda)
                            next_dir,f_or_b    = Glass.sample_lambda(direction, normal, t, scene.material, mat_id, rnd_lanmda)
                            brdf,brdf_pdf      = Glass.evaluate_pdf(normal,  next_dir, -direction,scene.material, mat_id)
                        else:
                            perfect_spec           = 0
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
                                        radiance    += UF.powerHeuristic(light_pdf, brdf_pdf ) / max(0.0001, light_pdf)*light_rad*light_tint  *throughout  *reflect_spec*brdf*abs(NdotL_surface)
                        
                            next_dir, f_or_b    = Disney.sample(direction, fnormal, scene.material, mat_id)
                            brdf,brdf_pdf       = Disney.evaluate_pdf(fnormal,  next_dir, -direction, scene.material, mat_id)
                            brdf               *= abs(normal.dot(next_dir))

                        next_origin   = UF.offset_ray(pos , ts.sign(f_or_b)*fnormal)

                        if (brdf_pdf > 0.0) & (UF.max_component(throughout) > 0.0):
                            throughout     *=  brdf * reflect_spec / brdf_pdf
                            depth     += 1
                        else:
                            break
                else:
                    dis = ti.sqrt(direction.x*direction.x + direction.z*direction.z)
                    beta         = ts.atan(direction.y , dis)
                    gamma        = ts.acos(direction.dot(self.sky.sun_dir[0]))
                    theta        = ts.clamp(0.5 * 3.1415926 - beta, 0.0, 0.5 * 3.1415926  )
                    ibl_emission = Hero.sky_sample(self.sky, theta, gamma, Lambda)
                    radiance    += throughout * ibl_emission * light_rad
                    break
             
            frame = float(cam.frame_gpu[0])
            self.AddSplat(radiance, i, j, Lambda, 1.0/(frame+1.0))