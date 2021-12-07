import sys
import os
sys.path.append("accel")
sys.path.append("texture")
import taichi as ti
import math
import numpy as np
import pywavefront
import trimesh
import SceneData as SCD
import UtilsFunc as UF
import taichi_glsl as ts
import LBvh as LBvh
import SahBvh as SBvh

import Texture as TX
import queue

MAX_STACK_SIZE =  32
@ti.data_oriented
class Scene:
    def __init__(self):
        self.maxboundarynp           = np.ones(shape=(1,3), dtype=np.float32)
        self.minboundarynp           = np.ones(shape=(1,3), dtype=np.float32)
        for i in range(3):
            self.maxboundarynp[0, i] = -UF.INF_VALUE
            self.minboundarynp[0, i] = UF.INF_VALUE

        self.light_cpu               = []
        self.material_cpu            = []
        self.vertex_cpu              = []
        self.vertex_index_cpu        = []
        self.shape_cpu               = []
        self.primitive_cpu           = []

        self.material                = ti.Vector.field(SCD.MAT_VEC_SIZE, dtype=ti.f32)
        self.vertex                  = ti.Vector.field(SCD.VER_VEC_SIZE, dtype=ti.f32)
        self.vertex_index            = ti.field(dtype=ti.i32)
        self.smooth_normal           = ti.Vector.field(3,  dtype=ti.f32)
        self.stack                   = ti.field( dtype=ti.i32)

        self.primitive               = ti.Vector.field(SCD.PRI_VEC_SIZE, dtype=ti.i32)
        self.shape                   = ti.Vector.field(SCD.SHA_VEC_SIZE, dtype=ti.f32)
        self.light                   = ti.field(dtype=ti.i32)
        self.light_area              = ti.field(dtype=ti.f32)      
                 
        self.material_count          = 0
        self.vertex_count            = 0
        self.primitive_count         = 0
        self.shape_count             = 0
        self.light_count             = 0
        self.vertex_max_index        = 0

        self.env                     = TX.Texture()
        self.env_power               = 0.0
    ########################host function#####################################

    @ti.pyfunc
    def add_obj(self, filename):
        '''
        mesh = trimesh.load(filename, force='mesh')
        scene = mesh.scene()
        visual = mesh.visual()
        '''
        
        scene = pywavefront.Wavefront(filename)
        scene.parse() 

        for name in scene.materials:
            ###process mat##
            material          = SCD.Material()
            if (scene.materials[name].emissive[0]> 1.0) & (scene.materials[name].emissive[1] > 1.0) & (scene.materials[name].emissive[2] > 1.0):
                material.type     = SCD.MAT_LIGHT
                material.setColor(scene.materials[name].emissive)
            elif scene.materials[name].transparency > 0.99:
                material.type     = SCD.MAT_DISNEY
                material.setMetal(0.0)
                material.setRough(0.5)
                material.setColor(scene.materials[name].diffuse)
            else:
                material.type     = SCD.MAT_GLASS
                material.setIor(scene.materials[name].optical_density)
                material.setExtinciton(scene.materials[name].shininess)
                material.setColor(scene.materials[name].diffuse)

            if scene.materials[name].texture != None:
                material.alebdoTex = float(scene.materials[name].texture)
            else:
                material.alebdoTex = -1
            self.material_cpu.append(material)


            ######process vert#########
            num_vert = len(scene.materials[name].vertices)
            v_format = scene.materials[name].vertex_format
            
            inner_index = 0
            while inner_index < num_vert:
                
                vertex   = SCD.Vertex()
                if v_format == 'T2F_V3F':
                    vertex.setPos(scene.materials[name].vertices, inner_index + 2)
                    vertex.setTex(scene.materials[name].vertices, inner_index + 0)
                    inner_index += 5
                    

                if v_format == 'T2F_N3F_V3F':
                    vertex.setPos(scene.materials[name].vertices, inner_index + 5)
                    vertex.setNormal(scene.materials[name].vertices, inner_index + 2)
                    vertex.setTex(scene.materials[name].vertices, inner_index + 0)
                    inner_index += 8

                if v_format == 'N3F_V3F':
                    vertex.setPos(scene.materials[name].vertices, inner_index + 3)
                    vertex.setNormal(scene.materials[name].vertices, inner_index + 0)
                    inner_index += 6

                if v_format== 'V3F':
                    vertex.setPos(scene.materials[name].vertices, inner_index + 0) 
                    inner_index += 3   

                for k in range(3):
                    self.maxboundarynp[0, k]   = max(vertex.pos[k], self.maxboundarynp[0, k])
                    self.minboundarynp[0, k]   = min(vertex.pos[k], self.minboundarynp[0, k])

                self.vertex_count += 1
                self.vertex_cpu.append(vertex)
                self.vertex_index_cpu.append(self.primitive_count)
                ######process triangle#########
                if self.vertex_count % 3  == 0 :
                    primitive                    = SCD.Primitive()
                    primitive.type               = SCD.PRIMITIVE_TRI
                    primitive.vertex_shape_index = self.vertex_count-3
                    primitive.mat_index          = self.material_count
                    self.primitive_cpu.append(primitive)
                    if material.type  == SCD.MAT_LIGHT:
                        self.light_cpu.append(self.primitive_count)
                        self.light_count += 1

                    self.primitive_count += 1
            self.material_count += 1




    @ti.pyfunc
    def cross(self, left, right):
        return [left[1] * right[2] - left[2] * right[1] , left[2] * right[0]  - left[0] *right[2] , left[0] * right[1]  - left[1] * right[0] ]

    @ti.pyfunc
    def length(self, v):
        return math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]) 

    @ti.pyfunc
    def minus(self, left, right):
        return [left[0]-right[0], left[1]-right[1], left[2]-right[2]]

    @ti.pyfunc
    def add(self, left, right):
        return [left[0]+right[0], left[1]+right[1], left[2]+right[2]]


    @ti.pyfunc
    def mul(self, left, right):
        return [left[0]*right[0], left[1]*right[1], left[2]*right[2]]


    @ti.pyfunc
    def cal_normal(self):
        for i in range(0, self.vertex_count, 3):
            if self.length(self.vertex_cpu[i].normal) == 0.0:
                v12 = self.minus(self.vertex_cpu[i+1].pos , self.vertex_cpu[i].pos)
                v13 = self.minus(self.vertex_cpu[i+2].pos , self.vertex_cpu[i].pos)
                n   = self.cross(v12, v13)

                l   = 1.0 / self.length(n)
                self.vertex_cpu[i].normal    = self.mul(n, [l,l,l])
                self.vertex_cpu[i+1].normal  = self.vertex_cpu[i].normal
                self.vertex_cpu[i+2].normal  = self.vertex_cpu[i].normal
            #print(i, "a", self.vertex_cpu[i].normal)

    @ti.pyfunc
    def add_env(self, filename, env_power):
        self.env.load_image("image/env.png")
        self.env_power  = env_power

    @ti.pyfunc
    def add_shape(self, shape, mat):
        if mat.type == SCD.MAT_LIGHT:
            self.light_cpu.append(self.primitive_count)
            self.light_count += 1

        primitive                    = SCD.Primitive()
        primitive.type               = SCD.PRIMITIVE_SHAPE
        primitive.mat_index          = self.material_count
        primitive.vertex_shape_index = self.shape_count

        self.primitive_cpu.append(primitive)
        self.primitive_count += 1

        self.shape_cpu.append(shape)
        self.shape_count += 1

        self.material_cpu.append(mat)
        self.material_count += 1


    @ti.pyfunc
    def write_data_debug(self):
        filename = "debug.obj"
        fo = open(filename, "w")
        
        vertex = self.vertex.to_numpy()
        for i in range(self.vertex_count):
            print ("v %f %f %f" %   (vertex[i, 0], vertex[i,1], vertex[i,2]), file = fo)
            print ("vn %f %f %f" %  (vertex[i, 3], vertex[i,4], vertex[i,5]), file = fo)
        
        for i in range(self.vertex_count//3):
            print ("f %d\%d %d\%d %d\%d " %  (3*i+1, 3*i+1, 3*i+2, 3*i+2, 3*i+3, 3*i+3), file = fo)
        fo.close()

    @ti.pyfunc
    def setup_data_cpu(self):
        print("***************mat:%d*******************"%(self.material_count))
        self.material_np  = np.zeros(shape=(self.material_count, SCD.MAT_VEC_SIZE), dtype=np.float32)
        for i in range(self.material_count):
            self.material_cpu[i].fillStruct(self.material_np, i)
            print("mat %d:"%(i), self.material_np[i,:])
        print("**************************************")

        print("***************vertex:%d*******************"%(self.vertex_count))
        self.cal_normal()
        self.vertex_np        = np.zeros(shape=(self.vertex_count, SCD.VER_VEC_SIZE), dtype=np.float32)
        self.smooth_normal_np = np.zeros(shape=(self.vertex_count, 3), dtype=np.float32)
        self.vertex_index_np  = np.zeros(shape=(self.vertex_count), dtype=np.int32)

        for i in range(self.vertex_count):
            self.vertex_cpu[i].fillStruct(self.vertex_np, i)
            self.vertex_index_np[i] = self.vertex_index_cpu[i]
            #print("ver %d:"%(i), self.vertex_np[i,:])
        print("**************************************")


        #self.primitive_bit = 0
        print("***************prim:%d *******************"%(self.primitive_count))
        self.primitive_np  = np.zeros(shape=(self.primitive_count, SCD.PRI_VEC_SIZE), dtype=np.int32)
        for i in range(self.primitive_count):
            self.primitive_cpu[i].fillStruct(self.primitive_np, i)
            #print("tri %d:"%(i), self.primitive_np[i,:])
        print("**************************************")

        print("***************light:%d*******************"%(self.light_count))
        if self.light_count > 0:
            self.light_np  = np.zeros(shape=(self.light_count), dtype=np.int32)
            for i in range(self.light_count):
                self.light_np[i] = self.light_cpu[i]
                #print("light %d:"%(i), self.light_np[i])
            ti.root.dense(ti.i, self.light_count     ).place(self.light)
        else:
            self.light_np  = np.zeros(shape=(1), dtype=np.float32)
            ti.root.dense(ti.i, 1     ).place(self.light)
        print("**************************************")

        print("***************shape:%d*******************"%(self.shape_count))
        if self.shape_count > 0:
            self.shape_np  = np.zeros(shape=(self.shape_count, SCD.SHA_VEC_SIZE), dtype=np.float32)
            for i in range(self.shape_count):
                self.shape_cpu[i].fillStruct(self.shape_np, i)
                print("shape %d:"%(i), self.shape_np[i,:])
            ti.root.dense(ti.i, self.shape_count     ).place(self.shape)
        else:
            self.shape_np  = np.zeros(shape=(1, SCD.SHA_VEC_SIZE), dtype=np.float32)
            ti.root.dense(ti.i, 1).place(self.shape)
        print("**************************************")

        print("***************bounding***********************")
        print(self.minboundarynp, self.maxboundarynp)
        print("**************************************")
   
        ti.root.dense(ti.i, 1  ).place(self.light_area) 
        ti.root.dense(ti.i, self.material_count  ).place(self.material)
        ti.root.dense(ti.i, self.vertex_count    ).place(self.vertex)
        ti.root.dense(ti.i, self.primitive_count ).place(self.primitive)

        ti.root.dense(ti.i, self.vertex_count    ).place(self.vertex_index)
        ti.root.dense(ti.i, self.vertex_count    ).place(self.smooth_normal )
        ti.root.dense(ti.ij, [self.vertex_count,MAX_STACK_SIZE] ).place(self.stack  )
        
        self.bvh = LBvh.Bvh(self.primitive_count, self.minboundarynp, self.maxboundarynp)
        self.bvh.setup_data_cpu()

        #self.bvh = SBvh.Bvh(self.primitive_count, self.minboundarynp, self.maxboundarynp)
        #self.bvh.setup_data_cpu(self.vertex_cpu, self.shape_cpu, self.primitive_cpu)

        if self.env_power == 0.0:
            self.env.load_image("image/black.png")

    @ti.pyfunc
    def setup_data_gpu(self):
        self.shape.from_numpy(self.shape_np)
        self.light.from_numpy(self.light_np)
        self.material.from_numpy(self.material_np)
        self.vertex.from_numpy(self.vertex_np)
        self.primitive.from_numpy(self.primitive_np)

        self.vertex_index.from_numpy(self.vertex_index_np)
        self.smooth_normal.from_numpy(self.smooth_normal_np)

        self.env.setup_data_gpu()
        self.bvh.setup_data_gpu(self.vertex, self.shape, self.primitive)



    ############algrithm##############
    @ti.func
    def UniformSampleSphere(self, u1,  u2):
        z = 1.0 - 2.0 * u1
        r = ti.sqrt(ts.clamp(1.0 - z * z, 0.0, 1.0))
        phi = 2.0 * 3.1415926 * u2
        x = r * ti.cos(phi)
        y = r * ti.sin(phi)
        return ti.Vector([x, y, z])

    @ti.func
    def get_prim_area(self, index):
        ret       = 0.0
        prim_type = UF.get_prim_type(self.primitive,index)
        if prim_type == SCD.PRIMITIVE_TRI:
            ver_index  = UF.get_prim_vindex(self.primitive, index)
            v1 = UF.get_vertex_pos(self.vertex, ver_index+0)
            v2 = UF.get_vertex_pos(self.vertex, ver_index+1)
            v3 = UF.get_vertex_pos(self.vertex, ver_index+2)
            a  = (v1-v2).norm()
            b  = (v1-v3).norm()
            c  = (v3-v2).norm()
            sum = (a+b+c) * 0.5
            ret = ti.sqrt(sum*(sum-a)*(sum-b)*(sum-c))
        else:
            sha_id   = UF.get_prim_vindex(self.primitive, index) 
            sha_type = UF.get_shape_type(self.shape, sha_id)
            if sha_type == SCD.SHPAE_SPHERE:
                r      = UF.get_shape_radius(self.shape, sha_id )
                ret = r*r*3.1415926
            elif sha_type == SCD.SHPAE_SPOT:
                r      = UF.get_shape_radius(self.shape, sha_id )
                ret = r*r*3.1415926
            elif sha_type == SCD.SHPAE_LASER:
                r      = UF.get_shape_radius(self.shape, sha_id )
                ret = r*r*3.1415926
        return ret


    @ti.func
    def get_prim_angle(self, index, v):
        ret       = 0.0
        prim_type = UF.get_prim_type(self.primitive,index)

        
        if prim_type == SCD.PRIMITIVE_TRI:
            ver_index  = UF.get_prim_vindex(self.primitive, index)
            v1 = UF.get_vertex_pos(self.vertex, ver_index+0)
            v2 = UF.get_vertex_pos(self.vertex, ver_index+1)
            v3 = UF.get_vertex_pos(self.vertex, ver_index+2)

            if ts.length(v1-v) < 0.00001:
                v12 = (v2 - v1).normalized()
                v13 = (v3 - v1).normalized()
                ret = v12.dot(v13)
            elif ts.length(v2-v) < 0.00001:
                v21 = (v1 - v2).normalized()
                v23 = (v3 - v2).normalized()
                ret = v21.dot(v23) 
            else:
                v31 = (v1 - v3).normalized()
                v32 = (v2 - v3).normalized()
                ret = v31.dot(v32)
        return ts.acos(ret)


    #https://www.zhihu.com/question/31706710?sort=created
    @ti.func
    def get_prim_random_point_normal(self, index):

        a = ti.random() 
        b = ti.random() 
        pos       = ti.Vector([0.0, 0.0, 0.0])
        normal    = ti.Vector([0.0, 0.0, 0.0])
        prim_type = UF.get_prim_type(self.primitive,index)
        if prim_type == SCD.PRIMITIVE_TRI:
            ver_index  = UF.get_prim_vindex(self.primitive, index)
            v1 = UF.get_vertex_pos(self.vertex, ver_index+0)
            v2 = UF.get_vertex_pos(self.vertex, ver_index+1)
            v3 = UF.get_vertex_pos(self.vertex, ver_index+2)
            n1 = UF.get_vertex_normal(self.vertex, ver_index+0)
            n2 = UF.get_vertex_normal(self.vertex, ver_index+1)
            n3 = UF.get_vertex_normal(self.vertex, ver_index+2)

            if(a+b>1.0):
                a = 1.0 - a
                b = 1.0 - b
            pos    = v1 + (v3-v1)*a + (v2-v1)*b 
            normal = ((1.0-a-b)*n1 + n2*a + n3*b).normalized()

        else:
            sha_id   = UF.get_prim_vindex(self.primitive, index) 
            sha_type = UF.get_shape_type(self.shape, sha_id)
            if sha_type == SCD.SHPAE_SPHERE:
                r      = UF.get_shape_radius(self.shape, sha_id )
                centre = UF.get_shape_pos(self.shape, sha_id )
        
                normal = self.UniformSampleSphere(a, b) 
                pos    = centre + normal * r
            elif sha_type == SCD.SHPAE_SPOT:
                normal = UF.get_shape_normal(self.shape, sha_id )
                pos    = UF.get_shape_pos(self.shape, sha_id )
            elif sha_type == SCD.SHPAE_LASER:
                normal = UF.get_shape_normal(self.shape, sha_id )
                pos    = UF.get_shape_pos(self.shape, sha_id )

        return pos,normal.normalized()


    @ti.func
    def get_random_light_prim_index(self):
        index = (int)(ti.random()  * self.light_count)
        if index >= self.light_count:
            index= self.light_count -1
        return self.light[index]

    @ti.func
    def sample_light(self):
        prim_index             = self.get_random_light_prim_index()
        light_pos,light_normal = self.get_prim_random_point_normal(prim_index)
        mat_id                 = UF.get_prim_mindex(self.primitive, prim_index)
        light_emission         = UF.get_material_color(self.material, mat_id)
        light_area             = self.get_prim_area(prim_index)
        light_choice_pdf       = 1.0 / (self.light_count * light_area)
        light_normal           = light_normal.normalized()

        light_dir,light_dir_pdf = UF.CosineSampleHemisphere_pdf(ti.random(), ti.random())
        light_dir               = UF.inverse_transform(light_dir, light_normal)

        

        prim_type = UF.get_prim_type(self.primitive,prim_index)
        if prim_type != SCD.PRIMITIVE_TRI:
            sha_id   = UF.get_prim_vindex(self.primitive, prim_index) 
            sha_type = UF.get_shape_type(self.shape, sha_id)
            if sha_type == SCD.SHPAE_SPOT:
                scale = UF.get_shape_scale(self.shape, sha_id)
                light_dir_pdf = 1.0
                r,phi         = UF.mapToDisk(ti.random(), ti.random())
                x1,x2         = UF.get_shape_xita(self.shape, sha_id)
                r1            = scale *ts.tan(x1)
                r2            = scale *ts.tan(x2)
                r            *= r2
                
                if r > r1 :
                    light_emission *= 1.0 - (r - r1) / (r2-r1)
                sampe_point   = ti.Vector([r*ts.cos(phi), r*ts.sin(phi), ti.sqrt(max(0.0, scale* scale - r*r))])
                light_dir     = UF.inverse_transform(sampe_point, light_normal)  

            elif sha_type == SCD.SHPAE_LASER:
                light_choice_pdf = 1.0 / (self.light_count)
                r             = UF.get_shape_radius(self.shape, sha_id)
                phi           = ti.random() * UF.M_PIf * 2.0
                sampe_point   = ti.Vector([r*ts.cos(phi), r*ts.sin(phi), 0.0])
                sampe_point   = UF.inverse_transform(sampe_point, light_normal) 

                light_dir     = light_normal 
                light_dir_pdf = 1.0
                light_pos    +=     sampe_point
                                
        return light_pos,light_normal, light_dir,light_emission,prim_index,light_choice_pdf,light_dir_pdf


    @ti.func
    def sample_li(self, pos):
        
        prim_index              = self.get_random_light_prim_index()
        light_pos,light_normal  = self.get_prim_random_point_normal(prim_index)
        mat_id                  = UF.get_prim_mindex(self.primitive, prim_index)
        light_emission          = UF.get_material_color(self.material, mat_id)
        light_area              = self.get_prim_area(prim_index)
        light_choice_pdf        = 1.0 / (float(self.light_count)*light_area)
        light_normal            = light_normal.normalized()
        light_dir               = pos - light_pos
        light_dist              = ts.length(light_dir)
        light_dir               = light_dir / light_dist
        NdotL                   = abs(light_dir.dot(light_normal))
        light_dir_pdf           = UF.CosineHemisphere_pdf(NdotL)
        visable                 = 1.0


        prim_type = UF.get_prim_type(self.primitive,prim_index)
        if prim_type != SCD.PRIMITIVE_TRI:
            sha_id   = UF.get_prim_vindex(self.primitive, prim_index) 
            sha_type = UF.get_shape_type(self.shape, sha_id)
            if sha_type == SCD.SHPAE_SPOT:

                light_dir_pdf = 1.0
                x1,x2         = UF.get_shape_xita(self.shape, sha_id)
                x             = ts.acos(NdotL)
                if x > x2 :
                    visable = 0.0
                elif x > x1:
                    visable *= 1.0 - (x-x1)/(x2-x1)  
            elif sha_type   == SCD.SHPAE_LASER:  
                light_choice_pdf        = 1.0 / (float(self.light_count))
                proj        = light_dir.dot(light_normal)*light_dist
                r           = ti.sqrt(light_dist*light_dist - proj*proj)
                limit_r     = UF.get_shape_radius(self.shape, sha_id)   
                if   (r >  limit_r):
                    visable = 0.0
                light_dir_pdf = 1.0
                #print(limit_r, r)
                   
        return light_pos,light_normal, light_dir,light_emission*visable, light_dist, prim_index,light_choice_pdf,light_dir_pdf 


    @ti.func
    def is_speculr(self, mat_id):
        ret         = 0
        mat_type    = UF.get_material_type(self.material, mat_id)
        if mat_type == SCD.MAT_GLASS:
            ret     = 1
        return ret

    @ti.func
    def intersect_prim(self, origin, direction, primitive_id):
        prim_type = UF.get_prim_type(self.primitive,primitive_id)
        hit_t     = UF.INF_VALUE
        hit_pos   = ti.Vector([0.0, 0.0, 0.0])
        hit_nor   = ti.Vector([0.0, 0.0, 0.0])
        hit_tex   = ti.Vector([0.0, 0.0, 0.0])
        hit_gnor  = ti.Vector([0.0, 0.0, 0.0])
        if prim_type == SCD.PRIMITIVE_TRI:
            hit_t, u,v = self.intersect_tri(origin, direction, primitive_id)
            if hit_t < UF.INF_VALUE:
                ver_index  = UF.get_prim_vindex(self.primitive, primitive_id)
                
                a = 1.0 - u-v
                b = u
                c = v

                v1 = UF.get_vertex_pos(self.vertex, ver_index+0)
                v2 = UF.get_vertex_pos(self.vertex, ver_index+1)
                v3 = UF.get_vertex_pos(self.vertex, ver_index+2)
                n1 = UF.get_vertex_normal(self.vertex, ver_index+0)
                n2 = UF.get_vertex_normal(self.vertex, ver_index+1)
                n3 = UF.get_vertex_normal(self.vertex, ver_index+2)  
                t1 = UF.get_vertex_uv(self.vertex, ver_index+0)
                t2 = UF.get_vertex_uv(self.vertex, ver_index+1)
                t3 = UF.get_vertex_uv(self.vertex, ver_index+2)   
                
                v13 = (v3-v1)
                v12 = (v2-v1)
                hit_gnor = v12.cross(v13)
                hit_pos  = a*v1 + b*v2 + c*v3
                hit_tex  = a*t1 + b*t2 + c*t3
                hit_nor  = a*n1 + b*n2 + c*n3                   
        else:
            sha_id   = UF.get_prim_vindex(self.primitive, primitive_id) 
            sha_type = UF.get_shape_type(self.shape, sha_id)
            if sha_type == SCD.SHPAE_SPHERE:
                #   h1    h2          -->two hitpoint
                # o--*--p--*--->d     -->Ray
                #   \   |
                #    \  |
                #     \ |
                #      c              -->circle centre
                r      = UF.get_shape_radius(self.shape, sha_id )
                centre = UF.get_shape_pos(self.shape, sha_id )
                oc     = centre - origin
                dis_oc_square = oc.dot(oc)
                dis_op        = direction.dot (oc)
                dis_cp        = ti.sqrt(dis_oc_square - dis_op * dis_op)
                if (dis_cp < r):
                        # h1 is nearer than h2
                        # because h1 = o + t*d
                        # so  |ch| = radius = |c - d - t*d| = |oc - td|
                        # so  radius*radius = (oc - td)*(oc -td) = oc*oc +t*t*d*d -2*t*(oc*d)
                        #so d*d*t^2   -2*(oc*d)* t + (oc*oc- radius*radius) = 0

                        #cal ax^2+bx+c = 0
                        a = direction.dot(direction)
                        b = -2.0 * dis_op
                        c = dis_oc_square - r*r

                        hit_t = (-b - ti.sqrt(b * b - 4.0 * a * c)) / 2.0 / a
                        hit_pos = origin + hit_t * direction
                        #h1 is nearer than h2
                        #float t2 = (-b + sqrt(b * b - 4.0f * a * c) ) / 2.0 / a;
                        #float3 h2 = o + t1 * d;
                        hit_nor = hit_pos - c
                        hit_gnor = hit_nor
            else:
                hit_t     = UF.INF_VALUE

        return hit_t, hit_pos, hit_gnor.normalized(), hit_nor.normalized(), hit_tex 


    @ti.func
    def intersect_tri(self, origin, direction, primitive_id):
        # https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/ray-triangle-intersection-geometric-solution
        t = UF.INF_VALUE
        u = 0.0
        v = 0.0
        
        vertex_id = UF.get_prim_vindex(self.primitive,primitive_id) 
        v0 = UF.get_vertex_pos(self.vertex, vertex_id+0)  
        v1 = UF.get_vertex_pos(self.vertex, vertex_id+1) 
        v2 = UF.get_vertex_pos(self.vertex, vertex_id+2) 
        E1 = v1 - v0
        E2 = v2 - v0

        P = direction.cross(E2)
        det = E1.dot(P)

        T = ti.Vector([0.0, 0.0, 0.0]) 
        if( det > 0.0 ):
            T = origin - v0
        else:
            T = v0 - origin
            det = -det

        if( det > 0.0 ):
            u = T.dot(P)
            if (( u >= 0.0) & (u <= det )):
                Q = T.cross(E1)
                v = direction.dot(Q)
                if((v >= 0.0) & (u + v <= det )):
                    t = E2.dot(Q)
                    fInvDet = 1.0 / det
                    t *= fInvDet
                    u *= fInvDet
                    v *= fInvDet
        return t,u,v



    @ti.func
    def intersect_prim_any(self, origin, direction, primitive_id):
        prim_type = UF.get_prim_type(self.primitive,primitive_id)
        hit_t     = UF.INF_VALUE


        if prim_type == SCD.PRIMITIVE_TRI:
            hit_t, u,v = self.intersect_tri(origin, direction, primitive_id)
        else:
            sha_id   = UF.get_prim_vindex(self.primitive, primitive_id) 
            sha_type = UF.get_shape_type(self.shape, sha_id)
            if sha_type == SCD.SHPAE_SPHERE:
                r      = UF.get_shape_radius(self.shape, sha_id )
                centre = UF.get_shape_pos(self.shape, sha_id )
                oc     = centre - origin
                dis_oc_square = oc.dot(oc)
                dis_op        = direction.dot (oc)
                dis_cp        = ti.sqrt(dis_oc_square - dis_op * dis_op)
                if (dis_cp < r):
                        a = direction.dot(direction)
                        b = -2.0 * dis_op
                        c = dis_oc_square - r*r

                        hit_t = (-b - ti.sqrt(b * b - 4.0 * a * c)) / 2.0 / a
            else:
                hit_t     = UF.INF_VALUE

        return hit_t
        
    @ti.func
    def closet_hit_shadow(self, origin, direction,stack, i,j, MAX_SIZE):
        hit_t           = UF.INF_VALUE
        stack[i,j, 0]   = 0
        stack_pos       = 0
        hit_prim        = -1

        while (stack_pos >= 0) & (stack_pos < MAX_SIZE):
            #pop
            node_index = stack[i, j, stack_pos]
            stack_pos  = stack_pos-1
            #type_n     = UF.get_node_type(self.bvh.compact_node, node_index)
            if UF.get_compact_node_type(self.bvh.compact_node, node_index) == SCD.IS_LEAF:
                shadow_prim          = UF.get_compact_node_prim(self.bvh.compact_node, node_index)
                t                    = self.intersect_prim_any(origin, direction, shadow_prim)
                if ( t < hit_t ) & (t > 0.0):
                    hit_t = t
                    hit_prim = shadow_prim
            else:
                min_v,max_v = UF.get_compact_node_min_max(self.bvh.compact_node, node_index)
                if UF.slabs(origin, direction,min_v,max_v) == 1:
                    left_node  = node_index+1
                    right_node = UF.get_compact_node_offset(self.bvh.compact_node, node_index)
                    #push
                    stack_pos              += 1
                    stack[i, j, stack_pos] = left_node
                    stack_pos              += 1
                    stack[i, j, stack_pos] = right_node
        return  hit_t, hit_prim

    # to cal detail intersect
    @ti.func
    def closet_hit(self, origin, direction, stack, i,j, MAX_SIZE):
        hit_t       = UF.INF_VALUE
        hit_pos     = ti.Vector([0.0, 0.0, 0.0]) 
        hit_normal  = ti.Vector([0.0, 0.0, 0.0]) 
        hit_gnormal = ti.Vector([0.0, 0.0, 0.0]) 
        hit_tex     = ti.Vector([0.0, 0.0, 0.0]) 
        hit_prim    = -1

        stack[i,j, 0]  = 0
        stack_pos       = 0
        while (stack_pos >= 0) & (stack_pos < MAX_SIZE):
            #pop
            node_index = stack[i, j, stack_pos]
            stack_pos  = stack_pos-1

            
            
            if UF.get_compact_node_type(self.bvh.compact_node, node_index) == SCD.IS_LEAF:
                prim_index          = UF.get_compact_node_prim(self.bvh.compact_node, node_index)
                t, pos, gnormal,normal, tex  = self.intersect_prim(origin, direction, prim_index)
                if ( t < hit_t ) & (t > 0.0):
                    hit_t       = t
                    hit_pos     = pos
                    hit_normal  = normal
                    hit_gnormal = gnormal
                    hit_tex     = tex
                    hit_prim    = prim_index
            else:
                min_v,max_v = UF.get_compact_node_min_max(self.bvh.compact_node, node_index)
                if UF.slabs(origin, direction,min_v,max_v) == 1:
                    left_node  = node_index+1
                    right_node = UF.get_compact_node_offset(self.bvh.compact_node, node_index)
                    #push
                    stack_pos              += 1
                    stack[i, j, stack_pos] = left_node
                    stack_pos              += 1
                    stack[i, j, stack_pos] = right_node

        if stack_pos == MAX_SIZE:
            print("overflow, need larger stack")

        return  hit_t, hit_pos, hit_gnormal, hit_normal, hit_tex, hit_prim


    @ti.kernel
    def total_area(self):
        for i in self.light:
            self.light_area[0] += self.get_prim_area(self.light[i])

    #smooth normal
    #http://www.bytehazard.com/articles/vertnorm.html
    @ti.kernel
    def process_normal(self):
        for i in self.vertex:
            v                     = UF.get_vertex_pos(self.vertex,i)
            n                     = UF.get_vertex_normal(self.vertex,i).normalized()
            f                     = self.vertex_index[i]
            self.smooth_normal[i] = n  * self.get_prim_angle(f, v) * self.get_prim_area(f)

            self.stack[i,0]       = 0
            stack_pos             = 0
            while (stack_pos >= 0) & (stack_pos < MAX_STACK_SIZE):
                #pop
                node_index = self.stack[i,stack_pos]
                stack_pos  = stack_pos-1
                type_n     = UF.get_compact_node_type(self.bvh.compact_node, node_index)

                if type_n == SCD.IS_LEAF:
                    prim_index = UF.get_compact_node_prim(self.bvh.compact_node, node_index)
                    prim_type  = UF.get_prim_type(self.primitive, prim_index)

                    if prim_type == SCD.PRIMITIVE_TRI:
                        ver_index  = UF.get_prim_vindex(self.primitive, prim_index)
                        for j in ti.static(range(3)):
                            neighbour_index = j + ver_index
                            if  i != neighbour_index:
                                neighbour_v              = UF.get_vertex_pos(self.vertex, neighbour_index)
                                neighbour_rormal         = UF.get_vertex_normal(self.vertex,neighbour_index).normalized()
                                if (ts.length(v-neighbour_v) < 0.000001) & (neighbour_rormal.dot(n) > 0.5):
                                    angle                     = self.get_prim_angle(prim_index, neighbour_v)
                                    self.smooth_normal[i]     += neighbour_rormal  * angle * self.get_prim_area(prim_index)
                                    #self.smooth_normal[i]    += neighbour_rormal

                else:
                    min_v,max_v = UF.get_compact_node_min_max(self.bvh.compact_node, node_index)
                    if (v.x >= min_v.x) & (v.y >= min_v.y) & (v.z >= min_v.z) & (v.x <= max_v.x) & (v.y <= max_v.y) & (v.z <= max_v.z):
                        #left_node,right_node   = UF.get_node_child(self.bvh.bvh_node,  node_index)  
                        offset     = UF.get_compact_node_offset(self.bvh.compact_node, node_index)        
                        #push
                        stack_pos              += 1
                        self.stack[i,stack_pos] = int(node_index+1)
                        stack_pos              += 1
                        self.stack[i,stack_pos] = int(offset)

        for i in self.vertex:
            UF.set_vertex_normal(self.vertex, i, self.smooth_normal[i].normalized())
                