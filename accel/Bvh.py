import taichi as ti
import math
import numpy as np
import pywavefront
import SceneData as SCD
import UtilsFunc as UF
import taichi_glsl as ts
import queue

HIT_TRI           = 0.0
HIT_SHA           = 1.0
IS_LEAF           = 1
IS_LIGHT          = 3.0
@ti.data_oriented
class Bvh:
    def __init__(self):

        self.min_boundary    = ti.Vector.field(3, dtype=ti.f32, shape=(1))
        self.max_boundary    = ti.Vector.field(3, dtype=ti.f32, shape=(1))
        self.maxboundarynp   = np.ones(shape=(1,3), dtype=np.float32)
        self.minboundarynp   = np.ones(shape=(1,3), dtype=np.float32)

        self.light_cpu       = []
        self.material_cpu    = []
        self.vertex_cpu      = []
        self.shape_cpu       = []
        self.primitive_cpu   = []
        self.material        = ti.Vector.field(SCD.MAT_VEC_SIZE, dtype=ti.f32)
        self.vertex          = ti.Vector.field(SCD.VER_VEC_SIZE, dtype=ti.f32)
        self.primitive       = ti.Vector.field(SCD.PRI_VEC_SIZE, dtype=ti.i32)
        self.shape           = ti.Vector.field(SCD.SHA_VEC_SIZE, dtype=ti.f32)
        self.light           = ti.field(dtype=ti.i32)

        self.material_count  = 0
        self.vertex_count    = 0
        self.primitive_count = 0
        self.shape_count     = 0
        self.light_count     = 0

        self.radix_count_zero = ti.field(dtype=ti.i32, shape=[1])
        self.radix_offset     = ti.Vector.field(2, dtype=ti.i32)
 
        self.morton_code_s    = ti.Vector.field(2, dtype=ti.i32)
        self.morton_code_d    = ti.Vector.field(2, dtype=ti.i32)
 
        self.bvh_node         = ti.Vector.field(SCD.NOD_VEC_SIZE, dtype=ti.f32)
        self.bvh_done         = ti.field(dtype=ti.i32, shape=[1])
    ########################host function#####################################

    @ti.pyfunc
    def get_pot_num(self, num):
        m = 1
        while m<num:
            m=m<<1
        return m >>1

    @ti.pyfunc
    def get_pot_bit(self, num):
        m   = 1
        cnt = 0
        while m<num:
            m=m<<1
            cnt += 1
        return cnt

    @ti.pyfunc
    def add_obj(self, filename):
        scene = pywavefront.Wavefront(filename)
        scene.parse() 

        for name in scene.materials:
            
            ###process mat##
            material          = SCD.Material()
            material.type     = 0
            material.setColor(scene.materials[name].diffuse)
            if (scene.materials[name].ambient[0]> 1.0) & (scene.materials[name].ambient[1] > 1.0) & (scene.materials[name].ambient[2] > 1.0):
                material.setEmission(scene.materials[name].ambient)
            material.setMetal(0.0)
            material.setRough(0.5)
            if scene.materials[name].texture != None:
                material.alebdoTex = float(scene.materials[name].texture)
            else:
                material.alebdoTex = -1
            self.material_cpu.append(material)

            #####light######
            is_light = 0
            if scene.materials[name].ambient[0]> 1.0:
                is_light = 1

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

                ######process triangle#########
                if self.vertex_count % 3  == 0 :
                    primitive              = SCD.Primitive()
                    primitive.type         = SCD.PRIMITIVE_TRI
                    primitive.vertex_shape_index = self.vertex_count-3
                    primitive.mat_index    = self.material_count
                    self.primitive_cpu.append(primitive)
                    if is_light == 1:
                        self.light_cpu.append(self.primitive_count)
                        self.light_count += 1
                    self.primitive_count += 1
            self.material_count += 1


    @ti.pyfunc
    def add_shape(self, shape, mat):
        emission = mat.getEmission()
        if (emission[0]> 1.0) & (emission[1] > 1.0) & (emission[2] > 1.0):
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
    def modify_mat(self, index, mat):

        if index < len(self.material_cpu):
            self.material_cpu[index ]=mat

    @ti.pyfunc
    def blelloch_scan_host(self, mask, move):
        self.radix_sort_predicate(mask, move)

        for i in range(1, self.primitive_bit+1):
            self.blelloch_scan_reduce(1<<i)

        for i in range(self.primitive_bit+1, 0, -1):
            self.blelloch_scan_downsweep(1<<i)

        #print(self.radix_offset.to_numpy(), self.radix_count_zero.to_numpy())

    @ti.pyfunc
    def radix_sort_host(self):
        for i in range(30):
            mask   = 0x00000001 << i
            self.blelloch_scan_host(mask, i)
            #print("********************", self.radix_count_zero.to_numpy())
            self.radix_sort_fill(mask, i)
        #self.print_morton_reslut()

    @ti.pyfunc
    def print_morton_reslut(self):
        tmp = self.morton_code_s.to_numpy()
        for i in range(1, self.primitive_count):
            if tmp[i,0] < tmp[i-1,0]:
                print(i, tmp[i,:], tmp[i-1,:], "!!!!!!!!!!wrong!!!!!!!!!!!!")
            elif tmp[i,0] == tmp[i-1,0]:
                print(i, tmp[i,:], tmp[i-1,:], "~~~~~~equal~~~~~~~~~~~~~")
        print("********************")

    @ti.pyfunc
    def print_node_info(self, bvh_node, index):
            is_leaf = int(bvh_node[index, 0])
            left    = int(bvh_node[index, 1])
            right   = int(bvh_node[index, 2])
            parent  = int(bvh_node[index, 3])
            minx    = bvh_node[index, 5]
            miny    = bvh_node[index, 6]
            minz    = bvh_node[index, 7]
            maxx    = bvh_node[index, 8]
            maxy    = bvh_node[index, 9]
            maxz    = bvh_node[index, 10]
            print("node:%d l:%d r:%d p:%d lf: %d  min:%.2f %.2f %.2f max:%.2f %.2f %.2f"%(index, left, right, parent, is_leaf, minx,miny,minz, maxx,maxy,maxz), file = self.fo)

    @ti.pyfunc
    def print_node_bf(self, bvh_node):
        self.fo = open("nodelist.txt", "w")
        q = queue.Queue(self.node_count)
        q.put(0)
        while (not q.empty()):
            index = q.get()
            self.print_node_info(bvh_node, index)

            is_leaf = int(bvh_node[index, 0])
            left    = int(bvh_node[index, 1])
            right   = int(bvh_node[index, 2])
            parent  = int(bvh_node[index, 3])
            if (parent == right) | (parent == left):
                print("wrong build!!!",index, left, right, parent, file = self.fo)
                return
            else:
                if is_leaf != IS_LEAF:
                    q.put(left)
                    q.put(right)
        self.fo.close()

    @ti.pyfunc
    def print_node_list(self, bvh_node):

        for i in range (self.primitive_count):
            self.print_node_info(bvh_node, i)


    @ti.pyfunc
    def setup_data_cpu(self):
        self.node_count      = self.primitive_count*2-1
        self.primitive_pot   = (self.get_pot_num(self.primitive_count)) << 1
        self.primitive_bit   = self.get_pot_bit(self.primitive_pot)

        ti.root.dense(ti.i, self.light_count     ).place(self.light)
        ti.root.dense(ti.i, self.material_count  ).place(self.material)
        ti.root.dense(ti.i, self.vertex_count    ).place(self.vertex)
        ti.root.dense(ti.i, self.primitive_count ).place(self.primitive)
        ti.root.dense(ti.i, self.primitive_count ).place(self.morton_code_s)
        ti.root.dense(ti.i, self.primitive_count ).place(self.morton_code_d)
        ti.root.dense(ti.i, self.primitive_pot   ).place(self.radix_offset)


        ti.root.dense(ti.i, self.node_count).place(self.bvh_node )


    @ti.pyfunc
    def setup_data_gpu(self):
        print("***************mat:%d*******************"%(self.material_count))
        material_np  = np.zeros(shape=(self.material_count, SCD.MAT_VEC_SIZE), dtype=np.float32)
        for i in range(self.material_count):
            self.material_cpu[i].fillStruct(material_np, i)
            print("mat %d:"%(i), material_np[i,:])
        print("**************************************")

        print("***************vertex:%d*******************"%(self.vertex_count))
        vertex_np  = np.zeros(shape=(self.vertex_count, SCD.VER_VEC_SIZE), dtype=np.float32)
        for i in range(self.vertex_count):
            self.vertex_cpu[i].fillStruct(vertex_np, i)
            #print("ver %d:"%(i), vertex_np[i,:])
        print("**************************************")


        #self.primitive_bit = 0
        print("***************prim:%d pot:%d bit:%d*******************"%(self.primitive_count, self.primitive_pot, self.primitive_bit))
        primitive_np  = np.zeros(shape=(self.primitive_count, SCD.PRI_VEC_SIZE), dtype=np.int32)
        for i in range(self.primitive_count):
            self.primitive_cpu[i].fillStruct(primitive_np, i)
            #print("tri %d:"%(i), primitive_np[i,:])
        print("**************************************")

        print("***************light:%d*******************"%(self.light_count))
        light_np  = np.zeros(shape=(self.light_count), dtype=np.int32)
        for i in range(self.light_count):
            light_np[i] = self.light_cpu[i]
            #print("light %d:"%(i), light_np[i])
        print("**************************************")

        print("***************shape:%d*******************"%(self.shape_count))
        if self.shape_count > 0:
            ti.root.dense(ti.i, self.shape_count     ).place(self.shape)
            shape_np  = np.zeros(shape=(self.shape_count, SCD.SHA_VEC_SIZE), dtype=np.float32)
            for i in range(self.shape_count):
                self.shape_cpu[i].fillStruct(shape_np, i)
                #print("shape %d:"%(i), shape_np[i,:])
            self.shape.from_numpy(shape_np)
        else:
            ti.root.dense(ti.i, 1     ).place(self.shape)
        print("**************************************")



        self.light.from_numpy(light_np)
        self.material.from_numpy(material_np)
        self.vertex.from_numpy(vertex_np)
        self.primitive.from_numpy(primitive_np)
        self.max_boundary.from_numpy(self.maxboundarynp)
        self.min_boundary.from_numpy(self.minboundarynp)
        #print(self.light.to_numpy())
        
        self.build_morton_3d()
        print("morton code is built")
        self.radix_sort_host()
        print("radix sort  is done")
        #self.print_morton_reslut()

        
        self.build_bvh()
        print("tree build  is done")
        
        done_prev = 0
        done_num  = 0
        while done_num < self.primitive_count-1:
            self.gen_aabb()
            done_num  = self.bvh_done.to_numpy()
            if done_num == done_prev:
                break
            done_prev = done_num
        if done_num != self.primitive_count-1:
            print("aabb gen error!!!!!!!!!!!!!!!!!!!")
        else:
            print("aabb gen suc")
        

        print("***************node:%d*******************"%(self.node_count))
        #bvh_node = self.bvh_node.to_numpy()
        #self.print_node_bf(bvh_node)
        #self.print_node_list(bvh_node)
        print("**************************************")
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
        return ret

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

            if(a+b>1.0):
                a = 1.0 - a
                b = 1.0 - b
            pos    = v1 + (v3-v1)*a + (v2-v1)*b 
            normal = ((v2-v1).cross(v3-v1))

        else:
            sha_id   = UF.get_prim_vindex(self.primitive, index) 
            sha_type = UF.get_shape_type(self.shape, sha_id)
            if sha_type == SCD.SHPAE_SPHERE:
                r      = UF.get_shape_radius(self.shape, sha_id )
                centre = UF.get_shape_pos(self.shape, sha_id )
        
                normal = self.UniformSampleSphere(a, b) 
                pos    = centre + normal * r

        return pos,normal.normalized()

    @ti.func
    def get_random_light_prim_index(self):
        index = (int)(ts.clamp(ti.random()  * self.light_count, 0.0, self.light_count))-1
        return self.light[index]

    @ti.func
    def intersect_prim(self, origin, direction, primitive_id):
        prim_type = UF.get_prim_type(self.primitive,primitive_id)
        hit_t     = UF.INF_VALUE
        hit_pos   = ti.Vector([0.0, 0.0, 0.0])
        hit_nor   = ti.Vector([0.0, 0.0, 0.0])
        hit_tex   = ti.Vector([0.0, 0.0, 0.0])

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
                
                hit_pos = a*v1 + b*v2 + c*v3
                hit_tex = a*t1 + b*t2 + c*t3
                
                if n1.norm() < 0.01:
                    v13 = (v3-v1)
                    v12 = (v2-v1)
                    hit_nor = v12.cross(v13)
                else:
                    hit_nor = a*n1 + b*n2 + c*n3
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

        return hit_t, hit_pos, hit_nor.normalized(), hit_tex 




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
        return hit_t


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
    def closet_hit_shadow(self, origin, direction,stack, i,j, MAX_SIZE):
        hit_t           = UF.INF_VALUE
        stack[i,j, 0]   = 0
        stack_pos       = 0
        hit_prim        = -1

        while (stack_pos >= 0) & (stack_pos < MAX_SIZE):
            #pop
            node_index = stack[i, j, stack_pos]
            stack_pos  = stack_pos-1
            type_n     = UF.get_node_type(self.bvh_node, node_index)

            if type_n == IS_LEAF:
                shadow_prim          = UF.get_node_prim(self.bvh_node, node_index)
                t                    = self.intersect_prim_any(origin, direction, shadow_prim)
                if ( t < hit_t ) & (t > 0.0):
                    hit_t = t
                    hit_prim = shadow_prim
            else:
                min_v,max_v = UF.get_node_min_max(self.bvh_node, node_index)
                if UF.slabs(origin, direction,min_v,max_v) == 1:
                    left_node,right_node   = UF.get_node_child(self.bvh_node,  node_index) 
                    #print(node_index, left_node, right_node)
                    #push
                    stack_pos              += 1
                    stack[i, j, stack_pos] = left_node
                    stack_pos              += 1
                    stack[i, j, stack_pos] = right_node
        return  hit_prim


    # to cal detail intersect
    @ti.func
    def closet_hit(self, origin, direction, stack, i,j, MAX_SIZE):
        hit_t      = UF.INF_VALUE
        hit_pos    = ti.Vector([0.0, 0.0, 0.0]) 
        hit_normal = ti.Vector([0.0, 0.0, 0.0]) 
        hit_tex    = ti.Vector([0.0, 0.0, 0.0]) 
        hit_prim   = -1

        stack[i,j, 0]  = 0
        stack_pos       = 0
        while (stack_pos >= 0) & (stack_pos < MAX_SIZE):
            #pop
            node_index = stack[i, j, stack_pos]
            stack_pos  = stack_pos-1
            type_n     = UF.get_node_type(self.bvh_node, node_index)

            if type_n == IS_LEAF:
                
                prim_index = UF.get_node_prim(self.bvh_node, node_index)
                t, pos, normal, tex  = self.intersect_prim(origin, direction, prim_index)
                if (t > 0.0) & (t < hit_t):
                    hit_t      = t
                    hit_pos    = pos
                    hit_normal = normal
                    hit_tex    = tex
                    hit_prim   = prim_index

            else:
                
                min_v,max_v = UF.get_node_min_max(self.bvh_node, node_index)
                if UF.slabs(origin, direction,min_v,max_v) == 1:
                    left_node,right_node   = UF.get_node_child(self.bvh_node,  node_index)          
                    #push
                    stack_pos              += 1
                    stack[i, j, stack_pos] = left_node
                    stack_pos              += 1
                    stack[i, j, stack_pos] = right_node
        return  hit_t, hit_pos, hit_normal, hit_tex, hit_prim


    @ti.func
    def determineRange(self, idx):
        l_r_range = ti.cast(ti.Vector([0, self.primitive_count-1]), ti.i32)

        if idx != 0:
            self_code = self.morton_code_s[idx][0]
            l_code    = self.morton_code_s[idx-1][0]
            r_code    = self.morton_code_s[idx+1][0]


            if  (l_code == self_code ) & (r_code == self_code) :
                l_r_range[0] = idx
                
                while  idx < self.primitive_count-1:
                    idx += 1
                    
                    if(idx >= self.primitive_count-1):
                        break

                    if (self.morton_code_s[idx][0] != self.morton_code_s[idx+1][0]):
                        break
                l_r_range[1] = idx 

            else:
                L_delta = UF.common_upper_bits(self_code, l_code)
                R_delta = UF.common_upper_bits(self_code, r_code)

                d = -1
                if R_delta > L_delta:
                    d = 1
                delta_min = min(L_delta, R_delta)
                l_max = 2
                delta = -1
                i_tmp = idx + d * l_max

                if ( (0 <= i_tmp) &(i_tmp < self.primitive_count)):
                    delta = UF.common_upper_bits(self_code, self.morton_code_s[i_tmp][0])


                while delta > delta_min:
                    l_max <<= 1
                    i_tmp = idx + d * l_max
                    delta = -1
                    if ( (0 <= i_tmp) & (i_tmp < self.primitive_count)):
                        delta = UF.common_upper_bits(self_code, self.morton_code_s[i_tmp][0])

                l = 0
                t = l_max >> 1

                while(t > 0):
                    i_tmp = idx + (l + t) * d
                    delta = -1
                    if ( (0 <= i_tmp) & (i_tmp < self.primitive_count)):
                        delta = UF.common_upper_bits(self_code, self.morton_code_s[i_tmp][0])
                    if(delta > delta_min):
                        l += t
                    t >>= 1

                l_r_range[0] = idx
                l_r_range[1] = idx + l * d
                if(d < 0):
                    tmp          = l_r_range[0]
                    l_r_range[0] = l_r_range[1]
                    l_r_range[1] = tmp 

        return l_r_range
        
    @ti.func
    def findSplit(self, first, last):
        first_code = self.morton_code_s[first][0]
        last_code  = self.morton_code_s[last][0]
        split = first
        if (first_code != last_code):
            delta_node = UF.common_upper_bits(first_code, last_code)

            stride = last - first
            while 1:
                stride = (stride + 1) >> 1
                middle = split + stride
                if (middle < last):
                    delta = UF.common_upper_bits(first_code, self.morton_code_s[middle][0])
                    if (delta > delta_node):
                        split = middle
                if stride <= 1:
                    break
        return split



    @ti.kernel
    def build_morton_3d(self):
        for i in self.primitive:
            pri_type  = UF.get_prim_type(self.primitive, i)
            if pri_type == SCD.PRIMITIVE_TRI:
                vertex_id = UF.get_prim_vindex(self.primitive, i)
                v0        = UF.get_vertex_pos(self.vertex, vertex_id)
                v1        = UF.get_vertex_pos(self.vertex, vertex_id+1)
                v2        = UF.get_vertex_pos(self.vertex, vertex_id+2)
                centre_p  = (v1 + v2 + v0) * (1.0/ 3.0)
                norm_p    = (centre_p - self.min_boundary[0])/(self.max_boundary[0] - self.min_boundary[0])

                self.morton_code_s[i][0] = UF.morton3D(norm_p.x, norm_p.y, norm_p.z)
                self.morton_code_s[i][1] = i
            elif self.shape_count > 0:
                shape_id = UF.get_prim_vindex(self.primitive, i)
                v        = UF.get_vertex_pos(self.shape, shape_id)
                self.morton_code_s[i][0] = UF.morton3D(v.x, v.y, v.z)
                self.morton_code_s[i][1] = i

    
    @ti.kernel
    def radix_sort_predicate(self,  mask: ti.i32, move: ti.i32):
        for i in self.radix_offset:
            if i < self.primitive_count:
                self.radix_offset[i][1]       = (self.morton_code_s[i][0] & mask ) >> move
                self.radix_offset[i][0]       = 1-self.radix_offset[i][1]
                ti.atomic_add(self.radix_count_zero[0], self.radix_offset[i][0]) 
            else:
                self.radix_offset[i][0]       = 0
                self.radix_offset[i][1]       = 0

 
    @ti.kernel
    def blelloch_scan_reduce(self, mod: ti.i32):
        for i in self.radix_offset:
            if (i+1)%mod == 0:
                prev_index = i - (mod>>1)
                self.radix_offset[i][0] += self.radix_offset[prev_index][0]
                self.radix_offset[i][1]+= self.radix_offset[prev_index][1]

    @ti.kernel
    def blelloch_scan_downsweep(self, mod: ti.i32):
        for i in self.radix_offset:

            if mod == (self.primitive_pot*2):
                self.radix_offset[self.primitive_pot-1] = ti.Vector([0,0])
            elif (i+1)%mod == 0:
                prev_index = i - (mod>>1)
                if prev_index >= 0:
                    tmpV   = self.radix_offset[prev_index]
                    self.radix_offset[prev_index] = self.radix_offset[i]
                    self.radix_offset[i] += tmpV

    @ti.kernel
    def radix_sort_fill(self,  mask: ti.i32, move: ti.i32):
        for i in self.morton_code_s:
            condition = (self.morton_code_s[i][0] & mask ) >> move
            
            if condition == 1:
                offset = self.radix_offset[i][1] + self.radix_count_zero[0]
                self.morton_code_d[offset] = self.morton_code_s[i]
            else:
                offset = self.radix_offset[i][0] 
                self.morton_code_d[offset] = self.morton_code_s[i]

        for i in self.morton_code_s:
            self.morton_code_s[i]    = self.morton_code_d[i]
            self.radix_count_zero[0] = 0

    
    @ti.kernel
    def build_bvh(self):
        
        for i in self.bvh_node:
            UF.init_bvh_node(self.bvh_node, i)
            self.bvh_done[0] = 0

        for i in self.bvh_node:
            if i >= self.primitive_count-1:
                UF.set_node_type(self.bvh_node, i, IS_LEAF)
                prim_index = self.morton_code_s[i-self.primitive_count+1][1]
                UF.set_node_prim(self.bvh_node, i, prim_index)
                prim_type  = UF.get_prim_type(self.primitive, prim_index)
                ver_index  = UF.get_prim_vindex(self.primitive, prim_index)
                min_v3     = ti.Vector([0.0, 0.0, 0.0])
                max_v3     = ti.Vector([0.0, 0.0, 0.0])

                if prim_type == SCD.PRIMITIVE_TRI:
                    v1 = UF.get_vertex_pos(self.vertex, ver_index+0)
                    v2 = UF.get_vertex_pos(self.vertex, ver_index+1)
                    v3 = UF.get_vertex_pos(self.vertex, ver_index+2)
                    min_v3 = v1
                    max_v3 = v1
                    for k in ti.static(range(3)):
                        min_v3[k] = min(min_v3[k], v2[k])
                        min_v3[k] = min(min_v3[k], v3[k])
                        max_v3[k] = max(max_v3[k], v2[k])
                        max_v3[k] = max(max_v3[k], v3[k])
                else:
                    pos      = UF.get_shape_pos(self.shape,ver_index )
                    sha_type = UF.get_shape_type(self.shape,ver_index )
                    if sha_type == SCD.SHPAE_SPHERE:
                        r      = UF.get_shape_radius(self.shape,ver_index )
                        min_v3 = pos + ti.Vector([-r, -r, -r])
                        max_v3 = pos + ti.Vector([r, r, r])  
                UF.set_node_min_max(self.bvh_node, i, min_v3,max_v3)
            
            else:
                UF.set_node_type(self.bvh_node, i, 1-IS_LEAF)
                l_r_range   = self.determineRange(i)
                spilt       = self.findSplit(l_r_range[0], l_r_range[1])

        
                #print(i, l_r_range, spilt)
 
                left_node   = spilt
                right_node  = spilt + 1

                if min(l_r_range[0], l_r_range[1]) == spilt :
                    left_node  += self.primitive_count - 1
            
                if max(l_r_range[0], l_r_range[1]) == spilt + 1:
                    right_node  += self.primitive_count - 1
                
                if l_r_range[0] == l_r_range[1]:
                    print("wrong", l_r_range, spilt, left_node, right_node)

                #if i == 26:
                #print(i, l_r_range, spilt, left_node, right_node)
                UF.set_node_left(self.bvh_node,   i, left_node)
                UF.set_node_right(self.bvh_node,  i, right_node)
                UF.set_node_parent(self.bvh_node, left_node, i)
                UF.set_node_parent(self.bvh_node, right_node, i)

    @ti.kernel
    def gen_aabb(self):
        for i in self.bvh_node:
            if (UF.get_node_has_box(self.bvh_node, i)   == 0):
                left_node,right_node   = UF.get_node_child(self.bvh_node,  i) 
                
                is_left_rdy  = UF.get_node_has_box(self.bvh_node, left_node)
                is_right_rdy = UF.get_node_has_box(self.bvh_node, right_node)

                if (is_left_rdy & is_right_rdy) > 0:
                    
                    l_min,l_max = UF.get_node_min_max(self.bvh_node, left_node)  
                    r_min,r_max = UF.get_node_min_max(self.bvh_node, right_node)  
                    UF.set_node_min_max(self.bvh_node, i, min(l_min, r_min),max(l_max, r_max))
                    #if i == 0:
                    self.bvh_done[0] += 1
                #print("ok", i, left_node, right_node)



                    
