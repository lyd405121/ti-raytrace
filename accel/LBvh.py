import taichi as ti
import math
import numpy as np
import pywavefront
import SceneData as SCD
import UtilsFunc as UF
import taichi_glsl as ts


HIT_TRI           = 0.0
HIT_SHA           = 1.0


@ti.data_oriented
class Bvh:
    def __init__(self, primitive_count, min_boundary, max_boundary):
        
        self.primitive_count = primitive_count
        self.minboundarynp   = min_boundary
        self.maxboundarynp   = max_boundary

        self.min_boundary    = ti.Vector.field(3, dtype=ti.f32, shape=(1))
        self.max_boundary    = ti.Vector.field(3, dtype=ti.f32, shape=(1))


        self.radix_count_zero = ti.field(dtype=ti.i32, shape=[1])
        self.radix_offset     = ti.Vector.field(2, dtype=ti.i32)
 
        self.morton_code_s    = ti.Vector.field(2, dtype=ti.i32)
        self.morton_code_d    = ti.Vector.field(2, dtype=ti.i32)
 
        self.bvh_node         = ti.Vector.field(SCD.NOD_VEC_SIZE, dtype=ti.f32)
        self.compact_node     = ti.Vector.field(SCD.CPNOD_VEC_SIZE, dtype=ti.f32)
        self.bvh_done         = ti.field(dtype=ti.i32, shape=[1])
        self.leaf_node_count  = 0
    ########################host function#####################################

    
    def get_pot_num(self, num):
        m = 1
        while m<num:
            m=m<<1
        return m >>1

    
    def get_pot_bit(self, num):
        m   = 1
        cnt = 0
        while m<num:
            m=m<<1
            cnt += 1
        return cnt

    
    def blelloch_scan_host(self, mask, move):
        self.radix_sort_predicate(mask, move)

        for i in range(1, self.primitive_bit+1):
            self.blelloch_scan_reduce(1<<i)

        for i in range(self.primitive_bit+1, 0, -1):
            self.blelloch_scan_downsweep(1<<i)

        #print(self.radix_offset.to_numpy(), self.radix_count_zero.to_numpy())

    
    def radix_sort_host(self):
        for i in range(30):
            mask   = 0x00000001 << i
            self.blelloch_scan_host(mask, i)
            #print("********************", self.radix_count_zero.to_numpy())
            self.radix_sort_fill(mask, i)

    
    def print_morton_reslut(self):
        tmp = self.morton_code_s.to_numpy()
        for i in range(0, self.primitive_count):
            #if i > 0:
            #    if tmp[i,0] < tmp[i-1,0]:
            #        print(i, tmp[i,:], tmp[i-1,:], "!!!!!!!!!!wrong!!!!!!!!!!!!")
            #    elif tmp[i,0] == tmp[i-1,0]:
            #        print(i, tmp[i,:], tmp[i-1,:], "~~~~~~equal~~~~~~~~~~~~~")

            if i > 0:
                if tmp[i,0] < tmp[i-1,0]:
                    print(i, tmp[i,:], tmp[i-1,:], "!!!!!!!!!!wrong!!!!!!!!!!!!")
                elif tmp[i,0] == tmp[i-1,0]:
                    print(i, tmp[i,:], tmp[i-1,:], "~~~~~~equal~~~~~~~~~~~~~")
                else:
                    print(i, tmp[i,:], tmp[i-1,:])
            else:
                print(i, tmp[i,:])
                
        print("********************")


    def print_node_info(self, bvh_node, index):
        is_leaf = int(bvh_node[index, 0]) & 0x0001
        offset  = int(bvh_node[index, 5])
        left    = int(bvh_node[index, 1])
        right   = int(bvh_node[index, 2])

        parent  = int(bvh_node[index, 3])
        prim_index = int(bvh_node[index, 4])
        
        min_point = [bvh_node[index, 6],bvh_node[index, 7],bvh_node[index, 8]]
        max_point = [bvh_node[index, 9],bvh_node[index, 10],bvh_node[index, 11]]
        chech_pass = 1

        if is_leaf == SCD.IS_LEAF:
            self.leaf_node_count += 1
        else:
            for i in range(3):
                if (min_point[i] != min(bvh_node[left, 6+i], bvh_node[right, 6+i])) & (max_point[i] != max(bvh_node[left, 9+i], bvh_node[right, 9+i])):
                    chech_pass = 0
                    break
        
        if chech_pass == 1:
            print("node:%d l:%d r:%d p:%d pri:%d lf: %d  offset:%d min:%.2f %.2f %.2f max:%.2f %.2f %.2f"%(index, left, right, parent, prim_index,is_leaf, offset, min_point[0],min_point[1],min_point[2],\
                max_point[0],max_point[1],max_point[2]), file = self.fo)
        else:
            print("wnode:%d l:%d r:%d p:%d pri:%d lf: %d  offset:%d min:%.2f %.2f %.2f max:%.2f %.2f %.2f"%(index, left, right, parent, prim_index,is_leaf, offset, min_point[0],min_point[1],min_point[2],\
                max_point[0],max_point[1],max_point[2]), file = self.fo)



    def print_compact_info(self, bvh_node, index):

        offset  = int(bvh_node[index, 2])
        prim_index = int(bvh_node[index, 1])

        min_point = [bvh_node[index, 3],bvh_node[index, 4],bvh_node[index, 5]]
        max_point = [bvh_node[index, 6],bvh_node[index, 7],bvh_node[index, 8]]
        
        print("node:%d pri:%d offset:%d min:%.2f %.2f %.2f max:%.2f %.2f %.2f"%(index, prim_index, offset, min_point[0],min_point[1],min_point[2],\
                max_point[0],max_point[1],max_point[2]), file = self.fo)

    def flatten_tree(self, compact_node, bvh_node, index):

        retOffset    = self.offset
        self.offset += 1

        is_leaf = int(bvh_node[index, 0]) & 0x0001
        left    = int(bvh_node[index, 1])
        right   = int(bvh_node[index, 2])

        compact_node[retOffset][0] = bvh_node[index, 0]
        for i in range(6):
            compact_node[retOffset][2+i] = bvh_node[index, 5+i]

        if is_leaf != SCD.IS_LEAF:
            self.flatten_tree(compact_node, bvh_node, left)
            compact_node[retOffset][1] = self.flatten_tree(compact_node,bvh_node, right)
            #print(self.offset, index, file = self.fo)
        else:
            compact_node[retOffset][1] = bvh_node[index, 4]
            
            


        return retOffset

    
    def build_compact_node(self, bvh_node, compact_node):
        self.fo = open("nodelist.txt", "w")
        
        self.offset = 0
        self.flatten_tree(compact_node,bvh_node,0)

        for i in range(self.node_count):
            self.print_compact_info(compact_node, i)
        self.fo.close()
        self.compact_node.from_numpy(compact_node)


    
    def setup_data_cpu(self):
        self.node_count      = self.primitive_count*2-1
        self.primitive_pot   = (self.get_pot_num(self.primitive_count)) << 1
        self.primitive_bit   = self.get_pot_bit(self.primitive_pot)
        
        ti.root.dense(ti.i, self.primitive_count ).place(self.morton_code_s)
        ti.root.dense(ti.i, self.primitive_count ).place(self.morton_code_d)
        ti.root.dense(ti.i, self.primitive_pot   ).place(self.radix_offset)


        ti.root.dense(ti.i, self.node_count).place(self.bvh_node )
        ti.root.dense(ti.i, self.node_count).place(self.compact_node )


    
    def setup_data_gpu(self, vertex, shape, primitive):
        self.max_boundary.from_numpy(self.maxboundarynp)
        self.min_boundary.from_numpy(self.minboundarynp)
        #print(self.light.to_numpy())
        
        #### lbvh build
        self.build_morton_3d(vertex, shape, primitive)
        print("morton code is built")
        self.radix_sort_host()
        print("radix sort  is done")
        #self.print_morton_reslut()
        self.build_lbvh(vertex, shape, primitive)
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
            print("aabb gen error!!!!!!!!!!!!!!!!!!!%d"%done_num)
        else:
            print("aabb gen suc")
        

        print("***************node:%d*******************"%(self.node_count))

        bvh_node     = self.bvh_node.to_numpy()
        compact_node = self.compact_node.to_numpy()
        self.build_compact_node(bvh_node,compact_node)
        print("**************************************")

    ############algrithm##############
    @ti.func
    def determineRange(self, idx):
        l_r_range = ti.cast(ti.Vector([0, self.primitive_count-1]), ti.i32)

        if idx != 0:
            self_code = self.morton_code_s[idx][0]
            l         = idx-1
            r         = idx+1
            l_code    = self.morton_code_s[l][0]
            r_code    = self.morton_code_s[r][0]

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
                L_delta   = UF.common_upper_bits(self_code, l_code)
                R_delta   = UF.common_upper_bits(self_code, r_code)

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
    def build_morton_3d(self, vertex:ti.template(), shape:ti.template(), primitive:ti.template()):
        for i in primitive:
            pri_type  = UF.get_prim_type(primitive, i)
            if pri_type == SCD.PRIMITIVE_TRI:
                vertex_id = UF.get_prim_vindex(primitive, i)
                v0        = UF.get_vertex_pos(vertex, vertex_id)
                v1        = UF.get_vertex_pos(vertex, vertex_id+1)
                v2        = UF.get_vertex_pos(vertex, vertex_id+2)
                centre_p  = (v1 + v2 + v0) * (1.0/ 3.0)
                norm_p    = (centre_p - self.min_boundary[0])/(self.max_boundary[0] - self.min_boundary[0])

                self.morton_code_s[i][0] = UF.morton3D(norm_p.x, norm_p.y, norm_p.z)
                self.morton_code_s[i][1] = i
            else:
                shape_id = UF.get_prim_vindex(primitive, i)
                v        = UF.get_vertex_pos(shape, shape_id)
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
    def build_lbvh(self, vertex:ti.template(), shape:ti.template(), primitive:ti.template()):
        
        for i in self.bvh_node:
            UF.init_bvh_node(self.bvh_node, i)
            self.bvh_done[0] = 0

        for i in self.bvh_node:
            if i >= self.primitive_count-1:
                UF.set_node_type(self.bvh_node, i, SCD.IS_LEAF)
                UF.set_node_prim_size(self.bvh_node, i, 1)

                prim_index = self.morton_code_s[i-self.primitive_count+1][1]
                UF.set_node_prim(self.bvh_node, i, prim_index)
                prim_type  = UF.get_prim_type(primitive, prim_index)
                ver_index  = UF.get_prim_vindex(primitive, prim_index)
                min_v3     = ti.Vector([0.0, 0.0, 0.0])
                max_v3     = ti.Vector([0.0, 0.0, 0.0])

                if prim_type == SCD.PRIMITIVE_TRI:
                    v1 = UF.get_vertex_pos(vertex, ver_index+0)
                    v2 = UF.get_vertex_pos(vertex, ver_index+1)
                    v3 = UF.get_vertex_pos(vertex, ver_index+2)
                    min_v3 = v1
                    max_v3 = v1
                    for k in ti.static(range(3)):
                        min_v3[k] = min(min_v3[k], v2[k])
                        min_v3[k] = min(min_v3[k], v3[k])
                        max_v3[k] = max(max_v3[k], v2[k])
                        max_v3[k] = max(max_v3[k], v3[k])
                else:
                    pos      = UF.get_shape_pos(shape,ver_index )
                    sha_type = UF.get_shape_type(shape,ver_index )
                    if sha_type == SCD.SHPAE_SPHERE:
                        r      = UF.get_shape_radius(shape,ver_index )
                        min_v3 = pos + ti.Vector([-r, -r, -r])
                        max_v3 = pos + ti.Vector([r, r, r])  
                UF.set_node_min_max(self.bvh_node, i, min_v3,max_v3)
            
            else:
                UF.set_node_type(self.bvh_node, i, 1-SCD.IS_LEAF)
                l_r_range   = self.determineRange(i)
                spilt       = self.findSplit(l_r_range[0], l_r_range[1])
 
                left_node   = spilt
                right_node  = spilt + 1

                if min(l_r_range[0], l_r_range[1]) == spilt :
                    left_node  += self.primitive_count - 1
            
                if max(l_r_range[0], l_r_range[1]) == spilt + 1:
                    right_node  += self.primitive_count - 1
                
                if l_r_range[0] == l_r_range[1]:
                    print(l_r_range, spilt, left_node, right_node,"wrong")
                #else:
                #    print(l_r_range, spilt,left_node, right_node)

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
                    self.bvh_done[0] += 1
                #print("ok", i, left_node, right_node)



                    
