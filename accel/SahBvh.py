import taichi as ti
import math
import numpy as np
import pywavefront
import SceneData as SCD
import UtilsFunc as UF
import taichi_glsl as ts
from queue import Queue

HIT_TRI           = 0.0
HIT_SHA           = 1.0

@ti.data_oriented
class Bvh:
    def __init__(self, primitive_count, min_boundary, max_boundary):
        
        self.primitive_count = primitive_count
        self.minboundarynp   = min_boundary
        self.maxboundarynp   = max_boundary
        self.boundary        = self.maxboundarynp - self.minboundarynp
        self.node            = []
        self.depth           = 0
        self.compact_node    = ti.Vector.field(SCD.CPNOD_VEC_SIZE, dtype=ti.f32)
        self.cur_sort_axis   = -1
    
    def get_face_bounds(self,  face_index):
        b = SCD.Bounds()  
        if self.primitive[face_index].type  == SCD.PRIMITIVE_TRI:
            index = self.primitive[face_index].vertex_shape_index
            b.Merge(self.vertex[index+0].pos)
            b.Merge(self.vertex[index+1].pos)
            b.Merge(self.vertex[index+2].pos)
        else:
            s  =  self.shape[self.primitive[face_index].vertex_shape_index]
            if s.type == SCD.SHPAE_SPHERE:
                r = s.getRadius()
                for k in range(3):
                    b.min_v3[k] = s.pos[k] -r
                    b.max_v3[k] = s.pos[k] +r
            else:
                b.Merge(s.pos)
        return b

    
    def get_face_array_bounds(self,  face_start, face_end):
        b = SCD.Bounds()   
        for i in range(face_start,face_end+1):
            b.MergeBox(self.get_face_bounds(i))
        return b
 
 
    
    def get_prim_centroid(self,  prim_index):
        c               = [0.0,0.0,0.0]
        if self.primitive[prim_index].type  == SCD.PRIMITIVE_TRI:
            index = self.primitive[prim_index].vertex_shape_index
            v1 = self.vertex[index+0].pos
            v2 = self.vertex[index+1].pos
            v3 = self.vertex[index+2].pos
            for k in range(3):
                c[k] = (v1[k]+v2[k]+v3[k]) / 3.0
        else:
            c  = self.shape[self.primitive[prim_index].vertex_shape_index].pos
        return c
        
    
    def sah_split(self,  face_start, face_end):
        bestAxis = 0
        bestIndex = 0
        bestCost = np.Infinity
        numFaces = face_end - face_start+1
        cumulativeLower = [0.0]*numFaces
        cumulativeUpper = [0.0]*numFaces      

        for a in range(3):
            #FaceSorter predicate(&m_vertices[0], &m_indices[0], m_numFaces*3, a)
            #std::sort(faces, faces+numFaces, predicate);           
            self.cpu_sorter(face_start, face_end, a)
            

            lower = SCD.Bounds()
            upper = SCD.Bounds()            
            for i in range(numFaces):
                lower.MergeBox(self.get_face_bounds(i+face_start))
                upper.MergeBox(self.get_face_bounds(numFaces-i-1+face_start))

                cumulativeLower[i] = lower.GetSurfaceArea()
                cumulativeUpper[numFaces-i-1]=upper.GetSurfaceArea()

                #if(face_start ==0)&(face_end == self.primitive_count-1):
                #    print(cumulativeLower[i], cumulativeUpper[numFaces-i-1], lower.min_v3, lower.max_v3,upper.min_v3,upper.max_v3 )

            invTotalSA = 1.0 / cumulativeUpper[0]           
            for i in range(numFaces-1):
                pBelow = cumulativeLower[i] * invTotalSA
                pAbove = cumulativeUpper[i] * invTotalSA            
                cost = 0.125 + (pBelow*i + pAbove*(numFaces-i))
                if (cost <= bestCost):
                    bestCost = cost
                    bestIndex = i
                    bestAxis = a   
        
        self.cpu_sorter(face_start, face_end, bestAxis)
        return bestIndex +1+ face_start

    
    def build_node(self, face_start, face_end):
        numFaces = face_end - face_start+1
        node = SCD.BVHNode()

        b     = self.get_face_array_bounds(face_start, face_end)
        node.min_v3 = b.min_v3
        node.max_v3 = b.max_v3 



        split = -1
        if numFaces == 1:
            node.is_leaf = 1
            node.prim_index = face_start
            node.left_node  = -1
            node.right_node = -1
        else:
            split = self.sah_split(face_start, face_end)
            node.is_leaf = 0
            node.prim_index = -1
            node.left_node  = self.node_index+1
            node.right_node = self.node_index+2
            print("node %d is done"%(self.node_index))
            self.node_index +=2 

        self.node.append(node)
        
        return split


    
    def build(self):
        self.node_index = 0
        start = Queue()
        end = Queue()
        start.put(0)
        end.put(self.primitive_count-1)

        while start.empty() == False:
            face_start = start.get()
            face_end   = end.get()
            split = self.build_node(face_start, face_end)
            
            if split != -1:
                start.put(face_start)
                start.put(split)
                end.put(split-1)
                end.put(face_end)

    
    def setup_data_cpu(self,  vertex, shape, primitive):
        self.centroid   = []
        self.primitive  = primitive
        self.vertex     = vertex
        self.shape      = shape

        for i in range(self.primitive_count):
            c = self.get_prim_centroid(i)
            self.centroid.append([c[0], c[1], c[2], i])
        print("centroid calculate end")
        self.build()
        print("sah build end")
        #self.debug()
        ti.root.dense(ti.i, len(self.node)).place(self.compact_node )


    
    def debug(self):
        fo = open("debug.obj", "w")

        vertex_index = 1
        l = len(self.node)
        for i in range(l):
            if self.node[i].is_leaf == 1:
                prim_index = self.node[i].prim_index
                if self.primitive[prim_index].type  == SCD.PRIMITIVE_TRI:
                    index = self.primitive[prim_index].vertex_shape_index
                    v1 = self.vertex[index+0].pos
                    v2 = self.vertex[index+1].pos
                    v3 = self.vertex[index+2].pos
                    print ("v %f %f %f" %   (v1[0], v1[1], v1[2]), file = fo)
                    print ("v %f %f %f" %   (v2[0], v2[1], v2[2]), file = fo)
                    print ("v %f %f %f" %   (v3[0], v3[1], v3[2]), file = fo)
                    print ("f %d %d %d" %   (vertex_index, vertex_index+1, vertex_index+2), file = fo)
                    vertex_index += 3
            else:
                min_v3 = self.node[i].min_v3
                max_v3 = self.node[i].max_v3
                
                print ("v %f %f %f" %   (min_v3[0], min_v3[1], min_v3[2]), file = fo)
                print ("v %f %f %f" %   (min_v3[0], min_v3[1], max_v3[2]), file = fo)
                print ("v %f %f %f" %   (max_v3[0], min_v3[1], max_v3[2]), file = fo)
                print ("v %f %f %f" %   (max_v3[0], min_v3[1], min_v3[2]), file = fo)
                print ("v %f %f %f" %   (min_v3[0], max_v3[1], min_v3[2]), file = fo)
                print ("v %f %f %f" %   (min_v3[0], max_v3[1], max_v3[2]), file = fo)
                print ("v %f %f %f" %   (max_v3[0], max_v3[1], max_v3[2]), file = fo)
                print ("v %f %f %f" %   (max_v3[0], max_v3[1], min_v3[2]), file = fo)
                
                print ("f %d %d %d %d" %   (vertex_index+0, vertex_index+1, vertex_index+2, vertex_index+3), file = fo)
                print ("f %d %d %d %d" %   (vertex_index+4, vertex_index+5, vertex_index+6, vertex_index+7), file = fo)
                print ("f %d %d %d %d" %   (vertex_index+0, vertex_index+1, vertex_index+5, vertex_index+4), file = fo)
                print ("f %d %d %d %d" %   (vertex_index+2, vertex_index+3, vertex_index+7, vertex_index+6), file = fo)
                print ("f %d %d %d %d" %   (vertex_index+1, vertex_index+2, vertex_index+6, vertex_index+5), file = fo)
                print ("f %d %d %d %d" %   (vertex_index+0, vertex_index+4, vertex_index+7, vertex_index+3), file = fo)
                vertex_index += 8

        fo.close()
    



    
    def setup_data_gpu(self, vertex, shape, primitive):
        self.offset = 0
        compact_node = self.compact_node.to_numpy()
        self.flatten_tree(compact_node, 0)
        self.compact_node.from_numpy(compact_node)
        print("flatten tree end")

        #primitive_np = primitive.to_numpy()
        #for i in range(self.primitive_count):
        #    self.primitive[i].fillStruct(primitive_np, i)
        #primitive.from_numpy(primitive_np)



    
    def partition(self,  low,  high, axis):
        i = low-1 
        pivot = self.centroid[high][axis]     
    
        for j in range(low , high): 
            # 当前元素小于或等于 pivot 
            if   self.centroid[j][axis]<= pivot: 
                i = i+1 
                self.centroid[i],self.centroid[j]= self.centroid[j],self.centroid[i]
                self.primitive[i],self.primitive[j]= self.primitive[j],self.primitive[i]
        self.centroid[i+1],self.centroid[high] = self.centroid[high],self.centroid[i+1] 
        self.primitive[i+1],self.primitive[high] = self.primitive[high],self.primitive[i+1] 
        return i+1 

    
    def cpu_sorter(self,  low,  high, axis):
        if axis == self.cur_sort_axis:
            return
        low_que  = Queue()
        high_que = Queue()

        low_que.put(low)
        high_que.put(high)

        while low_que.empty() == False:
            low  = low_que.get()
            high = high_que.get()

            if low < high: 
                pi = self.partition(low,high, axis) 
                low_que.put(low)
                high_que.put(pi-1)
                low_que.put(pi+1)
                high_que.put(high)
        self.cur_sort_axis = axis

    
    def find_best_axis(self,  boundary):
        if(boundary[0, 0] > boundary[0, 1]):
            if(boundary[0, 0] > boundary[0, 2]):
                return 0
            else:
                return 2
        else:
            if(boundary[0, 1] > boundary[0, 2]):
                return 1
            else:
                return 2
 
    
    def flatten_tree(self, compact_node, index):
        retOffset    = self.offset
        self.offset += 1

        is_leaf = self.node[index].is_leaf
        left    = self.node[index].left_node
        right   = self.node[index].right_node

        compact_node[retOffset][0] = self.node[index].is_leaf
        for i in range(3):
            compact_node[retOffset][3+i] = self.node[index].min_v3[i]
            compact_node[retOffset][6+i] = self.node[index].max_v3[i]

        if is_leaf != SCD.IS_LEAF:
            self.flatten_tree(compact_node, left)
            compact_node[retOffset][1] = self.flatten_tree(compact_node, right)
        else:
            compact_node[retOffset][1] = self.centroid[self.node[index].prim_index][3]

        return retOffset



                    
