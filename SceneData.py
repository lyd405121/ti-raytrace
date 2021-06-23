import taichi as ti
import numpy as np

########################### Important !!!! ###############################
#struct description

#material  : type_f      aleboTex_f   color_v3    param_v5                                 10
#            0  disney                            metallic_f    roughness_f 
#            1  glass                             ior           extinction  
#            2  light                                       
#            10 spectral                             

#shape     : type_f   pos_v3   param_v6                                                 10
#            1        sphere   radius
#            2        quad     v1      v2
#            3        spot     x1 x2 scale normal   
#            4        laser    radius X    X     normal                       

#vertex    : pos_v3   normal_v3 tex_v3                                                  9

#primitive : type(0:tri 1:shape) vertexIndex(shape_index) matIndex                      3
            

#            32bit         | 32bit    | 32bit       | 32bit      | 32bit     |96bit  |96bit 
#bvh_node  : is_leaf axis  |left_node  right_node   parent_node  prim_index   min_v3  max_v3   11
#             1bit   2bit       

#                32bit         |32bit       |32bit |96bit  |96bit 
#compact_node  : is_leaf axis  |prim_index  offset  min_v3  max_v3   9
#                1bit   2bit      

########################### Important !!!! ###############################
MAT_VEC_SIZE    = 10
VER_VEC_SIZE    = 9
PRI_VEC_SIZE    = 3
SHA_VEC_SIZE    = 10
NOD_VEC_SIZE    = 11
CPNOD_VEC_SIZE  = 9

SHPAE_NONE      = 0
SHPAE_SPHERE    = 1
SHPAE_QUAD      = 2
SHPAE_SPOT      = 3
SHPAE_LASER     = 4

PRIMITIVE_NONE  = 0
PRIMITIVE_TRI   = 1
PRIMITIVE_SHAPE = 2

MAT_DISNEY      = 0.0
MAT_GLASS       = 1.0
MAT_LIGHT       = 2.0
MAT_SPECTRAL   = 10.0

class Material:
    def __init__(self):
        self.type      = 0
        self.alebdoTex = 0 
        self.color     = [0.0, 0.0, 0.0]
        self.param     = [0.0, 0.0, 0.0, 0.0, 0.0]

    def setColor(self, color):
        self.color = color

    def setMetal(self, metal):
        self.param[0] = metal

    def setRough(self, rough):
        self.param[1] = rough

    def setIor(self, ior):
        self.param[0] = ior

    def setExtinciton(self, extinction):
        self.param[1] = extinction

    def fillStruct(self, np_data, index):
        np_data[index, 0] = float(self.type)
        np_data[index, 1] = float(self.alebdoTex)
        np_data[index, 2] = self.color[0]
        np_data[index, 3] = self.color[1]
        np_data[index, 4] = self.color[2]
        for i in range(5, MAT_VEC_SIZE):
            np_data[index, i] = self.param[i-5]

class Shape:
    def __init__(self):
        self.type      = 0
        self.pos       = [0.0, 0.0, 0.0]
        self.param     = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    def setRadius(self, radius):
        self.param[0] = radius

    def setXita(self, xita1, xita2):
        self.param[0] = xita1
        self.param[1] = xita2

    def setScale(self, scale):
        self.param[2] = scale

    def setV1(self, V1):
        self.param[0] = V1[0]
        self.param[1] = V1[1]
        self.param[2] = V1[2]

    def setV2(self, V2):
        self.param[3] = V2[0]
        self.param[4] = V2[1]
        self.param[5] = V2[2]


    def setNormal(self, normal):
        self.param[3] = normal[0]
        self.param[4] = normal[1]
        self.param[5] = normal[2]



    def fillStruct(self, np_data, index):
        np_data[index, 0] = float(self.type)
        np_data[index, 1] = self.pos[0]
        np_data[index, 2] = self.pos[1]
        np_data[index, 3] = self.pos[2]
        for i in range(4, VER_VEC_SIZE+1):
            np_data[index, i] = self.param[i-4]


class Vertex:
    def __init__(self):
        self.pos     = [0.0, 0.0, 0.0]
        self.normal  = [0.0, 0.0, 0.0]
        self.tex     = [0.0, 0.0, 0.0]
    
    def setPos(self, buf, offset):
        self.pos[0] = buf[offset +0 ]
        self.pos[1] = buf[offset +1 ]
        self.pos[2] = buf[offset +2 ]

    def setNormal(self, buf, offset):
        self.normal[0] = buf[offset +0 ]
        self.normal[1] = buf[offset +1 ]
        self.normal[2] = buf[offset +2 ]
    
    def setTex(self, buf, offset):
        self.tex[0] = buf[offset +0 ]
        self.tex[1] = buf[offset +1 ]
        self.tex[2] = 0.0


    def setTex3(self, buf, offset):
        self.tex[0] = buf[offset +0 ]
        self.tex[1] = buf[offset +1 ]
        self.tex[2] = buf[offset +2 ]

    def fillStruct(self, np_data, index):
        for i in range(0, 3):
            np_data[index, i+0] = self.pos[i]
            np_data[index, i+3] = self.normal[i]
            np_data[index, i+6] = self.tex[i]

class Primitive:
    def __init__(self):
        self.type                = 0
        self.vertex_shape_index  = 0
        self.mat_index           = 0

    def fillStruct(self, np_data, index):
        np_data[index, 0] = self.type
        np_data[index, 1] = self.vertex_shape_index
        np_data[index, 2] = self.mat_index