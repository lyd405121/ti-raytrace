import taichi as ti
########################### Important !!!! ###############################
#struct description

#material  : type_f aleboTex_f   color_v3      param_v5                                 10
#            0                                 emission_v3  metallic_f    roughness_f 
#            1                                 ior                       

#shape     : type_f   pos_v3   param_v6                                                 10
#            0        sphere   radius
#            1        quad     v1      v2
#            2        spot     radius  dis  normal                             
#vertex    : pos_v3   normal_v3 tex_v3                                                  9
#primitive : type(0:tri 1:shape) vertexIndex(shape_index) matIndex                      3
#bvh_node  : is_leaf  left_node    right_node  parent_node  prim_index  min_v3  max_v3 11
########################### Important !!!! ###############################


INF_VALUE = 1000000.0
########################### tool function ###############################
@ti.func
def get_material_type(material, index):
    return int(material[index][0]) 
@ti.func
def get_material_tex(material, index):
    return int(material[index][1]) 
@ti.func
def get_material_color(material, index):
    return ti.Vector([material[index][2], material[index][3], material[index][4] ])
@ti.func
def get_material_emission(material,  index):
    return ti.Vector([material[index][5], material[index][6], material[index][7] ])
@ti.func
def get_material_metallic(material,  index):
    return material[index][8]
@ti.func
def get_material_roughness(material, index):
    return material[index][9]
@ti.func
def get_material_ior(material, index):
    return 1.4

@ti.func
def get_vertex_pos(vertex, index):
    return ti.Vector([vertex[index][0], vertex[index][1], vertex[index][2] ])
@ti.func
def get_vertex_normal(vertex,  index):
    return ti.Vector([vertex[index][3], vertex[index][4], vertex[index][5] ])
@ti.func
def get_vertex_uv(vertex,  index):
    return ti.Vector([vertex[index][6], vertex[index][7], vertex[index][8] ])

@ti.func
def get_prim_type(primitive, index):
    return primitive[index][0]
@ti.func
def get_prim_vindex(primitive, index):
    return primitive[index][1]
@ti.func
def get_prim_mindex(primitive, index):
    return primitive[index][2]

    
@ti.func
def get_shape_type(shape, index):
    return int(shape[index][0])
@ti.func
def get_shape_pos(shape, index):
    return ti.Vector([shape[index][1], shape[index][2], shape[index][3] ])
@ti.func
def get_shape_radius(shape, index):
    return shape[index][4]
@ti.func
def get_shape_v1(shape, index):
    return ti.Vector([shape[index][4], shape[index][5], shape[index][6] ])
@ti.func
def get_shape_v2(shape, index):
    return ti.Vector([shape[index][7], shape[index][8], shape[index][9] ])
@ti.func
def get_shape_dis(shape, index):
    return shape[index][5]
@ti.func
def get_shape_normal(shape, index):
    return ti.Vector([shape[index][6], shape[index][7], shape[index][8] ])

@ti.func
def get_hit_t( hit_info):
    return hit_info[0]
@ti.func
def get_hit_prim( hit_info):
    return int(hit_info[10])
@ti.func
def get_hit_pos( hit_info):
    return ti.Vector([hit_info[1], hit_info[2], hit_info[3] ])
@ti.func
def get_hit_normal( hit_info):
    return ti.Vector([hit_info[4], hit_info[5], hit_info[6] ])
@ti.func
def get_hit_uv( hit_info):
    return ti.Vector([hit_info[7], hit_info[8], hit_info[9] ])

#bvh_node  : is_leaf  left_node    right_node  parent_node  prim_index  min_v3  max_v3 11
@ti.func
def init_bvh_node(bvh_node, index):
    bvh_node[index][0]  = -1.0
    bvh_node[index][1]  = -1.0
    bvh_node[index][2]  = -1.0
    bvh_node[index][3]  = -1.0
    bvh_node[index][4]  = -1.0
    bvh_node[index][5]  = INF_VALUE
    bvh_node[index][6]  = INF_VALUE
    bvh_node[index][7]  = INF_VALUE
    bvh_node[index][8]  = -INF_VALUE
    bvh_node[index][9]  = -INF_VALUE
    bvh_node[index][10] = -INF_VALUE
@ti.func
def set_node_type(bvh_node, index, type):
    bvh_node[index][0]  = float(type)
@ti.func
def set_node_left(bvh_node, index, left):
    bvh_node[index][1]  = float(left)
@ti.func
def set_node_right(bvh_node, index, right):
    bvh_node[index][2]  = float(right)
@ti.func
def set_node_parent(bvh_node, index, parent):
    bvh_node[index][3]  = float(parent)
@ti.func
def set_node_prim(bvh_node, index, prim):
    bvh_node[index][4]  = float(prim)
@ti.func
def set_node_min_max(bvh_node, index, minv,maxv):
    bvh_node[index][5]  = minv[0]
    bvh_node[index][6]  = minv[1]
    bvh_node[index][7]  = minv[2]
    bvh_node[index][8]  = maxv[0]
    bvh_node[index][9]  = maxv[1]
    bvh_node[index][10] = maxv[2]


@ti.func
def get_node_type(bvh_node, index):
    return int(bvh_node[index][0])
@ti.func
def get_node_child(bvh_node, index):
    return int(bvh_node[index][1]),int(bvh_node[index][2])
@ti.func
def get_node_parent(bvh_node, index):
    return int(bvh_node[index][3])
@ti.func
def get_node_prim(bvh_node, index):
    return int(bvh_node[index][4])
@ti.func
def get_node_min_max(bvh_node, index):
    return ti.Vector([bvh_node[index][5], bvh_node[index][6], bvh_node[index][7] ]),ti.Vector([bvh_node[index][8], bvh_node[index][9], bvh_node[index][10] ])
@ti.func
def get_node_has_box(bvh_node, index):
    return (bvh_node[index][5]  <= bvh_node[index][8]) & (bvh_node[index][6]  <= bvh_node[index][9]) & (bvh_node[index][7]  <= bvh_node[index][10])
###################################################################


############algrithm##############
@ti.func
def max_component( v):
    return max(v.z, max(v[0], v.y) )
@ti.func
def min_component( v):
    return min(v.z, min(v[0], v.y) )


@ti.func
def slabs(origin, direction, minv, maxv):
    # most effcient algrithm for ray intersect aabb 
    # en vesrion: https://www.researchgate.net/publication/220494140_An_Efficient_and_Robust_Ray-Box_Intersection_Algorithm
    # cn version: https://zhuanlan.zhihu.com/p/138259656

    
    ret  = 1
    tmin = 0.0
    tmax = INF_VALUE
    
    for i in ti.static(range(3)):
        if abs(direction[i]) < 0.000001:
            if ( (origin[i] < minv[i]) | (origin[i] > maxv[i])):
                ret = 0
        else:
            ood = 1.0 / direction[i] 
            t1 = (minv[i] - origin[i]) * ood 
            t2 = (maxv[i] - origin[i]) * ood
            if(t1 > t2):
                temp = t1 
                t1 = t2
                t2 = temp 
            if(t1 > tmin):
                tmin = t1
            if(t2 < tmax):
                tmax = t2 
            if(tmin > tmax) :
                ret=0
    return ret
    
    

    '''
    ret = 0
    t0 = (minv - origin) / direction
    t1 = (maxv - origin) / direction
    tmin = min(t0,t1)
    tmax = max(t0,t1)
    if max_component(tmin) <= min_component(tmax):
        ret = 1
    return ret
    '''

@ti.func
def expandBits( x):
    '''
    # nvidia blog : https://developer.nvidia.com/blog/thinking-parallel-part-iii-tree-construction-gpu/
    v = ( (v * 0x00010001) & 0xFF0000FF)
    v = ( (v * 0x00000101) & 0x0F00F00F)
    v = ( (v * 0x00000011) & 0xC30C30C3)
    v = ( (v * 0x00000005) & 0x49249249)
    taichi can not handle it, so i change that to bit operate
    '''
    x = (x | (x << 16)) & 0x030000FF
    x = (x | (x <<  8)) & 0x0300F00F
    x = (x | (x <<  4)) & 0x030C30C3
    x = (x | (x <<  2)) & 0x09249249
    return x


@ti.func
def common_upper_bits(lhs, rhs) :
    x    = lhs ^ rhs
    ret  = 0

    while 1:
        find  = x>>(31-ret)
        if (find == 1) |(ret == 31):
            ret  +=1
            break 
        ret  +=1
        #print(ret, lhs, rhs, x, find, ret)
    return ret

@ti.func
def morton3D(x, y, z):
    x = min(max(x * 1024.0, 0.0), 1023.0)
    y = min(max(y * 1024.0, 0.0), 1023.0)
    z = min(max(z * 1024.0, 0.0), 1023.0)
    xx = expandBits(ti.cast(x, dtype = ti.i32))
    yy = expandBits(ti.cast(y, dtype = ti.i32))
    zz = expandBits(ti.cast(z, dtype = ti.i32))
    #return zz  | (yy << 1) | (xx<<2)
    code = xx  | (yy << 1) | (zz<<2)
    if code == 0:
        print(x,y,z)
    return code