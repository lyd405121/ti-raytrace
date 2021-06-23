import taichi as ti
import taichi_glsl as ts
import math

from taichi_glsl.vector import normalize
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
#            3        spot     r1 r2 scale normal   
#            4        dir      radius X    X     normal    
#                                                
#vertex    : pos_v3   normal_v3 tex_v3                                                  9
#primitive : type(0:tri 1:shape) vertexIndex(shape_index) matIndex                      3

#            32bit         | 32bit    | 32bit       | 32bit      | 32bit     |96bit  |96bit 
#bvh_node  : is_leaf axis  |left_node  right_node   parent_node  prim_index   min_v3  max_v3   11
#             1bit   2bit       

#                32bit         |32bit       |32bit |96bit  |96bit 
#compact_node  : is_leaf axis  |prim_index  offset  min_v3  max_v3   9
#                1bit   2bit

########################### Important !!!! ###############################
AXIS_X            = 0
AXIS_Y            = 1
AXIS_Z            = 2
EPS               = 0.00001
M_PIf             = 3.1415956
INF_VALUE         = 1000000.0
k_B               = 1.38064852e-23
h                 = 6.62607015e-34
c                 = 299792458.0
xyz_to_srgb       = ti.Matrix([[3.240479, -1.537150, -0.498535],[-0.969256,  1.875991,  0.041556],[0.055648, -0.204043,  1.057311]])
srgb_to_xyz       = ti.Matrix([[0.412453, 0.357580, 0.180423 ],[0.212671, 0.715160, 0.072169],[0.019334, 0.119193, 0.950227]])

########################### color function ###############################

#http://brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html
@ti.func
def calc_matr_rgb_to_xyz(xy_r, xy_g, xy_b,XYZ_W):
    x_rgb = ti.Vector([xy_r.x, xy_g.x, xy_b.x])
    y_rgb = ti.Vector([xy_r.y, xy_g.y, xy_b.y])
    
    X_rgb = x_rgb / y_rgb
    Y_rgb = ti.Vector([1.0, 1.0, 1.0])
    Z_rgb = (ti.Vector([1.0, 1.0, 1.0]) - x_rgb - y_rgb ) / y_rgb
    
    S_rgb = ti.Matrix.rows([X_rgb,Y_rgb,Z_rgb]).inverse() @ XYZ_W[0]   
    return ti.Matrix.rows([S_rgb * X_rgb,S_rgb * Y_rgb,S_rgb * Z_rgb])


#https://en.wikipedia.org/wiki/Planck%27s_law#The_law
@ti.pyfunc
def Planck(Lambda, temperature):
    lambda_m = Lambda * 1.0e-9 #nm->m   
    #First radiation constant 2 h c²
    c_1L = 2.0 * h*c*c
    #Second radiation constant h c / k_B
    c_2  = h*c/k_B  
    numer = c_1L
    denom = pow(lambda_m,5.0) * (math.exp( c_2 / (lambda_m*temperature) ) - 1.0)

    value = numer / denom
    return value * 1.0e-9


@ti.func
def srgb_to_lrgb(srgb):
    ret = ti.Vector([0.0, 0.0, 0.0])
    for i in ti.static(range(3)):
        if srgb[i] < 0.04045:
            ret[i] = srgb[i] / 12.92
        else:
            ret[i] = pow((srgb[i] + 0.055) / 1.055, 2.4)
    return ret
   
@ti.func
def lrgb_to_srgb(lrgb):
    ret = ti.Vector([0.0, 0.0, 0.0])
    for i in ti.static(range(3)):
        if lrgb[i] < 0.0031308:
            ret[i] = lrgb[i] * 12.92
        else:
            ret[i] = 1.055 * pow(lrgb[i], 1.0 / 2.4) - 0.055
    return ts.clamp(ret, 0.0, 1.0)

@ti.func
def xyz_to_Yxy(xyz):
    ret = ti.Vector([0.0, 0.0, 0.0])
    coff = xyz[0] + xyz[1]+ xyz[2]
    if coff > 0.0:
        coff = 1.0 / coff
        ret  = ti.Vector([xyz[1], coff * xyz[0], coff * xyz[1]])
    return ret
@ti.func
def Yxy_to_xyz(yxy):
    ret = ti.Vector([0.0, 0.0, 0.0])
    if yxy[2] > 0.0:
        k = yxy[0] / yxy[2]
        ret = ti.Vector([k*yxy[1], yxy[0], k *(1.0 -yxy[1]-yxy[2])])
    return ret

#https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
@ti.func
def tone_ACES(x):
    a = 2.51
    b = 0.03
    c = 2.43
    d = 0.59
    e = 0.14
    return ts.clamp((x*(a*x+b))/(x*(c*x+d)+e),0.0, 1.0)



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
def get_material_metallic(material,  index):
    return material[index][5]
@ti.func
def get_material_roughness(material, index):
    return material[index][6]
@ti.func
def get_material_ior(material, index):
    return material[index][5]
@ti.func
def get_material_extinction(material, index):
    return material[index][6]

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
def set_vertex_normal(vertex,  index, n):
    vertex[index][3] = n[0]
    vertex[index][4] = n[1]
    vertex[index][5] = n[2]

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
def get_shape_xita(shape, index):
    return shape[index][4],shape[index][5]
@ti.func
def get_shape_scale(shape, index):
    return shape[index][6]

@ti.func
def get_shape_v1(shape, index):
    return ti.Vector([shape[index][4], shape[index][5], shape[index][6] ])
@ti.func
def get_shape_v2(shape, index):
    return ti.Vector([shape[index][7], shape[index][8], shape[index][9] ])

@ti.func
def get_shape_normal(shape, index):
    return ti.Vector([shape[index][7], shape[index][8], shape[index][9] ])

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

#            32bit         | 32bit    | 32bit       | 32bit      | 32bit     |96bit  |96bit 
#bvh_node  : is_leaf axis  |left_node  right_node   parent_node  prim_index   min_v3  max_v3   11
#             1bit   2bit     
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
    bvh_node[index][0] = float(int(bvh_node[index][0]) & (0xfffe | type))

@ti.func
def set_node_axis(bvh_node, index, axis):
    axis = axis<<1
    bvh_node[index][0] =float(int(bvh_node[index][0]) & (0xfff9 | type))

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
    return  int(bvh_node[index][0]) & 0x0001 
@ti.func
def get_node_axis(bvh_node, index):
    return  int(bvh_node[index][0]) & 0x0006 

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

#                32bit         |32bit       |32bit |96bit  |96bit 
#compact_node  : is_leaf axis  |prim_index  offset  min_v3  max_v3   9
#                1bit   2bit
@ti.func
def get_compact_node_type(bvh_node, index):
    return  int(bvh_node[index][0]) & 0x0001 
@ti.func
def get_compact_node_axis(bvh_node, index):
    return  int(bvh_node[index][0]) & 0x0006 
@ti.func
def get_compact_node_prim(bvh_node, index):
    return int(bvh_node[index][1])
@ti.func
def get_compact_node_offset(bvh_node, index):
    return int(bvh_node[index][2])
@ti.func
def get_compact_node_min_max(bvh_node, index):
    return ti.Vector([bvh_node[index][3], bvh_node[index][4], bvh_node[index][5] ]),ti.Vector([bvh_node[index][6], bvh_node[index][7], bvh_node[index][8] ])




###################################################################


############algrithm##############

@ti.func
def mapToDisk(u1,u2):
    phi = 0.0
    r   = 0.0
    a   = 2.0 *u1 - 1.0
    b   = 2.0 *u2 - 1.0
    if (a > -b) :
        if (a > b):
            r = a
            phi = (M_PIf / 4.0) * (b / a)
        else :
            r = b
            phi = (M_PIf / 4.0) * (2.0 - a / b)
    else:
        if (a < b):
            r = -a
            phi = (M_PIf / 4.0) * (4.0 + b / a)
        else:
            r = -b
            if b == 0.0:
                phi = 0.0
            else:
                phi = (M_PIf / 4.0) * (6.0 - a / b)
    #print(a,b,r,phi)
    return r, phi


@ti.func
def CosineHemisphere_pdf(cosTheta):
    return max(0.01, cosTheta/M_PIf)

@ti.func
def CosineSampleHemisphere( u1,  u2):
    r = ti.sqrt(u1)
    phi = 2.0*M_PIf * u2
    p =   ti.Vector([0.0,0.0,0.0])
    p.x = r * ti.cos(phi)
    p.y = r * ti.sin(phi)
    p.z = ti.sqrt(max(0.0, 1.0 - p.x*p.x - p.y*p.y))
    return p.normalized()

@ti.func
def CosineSampleHemisphere_pdf( u1,  u2):
    r = ti.sqrt(u1)
    phi = 2.0*M_PIf * u2
    p =   ti.Vector([0.0,0.0,0.0])
    p.x = r * ti.cos(phi)
    p.y = r * ti.sin(phi)
    p.z = ti.sqrt(max(0.0, 1.0 - p.x*p.x - p.y*p.y))
    p   = p.normalized()
    return p, CosineHemisphere_pdf(p.z)

@ti.func
def inverse_transform(dir, N):
    Normal   = N.normalized()
    Binormal = ti.Vector([0.0, 0.0, 0.0])
    if (abs(Normal.x) > abs(Normal.z)):
        Binormal.x = -Normal.y
        Binormal.y = Normal.x
        Binormal.z = 0.0
    else:
        Binormal.x = 0.0
        Binormal.y = -Normal.z
        Binormal.z = Normal.y
    Binormal = Binormal.normalized()
    Tangent  = Binormal.cross(Normal).normalized()
    return dir.x*Tangent + dir.y*Binormal + dir.z*Normal


@ti.func
def sqr(x):
     return x*x
@ti.func
def SchlickFresnel(u):
    m = ts.clamp(1.0-u, 0.0, 1.0)
    m2 = m*m
    return m2*m2*m
@ti.func
def GTR1(NDotH,  a):
    ret =1.0/ M_PIf
    if (a < 1.0):
        a2 = a*a
        t = 1.0 + (a2-1.0)*NDotH*NDotH
        ret = (a2-1.0) / (M_PIf*ti.log(a2)*t)
    return ret 
@ti.func
def GTR2(NDotH,  a):
    a2 = a*a
    t = 1.0 + (a2-1.0)*NDotH*NDotH
    return a2 / (M_PIf * t*t)
@ti.func
def smithG_GGX(NDotv,  alphaG):
    a = alphaG*alphaG
    b = NDotv*NDotv
    return 1.0/(NDotv + ti.sqrt(a + b - a*b))

@ti.func
def refract(InRay, N,  eta):
    suc  = -1.0 
    N_DOT_I = N.dot(InRay)
    k = 1.0 - eta * eta * (1.0 - N_DOT_I * N_DOT_I)
    R = ti.Vector([0.0,0.0,0.0])
    if k > 0.0:
        R = eta * InRay - (eta * N_DOT_I + ti.sqrt(k)) * N
        suc = 1.0
    return R,suc

@ti.func
def schlick(cosine,  index_of_refraction):
    r0 = (1.0 - index_of_refraction) / (1.0 + index_of_refraction)
    r0 = r0 * r0
    return r0 + (1.0 - r0) * pow((1.0 - cosine), 5.0)


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

# sometimes 3d model software will do normal smoothing,
# that will change the true geometry normal,so we use geometry normal as a ref
@ti.func
def faceforward(n,  i,  nref):
    return ts.sign(i.dot(nref)) * n

@ti.func
def srgb_to_lrgb(srgb):
    ret = ti.Vector([0.0, 0.0, 0.0])
    for i in ti.static(range(3)):
        if srgb[i] < 0.04045:
            ret[i] = srgb[i] / 12.92
        else:
            ret[i] = pow((srgb[i] + 0.055) / 1.055, 2.4)
    return ret

# https://refractiveindex.info/?shelf=glass&book=BK7&page=SCHOTT
@ti.func
def get_glass_ior(Lambda):
    Lambda      = Lambda/1000.0
    Lambda2     = Lambda*Lambda
    return      ti.sqrt(1.0 + 1.03961212 * Lambda2/ (Lambda2 -0.00600069867 )+ 0.231792344 * Lambda2/ (Lambda2 -0.0200179144 ) + 1.01046945 * Lambda2/ (Lambda2 -103.560653))

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
    ret  = 32


    while x > 0:
        x  = x>>1
        ret  -=1
        #print(ret, lhs, rhs, x, find, ret)
    #print(x)
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

#https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
@ti.kernel
def tone_map(exposure:ti.f32, input:ti.template(), output:ti.template()):
    for i,j in output:   
        output[i,j] =  lrgb_to_srgb(tone_ACES(input[i,j]*exposure))