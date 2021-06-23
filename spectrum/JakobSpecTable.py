import sys
import os
import time
import math
import taichi as ti
import numpy as np
import taichi_glsl as ts

ti.init(arch=ti.gpu)
ti.init(default_fp=ti.f64)

CIE_LAMBDA_MAX = 830
CIE_LAMBDA_MIN = 360
res            = 64
out_table_size = 3*3*res*res*res
lambda_num     = CIE_LAMBDA_MAX-CIE_LAMBDA_MIN+1
h = float(CIE_LAMBDA_MAX - CIE_LAMBDA_MIN) / float(lambda_num - 1)
RGB2SPEC_EPSILON =  1e-4

XYZ                 = ti.Vector.field(3, dtype=ti.f64, shape=[lambda_num])
rgb_tbl             = ti.Vector.field(3, dtype=ti.f64, shape=[lambda_num])
xyz_white_point     = ti.Vector.field(3, dtype=ti.f64, shape=[1])
debug_value         = ti.field(dtype=ti.f64, shape=[res,res])

rgb_coff            = ti.field( dtype=ti.f64, shape=[out_table_size])
scale               = ti.field( dtype=ti.f64, shape=[res])
lambda_tbl          = ti.field( dtype=ti.f64, shape=[lambda_num])
d65                 = ti.field( dtype=ti.f64, shape=[lambda_num])

xyz_to_srgb         = ti.Matrix([[3.240479, -1.537150, -0.498535],[-0.969256,  1.875991,  0.041556],[0.055648, -0.204043,  1.057311]])
srgb_to_xyz         = ti.Matrix([[0.412453, 0.357580, 0.180423 ],[0.212671, 0.715160, 0.072169],[0.019334, 0.119193, 0.950227]])

XYZ_np    = np.zeros(shape=(lambda_num,3),  dtype=np.float32)
d65_np    = np.zeros(shape=lambda_num,      dtype=np.float32)
lambda_np = np.zeros(shape=lambda_num,      dtype=np.float32)

@ti.func
def smoothstep( x):
    return x * x * (3.0 - 2.0 * x)

@ti.func
def sqr( x):
    return x * x

@ti.func
def sigmoid( x):
    return 0.5 * x / ti.sqrt(1.0 + x * x) + 0.5

@ti.func
def get_rgb(res_num, x,y,l):
    b      = scale[res_num]
    rgb    = ti.Vector([0.0, 0.0, 0.0])
    if l == 0:
        rgb[0] = b
        rgb[1] = x*b
        rgb[2] = y*b    
    elif l == 1:
        rgb[1] = b
        rgb[2] = x*b
        rgb[0] = y*b   
    else:
        rgb[2] = b
        rgb[0] = x*b
        rgb[1] = y*b   
    return rgb

@ti.func
def write_to_result(idx,coeffs):
    c0 = CIE_LAMBDA_MIN
    c1 = 1.0 / (CIE_LAMBDA_MAX - CIE_LAMBDA_MIN)
    A = coeffs[0]
    B = coeffs[1]
    C = coeffs[2]   
    rgb_coff[3*idx + 0] = float(A*(sqr(c1)))
    rgb_coff[3*idx + 1] = float(B*c1 - 2*A*c0*(sqr(c1)))
    rgb_coff[3*idx + 2] = float(C - B*c0*c1 + A*(sqr(c0*c1)))

@ti.func
def get_v_componet(b, p):
    ret = b[0]
    if p == 1:
        ret = b[1]
    elif p == 2:
        ret = b[2]
    return ret

@ti.func
def f( t):
    delta = 6.0 / 29.0
    ret   = 0.0
    if t > (delta*delta*delta):
        ret =  pow(t, 1.0 / 3.0)
    else:
        ret = t / (delta*delta * 3.0) + (4.0 / 29.0)

    return ret

@ti.func
def cie_lab(p):
    XYZ  = srgb_to_xyz @ p
    p[0] = 116.0 *  f(XYZ[1] / xyz_white_point[0][1]) - 16.0
    p[1] = 500.0 * (f(XYZ[0] / xyz_white_point[0][0]) - f(XYZ[1] / xyz_white_point[0][1]))
    p[2] = 200.0 * (f(XYZ[1] / xyz_white_point[0][1]) - f(XYZ[2] / xyz_white_point[0][2]))
    return p

@ti.func
def LUPDecompose(A) :
    Tol = 1e-15
    ret = 1
    P   = ti.Vector([0,1,2,3])

    '''
    for i in ti.static(range(3)):
        maxA = 0.0
        imax = i

        for k in ti.static(range(i,3)):
            absA = ti.abs(A[k,i])
            if absA > maxA:
                maxA = absA
                imax = k

        if (maxA < Tol): 
            ret = 0
            break

        if (imax != i) :
            P = swap_V(P, i, imax)
            A = swap_M(A, i, imax)
            P[3] += 1

        for j in ti.static(range(i+1,3)):
            A[j,i] /= A[i,i]
            for k in ti.static(range(i+1,3)):
                A[j,k] -= A[j,i] * A[i,k]
    '''

    ########### i =0 ##############################
    maxA = 0.0
    imax = 0

    for k in ti.static(range(3)):
        absA = ti.abs(A[k,0])
        if absA > maxA:
            maxA = absA
            imax = k

    if (maxA < Tol): 
        ret = 0

    if (imax != 0) :
        if imax == 1:
            P = ti.Vector([P[1],P[0],P[2],P[3]])
            A = ti.Matrix([[A[1,0], A[1,1], A[1,2]], [A[0,0], A[0,1], A[0,2]], [A[2,0], A[2,1], A[2,2]]]) 
        else:#IMAX = 2
            P = ti.Vector([P[2],P[1],P[0],P[3]])
            A = ti.Matrix([[A[2,0], A[2,1], A[2,2]],[A[1,0], A[1,1], A[1,2]],[A[0,0], A[0,1], A[0,2]]]) 
        P[3] += 1

    # j = 1
    A[1,0] /= A[0,0]
    A[1,1] -= A[1,0] * A[0,1]
    A[1,2] -= A[1,0] * A[0,2]

    # j = 2
    A[2,0] /= A[0,0]
    A[2,1] -= A[2,0] * A[0,1]
    A[2,2] -= A[2,0] * A[0,2]

    
    ########### i =1 ##############################
    maxA = 0.0
    imax = 1

    for k in ti.static(range(1,3)):
        absA = ti.abs(A[k,0])
        if absA > maxA:
            maxA = absA
            imax = k

    if (maxA < Tol): 
        ret = 0

    if (imax != 1) :#IMAX = 2
        P = ti.Vector([P[0],P[2],P[1],P[3]])
        A = ti.Matrix([[A[0,0], A[0,1], A[0,2]],[A[2,0], A[2,1], A[2,2]],[A[1,0], A[1,1], A[1,2]]]) 
        P[3] += 1

    # j = 2
    A[2,1] /= A[1,1]
    A[2,2] -= A[2,1] * A[1,2]

    #print(A)

    ########### i =2 ##############################
    if (ti.abs(A[2,2]) < Tol): 
        ret = 0

    
    B = A

    '''
    for k in ti.static(range(3)):
        if P[k] == 0:
            B[0, 0] = A[k,0] 
            B[0, 1] = A[k,1] 
            B[0, 2] = A[k,2] 
        elif P[k] == 1:
            B[1, 0] = A[k,0] 
            B[1, 1] = A[k,1] 
            B[1, 2] = A[k,2] 
        else:
            B[2, 0] = A[k,0] 
            B[2, 1] = A[k,1] 
            B[2, 2] = A[k,2] 
    '''
    return ret,B,P


@ti.func
def LUPSolve(A, P, b):
    x = ti.Vector([0.0, 0.0, 0.0])
    '''
    # can not compile
    for i in ti.static(range(3)):
        x = change_V(x, b, P[i])
        for k in ti.static(range(i)):
            x[i] -= A[i,k] * x[k]
    '''

    ### i= 0, k=[0,0)
    x[0] = get_v_componet(b, P[0])

    ###i = 1, k=[0,1)
    x[1] = get_v_componet(b, P[1])
    x[1] -= A[1,0] * x[0]

    ###i = 2, k=[0,2)
    x[2] = get_v_componet(b, P[2])
    x[2] -= A[2,0] * x[0]
    x[2] -= A[2,1] * x[1]

    '''
    for i in ti.static(range(2, -1)):
        for k in ti.static(range(i+1, 3)):
            x[i] -= A[i,k] * x[k]
        x[i] = x[i] / A[i,i]
    '''

    ###i = 2, k=[3,3)
    x[2] = x[2] / A[2,2]

    ###i = 1, k=[2,3) 
    x[1] -= A[1,2] * x[2]
    x[1] = x[1] / A[1,1]

    ###i = 0, k=[1,3)
    x[0] -= A[0,1] * x[1]
    x[0] -= A[0,2] * x[2]
    x[0]  = x[0] / A[0,0]


    return x

@ti.func
def eval_residual(coeffs, rgb):
    out = ti.Vector([0.0, 0.0, 0.0])
    i = 0
    while i < lambda_num:

        Lambda = (lambda_tbl[i] - CIE_LAMBDA_MIN) / (CIE_LAMBDA_MAX - CIE_LAMBDA_MIN)
        #print(Lambda)

        x = coeffs[0]
        x = x * Lambda + coeffs[1]
        x = x * Lambda + coeffs[2]

        s = sigmoid(x)
        out += rgb_tbl[i] * s
        i += 1
    return cie_lab(rgb)-cie_lab(out)


@ti.func
def eval_jacobian(coeffs, rgb):

    jac         = ti.Matrix([[0.0, 0.0, 0.0],[0.0, 0.0, 0.0],[0.0, 0.0, 0.0]])
    for i in ti.static(range(3)):
        tmp = coeffs
        tmp[i] -= RGB2SPEC_EPSILON
        r0 = eval_residual(tmp, rgb)

        tmp = coeffs
        tmp[i] += RGB2SPEC_EPSILON
        r1 = eval_residual(tmp, rgb)

        jac[0, i] = (r1[0] - r0[0]) / (2.0 * RGB2SPEC_EPSILON)
        jac[1, i] = (r1[1] - r0[1]) / (2.0 * RGB2SPEC_EPSILON)
        jac[2, i] = (r1[2] - r0[2]) / (2.0 * RGB2SPEC_EPSILON)
        
    return jac


@ti.func
def gauss_newton(rgb, coeffs, res_num):
    r        = 0.0
    iter_num = 0
    rv       = 1
    while iter_num < 15:
        residual = eval_residual(coeffs, rgb)
        J = eval_jacobian(coeffs, rgb)
        rv,J,P = LUPDecompose(J)
        
        
        if (rv != 1):
            print(res_num, iter_num, r, coeffs, rgb)
            break
        

        x = LUPSolve(J, P, residual)
        

        coeffs  -=  x
        r        =   residual.dot(residual)

        if (r < 0.000001):
            break

        coff_max = ti.max(ti.max(coeffs[0], coeffs[1]), coeffs[2])
        if (coff_max > 200.0):
            coeffs = coeffs * 200.0/ coff_max


        #print(res_num, iter_num, r, coeffs, rgb)
        iter_num += 1
    return rv,coeffs 

@ti.kernel
def pre_compute():
    for i in XYZ:
        weight = 3.0 / 8.0 * h
        if (i ==0) | (i == lambda_num-1):
            weight = weight
        elif ((i-1)%3 == 2):
            weight = weight * 2.0
        else:
            weight = weight * 3.0
        rgb_tbl[i]         += xyz_to_srgb @  XYZ[i] * d65[i] * weight
        xyz_white_point[0] += XYZ[i] * d65[i] * weight
    
    for i in scale:
        scale[i]= smoothstep(smoothstep(i / float(res - 1)))

@ti.kernel
def sovle(l:ti.i32):
    for i,j in debug_value:
        x = float(i) / float(res - 1)
        y = float(j) / float(res - 1)

        coeffs = ti.Vector([0.0, 0.0, 0.0])
        k     =   int(res / 5)      
        while k < res:
            
            rv,coeffs = gauss_newton(get_rgb(k,x,y,l), coeffs, k)
            if (rv != 1):
                print(i,j,k,"wrong")
                k = res
                break
            idx = ((l*res + k) * res + j)*res+i 
            write_to_result(idx,coeffs)
            k += 1

        
        k     =   int(res / 5)    
        coeffs = ti.Vector([0.0, 0.0, 0.0])
        while k >= 0:
            rv,coeffs = gauss_newton(get_rgb(k,x,y,l) , coeffs, k)
            if (rv != 1):
                print(i,j,k,"wrong")
                k = res
                break
            idx = ((l*res + k) * res + j)*res+i    
            write_to_result(idx,coeffs)
            k -= 1
            
def read_data():
    # csv data come from below:
    # http://cvrl.ioo.ucl.ac.uk/
    index = 0
    for line in open("ciexyz31_1.csv", "r"):
        values = line.split(',', 4)
        lambda_np[index] = int(values[0])
        XYZ_np[index,0]  = float(values[1])
        XYZ_np[index,1]  = float(values[2])
        XYZ_np[index,2]  = float(values[3])
        index += 1

    index = 0
    for line in open("Illuminantd65.csv", "r"):
        values = line.split(',', 2)
        lambda_index = int(values[0])
        if lambda_index >= 360:
            d65_np[index]  = float(values[1])
            #print(lambda_np[index], XYZ_np[index], d65_np[index])
            index += 1
    XYZ.from_numpy(XYZ_np)
    lambda_tbl.from_numpy(lambda_np)
    d65.from_numpy(d65_np)

read_data()

pre_compute()
scale_np           = scale.to_numpy()
xyz_white_point_np = xyz_white_point.to_numpy()
xyz_white_point.from_numpy(xyz_white_point_np /  xyz_white_point_np[0, 1])
rgb_tbl_np         = rgb_tbl.to_numpy()/ xyz_white_point_np[0, 1]
rgb_tbl.from_numpy(rgb_tbl_np )

sovle(0)
sovle(1)
sovle(2)


output_coff = rgb_coff.to_numpy()
fo = open("spec_table", "w")
print ("%d" %  (res), file = fo)
for i in range(0, res):
    print ("%.9g " %  (scale_np[i]), file = fo)
for i in range(0, out_table_size, 9):
    print ("%.9g %.9g %.9g %.9g %.9g %.9g %.9g %.9g %.9g" %  (output_coff[i],output_coff[i+1],output_coff[i+2],\
    output_coff[i+3],output_coff[i+4],output_coff[i+5],output_coff[i+6],output_coff[i+7],output_coff[i+8]), file = fo)
fo.close()

'''
fo = open("rgb_tbl.txt", "w")
for i in range(lambda_num):
    print ("%.9g %.9g %.9g" %  (rgb_tbl_np[i, 0],rgb_tbl_np[i, 1],rgb_tbl_np[i, 2]), file = fo)
fo.close()
'''
