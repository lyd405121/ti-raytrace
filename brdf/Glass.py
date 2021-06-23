import sys
import os
import taichi as ti
import math
import UtilsFunc as UF
import taichi_glsl as ts


@ti.func
def sample(dir, N, t,  mat_buf, mat_id):
    next_dir    = ti.Vector([0.0, 0.0, 0.0])
    w_out       = dir
    cos_theta_i = w_out.dot(N)
    ior         = UF.get_material_ior(mat_buf, mat_id)
    eta         = ior 
    
    probability = ti.random()   
    f_or_b      = 1.0
    R           = probability + 1.0
    
    if (cos_theta_i > 0.0):
        N      = -N
    else:
        cos_theta_i = -cos_theta_i
        eta = 1.0 /ior
        
    next_dir,suc = UF.refract(w_out, N, eta)
    if suc > 0.0:
        R = UF.schlick(cos_theta_i, ior)
    if (probability < R):
        next_dir = ts.reflect(w_out, N)
    else:
        f_or_b   = -1.0
    return next_dir, f_or_b




@ti.func
def sample_lambda(dir, N, t,  mat_buf, mat_id, Lambda):
    next_dir    = ti.Vector([0.0, 0.0, 0.0])
    w_out       = dir
    cos_theta_i = w_out.dot(N)
    ior         = UF.get_glass_ior(Lambda)
    eta         = ior  
    
    probability = ti.random()   
    f_or_b      = 1.0
    R           = probability + 1.0
    
    if (cos_theta_i > 0.0):
        N      = -N
        temp   = UF.get_material_extinction(mat_buf, mat_id)
    else:
        cos_theta_i = -cos_theta_i
        eta = 1.0 /ior
        
    next_dir,suc = UF.refract(w_out, N, eta)
    if suc > 0.0:
        R = UF.schlick(cos_theta_i, ior)
    if (probability < R):
        next_dir = ts.reflect(w_out, N)
    else:
        f_or_b   = -1.0
    return next_dir, f_or_b


@ti.func
def pdf(N, V, L, mat_buf, mat_id):
    return 1.0

@ti.func
def evaluate_pdf(N, V, L, mat_buf, mat_id):
    return 1.0, 1.0

@ti.func
def evaluate(N, V, L, mat_buf, mat_id):
    return 1.0
