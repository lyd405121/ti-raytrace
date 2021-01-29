import sys
import os
import taichi as ti
import math
import UtilsFunc as UF
import taichi_glsl as ts


@ti.data_oriented
class Glass:
    def __init__(self):
        self = self

    @ti.func
    def refract(self, InRay, N,  eta):
        suc  = -1.0 
        N_DOT_I = N.dot(InRay)
        k = 1.0 - eta * eta * (1.0 - N_DOT_I * N_DOT_I)
        R = ti.Vector([0.0,0.0,0.0])
        if k > 0.0:
            R = eta * InRay - (eta * N_DOT_I + ti.sqrt(k)) * N
            suc = 1.0
        return R,suc

    @ti.func
    def schlick(self, cosine,  index_of_refraction):
        r0 = (1.0 - index_of_refraction) / (1.0 + index_of_refraction)
        r0 = r0 * r0
        return r0 + (1.0 - r0) * pow((1.0 - cosine), 5.0)

    @ti.func
    def sample(self, dir, N, t, i, j, mat_buf, mat_id):
        next_dir    = ti.Vector([0.0, 0.0, 0.0])
        w_out       = dir
        cos_theta_i = w_out.dot(N)
        ior         = UF.get_material_ior(mat_buf, mat_id)
        eta         = ior

        probability = ti.random()   
        f_or_b      = 1.0
        R           = probability + 1.0
        extinction  = 1.0

        if (cos_theta_i > 0.0):
            N      = -N
            extinction = ti.exp(-0.1*t)
        else:
            cos_theta_i = -cos_theta_i
            eta = 1.0 /ior
            

        next_dir,suc = self.refract(w_out, N, eta)
        if suc > 0.0:
            R = self.schlick(cos_theta_i, ior)

        if (probability < R):
            next_dir = ts.reflect(w_out, N)
        else:
            f_or_b   = -1.0

        return next_dir, f_or_b*extinction


    @ti.func
    def evaluate(self, N, V, L, mat_buf, mat_id):
        #extinciton  = ti.exp(-0.5 * t)
        return UF.get_material_color(mat_buf, mat_id), 1.0
