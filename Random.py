import sys
import os
import taichi as ti
import math
import UtilsFunc as UF
import taichi_glsl as ts


@ti.data_oriented
class Random:
    def __init__(self):
        self.seed      = ti.field(dtype=ti.i32)

    @ti.pyfunc
    def setup_data_cpu(self, wid, hgt):
        self.wid = wid
        self.hgt = hgt
        ti.root.dense(ti.ij, (wid, hgt ) ).place(self.seed)
        


    @ti.func
    def tea(self, i, j, v0, N):
        self.seed[i,j] = v0
        v1             = i*self.wid + j
        
        s0 = 0
        n  = 0
        '''
        while n<N:
            s0 += 0x9e3779b9
            self.seed[i,j] += ((v1 << 4) + 0xa341316c) ^ (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4)
            v1 += ((self.seed[i,j] << 4) + 0xad90777d) ^ (self.seed[i,j] + s0) ^ ((self.seed[i,j] >> 5) + 0x7e95761e)
            n+=1
        '''
        while n<N:
            s0 += 0x013779b9
            self.seed[i,j] += ((v1 << 4) + 0x0141316c) ^ (v1 + s0) ^ ((v1 >> 5) + 0x01013ea4)
            v1 += ((self.seed[i,j] << 4) + 0x0190777d) ^ (self.seed[i,j] + s0) ^ ((self.seed[i,j] >> 5) + 0x0195761e)
            n+=1
        
    @ti.func
    def lcg(self, i, j):
        LCG_A = 1664525
        LCG_C = 1013904223
        self.seed[i,j] = (LCG_A * self.seed[i,j] + LCG_C)
        ret = self.seed[i,j] & 0x00FFFFFF
        return ret
 
    @ti.func
    def lcg2(self, i, j):
        self.seed[i,j] = (self.seed[i,j] * 8121 + 28411) % 134456
        return self.seed[i,j]

    @ti.func
    def rnd(self, i, j):
        m = (float)(self.lcg(i,j))
        n = (float)(0x01000000)
        return m / n

    @ti.kernel
    def init_seed(self, frame:ti.i32):
        for i, j in self.seed:
            self.tea(i, j, frame, 8)