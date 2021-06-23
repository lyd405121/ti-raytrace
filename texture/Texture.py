import taichi as ti
import math
import numpy as np
import cv2 as cv
import struct
import taichi_glsl as ts

@ti.data_oriented
class Texture:
    def __init__(self):
        self.buf        = ti.field(dtype=ti.i32)
        self.wid        = 0
        self.hgt        = 0
        self.channel    = 0
        self.size       = 0

    @ti.pyfunc
    def load_image(self, imagePath):
        img1 = cv.imread(imagePath)
        dim  = img1.shape
        self.wid        = dim[1]
        self.hgt        = dim[0]
        self.channel    = dim[2]
        self.size       = self.wid*self.hgt*self.channel

        ti.root.dense(ti.ij, (self.wid,self.hgt ) ).place(self.buf)
        self.np_img = np.zeros(shape=(self.wid,self.hgt ), dtype=np.int32)

        for i in range (self.hgt ):
            for j in range ( self.wid ):
                B = img1[i, j, 0]
                G = img1[i, j, 1]
                R = img1[i, j, 2]
                self.np_img[j, self.hgt - i -1] = (R<<16) | (G<<8)  | (B) 


    @ti.pyfunc
    def setup_data_gpu(self):
        self.buf.from_numpy(self.np_img)

    @ti.func
    def sample(self, v):
        x    = ts.clamp(int(v[0]), 0, self.wid -1)
        y    = ts.clamp(int(v[1]), 0, self.hgt -1)
        RGBA = self.buf[x, y]
        R = float((RGBA&0x00FF0000)>>16)/255.0
        G = float((RGBA&0x0000FF00)>>8)/255.0
        B = float((RGBA&0x000000FF)) /255.0
        return ti.Vector([R, G, B])

    @ti.func
    def texture2D(self, u, v):
        x = ts.clamp(u * self.wid, 0.0, self.wid -1.0)
        y = ts.clamp(v * self.hgt, 0.0, self.hgt -1.0)
        #   lt       rt
        #    *--------*
        #    |   ↑wbt |
        #    | ← *    |
        #    | wlr    |
        #    *--------*
        #   lb       rb
        lt   = ti.Vector([ti.floor(x), ti.floor(y)])
        rt   = lt + ti.Vector([1,0])
        lb   = lt + ti.Vector([0,1])
        rb   = lt + ti.Vector([1,1])
        wbt  = ts.fract(y)
        wlr  = ts.fract(x)
        #print(x,y,lt,wbt,wlr)
        return ts.mix( ts.mix(self.sample(lt), self.sample(rt), wlr),  ts.mix(self.sample(lb), self.sample(rb), wlr), wbt)