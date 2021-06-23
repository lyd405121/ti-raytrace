import taichi as ti
import math
import numpy as np
import taichi_glsl as ts
import UtilsFunc as UF

@ti.data_oriented
class Spectrum:
    def __init__(self):
        self.data         = ti.field(dtype=ti.f32)
        self.lambda_min   = 10000
        self.lambda_max   = 0
        self.lambda_range = 0
        self.size         = 0
        self.white_point  = ti.Vector.field(3, dtype=ti.f32, shape=(1))
        
    @ti.pyfunc
    def load_table(self, table_path):
        Data   = []
        for line in open(table_path, "r"):
            values = line.split(',', 2)
            v1     = float(values[1])
            v0     = float(values[0])
            Data.append(v1)
            if self.size == 0:
                self.lambda_min = v0
            self.lambda_max = v0
            self.size += 1
        
        self.data_np  = np.zeros(shape=(self.size), dtype=np.float32)
        for i in range(self.size):
            self.data_np[i] = Data[i]
        self.lambda_range = (self.lambda_max - self.lambda_min) / (self.size -1)
        ti.root.dense(ti.i, (self.size) ).place(self.data)


    @ti.pyfunc
    def setup_data_gpu(self):
        self.data.from_numpy(self.data_np)



    @ti.func
    def sample(self, Lambda):
        ret    = 0.0
        if  (Lambda>= self.lambda_min) & (Lambda <= self.lambda_max):
            offset = Lambda - self.lambda_min
            idx    = int (offset / self.lambda_range)
            w      = ts.fract(offset)
            ret    = ts.mix(self.data[idx], self.data[idx+1], w)
        return ret

    @ti.kernel
    def scale(self, coff:ti.f32):
        for i in self.data:
            self.data[i] *= coff

