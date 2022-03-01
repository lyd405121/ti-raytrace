import taichi as ti
import math
import numpy as np
import taichi_glsl as ts
RGB2SPEC_N_COEFFS = 3
@ti.data_oriented
class Rgb2Spec:
    def __init__(self):
        self.table_data  = ti.field(dtype=ti.f32)
        self.table_scale = ti.field(dtype=ti.f32)
        self.table_res   = 0
        self.table_size  = 0

    
    def load_table(self, table_path):

        index = 0
        for line in open(table_path, "r"):
            if index == 0:
                self.table_res      = int(line)
                self.table_size     = self.table_res*self.table_res*self.table_res*9
                self.dx             = 3
                self.dy             = 3 * self.table_res
                self.dz             = 3 * self.table_res * self.table_res
                self.table_scale_np = np.ones(shape=(self.table_res), dtype=np.float32)
                self.table_data_np  = np.ones(shape=(self.table_size), dtype=np.float32)
            elif index <= self.table_res:
                self.table_scale_np[index-1] = float(line)
            else:
                values = line.split(' ', 9)
                for i in range(9):
                    idx = (index-1-self.table_res) * 9 + i
                    self.table_data_np[idx] = float(values[i])
            index += 1
        #print(self.table_data_np)
        ti.root.dense(ti.i, (self.table_size    ) ).place(self.table_data)
        ti.root.dense(ti.i, (self.table_res     ) ).place(self.table_scale)

    
    def setup_data_gpu(self):
        self.table_data.from_numpy(self.table_data_np)
        self.table_scale.from_numpy(self.table_scale_np)

    @ti.func
    def fma(self, a,  b,  c):
        return a*b + c

    @ti.func
    def get_max_component(self, rgb):
        index = 0
        xyz = rgb
        if rgb[1] > rgb[0]:
            if rgb[2] > rgb[1]:
                index = 2
            else:
                index = 1
                xyz[0] = rgb[2]
                xyz[1] = rgb[0]
                xyz[2] = rgb[1]
        else:
            if rgb[2] > rgb[0]:
                index = 2
            else:
                index = 0
                xyz[0] = rgb[1]
                xyz[1] = rgb[2]
                xyz[2] = rgb[0]
        xyz[2] = max(0.00001, xyz[2]) 
        scale = float(self.table_res - 1) / xyz[2]
        xyz[0]    *= scale
        xyz[1]    *= scale
        return index,xyz[0],xyz[1],xyz[2]

    @ti.func
    def tri_linear(self, i, x0, y0, z0):
        dx, dy, dz = ti.static(self.dx, self.dy, self.dz)
        return ts.mix(ts.mix( ts.mix(self.table_data[i], self.table_data[i+dx], x0), ts.mix(self.table_data[i+dy], self.table_data[i+dy+dx], x0), y0),ts.mix( ts.mix(self.table_data[i+dz], self.table_data[i+dz+dx], x0), ts.mix(self.table_data[i+dy+dz], self.table_data[i+dx+dy+dz], x0), y0), z0)
        #return ts.mix(ts.mix(ts.mix(self.table_data[i+dx+dy+dz], self.table_data[i+dy+dz], x0), ts.mix(self.table_data[i+dz+dx], self.table_data[i+dz], x0),  y0),ts.mix( ts.mix(self.table_data[i+dy+dx], self.table_data[i+dy], x0), ts.mix(self.table_data[i+dx], self.table_data[i], x0),  y0), z0)

    @ti.func
    def find_interval(self,  size,  x):
        left = 0
        last_interval = size - 2
        size = last_interval

        while (size > 0):
            half   = size >> 1
            middle = left + half + 1

            if (self.table_scale[middle] <= x):
                left = middle
                size -= half + 1
            else:
                size = half
        return min(left, last_interval)

    @ti.func
    def fetch(self, rgb):
        rgb         = ts.clamp(rgb, 0.0, 1.0)
        index,x,y,z = self.get_max_component(rgb)
        
        out         = ti.Vector([0.0, 0.0, 0.0])
         
        xi = int(min(x, self.table_res - 2))
        yi = int(min(y, self.table_res - 2))
        zi = int(self.find_interval(self.table_res, z))
        offset = int((((index * self.table_res + zi) * self.table_res + yi) * self.table_res + xi) * RGB2SPEC_N_COEFFS)

        
        x0 = (x - xi)
        y0 = (y - yi)
        z0 = (z - self.table_scale[zi]) /(self.table_scale[zi + 1] - self.table_scale[zi])
        
        out[0] = self.tri_linear(offset,  x0, y0, z0)
        out[1] = self.tri_linear(offset+1,x0, y0, z0)
        out[2] = self.tri_linear(offset+2,x0, y0, z0)
        
        '''
        x1 = (x - xi)
        y1 = (y - yi)
        z1 = (z - self.table_scale[zi]) /(self.table_scale[zi + 1] - self.table_scale[zi])
        x0 = 1.0 - x1
        y0 = 1.0 - y1
        z0 = 1.0 - z1
        for j  in ti.static(range(RGB2SPEC_N_COEFFS)):
            out[j] = ((self.table_data[offset                    ] * x0 + self.table_data[offset + self.dx                    ] * x1) * y0 +      \
                      (self.table_data[offset + self.dy          ] * x0 + self.table_data[offset + self.dy + self.dx          ] * x1) * y1) * z0 +\
                     ((self.table_data[offset + self.dz          ] * x0 + self.table_data[offset + self.dz + self.dx          ] * x1) * y0 +      \
                      (self.table_data[offset + self.dz + self.dy] * x0 + self.table_data[offset + self.dz + self.dy + self.dx] * x1) * y1) * z1
            offset+= 1
        '''
        return out

    @ti.func
    def eval(self, coff, Lambda):
        x = self.fma(self.fma(coff[0], Lambda, coff[1]), Lambda, coff[2])
        y = 1.0 / ti.sqrt(self.fma(x, x, 1.0))
        return self.fma(0.5 * x, y, 0.5)