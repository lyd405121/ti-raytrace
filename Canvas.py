import taichi as ti
import math
import numpy as np
import taichi as ti
import taichi_glsl as ts
# FULL_FRAME_CAMERA_PARAM
FULL_HGT = 2.4
STACK_SIZE = 32
@ti.data_oriented
class Canvas:
    def __init__(self, sizex, sizey):
        
        self.frame_gpu  = ti.field(dtype=ti.i32, shape=(1))
        self.view       = ti.Matrix.field(4, 4, dtype=ti.f32, shape=(1))
        self.view_inv   = ti.Matrix.field(4, 4, dtype=ti.f32, shape=(1))

        self.eye        = ti.Vector.field(3, dtype=ti.f32, shape=[1])
        self.screenRes  = ti.Vector([sizex, sizey])
        self.hdr        = ti.Vector.field(3, dtype=ti.f32, shape=[sizex, sizey])
        self.hdr_reduce_min = ti.Vector.field(3, dtype=ti.f32, shape=[sizex//2+1, sizey//2+1])
        self.hdr_reduce_max = ti.Vector.field(3, dtype=ti.f32, shape=[sizex//2+1, sizey//2+1])
        self.img        = ti.Vector.field(3, dtype=ti.f32, shape=[sizex, sizey])


        self.stack      = ti.field(dtype=ti.i32, shape=[sizex, sizey, STACK_SIZE])
        self.wid = sizex
        self.hgt  = sizey

        self.focal      = 2.0
        self.ratio      = sizex / sizey

        #pinhole camera
        #https://www.microsoft.com/en-us/research/publication/a-flexible-new-technique-for-camera-calibration/
        self.fx         = self.focal * sizex / FULL_HGT
        self.fy         = self.fx 
        self.cx         = sizex * 0.5
        self.cy         = sizey * 0.5


        self.eye_np     = np.ones(shape=(1,3), dtype=np.float32)
        self.target     = np.array([0.0, 0.0, 0.0])
        self.up         = np.array([0.0, 1.0, 0.0])

        self.yaw        = 3.14
        self.pitch      = 0.0
        self.roll       = 0.0
        self.scale      = 1000.0

        self.frame_cpu  = np.zeros(shape=(1), dtype=np.int32)
        self.frame      = 0
        self.fps        = 30.0
        


    @ti.pyfunc
    def yaw_cam(self, targetx, targety, targetz):
        self.target[0] = targetx
        self.target[1] = targety
        self.target[2] = targetz
        if self.yaw < 3.14:
            self.set_view_point(self.yaw + 0.003, 0.0, 0.0, 3.0)

    @ti.pyfunc
    def pitch_cam(self, targetx, targety, targetz):
        self.target[0] = targetx
        self.target[1] = targety
        self.target[2] = targetz
        if self.pitch < 0.5:
            self.set_view_point(0.0, self.pitch + 0.003, 0.0, 3.0)

    @ti.pyfunc
    def update_cam(self):
        
        self.pitch = min(self.pitch, 1.57)
        self.pitch = max(self.pitch, -1.57)
        self.eye_np[0,0] = self.target[0] + self.scale * math.cos(self.pitch) * math.sin(self.yaw)
        self.eye_np[0,1] = self.target[1] + self.scale * math.sin(self.pitch)
        self.eye_np[0,2] = self.target[2] + self.scale * math.cos(self.pitch) * math.cos(self.yaw)
        self.up[0]  = -math.sin(self.pitch) * math.sin(self.yaw)
        self.up[1]  = math.cos(self.pitch)
        self.up[2]  = -math.sin(self.pitch) * math.cos(self.yaw)

        zaxis = self.eye_np[0,:] - self.target
        zaxis = zaxis / np.linalg.norm(zaxis) 
        xaxis = np.cross(self.up, zaxis)
        xaxis = xaxis / np.linalg.norm(xaxis)
        yaxis = np.cross(zaxis, xaxis)

        view_np = self.view.to_numpy()
        view_np[0] = ti.np.array([ [xaxis[0], xaxis[1], xaxis[2], -np.dot(xaxis, self.eye_np[0,:])], \
        [yaxis[0], yaxis[1], yaxis[2], -np.dot(yaxis,self.eye_np[0,:])], \
        [zaxis[0], zaxis[1], zaxis[2], -np.dot(zaxis,self.eye_np[0,:])], [0.0, 0.0, 0.0, 1.0] ])
        self.view.from_numpy(view_np)
        self.view_inv.from_numpy(np.linalg.inv(view_np))
        self.eye.from_numpy(self.eye_np)
            
        #print(view_np)

    @ti.pyfunc
    def set_view_point(self, yaw, pitch, roll,scale):
        self.pitch = pitch
        self.yaw = yaw
        self.roll = roll
        self.scale = scale
        self.update_cam()

    @ti.pyfunc
    def set_target(self, targetx, targety, targetz):
        self.target[0] = targetx
        self.target[1] = targety
        self.target[2] = targetz
        self.update_cam()


    @ti.pyfunc
    def update_frame(self):
        self.frame += 1
        self.frame_cpu[0] = self.frame 
        self.frame_gpu.from_numpy(self.frame_cpu)
        #print(self.frame_gpu.to_numpy())


    @ti.pyfunc
    def tone_map_host(self, exposure, gamma):
        #for i in range(9):
        #    self.get_hdr_mimax(pow(2,i))
        self.tone_map(exposure, gamma)

    @ti.func
    def get_ray_origin(self):
        return self.eye[0]

    @ti.func
    def get_ray_direction(self, u, v):
        jx  = 0.0
        jy  = 0.0
        if self.frame_gpu[0] != 0:
            jx  = ti.random() - 0.5
            jy  = ti.random() - 0.5
        x = (u+jx - self.cx) / self.fx
        y = (v+jy - self.cy) / self.fy
        z = -1.0
        wolrd_xyz = self.view_inv[0] @ ti.Vector([x, y, z, 0.0])
        return ti.Vector([wolrd_xyz.x, wolrd_xyz.y, wolrd_xyz.z]).normalized()

    @ti.func
    def get_neigbour_min(self, i, j, mod, src):
        out_min = src[i,j]
        if i +mod< self.wid-1:
            out_min = ti.min(out_min, src[i+mod,j])

        if j+mod < self.hgt-1:
            out_min = ti.min(out_min, src[i,j+mod])

        if (j+mod < self.hgt-1) & (i+mod < self.wid-mod):
            out_min = ti.min(out_min, src[i+mod,j+mod])
        return out_min

    @ti.func
    def get_neigbour_max(self, i, j, mod, src):
        out_max = src[i,j]
        if i+mod < self.wid-1:
            out_max = ti.max(out_max, src[i+mod,j])

        if j+mod < self.hgt-1:
            out_max = ti.max(out_max, src[i,j+mod])

        if (j+mod < self.hgt-1) & (i+mod < self.wid-1):
            out_max = ti.max(out_max, src[i+mod,j+mod])
        return out_max

    @ti.kernel
    def get_hdr_mimax(self, mod:ti.i32):
        for i,j in self.hdr_reduce_min:
            if mod == 1:
                self.hdr_reduce_min[i,j] = self.get_neigbour_min( i,j,mod,self.hdr)
                self.hdr_reduce_max[i,j] = self.get_neigbour_max( i,j,mod,self.hdr)
            else:
                self.hdr_reduce_min[i,j] = self.get_neigbour_min( i,j, mod,self.hdr_reduce_min)
                self.hdr_reduce_max[i,j] = self.get_neigbour_max( i,j, mod,self.hdr_reduce_max)

    #https://github.com/tizian/tonemapper/blob/master/src/operators/aces_unreal.h
    @ti.kernel
    def tone_map(self, exposure:ti.f32, gamma:ti.f32):
        for i,j in self.hdr:
            #r = self.hdr_reduce_max[0,0] - self.hdr_reduce_min[0,0]
            #self.img[i,j] = ts.clamp((self.hdr[i,j]-self.hdr_reduce_min[0,0])/r, 0.0, 1.0)

            rgb = exposure * self.hdr[i,j]
            self.img[i,j] = pow(rgb / (rgb + 0.155) * 1.019, gamma)

            #print(self.hdr[i,j])