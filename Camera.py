import taichi as ti
import math
import numpy as np
import taichi as ti
import taichi_glsl as ts
import UtilsFunc as UF

# FULL_FRAME_CAMERA_PARAM
FULL_HGT = 2.4

@ti.data_oriented
class Camera:
    def __init__(self, sizex, sizey, sample_count):
        
        self.frame_gpu  = ti.field(dtype=ti.i32, shape=(1))
        self.view       = ti.Matrix.field(4, 4, dtype=ti.f32, shape=(1))
        self.view_inv   = ti.Matrix.field(4, 4, dtype=ti.f32, shape=(1))

        self.eye        = ti.Vector.field(3, dtype=ti.f32, shape=[1])
        self.screenRes  = ti.Vector([sizex, sizey])


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

        self.yaw        = 0.0
        self.pitch      = 0.0
        self.roll       = 0.0
        self.scale      = 1000.0

        self.sample_count = int(math.sqrt(sample_count))
        self.sample_dis   = 1.0 / float(self.sample_count-1)
        self.frame_cpu    = np.zeros(shape=(1), dtype=np.int32)
        self.frame        = 0
        self.fps          = 30.0


    
    def yaw_cam(self, targetx, targety, targetz):
        self.target[0] = targetx
        self.target[1] = targety
        self.target[2] = targetz
        if self.yaw < 3.14:
            self.set_view_point(self.yaw + 0.003, 0.0, 0.0, 3.0)

    
    def pitch_cam(self, targetx, targety, targetz):
        self.target[0] = targetx
        self.target[1] = targety
        self.target[2] = targetz
        if self.pitch < 0.5:
            self.set_view_point(0.0, self.pitch + 0.003, 0.0, 3.0)

    
    def update(self):
        
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
        view_np[0] = np.array([ [xaxis[0], xaxis[1], xaxis[2], -np.dot(xaxis, self.eye_np[0,:])], \
        [yaxis[0], yaxis[1], yaxis[2], -np.dot(yaxis,self.eye_np[0,:])], \
        [zaxis[0], zaxis[1], zaxis[2], -np.dot(zaxis,self.eye_np[0,:])], [0.0, 0.0, 0.0, 1.0] ])
        self.view.from_numpy(view_np)
        self.view_inv.from_numpy(np.linalg.inv(view_np))
        self.eye.from_numpy(self.eye_np)
            
        #print(view_np)
        #print(self.fx,self.fy, self.cx, self.cy)

    
    def set_view_point(self, yaw, pitch, roll,scale):
        self.pitch = pitch
        self.yaw = yaw
        self.roll = roll
        self.scale = scale
        self.update()

    
    def set_target(self, targetx, targety, targetz):
        self.target[0] = targetx
        self.target[1] = targety
        self.target[2] = targetz
        self.update()


    
    def update_frame(self):
        self.frame += 1
        self.frame_cpu[0] = self.frame 
        self.frame_gpu.from_numpy(self.frame_cpu)
        #print(self.frame_gpu.to_numpy())


    @ti.func
    def get_ray_origin(self):
        return self.eye[0]

    @ti.func
    def get_optical_axis(self):
        return ti.Vector([self.view[0][2,0], self.view[0][2,1], self.view[0][2,2]])

    @ti.func
    def get_ray_direction(self, u, v):
        jx  = 0.0
        jy  = 0.0
        
        if (self.frame_gpu[0] != 0):
            jx  = ti.random() - 0.5
            jy  = ti.random() - 0.5
        x = (u+jx - self.cx) / self.fx
        y = (v+jy - self.cy) / self.fy
        z = -1.0
        wolrd_xyz = self.view_inv[0] @ ti.Vector([x, y, z, 0.0])
        return ti.Vector([wolrd_xyz.x, wolrd_xyz.y, wolrd_xyz.z]).normalized()

    @ti.func
    def get_image_point(self, p):
        pv  = self.view[0] @ ti.Vector([p.x, p.y, p.z, 1.0]) 

        #print(self.view[0] @ ti.Vector([-0.2, 4.0, 4.0, 1.0] ))
        u  = int(-pv.x/pv.z * self.fx + self.cx)
        v  = int(-pv.y/pv.z * self.fy + self.cy)
        wi = ti.Vector([0.0, 0.0, 0.0])  
        if (u<0) | (u>=self.wid) | (v<0) | (v>=self.hgt) | (pv.z > 0.0 ):
            u = -1
            v = -1
        else:
            wi = p - self.eye[0] 

        return ti.Vector([u, v]), wi.normalized()

