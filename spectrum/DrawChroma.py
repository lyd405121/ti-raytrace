import sys
import os
import time
import math

import taichi as ti
import numpy as np
import taichi_glsl as ts
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

ti.init(arch=ti.gpu)
imgSizeX   = 512
imgSizeY   = 512
lambda_num = 471
random_point_num = 100
#gui        = ti.GUI('Chromaticity diagram', res=(imgSizeX, imgSizeY))

xyz                 = ti.Vector.field(3, dtype=ti.f32, shape=[lambda_num])
XYZ                 = ti.Vector.field(3, dtype=ti.f32, shape=[lambda_num])

# x,y,is_inside_poly,r,g,b
random_point        = ti.Vector.field(6, dtype=ti.f32, shape=[random_point_num, random_point_num])

Lambda              = ti.field(dtype=ti.f32, shape=[lambda_num])
xyz_to_srgb         = ti.Matrix([[3.240479, -1.537150, -0.498535],[-0.969256,  1.875991,  0.041556],[0.055648, -0.204043,  1.057311]])
xyz_to_ciergb       = ti.Matrix([[2.689989, -1.276020, -0.413844],[-1.022095,  1.978261,  0.043821],[0.061203, -0.224411,  1.162859]])
xyz_to_rec2020      = ti.Matrix([[1.7166511880, -0.3556707838, -0.2533662814],[-0.6666843518,  1.6164812366,  0.0157685458],[0.0176398574, -0.0427706133,  0.9421031212]])

xyz_np  = np.zeros(shape=(lambda_num,3), dtype=np.float32)
XYZ_np  = np.zeros(shape=(lambda_num,3), dtype=np.float32)
d65_np  = np.zeros(shape=lambda_num, dtype=np.float32)
lambda_np = np.zeros(shape=lambda_num, dtype=np.int32)

@ti.func
def is_inside_chroma(x,y):
    ret = 0
    head = 0
    tail = lambda_num-1

    while head < lambda_num:
        xi = xyz[head][0]
        xj = xyz[tail][0]
        yi = xyz[head][1]
        yj = xyz[tail][1]
        if (( (yi>y) != (yj>y)) & (x < (xj-xi) * (y-yi) / (yj-yi) + xi) ):
            ret = 1- ret
        tail = head
        head += 1
        

    return ret


@ti.kernel
def XYZ_to_xy():
    for i in XYZ:
        xyz[i] =  XYZ[i] / XYZ[i].sum()

        

@ti.kernel
def scatter_point():
    for i,j in random_point:
        random_point[i,j][0] = float(i) / float(random_point_num)
        random_point[i,j][1] = float(j) / float(random_point_num)
        random_point[i,j][2] = float(is_inside_chroma(random_point[i,j][0], random_point[i,j][1]))

        rgb = ts.clamp(xyz_to_rec2020 @ ti.Vector([random_point[i,j][0],random_point[i,j][1],1.0-random_point[i,j][0]-random_point[i,j][1]]), 0.0, 1.0)
        random_point[i,j][3] = rgb.x
        random_point[i,j][4] = rgb.y
        random_point[i,j][5] = rgb.z

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
    Lambda.from_numpy(lambda_np)

read_data()
XYZ_to_xy()
scatter_point()

xyz_np = xyz.to_numpy()
point_np = random_point.to_numpy()



#fig=plt.figure()
#ax = axes3d.Axes3D(fig)
#for i in range(lambda_num):
#    ax.scatter3D(XYZ_np[i,0], XYZ_np[i,1],  XYZ_np[i,2], color=( rgb_np[i,0], rgb_np[i,1], rgb_np[i,2] )) 

plt.figure(figsize=(10,10))
plt.subplot(2, 2, 1)
plt.plot(lambda_np, XYZ_np[:,0],   color=(1.0, 0.0, 0.0))
plt.plot(lambda_np, XYZ_np[:,1],   color=(0.0, 1.0, 0.0))
plt.plot(lambda_np, XYZ_np[:,2],   color=(0.0, 0.0, 1.0))
plt.title('cie 1931 xyz')

plt.subplot(2, 2, 2)
plt.plot(lambda_np, d65_np,   color=(0.9, 0.9, 0.9))
plt.title('cie d65')

plt.subplot(2, 2, 3)

plt.axis("equal")
plt.xlim(0.0, 0.8)
plt.ylim(0.0, 0.9)

for i in range(random_point_num):
    for j in range(random_point_num):
        if point_np[i,j,2] > 0.5:
            plt.scatter(point_np[i,j,0], point_np[i,j,1], color=( point_np[i,j,3], point_np[i,j,4], point_np[i,j,5] ))
plt.plot(xyz_np[:,0], xyz_np[:,1],color=(0.0, 0.0, 0.0))
plt.plot([xyz_np[0,0],xyz_np[lambda_num-1,0]], [xyz_np[0,1],xyz_np[lambda_num-1,1]],color=(0.0, 0.0, 0.0))

plt.title('chroma')
plt.savefig('chroma.png',dpi=100)
plt.show()


os.system('pause')


