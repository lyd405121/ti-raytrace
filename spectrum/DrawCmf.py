import sys
import os
import time
import math

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d






title_name = 'canon_eos-5d-mkiv'
filename = 'canon_eos-5d-mkiv.csv'


lanmbda = []
spdx    = []
spdy    = []
spdz    = []

for line in open(filename, "r"):
    values = line.split(',', 4)
    lanmbda.append(int(values[0]))
    spdx.append(float(values[1]))
    spdy.append(float(values[2]))
    spdz.append(float(values[3]))

lambda_num = len(lanmbda)
lanmbda_np = np.zeros(shape=(lambda_num,1), dtype=np.float32)
spd_np     = np.zeros(shape=(lambda_num,3), dtype=np.float32)


for i in range(lambda_num):
    lanmbda_np[i, 0] = lanmbda[i]
    spd_np[i, 0]     = spdx[i]
    spd_np[i, 1]     = spdy[i]
    spd_np[i, 2]     = spdz[i]

plt.figure(figsize=(10,10))
plt.plot(lanmbda_np, spd_np[:,0],   color=(1.0, 0.0, 0.0))
plt.plot(lanmbda_np, spd_np[:,1],   color=(0.0, 1.0, 0.0))
plt.plot(lanmbda_np, spd_np[:,2],   color=(0.0, 0.0, 1.0))
plt.title('spectral power distribution')



plt.title(title_name)
plt.savefig(title_name+'.png',dpi=100)
plt.show()
os.system('pause')


