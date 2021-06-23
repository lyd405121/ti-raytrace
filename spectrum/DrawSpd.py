import sys
import os
import time
import math

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d





#title_name = 'red'
#filename = 'red-spec.csv'

title_name = 'd55'
filename = 'd55.csv'


lanmbda = []
spd     = []


for line in open(filename, "r"):
    values = line.split(',', 2)
    lanmbda.append(int(values[0]))
    spd.append(float(values[1]))

lambda_num = len(lanmbda)
lanmbda_np = np.zeros(shape=(lambda_num,1), dtype=np.float32)
spd_np     = np.zeros(shape=(lambda_num,1), dtype=np.float32)


for i in range(lambda_num):
    lanmbda_np[i, 0] = lanmbda[i]
    spd_np[i, 0]     = spd[i]


plt.figure(figsize=(10,10))
plt.plot(lanmbda_np, spd_np)
plt.title('spectral power distribution')



plt.title(title_name)
plt.savefig(title_name+'.png',dpi=100)
plt.show()
os.system('pause')


