import sys
import taichi as ti


sys.path.append("example")

import cornell_box as example
#import single_model as example
#import sky_dome as example
#import spectral_box as example
#import veach_bdpt as example
#import prism_rainbow as example

ret = 1
ex = example.example(512, 512, 512)
ex.build_scene()
while ret == 1:
    ret = ex.render()





