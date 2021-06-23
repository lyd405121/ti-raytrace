
import sys
sys.path.append("integrator")
import taichi as ti
import math
import Example
import Camera as Camera
import Scene as Scene
import UtilsFunc as UF
import PT_RGB


class example(Example.example):
    def __init__(self, imgSizeX, imgSizeY, sample_count):
        ti.init(arch=ti.gpu)

        Example.example.__init__(self, imgSizeX, imgSizeY, sample_count)
        self.scene.add_obj('model/cornell_box.obj')
        self.integrator       = PT_RGB.PathTrace(self.imgSizeX, self.imgSizeY, self.cam, self.scene,64)      

    @ti.pyfunc
    def build_scene(self):
        Example.example.build_scene(self)
        self.scene.total_area()
        print("********total light area:%f****"%(self.scene.light_area.to_numpy()))
        centre = self.scene.maxboundarynp+self.scene.minboundarynp
        size   = self.scene.maxboundarynp-self.scene.minboundarynp
        self.cam.scale = math.sqrt(size[0,0]*size[0,0] + size[0,1]*size[0,1] + size[0,2]*size[0,2])*0.8
        self.cam.set_target(centre[0,0]*0.5, centre[0,1]*0.5, centre[0,2]*0.5)
        self.cam.update()


    @ti.pyfunc
    def render(self):
        return Example.example.render(self)
