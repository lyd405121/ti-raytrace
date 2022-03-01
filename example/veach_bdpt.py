import sys
sys.path.append("integrator")
import Example
import taichi as ti
import math
import Camera as Camera
import Scene as Scene
import UtilsFunc as UF
import BDPT_RGB


class example(Example.example):
    def __init__(self, imgSizeX, imgSizeY, sample_count):
        ti.init(arch=ti.gpu)
        Example.example.__init__(self, imgSizeX, imgSizeY, sample_count)
        self.scene.add_obj('model/bdpt.obj')
        self.integrator    = BDPT_RGB.BDPT(self.imgSizeX, self.imgSizeY, self.cam, self.scene,64)      

    
    def build_scene(self):
        Example.example.build_scene(self)
        self.scene.process_normal()
        self.scene.total_area()
        
        print("********total light area:%f****"%(self.scene.light_area.to_numpy()))
        centre = self.scene.maxboundarynp+self.scene.minboundarynp
        size   = self.scene.maxboundarynp-self.scene.minboundarynp
        self.cam.scale = math.sqrt(size[0,0]*size[0,0] + size[0,1]*size[0,1] + size[0,2]*size[0,2])*0.5
        self.cam.set_target(centre[0,0]*0.5, centre[0,1]*0.5, centre[0,2]*0.5)
        self.cam.update()
        print("!!!!this example takes a long time to compile!!!!!!!!!!!")
    
    def render(self):
        return Example.example.render(self)

