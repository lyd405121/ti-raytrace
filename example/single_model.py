
import sys
sys.path.append("integrator")
import taichi as ti
import math
import Example
import Camera as Camera
import Scene as Scene
import UtilsFunc as UF
import PT_RGB
import SceneData as SCD
import Texture as TX

class example(Example.example):
    def __init__(self, imgSizeX, imgSizeY, sample_count):
        ti.init(arch=ti.gpu)

        Example.example.__init__(self, imgSizeX, imgSizeY, sample_count)

        #self.scene.add_obj('model/mc.obj')
        self.scene.add_obj('model/sphere.obj')
        #self.scene.add_obj('model/box.obj')
        #self.scene.add_obj('model/cylinder.obj')
        #scene.add_obj('model/Teapot.obj')


        self.scene.material_cpu[0].type = SCD.MAT_GLASS
        self.scene.material_cpu[0].setIor(1.3)
        self.scene.material_cpu[0].setExtinciton(5.0)
        #self.scene.material_cpu[0].setMetal(1.0)
        #self.scene.material_cpu[0].setRough(0.0)

        Example.example.add_sphere_light(self)
        self.scene.add_env("image/env.png", 5.0)
        self.integrator       = PT_RGB.PathTrace(self.imgSizeX, self.imgSizeY, self.cam, self.scene,64)      


    @ti.pyfunc
    def build_scene(self):
        Example.example.build_scene(self)

        self.scene.process_normal()
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
