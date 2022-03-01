import sys
sys.path.append("integrator")
import Example
import taichi as ti
import math
import Camera as Camera
import Scene as Scene
import UtilsFunc as UF
import BDPT_SPEC
import PT_RGB
import SceneData as SCD

class example(Example.example):
    def __init__(self, imgSizeX, imgSizeY, sample_count):
        ti.init(arch=ti.cpu)
        Example.example.__init__(self, imgSizeX, imgSizeY, sample_count)
        self.scene.add_obj('model/prism1.obj')
        self.add_sphere_light()
        self.add_laser_light()
        
        self.integrator    = BDPT_SPEC.BDPT(self.imgSizeX, self.imgSizeY, self.cam, self.scene, 1024)      
        #self.integrator    = PT_RGB.PathTrace(self.imgSizeX, self.imgSizeY, self.cam, self.scene, 1024)  



    def add_sphere_light(self):
        shape           = SCD.Shape()
        shape.type      = SCD.SHPAE_SPHERE
        shape.pos       = [0.0, 20.0, 0.0]
        shape.setRadius(5.0)

        mat      = SCD.Material()
        mat.type = SCD.MAT_LIGHT
        mat.setColor([500.0, 500.0, 500.0])
        self.scene.add_shape(shape, mat)

    def add_laser_light(self):
        shape           = SCD.Shape()
        shape.type      = SCD.SHPAE_LASER


        shape.pos       = [1.0, 0.0, 9.0]
        shape.setRadius(0.1)
        shape.setNormal([0.0, 0.0, -1.0])

        mat      = SCD.Material()
        mat.type = SCD.MAT_LIGHT
        mat.setColor([500.0, 500.0, 500.0])
        self.scene.add_shape(shape, mat)

    
    def build_scene(self):

        Example.example.build_scene(self)

        self.scene.total_area()
        

        print("********total light area:%f****"%(self.scene.light_area.to_numpy()))

        #cam.yaw   = 0.8
        #cam.scale = 20.0
        #cam.set_target(-50, 2.0, -93.0)

        self.cam.scale = 10.0
        self.cam.set_target(0.0, 0.0, 0.0)
        self.cam.update()
        print("!!!!this example takes a long time to compile!!!!!!!!!!!")
    
    def render(self):
        return Example.example.render(self)

