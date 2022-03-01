
import sys
sys.path.append("integrator")
import taichi as ti
import math
import Example
import Camera as Camera
import Scene as Scene
import UtilsFunc as UF
import PT_Spec
import SceneData as SCD

class example(Example.example):
    def __init__(self, imgSizeX, imgSizeY, sample_count):
        ti.init(arch=ti.gpu)

        Example.example.__init__(self, imgSizeX, imgSizeY, sample_count)
        self.scene.add_obj('model/cornell_box.obj')
        self.integrator       = PT_Spec.PathTrace(self.imgSizeX, self.imgSizeY, self.cam, self.scene,64)      

        self.scene.material_cpu[0].type        = SCD.MAT_SPECTRAL
        self.scene.material_cpu[1].type        = SCD.MAT_SPECTRAL
        self.scene.material_cpu[2].type        = SCD.MAT_SPECTRAL
        self.scene.material_cpu[0].alebdoTex   = 0
        self.scene.material_cpu[1].alebdoTex   = 1
        self.scene.material_cpu[2].alebdoTex   = 2

    
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

    
    def render(self):
        return Example.example.render(self)
