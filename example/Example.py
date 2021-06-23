
import sys
sys.path.append("integrator")
import taichi as ti
import math
import Camera as Camera
import Scene as Scene
import UtilsFunc as UF
import SceneData as SCD

class example:
    def __init__(self, imgSizeX, imgSizeY, sample_count):
        self.imgSizeX = imgSizeX
        self.imgSizeY = imgSizeY
        self.sample_count = sample_count
        self.gui          = ti.GUI('Render', res=(imgSizeX, imgSizeY))
        self.cam          = Camera.Camera(imgSizeX, imgSizeY, sample_count)
        self.scene        = Scene.Scene()

    def build_scene(self):
        self.scene.setup_data_cpu()
        self.integrator.setup_data_cpu()
        self.integrator.setup_data_gpu()
        self.scene.setup_data_gpu()


    def add_sphere_light(self):
        shape           = SCD.Shape()
        shape.type      = SCD.SHPAE_SPHERE
        shape.pos       = [0.0, 20.0, 0.0]
        shape.setRadius(5.0)

        mat      = SCD.Material()
        mat.type = SCD.MAT_LIGHT
        mat.setColor([50.0, 50.0, 50.0])
        self.scene.add_shape(shape, mat)

    def render(self):
        ret = 1
        if self.gui.running:
            if self.cam.frame_cpu < self.sample_count:
                self.integrator.render()
                UF.tone_map(0.5, self.integrator.hdr, self.integrator.rgb_film)
                self.gui.set_image(self.integrator.rgb_film.to_numpy())
                self.gui.show()
                self.cam.update_frame()

            elif self.cam.frame_cpu == self.sample_count:
                ti.imwrite(self.integrator.rgb_film, "out.png")
                self.cam.update_frame()
                self.gui.set_image(self.integrator.rgb_film.to_numpy())
                self.gui.show()
                ret = 0
            else:
                self.gui.set_image(self.integrator.rgb_film.to_numpy())
                self.gui.show()
        else:
            ret = 0
        return ret
