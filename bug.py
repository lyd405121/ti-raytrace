import taichi as ti
ti.init(arch=ti.gpu)

@ti.data_oriented
class Tool:
    @ti.func
    def unroll(self, v):
        for i in ti.static(range(3)):
            v[i]= 0


@ti.data_oriented
class Test:
    def __init__(self, tool):
        self.v  = ti.Vector.field(3, dtype=ti.f32)
        ti.root.dense(ti.i, 100 ).place(self.v) 
        self.tool = tool

    @ti.kernel
    def test(self):
        tool = ti.static(self.tool)
        for  i in self.v:
            tool.unroll(self.v[i])

tool = Tool()
test1 = Test(tool)
test1.test()





    