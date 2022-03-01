
import taichi as ti
@ti.data_oriented
class Vertex:
    def __init__(self, imgSizeX, imgSizeY, depth):
        self.imgSizeX  = imgSizeX
        self.imgSizeY  = imgSizeY
        self.depth     = depth

        self.pos       = ti.Vector.field(3, dtype=ti.f32)
        self.normal    = ti.Vector.field(3, dtype=ti.f32)
        self.snormal   = ti.Vector.field(3, dtype=ti.f32)
        self.beta      = ti.Vector.field(3, dtype=ti.f32)
        self.wo        = ti.Vector.field(3, dtype=ti.f32)
        self.fpdf      = ti.field( dtype=ti.f32)
        self.rpdf      = ti.field( dtype=ti.f32)
        self.type      = ti.field( dtype=ti.i32)
        self.prim      = ti.field( dtype=ti.i32)
        self.mat       = ti.field( dtype=ti.i32)
        self.delta     = ti.field( dtype=ti.i32)
        self.power     = ti.field( dtype=ti.f32)
    
    def setup_data_cpu(self):
        
        #ti.root.dense(ti.ijk, [self.imgSizeX, self.imgSizeY, self.depth] ).place(self.pos,self.normal,self.snormal,self.beta, self.wo, self.fpdf,self.rpdf,self.type ,self.prim,self.mat   )

    
        ti.root.dense(ti.ijk, [self.imgSizeX, self.imgSizeY, self.depth] ).place(self.pos       )
        ti.root.dense(ti.ijk, [self.imgSizeX, self.imgSizeY, self.depth] ).place(self.normal    )
        ti.root.dense(ti.ijk, [self.imgSizeX, self.imgSizeY, self.depth] ).place(self.snormal   )
        ti.root.dense(ti.ijk, [self.imgSizeX, self.imgSizeY, self.depth] ).place(self.beta      )
        ti.root.dense(ti.ijk, [self.imgSizeX, self.imgSizeY, self.depth] ).place(self.wo        )
        ti.root.dense(ti.ijk, [self.imgSizeX, self.imgSizeY, self.depth] ).place(self.fpdf      )
        ti.root.dense(ti.ijk, [self.imgSizeX, self.imgSizeY, self.depth] ).place(self.rpdf      )
        ti.root.dense(ti.ijk, [self.imgSizeX, self.imgSizeY, self.depth] ).place(self.type      )
        ti.root.dense(ti.ijk, [self.imgSizeX, self.imgSizeY, self.depth] ).place(self.prim      )
        ti.root.dense(ti.ijk, [self.imgSizeX, self.imgSizeY, self.depth] ).place(self.mat       )
        ti.root.dense(ti.ijk, [self.imgSizeX, self.imgSizeY, self.depth] ).place(self.delta     )
        ti.root.dense(ti.ijk, [self.imgSizeX, self.imgSizeY, self.depth] ).place(self.power     )
    
    def setup_data_gpu(self):
        # do nothing
        self.depth     = self.depth

    @ti.func
    def copy(self, depth, temp:ti.template(), i, j, k):
        self.pos[i,j,depth]     = temp.pos[i,j, k]    
        self.normal[i,j,depth]  = temp.normal[i,j, k] 
        self.snormal[i,j,depth] = temp.snormal[i,j, k]
        self.beta[i,j,depth]    = temp.beta[i,j, k]   
        self.wo[i,j,depth]      = temp.wo[i,j, k]     
        self.fpdf[i,j,depth]    = temp.fpdf[i,j, k]   
        self.rpdf[i,j,depth]    = temp.rpdf[i,j, k]   
        self.type[i,j,depth]    = temp.type[i,j, k]   
        self.prim[i,j,depth]    = temp.prim[i,j, k]   
        self.mat[i,j,depth]     = temp.mat[i,j, k ]
        self.delta[i,j,depth]   = temp.delta[i,j, k]
        self.power[i,j,depth]   = temp.power[i,j, k]