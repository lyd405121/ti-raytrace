import sys
import os
import taichi as ti
import math
import UtilsFunc as UF
import taichi_glsl as ts

M_PIf = 3.1415956
@ti.data_oriented
class Disney:
    def __init__(self):
                self = self

    @ti.func
    def sqr(self, x):
         return x*x

    @ti.func
    def SchlickFresnel(self, u):
        m = ts.clamp(1.0-u, 0.0, 1.0)
        m2 = m*m
        return m2*m2*m

    @ti.func
    def GTR1(self, NDotH,  a):
        ret =1.0/ M_PIf
        if (a < 1.0):
            a2 = a*a
            t = 1.0 + (a2-1.0)*NDotH*NDotH
            ret = (a2-1.0) / (M_PIf*ti.log(a2)*t)
        return ret 

    @ti.func
    def GTR2(self, NDotH,  a):
        a2 = a*a
        t = 1.0 + (a2-1.0)*NDotH*NDotH
        return a2 / (M_PIf * t*t)

    @ti.func
    def smithG_GGX(self, NDotv,  alphaG):
        a = alphaG*alphaG
        b = NDotv*NDotv
        return 1.0/(NDotv + ti.sqrt(a + b - a*b))

    @ti.func
    def CosineSampleHemisphere(self,  u1,  u2):

        r = ti.sqrt(u1)
        phi = 2.0*M_PIf * u2

        p =   ti.Vector([0.0,0.0,0.0])
        p.x = r * ti.cos(phi)
        p.y = r * ti.sin(phi)

        p.z = ti.sqrt(max(0.0, 1.0 - p.x*p.x - p.y*p.y))
        return p

    @ti.func
    def inverse_transform(self, dir, N):
        Normal   = N.normalized()
        Binormal = ti.Vector([0.0, 0.0, 0.0])

        if (abs(Normal.x) > abs(Normal.z)):
            Binormal.x = -Normal.y
            Binormal.y = Normal.x
            Binormal.z = 0.0
        else:
            Binormal.x = 0.0
            Binormal.y = -Normal.z
            Binormal.z = Normal.y

        Binormal = Binormal.normalized()
        Tangent  = Binormal.cross(Normal)
        return dir.x*Tangent + dir.y*Binormal + dir.z*Normal

    @ti.func
    def sample(self, dir, N, i, j, mat_buf, mat_id):
        next_dir = ti.Vector([0.0, 0.0, 0.0])
        inout    = 1.0
        metal = UF.get_material_metallic(mat_buf, mat_id)
        rough = UF.get_material_roughness(mat_buf, mat_id)

        diffuseRatio   = 0.5 * (1.0 - metal)
        specularAlpha  = max(0.001, rough)


        probability = ti.random() 
        r1          = ti.random() 
        r2          = ti.random() 

        if (probability < diffuseRatio):
            next_dir = self.CosineSampleHemisphere(r1, r2)
            next_dir = self.inverse_transform(next_dir, N)
        else:
            phi = r1 * 2.0* M_PIf
            cosTheta = ti.sqrt((1.0 - r2) / (1.0 + (specularAlpha*specularAlpha - 1.0) *r2))
            sinTheta = ti.sqrt(1.0 - (cosTheta * cosTheta))
            sinPhi = ti.sin(phi)
            cosPhi = ti.cos(phi)

            half = ti.Vector([sinTheta*cosPhi, sinTheta*sinPhi, cosTheta])
            half = self.inverse_transform(half, N)
            next_dir  = ts.reflect(dir, half)

        return next_dir,inout

    #https://github.com/wdas/brdf/blob/main/src/brdfs/disney.brdf
    @ti.func
    def evaluate(self, N, V, L, mat_buf, mat_id):
        outputC = ti.Vector([0.0, 0.0, 0.0])
        pdf     = -1.0
        NDotL = N.dot(L)
        NDotV = N.dot(V)
        if ((NDotL > 0.0) & (NDotV > 0.0)):
            H = (L + V).normalized()
            NDotH = H.dot(N)
            LDotH = H.dot(L)
            VDotH = H.dot(V)
            Cdlin = UF.get_material_color(mat_buf, mat_id)
            metal = UF.get_material_metallic(mat_buf, mat_id)
            rough = UF.get_material_roughness(mat_buf, mat_id)
            Cdlum = 0.3*Cdlin.x + 0.6*Cdlin.y + 0.1*Cdlin.z
            

            

            # spec=0.5 spectint=0 sheentint=0.5  
            Ctint = ti.Vector([1.0, 1.0, 1.0])
            if Cdlum > 0.0:
                Ctint = Cdlin / Cdlum

            Cspec0 = ts.mix(ti.Vector([0.04, 0.04, 0.04]), Cdlin, metal)
            Csheen = ts.mix(ti.Vector([1.0, 1.0, 1.0]), Ctint, 1.0)


            FL            = self.SchlickFresnel(NDotL)
            FV            = self.SchlickFresnel(NDotV)
            Fd90          = 0.5 + 2.0 * LDotH*LDotH * rough
            Fd            = ts.mix(1.0, Fd90, FL) * ts.mix(1.0, Fd90, FV)

            Fss90         = LDotH * LDotH* rough
            Fss           = ts.mix(1.0, Fss90, FL) * ts.mix(1.0, Fss90, FV)
            ss            = 1.25* (Fss * (1.0 / (NDotL + NDotV) - 0.5) + 0.5)

        


            specularAlpha = max(0.001, rough)
            Ds            = self.GTR2(NDotH, specularAlpha)
            FH            = self.SchlickFresnel(LDotH)
            Fs            = ts.mix(Cspec0, ti.Vector([1.0, 1.0, 1.0]), FH)
            roughg        = self.sqr(rough*0.5+ 0.5)
            Gs            = self.smithG_GGX(NDotL, roughg) * self.smithG_GGX(NDotV, roughg)

            Fsheen = FH * Csheen

            Dr = self.GTR1(NDotH,  0.001)
            Fr = ts.mix(0.04, 1.0, FH)
            Gr = self.smithG_GGX(NDotL, 0.25) * self.smithG_GGX(NDotV, 0.25)
            outputC = (Fsheen + Cdlin * (1.0 / M_PIf)) * Fd* (1.0 - metal) + Gs * Fs * Ds 
            outputC *= NDotL

            diffuseRatio = 0.5 * (1.0 - metal)
            specularRatio = 1.0 - diffuseRatio
            pdfGTR2 = Ds * NDotH
            pdfGTR1 = Dr * NDotH

            pdfSpec = pdfGTR2 / (4.0 * abs(LDotH))
            pdfDiff = NDotL / M_PIf

            pdf = diffuseRatio * pdfDiff + specularRatio * pdfSpec
        
        return outputC , pdf
