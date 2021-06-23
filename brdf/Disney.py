import sys
import os
import taichi as ti
import math
import UtilsFunc as UF
import taichi_glsl as ts





@ti.func
def diffuse_pdf(NDotL):
    #return  abs(NDotL)/ UF.M_PIf
    return  1.0/ UF.M_PIf

@ti.func
def sample(dir, N, mat_buf, mat_id):
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
        next_dir = UF.CosineSampleHemisphere(r1, r2)
        next_dir = UF.inverse_transform(next_dir, N)
    else:
        phi = r1 * 2.0* UF.M_PIf
        cosTheta = ti.sqrt((1.0 - r2) / (1.0 + (specularAlpha*specularAlpha - 1.0) *r2))
        sinTheta = ti.sqrt(1.0 - (cosTheta * cosTheta))
        sinPhi = ti.sin(phi)
        cosPhi = ti.cos(phi)
        half = ti.Vector([sinTheta*cosPhi, sinTheta*sinPhi, cosTheta])
        half = UF.inverse_transform(half, N)
        next_dir  = ts.reflect(dir, half)
    return next_dir,inout

    
@ti.func
def pdf(N, V, L, mat_buf, mat_id):
    pdf     = 0.0
    NDotL = N.dot(L)
    NDotV = N.dot(V)
    if ((NDotL > 0.0) & (NDotV > 0.0)):
        H = (L + V).normalized()
        NDotH = H.dot(N)
        LDotH = H.dot(L)
        metal = UF.get_material_metallic(mat_buf, mat_id)
        rough = UF.get_material_roughness(mat_buf, mat_id)
        specularAlpha = max(0.001, rough)
        Ds            = UF.GTR2(NDotH, specularAlpha)
        diffuseRatio  = 0.5 * (1.0 - metal)
        specularRatio = 1.0 - diffuseRatio
        pdfGTR2       = Ds * NDotH
        pdfSpec       = pdfGTR2 / (4.0 * abs(LDotH))
        pdfDiff       = diffuse_pdf(NDotL)
        pdf           = diffuseRatio * pdfDiff + specularRatio * pdfSpec
    
    return pdf

@ti.func
def evaluate_pdf(N, V, L, mat_buf, mat_id):
    outputC = 0.0
    pdf     = -1.0
    NDotL = N.dot(L)
    NDotV = N.dot(V)
    if ((NDotL > 0.0) & (NDotV > 0.0)):
        H = (L + V).normalized()
        NDotH = H.dot(N)
        LDotH = H.dot(L)
        #VDotH = H.dot(V)
        #Cdlin = UF.get_material_color(mat_buf, mat_id)
        metal = UF.get_material_metallic(mat_buf, mat_id)
        rough = UF.get_material_roughness(mat_buf, mat_id)
        #Cdlum = 0.3*Cdlin.x + 0.6*Cdlin.y + 0.1*Cdlin.z
        
        
        # spec=0.5 spectint=0 sheentint=0.5  
        #Ctint = ti.Vector([1.0, 1.0, 1.0])
        #if Cdlum > 0.0:
        #    Ctint = Cdlin / Cdlum
        Cspec0 = ts.mix(0.04, 1.0, metal)
        Csheen = 0.5
        FL            = UF.SchlickFresnel(NDotL)
        FV            = UF.SchlickFresnel(NDotV)
        Fd90          = 0.5 + 2.0 * LDotH*LDotH * rough
        Fd            = ts.mix(1.0, Fd90, FL) * ts.mix(1.0, Fd90, FV)
        specularAlpha = max(0.001, rough)
        Ds            = UF.GTR2(NDotH, specularAlpha)
        FH            = UF.SchlickFresnel(LDotH)
        Fs            = ts.mix(Cspec0, 1.0, FH)
        roughg        = UF.sqr(rough*0.5+ 0.5)
        Gs            = UF.smithG_GGX(NDotL, roughg) * UF.smithG_GGX(NDotV, roughg)
        Fsheen        = FH * Csheen
        outputC       = (Fsheen + 1.0 / UF.M_PIf) * Fd* (1.0 - metal) + Gs * Fs * Ds 

        diffuseRatio  = 0.5 * (1.0 - metal)
        specularRatio = 1.0 - diffuseRatio
        pdfGTR2       = Ds * NDotH
        pdfSpec       = pdfGTR2 / (4.0 * abs(LDotH))
        pdfDiff       = diffuse_pdf(NDotL)
        pdf           = diffuseRatio * pdfDiff + specularRatio * pdfSpec
    
    return outputC , pdf

#https://github.com/wdas/brdf/blob/main/src/brdfs/disney.brdf
@ti.func
def evaluate(N, V, L, mat_buf, mat_id):
    outputC = 0.0
    NDotL = N.dot(L)
    NDotV = N.dot(V)
    if ((NDotL > 0.0) & (NDotV > 0.0)):
        H = (L + V).normalized()
        NDotH = H.dot(N)
        LDotH = H.dot(L)
        #VDotH = H.dot(V)
        #Cdlin = UF.get_material_color(mat_buf, mat_id)
        metal = UF.get_material_metallic(mat_buf, mat_id)
        rough = UF.get_material_roughness(mat_buf, mat_id)
        #Cdlum = 0.3*Cdlin.x + 0.6*Cdlin.y + 0.1*Cdlin.z
        # spec=0.5 spectint=0 sheentint=0.5  
        #Ctint = ti.Vector([1.0, 1.0, 1.0])
        #if Cdlum > 0.0:
        #    Ctint = Cdlin / Cdlum
        Cspec0 = ts.mix(0.04, 1.0, metal)
        Csheen = 0.5
        FL            = UF.SchlickFresnel(NDotL)
        FV            = UF.SchlickFresnel(NDotV)
        Fd90          = 0.5 + 2.0 * LDotH*LDotH * rough
        Fd            = ts.mix(1.0, Fd90, FL) * ts.mix(1.0, Fd90, FV)
        specularAlpha = max(0.001, rough)
        Ds            = UF.GTR2(NDotH, specularAlpha)
        FH            = UF.SchlickFresnel(LDotH)
        Fs            = ts.mix(Cspec0, 1.0, FH)
        roughg        = UF.sqr(rough*0.5+ 0.5)
        Gs            = UF.smithG_GGX(NDotL, roughg) * UF.smithG_GGX(NDotV, roughg)
        Fsheen        = FH * Csheen
        outputC       = (Fsheen + 1.0 / UF.M_PIf) * Fd* (1.0 - metal) + Gs * Fs * Ds 
    return outputC 
