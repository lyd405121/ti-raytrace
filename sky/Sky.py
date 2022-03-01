import sys
import os
import time
import math
import taichi as ti
import numpy as np
import taichi_glsl as ts

MATH_PI    = 3.141592653589793
LAMDDA_DIV = 11
ALBEDO_NUM = 2
TURB_NUM   = 10
THETA_NUM  = 9
GAMMA_NUM  = 6
PIECES     = 45
ORDER      = 4
DATA_NUM   = TURB_NUM*ALBEDO_NUM*THETA_NUM*GAMMA_NUM
RAD_NUM    = TURB_NUM*ALBEDO_NUM*6
SOLAR_NUM  = TURB_NUM*PIECES*ORDER
DARK_NUM   = 6

MIN_LAMBDA = 320.0
MAX_LAMBDA = 720.0


@ti.data_oriented
class Sky:
    def __init__(self, turbidity=3.0, albedo=0.5, elevation=10.0*MATH_PI / 180.0):
        
        self.configs         = ti.field(dtype=ti.f32, shape=(LAMDDA_DIV, THETA_NUM))
        self.radiances       = ti.field(dtype=ti.f32, shape=(LAMDDA_DIV))
        self.data            = ti.field(dtype=ti.f32, shape=(LAMDDA_DIV, DATA_NUM))
        self.data_rad        = ti.field(dtype=ti.f32, shape=(LAMDDA_DIV, RAD_NUM))
        self.data_solar      = ti.field(dtype=ti.f32, shape=(LAMDDA_DIV, SOLAR_NUM))
        self.data_dark       = ti.field(dtype=ti.f32, shape=(LAMDDA_DIV, DARK_NUM))
        self.sun_dir         = ti.Vector.field(3, dtype=ti.f32, shape=(1))

        self.configs_np      = np.zeros(shape=(LAMDDA_DIV, THETA_NUM), dtype=np.float32)
        self.radiances_np    = np.zeros(shape=(LAMDDA_DIV), dtype=np.float32)
        self.data_np         = np.zeros(shape=(LAMDDA_DIV, DATA_NUM), dtype=np.float32)
        self.data_rad_np     = np.zeros(shape=(LAMDDA_DIV, RAD_NUM), dtype=np.float32)
        self.data_solar_np   = np.zeros(shape=(LAMDDA_DIV, SOLAR_NUM), dtype=np.float32)
        self.data_dark_np    = np.zeros(shape=(LAMDDA_DIV, DARK_NUM), dtype=np.float32)       
        self.sun_dir_np      = np.zeros(shape=(1, 3), dtype=np.float32)
     
        self.turbidity       = turbidity
        self.solar_radius    = 0.51 * MATH_PI / 180.0 / 2.0
        self.albedo          = albedo
        self.elevation       = elevation  

        i = 0
        for line in open("sky\data.csv", "r"):
            values = line.split(',', DATA_NUM)
            for j in range(DATA_NUM):
                self.data_np[i,j] = values[j]
            i += 1


        i = 0
        for line in open("sky\data_rad.csv", "r"):
            values = line.split(',', RAD_NUM)
            for j in range(RAD_NUM):
                self.data_rad_np[i,j] = values[j]
            i += 1

        i = 0
        for line in open("sky\data_solar.csv", "r"):
            values = line.split(',', SOLAR_NUM)
            for j in range(SOLAR_NUM):
                self.data_solar_np[i,j] = values[j]
            i += 1

        i = 0       
        for line in open("sky\data_dark.csv", "r"):
            values = line.split(',', DARK_NUM)
            for j in range(DARK_NUM):
                self.data_dark_np[i,j] = values[j]
            i += 1       

        #ti.root.dense(ti.i, (self.size) ).place(self.data)

    
    def setup_data_gpu(self):
        self.update()

        self.configs.from_numpy(self.configs_np) 
        self.radiances.from_numpy(self.radiances_np)

        self.data.from_numpy(self.data_np) 
        self.data_rad.from_numpy(self.data_rad_np)          
        self.data_solar.from_numpy(self.data_solar_np)        
        self.data_dark.from_numpy(self.data_dark_np)  

        self.sun_dir_np[0,0] = 0.0
        self.sun_dir_np[0,1] = math.sin(self.elevation)
        self.sun_dir_np[0,2] = math.cos(self.elevation)
        self.sun_dir.from_numpy(self.sun_dir_np )
        #print(self.sun_dir_np)

    
    def formula(self,t, A0,A1,A2,A3,A4,A5):
        return pow(1.0-t, 5.0) * A0  + 5.0  * pow(1.0-t, 4.0) * t * A1 +\
            10.0*pow(1.0-t, 3.0)*pow(t, 2.0) * A2 +10.0*pow(1.0-t, 2.0)*pow(t, 3.0) * A3 +\
            5.0*(1.0-t)*pow(t, 4.0) * A4 + pow(t, 5.0)  * A5

    
    def update(self):
        albedo = self.albedo
        int_turbidity   = int(self.turbidity)
        turbidity_rem   = self.turbidity - float(int_turbidity)
        solar_elevation = pow(self.elevation / (MATH_PI / 2.0), (1.0 / 3.0))

        #configs
        index = 9 * 6 * (int_turbidity-1)
        for j in range(LAMDDA_DIV):
            for i in range(THETA_NUM):
                self.configs_np[j, i] =  (1.0-albedo) * (1.0 - turbidity_rem) \
                * self.formula(solar_elevation,self.data_np[j, index + i],self.data_np[j, index + i+9],self.data_np[j, index + i+18],self.data_np[j, index + i+27],self.data_np[j, index + i+36],self.data_np[j, index + i+45])

        index = 9*6*10 + 9*6*(int_turbidity-1)
        for j in range(LAMDDA_DIV):
            for i in range(THETA_NUM):
                self.configs_np[j, i] +=  (albedo) * (1.0 - turbidity_rem)\
                * self.formula(solar_elevation,self.data_np[j, index + i],self.data_np[j, index + i+9],self.data_np[j, index + i+18],self.data_np[j, index + i+27],self.data_np[j, index + i+36],self.data_np[j, index + i+45])
        
        if(int_turbidity < 10):
            index = 9*6*int_turbidity
            for j in range(LAMDDA_DIV):
                for i in range(THETA_NUM):
                    self.configs_np[j, i] += (1.0-albedo) * (turbidity_rem)\
                    * self.formula(solar_elevation,self.data_np[j, index + i],self.data_np[j, index + i+9],self.data_np[j, index + i+18],self.data_np[j, index + i+27],self.data_np[j, index + i+36],self.data_np[j, index + i+45])

            index = 9*6*10 + 9*6*(int_turbidity)       
            for j in range(LAMDDA_DIV):
                for i in range(THETA_NUM):
                    self.configs_np[j, i] +=         (albedo) * (turbidity_rem)\
                    * self.formula(solar_elevation,self.data_np[j, index + i],self.data_np[j, index + i+9],self.data_np[j, index + i+18],self.data_np[j, index + i+27],self.data_np[j, index + i+36],self.data_np[j, index + i+45])


        #radiance
        index = 6 * (int_turbidity-1)
        for i in range(LAMDDA_DIV):
            self.radiances_np[i] =  (1.0-albedo) * (1.0 - turbidity_rem) \
            * self.formula(solar_elevation,self.data_rad_np[i, index + 0],self.data_rad_np[i, index + 1],self.data_rad_np[i, index + 2],self.data_rad_np[i, index + 3],self.data_rad_np[i, index + 4],self.data_rad_np[i, index + 5])

        index =6*10 + 6*(int_turbidity-1)
        for i in range(LAMDDA_DIV):
            self.radiances_np[i] +=  (albedo) * (1.0 - turbidity_rem) \
            * self.formula(solar_elevation,self.data_rad_np[i, index + 0],self.data_rad_np[i, index + 1],self.data_rad_np[i, index + 2],self.data_rad_np[i, index + 3],self.data_rad_np[i, index + 4],self.data_rad_np[i, index + 5])

        if(int_turbidity < 10):
            index = 6*int_turbidity
            for i in range(LAMDDA_DIV):
                self.radiances_np[i] +=  (1.0-albedo) * (turbidity_rem) \
                * self.formula(solar_elevation,self.data_rad_np[i, index + 0],self.data_rad_np[i, index + 1],self.data_rad_np[i, index + 2],self.data_rad_np[i, index + 3],self.data_rad_np[i, index + 4],self.data_rad_np[i, index + 5])
            index = 6*10 + 6*(int_turbidity)
            for i in range(LAMDDA_DIV):
                self.radiances_np[i] +=  (albedo) * (turbidity_rem) \
                * self.formula(solar_elevation,self.data_rad_np[i, index + 0],self.data_rad_np[i, index + 1],self.data_rad_np[i, index + 2],self.data_rad_np[i, index + 3],self.data_rad_np[i, index + 4],self.data_rad_np[i, index + 5])
        

        #print(self.configs_np)
        #print(self.radiances_np)

    @ti.func
    def sr_internal(self, turbidity, wl,elevation):
        pos =(int) (pow(2.0*elevation / MATH_PI, 1.0/3.0) * PIECES)

        if ( pos > 44 ):
             pos = 44

        break_x =pow(( float(pos) /  float(PIECES)), 3.0) * (MATH_PI * 0.5)
        index = ORDER*PIECES *turbidity + ORDER *(pos+1) -1
        ret = 0.0
        x = elevation - break_x
        x_exp = 1.0


        

        i = 0
        while i < ORDER:
            ret += x_exp * self.data_solar[wl, index]
            x_exp *= x
            i += 1
            index -= 1

        return ret 


    @ti.func
    def solar_radiance_internal(self, wl, theta, gamma):
        expM = ti.exp(self.configs[wl,4] * gamma)
        rayM = ti.cos(gamma)*ti.cos(gamma)
        mieM = (1.0 + ti.cos(gamma)*ti.cos(gamma)) / pow((1.0 + self.configs[wl,8]*self.configs[wl,8] - 2.0*self.configs[wl,8]*ti.cos(gamma)), 1.5)
        zenith = ti.sqrt(ti.cos(theta))

        return (1.0 + self.configs[wl,0] * ti.exp(self.configs[wl,1] / (ti.cos(theta) + 0.01))) *\
            (self.configs[wl,2] + self.configs[wl,3] * expM + self.configs[wl,5] * rayM + self.configs[wl,6] * mieM + self.configs[wl,7] * zenith)

    @ti.func
    def solar_radiance_internal2(self, wavelength, elevation, gamma):
        sol_rad_sin = ti.sin(self.solar_radius)
        ar2 = 1.0 / ( sol_rad_sin * sol_rad_sin )
        singamma = ti.sin(gamma)
        sc2 = 1.0 - ar2 * singamma * singamma
        
        direct_radiance = 0.0
        if (sc2 < 0.0 ):
            sc2 = 0.0
            direct_radiance = 0.0
        else:
            sampleCosine = ti.sqrt (sc2)
    
            turb_low  = int(self.turbidity) - 1
            turb_frac = self.turbidity - float(turb_low + 1)
    
            if ( turb_low == 9 ):
                turb_low  = 8
                turb_frac = 1.0
    
            wl_low  = int((wavelength - 320.0) / 40.0)
            wl_frac = ts.fract(wavelength/ 40.0) 
    
            if ( wl_low == 10 ):
                wl_low = 9
                wl_frac = 1.0
    
            direct_radiance = ts.mix(ts.mix(self.sr_internal(turb_low, wl_low, elevation), self.sr_internal(turb_low, wl_low+1, elevation),wl_frac), \
                            ts.mix(self.sr_internal(turb_low+1, wl_low, elevation), self.sr_internal(turb_low+1, wl_low+1, elevation),wl_frac), turb_frac)
    
            
            ldCoefficient = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            for i in ti.static(range(6)):
                ldCoefficient[i] = ts.mix(self.data_dark[wl_low, i],self.data_dark[wl_low+1, i], wl_frac)
    
            
            darkeningFactor   = ldCoefficient[0] + ldCoefficient[1] * sampleCosine + ldCoefficient[2] * pow( sampleCosine, 2.0 )+ ldCoefficient[3] * pow( sampleCosine, 3.0 )+ ldCoefficient[4] * pow( sampleCosine, 4.0 )+ ldCoefficient[5] * pow( sampleCosine, 5.0 )
            direct_radiance  *= darkeningFactor
        return direct_radiance

    @ti.func
    def solar_radiance(self, theta, gamma, wavelength):
        low_wl = int((wavelength - 320.0 ) / 40.0)
        result = 0.0
        if ( low_wl >= 0) & ( low_wl < 11 ):
            interp  = ts.fract((wavelength - 320.0 ) / 40.0)
            val_low = self.solar_radiance_internal(low_wl,theta,gamma) *self.radiances[low_wl]

            if ( interp < 1e-6 ):
                result = val_low
            else:
                result = ( 1.0 - interp ) * val_low
                if ( low_wl+1 < 11 ):
                    result +=interp * self.solar_radiance_internal(low_wl+1,theta,gamma) * self.radiances[low_wl+1]
        return result

    @ti.func
    def get_solar_radiance(self, theta, gamma, wavelength):
        ret = 0.0
        if (wavelength >= MIN_LAMBDA ) & ((wavelength <= MAX_LAMBDA )):
            #direct_sun_light   = self.solar_radiance_internal2(wavelength, MATH_PI/2.0-theta, gamma)
            indirect_sun_light = self.solar_radiance(theta, gamma, wavelength)
            ret = indirect_sun_light
        return ret