import taichi as ti
import math
import UtilsFunc as UF

SAMPLE_WAVELENGTHS = 4
LAMBDA_MIN        = 360.0
LAMBDA_MAX        = 760.0
LAMBDA_STEP       = (LAMBDA_MAX - LAMBDA_MIN)/SAMPLE_WAVELENGTHS

@ti.func
def sample(spec, Lambda0):
    ret = ti.Vector([0.0,0.0,0.0,0.0])
    for i in ti.static(range(SAMPLE_WAVELENGTHS)):
        Lambda = Lambda0+i*LAMBDA_STEP
        ret[i] = spec.sample(Lambda)
    return ret

@ti.func
def sample_xyz(sensor, Lambda0):
    x_rad = ti.Vector([0.0,0.0,0.0,0.0])
    y_rad = ti.Vector([0.0,0.0,0.0,0.0])
    z_rad = ti.Vector([0.0,0.0,0.0,0.0])
    for i in ti.static(range(SAMPLE_WAVELENGTHS)):
        Lambda   = Lambda0+i*LAMBDA_STEP
        xyz      = sensor.sample(Lambda)
        x_rad[i] = xyz[0]
        y_rad[i] = xyz[1]
        z_rad[i] = xyz[2]
    return x_rad,y_rad,z_rad


@ti.func
def get_rnd_hero(Lambda0):
    index = int(ti.random() * SAMPLE_WAVELENGTHS)
    return index, Lambda0 + index*LAMBDA_STEP

@ti.func
def get_extinction_hero(Lambda0, t):
    ret = ti.Vector([0.0,0.0,0.0,0.0])
    for i in ti.static(range(SAMPLE_WAVELENGTHS)):
        Lambda = Lambda0+i*LAMBDA_STEP
        ret[i] = ti.exp(-t / Lambda)
    return ret


@ti.func
def srgb_to_spec(rgb2spec, srgb, Lambda0):
    #Lambda0 = 423.856689
    #srgb = ti.Vector([0.00392156886, 0.0627451017, 0.129411772])
    #lrgb = ti.Vector([0.000303526991, 0.00913405698, 0.0212190095])

    lrgb = UF.srgb_to_lrgb(srgb)
    coff = rgb2spec.fetch(lrgb)
    ret = ti.Vector([0.0, 0.0, 0.0, 0.0])
    for i in ti.static(range(SAMPLE_WAVELENGTHS)):
        ret[i] = rgb2spec.eval(coff, Lambda0+i*LAMBDA_STEP)
    return ret

@ti.func
def sky_sample(sky, theta, gamma,  Lambda0):
    ret = ti.Vector([0.0,0.0,0.0,0.0])

    #theta = 1.000000
    #gamma = 0.781103
    #Lambda0 = 397.063538

    for i in ti.static(range(SAMPLE_WAVELENGTHS)):
        ret[i] = sky.get_solar_radiance(theta,  gamma ,Lambda0+i*LAMBDA_STEP)

    return ret


@ti.func
def spec_to_ciexyz(x_bar,y_bar,z_bar, spec, Lambda0):
    x_flux       = sample(x_bar, Lambda0) * spec
    y_flux       = sample(y_bar, Lambda0) * spec
    z_flux       = sample(z_bar, Lambda0) * spec

    #kind of monte carlo
    x_integrate  = x_flux * (x_bar.lambda_max - x_bar.lambda_min) / SAMPLE_WAVELENGTHS
    y_integrate  = y_flux * (y_bar.lambda_max - y_bar.lambda_min) / SAMPLE_WAVELENGTHS
    z_integrate  = z_flux * (z_bar.lambda_max - z_bar.lambda_min) / SAMPLE_WAVELENGTHS

    return ti.Vector([x_integrate.sum(), y_integrate.sum(), z_integrate.sum()])
