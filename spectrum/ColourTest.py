from colour.plotting import *
import numpy as np
import colour

RGB = np.random.random((1, 1, 3))
plot_RGB_chromaticities_in_chromaticity_diagram_CIE1931(RGB, 'ITU-R BT.709',colourspaces=['ACEScg', 'S-Gamut', 'Pointer Gamut'])

#plot_visible_spectrum('CIE 1931 2 Degree Standard Observer')
#plot_single_illuminant_sd('D65')