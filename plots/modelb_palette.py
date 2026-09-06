"""Semantic color definitions for the Model-B manuscript figures."""
import numpy as np
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

# Reference / method colors
REFERENCE = '#262626'      # direct Model B / exact reference
DM = '#0072B2'             # learned diffusion model
ALT = '#D55E00'            # secondary learned method / solver
GUIDE = '#7A7A7A'          # theory guides, asymptotic laws
ZERO = '#B0B0B0'           # y=0 or ratio=1 guide

# Fixed mode colors. Keep these identical in Fig. 1(d), Fig. 2, and Fig. S2.
MODE_COLORS = ['#D55E00', '#0072B2', '#CC79A7', '#009E73', '#E69F00', '#56B4E9']
MODE_MARKERS = ['o', 's', 'D', '^', 'v', 'P']

# Ordered physical times. Avoid the very dark and very yellow ends.
def time_colors(n):
    return mpl.colormaps['viridis'](np.linspace(0.12, 0.82, n))

# Signed scalar field. Always use symmetric vmin/vmax about zero.
FIELD_CMAP = LinearSegmentedColormap.from_list(
    'modelb_field', ['#2F63AC', '#F4F4F2', '#B7433E'], N=256
)
