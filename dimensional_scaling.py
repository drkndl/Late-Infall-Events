# Functions to allow scaling across Mstar

import numpy as np
from pathlib import Path
from read import get_domain_spherical, get_data, load_par_file, get_param_value
import matplotlib.pyplot as plt
from no_thoughts_just_plots import quiver_plot_3d, contours_3D, plot_surf_dens, plot_twist_arrows, plot_total_disks_bonanza, cyl_2D_plot, XY_2D_plot
import astropy.constants as c
import pandas as pd
au = c.au.cgs.value
G = 6.67e-8               # Gravitational constant in cgs units
Msun = 1.989e33           # Mass of the Sun in g
Mstar = 0.7 * Msun        # Mass of the primary star in IRAS 04125+2902 (Barber et al. 2024)
dt = 1.87e7               # Timestep length of simulations in sec
ninterm = 200             # Total number of timesteps between outputs in FARGO simulations
stoky = 3.156e7 * 1e3     # 1 kyr in sec


def rescale_t(t, f):
    """
    Function to rescale time based on scaling parameter f of stellar mass
    """

    tprime = np.sqrt(f) * t 
    return tprime


def rescale_v(v, f):
    """
    Function to rescale velocity based on scaling parameter f of stellar mass
    """

    vprime = np.sqrt(f) * v
    return vprime 


def rescale_Mdot(Mdot, f):
    """
    Function to rescale accretion rate based on scaling parameter f of stellar mass
    """

    Mdotprime = np.sqrt(f) * Mdot 
    return Mdotprime 

