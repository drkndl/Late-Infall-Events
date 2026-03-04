# Calculate orbit parameters

import numpy as np
from pathlib import Path
from read import get_domain_spherical, get_data, load_par_file, get_param_value
import matplotlib.pyplot as plt
import astropy.constants as c

au = c.au.cgs.value
G = 6.67e-8               # Gravitational constant in cgs units
Msun = 1.989e33           # Mass of the Sun in g
Mstar = 0.7 * Msun        # Mass of the primary star in IRAS 04125+2902 (Barber et al. 2024)
dt = 1.87e7               # Timestep length of simulations in sec
ninterm = 200             # Total number of timesteps between outputs in FARGO simulations
stoky = 3.156e7 * 1e3     # 1 kyr in sec


def b_crit(v_inf):
    """
    Calculates the critical impact parameter below which accretion of cloud can occur


    Inputs:
    ------
    vinf:       Velocity of cloud at infinite distance [cm/s]

    Outputs:
    -------
    bcrit:      Critical impact parameter [cm]
    """

    bcrit = G * Mstar / v_inf**2
    return bcrit


def eccentricity(b):
    """
    Calculates eccentricity of the hyperbolic trajectory of the cloud

    Inputs:
    ------
    b:         Fraction of critical impact parameter used in simulation (i.e. b/b_crit) [float]

    Outputs:
    -------
    e:         Orbit eccentricity
    """

    e = np.sqrt(1 + b**2)
    return e 


def periapsis(bcrit, b):
    """
    Calculates the periapsis, or the distance of closest approach of the cloud

    Inputs:
    ------
    b:         Fraction of critical impact parameter used in simulation (i.e. b/b_crit) [float]
    bcrit:     Critical impact parameter [float]

    Outputs:
    -------
    rp:     Periapsis distance, or the distance of closest approach [cm, float]
    """

    rp = bcrit * (np.sqrt(1 + b**2) - 1)
    return rp 


def main():

    folder = Path("../cloud_disk_it450_rotX45/")                        # Folder with the output files
    # folder = Path("../fargo3d/outputs/cloud_disk_it450_rotX45")         # Folder with the output files (BinAC2)
    fig_imgs = Path("cloud_disk_it450_rotX45/imgs/")                    # Folder to save images
    sim_name = str(fig_imgs).split('/')[0]                         # Simulation name (for plot labels)
    
    # Load some simulation parameters
    b = get_param_value('ImpactParameter', sim_name) 
    vinf = get_param_value('VInf', sim_name)
    Rcloud = get_param_value('CloudletRadius', sim_name)

    e = eccentricity(b)
    bcrit = b_crit(vinf) 
    rp = periapsis(bcrit, b)

    print(f"Distance of closest approach: {(rp/au):.2f} AU")
    print(f"Cloudlet radius: {(Rcloud/au):.2f}")
    print(f"Eccentricity: {e:.2f}")


if __name__=="__main__":
    main()