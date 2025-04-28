import numpy as np
# from pathlib import Path


def get_domain_spherical(folder):
    """
    Read and save spherical coordinates from the FARGO domain files
    
    Inputs:
    ------
    folder:     Directory where the domain files are saved
    
    Outputs:
    -------
    domains:    Dictionary of theta, r, phi coordinates in cgs units
    """

    # Getting the spherical coordinate values from the FARGO3D domain files
    domains_min = {}
    real_to_fargo = {'phi': 'x', 'r':'y', 'theta':'z'}
    for key in ["r", "phi", "theta"]:
        domains_min[key] = np.loadtxt(folder / f"domain_{real_to_fargo[key]}.dat")
        if key != "phi": domains_min[key] = domains_min[key][3:-3] # Ghost cells

    # Since domain values are at cell interfaces, centering the domains
    domains = {}
    for key in domains_min.keys():
        domains[key] = (domains_min[key][1:] + domains_min[key][:-1]) / 2

    return domains


# Obtaining and reshaping physical quantity from scalar field files
def get_data(folder, quant, iter, domains):
    """
    Obtain physical quantity from FARGO3D's scalar fields e.g. dens
    
    Inputs:
    ------
    quant:    Scalar field keyword (str)
    iter:     Simulation snapshot (int)
    domains:  Centered domain data from domain*.dat files
    
    Outputs:
    -------
    data:     3D physical scalar field data
    """
    
    data = np.fromfile(folder / f"gas{quant}{iter}.dat").reshape(domains["theta"].size, domains["r"].size, domains["phi"].size)
    return data