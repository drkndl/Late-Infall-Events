# Checking the time evolution of energies to determine boundedness of cloudlet orbit

import numpy as np 
import matplotlib.pyplot as plt 
from pathlib import Path
from read import get_data, get_domain_spherical, load_par_file
from analysis import calc_simtime, sph_to_cart, vel_sph_to_cart


def track_cloudlet(Rout):
    """
    Tracks the position of the cloudlet throughout the simulation
    """

    return d 


def calculate_energy(v, M, r):
    """
    Calculates the total energy at different points in the simulation to check if the cloudlet orbit is hyperbolic or elliptical
    """

    return eps


def main():

    folder = Path("../cloud_nodisk_it450_rotXY90/")                        # Folder with the output files
    # folder = Path("../fargo3d/outputs/cloud_nodisk_it450_rotXY45")       # Folder with the output files (BinAC2)
    fig_imgs = Path("cloud_nodisk_it450_rotXY90/imgs/")                    # Folder to save images
    it = 450                                                       # FARGO snapshot of interest
    sim_name = str(fig_imgs).split('/')[0]                         # Simulation name (for plot labels)
    sim_params = load_par_file(f"{sim_name}/{sim_name}.par")       # Loading simulation parameters from the .par file
    
    # Save simulation parameters into dictionary for plotting purposes
    plot_args = {"Time": f"{int(calc_simtime(it))} kyr", "b": sim_params['ImpactParameter']}
    

    ###################### Load data (theta = 175, r = 150, phi = 100) ################################


    domains = get_domain_spherical(folder)
    rho = get_data(folder, "dens", it, domains)         # Load 3D array of density values            
    vphi = get_data(folder, "vx", it, domains)          # Load 3D array of azimuthal velocities v_phi
    vrad = get_data(folder, "vy", it, domains)          # Load 3D array of radial velocities v_rad
    vthe = get_data(folder, "vz", it, domains)          # Load 3D array of colatitude velocities v_theta
    energy =  get_data(folder, "energy", it, domains)   # Load 3D array of energies 

    vsph = np.sqrt(vphi**2 + vrad**2 + vthe**2)         # Total velocities in spherical coordinates

    THETA, R, PHI = np.meshgrid(domains["theta"], domains["r"], domains["phi"], indexing="ij")
    X, Y, ZCYL, RCYL = sph_to_cart(THETA, R, PHI)       # Meshgrid of Cartesian coordinates

    # Cartesian velocities
    vx, vy, vz = vel_sph_to_cart(vthe, vrad, vphi, THETA, PHI)


if __name__== "__main__":
    main()