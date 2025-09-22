import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib import cm 
import matplotlib.colors as mcolors
from pathlib import Path
from read import get_domain_spherical, get_data, load_par_file
from analysis import calc_cell_volume, calc_mass, sph_to_cart, calc_simtime
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import astropy.constants as c
import pandas as pd
au = c.au.cgs.value
G = 6.67e-8               # Gravitational constant in cgs units
Msun = 1.989e33           # Mass of the Sun in g
Mstar = 0.7 * Msun        # Mass of the primary star in IRAS 04125+2902 (Barber et al. 2024)
dt = 1.87e7               # Timestep length of simulations in sec
ninterm = 200             # Total number of timesteps between outputs in FARGO simulations
stoky = 3.156e7 * 1e3     # 1 kyr in sec


def main():

    folder = Path("../cloud_disk_it450_rotXY30/")         # Folder with the output files
    fig_imgs = Path("cloud_disk_it450_rotXY30/imgs/")     # Folder to save images
    it = 450                                                       # FARGO snapshot of interest
    sim_name = str(fig_imgs).split('/')[0]                         # Simulation name (for plot labels)
    

    ############# Load data for single snapshot (theta = 175, r = 150, phi = 100) ######################


    domains = get_domain_spherical(folder)
    rho = get_data(folder, "dens", it, domains)         # Load 3D array of density values            

    THETA, R, PHI = np.meshgrid(domains["theta"], domains["r"], domains["phi"], indexing="ij")
    X, Y, ZCYL, RCYL = sph_to_cart(THETA, R, PHI)       # Meshgrid of Cartesian coordinates

    cell_volume = calc_cell_volume(domains["theta"], domains["r"], domains["phi"])
    mass = calc_mass(rho, cell_volume)


    ############################## Load data for multiple snapshots ####################################


    # Load mass values at multiple iterations 
    mass_allit = []

    for i in range(0, it+1, 10):     # loading density every 10 iterations
        rho_i = get_data(folder, "dens", i, domains)
        mass_i = calc_mass(rho_i, cell_volume)
        mass_allit.append(mass_i)

    mass_allit = np.asarray(mass_allit)


    ############################# Mass in each spherical shell ########################################


    # Mass in each spherical shell for a single iteration
    shell_mass = np.sum(mass, axis=(0,2))                       # Shell mass in shape (nr-1)

    fig, ax = plt.subplots()
    ax.plot(np.log10(domains["r"]/au)[:-1], np.log10(shell_mass))
    ax.set_xlabel(r"$\log(r)$ [AU]")
    ax.set_ylabel(r"$\log(M(r))$ [g]")
    ax.set_title(f"{sim_name}: logM(r) vs logr")
    plt.savefig(f'{fig_imgs}/logM_vs_logr_it{it}.png')
    plt.show()

    # Time evolution of mass in spherical shells
    shell_mass_allit = np.sum(mass_allit, axis=(1,3))           # Shell mass in shape (evol_it, nr-1)
    evol_it = len(shell_mass_allit[:,1])                        # Total number of snapshots loaded
    dtkyrs = calc_simtime(np.asarray(range(0, it+1, 10)))       # Convert iterations to kyrs
    
    cols = cm.get_cmap('viridis', evol_it)
    fig, ax = plt.subplots()
    for i in range(evol_it):
        plt.plot(np.log10(domains["r"]/au)[:-1], np.log10(shell_mass_allit[i]), color=cols(i))
    ax.set_xlabel(r"$\log(r)$ [AU]")
    ax.set_ylabel(r"$\log(M(r))$ [g]")
    ax.set_title(f"{sim_name}: logM(r) vs logr time evolution")

    norm = mcolors.Normalize(vmin=min(dtkyrs), vmax=max(dtkyrs))
    sm = cm.ScalarMappable(cmap=cols, norm=norm)
    sm.set_array([])  # Required for older matplotlib versions

    # Add the colorbar
    cbar = plt.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("Time [kyr]")
    plt.savefig(f'{fig_imgs}/logM_vs_logr_timeevol.png')
    plt.show()


    ############################## log(dM/dr) in each spherical shell ######################################




if __name__ == "__main__":
    main()
    