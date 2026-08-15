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

# Global plot formatting 
plt.rcParams['lines.linewidth'] = 2.5
plt.rcParams['axes.labelsize'] = 18     # x/y label size
plt.rcParams['xtick.labelsize'] = 16     # x-tick label size
plt.rcParams['ytick.labelsize'] = 16     # y-tick label size
plt.rcParams['legend.fontsize'] = 16     # legend font size


def main():

    folders = [Path("F:/cloud_disk_it450_rotX45/"), Path("F:/cloud_disk_it450_cmass10_rotX45/"), Path("F:/cloud_disk_it450_retro_rotX45/"), Path("F:/cloud_disk_it450_b01_rotX45/")]                        # Folder with the output files
    # folder = Path("../fargo3d/outputs/cloud_disk_it450_retro_rotY45")     # Folder with the output files (BinAC2)
    fig_imgs = Path("paper_plots/")                  # Folder to save images
    it = 450                                                             # FARGO snapshot of interest
    # sim_name = str(fig_imgs).split('/')[0]                               # Simulation name (for plot labels)


    ############# Load data for single snapshot (theta = 175, r = 150, phi = 100) ######################

    fig, axes = plt.subplots(2, 2, figsize=(15, 12), sharex=True, sharey=True)
    axes_flat = axes.flatten()
    label_texts = [r'$M_c/M_d=0.45$', r'$M_c/M_d=4.5$', 'Retrograde', r'$b/b_{crit}=0.1$']

    for dataset_idx, folder in enumerate(folders):

        ax = axes_flat[dataset_idx]

        domains = get_domain_spherical(folder)
        rho = get_data(folder, "dens", it, domains)         # Load 3D array of density values   
        vrad = get_data(folder, "vy", it, domains)          # Load 3D array of radial velocities v_rad
        cell_volume = calc_cell_volume(domains["theta"], domains["r"], domains["phi"])
        mass = calc_mass(rho, cell_volume)


        ############################## Load data for multiple snapshots ####################################


        # Load mass values at multiple iterations 
        mass_allit = []
        
        for i in range(0, it+1, 10):     # loading density and vrad every 10 iterations
            rho_i = get_data(folder, "dens", i, domains)          
            mass_i = calc_mass(rho_i, cell_volume)
            mass_allit.append(mass_i)

        mass_allit = np.asarray(mass_allit)
        allit_years = calc_simtime(np.asarray(range(0, it+1, 10)))       # Convert iterations to kyrs


        ############################# Mass in each spherical shell ########################################


        # Mass in each spherical shell for a single iteration
        shell_mass = np.sum(mass, axis=(0,2))                       # Shell mass in shape (nr-1)

        # # Time evolution of mass in spherical shells
        shell_mass_allit = np.sum(mass_allit, axis=(1,3))           # Shell mass in shape (evol_it, nr-1)
        evol_it = len(shell_mass_allit[:,1])                        # Total number of snapshots loaded
        dtkyrs = calc_simtime(np.asarray(range(0, it+1, 10)))       # Convert iterations to kyrs
        
        cols = cm.get_cmap('viridis', evol_it)

        ############################################### Cumulative mass ###############################################

        M_cumsum = np.cumsum(shell_mass)
        M_cumsum_allit = np.cumsum(shell_mass_allit, axis=1)

        ############################### log ((dM_cumsum) / dr) in each spherical shell ##############################

        dM_cum = np.diff(M_cumsum)
        dM_cum_allit = np.diff(M_cumsum_allit, axis=1)

        ############################### log((dM_cumsum) / dlog(r)) in each spherical shell ##############################

        dlogR = np.diff(np.log10(domains["r"][:-1]))
        
        for i in range(evol_it):
            ax.plot(np.log10(domains["r"]/au)[:-2], np.log10(dM_cum_allit[i]/dlogR), color=cols(i))
        # ax.set_xlabel(r"$\log(r)$ [AU]")
        # ax.set_ylabel(r"$\mathrm{\log(dM_{cum}(r)/d\log(r))}$")
        ax.axvline(2, linestyle=":", color="black")
        ax.set_ylim(27, 34)

        ax.text(0.97, 0.95, label_texts[dataset_idx],
        transform=ax.transAxes,
        ha='right', va='top',
        fontsize=20, color='black',
        bbox=dict(facecolor='white', edgecolor='black', boxstyle='square',pad=0.3))

        if dataset_idx in [2, 3]:
            ax.set_xlabel(r"$\log(r)$ [AU]")
        if dataset_idx in [0, 2]:
            ax.set_ylabel(r"$\mathrm{\log(dM_{cum}(r)/d\log(r))}$")

    norm = mcolors.Normalize(vmin=min(dtkyrs), vmax=max(dtkyrs))
    sm = cm.ScalarMappable(cmap=cols, norm=norm)
    sm.set_array([])
    fig.tight_layout()
    cbar = fig.colorbar(sm, ax=axes, shrink=0.8, pad=0.03)
    cbar.set_label("Time [kyr]")

    plt.savefig(f'{fig_imgs}/dMcumdlogr_vs_logr_timeevol.pdf', dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
    
