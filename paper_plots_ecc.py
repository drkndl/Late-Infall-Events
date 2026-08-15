import numpy as np
from pathlib import Path
from read import get_domain_spherical, get_data, load_par_file, get_param_value
from analysis import sph_to_cart, vel_sph_to_cart, centering, calc_angular_momentum, calc_cell_volume, calc_eccen, calc_LRL, calc_mass, calc_surfdens, isolate_disk, calc_L_average, calc_simtime, calc_inc_twist, calc_whirl, calc_total_L, ini_cloudlet_pos, isolate_outer_disk, calc_e_average
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import cm
import matplotlib.colors as colors
import colormaps as cmaps
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import pandas as pd
import os
from no_thoughts_just_plots import XY_2D_plot, cyl_2D_plot, quiver_plot_3d, contours_3D, plot_surf_dens, plot_twist_arrows, make_evol_GIF, plot_total_disks_bonanza, plot_disk_sep, vel_cyl_2D
import astropy.constants as c
au = c.au.cgs.value
G = 6.67e-8               # Gravitational constant in cgs units
Msun = 1.989e33           # Mass of the Sun in g
Mstar = 0.7 * Msun        # Mass of the primary star in IRAS 04125+2902 (Barber et al. 2024)
dt = 1.87e7               # Timestep length of simulations in sec
ninterm = 200             # Total number of timesteps between outputs in FARGO simulations
stoky = 3.156e7 * 1e3     # 1 kyr in sec

# Global plot formatting 
plt.rcParams['lines.linewidth'] = 3
plt.rcParams['axes.labelsize'] = 20     # x/y label size
plt.rcParams['xtick.labelsize'] = 18     # x-tick label size
plt.rcParams['ytick.labelsize'] = 18     # y-tick label size
plt.rcParams['legend.fontsize'] = 18     # legend font size


def main():


    folders = [Path("F:/cloud_disk_it450_rotX45/"), Path("F:/cloud_disk_it450_cmass10_rotX45/")]                        # Folder with the FARGO output files
    # folder = Path("../fargo3d/outputs/cloud_disk_it450_rotX45/")          # Folder with the FARGO output files (Binac2)
    fig_imgs = Path("paper_plots/")                      # Folder to save images    
    iter_total = 450                                     # FARGO snapshot

    first_it = 100
    iter_check = np.arange(first_it, iter_total+1, 50)                       # Some iterations to plot
    sim_name = str(fig_imgs).split('/')[0]                                     # Simulation name (for plot labels)
    dt_years = calc_simtime(np.asarray(range(first_it, iter_total+1, 10)))       # Convert iterations to kyrs
    dtkyrs_check = calc_simtime(iter_check)                                  # Convert iterations into kyrs

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
    images = []
    
    for dataset_idx, folder in enumerate(folders):

        ax = axes[dataset_idx]
        e_it = []                                                 # List to save eccentricity at each iteration
    
    ################################# Load coordinates  ################################

        domains = get_domain_spherical(folder)
        THETA, R, PHI = np.meshgrid(domains["theta"], domains["r"], domains["phi"], indexing="ij")
        rc = 0.5 * (domains["r"][1:] + domains["r"][:-1])
        X, Y, ZCYL, RCYL = sph_to_cart(THETA, R, PHI)       # Meshgrid of Cartesian coordinates

        for it in range(first_it, iter_total+1, 10):
        # for it in iter_check:

            ###################### Load data for each iteration #############################

            rho = get_data(folder, "dens", it, domains)         # Load 3D array of density values            
            vphi = get_data(folder, "vx", it, domains)          # Load 3D array of azimuthal velocities v_phi
            vrad = get_data(folder, "vy", it, domains)          # Load 3D array of radial velocities v_rad
            vthe = get_data(folder, "vz", it, domains)          # Load 3D array of colatitude velocities v_theta

            vsph = np.sqrt(vphi**2 + vrad**2 + vthe**2)         # Total velocities in spherical coordinates

            # Cartesian velocities
            vx, vy, vz = vel_sph_to_cart(vthe, vrad, vphi, THETA, PHI)

            # Central coordinates of the primary
            Px, Py, Pz = 0, 0, 0                                # Primary is in the centre of the simulation

            ###################### Calculate physical quantities ###############################


            # Interpolate the densities & coordinates to the cell centres so that the array shape matches with mass & L
            rho_c = centering(rho)
            X_c = centering(X)
            Y_c = centering(Y)
            Z_c = centering(ZCYL)
            vx_c = centering(vx)
            vy_c = centering(vy)
            vz_c = centering(vz)

            cell_volume = calc_cell_volume(domains["theta"], domains["r"], domains["phi"])
            mass = calc_mass(rho, cell_volume)
            surf_dens = calc_surfdens(rho, domains["theta"], domains["r"], domains["phi"])
            Lx, Ly, Lz = calc_angular_momentum(mass, X, Y, ZCYL, vx, vy, vz)
            Ax, Ay, Az = calc_LRL(mass, Mstar, vx_c, vy_c, vz_c, Lx, Ly, Lz, X_c, Y_c, Z_c)         # Laplace-Runge-Lenz vector 3D
            ex, ey, ez = calc_eccen(Ax, Ay, Az, mass, Mstar)                                        # Eccentricity 3D
            e = np.sqrt(ex**2 + ey**2 + ez**2)                                                      # Absolute eccentricity 3D 


            ########################### Isolating the warp in the primary disk #####################


            # Note 1: I am using centered densities to isolate the warp to match the indices corresponding to the warp with the angular momenta indices
            # Note 2: The warp_ids itself is a 3D Boolean array, but when applied to another array such as x[warp_ids], the latter array becomes 1D
            warp_thresh = -17   # log of density threshold for which we can see the warp in the primary
            warp_buffer = 500     # Isolates a box of 2 * warp_buffer around the star (AU)
            rho_c_warp, vx_c_warp, vy_c_warp, vz_c_warp, Lx_c_warp, Ly_c_warp, Lz_c_warp, warp_ids = isolate_disk(X_c, Y_c, Z_c, Px * au, Py * au, Pz * au, warp_buffer * au, rho_c, vx_c, vy_c, vz_c, Lx, Ly, Lz, warp_thresh) 

            Ax_warp, Ay_warp, Az_warp = calc_LRL(mass, Mstar, vx_c_warp, vy_c_warp, vz_c_warp, Lx_c_warp, Ly_c_warp, Lz_c_warp, X_c, Y_c, Z_c)
            ex_warp, ey_warp, ez_warp = calc_eccen(Ax_warp, Ay_warp, Az_warp, mass, Mstar)
            eavg_disk = calc_e_average(ex_warp, ey_warp, ez_warp, mass)
            e_it.append(eavg_disk)

        # Contour plot of time evolution of radial profile of mass-averaged eccentricity of the disk only (not the whole sim space) 
        e_it = np.array(e_it, dtype=float)
        print(f"Max eccentricity: {np.max(e_it)}")

        ee = ax.imshow(e_it, cmap=cmaps.haline, aspect='auto', origin='lower', extent=[(np.log10(domains["r"]/au)).min(), (np.log10(domains["r"]/au)).max(), dt_years.min(), dt_years.max()], vmin=0.0, vmax=1.0)
        images.append(ee)



        ax.set_xlabel("log(R / AU)")
        if dataset_idx == 0:
            ax.set_ylabel("Time [kyr]")
            label_text = r'$M_c/M_d=0.45$' 
        else:
            label_text = r'$M_c/M_d=4.5$'
        ax.text(
            0.97, 0.95, label_text,
            transform=ax.transAxes,       # use axes-fraction coords, not data coords
            ha='right', va='top',
            fontsize=20,color='black',
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='square', pad=0.3)
        )
    
    fig.tight_layout()
    cbar = fig.colorbar(images[0], ax=axes, orientation='horizontal', location='top', shrink=0.9, pad=0.05)
    cbar.set_label("e")
    plt.savefig(f"{fig_imgs}/e_time_evol_diskonly.pdf", bbox_inches="tight", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
