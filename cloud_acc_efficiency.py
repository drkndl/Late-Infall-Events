# Calculate efficiency of cloud accretion onto disk during late infall 

import numpy as np
from pathlib import Path
from read import get_domain_spherical, get_data, load_par_file, get_param_value
import matplotlib.pyplot as plt
from analysis import calc_simtime, sph_to_cart, vel_sph_to_cart, centering, calc_cell_volume, calc_mass, calc_surfdens, calc_angular_momentum, calc_LRL, calc_eccen, isolate_disk, ini_cloudlet_pos
from no_thoughts_just_plots import quiver_plot_3d, contours_3D, plot_surf_dens, plot_twist_arrows, plot_total_disks_bonanza, cyl_2D_plot, XY_2D_plot, velocity_streamlines, plotly_quiver3D, vel_cyl_2D, plot_disk_sep, interactive_2D
import astropy.constants as c
import pandas as pd
# import pyvista as pv 

# pv.global_theme.allow_empty_mesh = True
au = c.au.cgs.value
G = 6.67e-8               # Gravitational constant in cgs units
Msun = 1.989e33           # Mass of the Sun in g
Mstar = 0.7 * Msun        # Mass of the primary star in IRAS 04125+2902 (Barber et al. 2024)
dt = 1.87e7               # Timestep length of simulations in sec
ninterm = 200             # Total number of timesteps between outputs in FARGO simulations
stoky = 3.156e7 * 1e3     # 1 kyr in sec

def main():

    folder = Path("../cloud_disk_it450_b09_rotX45/")                        # Folder with the FARGO output files
    # folder = Path("../fargo3d/outputs/cloud_disk_it450_b09_rotX45/")      # Folder with the FARGO output files (Binac2)
    fig_imgs = Path("cloud_disk_it450_b09_rotX45/imgs/")                    # Folder to save images    
    iter_total = 450                                     # FARGO snapshot

    first_it = 0
    sim_name = str(fig_imgs).split('/')[0]                                     # Simulation name (for plot labels)
    dt_years = calc_simtime(np.asarray(range(first_it, iter_total+1, 10)))     # Convert iterations to kyrs

    disk_mass_allit = []                             # List to save the mass of the disk at each iteration

    
    ################################# Load coordinates  ################################

    domains = get_domain_spherical(folder)
    THETA, R, PHI = np.meshgrid(domains["theta"], domains["r"], domains["phi"], indexing="ij")
    rc = 0.5 * (domains["r"][1:] + domains["r"][:-1])
    X, Y, ZCYL, RCYL = sph_to_cart(THETA, R, PHI)       # Meshgrid of Cartesian coordinates

    # Calculating cloudlet mass
    cloud_mass_ini = get_param_value('CloudletMass', sim_name)
    print(f"Cloudlet mass: {cloud_mass_ini/Msun:.4f} Msun, {cloud_mass_ini:.2e} g")

    for it in range(first_it, iter_total+1, 10):
    

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
        Lx, Ly, Lz = calc_angular_momentum(mass, X, Y, ZCYL, vx, vy, vz)


        ########################### Isolating the warp in the primary disk #####################


        # Note 1: I am using centered densities to isolate the warp to match the indices corresponding to the warp with the angular momenta indices
        # Note 2: The warp_ids itself is a 3D Boolean array, but when applied to another array such as x[warp_ids], the latter array becomes 1D
        warp_thresh = -17   # log of density threshold for which we can see the warp in the primary
        warp_buffer = 500     # Isolates a box of 2 * warp_buffer around the star (AU)
        rho_c_warp, vx_c_warp, vy_c_warp, vz_c_warp, Lx_c_warp, Ly_c_warp, Lz_c_warp, warp_ids = isolate_disk(X_c, Y_c, Z_c, Px * au, Py * au, Pz * au, warp_buffer * au, rho_c, vx_c, vy_c, vz_c, Lx, Ly, Lz, warp_thresh) 


        ######################## Calculating the mass of the warped/broken disk ###################


        disk_mass_radial_it = np.nansum(rho_c_warp * cell_volume, axis=(0,2))
        # plt.plot(np.log10(domains["r"][:-1]/au), np.log10(disk_mass_radial_it))
        # plt.show()        
        disk_mass_it = np.sum(disk_mass_radial_it)
        print((it))
        print(f"Disk mass: {disk_mass_it/Msun:.4f} Msun, {disk_mass_it:.2e} g")
        print(f"Cloud mass: {cloud_mass_ini/Msun:.4f} Msun, {cloud_mass_ini:.2e} g")
        print(f"Rout 30 AU Mc/Md ratio: {cloud_mass_ini/disk_mass_it:.4f}")
        disk_mass_allit.append(disk_mass_it)


    ############################## Calculate cloudlet accretion efficiency ###################################


    disk_mass_initial = disk_mass_allit[0]
    cloud_mass_accreted = disk_mass_allit - disk_mass_initial
    cloud_acc_eff = cloud_mass_accreted / cloud_mass_ini * 100
    print(cloud_mass_ini)

    fig, ax = plt.subplots()
    ax.plot(dt_years, cloud_mass_accreted/Msun)
    ax.axhline(cloud_mass_ini/Msun, color='black', ls=":")
    ax.set_xlabel(r"Time t [kyr]")
    ax.set_ylabel(r"$\log(M_{cloud,acc})$ [$M_\odot$]")
    # ax.set_title("Disk surface density profile")
    fig.tight_layout()
    plt.savefig(f'{fig_imgs}/cloud_mass_accreted.png')
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(dt_years, cloud_acc_eff)
    ax.set_xlabel(r"$Time [kyr]$")
    ax.set_ylabel(r"Cloud Accretion Efficiency [%] $\frac{M_{cloud,acc}}{M_{cloud}}$")
    # ax.set_title("Disk surface density profile")
    fig.tight_layout()
    plt.savefig(f'{fig_imgs}/cloud_accretion_efficiency.png')
    plt.show()

if __name__ == "__main__":
    main()