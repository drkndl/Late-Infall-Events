# Analysis of parameter study across inclination axes, Mcloud / Mdisk, Rout 

import numpy as np
from pathlib import Path
from read import get_domain_spherical, get_data, load_par_file
import matplotlib.pyplot as plt
from analysis import calc_cell_volume, calc_mass, sph_to_cart, calc_simtime, vel_sph_to_cart, centering, calc_angular_momentum, isolate_disk, calc_L_average, calc_inc_twist
from accretion import scale_height, calc_accretion
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


def main():


    folders = [Path("../fargo3d/outputs/cloud_disk_it450_rotX45"), Path("../fargo3d/outputs/cloud_disk_it450_rotY45"), Path("../fargo3d/outputs/cloud_disk_it450_Rout30_rotX45"), Path("../fargo3d/outputs/cloud_disk_it450_Rout30_rotY45"), Path("../fargo3d/outputs/cloud_disk_it450_cmass10_rotX45"), Path("../fargo3d/outputs/cloud_disk_it450_cmass10_rotY45"), Path("../fargo3d/outputs/cloud_disk_it450_cmass10_Rout30_rotX45"), Path("../fargo3d/outputs/cloud_disk_it450_cmass10_Rout30_rotY45")]

    folders_labels = {"cloud_disk_it450_rotX45": r"$\mathrm{X = 45\degree, M_{cloud} / M_{disk}=0.45, R_{out} = 100 AU}$", "cloud_disk_it450_rotY45": r"$\mathrm{Y = 45\degree, M_{cloud} / M_{disk}=0.45, R_{out} = 100 AU}$", 
    "cloud_disk_it450_Rout30_rotX45": r"$\mathrm{X = 45\degree, M_{cloud} / M_{disk}=0.45, R_{out} = 30 AU}$",
    "cloud_disk_it450_Rout30_rotY45": r"$\mathrm{Y = 45\degree, M_{cloud} / M_{disk}=0.45, R_{out} = 30 AU}$",
    "cloud_disk_it450_cmass10_rotX45": r"$\mathrm{X = 45\degree, M_{cloud} / M_{disk}=4.5, R_{out} = 100 AU}$",
    "cloud_disk_it450_cmass10_rotY45": r"$\mathrm{Y = 45\degree, M_{cloud} / M_{disk}=4.5, R_{out} = 100 AU}$",
    "cloud_disk_it450_cmass10_Rout30_rotX45": r"$\mathrm{X = 45\degree, M_{cloud} / M_{disk}=4.5, R_{out} = 30 AU}$",
    "cloud_disk_it450_cmass10_Rout30_rotY45": r"$\mathrm{Y = 45\degree, M_{cloud} / M_{disk}=4.5, R_{out} = 30 AU}$"}

    disk_mass_folder = {}              # Disk masses m(t, theta, r, phi) for all sims
    disk_inc_avg_folder = {}           # Average disk inclination inc(t) for all sims
    disk_twist_avg_folder = {}         # Average disk twist twist(t) for all sims
    disk_Mdot_folder = {}              # Mass accretion rate onto star Mdot(t) for all sims

    N = 10                                             # Load data for every N iterations

    # Central coordinates of the primary
    Px, Py, Pz = 0, 0, 0                                # Primary is in the centre of the simulation

    for f in folders:
        
        # Load simulation 
        f_sim_name = str(f).split('/')[3]                       # Simulation name (for plot labels)
        domains = get_domain_spherical(f)                       # Load coordinates  
        it = 450                                                # Final iteration (t=53kyr)    

        # Load simulation domains and create spherical and Cartesian meshgrids
        THETA, R, PHI = np.meshgrid(domains["theta"], domains["r"], domains["phi"], indexing="ij")
        X, Y, ZCYL, RCYL = sph_to_cart(THETA, R, PHI)                               # Meshgrid of Cartesian coordinates  
        cell_volume = calc_cell_volume(domains["theta"], domains["r"], domains["phi"])   # Cell volumes 
        X_c = centering(X)
        Y_c = centering(Y)
        Z_c = centering(ZCYL)
        
        # Save density, mass, vrad, average inclination, average twist at multiple iterations 
        rho_allit = []
        vrad_allit = []
        mass_allit = []
        inc_avg_allit = []
        twist_avg_allit = []


        ######################## Calculating inc, twist values ####################################


        for i in range(0, it+1, N):     

            # Loading density and velocities every N iterations
            rho_i = get_data(f, "dens", i, domains)
            vrad_i = get_data(f, "vy", i, domains) 
            vphi_i = get_data(f, "vx", i, domains) 
            vthe_i = get_data(f, "vz", i, domains) 

            rho_allit.append(rho_i)
            vrad_allit.append(vrad_i)   

            # Converting spherical velocities to Cartesian velocities
            vx_i, vy_i, vz_i = vel_sph_to_cart(vthe_i, vrad_i, vphi_i, THETA, PHI)

            # Centering rho, v
            rho_c_i = centering(rho_i)
            vx_c_i = centering(vx_i)
            vy_c_i = centering(vy_i)
            vz_c_i = centering(vz_i)

            # Calculating mass and angular momentum
            mass_i = calc_mass(rho_i, cell_volume)
            mass_allit.append(mass_i)
            Lx_i, Ly_i, Lz_i = calc_angular_momentum(mass_i, X, Y, ZCYL, vx_i, vy_i, vz_i)

            # Isolating the warped/broken disk
            warp_thresh = -15   # log of density threshold for which we can see the warp in the primary
            warp_buffer = 500   # Isolates a box of 2 * warp_buffer around the star (AU)
            _, _, _, _, Lx_c_warp_i, Ly_c_warp_i, Lz_c_warp_i, _ = isolate_disk(X_c, Y_c, Z_c, Px * au, Py * au, Pz * au, warp_buffer * au, rho_c_i, vx_c_i, vy_c_i, vz_c_i, Lx_i, Ly_i, Lz_i, warp_thresh) 

            # Calculating inclination, twist in the disk and saving the radial averages
            Lx_warp_avg_i, Ly_warp_avg_i, Lz_warp_avg_i = calc_L_average(Lx_c_warp_i, Ly_c_warp_i, Lz_c_warp_i, mass_i)
            inc_i, twist_i = calc_inc_twist(Lx_warp_avg_i, Ly_warp_avg_i, Lz_warp_avg_i, domains["r"], savefig=False, plot=False)
            print(np.shape(inc_i), np.nanmean(inc_i))
            inc_avg_allit.append(np.nanmean(inc_i))
            twist_avg_allit.append(np.nanmean(twist_i))            

        vrad_allit = np.asarray(vrad_allit)
        rho_allit = np.asarray(rho_allit)
        mass_allit = np.asarray(mass_allit)
        inc_avg_allit = np.asarray(inc_avg_allit)
        twist_avg_allit = np.asarray(twist_avg_allit)

        disk_mass_folder[f_sim_name] = mass_allit
        disk_inc_avg_folder[f_sim_name] = inc_avg_allit
        disk_twist_avg_folder[f_sim_name] = twist_avg_allit

        allit_years = calc_simtime(np.asarray(range(0, it+1, N)))       # Convert iterations to kyrs


        ############################### Calculating mass accretion values ##################################


        R0 = 5.2 * au                         # As defined in FARGO3D [cm]
        # f = sim_params['FlaringIndex']      # Flaring index
        # h0 = sim_params['AspectRatio']      # Aspect ratio
        f = 0.25                              # Flaring index (from setups/cloud_disk.par)
        h0 = 0.03799                          # Aspect ratio (from setups/cloud_disk.par)
        Hc = scale_height(domains["r"][0], h0, R0, f)
        zmax = 4 * Hc
        r0 = domains["r"][0]     # Taking the innermost radius to check accretion onto star
        dotM_in_allit = []
        # dotM_out_allit = []

        for i in range(len(allit_years)):
            _, dotM_in_i, dotM_out_i = calc_accretion(rho_allit[i], vrad_allit[i], domains["theta"], r0, 0, domains["phi"], zmax, Msun)
            dotM_in_allit.append(dotM_in_i)
            # dotM_out_allit.append(dotM_out_i)

        dotM_in_allit = np.asarray(dotM_in_allit)
        # dotM_out_allit = np.asarray(dotM_out_allit)

        disk_Mdot_folder[f_sim_name] = dotM_in_allit
        # Mdot_out_allincs_nodisk[f_sim_name] = dotM_out_allit

    
    ############################################ Plotting ############################################


    # Plotting inc_avg vs time 
    fig, ax = plt.subplots()
    for key, value in disk_inc_avg_folder.items():
        ax.plot(allit_years, value, label=folders_labels[key])
        # ax.plot(allit_years, dotM_out_allit, label="Outward flux")
    ax.set_xlabel(r"Time [kyr]")
    ax.set_ylabel(r"$\mathrm{inc_{avg}}$")
    ax.set_title(fr"Time Evolution of Average Inclinations")
    fig.tight_layout()
    ax.legend(loc="lower right")   # loc='upper left', 
    plt.savefig('param_study_inc_avg_vs_t.png')
    plt.show()

    # Plotting twist_avg vs time 
    fig, ax = plt.subplots()
    for key, value in disk_twist_avg_folder.items():
        ax.plot(allit_years, value, label=folders_labels[key])
        # ax.plot(allit_years, dotM_out_allit, label="Outward flux")
    ax.set_xlabel(r"Time [kyr]")
    ax.set_ylabel(r"$\mathrm{twist_{avg}}$")
    ax.set_title(fr"Time Evolution of Average Twist")
    fig.tight_layout()
    ax.legend(loc="lower right")   # loc='upper left', 
    plt.savefig('param_study_twist_avg_vs_t.png')
    plt.show()


if __name__ == "__main__":
    main()