# Post-processing on FARGO3D output files 

import numpy as np
from pathlib import Path
from read import get_domain_spherical, get_data
from analysis import sph_to_cart, vel_sph_to_cart, centering, calc_angular_momentum, calc_cell_volume, calc_eccen, calc_LRL, calc_mass, calc_surfdens, isolate_warp, calc_L_average, calc_simtime
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
# import imageio
import os
from no_thoughts_just_plots import quiver_plot_3d, contours_3D, plot_surf_dens, plot_twist_arrows
import astropy.constants as c
au = c.au.cgs.value
G = 6.67e-8               # Gravitational constant in cgs units
Msun = 1.989e33           # Mass of the Sun in g
Mstar = 0.7 * Msun        # Mass of the primary star in IRAS 04125+2902 (Barber et al. 2024)
dt = 1.87e7               # Timestep length of simulations in sec
ninterm = 200              # Total number of timesteps between outputs in FARGO simulations
stoky = 3.156e7 * 1e3     # 1 kyr in sec


def main():


    folder = Path("../iras04125_lowres_it450_b10/")         # Folder with the FARGO output files
    fig_imgs = Path("cloudlet_lowres_it450_b10/imgs/")      # Folder to save images    
    iter_total = 450                                    # FARGO snapshot

    iter_check = np.arange(100, iter_total+1, 25)                       # Some iterations to plot
    sim_name = str(folder).split('/')[1]                                # Simulation name (for plot labels)
    dt_years = calc_simtime(np.asarray(range(100, iter_total+1)))       # Convert iterations to kyrs
    dtkyrs_check = calc_simtime(iter_check)                             # Convert iterations into kyrs

    inc_it = []                                               # List to save disk inclination at each iteration
    prec_it = []                                              # List to save disk precession at each iteration
    surf_dens_iter = []                                       # List to save surface density at each iter_check
    r_surf_dens_iter = []                                     # List to save surface density radii at each iter_check
    inc_itercheck = []                                        # List to save disk inclination at each iter_check
    prec_itercheck = []                                       # List to save disk precession at each iter_check

    
    ################################# Load coordinates  ################################

    domains = get_domain_spherical(folder)
    THETA, R, PHI = np.meshgrid(domains["theta"], domains["r"], domains["phi"], indexing="ij")
    rc = 0.5 * (domains["r"][1:] + domains["r"][:-1])
    X, Y, ZCYL, RCYL = sph_to_cart(THETA, R, PHI)       # Meshgrid of Cartesian coordinates

    for it in range(100, iter_total+1):
    # for it in iter_check:

        ###################### Load data for each iteration #############################

        rho = get_data(folder, "dens", it, domains)         # Load 3D array of density values            
        vphi = get_data(folder, "vx", it, domains)          # Load 3D array of azimuthal velocities v_phi
        vrad = get_data(folder, "vy", it, domains)          # Load 3D array of radial velocities v_rad
        vthe = get_data(folder, "vz", it, domains)          # Load 3D array of colatitude velocities v_theta

        vsph = np.sqrt(vphi**2 + vrad**2 + vthe**2)         # Total velocities in spherical coordinates

        # Cartesian velocities
        vx, vy, vz = vel_sph_to_cart(vthe, vrad, vphi, THETA, PHI)

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


        ########################### Isolating the warp in the primary disk #####################


        # Note 1: I am using centered densities to isolate the warp to match the indices corresponding to the warp with the angular momenta indices
        # Note 2: The warp_ids itself is a 3D Boolean array, but when applied to another array such as x[warp_ids], the latter array becomes 1D
        warp_thresh = -15   # log of density threshold for which we can see the warp in the primary
        rho_c_warp, vx_c_warp, vy_c_warp, vz_c_warp, Lx_c_warp, Ly_c_warp, Lz_c_warp, warp_ids = isolate_warp(rho_c, vx_c, vy_c, vz_c, Lx, Ly, Lz, warp_thresh) 

        # Find the radial extent of the warp
        r_warp_extent = np.sqrt(X_c[warp_ids]**2 +  Y_c[warp_ids]**2 + Z_c[warp_ids]**2) / au
        mask = (domains["r"]/au >= r_warp_extent.min()) & (domains["r"]/au <= r_warp_extent.max())
        r_select = domains["r"][mask]
        surf_dens_select = surf_dens[mask]


        #################################### Plotting warp properties #####################################


        # Plotting the warp densities 
        # contours_3D(X_c/au, Y_c/au, Z_c/au, rho_c_warp, xlabel='X [AU]', ylabel='Y [AU]', zlabel='Z [AU]', colorbarlabel=r'$\rho [g/cm^3]$', title=rf'{sim_name} $\log(\rho)$ above $\rho = 10^{{{warp_thresh}}} g/cm^3$, t = {int(it * dt * ninterm / stoky)} kyr', savefig=True, figfolder=f'{fig_imgs}/warp_dens_thresh{warp_thresh}_it{it}.png', showfig=False)

        # Plotting the Cartesian warp angular momenta
        # quiver_plot_3d(X_c/au, Y_c/au, Z_c/au, Lx_c_warp, Ly_c_warp, Lz_c_warp, stagger=2, length=5, title="Warp angular momenta", colorbarlabel="logL", savefig=False, figfolder=f'../warp_L_thresh{warp_thresh}_it{it}.png', logmag=True, ignorecol=True, showfig=False)

        # Calculating the radial profile of warp inclination and twist
        inc, twist = calc_L_average(Lx_c_warp, Ly_c_warp, Lz_c_warp, domains["r"], savefig=False, plot=False)
        inc_it.append(inc)
        prec_it.append(twist)

        # Plotting warp twist quiver plot
        plot_twist_arrows(Lx_c_warp, Ly_c_warp, Lz_c_warp, domains["r"], r_select, title=f"Warp twist {sim_name} t={int(calc_simtime(it))} kyr", savefig=True, figfolder=f'{fig_imgs}/twist_evol/warp_twist_arrows_it{it}.png', showfig=False)

        # Calculating warp surface density
        if it in iter_check:

            r_surf_dens_iter.append(r_select)
            surf_dens_iter.append(surf_dens_select)
            inc_itercheck.append(inc)
            prec_itercheck.append(twist)

    
    ewfwjlfwl
    # Plot time evolution of warp inclination in 2D for some specific iters in iter_check
    cols = cm.get_cmap('viridis', len(iter_check))
    for i in range(len(iter_check)):
        plt.plot(rc/au, inc_itercheck[i], color=cols(i), label=f"{int(dtkyrs_check[i])} kyr")
    plt.xlabel("R [AU]")
    plt.ylabel("Warp inclination [°]")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title(f"{sim_name}: Time evolution of warp inclination")
    plt.tight_layout()
    plt.savefig(f'{fig_imgs}/warp_inc_evol.png')
    plt.show()

    # Plot time evolution of warp inclination in 3D for all iters 
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    r_warp_extent = np.sqrt(X_c[warp_ids]**2 +  Y_c[warp_ids]**2 + Z_c[warp_ids]**2) / au
    mask = (rc/au >= r_warp_extent.min()) & (rc/au <= r_warp_extent.max())
    for i in range(len(dt_years[::4])):
        r_select = rc[mask]
        inc_it_select = inc_it[i][mask]
        ax.plot(r_select/au, [dt_years[::4][i]] * len(r_select), inc_it_select, color=plt.cm.viridis(i/len(dt_years[::4])))

    ax.view_init(elev=35, azim=-31)
    ax.set_xlabel('R [AU]')
    ax.set_ylabel('Time [kyr]')
    ax.set_zlabel('Warp inclination [°]')
    ax.set_title(f'{sim_name}: Time evolution of warp inclination')
    # plt.colorbar(surf, label='Angle')
    plt.tight_layout()
    plt.savefig(f'{fig_imgs}/warp_inc_evol_3D.png')
    plt.show()

    # Plot time evolution of warp precession in 2D for some specific iters in iter_check
    for i in range(len(iter_check)):
        plt.plot(rc/au, prec_itercheck[i], color=cols(i), label=f"{int(dtkyrs_check[i])} kyr")
    plt.xlabel("R [AU]")
    plt.ylabel("Warp precession [°]")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title(f"{sim_name}: Time evolution of warp twist")
    plt.tight_layout()
    plt.savefig(f'{fig_imgs}/warp_twist_evol.png')
    plt.show()

    # Plot time evolution of warp precession in 3D for all iters 
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    r_warp_extent = np.sqrt(X_c[warp_ids]**2 +  Y_c[warp_ids]**2 + Z_c[warp_ids]**2) / au
    mask = (rc/au >= r_warp_extent.min()) & (rc/au <= r_warp_extent.max())
    for i in range(len(dt_years[::4])):
        r_select = rc[mask]
        prec_it_select = prec_it[i][mask]
        ax.plot(r_select/au, [dt_years[::4][i]] * len(r_select), prec_it_select, color=plt.cm.viridis(i/len(dt_years[::4])))

    ax.view_init(elev=35, azim=-31)
    ax.set_xlabel('R [AU]')
    ax.set_ylabel('Time [kyr]')
    ax.set_zlabel('Warp precession [°]')
    ax.set_title(f'{sim_name}: Time evolution of warp precession')
    # plt.colorbar(surf, label='Angle')
    plt.tight_layout()
    plt.savefig(f'{fig_imgs}/warp_twist_evol_3D.png')
    plt.show()

    # Plot time evolution of surface densities
    for i in range(len(surf_dens_iter)):
        plt.plot(r_surf_dens_iter[i]/au, surf_dens_iter[i], color=cols(i), label=f"{int(dtkyrs_check[i])} kyr")
    plt.xlabel("R [AU]")
    plt.ylabel(r"$\Sigma [g/cm^2]$")
    plt.title(f"{sim_name}: Time evolution of warp surface density")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'{fig_imgs}/warp_surfdens_evol.png')
    plt.show()

    # Make a time evolution GIF out of the 3D surface density and twist plots
    # filenames = sorted([f for f in os.listdir(fig_imgs) if f.endswith('.png')])
    # with imageio.get_writer(f'{fig_imgs}/warp_surfdens_movie.gif', mode='I') as writer:
    #     for filename in filenames:
    #         image = imageio.imread(filename)
    #         writer.append_data(image)

if __name__ == "__main__":
    main()
