import numpy as np
import matplotlib.pyplot as plt 
from pathlib import Path
from read import get_domain_spherical, get_data, load_par_file, get_param_value
import astropy.constants as c
from scipy.interpolate import griddata
from analysis import calc_cell_volume, calc_mass, sph_to_cart, calc_simtime, vel_sph_to_cart, centering, calc_angular_momentum, isolate_disk, calc_L_average, calc_inc_twist, calc_whirl, calc_total_L, ini_cloudlet_pos, isolate_outer_disk
from accretion import scale_height, calc_accretion
# from no_thoughts_just_plots import velocity_streamlines

au = c.au.cgs.value
G = 6.67e-8               # Gravitational constant in cgs units
Msun = 1.989e33           # Mass of the Sun in g
Mstar = 0.7 * Msun        # Mass of the primary star in IRAS 04125+2902 (Barber et al. 2024)
dt = 1.87e7               # Timestep length of simulations in sec
ninterm = 200             # Total number of timesteps between outputs in FARGO simulations
stoky = 3.156e7 * 1e3     # 1 kyr in sec

# Global plot formatting 
plt.rcParams['lines.linewidth'] = 2.5
plt.rcParams['axes.labelsize'] = 16     # x/y label size
plt.rcParams['xtick.labelsize'] = 14     # x-tick label size
plt.rcParams['ytick.labelsize'] = 14     # y-tick label size
plt.rcParams['legend.fontsize'] = 12     # legend font size


def main():
    
    folders = [Path("F:/cloud_disk_it450_rotX45/"), Path("F:/cloud_disk_it450_retro_rotX45/")]
    fig_imgs = Path("paper_plots/")                    # Folder to save images
    it = 450                                                       # FARGO snapshot of interest
    # sim_name = str(fig_imgs).split('/')[0]                         # Simulation name (for plot labels)

    row_labels = ["Prograde", "Retrograde"]

    fig, axes = plt.subplots(2, 3, figsize=(10, 6), sharex=True, sharey=True)
    showfig=True
    savefig=True
    
    for dataset_idx, folder in enumerate(folders):

        ###################### Load data (theta = 175, r = 150, phi = 100) ################################

        domains = get_domain_spherical(folder)
        rho = get_data(folder, "dens", it, domains)         # Load 3D array of density values            
        vphi = get_data(folder, "vx", it, domains)          # Load 3D array of azimuthal velocities v_phi
        vrad = get_data(folder, "vy", it, domains)          # Load 3D array of radial velocities v_rad
        vthe = get_data(folder, "vz", it, domains)          # Load 3D array of colatitude velocities v_theta

        vsph = np.sqrt(vphi**2 + vrad**2 + vthe**2)         # Total velocities in spherical coordinates

        THETA, R, PHI = np.meshgrid(domains["theta"], domains["r"], domains["phi"], indexing="ij")
        X, Y, ZCYL, RCYL = sph_to_cart(THETA, R, PHI)       # Meshgrid of Cartesian coordinates

        # Cartesian velocities
        vx, vy, vz = vel_sph_to_cart(vthe, vrad, vphi, THETA, PHI)

        # Central coordinates of the primary
        Px, Py, Pz = 0, 0, 0                                # Primary is in the centre of the simulation


        ############################# Calculate physical quantities ######################################


        # Interpolate the densities & coordinates to the cell centres so that the array shape matches with mass & L
        rho_c = centering(rho)
        X_c = centering(X)
        Y_c = centering(Y)
        Z_c = centering(ZCYL)
        vx_c = centering(vx)
        vy_c = centering(vy)
        vz_c = centering(vz)

        cell_volume = calc_cell_volume(domains["theta"], domains["r"], domains["phi"])          # Volume 3D
        mass = calc_mass(rho, cell_volume)                                                      # Mass 3D
        # surf_dens = calc_surfdens(rho, domains["theta"], domains["r"], domains["phi"])          # Surface density 3D
        Lx, Ly, Lz = calc_angular_momentum(mass, X, Y, ZCYL, vx, vy, vz)                        # Angular momentum 3D
        # Ax, Ay, Az = calc_LRL(mass, Mstar, vx_c, vy_c, vz_c, Lx, Ly, Lz, X_c, Y_c, Z_c)         # Laplace-Runge-Lenz vector 3D
        # ex, ey, ez = calc_eccen(Ax, Ay, Az, mass, Mstar)                                        # Eccentricity 3D
        # e = np.sqrt(ex**2 + ey**2 + ez**2)                                                      # Absolute eccentricity 3D

        ########################### Isolating the warp in the primary disk ###############################


        # Note 1: I am using centered densities to isolate the warp to match the indices corresponding to the warp with the angular momenta indices
        # Note 2: The warp_ids itself is a 3D Boolean array, but when applied to another array such as x[warp_ids], the latter array becomes 1D
        warp_thresh = -17   # log of density threshold for which we can see the warp in the primary
        warp_buffer = 500   # Isolates a box of 2 * warp_buffer around the star (AU)
        rho_c_warp, vx_c_warp, vy_c_warp, vz_c_warp, Lx_c_warp, Ly_c_warp, Lz_c_warp, warp_ids = isolate_disk(X_c, Y_c, Z_c, Px * au, Py * au, Pz * au, warp_buffer * au, rho_c, vx_c, vy_c, vz_c, Lx, Ly, Lz, warp_thresh) 
    

        # Row label on the leftmost axis
        axes[dataset_idx, 2].set_ylabel(row_labels[dataset_idx], fontsize=14, rotation=270, labelpad=15)
        axes[dataset_idx, 2].yaxis.set_label_position("right")
        irad = np.where(domains["r"]/au < 200)[0][-1]
        targets = np.array([50, 90, 120])   
        ithetas = [np.abs(np.rad2deg(domains["theta"]) - t).argmin() for t in targets]

        for col_idx, theta_i in enumerate(ithetas):

            ax = axes[dataset_idx, col_idx]

            Xc = X_c[theta_i, :irad, ...] / au
            Yc = Y_c[theta_i, :irad, ...] / au
            Uc = vx_c_warp[theta_i, :irad, ...]
            Vc = vy_c_warp[theta_i, :irad, ...]

            x_reg = np.linspace(Xc.min(), Xc.max(), 300)
            y_reg = np.linspace(Yc.min(), Yc.max(), 300)
            Xg, Yg = np.meshgrid(x_reg, y_reg)

            Ugrid   = griddata((Xc.ravel(), Yc.ravel()), Uc.ravel(), (Xg, Yg))
            Vgrid   = griddata((Xc.ravel(), Yc.ravel()), Vc.ravel(), (Xg, Yg))
            RHOgrid = griddata((Xc.ravel(), Yc.ravel()), rho_c_warp[theta_i, :irad, ...].ravel(), (Xg, Yg))

            map = ax.pcolormesh(Xg, Yg, np.log10(RHOgrid), cmap="Spectral_r", vmin=-19, vmax=-11)
            ax.streamplot(Xg, Yg, Ugrid, Vgrid, color="black")
            ax.set_aspect("equal")
            ax.set_title(rf"$\theta$ = {theta_i}$\degree$")

    fig.supxlabel(r"X [AU]")
    fig.supylabel(r"Y [AU]")

    plt.tight_layout(rect=[0, 0, 0.85, 1])

    cax = fig.add_axes([0.87, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(map, cax=cax)
    cbar.set_label(r"$\log(\rho)$")

    if savefig:
        plt.savefig(f'{fig_imgs}/vel_streamlines_thresh{warp_thresh}_it{it}.pdf', bbox_inches="tight", dpi=300)

    if showfig:
        plt.show()
    else:
        plt.close()




if __name__=="__main__":
    main() 

