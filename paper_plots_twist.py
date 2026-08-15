import numpy as np
from pathlib import Path
from read import get_domain_spherical, get_data
import matplotlib.pyplot as plt
from analysis import calc_cell_volume, calc_mass, sph_to_cart, calc_simtime, vel_sph_to_cart, centering, calc_angular_momentum, isolate_disk, calc_L_average, calc_inc_twist
import astropy.constants as c
au = c.au.cgs.value
G = 6.67e-8               # Gravitational constant in cgs units
Msun = 1.989e33           # Mass of the Sun in g
Mstar = 0.7 * Msun        # Mass of the primary star in IRAS 04125+2902 (Barber et al. 2024)
dt = 1.87e7               # Timestep length of simulations in sec
ninterm = 200             # Total number of timesteps between outputs in FARGO simulations
stoky = 3.156e7 * 1e3     # 1 kyr in sec

# Global plot formatting 
plt.rcParams['lines.linewidth'] = 4.5
plt.rcParams['axes.labelsize'] = 19     # x/y label size
plt.rcParams['xtick.labelsize'] = 18     # x-tick label size
plt.rcParams['ytick.labelsize'] = 18     # y-tick label size
plt.rcParams['legend.fontsize'] = 19     # legend font size

colours = ['#0072B2', '#E69F00', '#009E73', '#CC79A7', '#D55E00']
# blue,      orange,     green,      pink,      vermillion

def find_twist_cutoff(twist, abs_jump_thresh=15, window=5, sustain_frac=0.8):
    """
    Find the first index where twist jumps by more than abs_jump_thresh
    (degrees) between adjacent bins AND stays away from the pre-jump
    level for at least `window` points afterward (to distinguish a
    sustained artifact jump from a single noisy outlier).
    """
    twist = np.asarray(twist)
    dtwist = np.diff(twist)

    for i in range(len(dtwist)):
        if np.abs(dtwist[i]) > abs_jump_thresh:
            pre_val = twist[i]
            ahead = twist[i+1:i+1+window]
            if len(ahead) == 0:
                continue
            # require most of the following points to stay on the far side of the jump
            far_side = np.abs(ahead - pre_val) > abs_jump_thresh * 0.7
            if np.mean(far_side) >= sustain_frac:
                return i + 1  # cut right before the jump

    return len(twist)


def main():

    # Simulation data locally
    folders = [Path("F:/cloud_disk_it450_rotX45"), Path("F:/cloud_disk_it450_cmass5_rotX45"), Path("F:/cloud_disk_it450_cmass7_rotX45"), Path("F:/cloud_disk_it450_cmass10_rotX45"), Path("F:/cloud_disk_it450_cmass15_rotX45")] 

    folders_labels = {"cloud_disk_it450_rotX45": r"$\mathrm{M_{c} / M_{d}=0.45}$",
    "cloud_disk_it450_cmass5_rotX45": r"$\mathrm{M_{c} / M_{d}=1.5}$",
    "cloud_disk_it450_cmass7_rotX45": r"$\mathrm{M_{c} / M_{d}=3}$",
    "cloud_disk_it450_cmass10_rotX45": r"$\mathrm{M_{c} / M_{d}=4.5}$",
    "cloud_disk_it450_cmass15_rotX45": r"$\mathrm{M_{c} / M_{d}=8}$"} 

    disk_twist_folder = {}             # Disk twists twist(r, t) at all radii and all timesteps for all sims

    N = 10                                             # Load data for every N iterations

    # Central coordinates of the primary
    Px, Py, Pz = 0, 0, 0                                # Primary is in the centre of the simulation

    for f in folders:
        
        # Load simulation 
        f_sim_name = f.name                       # Simulation name (for plot labels)
        domains = get_domain_spherical(f)                       # Load coordinates  
        it = 450                                                # Final iteration (t=53kyr)    

        # Load simulation domains and create spherical and Cartesian meshgrids
        THETA, R, PHI = np.meshgrid(domains["theta"], domains["r"], domains["phi"], indexing="ij")
        X, Y, ZCYL, RCYL = sph_to_cart(THETA, R, PHI)                               # Meshgrid of Cartesian coordinates 
        RCYL_c = centering(RCYL)
        ZCYL_c = centering(ZCYL) 
        cell_volume = calc_cell_volume(domains["theta"], domains["r"], domains["phi"])   # Cell volumes 
        X_c = centering(X)
        Y_c = centering(Y)
        Z_c = centering(ZCYL)

        rc = 0.5 * (domains["r"][1:] + domains["r"][:-1])
        
        # Save density, mass, vrad, average inclination, average twist at multiple iterations 
        twist_allit = []                           


        ######################## Calculating mass, inc, twist values ####################################


        for i in range(0, it+1, N):     

            # Loading density and velocities every N iterations
            rho_i = get_data(f, "dens", i, domains)
            vrad_i = get_data(f, "vy", i, domains) 
            vphi_i = get_data(f, "vx", i, domains) 
            vthe_i = get_data(f, "vz", i, domains)    

            # Converting spherical velocities to Cartesian velocities
            vx_i, vy_i, vz_i = vel_sph_to_cart(vthe_i, vrad_i, vphi_i, THETA, PHI)

            # Centering rho, v
            rho_c_i = centering(rho_i)
            vx_c_i = centering(vx_i)
            vy_c_i = centering(vy_i)
            vz_c_i = centering(vz_i)

            # Calculating mass and angular momentum
            mass_i = calc_mass(rho_i, cell_volume)
            Lx_i, Ly_i, Lz_i = calc_angular_momentum(mass_i, X, Y, ZCYL, vx_i, vy_i, vz_i)

            # Isolating the warped/broken disk
            warp_thresh = -17   # log of density threshold for which we can see the warp in the primary
            warp_buffer = 500   # Isolates a box of 2 * warp_buffer around the star (AU)
            rho_c_warp_i, vx_c_warp_i, vy_c_warp_i, vz_c_warp_i, Lx_c_warp_i, Ly_c_warp_i, Lz_c_warp_i, _ = isolate_disk(X_c, Y_c, Z_c, Px * au, Py * au, Pz * au, warp_buffer * au, rho_c_i, vx_c_i, vy_c_i, vz_c_i, Lx_i, Ly_i, Lz_i, warp_thresh) 

            # Calculating inclination, twist in the disk and saving the radial averages
            Lx_warp_avg_i, Ly_warp_avg_i, Lz_warp_avg_i = calc_L_average(Lx_c_warp_i, Ly_c_warp_i, Lz_c_warp_i, mass_i)
            inc_i, twist_i = calc_inc_twist(Lx_warp_avg_i, Ly_warp_avg_i, Lz_warp_avg_i, domains["r"], savefig=False, plot=False)
            twist_allit.append(twist_i)

        twist_allit = np.asarray(twist_allit)
        disk_twist_folder[f_sim_name] = twist_allit
        allit_years = calc_simtime(np.asarray(range(0, it+1, N)))       # Convert iterations to kyrs

    
    ############################################ Time evolution plots ############################################

    # Plotting twist_final vs R
    fig, ax = plt.subplots(figsize=(11, 7))
    current_color_index = -1
    last_base = None
    ls = "-"
    r_plot = np.log10(domains["r"]/au)[:-1]

    for key, value in disk_twist_folder.items():
        
        current_color_index = (current_color_index + 1) % len(colours)
        colour = colours[current_color_index]

        twist_final = value[-1, :]
        cutoff = find_twist_cutoff(twist_final)

        ax.plot(r_plot[:cutoff], twist_final[:cutoff], linestyle=ls, color=colour, label=folders_labels[key])   # -1 corresponds to last iteration

    ax.set_xlabel(r"$\log(r)$ [AU]")
    ax.set_ylabel(r"$\gamma$ $(\degree)$")
    ax.axvline(x=2, color='black', linestyle=':', linewidth=4.5)
    # ax.set_title(fr"Twist vs logr (53 kyr) $(\mathrm{{\rho \geq 10^{warp_thresh}}})$")
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=False) 
    plt.tight_layout()
    plt.savefig(f'paper_plots/param_study_twist_final_iter_vs_r_warp{warp_thresh}.pdf', dpi=300, bbox_inches="tight")
    plt.show()

    
if __name__ == "__main__":
    main()