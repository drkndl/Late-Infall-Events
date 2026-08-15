import numpy as np
from pathlib import Path
from read import get_domain_spherical, get_data
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from analysis import calc_cell_volume, calc_mass, sph_to_cart, vel_sph_to_cart, centering, calc_angular_momentum, isolate_disk, calc_L_average, calc_inc_twist
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

colors = ['#0072B2', '#EE7733']  

def main():

    # Simulation data locally
    folders = [Path("F:/cloud_disk_it450_rotX45"), Path("F:/cloud_disk_it450_Rout30_correct_rotX45"), Path("F:/cloud_disk_it450_cmass10_rotX45"), Path("F:/cloud_disk_it450_Rout30_cmass10_correct_rotX45")] 

    # Central coordinates of the primary
    Px, Py, Pz = 0, 0, 0                                # Primary is in the centre of the simulation
    fig, ax = plt.subplots(2, 1, figsize=(9, 10), sharex=True)

    for idx, f in enumerate(folders):
        
        # Load simulation 
        f_sim_name = f.name                       # Simulation name (for plot labels)
        domains = get_domain_spherical(f)                       # Load coordinates  
        i = 450                                                # Final iteration (t=53kyr)    

        # Load simulation domains and create spherical and Cartesian meshgrids
        THETA, R, PHI = np.meshgrid(domains["theta"], domains["r"], domains["phi"], indexing="ij")
        X, Y, ZCYL, RCYL = sph_to_cart(THETA, R, PHI)                               # Meshgrid of Cartesian coordinates 
        RCYL_c = centering(RCYL)
        ZCYL_c = centering(ZCYL) 
        cell_volume = calc_cell_volume(domains["theta"], domains["r"], domains["phi"])   # Cell volumes 
        X_c = centering(X)
        Y_c = centering(Y)
        Z_c = centering(ZCYL)

        # Save density, mass, vrad, average inclination, average twist at multiple iterations 
        rho_allit = []
        vrad_allit = []
        mass_allit = []
        inc_allit =[]

        ######################## Calculating mass, inc, twist values ####################################     

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
        warp_thresh = -17   # log of density threshold for which we can see the warp in the primary
        warp_buffer = 500   # Isolates a box of 2 * warp_buffer around the star (AU)
        rho_c_warp_i, vx_c_warp_i, vy_c_warp_i, vz_c_warp_i, Lx_c_warp_i, Ly_c_warp_i, Lz_c_warp_i, _ = isolate_disk(X_c, Y_c, Z_c, Px * au, Py * au, Pz * au, warp_buffer * au, rho_c_i, vx_c_i, vy_c_i, vz_c_i, Lx_i, Ly_i, Lz_i, warp_thresh) 

        # Calculating inclination, twist in the disk and saving the radial averages
        Lx_warp_avg_i, Ly_warp_avg_i, Lz_warp_avg_i = calc_L_average(Lx_c_warp_i, Ly_c_warp_i, Lz_c_warp_i, mass_i)
        inc_i, twist_i = calc_inc_twist(Lx_warp_avg_i, Ly_warp_avg_i, Lz_warp_avg_i, domains["r"], savefig=False, plot=False)
        inc_allit.append(inc_i) 

        inc_allit = np.asarray(inc_allit)[0]
        
        subplot_idx = idx // 2
        linestyle = '-' if idx % 2 == 0 else '--'
        ax[subplot_idx].plot(np.log10(domains["r"]/au)[:-1], inc_allit, color=colors[subplot_idx], linestyle=linestyle, label=f_sim_name)
        ax[subplot_idx].set_ylim(0, 50)

    first_label = [r"$R_\mathrm{disk}=100\,\mathrm{au}$", r"$R_\mathrm{disk}=30\,\mathrm{au}$"]
    first_handles = [Line2D([0], [0], color='black', linestyle='-'),
                    Line2D([0], [0], color='black', linestyle='--')]
    leg1 = ax[1].legend(first_handles, first_label, loc="lower left")

    # Legend 2: color meaning, shown on second subplot
    second_label = [r"$M_\mathrm{c}/M_\mathrm{d}=0.45$", r"$M_\mathrm{c}/M_\mathrm{d}=4.5$"]
    second_handles = [Line2D([0], [0], color=colors[0], linestyle='-'),
                    Line2D([0], [0], color=colors[1], linestyle='-')]
    ax[1].legend(second_handles, second_label, loc="upper left")
    ax[1].add_artist(leg1)

    ax[0].set_ylabel(r"$\beta (\degree)$")  # whatever your y-label is
    ax[1].set_xlabel(r"$\log(r)$ [AU]")
    ax[1].set_ylabel(r"$\beta (\degree)$")

    plt.tight_layout()
    plt.savefig(f'paper_plots/Rdisk_inc_effect.pdf', dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()