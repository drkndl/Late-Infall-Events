# Calculate efficiency of cloud accretion onto disk during late infall 

import numpy as np
from pathlib import Path
from read import get_domain_spherical, get_data, load_par_file, get_param_value
import matplotlib.pyplot as plt
from analysis import calc_simtime, sph_to_cart, vel_sph_to_cart, centering, calc_cell_volume, calc_mass, scale_height, calc_angular_momentum, isolate_disk
from accretion import calc_accretion
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
plt.rcParams['axes.labelsize'] = 22     # x/y label size
plt.rcParams['xtick.labelsize'] = 20     # x-tick label size
plt.rcParams['ytick.labelsize'] = 20     # y-tick label size
plt.rcParams['legend.fontsize'] = 22     # legend font size


def stellar_accretion_mass(Mdot, t_arr):
    """
    Calculates the total mass lost to accretion onto the central star at each timestep. Essentially a numerical integration of Mdot using the Trapezoidal rule.

    Inputs:
    ------
    Mdot:             Stellar accretion rate in Msun/yr units
    t_arr:            Timestep array in kyr

    Outputs:
    -------
    M_stellar_acc:    Mass lost to stellar accretion at each timestep in g
    """

    # Converting Mdot from Msun/yr to g/s
    Mdot = Mdot * Msun / 3.154e7 

    # Converting time from kyr to s
    t_arr = t_arr * 3.154e7 * 1e3

    M_stellar_acc = np.zeros_like(Mdot)
    for i in range(1, len(t_arr)):
        M_stellar_acc[i] = M_stellar_acc[i-1] + 0.5 * (Mdot[i] + Mdot[i-1]) * (t_arr[i] - t_arr[i-1])

    return M_stellar_acc


def main():

    folders = [Path("F:/cloud_disk_it450_b01_rotX45/"), Path("F:/cloud_disk_it450_retro_rotX45/")]                        # Folder with the FARGO output files
    # folder = Path("../fargo3d/outputs/cloud_disk_it450_rotX45/")      # Folder with the FARGO output files (Binac2)
    fig_imgs = Path("paper_plots/")                    # Folder to save images    
    iter_total = 450                                     # FARGO snapshot

    first_it = 0

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)

    for dataset_idx, folder in enumerate(folders):

        ax = axes[dataset_idx]
        sim_name = folder.name                                     # Simulation name (for plot labels)
        dt_years = calc_simtime(np.asarray(range(first_it, iter_total+1, 10)))     # Convert iterations to kyrs

        disk_mass_allit = []                             # List to save the mass of the disk at each iteration
        rho_allit = []                                   # List to save density of all cells at each iteration
        vrad_allit = []                                  # List to save radial velocities of all cells at each iteration 

        
        ################################# Load coordinates  ################################

        domains = get_domain_spherical(folder)
        THETA, R, PHI = np.meshgrid(domains["theta"], domains["r"], domains["phi"], indexing="ij")
        rc = 0.5 * (domains["r"][1:] + domains["r"][:-1])
        X, Y, ZCYL, RCYL = sph_to_cart(THETA, R, PHI)       # Meshgrid of Cartesian coordinates

        # Loading all required basic set up parameters
        h0 = get_param_value("AspectRatio", sim_name)      # Aspect ratio
        f = get_param_value("FlaringIndex", sim_name)      # Flaring index
        p = get_param_value('SigmaSlope', sim_name)        # Power law slope of surface densities
        sigma0 = get_param_value('Sigma0', sim_name)       # Midplane surface density at R0 [g/cm^3]
        alpha = get_param_value('Alpha', sim_name)         # Alpha viscosity value
        Rin = get_param_value('Ymin', sim_name)            # Disk inner radius [cm]
        Rout = get_param_value('Rout', sim_name)           # Disk outer radius [cm]
        Nr = get_param_value('Ny', sim_name)               # Radial resolution (in FARGO3D, x=theta, y=r, z=phi)
        R0 = 5.2 * au                                      # As defined in FARGO3D [cm]

        # Calculating cloudlet mass
        cloud_mass_ini = get_param_value('CloudletMass', sim_name)
        print(f"Cloudlet mass: {cloud_mass_ini/Msun:.4f} Msun, {cloud_mass_ini:.2e} g")

        for it in range(first_it, iter_total+1, 10):
        

            ###################### Load data for each iteration #############################


            rho = get_data(folder, "dens", it, domains)         # Load 3D array of density values            
            vphi = get_data(folder, "vx", it, domains)          # Load 3D array of azimuthal velocities v_phi
            vrad = get_data(folder, "vy", it, domains)          # Load 3D array of radial velocities v_rad
            vthe = get_data(folder, "vz", it, domains)          # Load 3D array of colatitude velocities v_theta

            rho_allit.append(rho)
            vrad_allit.append(vrad)

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
            disk_mass_it = np.sum(disk_mass_radial_it)
            print((it))
            print(f"Disk mass: {disk_mass_it/Msun:.4f} Msun, {disk_mass_it:.2e} g")
            print(f"Cloud mass: {cloud_mass_ini/Msun:.4f} Msun, {cloud_mass_ini:.2e} g")
            print(f"Rout 30 AU Mc/Md ratio: {cloud_mass_ini/disk_mass_it:.4f}")
            disk_mass_allit.append(disk_mass_it)


        ####################### Calculating accretion onto the star ###############################

        vrad_allit = np.asarray(vrad_allit)
        rho_allit = np.asarray(rho_allit)

        dotM_tot_allit = []
        dotM_in_allit = []
        dotM_out_allit = []

        Hc = scale_height(domains["r"][0], h0, R0, f)
        zmax = 4 * Hc
        r0 = domains["r"][0]     # Taking the innermost radius to check accretion onto star

        for i in range(len(dt_years)):
            dotM_tot_i, dotM_in_i, dotM_out_i = calc_accretion(rho_allit[i], vrad_allit[i], domains["theta"], r0, 0, domains["phi"], zmax, Msun)
            dotM_tot_allit.append(dotM_tot_i)
            dotM_in_allit.append(dotM_in_i)
            dotM_out_allit.append(dotM_out_i)

        dotM_tot_allit = np.asarray(dotM_tot_allit)
        dotM_in_allit = np.asarray(dotM_in_allit)
        dotM_out_allit = np.asarray(dotM_out_allit)


        ############################## Calculate net Mc accreted, net cloudlet accretion efficiency ###############################


        disk_mass_initial = disk_mass_allit[0]
        cloud_mass_accreted = disk_mass_allit - disk_mass_initial
        cloud_acc_eff = cloud_mass_accreted / disk_mass_initial * 100
        print(cloud_mass_ini/Msun)

        ############################## Calculate absolute Mc accreted, abs cloud accretion efficiency ##############################

        M_stellar_acc = stellar_accretion_mass(np.abs(dotM_tot_allit), dt_years)
        abs_cloud_mass_acc = cloud_mass_accreted + M_stellar_acc
        abs_cloud_acc_eff = abs_cloud_mass_acc / cloud_mass_ini * 100

        ax.plot(dt_years, cloud_mass_accreted/Msun, color="red", label=r"$M_\mathrm{d,net}$")
        ax.plot(dt_years, abs_cloud_mass_acc/Msun, color="blue", label=r"$M_\mathrm{d,abs}$")
        ax.axvline(8, ls="--", lw=2, color="black")
        ax.set_xlabel(r"Time t [kyr]")
        if dataset_idx == 0:
            ax.set_ylabel(r"$\log(M)$ [$M_\odot$]")
            ax.legend()

        label_lines = [r"$b/b_\mathrm{crit} = 0.1$"]
        if "retro" in sim_name:
            label_text = "Retrograde"
        else:
            label_text = r"$b/b_\mathrm{crit} = 0.1$"

        ax.text(
            0.05, 0.95, label_text,
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=22,
            bbox=dict(boxstyle="square,pad=0.3", facecolor="white",
                      edgecolor="black", alpha=0.8),
        )

    fig.tight_layout()
    plt.savefig(f'{fig_imgs}/cloud_mass_accreted2.pdf', dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()