import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib import cm 
import matplotlib.colors as mcolors
from pathlib import Path
import colormaps as cmaps
from read import get_domain_spherical, get_data, load_par_file, get_param_value
from analysis import calc_cell_volume, calc_mass, sph_to_cart, calc_simtime
from check_mass import surf_dens_profile
from no_thoughts_just_plots import load_sciviscolor_colormaps
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import matplotlib.colors as colors
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
plt.rcParams['axes.labelsize'] = 14     # x/y label size
plt.rcParams['xtick.labelsize'] = 12     # x-tick label size
plt.rcParams['ytick.labelsize'] = 12     # y-tick label size
plt.rcParams['legend.fontsize'] = 11     # legend font size

# cmap1 = load_sciviscolor_colormaps("colourmaps/discrete-2-5-discrete-gr-ye-rd-dark.xml")
# cmap1 = list(cmap1.values())[0]
cmap1 = cmaps.agsunset


def scale_height(r, h0, R0, f):
    """
    Calculates the scale height of the disk at a given radius r0

    Inputs:
    ------
    r:            Radius at which pressure scale height is calcualted [cm] (float)
    h0:           Aspect ratio (float)
    R0:           Radius at which aspect ratio is defined (FARGO3D standard) [cm] (float)
    f:            Disk flaring index (float)

    Outputs:
    -------
    Hc:           Scale height at given radius r0 [cm]
    """

    Hc = h0 * r * np.power(r / R0, f)             
    return Hc 


def omega_kepler(Mstar, r):
    """
    Calculates the Keplerian velocities for an array of radii
    
    Inputs:
    ------
    Mstar:   Mass of the star [g]
    r:       1D array of radii [cm]

    Outputs:
    -------
    omega_k: 1D array of Keplerian velocities [/s]
    """

    omega_k = np.sqrt(G * Mstar / r**3)
    return omega_k


def calc_accretion(rho, vr, theta, r0, r0_id, phi, max_height, Msun, min_height=None):
    """
    Function to check if cloudlet mass is accreted onto the star

    Inputs:
    ------
    rho:          3D array of densities [g/cm^3] (shape: ntheta, nr, nphi)
    vr:           3D array of radial velocities [cm/s] (shape: ntheta, nr, nphi)
    theta:        1D array of polar angles [radians] (shape: ntheta)
    r0:           Radius at which accretion is calculated [cm] (float)
    r0_id:        Array index of r0 corresponding to domains["r"] (and therefore all physical quantities like rho, vrad)
    phi:          1D array of azimuthal angles (shape: nphi)
    max_height:   Maximum height within which we calculate accretion [cm]
    Msun:         Mass of the Sun [g]

    Outputs:
    -------
    dotM_total:       Total mass flux in and out of shell [g/s]
    dotM_in:          Inward accretion [g/s]
    dotM_out:         Outward flux [g/s]
    """

    z = r0 * np.cos(theta)                          # Disk heights at given radius

    # Boolean mask selecting only polar angles within max_height so that we ignore cloudlet
    if min_height == None:
        theta_mask = np.abs(z) <= max_height
        theta_sel = theta[theta_mask]
        print(np.round(np.degrees(theta_sel), 1))

    elif min_height != None:
        theta_mask = (np.abs(z) >= min_height) & (np.abs(z) <= max_height)
        theta_sel = theta[theta_mask]   
        print(np.round(np.degrees(theta_sel), 1))
    
    dtheta_sel = np.gradient(theta_sel)
    dphi = np.gradient(phi)
    rho0_thetamask = rho[:, r0_id, :][theta_mask, :]              # shape (ntheta_mask, nphi)
    vr0_thetamask  = vr[:, r0_id, :][theta_mask, :]               # shape (ntheta_mask, nphi)

    theta2d, phi2d = np.meshgrid(theta_sel, phi, indexing='ij')
    dtheta2d, dphi2d = np.meshgrid(dtheta_sel, dphi, indexing='ij')
    dA = r0**2 * np.sin(theta2d) * dtheta2d * dphi2d

    mass_flux = rho0_thetamask * vr0_thetamask * dA        # (g/s) 

    dotM_total = np.sum(mass_flux)
    dotM_out = np.sum(mass_flux[vr0_thetamask > 0])         # Assuming vr > 0 is outward
    dotM_in  = np.sum(mass_flux[vr0_thetamask < 0])         # Assuming vr < 0 is inward

    # Converting it from g/s -> Msun/kyr -> Msun/yr
    dotM_total = dotM_total / Msun * stoky / 1e3       
    dotM_in = dotM_in / Msun * stoky / 1e3
    dotM_out = dotM_out / Msun * stoky / 1e3

    return dotM_total, dotM_in, dotM_out


def calc_accretion_theoretical(sigma, H, ok, alpha):
    """
    Function to calculate the theoretical mass accretion values Mdot = 3 pi sigma nu
    
    Inputs:
    -------
    """

    cs = H * ok                                     # Isothermal sound speed [cm/s]
    nu = alpha * cs**2 / ok                         # Viscosity (?) [cgs]
    Mdot_theo = 3 * np.pi * sigma * nu              # Theoretical mass accretion rate [g/s]
    Mdot_theo = Mdot_theo / Msun * stoky / 1e3      # Theoretical mass accretion rate [Msun/yr]
    return Mdot_theo


def main():


    folder = Path("../cloud_disk_it450_cmass10_Rout30_rotX45/")                    # Folder with the output files
    # folder = Path("../fargo3d/outputs/cloud_disk_it450_cmass10_Rout30_rotX45")       # Folder with the output files (BinAC2)
    fig_imgs = Path("cloud_disk_it450_cmass10_Rout30_rotX45/imgs/")                  # Folder to save images
    it = 450                                                             # FARGO snapshot of interest
    sim_name = str(fig_imgs).split('/')[0]                               # Simulation name (for plot labels)

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
    

    ############# Load data for single snapshot (theta = 175, r = 150, phi = 100) ######################


    domains = get_domain_spherical(folder)
    rho = get_data(folder, "dens", it, domains)         # Load 3D array of density values   
    vrad = get_data(folder, "vy", it, domains)          # Load 3D array of radial velocities v_rad
    cell_volume = calc_cell_volume(domains["theta"], domains["r"], domains["phi"])
    mass = calc_mass(rho, cell_volume)

    Hc = scale_height(domains["r"][0], h0, R0, f)
    zmax = 4 * Hc
    r0 = domains["r"][0]     # Taking the innermost radius to check accretion onto star
    dotM_tot, dotM_in, dotM_out = calc_accretion(rho, vrad, domains["theta"], r0, 0, domains["phi"], zmax, Msun)
    print(f"Total flux across inner shell: {dotM_tot:.3e} Msun/yr")
    print(f"Outflow: {dotM_out:.3e} Msun/yr, Inflow: {dotM_in:.3e} Msun/yr")


    ############################## Load data for multiple snapshots ####################################


    # Load density and radial velocity values at multiple iterations 
    rho_allit = []
    vrad_allit = [] 

    for i in range(0, it+1, 10):     # loading density and vrad every 10 iterations
        rho_i = get_data(folder, "dens", i, domains)
        vrad_i = get_data(folder, "vy", i, domains)          
        rho_allit.append(rho_i)
        vrad_allit.append(vrad_i)
    
    vrad_allit = np.asarray(vrad_allit)
    rho_allit = np.asarray(rho_allit)
    allit_years = calc_simtime(np.asarray(range(0, it+1, 10)))       # Convert iterations to kyrs


    #################################### Accretion onto star ##########################################


    dotM_tot_allit = []
    dotM_in_allit = []
    dotM_out_allit = []
    Hc = scale_height(domains["r"][0], h0, R0, f)
    zmax = 4 * Hc
    for i in range(len(allit_years)):
        dotM_tot_i, dotM_in_i, dotM_out_i = calc_accretion(rho_allit[i], vrad_allit[i], domains["theta"], r0, 0, domains["phi"], zmax, Msun)
        dotM_tot_allit.append(dotM_tot_i)
        dotM_in_allit.append(dotM_in_i)
        dotM_out_allit.append(dotM_out_i)

    dotM_tot_allit = np.asarray(dotM_tot_allit)
    dotM_in_allit = np.asarray(dotM_in_allit)
    dotM_out_allit = np.asarray(dotM_out_allit)

    # Plotting the logarithmic mass fluxes 
    fig, ax = plt.subplots()
    # plt.plot(allit_years, np.abs(dotM_tot_allit), label="Total flux")
    plt.plot(allit_years, np.log10(-dotM_in_allit), label="Log Inward flux")
    plt.plot(allit_years, dotM_out_allit, label="Outward flux")
    ax.set_xlabel(r"Time [kyr]")
    ax.set_ylabel(r"$\mathrm{\log\dot{M}}$ [$M_{sun}$/yr]")
    ax.set_title(fr"{sim_name}: $\mathrm{{\log\dot{{M}}}}$ vs t (R = 10 AU)")
    plt.legend(loc="lower right")
    plt.savefig(f'{fig_imgs}/logMdot_vs_t_it{it}.png')
    plt.show()


    ########################### Check accretion for different max heights #############################


    # Defining max heights and the corresponding plot labels
    Hc = scale_height(domains["r"][0], h0, R0, f)
    zmax_array = [Hc, 2 * Hc, 4 * Hc, 5 * Hc, 10 * Hc, 20 * Hc, 30 * Hc]
    zmax_labels = {}

    for zmax in zmax_array:
        z = r0 * np.cos(domains["theta"])               # Disk heights at given radius
        theta_mask = np.abs(z) <= zmax            # Boolean mask selecting only polar angles within max_height so that we ignore cloudlet
        theta_sel = domains["theta"][theta_mask]
        theta_sel_min, theta_sel_max = np.min(np.round(np.degrees(theta_sel), 1)), np.max(np.round(np.degrees(theta_sel), 1))
        # zmax_labels[zmax] = f"{int(zmax/Hc)}Hc ({theta_sel_min}$\degree$ - {theta_sel_max}$\degree$)"
        zmax_labels[zmax] = f"{int(zmax/Hc)}Hc"

    Mdot_in_allzmax = {}
    Mdot_out_allzmax = {}
    
    for z in zmax_array:

        print(z)
        dotM_in_allit = []
        dotM_out_allit = []
        for i in range(len(allit_years)):

            _, dotM_in_z, dotM_out_z = calc_accretion(rho_allit[i], vrad_allit[i], domains["theta"], r0, 0, domains["phi"], z, Msun)
            dotM_in_allit.append(dotM_in_z)
            dotM_out_allit.append(dotM_out_z)

        dotM_in_allit = np.asarray(dotM_in_allit)
        dotM_out_allit = np.asarray(dotM_out_allit)

        # Adding time evolution of accretion rates to corresponding max height value in the dictionary
        Mdot_in_allzmax[z] = dotM_in_allit
        Mdot_out_allzmax[z] = dotM_out_allit

    # Plotting the inward mass fluxes for all max heights
    fig, ax = plt.subplots()
    i=0
    for key, value in Mdot_in_allzmax.items():
        ax.plot(allit_years, np.log10(-value), label=f'{zmax_labels[key]}', color=cmap1(i / (len(Mdot_in_allzmax)-1)))
        i+=1
    ax.set_xlabel(r"Time [kyr]")
    ax.set_ylabel(r"$\mathrm{\log\dot{M}}$ [$M_{sun}$/yr]")
    ax.set_title(fr"{sim_name}: Inward $\mathrm{{\log\dot{{M}}}}$ vs t (R = 10 AU)")
    fig.tight_layout()
    ax.legend(loc="upper right")   # loc='upper left', 
    plt.savefig(f'{fig_imgs}/logMdot_vs_t_all_zmax.png')
    plt.show()

    # Plotting the outward mass fluxes for all max heights as a sanity check (SHOULD BE ZERO!)
    # fig, ax = plt.subplots()
    # for key, value in Mdot_in_allzmax.items():
    #     ax.plot(allit_years, value, label=zmax_labels[key])
    # ax.set_xlabel(r"Time [kyr]")
    # ax.set_ylabel(r"$\mathrm{\log\dot{M}}$ [$M_{sun}$/yr]")
    # ax.set_title(fr"{sim_name}: Outward $\mathrm{{\log\dot{{M}}}}$ vs t (R = 10 AU)")
    # fig.tight_layout()
    # ax.legend(loc="lower right")   # loc='upper left', 
    # plt.savefig(f'{fig_imgs}/logMoutdot_vs_t_all_zmax.png')
    plt.show()


    ################################# Compare theoretical and actual mass accretion ###################################


    r_theo = np.logspace(np.log10(Rin / au), np.log10(Rout / au), Nr) * au       # Radius array
    omega_k = omega_kepler(Mstar, r_theo)                                        # Keplerian velocities
    Hc_arr = scale_height(r_theo, h0, R0, f)                                     # Pressure scale heights 
    sigma = surf_dens_profile(sigma0, p, R0, r_theo, Rout, plot=False)           # Surface densities
    Mdot_theo = calc_accretion_theoretical(sigma, Hc_arr, omega_k, alpha)        # Theoretical mass accretion rate

    # plt.plot(np.log10(r_theo/au), Mdot_theo)
    # plt.xlabel("logR")
    # plt.ylabel(r"$\mathrm{\log\dot{M}}$ [$M_{sun}$/yr]")
    # plt.title("Theoretical Mass Accretion Rate")
    # plt.savefig(f"{fig_imgs}/Mdot_theoretical.png")
    # plt.show()


    ########################### Check accretion at different radii for different max heights #############################


    # Defining max heights and the corresponding plot labels
    zmax_array = [1, 2, 4, 5, 10, 20]
    zmax_labels = {}

    # Defining radii at which I want to plot accretion at different max heights
    idx_20 = np.argmin(np.abs(domains["r"]/au - 20))    # Index of 20 AU
    idx_50 = np.argmin(np.abs(domains["r"]/au - 50))    # Index of 50 AU
    idx_100  =np.argmin(np.abs(domains["r"]/au - 100))  # Index of 100 AU
    r20 = domains["r"][idx_20]
    r50 = domains["r"][idx_50]
    r100 = domains["r"][idx_100]
    r_for_acc = {r0: 0, r20: idx_20, r50: idx_50, r100: idx_100}

    # Defining subplots to plot inward accretion
    # fig, axes = plt.subplots(2, 2, figsize=(8,6), sharex=True, sharey=True)
    # axes = axes.flatten()
    # plot_counter=0
    
    # for r_acc, r_acc_id in r_for_acc.items():

    #     Mdot_in_allzmax = {}
    #     Hc = scale_height(r_acc, h0, R0, f)

    #     for z_scale in zmax_array:

    #         print(z_scale)
    #         zmax = z_scale * Hc         # Defining max_height for scale height at given radius

    #         # Defining zmax_labels
    #         z = r_acc * np.cos(domains["theta"])      # Disk heights at given radius
    #         theta_mask = np.abs(z) <= zmax            # Boolean mask selecting only polar angles within max_height so that we ignore cloudlet
    #         theta_sel = domains["theta"][theta_mask]
    #         theta_sel_min, theta_sel_max = np.min(np.round(np.degrees(theta_sel), 1)), np.max(np.round(np.degrees(theta_sel), 1))
    #         # zmax_labels[zmax] = fr"{int(zmax/Hc)}Hc ({theta_sel_min}$\degree$-{theta_sel_max}$\degree$)"
    #         zmax_labels[zmax] = fr"{int(zmax/Hc)}Hc"

    #         dotM_in_allit = []          # Inward accretion rates for given radius and given scale height at every 10 iters
    #         for i in range(len(allit_years)):

    #             _, dotM_in_z, _ = calc_accretion(rho_allit[i], vrad_allit[i], domains["theta"], r_acc, r_acc_id, domains["phi"], zmax, Msun)
    #             dotM_in_allit.append(dotM_in_z)

    #         dotM_in_allit = np.asarray(dotM_in_allit)

    #         # Adding time evolution of accretion rates to corresponding max height value in the dictionary
    #         Mdot_in_allzmax[zmax] = dotM_in_allit

    #     # Plotting the inward mass fluxes for all max heights at given radius
    #     i=0
    #     for key, value in Mdot_in_allzmax.items():
    #         axes[plot_counter].plot(allit_years, np.log10(-value), label=f'{zmax_labels[key]}', color=cmap1(i / (len(Mdot_in_allzmax)-1)))
    #         i+=1 
    #     # axes[plot_counter].set_xlabel(r"Time [kyr]")
    #     # axes[plot_counter].set_ylabel(r"$\mathrm{\log\dot{M}}$ [$M_{sun}$/yr]")
    #     axes[plot_counter].set_title(fr"R = {int(r_acc/au)} AU")
    #     handles, labels = axes[plot_counter].get_legend_handles_labels()
    #     plot_counter += 1
       
    # fig.legend(handles, labels, loc="upper center", ncol=6, frameon=False)
    # fig.supxlabel(r"Time [kyr]")  
    # fig.supylabel(r"$\mathrm{\log\dot{M}}$ [$M_{sun}$/yr]")  
    # fig.suptitle(f"{sim_name}: Inward flux", fontsize=10, y=0.95)  
    # fig.tight_layout()
    # plt.savefig(f'{fig_imgs}/logMdot_vs_t_all_radii_all_zmax.png')
    # plt.show()

    # # Defining subplots to plot outward accretion
    # fig, axes = plt.subplots(2, 2, figsize=(8,6), sharex=True, sharey=True)
    # axes = axes.flatten()
    # plot_counter=0
    
    # for r_acc, r_acc_id in r_for_acc.items():

    #     Mdot_out_allzmax = {}
    #     Hc = scale_height(r_acc, h0, R0, f)
    #     print(r_acc/au, Hc/au)

    #     for z_scale in zmax_array:

    #         print(z_scale)
    #         zmax = z_scale * Hc         # Defining max_height for scale height at given radius

    #         # Defining zmax_labels
    #         z = r_acc * np.cos(domains["theta"])      # Disk heights at given radius
    #         theta_mask = np.abs(z) <= zmax            # Boolean mask selecting only polar angles within max_height so that we ignore cloudlet
    #         theta_sel = domains["theta"][theta_mask]
    #         theta_sel_min, theta_sel_max = np.min(np.round(np.degrees(theta_sel), 1)), np.max(np.round(np.degrees(theta_sel), 1))
    #         # zmax_labels[zmax] = fr"{int(zmax/Hc)}Hc ({theta_sel_min}$\degree$-{theta_sel_max}$\degree$)"
    #         zmax_labels[zmax] = fr"{int(zmax/Hc)}Hc"

    #         dotM_out_allit = []          # Outward accretion rates for given radius and given scale height at every 10 iters
    #         for i in range(len(allit_years)):

    #             _, _, dotM_out_z = calc_accretion(rho_allit[i], vrad_allit[i], domains["theta"], r_acc, r_acc_id, domains["phi"], zmax, Msun)
    #             dotM_out_allit.append(dotM_out_z)

    #         dotM_out_allit = np.asarray(dotM_out_allit)

    #         # Adding time evolution of accretion rates to corresponding max height value in the dictionary
    #         Mdot_out_allzmax[zmax] = dotM_out_allit

    #     # Plotting the outward mass fluxes for all max heights at given radius
    #     i=0
    #     for key, value in Mdot_out_allzmax.items():
    #         if int(r_acc/au) == 10:
    #             axes[plot_counter].plot(allit_years, value, label=f'{zmax_labels[key]}', color=cmap1(i / (len(Mdot_out_allzmax)-1)))
    #             i+=1
    #         else:
    #             axes[plot_counter].plot(allit_years, np.log10(value), label=f'{zmax_labels[key]}', color=cmap1(i / (len(Mdot_out_allzmax)-1)))
    #             i+=1
    #     # axes[plot_counter].set_xlabel(r"Time [kyr]")
    #     # axes[plot_counter].set_ylabel(r"$\mathrm{\log\dot{M}}$ [$M_{sun}$/yr]")
    #     axes[plot_counter].set_title(fr"R = {int(r_acc/au)} AU")
    #     handles, labels = axes[plot_counter].get_legend_handles_labels()
    #     plot_counter += 1
       
    # fig.legend(handles, labels, loc="upper center", ncol=6, frameon=False)
    # fig.supxlabel(r"Time [kyr]")  
    # fig.supylabel(r"$\mathrm{\log\dot{M}}$ [$M_{sun}$/yr]")
    # fig.suptitle(f"{sim_name}: Outward flux", fontsize=10, y=0.95)  
    # fig.tight_layout()
    # plt.savefig(f'{fig_imgs}/logMoutdot_vs_t_all_radii_all_zmax.png')
    # plt.show()


    # Defining subplots to plot net accretion
    fig, axes = plt.subplots(2, 2, figsize=(8,6), sharex=True, sharey=True)
    axes = axes.flatten()
    plot_counter=0
    
    for r_acc, r_acc_id in r_for_acc.items():

        Mdot_net_allzmax = {}
        Hc = scale_height(r_acc, h0, R0, f)
        print(r_acc/au, Hc/au)

        for z_scale in zmax_array:

            print(z_scale)
            zmax = z_scale * Hc         # Defining max_height for scale height at given radius

            # Defining zmax_labels
            z = r_acc * np.cos(domains["theta"])      # Disk heights at given radius
            theta_mask = np.abs(z) <= zmax            # Boolean mask selecting only polar angles within max_height so that we ignore cloudlet
            theta_sel = domains["theta"][theta_mask]
            theta_sel_min, theta_sel_max = np.min(np.round(np.degrees(theta_sel), 1)), np.max(np.round(np.degrees(theta_sel), 1))
            # zmax_labels[zmax] = fr"{int(zmax/Hc)}Hc ({theta_sel_min}$\degree$-{theta_sel_max}$\degree$)"
            zmax_labels[zmax] = fr"{int(zmax/Hc)}Hc"

            dotM_net_allit = []          # Outward accretion rates for given radius and given scale height at every 10 iters
            for i in range(len(allit_years)):

                dotM_net_z, _, _ = calc_accretion(rho_allit[i], vrad_allit[i], domains["theta"], r_acc, r_acc_id, domains["phi"], zmax, Msun)
                dotM_net_allit.append(dotM_net_z)

            dotM_net_allit = np.asarray(dotM_net_allit)

            # Adding time evolution of accretion rates to corresponding max height value in the dictionary
            Mdot_net_allzmax[zmax] = dotM_net_allit

        # Plotting the net mass fluxes for all max heights at given radius
        for i, (key, value) in enumerate(Mdot_net_allzmax.items()):

            # Plotting negative accretion rates as dotted lines and positive values as solid lines
            accr_pos = np.where(value > 0, value, np.nan)
            accr_neg = np.where(value < 0, value, np.nan)

            # Plotting positive accretion rates
            axes[plot_counter].plot(allit_years, np.log10(np.abs(accr_pos)), linestyle='-', color=cmap1(i / (len(Mdot_net_allzmax)-1)), label=zmax_labels[key])

            # Plotting negative accretion rates
            axes[plot_counter].plot(allit_years, np.log10(np.abs(accr_neg)), linestyle='--', lw=1.5, color=cmap1(i / (len(Mdot_net_allzmax)-1)))

        axes[plot_counter].set_title(fr"R = {int(r_acc/au)} AU")
        handles, labels = axes[plot_counter].get_legend_handles_labels()
        plot_counter += 1
       
    fig.legend(handles, labels, loc="upper center", ncol=6, frameon=False)
    fig.supxlabel(r"Time [kyr]")  
    fig.supylabel(r"$\mathrm{\log\dot{M}}$ [$M_{sun}$/yr]")
    fig.suptitle(f"{sim_name}: Net flux", fontsize=10, y=0.95)  
    fig.tight_layout()
    plt.savefig(f'{fig_imgs}/logMnetdot_vs_t_all_radii_all_zmax.png')
    plt.show()


    ######################### Is the accretion systematic at different heights? ###########################


    # Defining max heights and the corresponding plot labels
    zmin_array = [1, 2, 4, 5, 10]
    zmax_array = [2, 4, 5, 10, 20]
    zrange_labels = {}

    # Defining radii at which I want to plot accretion at different max heights
    idx_20 = np.argmin(np.abs(domains["r"]/au - 20))    # Index of 20 AU
    idx_50 = np.argmin(np.abs(domains["r"]/au - 50))    # Index of 50 AU
    idx_100  =np.argmin(np.abs(domains["r"]/au - 100))  # Index of 100 AU
    r20 = domains["r"][idx_20]
    r50 = domains["r"][idx_50]
    r100 = domains["r"][idx_100]
    r_for_acc = {r0: 0, r20: idx_20, r50: idx_50, r100: idx_100}

    # # Defining subplots to plot inward accretion
    # fig, axes = plt.subplots(2, 2, figsize=(8,6), sharex=True, sharey=True)
    # axes = axes.flatten()
    # plot_counter=0
    
    # for r_acc, r_acc_id in r_for_acc.items():

    #     Mdot_in_allzrange = {}
    #     Hc = scale_height(r_acc, h0, R0, f)

    #     for z_scale_i in range(len(zmax_array)):

    #         zmax = zmax_array[z_scale_i] * Hc         # Defining max_height for scale height at given radius
    #         zmin = zmin_array[z_scale_i] * Hc         # Defining min_height for scale height at given radius

    #         # Defining zrange_labels
    #         zrange_labels[zmin] = fr"{int(zmin/Hc)}Hc - {int(zmax/Hc)}Hc"

    #         dotM_in_allit = []          # Inward accretion rates for given radius and given scale height at every 10 iters
    #         for i in range(len(allit_years)):

    #             _, dotM_in_z, _ = calc_accretion(rho_allit[i], vrad_allit[i], domains["theta"], r_acc, r_acc_id, domains["phi"], zmax, Msun, zmin)
    #             dotM_in_allit.append(dotM_in_z)

    #         dotM_in_allit = np.asarray(dotM_in_allit)

    #         # Adding time evolution of accretion rates to corresponding max height value in the dictionary
    #         Mdot_in_allzrange[zmin] = dotM_in_allit

    #     # Plotting the inward mass fluxes for all max heights at given radius
    #     i=0
    #     for key, value in Mdot_in_allzrange.items():
    #         axes[plot_counter].plot(allit_years, np.log10(-value), label=f'{zrange_labels[key]}', color=cmap1(i / (len(Mdot_in_allzrange)-1)))
    #         i+=1
    #     # axes[plot_counter].set_xlabel(r"Time [kyr]")
    #     # axes[plot_counter].set_ylabel(r"$\mathrm{\log\dot{M}}$ [$M_{sun}$/yr]")
    #     axes[plot_counter].set_title(fr"R = {int(r_acc/au)} AU")
    #     handles, labels = axes[plot_counter].get_legend_handles_labels()
    #     plot_counter += 1
       
    # fig.legend(handles, labels, loc="upper center", ncol=6, frameon=False)
    # fig.suptitle(f"{sim_name}: Inward flux between diff heights", fontsize=10, y=0.95)  
    # fig.supxlabel(r"Time [kyr]")  
    # fig.supylabel(r"$\mathrm{\log\dot{M}}$ [$M_{sun}$/yr]")
    # fig.tight_layout()
    # plt.savefig(f'{fig_imgs}/logMdot_vs_t_all_radii_all_zrange.png')
    # plt.show()

    # # Defining subplots to plot inward accretion
    # fig, axes = plt.subplots(2, 2, figsize=(8,6), sharex=True, sharey=True)
    # axes = axes.flatten()
    # plot_counter=0
    
    # for r_acc, r_acc_id in r_for_acc.items():

    #     Mdot_out_allzrange = {}
    #     Hc = scale_height(r_acc, h0, R0, f)
    #     print(r_acc/au, Hc/au)

    #     for z_scale_i in range(len(zmax_array)):

    #         zmax = zmax_array[z_scale_i] * Hc         # Defining max_height for scale height at given radius
    #         zmin = zmin_array[z_scale_i] * Hc         # Defining min_height for scale height at given radius

    #         # Defining zrange_labels
    #         zrange_labels[zmin] = fr"{int(zmin/Hc)}Hc - {int(zmax/Hc)}Hc"

    #         dotM_out_allit = []          # Outward accretion rates for given radius and given scale height at every 10 iters
    #         for i in range(len(allit_years)):

    #             _, _, dotM_out_z = calc_accretion(rho_allit[i], vrad_allit[i], domains["theta"], r_acc, r_acc_id, domains["phi"], zmax, Msun, zmin)
    #             dotM_out_allit.append(dotM_out_z)

    #         dotM_out_allit = np.asarray(dotM_out_allit)

    #         # Adding time evolution of accretion rates to corresponding max height value in the dictionary
    #         Mdot_out_allzrange[zmin] = dotM_out_allit

    #     # Plotting the outward mass fluxes for all max heights at given radius
    #     i=0
    #     for key, value in Mdot_out_allzrange.items():
    #         if int(r_acc/au) == 10:
    #             axes[plot_counter].plot(allit_years, value, label=f'{zrange_labels[key]}', color=cmap1(i / (len(Mdot_out_allzrange)-1)))
    #             i+=1
    #         else:
    #             axes[plot_counter].plot(allit_years, np.log10(value), label=f'{zrange_labels[key]}', color=cmap1(i / (len(Mdot_out_allzrange)-1)))
    #             i+=1
    #     # axes[plot_counter].set_xlabel(r"Time [kyr]")
    #     # axes[plot_counter].set_ylabel(r"$\mathrm{\log\dot{M}}$ [$M_{sun}$/yr]")
    #     axes[plot_counter].set_title(fr"R = {int(r_acc/au)} AU")
    #     handles, labels = axes[plot_counter].get_legend_handles_labels()
    #     plot_counter += 1
       
    # fig.legend(handles, labels, loc="upper center", ncol=6, frameon=False)
    # fig.supxlabel(r"Time [kyr]")  
    # fig.supylabel(r"$\mathrm{\log\dot{M}}$ [$M_{sun}$/yr]")
    # fig.suptitle(f"{sim_name}: Outward flux between diff heights", fontsize=10, y=0.95)  
    # fig.tight_layout()
    # plt.savefig(f'{fig_imgs}/logMoutdot_vs_t_all_radii_all_zrange.png')
    # plt.show()


    # Defining subplots to plot net accretion
    fig, axes = plt.subplots(2, 2, figsize=(8,6), sharex=True, sharey=True)
    axes = axes.flatten()
    plot_counter=0
    
    for r_acc, r_acc_id in r_for_acc.items():

        Mdot_net_allzrange = {}
        Hc = scale_height(r_acc, h0, R0, f)
        print(r_acc/au, Hc/au)

        for z_scale_i in range(len(zmax_array)):

            zmax = zmax_array[z_scale_i] * Hc         # Defining max_height for scale height at given radius
            zmin = zmin_array[z_scale_i] * Hc         # Defining min_height for scale height at given radius

            # Defining zrange_labels
            zrange_labels[zmin] = fr"{int(zmin/Hc)}Hc - {int(zmax/Hc)}Hc"

            dotM_net_allit = []          # Net accretion rates for given radius and given scale height at every 10 iters
            for i in range(len(allit_years)):

                dotM_net_z, _, _ = calc_accretion(rho_allit[i], vrad_allit[i], domains["theta"], r_acc, r_acc_id, domains["phi"], zmax, Msun, zmin)
                dotM_net_allit.append(dotM_net_z)

            dotM_net_allit = np.asarray(dotM_net_allit)

            # Adding time evolution of accretion rates to corresponding max height value in the dictionary
            Mdot_net_allzrange[zmin] = dotM_net_allit

        # Plotting the net mass fluxes for all max heights at given radius
        for i, (key, value) in enumerate(Mdot_net_allzrange.items()):

            # Plotting negative accretion rates as dotted lines and positive values as solid lines
            accr_pos = np.where(value > 0, value, np.nan)
            accr_neg = np.where(value < 0, value, np.nan)

            # Plotting positive accretion rates
            axes[plot_counter].plot(allit_years, np.log10(np.abs(accr_pos)), linestyle='-', color=cmap1(i / (len(Mdot_net_allzrange)-1)), label=zrange_labels[key])

            # Plotting negative accretion rates
            axes[plot_counter].plot(allit_years, np.log10(np.abs(accr_neg)), linestyle='--', lw=1.5, color=cmap1(i / (len(Mdot_net_allzrange)-1)))

        axes[plot_counter].set_title(fr"R = {int(r_acc/au)} AU")
        handles, labels = axes[plot_counter].get_legend_handles_labels()
        plot_counter += 1
       
    fig.legend(handles, labels, loc="upper center", ncol=6, frameon=False)
    fig.supxlabel(r"Time [kyr]")  
    fig.supylabel(r"$\mathrm{\log\dot{M}}$ [$M_{sun}$/yr]")
    fig.suptitle(f"{sim_name}: Net flux between diff heights", fontsize=10, y=0.95)  
    fig.tight_layout()
    plt.savefig(f'{fig_imgs}/logMnetdot_vs_t_all_radii_all_zrange.png')
    plt.show()


    ########################## 2D accretion across radii and iteration times #######################


    # Initializing arrays of shape (t, r)
    Mdot_in_2D = np.zeros([len(allit_years), len(domains["r"])])
    Mdot_out_2D = np.zeros([len(allit_years), len(domains["r"])])
    Mdot_net_2D = np.zeros([len(allit_years), len(domains["r"])])

    for i in range(len(allit_years)):
        for j in range(len(domains["r"])):

            # Calculating scale height and maximum height at which to calculate accretion at given radius
            Hc_j = scale_height(domains["r"][j], h0, R0, f)
            zmax_j = 4 * Hc_j

            # Calculating accretion
            dotM_net, dotM_in, dotM_out = calc_accretion(rho_allit[i], vrad_allit[i], domains["theta"], domains["r"][j], j, domains["phi"], zmax_j, Msun)
            Mdot_in_2D[i, j] = dotM_in
            Mdot_out_2D[i, j] = dotM_out
            Mdot_net_2D[i, j] = dotM_net

    # Plotting heat maps of the 2D accretion values
    # fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(5,7), sharex=True)

    # c1 = ax1.imshow(np.log10(-Mdot_in_2D), extent=[np.log10(domains["r"].min()/au), np.log10(domains["r"].max()/au), allit_years.min(), allit_years.max()], origin="lower", cmap=cmaps.matter, aspect='auto', vmin=-17.5, vmax=-5)
    # fig.colorbar(c1, ax=ax1, label=r"$\mathrm{\log\dot{M} [M_{sun}/yr]}$")
    # ax1.set_title("Inward flux")
    # ax1.set_ylabel("Time [kyr]")

    # c2 = ax2.imshow(np.log10(Mdot_out_2D), extent=[np.log10(domains["r"].min()/au), np.log10(domains["r"].max()/au), allit_years.min(), allit_years.max()], origin="lower", cmap=cmaps.matter, aspect='auto', vmin=-17.5, vmax=-5)
    # fig.colorbar(c2, ax=ax2, label=r"$\mathrm{\log\dot{M} [M_{sun}/yr]}$")
    # ax2.set_title("Outward flux")
    # ax2.set_ylabel("Time [kyr]")

    # c3 = ax3.imshow(np.sign(Mdot_net_2D) * np.log10(np.abs(Mdot_net_2D)), extent=[np.log10(domains["r"].min()/au), np.log10(domains["r"].max()/au), allit_years.min(), allit_years.max()], origin="lower", cmap=cmaps.BlueRed, aspect='auto', vmin=-17.5, vmax=17.5)
    # fig.colorbar(c3, ax=ax3, label=r"$\mathrm{\log\dot{M} [M_{sun}/yr]}$")
    # ax3.set_title("Net accretion")
    # ax3.set_xlabel("log(R) [AU]")
    # ax3.set_ylabel("Time [kyr]")
    
    # Add arrows to show direction of net accretion
    # Xtemp, Ytemp = np.meshgrid(np.log10(domains["r"]/au), allit_years)
    # U = np.sign(Mdot_net_2D)  # horizontal direction (positive = outward)
    # V = np.zeros_like(U)      # no vertical component
    # arrow_cmap = plt.cm.bwr   # blue–white–red
    # arrow_norm = colors.Normalize(vmin=-1, vmax=1)
    # step = (slice(None, None, 5), slice(None, None, 10))  # Downsampling to avoid clutter
    # q1 = ax3.quiver(Xtemp[step], Ytemp[step], U[step], V[step], U[step], cmap=arrow_cmap, norm=arrow_norm, scale=20)
    
    # fig.tight_layout()
    # plt.savefig(f'{fig_imgs}/logMdot_2D.png', bbox_inches="tight")
    # plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(5,4))
    c3 = ax.imshow(np.sign(Mdot_net_2D) * np.log10(np.abs(Mdot_net_2D)), extent=[np.log10(domains["r"].min()/au), np.log10(domains["r"].max()/au), allit_years.min(), allit_years.max()], origin="lower", cmap=cmaps.BlueRed, aspect='auto')
    cbar = fig.colorbar(c3, ax=ax)
    #Adjusting colorbar tick labels
    exponents = np.arange(-10, -1, 2)   
    ticks = np.concatenate([-exponents[::-1], [0], exponents])
    print(ticks)
    ticklabels = (
        [fr"$-10^{{{exp}}}$" for exp in exponents[::-1]] +
        ["0"] +
        [fr"$10^{{{exp}}}$" for exp in exponents]
    )
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(ticklabels)
    cbar.set_label(r"Accretion rate [$M_\odot/\mathrm{yr}$]")
    ax.set_title("Net accretion")
    ax.set_xlabel("log(R) [AU]")
    ax.set_ylabel("Time [kyr]")
    fig.tight_layout()
    plt.savefig(f'{fig_imgs}/logMnetdot_2D.png', bbox_inches="tight")
    plt.show()

    ####################### Compare Mdot for different no disk inclinations ########################


    inc_nodisk_folders = [Path("../fargo3d/outputs/cloud_nodisk_it450_rotX45"), Path("../fargo3d/outputs/cloud_nodisk_it450_rotXY45"), Path("../fargo3d/outputs/cloud_nodisk_it450_rotXY30"), Path("../fargo3d/outputs/cloud_nodisk_it450_rotXY90"), Path("../fargo3d/outputs/iras04125_lowres_it450_nocomp")]
    inc_nodisk_labels = {"iras04125_lowres_it450_nocomp": r"$\mathrm{i = 0\degree}$", "cloud_nodisk_it450_rotX45": r"$\mathrm{i_X = 45\degree}$", "cloud_nodisk_it450_rotXY45": r"$\mathrm{i_{XY} = 45\degree}$", "cloud_nodisk_it450_rotXY30": r"$\mathrm{i_{XY} = 30\degree}$", "cloud_nodisk_it450_rotXY90": r"$\mathrm{i_{XY} = 90\degree}$"}

    Mdot_in_allincs_nodisk = {}
    # Mdot_out_allincs_nodisk = {}

    for f in inc_nodisk_folders:
        
        f_sim_name = str(f).split('/')[3]                       # Simulation name (for plot labels)
        domains = get_domain_spherical(f)                       # Load coordinates         
        
        # Load density and radial velocity values at multiple iterations 
        rho_allit = []
        vrad_allit = [] 

        for i in range(0, it+1, 10):     # loading density and vrad every 10 iterations
            rho_i = get_data(f, "dens", i, domains)
            vrad_i = get_data(f, "vy", i, domains)          
            rho_allit.append(rho_i)
            vrad_allit.append(vrad_i)

        vrad_allit = np.asarray(vrad_allit)
        rho_allit = np.asarray(rho_allit)
        allit_years = calc_simtime(np.asarray(range(0, it+1, 10)))       # Convert iterations to kyrs

        dotM_in_allit = []
        # dotM_out_allit = []

        for i in range(len(allit_years)):
            _, dotM_in_i, dotM_out_i = calc_accretion(rho_allit[i], vrad_allit[i], domains["theta"], domains["r"], domains["phi"], Hc, zmax, Msun)
            dotM_in_allit.append(dotM_in_i)
            # dotM_out_allit.append(dotM_out_i)

        dotM_in_allit = np.asarray(dotM_in_allit)
        # dotM_out_allit = np.asarray(dotM_out_allit)

        Mdot_in_allincs_nodisk[f_sim_name] = dotM_in_allit
        # Mdot_out_allincs_nodisk[f_sim_name] = dotM_out_allit

    # Plotting the logarithmic mass fluxes 
    fig, ax = plt.subplots()
    for key, value in Mdot_in_allincs_nodisk.items():
        ax.plot(allit_years, np.log10(-value), label=inc_nodisk_labels[key])
        # ax.plot(allit_years, dotM_out_allit, label="Outward flux")
    ax.set_xlabel(r"Time [kyr]")
    ax.set_ylabel(r"$\mathrm{\log\dot{M}}$ [$M_{sun}$/yr]")
    ax.set_title(fr"Cloudlet inclinations (no disk): Inward $\mathrm{{\log\dot{{M}}}}$ vs t (R = 10 AU)")
    fig.tight_layout()
    ax.legend(loc="lower right")   # loc='upper left', 
    plt.savefig('logMdot_vs_t_all_incs_nodisk.png')
    plt.show()

   
    ####################### Compare Mdot for different impact parameters ########################


    b_folders = [Path("../fargo3d/outputs/cloud_disk_it450_b01_rotXY45"), Path("../fargo3d/outputs/cloud_disk_it450_rotXY45"), Path("../fargo3d/outputs/cloud_disk_it450_b025_rotXY45"), Path("../fargo3d/outputs/cloud_disk_it450_b075_rotXY45")]
    b_labels = {"cloud_disk_it450_b01_rotXY45": r"$\mathrm{b / b_{crit} = 0.1}$", "cloud_disk_it450_rotXY45": r"$\mathrm{b / b_{crit} = 0.5}$", "cloud_disk_it450_b025_rotXY45": r"$\mathrm{b / b_{crit} = 0.25}$", "cloud_disk_it450_b075_rotXY45": r"$\mathrm{b / b_{crit} = 0.75}$"}

    Mdot_in_allb = {}
    # Mdot_out_allb = {}

    for f in b_folders:
        
        f_sim_name = str(f).split('/')[3]                       # Simulation name (for plot labels)
        domains = get_domain_spherical(f)                       # Load coordinates         
        
        # Load density and radial velocity values at multiple iterations 
        rho_allit = []
        vrad_allit = [] 

        for i in range(0, it+1, 10):     # loading density and vrad every 10 iterations
            rho_i = get_data(f, "dens", i, domains)
            vrad_i = get_data(f, "vy", i, domains)          
            rho_allit.append(rho_i)
            vrad_allit.append(vrad_i)

        vrad_allit = np.asarray(vrad_allit)
        rho_allit = np.asarray(rho_allit)
        allit_years = calc_simtime(np.asarray(range(0, it+1, 10)))       # Convert iterations to kyrs

        dotM_in_allit = []
        # dotM_out_allit = []

        for i in range(len(allit_years)):
            _, dotM_in_i, dotM_out_i = calc_accretion(rho_allit[i], vrad_allit[i], domains["theta"], domains["r"], domains["phi"], Hc, zmax, Msun)
            dotM_in_allit.append(dotM_in_i)
            # dotM_out_allit.append(dotM_out_i)

        dotM_in_allit = np.asarray(dotM_in_allit)
        # dotM_out_allit = np.asarray(dotM_out_allit)

        Mdot_in_allb[f_sim_name] = dotM_in_allit
        # Mdot_out_allb[f_sim_name] = dotM_out_allit

    # Plotting the logarithmic mass fluxes 
    fig, ax = plt.subplots()
    for key, value in Mdot_in_allb.items():
        ax.plot(allit_years, np.log10(-value), label=b_labels[key])
        # ax.plot(allit_years, dotM_out_allit, label="Outward flux")
    ax.set_xlabel(r"Time [kyr]")
    ax.set_ylabel(r"$\mathrm{\log\dot{M}}$ [$M_{sun}$/yr]")
    ax.set_title(fr"Cloudlet impact parameters: Inward $\mathrm{{\log\dot{{M}}}}$ vs t (R = 10 AU)")
    fig.tight_layout()
    ax.legend(loc="lower right")   # loc='upper left', 
    plt.savefig('logMdot_vs_t_all_b.png')
    plt.show()


    ####################### Compare Mdot for different inclinations ########################


    inc_folders = [Path("../fargo3d/outputs/cloud_disk_it450"), Path("../fargo3d/outputs/cloud_disk_it450_rotX45"), Path("../fargo3d/outputs/cloud_disk_it450_rotXY45"), Path("../fargo3d/outputs/cloud_disk_it450_rotXY30"), Path("../fargo3d/outputs/cloud_disk_it450_rotXY90")]

    Mdot_in_allincs = {}
    # Mdot_out_allincs = {}

    for f in inc_folders:
        
        f_sim_name = str(f).split('/')[3]                       # Simulation name (for plot labels)
        domains = get_domain_spherical(f)                       # Load coordinates         
        
        # Load density and radial velocity values at multiple iterations 
        rho_allit = []
        vrad_allit = [] 

        for i in range(0, it+1, 10):     # loading density and vrad every 10 iterations
            rho_i = get_data(f, "dens", i, domains)
            vrad_i = get_data(f, "vy", i, domains)          
            rho_allit.append(rho_i)
            vrad_allit.append(vrad_i)

        vrad_allit = np.asarray(vrad_allit)
        rho_allit = np.asarray(rho_allit)
        allit_years = calc_simtime(np.asarray(range(0, it+1, 10)))       # Convert iterations to kyrs

        dotM_in_allit = []
        # dotM_out_allit = []

        for i in range(len(allit_years)):
            _, dotM_in_i, dotM_out_i = calc_accretion(rho_allit[i], vrad_allit[i], domains["theta"], domains["r"], domains["phi"], Hc, zmax, Msun)
            dotM_in_allit.append(dotM_in_i)
            # dotM_out_allit.append(dotM_out_i)

        dotM_in_allit = np.asarray(dotM_in_allit)
        # dotM_out_allit = np.asarray(dotM_out_allit)

        Mdot_in_allincs[f_sim_name] = dotM_in_allit
        # Mdot_out_allincs[f_sim_name] = dotM_out_allit

    # Plotting the logarithmic mass fluxes 
    fig, ax = plt.subplots()
    for key, value in Mdot_in_allincs.items():
        ax.plot(allit_years, np.log10(-value), label=key)
        # ax.plot(allit_years, dotM_out_allit, label="Outward flux")
    ax.set_xlabel(r"Time [kyr]")
    ax.set_ylabel(r"$\mathrm{\log\dot{M}}$ [$M_{sun}$/yr]")
    ax.set_title(fr"Cloudlet inclinations: Inward $\mathrm{{\log\dot{{M}}}}$ vs t (R = 10 AU)")
    fig.tight_layout()
    ax.legend(loc="lower right")   # loc='upper left', 
    plt.savefig('logMdot_vs_t_all_incs.png')
    plt.show()


    ####################### Compare radial mass distributions for different cloudlet masses ########################


    cmass_folders = [Path("../fargo3d/outputs/cloud_disk_it450"), Path("../fargo3d/outputs/cloud_disk_it450_cmass01"), Path("../fargo3d/outputs/cloud_disk_it450_cmass10")]
    cmass_labels = {"cloud_disk_it450_cmass01": r"$\mathrm{M_{cloud} / M_{disk}} = 0.045$", "cloud_disk_it450_cmass10": r"$\mathrm{M_{cloud} / M_{disk}} = 4.5$", "cloud_disk_it450": r"$\mathrm{M_{cloud} / M_{disk}} = 0.45$"}

    Mdot_in_allcmass = {}
    # Mdot_out_allcmass = {}

    for f in cmass_folders:
        
        f_sim_name = str(f).split('/')[3]                       # Simulation name (for plot labels)
        domains = get_domain_spherical(f)                       # Load coordinates         
        
        # Load density and radial velocity values at multiple iterations 
        rho_allit = []
        vrad_allit = [] 

        for i in range(0, it+1, 10):     # loading density and vrad every 10 iterations
            rho_i = get_data(f, "dens", i, domains)
            vrad_i = get_data(f, "vy", i, domains)          
            rho_allit.append(rho_i)
            vrad_allit.append(vrad_i)

        vrad_allit = np.asarray(vrad_allit)
        rho_allit = np.asarray(rho_allit)
        allit_years = calc_simtime(np.asarray(range(0, it+1, 10)))       # Convert iterations to kyrs

        dotM_in_allit = []
        # dotM_out_allit = []

        for i in range(len(allit_years)):
            _, dotM_in_i, dotM_out_i = calc_accretion(rho_allit[i], vrad_allit[i], domains["theta"], domains["r"], domains["phi"], Hc, zmax, Msun)
            dotM_in_allit.append(dotM_in_i)
            # dotM_out_allit.append(dotM_out_i)

        dotM_in_allit = np.asarray(dotM_in_allit)
        # dotM_out_allit = np.asarray(dotM_out_allit)

        Mdot_in_allcmass[f_sim_name] = dotM_in_allit
        # Mdot_out_allcmass[f_sim_name] = dotM_out_allit

    # Plotting the logarithmic mass fluxes 
    fig, ax = plt.subplots()
    for key, value in Mdot_in_allcmass.items():
        ax.plot(allit_years, np.log10(-value), label=cmass_labels[key])
        # ax.plot(allit_years, dotM_out_allit, label="Outward flux")
    ax.set_xlabel(r"Time [kyr]")
    ax.set_ylabel(r"$\mathrm{\log\dot{M}}$ [$M_{sun}$/yr]")
    ax.set_title(fr"Cloudlet masses: Inward $\mathrm{{\log\dot{{M}}}}$ vs t (R = 10 AU)")
    fig.tight_layout()
    ax.legend(loc="lower right")   # loc='upper left', 
    plt.savefig('logMdot_vs_t_all_cmass.png')
    plt.show()


if __name__ == "__main__":
    main()