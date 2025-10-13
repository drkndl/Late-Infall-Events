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


def scale_height(r0, h0, R0, f):
    """
    Calculates the scale height of the disk at a given radius r0

    Inputs:
    ------
    h0:           Aspect ratio (float)
    R0:           Radius at which aspect ratio is defined (FARGO3D standard) [cm] (float)
    f:            Disk flaring index (float)

    Outputs:
    -------
    Hc:           Scale height at given radius r0 [cm]
    """

    Hc = h0 * r0 * np.power(r0 / R0, f)             
    return Hc


def check_accretion(rho, vr, theta, r, phi, Hc, max_height, Msun):
    """
    Function to check if cloudlet mass is accreted onto the star

    Inputs:
    ------
    rho:          3D array of densities [g/cm^3] (shape: ntheta, nr, nphi)
    vr:           3D array of radial velocities [cm/s] (shape: ntheta, nr, nphi)
    theta:        1D array of polar angles [radians] (shape: ntheta)
    r:            1D array of radius [cm] (shape: nr)
    phi:          1D array of azimuthal angles (shape: nphi)
    Hc:           Scale height at given radius r0 [cm]
    max_height:   Maximum height within which we calculate accretion [cm]
    Msun:         Mass of the Sun [g]

    Outputs:
    -------
    dotM_total:   Total mass flux in and out of shell [g/s]
    dotM_in:      Inward accretion [g/s]
    dotM_out:     Outward flux [g/s]
    """

    r0 = r[0]                                       # Taking the innermost radius to check accretion onto star
    z = r0 * np.cos(theta)                          # Disk heights at inner radius

    # Boolean mask selecting only polar angles within max_height so that we ignore cloudlet
    theta_mask = np.abs(z) <= max_height
    theta_sel = theta[theta_mask]
    # print(np.round(np.degrees(theta_sel), 1))
    
    dtheta_sel = np.gradient(theta_sel)
    dphi = np.gradient(phi)
    rho0_thetamask = rho[:, 0, :][theta_mask, :]              # shape (ntheta_mask, nphi)
    vr0_thetamask  = vr[:, 0, :][theta_mask, :]               # shape (ntheta_mask, nphi)

    theta2d, phi2d = np.meshgrid(theta_sel, phi, indexing='ij')
    dtheta2d, dphi2d = np.meshgrid(dtheta_sel, dphi, indexing='ij')
    dA = r0**2 * np.sin(theta2d) * dtheta2d * dphi2d

    mass_flux = rho0_thetamask * vr0_thetamask * dA        # (g/s) 

    dotM_total = np.sum(mass_flux)
    dotM_out = np.sum(mass_flux[vr0_thetamask > 0])         # Assuming vr > 0 is outward
    dotM_in  = np.sum(mass_flux[vr0_thetamask < 0])         # Assuming vr < 0 is inward

    # Converting it from g/s to Msun/s
    dotM_total = dotM_total / Msun
    dotM_in = dotM_in / Msun
    dotM_out = dotM_out / Msun

    return dotM_total, dotM_in, dotM_out


def main():


    # folder = Path("../cloud_nodisk_it450_rotXY90/")                        # Folder with the output files
    folder = Path("../fargo3d/outputs/cloud_nodisk_it450_rotXY90")     # Folder with the output files (BinAC2)
    fig_imgs = Path("cloud_nodisk_it450_rotXY90/imgs/")                  # Folder to save images
    it = 450                                                             # FARGO snapshot of interest
    sim_name = str(fig_imgs).split('/')[0]                               # Simulation name (for plot labels)
    sim_params = load_par_file(f"{sim_name}/{sim_name}.par")             # Loading simulation parameters from the .par file

    R0 = 5.2 * au                         # As defined in FARGO3D [cm]
    # f = sim_params['FlaringIndex']      # Flaring index
    # h0 = sim_params['AspectRatio']      # Aspect ratio
    f = 0.25                              # Flaring index (from setups/cloud_disk.par)
    h0 = 0.03799                          # Aspect ratio (from setups/cloud_disk.par)


    ############# Load data for single snapshot (theta = 175, r = 150, phi = 100) ######################


    domains = get_domain_spherical(folder)
    rho = get_data(folder, "dens", it, domains)         # Load 3D array of density values   
    vrad = get_data(folder, "vy", it, domains)          # Load 3D array of radial velocities v_rad
    cell_volume = calc_cell_volume(domains["theta"], domains["r"], domains["phi"])
    mass = calc_mass(rho, cell_volume)

    Hc = scale_height(domains["r"][0], h0, R0, f)
    zmax = 4 * Hc
    dotM_tot, dotM_in, dotM_out = check_accretion(rho, vrad, domains["theta"], domains["r"], domains["phi"], Hc, zmax, Msun)
    print(f"Total flux across inner shell: {dotM_tot:.3e} Msun/s")
    print(f"Outflow: {dotM_out:.3e} Msun/s, Inflow: {dotM_in:.3e} Msun/s")


    ############################## Load data for multiple snapshots ####################################


    # Load density and radial velocity values at multiple iterations 
    mass_allit = []
    rho_allit = []
    vrad_allit = [] 

    for i in range(0, it+1, 10):     # loading density and vrad every 10 iterations
        rho_i = get_data(folder, "dens", i, domains)
        vrad_i = get_data(folder, "vy", i, domains)          
        mass_i = calc_mass(rho_i, cell_volume)
        rho_allit.append(rho_i)
        vrad_allit.append(vrad_i)
        mass_allit.append(mass_i)

    mass_allit = np.asarray(mass_allit)
    vrad_allit = np.asarray(vrad_allit)
    rho_allit = np.asarray(rho_allit)
    allit_years = calc_simtime(np.asarray(range(0, it+1, 10)))       # Convert iterations to kyrs


    #################################### Accretion onto star ##########################################


    dotM_tot_allit = []
    dotM_in_allit = []
    dotM_out_allit = []
    for i in range(len(allit_years)):
        dotM_tot_i, dotM_in_i, dotM_out_i = check_accretion(rho_allit[i], vrad_allit[i], domains["theta"], domains["r"], domains["phi"], Hc, zmax, Msun)
        dotM_tot_allit.append(dotM_tot_i)
        dotM_in_allit.append(dotM_in_i)
        dotM_out_allit.append(dotM_out_i)

    dotM_tot_allit = np.asarray(dotM_tot_allit)
    dotM_in_allit = np.asarray(dotM_in_allit)
    dotM_out_allit = np.asarray(dotM_out_allit)

    # Plotting the mass fluxes 
    fig, ax = plt.subplots()
    # plt.plot(allit_years, np.abs(dotM_tot_allit), label="Total flux")
    plt.plot(allit_years, np.log10(-dotM_in_allit), label="Log Inward flux")
    plt.plot(allit_years, dotM_out_allit, label="Outward flux")
    ax.set_xlabel(r"Time [kyr]")
    ax.set_ylabel(r"$\mathrm{\log\dot{M}}$ [$M_{sun}$/s]")
    ax.set_title(fr"{sim_name}: $\mathrm{{\log\dot{{M}}}}$ vs t (R = 10 AU)")
    plt.legend()
    plt.savefig(f'{fig_imgs}/logMdot_vs_t_it{it}.png')
    plt.show()

    fig, ax = plt.subplots()
    # plt.plot(allit_years, np.abs(dotM_tot_allit), label="Total flux")
    plt.plot(allit_years, dotM_in_allit, label="Inward flux")
    plt.plot(allit_years, dotM_out_allit, label="Outward flux")
    ax.set_xlabel(r"Time [kyr]")
    ax.set_ylabel(r"$\mathrm{\dot{M}}$ [$M_{sun}$/s]")
    ax.set_title(fr"{sim_name}: $\mathrm{{\dot{{M}}}}$ vs t (R = 10 AU)")
    plt.legend()
    plt.savefig(f'{fig_imgs}/Mdot_vs_t_it{it}.png')
    plt.show()


    ############################# Mass in each spherical shell ########################################


    # Mass in each spherical shell for a single iteration
    shell_mass = np.sum(mass, axis=(0,2))                       # Shell mass in shape (nr-1)

    fig, ax = plt.subplots()
    ax.plot(np.log10(domains["r"]/au)[:-1], np.log10(shell_mass))
    plt.axvline(2, linestyle=":", color="black")
    plt.text(1.9, 29, "disk edge", rotation=90, verticalalignment='center')
    ax.set_xlabel(r"$\log(r)$ [AU]")
    ax.set_ylabel(r"$\mathrm{\log(M_{shell}(r))}$ [g]")
    ax.set_title(fr"{sim_name}: $\mathrm{{\log(M_{{shell}}(r))}}$ vs logr")
    plt.savefig(f'{fig_imgs}/logM_vs_logr_it{it}.png')
    plt.show()

    # Time evolution of mass in spherical shells
    shell_mass_allit = np.sum(mass_allit, axis=(1,3))           # Shell mass in shape (evol_it, nr-1)
    evol_it = len(shell_mass_allit[:,1])                        # Total number of snapshots loaded
    dtkyrs = calc_simtime(np.asarray(range(0, it+1, 10)))       # Convert iterations to kyrs
    
    cols = cm.get_cmap('viridis', evol_it)
    fig, ax = plt.subplots()
    for i in range(evol_it):
        plt.plot(np.log10(domains["r"]/au)[:-1], np.log10(shell_mass_allit[i]), color=cols(i))
    ax.set_xlabel(r"$\log(r)$ [AU]")
    ax.set_ylabel(r"$\mathrm{\log(M_{shell}(r))}$ [g]")
    plt.axvline(2, linestyle=":", color="black")
    plt.text(1.9, 24, "disk edge", rotation=90, verticalalignment='center')
    ax.set_title(fr"{sim_name}: $\mathrm{{\log(M_{{shell}}(r))}}$ vs logr time evolution")

    norm = mcolors.Normalize(vmin=min(dtkyrs), vmax=max(dtkyrs))     # Colorbar formatting
    sm = cm.ScalarMappable(cmap=cols, norm=norm)
    sm.set_array([])  
    cbar = plt.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("Time [kyr]")
    plt.savefig(f'{fig_imgs}/logM_vs_logr_timeevol.png')
    plt.show()


    ############################## log(dM/dr) in each spherical shell ######################################


    # Mass in each spherical shell for a single iteration
    shell_dM = np.diff(shell_mass)
    dR = np.diff(domains["r"][:-1])

    fig, ax = plt.subplots()
    ax.plot(np.log10(domains["r"]/au)[:-2], np.log10(shell_dM/dR))
    plt.axvline(2, linestyle=":", color="black")
    plt.text(1.9, 14, "disk edge", rotation=90, verticalalignment='center')
    ax.set_xlabel(r"$\log(r)$ [AU]")
    ax.set_ylabel(r"$\mathrm{\log(dM_{shell}(r)/dr)}$")
    ax.set_title(rf"{sim_name}: $\mathrm{{\log(dM_{{shell}}(r)/dr)}}$ vs logr")
    plt.savefig(f'{fig_imgs}/logdMdr_vs_logr_it{it}.png')
    plt.show()

    # Time evolution of mass in spherical shells
    shell_mass_allit = np.sum(mass_allit, axis=(1,3))           # Shell mass in shape (evol_it, nr-1)
    shell_dM_allit = np.diff(shell_mass_allit)
    
    fig, ax = plt.subplots()
    for i in range(evol_it):
        plt.plot(np.log10(domains["r"]/au)[:-2], np.log10(shell_dM_allit[i]/dR), color=cols(i))
    ax.set_xlabel(r"$\log(r)$ [AU]")
    ax.set_ylabel(r"$\mathrm{\log(dM_{shell}(r)/dr)}$")
    plt.axvline(2, linestyle=":", color="black")
    plt.text(1.9, 10, "disk edge", rotation=90, verticalalignment='center')
    ax.set_title(fr"{sim_name}: $\mathrm{{\log(dM_{{shell}}(r)/dr)}}$ vs logr time evolution")

    norm = mcolors.Normalize(vmin=min(dtkyrs), vmax=max(dtkyrs))     # Colorbar formatting
    sm = cm.ScalarMappable(cmap=cols, norm=norm)
    sm.set_array([])  
    cbar = plt.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("Time [kyr]")
    plt.savefig(f'{fig_imgs}/logdMdr_vs_logr_timeevol.png')
    plt.show()


    ############################################### Cumulative mass ###############################################


    M_cumsum = np.cumsum(shell_mass)
    
    fig, ax = plt.subplots()
    ax.plot(np.log10(domains["r"]/au)[:-1], np.log10(M_cumsum))
    plt.axvline(2, linestyle=":", color="black")
    plt.text(1.9, 29, "disk edge", rotation=90, verticalalignment='center')
    ax.set_xlabel(r"$\log(r)$ [AU]")
    ax.set_ylabel(r"$\mathrm{\log(M_{cum}(r))}$")
    ax.set_title(fr"{sim_name}: $\mathrm{{\log(M_{{cum}}(r))}}$ vs logr")
    plt.savefig(f'{fig_imgs}/cumlogM_vs_logr_it{it}.png')
    plt.show()

    M_cumsum_allit = np.cumsum(shell_mass_allit, axis=1)
    
    fig, ax = plt.subplots()
    for i in range(evol_it):
        plt.plot(np.log10(domains["r"]/au)[:-1], np.log10(M_cumsum_allit[i]), color=cols(i))
    ax.set_xlabel(r"$\log(r)$ [AU]")
    ax.set_ylabel(r'$\mathrm{{\log(M_{{cum}}(r))}}$')
    plt.axvline(2, linestyle=":", color="black")
    plt.text(1.9, 29, "disk edge", rotation=90, verticalalignment='center')
    ax.set_title(fr"{sim_name}: $\mathrm{{\log(M_{{cum}}(r))}}$ vs logr time evol")

    norm = mcolors.Normalize(vmin=min(dtkyrs), vmax=max(dtkyrs))     # Colorbar formatting
    sm = cm.ScalarMappable(cmap=cols, norm=norm)
    sm.set_array([])  
    cbar = plt.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("Time [kyr]")
    plt.savefig(f'{fig_imgs}/cumlogM_vs_logr_timeevol.png')
    plt.show()


    ############################### log ((dM_cumsum) / dr) in each spherical shell ##############################


    dM_cum = np.diff(M_cumsum)
    
    fig, ax = plt.subplots()
    ax.plot(np.log10(domains["r"]/au)[:-2], np.log10(dM_cum/dR))
    plt.axvline(2, linestyle=":", color="black")
    plt.text(1.9, 15, "disk edge", rotation=90, verticalalignment='center')
    ax.set_xlabel(r"$\log(r)$ [AU]")
    ax.set_ylabel(r"$\log(\mathrm{dM_{cum}(r)/dr})$")
    ax.set_title(fr"{sim_name}: log($\mathrm{{dM_{{cum}}(r)/dr}}$) vs logr")
    plt.savefig(f'{fig_imgs}/logdMcumdr_vs_logr_it{it}.png')
    plt.show()

    dM_cum_allit = np.diff(M_cumsum_allit, axis=1)
    
    fig, ax = plt.subplots()
    for i in range(evol_it):
        plt.plot(np.log10(domains["r"]/au)[:-2], np.log10(dM_cum_allit[i]/dR), color=cols(i))
    ax.set_xlabel(r"$\log(r)$ [AU]")
    ax.set_ylabel(r"$\log(\mathrm{dM_{cum}(r)/dr})$")
    plt.axvline(2, linestyle=":", color="black")
    plt.text(1.9, 10, "disk edge", rotation=90, verticalalignment='center')
    ax.set_title(fr"{sim_name}: log($\mathrm{{dM_{{cum}}(r)/dr}}$) vs logr time evol")

    norm = mcolors.Normalize(vmin=min(dtkyrs), vmax=max(dtkyrs))     # Colorbar formatting
    sm = cm.ScalarMappable(cmap=cols, norm=norm)
    sm.set_array([])  
    cbar = plt.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("Time [kyr]")
    plt.savefig(f'{fig_imgs}/logdMcumdr_vs_logr_timeevol.png')
    plt.show()


    ############################### log((dM_cumsum) / dlog(r)) in each spherical shell ##############################


    dlogR = np.diff(np.log10(domains["r"][:-1]))
    
    fig, ax = plt.subplots()
    ax.plot(np.log10(domains["r"]/au)[:-2], np.log10(dM_cum/dlogR))
    plt.axvline(2, linestyle=":", color="black")
    plt.text(1.9, 30, "disk edge", rotation=90, verticalalignment='center')
    ax.set_xlabel(r"$\log(r)$ [AU]")
    ax.set_ylabel(r"$\mathrm{\log(dM_{cum}(r)/d\log(r))}$")
    ax.set_title(fr"{sim_name}: $\mathrm{{\log(dM_{{cum}}(r)/d\log(r))}}$ vs logr")
    plt.savefig(f'{fig_imgs}/dMcumdlogr_vs_logr_it{it}.png')
    plt.show()
    
    fig, ax = plt.subplots()
    for i in range(evol_it):
        plt.plot(np.log10(domains["r"]/au)[:-2], np.log10(dM_cum_allit[i]/dlogR), color=cols(i))
    ax.set_xlabel(r"$\log(r)$ [AU]")
    ax.set_ylabel(r"$\mathrm{\log(dM_{cum}(r)/d\log(r))}$")
    plt.axvline(2, linestyle=":", color="black")
    plt.text(1.9, 26, "disk edge", rotation=90, verticalalignment='center')
    ax.set_title(fr"{sim_name}: $\mathrm{{\log(dM_{{cum}}(r)/d\log(r))}}$ vs logr time evol")

    norm = mcolors.Normalize(vmin=min(dtkyrs), vmax=max(dtkyrs))     # Colorbar formatting
    sm = cm.ScalarMappable(cmap=cols, norm=norm)
    sm.set_array([])  
    cbar = plt.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("Time [kyr]")
    plt.savefig(f'{fig_imgs}/dMcumdlogr_vs_logr_timeevol.png')
    plt.show()


    ##################################### Comparing different logM vs logr plots ################################

    
    fig, ax = plt.subplots()
    ax.plot(np.log10(domains["r"]/au)[:-1], np.log10(shell_mass), label=r"$\mathrm{\log(M_{shell}(r))}$")
    ax.plot(np.log10(domains["r"]/au)[:-2], np.log10(dM_cum/dR), label=r"$\log(\mathrm{dM_{cum}(r)/dr})$")
    ax.plot(np.log10(domains["r"]/au)[:-2], np.log10(dM_cum/dlogR), label=r"$\mathrm{{\log(dM_{{cum}}(r)/d\log(r))}}$")
    plt.axvline(2, linestyle=":", color="black")
    plt.text(1.9, 23, "disk edge", rotation=90, verticalalignment='center')
    ax.set_xlabel(r"$\log(r)$ [AU]")
    # ax.set_ylabel(r"$\mathrm{\log(dM_{cum}(r)/d\log(r))}$")
    ax.set_title(fr"{sim_name}: Different logM vs logr")
    ax.legend()
    plt.savefig(f'{fig_imgs}/compare_logM_vs_logr_it{it}.png')
    plt.show()


    ####################### Compare radial mass dist., Mdot for different no disk inclinations ########################


    inc_nodisk_folders = [Path("../fargo3d/outputs/cloud_nodisk_it450_rotX45"), Path("../fargo3d/outputs/cloud_nodisk_it450_rotXY45"), Path("../fargo3d/outputs/cloud_nodisk_it450_rotXY30"), Path("../fargo3d/outputs/cloud_nodisk_it450_rotXY90")]
    inc_nodisk_labels = {"cloud_nodisk_it450_rotX45": r"$\mathrm{i_X = 45\degree}$", "cloud_nodisk_it450_rotXY45": r"$\mathrm{i_{XY} = 45\degree}$", "cloud_nodisk_it450_rotXY30": r"$\mathrm{i_{XY} = 30\degree}$", "cloud_nodisk_it450_rotXY90": r"$\mathrm{i_{XY} = 90\degree}$"}

    ########## Mass distribution calculations

    shell_mass_allincs_nodisk = {}
    cum_mass_allincs_nodisk = {}

    for f in inc_nodisk_folders:
        
        f_sim_name = str(f).split('/')[3]                       # Simulation name (for plot labels)
        domains = get_domain_spherical(f)                       # Load coordinates
        f_rho = get_data(f, "dens", it, domains)                # Load 3D array of density values            

        cell_volume = calc_cell_volume(domains["theta"], domains["r"], domains["phi"])
        f_mass = calc_mass(f_rho, cell_volume)

        # Mass in each spherical shell for a single iteration
        f_shell_mass = np.sum(f_mass, axis=(0,2))                       # Shell mass in shape (nr-1)
        shell_mass_allincs_nodisk[f_sim_name] = f_shell_mass

        # Cumulative mass in each spherical shell for a single iteration
        f_M_cumsum = np.cumsum(f_shell_mass)
        cum_mass_allincs_nodisk[f_sim_name] = f_M_cumsum

    fig, ax = plt.subplots()
    for key, value in shell_mass_allincs_nodisk.items():
        ax.plot(np.log10(domains["r"]/au)[:-1], np.log10(value), label=inc_nodisk_labels[key])
    ax.set_xlabel(r"$\log(r)$ [AU]")
    ax.set_ylabel(r"$\mathrm{\log(M_{shell}(r))}$ [g]")
    plt.axvline(2, linestyle=":", color="black")
    plt.text(1.9, 26, "disk edge", rotation=90, verticalalignment='center')
    ax.set_title(f"Cloudlet inclinations (no disks): $\mathrm{{\log(M_{{shell}}(r))}}$ vs logr (53kyr)")
    ax.legend()
    plt.savefig('logM_vs_logr_all_incs_nodisk.png')
    plt.show()

    fig, ax = plt.subplots()
    # inset_ax = inset_axes(ax, width="35%", height="35%", bbox_to_anchor=(0.6, 0.25, 0.95, 0.95), bbox_transform=fig.transFigure, loc="lower left")      
    # y1, y2 = 31.7, 31.9
    # x1, x2 = 1.5, 3.5
    # inset_ax.set_ylim(y1, y2)
    # inset_ax.set_xlim(x1, x2)
    for key, value in cum_mass_allincs_nodisk.items():
        ax.plot(np.log10(domains["r"]/au)[:-1], np.log10(value), label=inc_nodisk_labels[key])
        # inset_ax.plot(np.log10(domains["r"]/au)[:-1], np.log10(value))
    ax.set_xlabel(r"$\log(r)$ [AU]")
    ax.set_ylabel(r"$\mathrm{\log(M_{cum}(r))}$")
    ax.axvline(2, linestyle=":", color="black")
    ax.text(1.9, 29, "disk edge", rotation=90, verticalalignment='center')
    # inset_ax.axvline(2, linestyle=":", color="black")
    # inset_ax.tick_params(axis='both', labelsize=8)
    ax.set_title(fr"Cloudlet inclinations (no disks): $\mathrm{{\log(M_{{cum}}(r))}}$ vs logr (53kyr)")
    fig.tight_layout()
    ax.legend(loc="upper right")   # loc='upper left', 
    plt.savefig('cumlogM_vs_logr_all_incs_nodisk.png')
    plt.show()

    ###### Mass accretion calculations

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
            _, dotM_in_i, dotM_out_i = check_accretion(rho_allit[i], vrad_allit[i], domains["theta"], domains["r"], domains["phi"], Hc, zmax, Msun)
            dotM_in_allit.append(dotM_in_i)
            # dotM_out_allit.append(dotM_out_i)

        dotM_in_allit = np.asarray(dotM_in_allit)
        # dotM_out_allit = np.asarray(dotM_out_allit)

        Mdot_in_allincs_nodisk[f_sim_name] = dotM_in_allit
        # Mdot_out_allincs_nodisk[f_sim_name] = dotM_out_allit

    # Plotting the mass fluxes 
    fig, ax = plt.subplots()
    for key, value in Mdot_in_allincs_nodisk.items():
        ax.plot(allit_years, np.log10(-value), label=inc_nodisk_labels[key])
        # ax.plot(allit_years, dotM_out_allit, label="Outward flux")
    ax.set_xlabel(r"Time [kyr]")
    ax.set_ylabel(r"$\mathrm{\log\dot{M}}$ [$M_{sun}$/s]")
    ax.set_title(fr"Cloudlet inclinations (no disk): Inward $\mathrm{{\log\dot{{M}}}}$ vs t (R = 10 AU)")
    fig.tight_layout()
    ax.legend(loc="upper right")   # loc='upper left', 
    plt.savefig('logMdot_vs_t_all_incs_nodisk.png')
    plt.show()

    fig, ax = plt.subplots()
    for key, value in Mdot_in_allincs_nodisk.items():
        ax.plot(allit_years, value, label=inc_nodisk_labels[key])
        # ax.plot(allit_years, dotM_out_allit, label="Outward flux")
    ax.set_xlabel(r"Time [kyr]")
    ax.set_ylabel(r"$\mathrm{\dot{M}}$ [$M_{sun}$/s]")
    ax.set_title(fr"Cloudlet inclinations (no disk): Inward $\mathrm{{\dot{{M}}}}$ vs t (R = 10 AU)")
    fig.tight_layout()
    ax.legend(loc="upper right")   # loc='upper left', 
    plt.savefig('Mdot_vs_t_all_incs_nodisk.png')
    plt.show()
        

   
    ####################### Compare radial mass dist., Mdot for different impact parameters ########################


    b_folders = [Path("../fargo3d/outputs/cloud_disk_it450_rotXY45"), Path("../fargo3d/outputs/cloud_disk_it450_b025_rotXY45"), Path("../fargo3d/outputs/cloud_disk_it450_b075_rotXY45")]
    b_labels = {"cloud_disk_it450_rotXY45": r"$\mathrm{b / b_{crit} = 0.5}$", "cloud_disk_it450_b025_rotXY45": r"$\mathrm{b / b_{crit} = 0.25}$", "cloud_disk_it450_b075_rotXY45": r"$\mathrm{b / b_{crit} = 0.75}$"}

    ####### Mass distribution calculations

    shell_mass_allb = {}
    cum_mass_allb = {}

    for f in b_folders:
        
        f_sim_name = str(f).split('/')[3]                       # Simulation name (for plot labels)
        domains = get_domain_spherical(f)                       # Load coordinates
        f_rho = get_data(f, "dens", it, domains)                # Load 3D array of density values            

        cell_volume = calc_cell_volume(domains["theta"], domains["r"], domains["phi"])
        f_mass = calc_mass(f_rho, cell_volume)

        # Mass in each spherical shell for a single iteration
        f_shell_mass = np.sum(f_mass, axis=(0,2))                       # Shell mass in shape (nr-1)
        shell_mass_allb[f_sim_name] = f_shell_mass

        # Cumulative mass in each spherical shell for a single iteration
        f_M_cumsum = np.cumsum(f_shell_mass)
        cum_mass_allb[f_sim_name] = f_M_cumsum

    fig, ax = plt.subplots()
    for key, value in shell_mass_allb.items():
        ax.plot(np.log10(domains["r"]/au)[:-1], np.log10(value), label=b_labels[key])
    ax.set_xlabel(r"$\log(r)$ [AU]")
    ax.set_ylabel(r"$\mathrm{\log(M_{shell}(r))}$ [g]")
    plt.axvline(2, linestyle=":", color="black")
    plt.text(1.9, 26, "disk edge", rotation=90, verticalalignment='center')
    ax.set_title(f"Cloudlet impact parameters: $\mathrm{{\log(M_{{shell}}(r))}}$ vs logr (53kyr)")
    ax.legend()
    plt.savefig('logM_vs_logr_all_b.png')
    plt.show()

    fig, ax = plt.subplots()
    inset_ax = inset_axes(ax, width="35%", height="35%", bbox_to_anchor=(0.6, 0.25, 0.95, 0.95), bbox_transform=fig.transFigure, loc="lower left")      
    y1, y2 = 31.7, 31.9
    x1, x2 = 1.5, 3.5
    inset_ax.set_ylim(y1, y2)
    inset_ax.set_xlim(x1, x2)
    for key, value in cum_mass_allb.items():
        ax.plot(np.log10(domains["r"]/au)[:-1], np.log10(value), label=b_labels[key])
        inset_ax.plot(np.log10(domains["r"]/au)[:-1], np.log10(value))
    ax.set_xlabel(r"$\log(r)$ [AU]")
    ax.set_ylabel(r"$\mathrm{\log(M_{cum}(r))}$")
    ax.axvline(2, linestyle=":", color="black")
    ax.text(1.9, 29, "disk edge", rotation=90, verticalalignment='center')
    inset_ax.axvline(2, linestyle=":", color="black")
    inset_ax.tick_params(axis='both', labelsize=8)
    ax.set_title(fr"Cloudlet impact parameters: $\mathrm{{\log(M_{{cum}}(r))}}$ vs logr (53kyr)")
    fig.tight_layout()
    ax.legend(loc="upper right")   # loc='upper left', 
    plt.savefig('cumlogM_vs_logr_all_b.png')
    plt.show()


    ###### Mass accretion calculations

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
            _, dotM_in_i, dotM_out_i = check_accretion(rho_allit[i], vrad_allit[i], domains["theta"], domains["r"], domains["phi"], Hc, zmax, Msun)
            dotM_in_allit.append(dotM_in_i)
            # dotM_out_allit.append(dotM_out_i)

        dotM_in_allit = np.asarray(dotM_in_allit)
        # dotM_out_allit = np.asarray(dotM_out_allit)

        Mdot_in_allb[f_sim_name] = dotM_in_allit
        # Mdot_out_allb[f_sim_name] = dotM_out_allit

    # Plotting the mass fluxes 
    fig, ax = plt.subplots()
    for key, value in Mdot_in_allb.items():
        ax.plot(allit_years, np.log10(-value), label=b_labels[key])
        # ax.plot(allit_years, dotM_out_allit, label="Outward flux")
    ax.set_xlabel(r"Time [kyr]")
    ax.set_ylabel(r"$\mathrm{\log\dot{M}}$ [$M_{sun}$/s]")
    ax.set_title(fr"Cloudlet impact parameters: Inward $\mathrm{{\log\dot{{M}}}}$ vs t (R = 10 AU)")
    fig.tight_layout()
    ax.legend(loc="upper right")   # loc='upper left', 
    plt.savefig('logMdot_vs_t_all_b.png')
    plt.show()

    fig, ax = plt.subplots()
    for key, value in Mdot_in_allb.items():
        ax.plot(allit_years, value, label=b_labels[key])
        # ax.plot(allit_years, dotM_out_allit, label="Outward flux")
    ax.set_xlabel(r"Time [kyr]")
    ax.set_ylabel(r"$\mathrm{\dot{M}}$ [$M_{sun}$/s]")
    ax.set_title(fr"Cloudlet impact parameters: Inward $\mathrm{{\dot{{M}}}}$ vs t (R = 10 AU)")
    fig.tight_layout()
    ax.legend(loc="upper right")   # loc='upper left', 
    plt.savefig('Mdot_vs_t_all_b.png')
    plt.show()


    ####################### Compare radial mass dist., Mdot for different inclinations ########################


    inc_folders = [Path("../fargo3d/outputs/cloud_disk_it450"), Path("../fargo3d/outputs/cloud_disk_it450_rotX45"), Path("../fargo3d/outputs/cloud_disk_it450_rotXY45"), Path("../fargo3d/outputs/cloud_disk_it450_rotXY30"), Path("../fargo3d/outputs/cloud_disk_it450_rotXY90")]

    ###### Mass distribution calculations

    shell_mass_allincs = {}
    cum_mass_allincs = {}

    for f in inc_folders:
        
        f_sim_name = str(f).split('/')[1]                       # Simulation name (for plot labels)
        domains = get_domain_spherical(f)                       # Load coordinates
        f_rho = get_data(f, "dens", it, domains)                # Load 3D array of density values            

        cell_volume = calc_cell_volume(domains["theta"], domains["r"], domains["phi"])
        f_mass = calc_mass(f_rho, cell_volume)

        # Mass in each spherical shell for a single iteration
        f_shell_mass = np.sum(f_mass, axis=(0,2))                       # Shell mass in shape (nr-1)
        shell_mass_allincs[f_sim_name] = f_shell_mass

        # Cumulative mass in each spherical shell for a single iteration
        f_M_cumsum = np.cumsum(f_shell_mass)
        cum_mass_allincs[f_sim_name] = f_M_cumsum

    fig, ax = plt.subplots()
    for key, value in shell_mass_allincs.items():
        ax.plot(np.log10(domains["r"]/au)[:-1], np.log10(value), label=key)
    ax.set_xlabel(r"$\log(r)$ [AU]")
    ax.set_ylabel(r"$\log(M(r))$ [g]")
    plt.axvline(2, linestyle=":", color="black")
    plt.text(1.9, 29, "disk edge", rotation=90, verticalalignment='center')
    ax.set_title(f"Cloudlet inclinations: logM(r) vs logr (53kyr)")
    ax.legend()
    plt.savefig('logM_vs_logr_all_incs.png')
    plt.show()

    fig, ax = plt.subplots()
    inset_ax = inset_axes(ax, width="35%", height="35%", bbox_to_anchor=(0.6, 0.25, 0.95, 0.95), bbox_transform=fig.transFigure, loc="lower left")      
    y1, y2 = 31.82, 31.9
    x1, x2 = 1.5, 3.5
    inset_ax.set_ylim(y1, y2)
    inset_ax.set_xlim(x1, x2)
    for key, value in cum_mass_allincs.items():
        ax.plot(np.log10(domains["r"]/au)[:-1], np.log10(value), label=key)
        inset_ax.plot(np.log10(domains["r"]/au)[:-1], np.log10(value))
    ax.set_xlabel(r"$\log(r)$ [AU]")
    ax.set_ylabel(r"$\log(\Sigma M(r))$")
    ax.axvline(2, linestyle=":", color="black")
    ax.text(1.9, 29, "disk edge", rotation=90, verticalalignment='center')
    inset_ax.axvline(2, linestyle=":", color="black")
    inset_ax.tick_params(axis='both', labelsize=8)
    ax.set_title(fr"Cloudlet inclinations: $\Sigma$log(M(r)) vs logr (53kyr)")
    fig.tight_layout()
    ax.legend(loc="upper right")   # loc='upper left', 
    plt.savefig('cumlogM_vs_logr_all_incs.png')
    plt.show()


    ###### Mass accretion calculations

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
            _, dotM_in_i, dotM_out_i = check_accretion(rho_allit[i], vrad_allit[i], domains["theta"], domains["r"], domains["phi"], Hc, zmax, Msun)
            dotM_in_allit.append(dotM_in_i)
            # dotM_out_allit.append(dotM_out_i)

        dotM_in_allit = np.asarray(dotM_in_allit)
        # dotM_out_allit = np.asarray(dotM_out_allit)

        Mdot_in_allincs[f_sim_name] = dotM_in_allit
        # Mdot_out_allincs[f_sim_name] = dotM_out_allit

    # Plotting the mass fluxes 
    fig, ax = plt.subplots()
    for key, value in Mdot_in_allincs.items():
        ax.plot(allit_years, np.log10(-value), label=key)
        # ax.plot(allit_years, dotM_out_allit, label="Outward flux")
    ax.set_xlabel(r"Time [kyr]")
    ax.set_ylabel(r"$\mathrm{\log\dot{M}}$ [$M_{sun}$/s]")
    ax.set_title(fr"Cloudlet inclinations: Inward $\mathrm{{\log\dot{{M}}}}$ vs t (R = 10 AU)")
    fig.tight_layout()
    ax.legend(loc="upper right")   # loc='upper left', 
    plt.savefig('logMdot_vs_t_all_incs.png')
    plt.show()

    fig, ax = plt.subplots()
    for key, value in Mdot_in_allincs.items():
        ax.plot(allit_years, value, label=key)
        # ax.plot(allit_years, dotM_out_allit, label="Outward flux")
    ax.set_xlabel(r"Time [kyr]")
    ax.set_ylabel(r"$\mathrm{\dot{M}}$ [$M_{sun}$/s]")
    ax.set_title(fr"Cloudlet inclinations: Inward $\mathrm{{\dot{{M}}}}$ vs t (R = 10 AU)")
    fig.tight_layout()
    ax.legend(loc="upper right")   # loc='upper left', 
    plt.savefig('Mdot_vs_t_all_incs.png')
    plt.show()


    ####################### Compare radial mass distributions for different cloudlet masses ########################


    cmass_folders = [Path("../fargo3d/outputs/cloud_disk_it450"), Path("../fargo3d/outputs/cloud_disk_it450_cmass01"), Path("../fargo3d/outputs/cloud_disk_it450_cmass10")]
    cmass_labels = {"cloud_disk_it450_cmass01": r"$\mathrm{M_{cloud} / M_{disk}} = 0.045$", "cloud_disk_it450_cmass10": r"$\mathrm{M_{cloud} / M_{disk}} = 4.5$", "cloud_disk_it450": r"$\mathrm{M_{cloud} / M_{disk}} = 0.45$"}

    ###### mass distribution calculations

    shell_mass_allcmass = {}
    cum_mass_allcmass = {}

    for f in cmass_folders:
        
        f_sim_name = str(f).split('/')[3]                       # Simulation name (for plot labels)
        domains = get_domain_spherical(f)                       # Load coordinates
        f_rho = get_data(f, "dens", it, domains)                # Load 3D array of density values            

        cell_volume = calc_cell_volume(domains["theta"], domains["r"], domains["phi"])
        f_mass = calc_mass(f_rho, cell_volume)

        # Mass in each spherical shell for a single iteration
        f_shell_mass = np.sum(f_mass, axis=(0,2))                       # Shell mass in shape (nr-1)
        shell_mass_allcmass[f_sim_name] = f_shell_mass

        # Cumulative mass in each spherical shell for a single iteration
        f_M_cumsum = np.cumsum(f_shell_mass)
        cum_mass_allcmass[f_sim_name] = f_M_cumsum

    fig, ax = plt.subplots()
    for key, value in shell_mass_allcmass.items():
        ax.plot(np.log10(domains["r"]/au)[:-1], np.log10(value), label=cmass_labels[key])
    ax.set_xlabel(r"$\log(r)$ [AU]")
    ax.set_ylabel(r"$\log(M(r))$ [g]")
    plt.axvline(2, linestyle=":", color="black")
    plt.text(1.9, 29, "disk edge", rotation=90, verticalalignment='center')
    ax.set_title(f"Cloudlet masses: logM(r) vs logr (53kyr)")
    ax.legend()
    plt.savefig('logM_vs_logr_all_cmass.png')
    plt.show()

    fig, ax = plt.subplots()
    for key, value in cum_mass_allcmass.items():
        ax.plot(np.log10(domains["r"]/au)[:-1], np.log10(value), label=cmass_labels[key])
    ax.set_xlabel(r"$\log(r)$ [AU]")
    ax.set_ylabel(r"$\log(\Sigma M(r))$")
    plt.axvline(2, linestyle=":", color="black")
    plt.text(1.9, 29, "disk edge", rotation=90, verticalalignment='center')
    ax.set_title(fr"Cloudlet masses: $\Sigma$log(M(r)) vs logr (53kyr)")
    fig.tight_layout()
    ax.legend(loc="lower right")   # loc='upper left', 
    plt.savefig('cumlogM_vs_logr_all_cmass.png')
    plt.show()


    ###### Mass accretion calculations

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
            _, dotM_in_i, dotM_out_i = check_accretion(rho_allit[i], vrad_allit[i], domains["theta"], domains["r"], domains["phi"], Hc, zmax, Msun)
            dotM_in_allit.append(dotM_in_i)
            # dotM_out_allit.append(dotM_out_i)

        dotM_in_allit = np.asarray(dotM_in_allit)
        # dotM_out_allit = np.asarray(dotM_out_allit)

        Mdot_in_allcmass[f_sim_name] = dotM_in_allit
        # Mdot_out_allcmass[f_sim_name] = dotM_out_allit

    # Plotting the mass fluxes 
    fig, ax = plt.subplots()
    for key, value in Mdot_in_allcmass.items():
        ax.plot(allit_years, np.log10(-value), label=cmass_labels[key])
        # ax.plot(allit_years, dotM_out_allit, label="Outward flux")
    ax.set_xlabel(r"Time [kyr]")
    ax.set_ylabel(r"$\mathrm{\log\dot{M}}$ [$M_{sun}$/s]")
    ax.set_title(fr"Cloudlet masses: Inward $\mathrm{{\log\dot{{M}}}}$ vs t (R = 10 AU)")
    fig.tight_layout()
    ax.legend(loc="upper right")   # loc='upper left', 
    plt.savefig('logMdot_vs_t_all_cmass.png')
    plt.show()

    fig, ax = plt.subplots()
    for key, value in Mdot_in_allcmass.items():
        ax.plot(allit_years, value, label=cmass_labels[key])
        # ax.plot(allit_years, dotM_out_allit, label="Outward flux")
    ax.set_xlabel(r"Time [kyr]")
    ax.set_ylabel(r"$\mathrm{\dot{M}}$ [$M_{sun}$/s]")
    ax.set_title(fr"Cloudlet masses: Inward $\mathrm{{\dot{{M}}}}$ vs t (R = 10 AU)")
    fig.tight_layout()
    ax.legend(loc="upper right")   # loc='upper left', 
    plt.savefig('Mdot_vs_t_all_cmass.png')
    plt.show()


if __name__ == "__main__":
    main()
    
