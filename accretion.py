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


def calc_accretion(rho, vr, theta, r0, r0_id, phi, max_height, Msun):
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
    dotM_total:   Total mass flux in and out of shell [g/s]
    dotM_in:      Inward accretion [g/s]
    dotM_out:     Outward flux [g/s]
    """

    z = r0 * np.cos(theta)                          # Disk heights at given radius

    # Boolean mask selecting only polar angles within max_height so that we ignore cloudlet
    theta_mask = np.abs(z) <= max_height
    theta_sel = theta[theta_mask]
    # print(np.round(np.degrees(theta_sel), 1))
    
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


def main():


    # folder = Path("../cloud_disk_it450_rotX45/")                    # Folder with the output files
    folder = Path("../fargo3d/outputs/cloud_disk_it450_rotX45")       # Folder with the output files (BinAC2)
    fig_imgs = Path("cloud_disk_it450_rotX45/imgs/")                  # Folder to save images
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
    r0 = domains["r"][0]     # Taking the innermost radius to check accretion onto star
    dotM_tot, dotM_in, dotM_out = calc_accretion(rho, vrad, domains["theta"], r0, 0, domains["phi"], zmax, Msun)
    print(f"Total flux across inner shell: {dotM_tot:.3e} Msun/yr")
    print(f"Outflow: {dotM_out:.3e} Msun/yr, Inflow: {dotM_in:.3e} Msun/yr")
    

    ############################## Load data for multiple snapshots ####################################


    # Load density and radial velocity values at multiple iterations 
    rho_allit = []
    vrad_allit = [] 

    for i in range(0, it+1, 5):     # loading density and vrad every 5 iterations
        rho_i = get_data(folder, "dens", i, domains)
        vrad_i = get_data(folder, "vy", i, domains)          
        rho_allit.append(rho_i)
        vrad_allit.append(vrad_i)
    
    vrad_allit = np.asarray(vrad_allit)
    rho_allit = np.asarray(rho_allit)
    allit_years = calc_simtime(np.asarray(range(0, it+1, 5)))       # Convert iterations to kyrs


    #################################### Accretion onto star ##########################################


    dotM_tot_allit = []
    dotM_in_allit = []
    dotM_out_allit = []
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


    zmax_array = [Hc, 2 * Hc, 3 * Hc, 4 * Hc, 5 * Hc, 10 * Hc]
    zmax_labels = {Hc: "Hc", 2 * Hc: "2Hc", 3 * Hc: "3Hc", 4 * Hc: "4Hc", 5 * Hc: "5Hc", 10 * Hc: "10Hc"}
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
    for key, value in Mdot_in_allzmax.items():
        ax.plot(allit_years, np.log10(-value), label=zmax_labels[key])
    ax.set_xlabel(r"Time [kyr]")
    ax.set_ylabel(r"$\mathrm{\log\dot{M}}$ [$M_{sun}$/yr]")
    ax.set_title(fr"{sim_name}: Inward $\mathrm{{\log\dot{{M}}}}$ vs t (R = 10 AU)")
    fig.tight_layout()
    ax.legend(loc="lower right")   # loc='upper left', 
    plt.savefig(f'{fig_imgs}/logMdot_vs_t_all_zmax.png')
    plt.show()

    # Plotting the outward mass fluxes for all max heights as a sanity check (SHOULD BE ZERO!)
    fig, ax = plt.subplots()
    for key, value in Mdot_in_allzmax.items():
        ax.plot(allit_years, value, label=zmax_labels[key])
    ax.set_xlabel(r"Time [kyr]")
    ax.set_ylabel(r"$\mathrm{\log\dot{M}}$ [$M_{sun}$/yr]")
    ax.set_title(fr"{sim_name}: Outward $\mathrm{{\log\dot{{M}}}}$ vs t (R = 10 AU)")
    fig.tight_layout()
    ax.legend(loc="lower right")   # loc='upper left', 
    plt.savefig(f'{fig_imgs}/logMoutdot_vs_t_all_zmax.png')
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
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(5,7), sharex=True)

    c1 = ax1.imshow(np.log10(-Mdot_in_2D), extent=[np.log10(domains["r"].min()/au), np.log10(domains["r"].max()/au), allit_years.min(), allit_years.max()], origin="lower", cmap="inferno", aspect='auto', vmin=-17.5, vmax=-5)
    fig.colorbar(c1, ax=ax1, label=r"$\mathrm{\log\dot{M} [M_{sun}/yr]}$")
    ax1.set_title("Inward flux")
    ax1.set_ylabel("Time [kyr]")

    c2 = ax2.imshow(np.log10(Mdot_out_2D), extent=[np.log10(domains["r"].min()/au), np.log10(domains["r"].max()/au), allit_years.min(), allit_years.max()], origin="lower", cmap="inferno", aspect='auto', vmin=-17.5, vmax=-5)
    fig.colorbar(c2, ax=ax2, label=r"$\mathrm{\log\dot{M} [M_{sun}/yr]}$")
    ax2.set_title("Outward flux")
    ax2.set_ylabel("Time [kyr]")

    c3 = ax3.imshow(np.log10(Mdot_net_2D), extent=[np.log10(domains["r"].min()/au), np.log10(domains["r"].max()/au), allit_years.min(), allit_years.max()], origin="lower", cmap="inferno", aspect='auto', vmin=-17.5, vmax=-5)
    fig.colorbar(c3, ax=ax3, label=r"$\mathrm{\log\dot{M} [M_{sun}/yr]}$")
    ax3.set_title("Net accretion")
    ax3.set_xlabel("log(R) [AU]")
    ax3.set_ylabel("Time [kyr]")
    fig.tight_layout()
    plt.savefig(f'{fig_imgs}/logMdot_2D.png', bbox_inches="tight")
    plt.show()
    efjenkvnkev


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