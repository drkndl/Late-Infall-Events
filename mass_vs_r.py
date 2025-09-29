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


def main():


    folder = Path("../cloud_disk_it450_cmass10/")         # Folder with the output files
    fig_imgs = Path("cloud_disk_it450_cmass10/imgs/")     # Folder to save images
    it = 450                                                       # FARGO snapshot of interest
    sim_name = str(fig_imgs).split('/')[0]                         # Simulation name (for plot labels)
    

    ############# Load data for single snapshot (theta = 175, r = 150, phi = 100) ######################


    domains = get_domain_spherical(folder)
    rho = get_data(folder, "dens", it, domains)         # Load 3D array of density values            

    cell_volume = calc_cell_volume(domains["theta"], domains["r"], domains["phi"])
    mass = calc_mass(rho, cell_volume)


    ############################## Load data for multiple snapshots ####################################


    # Load mass values at multiple iterations 
    mass_allit = []

    for i in range(0, it+1, 10):     # loading density every 10 iterations
        rho_i = get_data(folder, "dens", i, domains)
        mass_i = calc_mass(rho_i, cell_volume)
        mass_allit.append(mass_i)

    mass_allit = np.asarray(mass_allit)


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


    ####################### Compare radial mass distributions for different inclinations ########################

    efkwfkfw
    inc_folders = [Path("../cloud_disk_it450"), Path("../cloud_disk_it450_rotX45"), Path("../cloud_disk_it450_rotXY45"), Path("../cloud_disk_it450_rotXY30"), Path("../cloud_disk_it450_rotXY90")]

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


    ####################### Compare radial mass distributions for different cloudlet masses ########################


    cmass_folders = [Path("../cloud_disk_it450"), Path("../cloud_disk_it450_cmass01"), Path("../cloud_disk_it450_cmass10")]
    cmass_labels = {"cloud_disk_it450_cmass01": r"$\mathrm{M_{cloud} / M_{disk}} = 0.045$", "cloud_disk_it450_cmass10": r"$\mathrm{M_{cloud} / M_{disk}} = 4.5$", "cloud_disk_it450": r"$\mathrm{M_{cloud} / M_{disk}} = 0.45$"}

    shell_mass_allcmass = {}
    cum_mass_allcmass = {}

    for f in cmass_folders:
        
        f_sim_name = str(f).split('/')[1]                       # Simulation name (for plot labels)
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


if __name__ == "__main__":
    main()
    