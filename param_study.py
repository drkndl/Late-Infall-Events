# Analysis of parameter study across inclination axes, Mcloud / Mdisk, Rout 

import numpy as np
from pathlib import Path
from read import get_domain_spherical, get_data, load_par_file, get_param_value
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import colormaps as cmaps
from analysis import calc_cell_volume, calc_mass, sph_to_cart, calc_simtime, calc_LRL, calc_eccen, vel_sph_to_cart, centering, calc_angular_momentum, isolate_disk, calc_L_average, calc_e_average, calc_inc_twist, calc_whirl, calc_total_L, ini_cloudlet_pos, isolate_outer_disk
from accretion import scale_height, calc_accretion
from cloud_acc_efficiency import stellar_accretion_mass
from no_thoughts_just_plots import param_study_plot, make_evol_GIF, load_sciviscolor_colormaps
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
plt.rcParams['lines.linewidth'] = 3.5
plt.rcParams['axes.labelsize'] = 15     # x/y label size
plt.rcParams['xtick.labelsize'] = 12     # x-tick label size
plt.rcParams['ytick.labelsize'] = 12     # y-tick label size
plt.rcParams['legend.fontsize'] = 14     # legend font size

# Colours for usual parameter study
cmap1 = cmaps.agsunset
colorset1 = cmap1(np.linspace(0, 1, cmap1.N))[:2]
cmap2 = cmaps.sunsetdark
colorset2 = cmap2(np.linspace(0, 1, cmap2.N))[:2]
colours = np.concatenate((colorset1, colorset2), axis=0)

def main():

    # Simulation data in BinAC2
    # folders = [Path("../fargo3d/outputs/cloud_disk_it450_rotX45"), Path("../fargo3d/outputs/cloud_disk_it450_rotY45"), Path("../fargo3d/outputs/cloud_disk_it450_Rout30_rotX45"), Path("../fargo3d/outputs/cloud_disk_it450_Rout30_rotY45"), Path("../fargo3d/outputs/cloud_disk_it450_cmass10_rotX45"), Path("../fargo3d/outputs/cloud_disk_it450_cmass10_rotY45"), Path("../fargo3d/outputs/cloud_disk_it450_cmass10_Rout30_rotX45"), Path("../fargo3d/outputs/cloud_disk_it450_cmass10_Rout30_rotY45")]

    # Simulation data locally
    folders = [Path("../cloud_disk_it450_rotX45"), Path("../cloud_disk_it450_rotY45"), Path("../cloud_disk_it450_Rout30_correct_rotX45"), Path("../cloud_disk_it450_Rout30_correct_rotY45"), Path("../cloud_disk_it450_cmass10_rotX45"), Path("../cloud_disk_it450_cmass10_rotY45"), Path("../cloud_disk_it450_Rout30_cmass10_correct_rotX45"), Path("../cloud_disk_it450_Rout30_cmass10_correct_rotY45")] #, Path("../cloud_disk_it450_retro_rotX45"), Path("../cloud_disk_it450_retro_rotY45")]

    # Simulation data locally (prograde vs retrograde comparison)
    # folders = [Path("../cloud_disk_it450_rotX45"), Path("../cloud_disk_it450_rotY45"), Path("../cloud_disk_it450_retro_rotX45"), Path("../cloud_disk_it450_retro_rotY45")]

    folders_labels = {"cloud_disk_it450_rotX45": r"$\mathrm{X_{45}, M_{c} / M_{d}=0.45, R_{out} = 100}$", "cloud_disk_it450_rotY45": r"$\mathrm{Y_{45}, M_{c} / M_{d}=0.45, R_{out} = 100}$", 
    "cloud_disk_it450_Rout30_correct_rotX45": r"$\mathrm{X_{45}, M_{c} / M_{d}=0.45, R_{out} = 30}$",
    "cloud_disk_it450_Rout30_correct_rotY45": r"$\mathrm{Y_{45}, M_{c} / M_{d}=0.45, R_{out} = 30}$",
    "cloud_disk_it450_cmass10_rotX45": r"$\mathrm{X_{45}, M_{c} / M_{d}=4.5, R_{out} = 100}$",
    "cloud_disk_it450_cmass10_rotY45": r"$\mathrm{Y_{45}, M_{c} / M_{d}=4.5, R_{out} = 100}$",
    "cloud_disk_it450_Rout30_cmass10_correct_rotX45": r"$\mathrm{X_{45}, M_{c} / M_{d}=4.5, R_{out} = 30}$",
    "cloud_disk_it450_Rout30_cmass10_correct_rotY45": r"$\mathrm{Y_{45}, M_{c} / M_{d}=4.5, R_{out} = 30}$"} #,
    # "cloud_disk_it450_retro_rotX45": r"$\mathrm{X_{45, retro}, M_{c} / M_{d}=0.45, R_{out} = 100}$",
    # "cloud_disk_it450_retro_rotY45": r"$\mathrm{Y_{45, retro}, M_{c} / M_{d}=0.45, R_{out} = 100}$"}

    style_legend = [
    Line2D([0], [0], color='black', lw=3.5, linestyle='-', label=r"$X_{45}$"),
    Line2D([0], [0], color='black', lw=3.5, linestyle='--', label=r"$Y_{45}$")]

    color_legend = [
    Line2D([0], [0], color=colours[0], lw=3.5, label=r"$M_c/M_d=0.45,\ R_{out}=100$"),
    Line2D([0], [0], color=colours[1], lw=3.5, label=r"$M_c/M_d=0.45,\ R_{out}=30$"),
    Line2D([0], [0], color=colours[2], lw=3.5, label=r"$M_c/M_d=4.5,\ R_{out}=100$"),
    Line2D([0], [0], color=colours[3], lw=3.5, label=r"$M_c/M_d=4.5,\ R_{out}=30$")]

    ############# NICE PLOTS LABELS ###############
    # folders_labels = {"cloud_disk_it450_rotX45": r"$M_{c} / M_{d}=0.45, R_{out} = \mathrm{100 AU}}$", "cloud_disk_it450_rotY45": r"$\mathrm{Y_{45}, M_{c} / M_{d}=0.45, R_{out} = 100}$", 
    # "cloud_disk_it450_Rout30_rotX45": r"$M_{c} / M_{d}=0.45, R_{out} = \mathrm{30 AU}}$",
    # "cloud_disk_it450_Rout30_rotY45": r"$\mathrm{Y_{45}, M_{c} / M_{d}=0.45, R_{out} = \mathrm{30 AU}}$",
    # "cloud_disk_it450_cmass10_rotX45": r"$M_{c} / M_{d}=4.5, R_{out} = \mathrm{100 AU}}$",
    # "cloud_disk_it450_cmass10_rotY45": r"$\mathrm{Y_{45}, M_{c} / M_{d}=4.5, R_{out} = \mathrm{100 AU}}$",
    # "cloud_disk_it450_cmass10_Rout30_rotX45": r"$M_{c} / M_{d}=4.5, R_{out} = \mathrm{30 AU}}$",
    # "cloud_disk_it450_cmass10_Rout30_rotY45": r"$\mathrm{Y_{45}, M_{c} / M_{d}=4.5, R_{out} = \mathrm{30 AU}}$"} #,
    # "cloud_disk_it450_retro_rotX45": r"$\mathrm{X_{45, retro}, M_{c} / M_{d}=0.45, R_{out} = 100}$",
    # "cloud_disk_it450_retro_rotY45": r"$\mathrm{Y_{45, retro}, M_{c} / M_{d}=0.45, R_{out} = 100}$"}

    # folders_labels = {"cloud_disk_it450_rotX45": r"$\mathrm{X_{45, pro}, M_{c} / M_{d}=0.45, R_{out} = 100}$", "cloud_disk_it450_rotY45": r"$\mathrm{Y_{45, pro}, M_{c} / M_{d}=0.45, R_{out} = 100}$", "cloud_disk_it450_retro_rotX45": r"$\mathrm{X_{45, retro}, M_{c} / M_{d}=0.45, R_{out} = 100}$",
    # "cloud_disk_it450_retro_rotY45": r"$\mathrm{Y_{45, retro}, M_{c} / M_{d}=0.45, R_{out} = 100}$"}

    disk_mass_folder = {}              # Disk masses m(t, theta, r, phi) for all sims
    disk_inc_avg_folder = {}           # Average disk inclination inc(t) for all sims
    disk_twist_avg_folder = {}         # Average disk twist twist(t) for all sims
    disk_inc_folder = {}               # Inclinations inc(r, t) at all radii and all timesteps for all sims
    disk_twist_folder = {}             # Disk twists twist(r, t) at all radii and all timesteps for all sims
    disk_Mdot_folder = {}              # Mass accretion rate onto star Mdot(t) for all sims
    disk_whirl_folder = {}             # Disk whirl at each timestep for all sims
    Mcumsum_folder ={}                 # Cumulative mass values M_cumsum(r, t) for all sims
    dMcumdlogr_folder ={}              # log(dM_cumsum/dlogr)(r, t) for all sims
    Mcloud_acc_folder = {}             # M_cloud,acc(t) for all sims
    abs_Mcloud_acc_folder = {}         # Absolute M_cloud,acc(t) for all sims
    cloud_acc_eff_folder = {}          # Cloud accretion efficiency(t) for all sims
    abs_cloud_acc_eff_folder = {}      # Absolute cloud accretion efficiency(t) for all sims
    e_folder = {}                      # Radial profile of eccentricities for all sims

    N = 10                                             # Load data for every N iterations

    # Central coordinates of the primary
    Px, Py, Pz = 0, 0, 0                                # Primary is in the centre of the simulation

    for f in folders:
        
        # Load simulation 
        f_sim_name = str(f).split('/')[1]                       # Simulation name (for plot labels)
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

        cloud_dist = get_param_value("DistIni", f_sim_name)
        rho0 = get_data(f, "dens", 0, domains)         # Load 3D array of density values at first iteration
        cloud_phi = ini_cloudlet_pos(cloud_dist, rho0, 1e-17, domains["r"], domains["phi"])

        # Calculating cloudlet mass
        cloud_mass_ini = get_param_value('CloudletMass', f_sim_name)
        print(f"Cloudlet mass: {cloud_mass_ini/Msun:.4f} Msun, {cloud_mass_ini:.2e} g")

        rc = 0.5 * (domains["r"][1:] + domains["r"][:-1])
        
        # Save density, mass, vrad, average inclination, average twist at multiple iterations 
        rho_allit = []
        vrad_allit = []
        mass_allit = []
        inc_avg_allit = []
        twist_avg_allit = []
        inc_allit =[]
        twist_allit = []
        di_dr_allit = []
        whirl_allit = []
        disk_mass_allit = []     
        e_allit = []                        


        ######################## Calculating mass, inc, twist values ####################################


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
            warp_thresh = -17   # log of density threshold for which we can see the warp in the primary
            warp_buffer = 500   # Isolates a box of 2 * warp_buffer around the star (AU)
            rho_c_warp_i, vx_c_warp_i, vy_c_warp_i, vz_c_warp_i, Lx_c_warp_i, Ly_c_warp_i, Lz_c_warp_i, _ = isolate_disk(X_c, Y_c, Z_c, Px * au, Py * au, Pz * au, warp_buffer * au, rho_c_i, vx_c_i, vy_c_i, vz_c_i, Lx_i, Ly_i, Lz_i, warp_thresh) 

            # Calculating the warped/broken disk mass 
            disk_mass_radial_i = np.nansum(rho_c_warp_i * cell_volume, axis=(0,2))
            # plt.plot(np.log10(domains["r"][:-1]/au), np.log10(disk_mass_radial_it))
            # plt.show()        
            disk_mass_i = np.sum(disk_mass_radial_i)
            # print(f"Disk mass: {disk_mass_i/Msun:.4f} Msun, {disk_mass_i:.2e} g")
            disk_mass_allit.append(disk_mass_i)

            Ax_warp_i, Ay_warp_i, Az_warp_i = calc_LRL(mass_i, Mstar, vx_c_warp_i, vy_c_warp_i, vz_c_warp_i, Lx_c_warp_i, Ly_c_warp_i, Lz_c_warp_i, X_c, Y_c, Z_c)
            ex_warp_i, ey_warp_i, ez_warp_i = calc_eccen(Ax_warp_i, Ay_warp_i, Az_warp_i, mass_i, Mstar)
            eavg_disk_i = calc_e_average(ex_warp_i, ey_warp_i, ez_warp_i, mass_i)
            e_allit.append(eavg_disk_i)

            # Calculating inclination, twist in the disk and saving the radial averages
            Lx_warp_avg_i, Ly_warp_avg_i, Lz_warp_avg_i = calc_L_average(Lx_c_warp_i, Ly_c_warp_i, Lz_c_warp_i, mass_i)
            inc_i, twist_i = calc_inc_twist(Lx_warp_avg_i, Ly_warp_avg_i, Lz_warp_avg_i, domains["r"], savefig=False, plot=False)
            inc_allit.append(inc_i)
            twist_allit.append(twist_i)
            inc_avg_allit.append(np.nanmean(inc_i))
            twist_avg_allit.append(np.nanmean(twist_i))    

            # Finding the radial separation between the inner and outer disks at the discontinuity of dinc/dr
            print(inc_i.shape, rc.shape)
            di_dr = np.gradient(inc_i, rc)
            di_dr_allit.append(di_dr)
            r_break_i = rc[np.nanargmax(np.abs(di_dr))]
            print("r_break:", r_break_i/au)

            # Isolating the outer disk using r_break and a density threshold
            R_c = centering(R)
            RCYL_c = centering(RCYL)
            outer_thresh = -17
            outer_rho_i, Lx_outer_i, Ly_outer_i, Lz_outer_i, outer_ids = isolate_outer_disk(R_c, RCYL_c, Z_c, r_break_i, rho_c_i, Lx_i, Ly_i, Lz_i, threshold=outer_thresh)
            r_outer_extent = np.sqrt(X_c[outer_ids]**2 +  Y_c[outer_ids]**2 + Z_c[outer_ids]**2) / au
            mask = (domains["r"]/au >= r_outer_extent.min()) & (domains["r"]/au <= r_outer_extent.max())
            r_outer = domains["r"][mask]

            # Radially averaged outer disk momenta
            Lx_outer_avg_i, Ly_outer_avg_i, Lz_outer_avg_i = calc_L_average(Lx_outer_i, Ly_outer_i, Lz_outer_i, mass_i)
            # plot_twist_arrows(Lx_outer_avg, Ly_outer_avg, Lz_outer_avg, domains["r"], r_outer, sim_params=None, title=f"{sim_name}: Outer Disk Twist", savefig=True, figfolder=f'{fig_imgs}/outer_twist_arrows_it{it}_dens{outer_thresh}.png', bbox_inches="tight", showfig=True)

            # Total outer disk momenta
            Lx_outer_disk_i, Ly_outer_disk_i, Lz_outer_disk_i = calc_total_L(Lx_outer_avg_i, Ly_outer_avg_i, Lz_outer_avg_i)
            outer_whirl = calc_whirl(Lx_outer_disk_i, Ly_outer_disk_i, Lz_outer_disk_i, cloud_phi)
            print("OUTER DISK WHIRL: ", outer_whirl)
            whirl_allit.append(outer_whirl)    

            # Calculating and plotting the total angular momentum of the warped disk
            # Lx_disk_i, Ly_disk_i, Lz_disk_i = calc_total_L(Lx_warp_avg_i, Ly_warp_avg_i, Lz_warp_avg_i)
            # whirl = calc_whirl(Lx_disk_i, Ly_disk_i, Lz_disk_i, cloud_phi)
            # whirl_allit.append(whirl)    

        vrad_allit = np.asarray(vrad_allit)
        rho_allit = np.asarray(rho_allit)
        mass_allit = np.asarray(mass_allit)
        inc_avg_allit = np.asarray(inc_avg_allit)
        twist_avg_allit = np.asarray(twist_avg_allit)
        inc_allit = np.asarray(inc_allit)
        twist_allit = np.asarray(twist_allit)
        whirl_allit = np.asarray(whirl_allit)
        disk_mass_allit = np.asarray(disk_mass_allit)
        e_allit = np.asarray(e_allit)

        disk_mass_initial = disk_mass_allit[0]
        cloud_mass_accreted = disk_mass_allit - disk_mass_initial
        cloud_acc_eff = cloud_mass_accreted / cloud_mass_ini * 100

        disk_mass_folder[f_sim_name] = mass_allit
        disk_inc_avg_folder[f_sim_name] = inc_avg_allit
        disk_twist_avg_folder[f_sim_name] = twist_avg_allit
        disk_inc_folder[f_sim_name] = inc_allit
        disk_twist_folder[f_sim_name] = twist_allit
        disk_whirl_folder[f_sim_name] = whirl_allit
        Mcloud_acc_folder[f_sim_name] = cloud_mass_accreted / Msun
        cloud_acc_eff_folder[f_sim_name] = cloud_acc_eff
        e_folder[f_sim_name] = e_allit

        allit_years = calc_simtime(np.asarray(range(0, it+1, N)))       # Convert iterations to kyrs


        ################ Calculating M_cumsum and dlogMcum/dlogr values for each sim ####################


        shell_mass_allit = np.sum(mass_allit, axis=(1,3))              # Shell mass in shape (nt, nr-1)
        M_cumsum_allit = np.cumsum(shell_mass_allit, axis=1)           # Cumulative sum mass in shape (nt, nr-1)
        dM_cum_allit = np.diff(M_cumsum_allit, axis=1)
        dlogR = np.diff(np.log10(domains["r"][:-1]))

        Mcumsum_folder[f_sim_name] = M_cumsum_allit
        dMcumdlogr_folder[f_sim_name] = np.log10(dM_cum_allit/dlogR)   # log(dMcum/dlogr) in shape (nt, nr-2)


        ############################### Calculating mass accretion values ##################################

        # Loading all required basic set up parameters
        h0 = get_param_value("AspectRatio", f_sim_name)      # Aspect ratio
        f = get_param_value("FlaringIndex", f_sim_name)      # Flaring index
        R0 = 5.2 * au                                      # As defined in FARGO3D [cm]
        
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

        # Calculating absolute cloud mass accreted and absolute cloud accretion efficiency
        M_stellar_acc = stellar_accretion_mass(np.abs(dotM_in_allit), allit_years)
        abs_cloud_mass_acc = cloud_mass_accreted + M_stellar_acc
        abs_cloud_acc_eff = abs_cloud_mass_acc / cloud_mass_ini * 100

        abs_Mcloud_acc_folder[f_sim_name] = abs_cloud_mass_acc / Msun
        abs_cloud_acc_eff_folder[f_sim_name] = abs_cloud_acc_eff


    
    ############################################ Time evolution plots ############################################


    # Plotting inc_avg vs time 
    fig, ax = plt.subplots(figsize=(11, 7))
    param_study_plot(fig, ax, disk_inc_avg_folder, allit_years, folders_labels, colours, xlabel=r"Time [kyr]", ylabel=r"$\mathrm{inc_{avg} [deg]}$", title=fr"Time Evolution of Average Inclinations $(\mathrm{{\rho \geq 10^{warp_thresh}}})$", figfolder=f'nice_plots_correct/param_study_inc_avg_vs_t_warp{warp_thresh}.png', savefig=True, showfig=True)

    # Plotting twist_avg vs time 
    fig, ax = plt.subplots(figsize=(11, 7))
    param_study_plot(fig, ax, disk_twist_avg_folder, allit_years, folders_labels, colours, xlabel=r"Time [kyr]", ylabel=r"$\mathrm{twist_{avg} [deg]}$", title=fr"Time Evolution of Average Twist $(\mathrm{{\rho \geq 10^{warp_thresh}}})$", figfolder=f'nice_plots_correct/param_study_twist_avg_vs_t_warp{warp_thresh}.png', savefig=True, showfig=True)

    # Plotting net Mcloud,acc vs time
    fig, ax = plt.subplots(figsize=(11, 7))
    param_study_plot(fig, ax, Mcloud_acc_folder, allit_years, folders_labels, colours, xlabel=r"Time [kyr]", ylabel=r"Net $M_{cloud,acc}$ [$M_\odot$]", title=fr"Cloud mass accreted $(\mathrm{{\rho \geq 10^{warp_thresh}}})$", figfolder=f'nice_plots_correct/param_study_Mcloudacc_vs_t_warp{warp_thresh}.png', savefig=True, showfig=True)

    # Plotting net cloud accretion efficiency vs time
    fig, ax = plt.subplots(figsize=(11, 7))
    param_study_plot(fig, ax, cloud_acc_eff_folder, allit_years, folders_labels, colours, xlabel=r"Time [kyr]", ylabel=r"Net $\mathrm{\frac{M_{cloud,acc}}{M_{cloud}} x 100}$ [%]", title=fr"Cloud accretion efficiency $(\mathrm{{\rho \geq 10^{warp_thresh}}})$", figfolder=f'nice_plots_correct/param_study_cloudacceff_vs_t_warp{warp_thresh}.png', savefig=True, showfig=True)

    # Plotting absolute Mcloud,acc vs time
    fig, ax = plt.subplots(figsize=(11, 7))
    param_study_plot(fig, ax, abs_Mcloud_acc_folder, allit_years, folders_labels, colours, xlabel=r"Time [kyr]", ylabel=r"Absolute $M_{cloud,acc}$ [$M_\odot$]", title=fr"Cloud mass accreted $(\mathrm{{\rho \geq 10^{warp_thresh}}})$", figfolder=f'nice_plots_correct/param_study_abs_Mcloudacc_vs_t_warp{warp_thresh}.png', savefig=True, showfig=True)

    # Plotting absolute cloud accretion efficiency vs time
    fig, ax = plt.subplots(figsize=(11, 7))
    param_study_plot(fig, ax, abs_cloud_acc_eff_folder, allit_years, folders_labels, colours, xlabel=r"Time [kyr]", ylabel=r"Absolute $\mathrm{\frac{M_{cloud,acc}}{M_{cloud}} x 100}$ [%]", title=fr"Cloud accretion efficiency $(\mathrm{{\rho \geq 10^{warp_thresh}}})$", figfolder=f'nice_plots_correct/param_study_abs_cloudacceff_vs_t_warp{warp_thresh}.png', savefig=True, showfig=True)


    # Plotting whirl vs time
    fig, ax = plt.subplots(figsize=(11, 7))
    current_color_index = -1
    last_base = None
    for key, value in disk_whirl_folder.items():

        # Plotting rotX simulations in solid lines and rotY simulations in dashed lines (but same colour for easy comparison)
        if "rotX" in key:
            base = key.replace("rotX", "")
            ls = "-"
        elif "rotY" in key:
            base = key.replace("rotY", "")
            ls = "--"        
        # Only change color when we encounter a new base (first time we see either X or Y)
        if base != last_base:
            current_color_index = (current_color_index + 1) % len(colours)
            last_base = base

        colour = colours[current_color_index]
        ax.plot(allit_years[5:], value[5:], linestyle=ls, color=colour, label=folders_labels[key])
        
    ax.set_xlabel(r"Time [kyr]")
    ax.set_ylabel(r"Whirl [deg]")
    # ax.set_title(fr"Time Evolution of Whirl $(\mathrm{{\rho \geq 10^{warp_thresh}}})$")
    leg1 = ax.legend(handles=style_legend, loc="upper left", frameon=False)
    ax.add_artist(leg1)
    ax.legend(handles=color_legend, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False)
    plt.tight_layout() 
    plt.savefig(f'nice_plots_correct/param_study_whirl_vs_t_warp{warp_thresh}.png', bbox_inches="tight")
    plt.show()

    # Plotting absolute values of twist_avg vs time 
    fig, ax = plt.subplots(figsize=(11, 7))
    current_color_index = -1
    last_base = None
    for key, value in disk_twist_avg_folder.items():

        # Plotting rotX simulations in solid lines and rotY simulations in dashed lines (but same colour for easy comparison)
        if "rotX" in key:
            base = key.replace("rotX", "")
            ls = "-"
        elif "rotY" in key:
            base = key.replace("rotY", "")
            ls = "--"       
        # Only change color when we encounter a new base (first time we see either X or Y)
        if base != last_base:
            current_color_index = (current_color_index + 1) % len(colours)
            last_base = base

        colour = colours[current_color_index]
        ax.plot(allit_years, np.abs(value), linestyle=ls, color=colour, label=folders_labels[key])
        
    ax.set_xlabel(r"Time [kyr]")
    ax.set_ylabel(r"$\mathrm{\vert twist_{avg}\vert [deg]}$")
    # ax.set_title(fr"Time Evolution of Absolute Values of Average Twist $(\mathrm{{\rho \geq 10^{warp_thresh}}})
    leg1 = ax.legend(handles=style_legend, loc="upper left", frameon=False)
    ax.add_artist(leg1)
    ax.legend(handles=color_legend, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False)
    plt.tight_layout() 
    plt.savefig(f'nice_plots_correct/param_study_absolute_twist_avg_vs_t_warp{warp_thresh}.png', bbox_inches="tight")
    plt.show()

    # # Plotting time evolution of cumulative mass at 100 AU
    # fig, ax = plt.subplots(figsize=(11, 7))
    # current_color_index = -1
    # last_base = None
    # for key, value in Mcumsum_folder.items():

    #     # Plotting rotX simulations in solid lines and rotY simulations in dashed lines (but same colour for easy comparison)
    #     if "rotX" in key:
    #         base = key.replace("rotX", "")
    #         ls = "-"
    #     elif "rotY" in key:
    #         base = key.replace("rotY", "")
    #         ls = "--"
    #     # Only change color when we encounter a new base (first time we see either X or Y)
    #     if base != last_base:
    #         current_color_index = (current_color_index + 1) % len(colours)
    #         last_base = base

    #     colour = colours[current_color_index]
    #     ax.plot(np.log10(domains["r"]/au)[:-1], np.log10(value[-1, :]))   # -1 corresponds to last iteration

    # ax.set_xlabel(r"$\log(r)$ [AU]")
    # ax.set_ylabel(r"$\mathrm{\log(M_{cum}(r))}$")
    # ax.set_title(fr"Time Evolution of $\mathrm{{\log(M_{{cum}}(r))}}$ (100 AU)")
    # leg1 = ax.legend(handles=style_legend, loc="upper left", frameon=False)
    # ax.add_artist(leg1)
    # ax.legend(handles=color_legend, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False)
    # plt.tight_layout()
    # plt.savefig('param_study_Mcumsum_100AU_vs_t.png', bbox_inches="tight")
    # plt.show()

    # # Plotting time evolution of dMcumdlogr at 100 AU
    # fig, ax = plt.subplots(figsize=(11, 7))
    # current_color_index = -1
    # last_base = None
    # for key, value in dMcumdlogr_folder.items():

    #     # Plotting rotX simulations in solid lines and rotY simulations in dashed lines (but same colour for easy comparison)
    #     if "rotX" in key:
    #         base = key.replace("rotX", "")
    #         ls = "-"
    #     elif "rotY" in key:
    #         base = key.replace("rotY", "")
    #         ls = "--"  
    #     # Only change color when we encounter a new base (first time we see either X or Y)
    #     if base != last_base:
    #         current_color_index = (current_color_index + 1) % len(colours)
    #         last_base = base

    #     colour = colours[current_color_index]
    #     ax.plot(np.log10(domains["r"]/au)[:-2], value[-1, :])   # -1 corresponds to last iteration

    # ax.set_xlabel(r"$\log(r)$ [AU]")
    # ax.set_ylabel(r"$\mathrm{\log(dM_{cum}(r)/d\log(r))}$")
    # ax.set_title(fr"Time Evolution of $\mathrm{{\log(dM_{{cum}}(r)/d\log(r))}}$ (100 AU)")
    # leg1 = ax.legend(handles=style_legend, loc="upper left", frameon=False)
    # ax.add_artist(leg1)
    # ax.legend(handles=color_legend, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False)
    # plt.tight_layout()
    # plt.savefig('param_study_dMcumdlogr_100AU_vs_t.png', bbox_inches="tight")
    # plt.show()

    # Plotting the mass accretion rate onto star vs time
    fig, ax = plt.subplots(figsize=(11, 7))
    current_color_index = -1
    last_base = None
    for key, value in disk_Mdot_folder.items():

        # Plotting rotX simulations in solid lines and rotY simulations in dashed lines (but same colour for easy comparison)
        if "rotX" in key:
            base = key.replace("rotX", "")
            ls = "-"
        elif "rotY" in key:
            #
            base = key.replace("rotY", "")
            ls = "--"        
        # Only change color when we encounter a new base (first time we see either X or Y)
        if base != last_base:
            current_color_index = (current_color_index + 1) % len(colours)
            last_base = base

        colour = colours[current_color_index]
        ax.plot(allit_years, np.log10(-value), linestyle=ls, color=colour, label=folders_labels[key])
        
    ax.set_xlabel(r"Time [kyr]")
    ax.set_ylabel(r"$\mathrm{\log\dot{M}}$ [$M_{sun}$/yr]")
    # ax.set_title(fr"Time Evolution of Mass Accretion Rate onto Star $(\mathrm{{\rho \geq 10^{warp_thresh}}})$
    leg1 = ax.legend(handles=style_legend, loc="upper left", frameon=False)
    ax.add_artist(leg1)
    ax.legend(handles=color_legend, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False)
    plt.tight_layout()
    plt.savefig(f'nice_plots_correct/param_study_clean_Mdot_vs_t_warp{warp_thresh}.png', bbox_inches="tight")
    plt.show()


    ################################ Final timestep values vs R plots ######################################


    # Plotting inc_final vs R
    fig, ax = plt.subplots(figsize=(11, 7))
    current_color_index = -1
    last_base = None
    for key, value in disk_inc_folder.items():

        # Plotting rotX simulations in solid lines and rotY simulations in dashed lines (but same colour for easy comparison)
        if "rotX" in key:
            base = key.replace("rotX", "")
            ls = "-"
        elif "rotY" in key:
            base = key.replace("rotY", "")
            ls = "--"        
        # Only change color when we encounter a new base (first time we see either X or Y)
        if base != last_base:
            current_color_index = (current_color_index + 1) % len(colours)
            last_base = base

        colour = colours[current_color_index]
        ax.plot(np.log10(domains["r"]/au)[:-1], value[-1, :], linestyle=ls, color=colour, label=folders_labels[key])   # -1 corresponds to last iteration

    ax.set_xlabel(r"$\log(r)$ [AU]")
    ax.set_ylabel(r"Disk Inclination $(\degree)$")
    # ax.set_title(fr"Disk Inclination vs logr (53 kyr) $(\mathrm{{\rho \geq 10^{{{warp_thresh}}}}})$
    leg1 = ax.legend(handles=style_legend, loc="upper left", frameon=False)
    ax.add_artist(leg1)
    ax.legend(handles=color_legend, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False)
    plt.tight_layout()
    plt.savefig(f'nice_plots_correct/param_study_inc_final_iter_vs_r_warp{warp_thresh}.png', bbox_inches="tight")
    plt.show()

    # Plotting twist_final vs R
    fig, ax = plt.subplots(figsize=(11, 7))
    current_color_index = -1
    last_base = None
    for key, value in disk_twist_folder.items():

        # Plotting rotX simulations in solid lines and rotY simulations in dashed lines (but same colour for easy comparison)
        if "rotX" in key:
            base = key.replace("rotX", "")
            ls = "-"
        elif "rotY" in key:
            base = key.replace("rotY", "")
            ls = "--"
        
        # Only change color when we encounter a new base (first time we see either X or Y)
        if base != last_base:
            current_color_index = (current_color_index + 1) % len(colours)
            last_base = base
        colour = colours[current_color_index]
        ax.plot(np.log10(domains["r"]/au)[:-1], value[-1, :], linestyle=ls, color=colour, label=folders_labels[key])   # -1 corresponds to last iteration

    ax.set_xlabel(r"$\log(r)$ [AU]")
    ax.set_ylabel(r"Disk Twist $(\degree)$")
    # ax.set_title(fr"Twist vs logr (53 kyr) $(\mathrm{{\rho \geq 10^{warp_thresh}}})$
    leg1 = ax.legend(handles=style_legend, loc="upper left", frameon=False)
    ax.add_artist(leg1)
    ax.legend(handles=color_legend, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False)
    plt.tight_layout()
    plt.savefig(f'nice_plots_correct/param_study_twist_final_iter_vs_r_warp{warp_thresh}.png', bbox_inches="tight")
    plt.show()


    # Plotting absolute twist_final vs R
    fig, ax = plt.subplots(figsize=(11, 7))
    current_color_index = -1
    last_base = None
    for key, value in disk_twist_folder.items():

        # Plotting rotX simulations in solid lines and rotY simulations in dashed lines (but same colour for easy comparison)
        if "rotX" in key:
            base = key.replace("rotX", "")
            ls = "-"
        elif "rotY" in key:
            base = key.replace("rotY", "")
            ls = "--"        
        # Only change color when we encounter a new base (first time we see either X or Y)
        if base != last_base:
            current_color_index = (current_color_index + 1) % len(colours)
            last_base = base

        colour = colours[current_color_index]
        ax.plot(np.log10(domains["r"]/au)[:-1], np.abs(value[-1, :]), linestyle=ls, color=colour, label=folders_labels[key])   # -1 corresponds to last iteration

    ax.set_xlabel(r"$\log(r)$ [AU]")
    ax.set_ylabel(r"Absolute Disk Twist $(\vert\degree\vert)$")
    # ax.set_title(fr"Absolute twist vs logr (53 kyr) $(\mathrm{{\rho \geq 10^{warp_thresh}}})$
    leg1 = ax.legend(handles=style_legend, loc="upper left", frameon=False)
    ax.add_artist(leg1)
    ax.legend(handles=color_legend, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False)
    plt.tight_layout()
    plt.savefig(f'nice_plots_correct/param_study_absolute_twist_final_iter_vs_r_warp{warp_thresh}.png', bbox_inches="tight")
    plt.show()


    # Plotting Mcumsum_final vs R
    fig, ax = plt.subplots(figsize=(11, 7))
    current_color_index = -1
    last_base = None
    for key, value in Mcumsum_folder.items():

        # Plotting rotX simulations in solid lines and rotY simulations in dashed lines (but same colour for easy comparison)
        if "rotX" in key:
            base = key.replace("rotX", "")
            ls = "-"
        elif "rotY" in key:
            base = key.replace("rotY", "")
            ls = "--"
        # Only change color when we encounter a new base (first time we see either X or Y)
        if base != last_base:
            current_color_index = (current_color_index + 1) % len(colours)
            last_base = base

        colour = colours[current_color_index]
        ax.plot(np.log10(domains["r"]/au)[:-1], np.log10(value[-1, :]), linestyle=ls, color=colour, label=folders_labels[key])   # -1 corresponds to last iteration

    ax.set_xlabel(r"$\log(r)$ [AU]")
    ax.set_ylabel(r"$\mathrm{\log(M_{cum}(r))}$")
    # ax.set_title(fr"$\mathrm{{\log(M_{{cum}}(r))}}$ vs logr (53 kyr) $(\mathrm{{\rho \geq 10^{warp_thresh}}})$
    leg1 = ax.legend(handles=style_legend, loc="upper left", frameon=False)
    ax.add_artist(leg1)
    ax.legend(handles=color_legend, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False)
    plt.tight_layout()
    plt.savefig(f'nice_plots_correct/param_study_Mcumsum_final_iter_vs_r_warp{warp_thresh}.png', bbox_inches="tight")
    plt.show()

    # Plotting dMcumdlogr_final vs R
    fig, ax = plt.subplots(figsize=(11, 7))
    current_color_index = -1
    last_base = None
    for key, value in dMcumdlogr_folder.items():

        # Plotting rotX simulations in solid lines and rotY simulations in dashed lines (but same colour for easy comparison)
        if "rotX" in key:
            base = key.replace("rotX", "")
            ls = "-"
        elif "rotY" in key:
            base = key.replace("rotY", "")
            ls = "--"
        # Only change color when we encounter a new base (first time we see either X or Y)
        if base != last_base:
            current_color_index = (current_color_index + 1) % len(colours)
            last_base = base

        colour = colours[current_color_index]
        ax.plot(np.log10(domains["r"]/au)[:-2], value[-1, :], linestyle=ls, color=colour, label=folders_labels[key])   # -1 corresponds to last iteration

    ax.set_xlabel(r"$\log(r)$ [AU]")
    ax.set_ylabel(r"$\mathrm{\log(dM_{cum}(r)/d\log(r))}$")
    # ax.set_title(fr"$\mathrm{{\log(dM_{{cum}}(r)/d\log(r))}}$ vs logr (53 kyr) $(\mathrm{{\rho \geq 10^{warp_thresh}}})$
    leg1 = ax.legend(handles=style_legend, loc="upper left", frameon=False)
    ax.add_artist(leg1)
    ax.legend(handles=color_legend, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False)
    plt.tight_layout()
    plt.savefig(f'nice_plots_correct/param_study_dMcumdlogr_final_iter_vs_r_warp{warp_thresh}.png', bbox_inches="tight")
    plt.show()

    # Plotting eccentricity_final vs R
    fig, ax = plt.subplots(figsize=(11, 7))
    current_color_index = -1
    last_base = None
    for key, value in e_folder.items():

        # Plotting rotX simulations in solid lines and rotY simulations in dashed lines (but same colour for easy comparison)
        if "rotX" in key:
            base = key.replace("rotX", "")
            ls = "-"
        elif "rotY" in key:
            base = key.replace("rotY", "")
            ls = "--"
        # Only change color when we encounter a new base (first time we see either X or Y)
        if base != last_base:
            current_color_index = (current_color_index + 1) % len(colours)
            last_base = base

        colour = colours[current_color_index]
        ax.plot(np.log10(domains["r"]/au)[:-1], value[-1, :], linestyle=ls, color=colour, label=folders_labels[key])   # -1 corresponds to last iteration

    ax.set_xlabel(r"$\log(r)$ [AU]")
    ax.set_ylabel(r"e")
    leg1 = ax.legend(handles=style_legend, loc="upper left", frameon=False)
    ax.add_artist(leg1)
    ax.legend(handles=color_legend, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False)
    plt.tight_layout()
    plt.savefig(f'nice_plots_correct/param_study_e_final_iter_vs_r_warp{warp_thresh}.png')
    plt.show()

    # Making a GIF to show time evolution of dMcumdlogr vs logR
    # for t in range(len(allit_years)):
    #     fig, ax = plt.subplots(figsize=(11, 7))
    #     current_color_index = -1
    #     last_base = None
    #     for key, value in dMcumdlogr_folder.items():

    #         # Plotting rotX simulations in solid lines and rotY simulations in dashed lines (but same colour for easy comparison)
    #         if "rotX" in key:
    #             base = key.replace("rotX", "")
    #             ls = "-"
    #         elif "rotY" in key:
    #             base = key.replace("rotY", "")
    #             ls = "--"
    #         # Only change color when we encounter a new base (first time we see either X or Y)
    #         if base != last_base:
    #             current_color_index = (current_color_index + 1) % len(colours)
    #             last_base = base

    #         colour = colours[current_color_index]
    #         ax.plot(np.log10(domains["r"]/au)[:-2], value[t, :], linestyle=ls, color=colour, label=folders_labels[key])   # -1 corresponds to last iteration

    #     ax.set_xlabel(r"$\log(r)$ [AU]")
    #     ax.set_ylabel(r"$\mathrm{\log(dM_{cum}(r)/d\log(r))}$")
    #     ax.set_title(fr"$\mathrm{{\log(dM_{{cum}}(r)/d\log(r))}}$ vs logr ({int(allit_years[t])} kyr) $(\mathrm{{\rho \geq 10^{warp_thresh}}})$")
    # leg1 = ax.legend(handles=style_legend, loc="upper left", frameon=False)
    # ax.add_artist(leg1)
    # ax.legend(handles=color_legend, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False)
    #     ax.set_ylim(26, 33)
 
    #     plt.tight_layout()
    #     plt.savefig(f'nice_plots_correct/param_study_dMcumdlogr_it{t}.png', bbox_inches="tight")
    #     plt.close()

    # make_evol_GIF(".", "param_study_dMcumdlogr_it", f"param_study_dMcumdlogr_warp{warp_thresh}_movie")



if __name__ == "__main__":
    main()