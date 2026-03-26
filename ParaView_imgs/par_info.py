#!/usr/bin/env python
import numpy as np
from scipy.integrate import trapezoid
from pathlib import Path
import sys

from astropy import units as u, constants as c
kyr = (1.e3*u.year).cgs.value
Msun = c.M_sun.cgs.value
G = 6.674e-8
MSTAR_DEFAULT = 1.9891e33
MSTAR_IRAS = 1.39237e33
MSTAR_0_5 = 9.9455e+32
MSTAR_2_0 = 3.9782e+33
MSTAR = MSTAR_DEFAULT
AU = 1.49597871e13
R0 = 5.2*AU
R_MU = 36149835.0
STEFANK = 5.6705e-5

def load_file(file):
    vars = {}
    with open(file, 'r') as f:
        for l in f.readlines():
            if l.isspace() or l.startswith("#"): continue
            key, val = l.split()[:2]
            vars[key.strip().upper()] = val.strip()
    return vars

def calc_cloud_properties(vars):
    r_cloudlet = float(vars["CLOUDLETRADIUS"])
    M_cloudlet = float(vars["CLOUDLETMASS"])
    rho_cloudlet = M_cloudlet*3/(4*np.pi*r_cloudlet**3)
    return r_cloudlet, rho_cloudlet, M_cloudlet
    
def print_tff(vars):
    v_inf = float(vars["VINF"])
    d_cloud = float(vars["DISTINI"])
    tff = d_cloud/v_inf
    print(f"Free-fall timescale: {tff:.4e} s ({tff/kyr:.3f} kyr)")
    
def print_cloud(vars):
    r_cloudlet, rho_cloudlet, M_cloudlet = calc_cloud_properties(vars)
    print(f"Cloudlet radius: {r_cloudlet/AU:.2f} AU")
    print(f"Cloudlet density: {rho_cloudlet:.2e} g/cm²")
    print(f"Cloudlet mass: {M_cloudlet:.1e} g ({M_cloudlet/MSTAR:.2e} Msun)")
    
def print_mass_ratios(vars):
    _, _, M_cloudlet = calc_cloud_properties(vars)
    ymin = float(vars["YMIN"])
    ymax = float(vars["YMAX"])
    r_grid = np.linspace(ymin, ymax, 100)
    rout = float(vars["ROUT"])
    z_o = np.minimum((r_grid-rout)/(0.05*rout), 500)
    a = float(vars["SIGMASLOPE"])
    Sigma0 = float(vars["SIGMA0"])
    Sigma = Sigma0*(r_grid/R0)**(-a)/(1+np.exp(z_o))
    M_disk = trapezoid(2*np.pi*r_grid*Sigma, r_grid)
    print(f"Mdisk/Mstar: {M_disk/MSTAR*100:.5f}%")
    print(f"Mdisk/Msun: {M_disk/Msun*100:.5f}%")
    print(f"Mcloud/Mdisk: {M_cloudlet/M_disk*100:.5f}%")
    
def print_cloud_orbit(vars):
    dist_cloud = float(vars["DISTINI"])
    v_inf = float(vars["VINF"])
    b_bcrit = float(vars["IMPACTPARAMETER"])
    is_prograde = bool(float(vars["PROGRADE"]))
    rotax_first = float(vars["ROTAXFIRST"])
    rotangle_x = float(vars["ROTANGLEX"])
    rotangle_y = float(vars["ROTANGLEY"])
    r_cloud = float(vars["CLOUDLETRADIUS"])
    b_crit = G*MSTAR/v_inf**2
    b_bcrit_min = b_bcrit-r_cloud/b_crit
    b_bcrit_max = b_bcrit+r_cloud/b_crit
    r_close = b_crit*(np.sqrt(1+b_bcrit**2)-1)
    r_close_min = b_crit*(np.sqrt(1+b_bcrit_min**2)-1)
    r_close_max = b_crit*(np.sqrt(1+b_bcrit_max**2)-1)
    trueanomaly = np.arccos((b_bcrit**2*b_crit-dist_cloud)/(dist_cloud*np.sqrt(1+b_bcrit**2)))
    specific_am = b_bcrit*b_crit*v_inf
    specific_am_min = b_bcrit_min*b_crit*v_inf
    specific_am_max = b_bcrit_max*b_crit*v_inf
    kepler_radius_am = specific_am**2/(G*MSTAR)
    kepler_radius_am_min = specific_am_min**2/(G*MSTAR)
    kepler_radius_am_max = specific_am_max**2/(G*MSTAR)
    flightpathangle = np.arctan(np.sin(trueanomaly)/(np.cos(trueanomaly)+(1+b_bcrit)**(-1./2.)))
    velocityangle = flightpathangle - trueanomaly + np.pi/2
    speed = v_inf * np.sqrt(2*b_crit/dist_cloud+1)
    x_ini0 = np.cos(trueanomaly)*dist_cloud
    y_ini0 = np.sin(trueanomaly)*dist_cloud
    z_ini0 = 0.0
    vx_ini0 = speed * np.cos(velocityangle)
    vy_ini0 = -speed * np.sin(velocityangle)
    vz_ini0 = 0.0
    if is_prograde:
        y_ini0 = -y_ini0
        vy_ini0 = -vy_ini0
    if rotangle_x != 0.0 and rotangle_y == 0.0:
        rotanglex = rotangle_x * np.pi / 180.0
        x_ini = x_ini0
        y_ini = np.cos(rotanglex) * y_ini0 - np.sin(rotanglex) * z_ini0
        z_ini = np.sin(rotanglex) * y_ini0 + np.cos(rotanglex) * z_ini0
        vx_ini = vx_ini0
        vy_ini = np.cos(rotanglex) * vy_ini0 - np.sin(rotanglex) * vz_ini0
        vz_ini = np.sin(rotanglex) * vy_ini0 + np.cos(rotanglex) * vz_ini0
    elif rotangle_y != 0.0 and rotangle_x == 0.0:
        rotangley = rotangle_y * np.pi / 180.0
        x_ini = np.cos(rotangley) * x_ini0 + np.sin(rotangley) * z_ini0
        y_ini = y_ini0
        z_ini = -np.sin(rotangley) * x_ini0 + np.cos(rotangley) * z_ini0
        vx_ini = np.cos(rotangley) * vx_ini0 + np.sin(rotangley) * vz_ini0
        vy_ini = vy_ini0
        vz_ini = -np.sin(rotangley) * vx_ini0 + np.cos(rotangley) * vz_ini0
    elif rotangle_x !=0.0 and rotangle_y != 0.0 and rotax_first == 0:
        rotanglex = rotangle_x * np.pi / 180.0
        rotangley = rotangle_y * np.pi / 180.0
        x_ini1 = x_ini0
        y_ini1 = np.cos(rotanglex) * y_ini0 - np.sin(rotanglex) * z_ini0
        z_ini1 = np.sin(rotanglex) * y_ini0 + np.cos(rotanglex) * z_ini0
        vx_ini1 = vx_ini0
        vy_ini1 = np.cos(rotanglex) * vy_ini0 - np.sin(rotanglex) * vz_ini0
        vz_ini1 = np.sin(rotanglex) * vy_ini0 + np.cos(rotanglex) * vz_ini0
        x_ini = np.cos(rotangley) * x_ini1 + np.sin(rotangley) * z_ini1
        z_ini = -np.sin(rotangley) * x_ini1 + np.cos(rotangley) * z_ini1
        y_ini = y_ini1
        vx_ini = np.cos(rotangley) * vx_ini1 + np.sin(rotangley) * vz_ini1
        vy_ini = vy_ini1
        vz_ini = -np.sin(rotangley) * vx_ini1 + np.cos(rotangley) * vz_ini1
    elif rotangle_x !=0.0 and rotangle_y != 0.0 and rotax_first == 1:
        rotanglex = rotangle_x * np.pi / 180.0
        rotangley = rotangle_y * np.pi / 180.0
        x_ini1 = np.cos(rotangley) * x_ini0 + np.sin(rotangley) * z_ini0
        y_ini1 = y_ini0
        z_ini1 = -np.sin(rotangley) * x_ini0 + np.cos(rotangley) * z_ini0
        vx_ini1 = np.cos(rotangley) * vx_ini0 + np.sin(rotangley) * vz_ini0
        vy_ini1 = vy_ini0
        vz_ini1 = -np.sin(rotangley) * vx_ini0 + np.cos(rotangley) * vz_ini0
        x_ini = x_ini1
        y_ini = np.cos(rotanglex) * y_ini1 - np.sin(rotanglex) * z_ini1
        z_ini = np.sin(rotanglex) * y_ini1 + np.cos(rotanglex) * z_ini1
        vx_ini = vx_ini1
        vy_ini = np.cos(rotanglex) * vy_ini1 - np.sin(rotanglex) * vz_ini1
        vz_ini = np.sin(rotanglex) * vy_ini1 + np.cos(rotanglex) * vz_ini1
    else:
        x_ini = x_ini0
        y_ini = y_ini0
        z_ini = z_ini0
        vx_ini = vx_ini0
        vy_ini = vy_ini0
        vz_ini = vz_ini0
    print(f"Initial distance: {dist_cloud/AU:.1f} AU")
    print(f"Initial position vector: ({x_ini/AU:.1f}, {y_ini/AU:.1f}, {z_ini/AU:.1f}) AU")
    print(f"Initial speed: {speed/1e5:.2e} km/s")
    print(f"Initial velocity vector: ({vx_ini/1e5:.2e}, {vy_ini/1e5:.2e}, {vz_ini/1e5:.2e}) km/s")
    print(f"Impact parameter: {b_bcrit:.2f} + {b_bcrit_max:.2f} - {b_bcrit_min:.2f} b_crit = {b_bcrit*b_crit/AU:.2f} + {(b_bcrit*b_crit+r_cloud)/AU:.2f} - {(b_bcrit*b_crit-r_cloud)/AU:.2f} AU")
    print(f"Eccentricity: {np.sqrt(1+b_bcrit**2):.2f}")
    print(f"Radius of closest encounter: {r_close/AU:.1f} + {r_close_max/AU:.1f} - {r_close_min/AU:.1f} AU")
    print(f"Initial true anomaly: {trueanomaly/np.pi:.2f} pi")
    print(f"Specific angular momentum: {specific_am:.2e} + {specific_am_min:.2e} - {specific_am_max:.2e} cm^2/s")
    print(f"Keplerian radius: {kepler_radius_am/AU:.1f} + {kepler_radius_am_max/AU:.1f} - {kepler_radius_am_min/AU:.1f} AU")

def print_all(filenames, vars):
    for fname in filenames:
        if fname is None: continue
        print("+++", fname.stem + fname.suffix, "+++")
    for print_func in [print_tff, print_cloud, print_mass_ratios, print_cloud_orbit]:
        print("-----------------------------")
        print_func(vars)

def main():
    if len(sys.argv) < 2 or len(sys.argv) > 3: sys.exit(1)
    fargo_parfile = Path(sys.argv[1])
    supplement_parfile = None
    if len(sys.argv) == 3: supplement_parfile = Path(sys.argv[2])
    if not (fargo_parfile.exists() and fargo_parfile.is_file()): sys.exit(2)
    if supplement_parfile is not None and not (supplement_parfile.exists() and supplement_parfile.is_file()): sys.exit(2)
    vars = load_file(fargo_parfile)
    if supplement_parfile is not None:
        vars.update(load_file(supplement_parfile))
    print_all([fargo_parfile,supplement_parfile], vars)
    sys.exit(0)

if __name__ == "__main__":
    main()
