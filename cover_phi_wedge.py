import numpy as np
from pathlib import Path
from read import get_domain_spherical, get_data


def cover_phi_wedge(data, theta, r, phi):
    """
    Covers up a wedge of phi that is missing from the disk (thanks to my low resolution sims) which makes it look like somebody had a tiny slice of disk pizza. This function makes the disk pizza whole again.
    
    Inputs:
    ------
    data:       3D data of shape (ntheta, nr, nphi)
    phi:        1D array of phi values

    Outputs:
    -------
    data_ext:   3D data of shape (ntheta, nr, nphi) which is now without a missing wedge in phi
    phi_ext:    1D array of phi values without a missing wedge
    """

    nth,nr,nph = data.shape
    assert nth == len(theta)
    assert nr == len(r)
    assert nph == len(phi)
    
    # To ensure that we have no "empty wedge" near the extrema of phi 
    data_ext           = np.zeros((nth,nr,nph+2))
    data_ext[:,:,1:-1] = data[:,:,:]
    data_ext[:,:,0]    = data_ext[:,:,-2]
    data_ext[:,:,-1]   = data_ext[:,:,1]
    phi_ext            = np.zeros(nph+2)
    phi_ext[1:-1]      = phi[:]
    phi_ext[0]         = phi_ext[-2] - 2*np.pi
    phi_ext[-1]        = phi_ext[1]  + 2*np.pi

    return data_ext, phi_ext


def write_dat(folder, quant, iter, data_ext):
    """
    Creates a new dat file with the extended data 

    Inputs:
    ------
    folder:   
    quant:      Scalar field keyword (str)
    iter:       Simulation snapshot (int)
    data_ext:   Full pizza data (ntheta, nr, nphi+2)
    """

    np.ascontiguousarray(data_ext, dtype=np.float64).tofile(folder / f"gas{quant}{iter}.dat")
    return 


def main():

    ###################### Load data (theta = 175, r = 150, phi = 100) ################################

    # folder = Path("../cloud_disk_it450_retro_rotX45/")                            # Folder with the output files
    folder = Path("../fargo3d/outputs/cloud_disk_it450_retro_rotX45")               # Folder with the output files (BinAC2)
    ext_folder = Path("../fargo3d/outputs/cloud_disk_it450_retro_rotX45_vtk/")      # Folder to save images
    it = 450                                                                  # FARGO snapshot of interest
    domains = get_domain_spherical(folder)
    rho = get_data(folder, "dens", it, domains)         # Load 3D array of density values
    vphi = get_data(folder, "vx", it, domains)          # Load 3D array of azimuthal velocities v_phi
    vrad = get_data(folder, "vy", it, domains)          # Load 3D array of radial velocities v_rad
    vthe = get_data(folder, "vz", it, domains)          # Load 3D array of colatitude velocities v_theta
    energy = get_data(folder, "energy", it, domains)    # Load 3D array of energies 

    #############################   Make azimuthal pizza full #########################################

    rho_ext, phi_ext = cover_phi_wedge(rho, domains["theta"], domains["r"], domains["phi"])      # Cover empty wedge in dens
    vphi_ext, phi_ext = cover_phi_wedge(vphi, domains["theta"], domains["r"], domains["phi"])    # Cover empty wedge in vx
    vrad_ext, phi_ext = cover_phi_wedge(vrad, domains["theta"], domains["r"], domains["phi"])    # Cover empty wedge in vy
    vthe_ext, phi_ext = cover_phi_wedge(vthe, domains["theta"], domains["r"], domains["phi"])    # Cover empty wedge in vz 
    E_ext, phi_ext = cover_phi_wedge(energy, domains["theta"], domains["r"], domains["phi"])     # Cover empty wedge in energy 

    #################################### Write new data ################################################

    write_dat(ext_folder, "dens", it, rho_ext)       # Write 3D array of densities into dat file (nth, nr, nphi+2)
    write_dat(ext_folder, "vx", it, vphi_ext)        # Write 3D array of vx into dat file (nth, nr, nphi+2)
    write_dat(ext_folder, "vy", it, vrad_ext)        # Write 3D array of vy into dat file (nth, nr, nphi+2)
    write_dat(ext_folder, "vz", it, vthe_ext)        # Write 3D array of vz into dat file (nth, nr, nphi+2)
    write_dat(ext_folder, "energy", it, E_ext)       # Write 3D array of energy into dat file (nth, nr, nphi+2)
    

if __name__ == "__main__":
    main()