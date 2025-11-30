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
    pizza:      3D data of shape (ntheta, nr, nphi) which is now without a missing wedge in phi
    pizza_phi:  1D array of phi values without a missing wedge
    """

    nth,nr,nph = data.shape
    assert nth == len(theta)
    assert nr == len(r)
    assert nph == len(phi)
    
    # To ensure that we have no "empty wedge" near the extrema of phi:
    data_ext           = np.zeros((nth,nr,nph+2))
    data_ext[:,:,1:-1] = data[:,:,:]
    data_ext[:,:,0]    = data_ext[:,:,-2]
    data_ext[:,:,-1]   = data_ext[:,:,1]
    phi_ext            = np.zeros(nph+2)
    phi_ext[1:-1]      = phi[:]
    phi_ext[0]         = phi_ext[-2] - 2*np.pi
    phi_ext[-1]        = phi_ext[1]  + 2*np.pi

    return data_ext, phi_ext


def main():

    cover_phi_wedge()
    

if __name__ == "__main__":
    main()