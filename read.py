import numpy as np
# from pathlib import Path


def get_domain_spherical(folder):
    """
    Read and save spherical coordinates from the FARGO domain files
    
    Inputs:
    ------
    folder:     Directory where the domain files are saved
    
    Outputs:
    -------
    domains:    Dictionary of theta, r, phi coordinates in cgs units
    """

    # Getting the spherical coordinate values from the FARGO3D domain files
    domains_min = {}
    real_to_fargo = {'phi': 'x', 'r':'y', 'theta':'z'}
    for key in ["r", "phi", "theta"]:
        domains_min[key] = np.loadtxt(folder / f"domain_{real_to_fargo[key]}.dat")
        if key != "phi": domains_min[key] = domains_min[key][3:-3] # Ghost cells

    # Since domain values are at cell interfaces, centering the domains
    domains = {}
    for key in domains_min.keys():
        domains[key] = (domains_min[key][1:] + domains_min[key][:-1]) / 2

    return domains


# Obtaining and reshaping physical quantity from scalar field files
def get_data(folder, quant, iter, domains):
    """
    Obtain physical quantity from FARGO3D's scalar fields e.g. dens
    
    Inputs:
    ------
    quant:    Scalar field keyword (str)
    iter:     Simulation snapshot (int)
    domains:  Centered domain data from domain*.dat files
    
    Outputs:
    -------
    data:     3D physical scalar field data
    """
    
    data = np.fromfile(folder / f"gas{quant}{iter}.dat").reshape(domains["theta"].size, domains["r"].size, domains["phi"].size)
    return data



def load_par_file(filepath):
    """
    Load the parameters in the simulation par file as a dictionary of param_name: param_value

    Inputs:
    ------
    filepath:    Path to the .par file

    Outputs:
    -------
    params:      dictionary of param_name: param_value (for e.g. "AspectRatio": 0.03799)
    """
    
    params = {}
    with open(filepath, 'r') as file:
        for line in file:
            line = line.strip()

            if not line or line.startswith('#'):
                continue  # Skip comments or empty lines

            parts = line.split(maxsplit=2)
            if len(parts) >= 2:
                key = parts[0]
                value_str = parts[1]
                try:
                    value = float(value_str)
                    if value.is_integer():
                        value = int(value)
                except ValueError:
                    value = value_str  # Keep as string if not numeric
                params[key] = value

    return params


def get_param_value(param_name, sim_name):
    """
    Obtains the value of a simulation parameter by parsing through {sim_name}/{sim_name}.par and setup_{setup_name}/{setup_name}.par files. {sim_name}.par corresponds to parameter file present in fargo3d/in whereas {setup_name}.par corresponds to parameter file present in fargo3d/setups. If a parameter name is present in both files, the value in {sim_name}.par takes precedence over the value in {setup_name}.par
    """

    sim_params = load_par_file(f"{sim_name}\\{sim_name}.par")
    setup_name = sim_params['Setup']
    setup_params = load_par_file(f"setup_{setup_name}/{setup_name}.par")
    param_value = setup_params[param_name]
    
    if param_name in sim_params.keys():
        param_value = sim_params[param_name]
    
    return param_value