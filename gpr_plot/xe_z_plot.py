# At the top of your script
import importlib.util
import sys

# Explicitly load the module from the specific file
spec = importlib.util.spec_from_file_location(
    "classy", 
    "/users/smq24hc/cosmo/code/classy_gp/python/build/lib.linux-x86_64-cpython-39/classy.cpython-39-x86_64-linux-gnu.so"
)
classy = importlib.util.module_from_spec(spec)
sys.modules["classy"] = classy
spec.loader.exec_module(classy)
Class = classy.Class

import numpy as np
import matplotlib.pyplot as plt
import os


plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'cm'  # Use Computer Modern font for math text

# For interpolation
import scipy.interpolate as interp

# For reading MCMC chains
import getdist
from getdist import plots, MCSamples

################################################################################
# 1) Load the chains from GetDist
################################################################################
# Path to your chain directory:
path_to_chain_gprgeneral = '/users/smq24hc/cosmo/gpr_general/gpr_general'

# Load samples (ignore the first 30% by default)
gprgeneral_samples = getdist.loadMCSamples(
    path_to_chain_gprgeneral, 
    settings={'ignore_rows': 0.3}
)

# Get the underlying raw chain data as a NumPy array
# `getSamples()` will have shape (N_points, N_parameters).
all_samples = gprgeneral_samples.samples

# And get the associated weights:
all_weights = gprgeneral_samples.weights

# Optional: check the parameter names to confirm the indexing
param_names = gprgeneral_samples.getParamNames().list()

################################################################################
# 2) Shuffle and downsample the chains
################################################################################
N_samples = 300000  # Reduced from 20000 to make the test run faster - increase as needed
N_total = all_samples.shape[0]
indices = np.arange(N_total)
np.random.shuffle(indices)

if N_total > N_samples:
    indices = indices[:N_samples]
else:
    N_samples = N_total  # Use all if you have fewer than N_samples

# We'll keep track of the relevant subset of samples and weights
samples_subset = all_samples[indices]
weights_subset = all_weights[indices]

################################################################################
# A helper function to compute weighted quantiles
################################################################################
def weighted_quantile(values, quantiles, weights=None):
    """
    Compute (weighted) quantiles from a 1D numpy array.
    """
    values = np.array(values, dtype=float)
    quantiles = np.array(quantiles, dtype=float)

    if weights is None:
        weights = np.ones_like(values)
    else:
        weights = np.array(weights, dtype=float)

    # Sort values and weights together by values
    sorter = np.argsort(values)
    values_sorted = values[sorter]
    weights_sorted = weights[sorter]

    # Normalized cumulative weights from 0 to 1
    cumulative_weights = np.cumsum(weights_sorted)
    cumulative_weights /= cumulative_weights[-1]

    # Interpolate quantiles
    results = np.zeros_like(quantiles, dtype=float)
    for i, q in enumerate(quantiles):
        # index of first cume_weight >= q
        idx = np.searchsorted(cumulative_weights, q)
        if idx == 0:
            results[i] = values_sorted[0]
        elif idx >= len(values_sorted):
            results[i] = values_sorted[-1]
        else:
            # linear interpolation
            t = ((q - cumulative_weights[idx-1]) /
                 (cumulative_weights[idx] - cumulative_weights[idx-1]))
            results[i] = values_sorted[idx-1] + t*(values_sorted[idx] - values_sorted[idx-1])

    return results

################################################################################
# 3) For each sample, set the reionization parameters in CLASS and run
################################################################################
# Define z_c
z_c = 30.0  # You can change this to your preferred value

# We'll store all the x_e(z) curves here
zmax = 800
nz = 1000
z_array = np.linspace(0, zmax, nz)
xe_matrix = np.zeros((N_samples, nz))

# Arrays to store tau_lowz and tau_highz
tau_lowz_values = np.zeros(N_samples)
tau_highz_values = np.zeros(N_samples)

# Some indices in your chain for convenience:
idx_zs = [param_names.index(f'z{i+1}') for i in range(20)]  # z1..z20
idx_xe1   = param_names.index('xe1')
idx_xe2   = param_names.index('xe2')
idx_logxe = [param_names.index(f'log_xe{i+3}') for i in range(18)]  # log_xe3..log_xe20

idx_gpr_step_sharp = param_names.index('gpr_reio_step_sharpness')
idx_sigma_f        = param_names.index('gpr_sigma_f')
idx_l              = param_names.index('gpr_l')
idx_z_min          = param_names.index('gpr_z_min')
idx_z_max          = param_names.index('gpr_z_max')
idx_z_transition   = param_names.index('gpr_z_transition')
idx_n_low_float    = param_names.index('n_low_float')
idx_n_high_float   = param_names.index('n_high_float')

for i, idx in enumerate(indices):
    # Progress indicator
    if i % 10 == 0:
        print(f"Processing sample {i+1}/{N_samples}")
        
    # --------------------------------------------------------------------------
    # 3(a) Extract the GPR parameters from the chain
    # --------------------------------------------------------------------------
    chain_entry   = samples_subset[i]
    chain_zvals   = chain_entry[idx_zs]  # z1..z20
    chain_xe1     = chain_entry[idx_xe1]
    chain_xe2     = chain_entry[idx_xe2]
    chain_logvals = chain_entry[idx_logxe]  # log_xe3..log_xe20

    gpr_reio_step_sharpness = chain_entry[idx_gpr_step_sharp]
    gpr_sigma_f             = chain_entry[idx_sigma_f]
    gpr_l                   = chain_entry[idx_l]
    gpr_z_min               = chain_entry[idx_z_min]
    gpr_z_max               = chain_entry[idx_z_max]
    gpr_z_transition        = chain_entry[idx_z_transition]
    n_low_float             = chain_entry[idx_n_low_float]
    n_high_float            = chain_entry[idx_n_high_float]

    # Build the comma-separated strings needed by the GPR-ified CLASS
    z_list_str = ",".join([f"{zv:.3f}" for zv in chain_zvals])
    xe_list = [chain_xe1, chain_xe2] + [10**lv for lv in chain_logvals]
    xe_list_str = ",".join([f"{xv:.6f}" for xv in xe_list])

    # Convert the float to an int for n_low, n_high:
    gpr_n_low  = int(n_low_float)
    gpr_n_high = int(n_high_float)

    # --------------------------------------------------------------------------
    # 3(b) Prepare the full cosmological parameter dictionary
    #     (baseline + GPR reionization)
    # --------------------------------------------------------------------------
    cosmo_params = {
        # Baseline cosmology
        'H0'         : 67.27,
        'omega_b'    : 0.02236,
        'omega_cdm'  : 0.1202,
        'A_s'        : 2.101e-9,
        'n_s'        : 0.9649,

        # GPR-based reionization
        'reio_parametrization': 'reio_gpr_tanh',
        'gpr_reio_num'        : 20,
        'gpr_reio_z'          : z_list_str,
        'gpr_reio_xe'         : xe_list_str,
        'gpr_reio_step_sharpness': gpr_reio_step_sharpness,
        'gpr_sigma_f'         : gpr_sigma_f,
        'gpr_l'               : gpr_l,

        # Ranges and discretization
        'gpr_z_min'       : gpr_z_min,
        'gpr_z_max'       : gpr_z_max,
        'gpr_z_transition': gpr_z_transition,
        'gpr_n_low'       : gpr_n_low,
        'gpr_n_high'      : gpr_n_high,

        # Add z_c and z_max for tau calculation
        'z_c'           : z_c,
        'z_max'         : 800.0,  # Set z_max for high-z range

        # Output setup
        'output'         : 'tCl',
        'lensing'        : 'no',
        'l_max_scalars'  : 2,
        'non_linear'     : 'none',

        # Neutrinos, etc. 
        'N_ncdm': 1,
        'N_ur'  : 2.0328,
    }

    # --------------------------------------------------------------------------
    # 3(c) Run CLASS
    # --------------------------------------------------------------------------
    cosmo = Class()
    cosmo.set(cosmo_params)
    cosmo.compute()

    # --------------------------------------------------------------------------
    # 3(d) Get the thermodynamics output and interpolate x_e(z)
    # --------------------------------------------------------------------------
    thermo = cosmo.get_thermodynamics()   # dictionary with keys 'z', 'x_e', ...
    #tau_reio = cosmo.tau_reio() 
    #print(tau_reio)
    #tau_high = cosmo.tau_highz() 
    #print(tau_high)
    z_th   = thermo['z']
    xe_th  = thermo['x_e']

    # CLASS typically returns arrays from high z -> low z, so reverse if needed
    if z_th[0] > z_th[-1]:
        z_th  = z_th[::-1]
        xe_th = xe_th[::-1]

    # Interpolate x_e(z) in linear space of z:
    f_xe = interp.interp1d(z_th, xe_th, kind='linear',
                           bounds_error=False, fill_value='extrapolate')

    xe_matrix[i, :] = f_xe(z_array)

    # Clean up
    cosmo.struct_cleanup()
    cosmo.empty()

################################################################################
# 4) Compute median, 68% band, and 95% band of x_e(z) at each z, using weights
################################################################################
quantiles_to_get = [0.025, 0.16, 0.50, 0.84, 0.975]

xe_median   = np.zeros(nz)
xe_lower_68 = np.zeros(nz)
xe_upper_68 = np.zeros(nz)
xe_lower_95 = np.zeros(nz)
xe_upper_95 = np.zeros(nz)

for iz in range(nz):
    qvals = weighted_quantile(
        xe_matrix[:, iz],
        quantiles=quantiles_to_get,
        weights=weights_subset
    )
    (xe_lower_95[iz],
     xe_lower_68[iz],
     xe_median[iz],
     xe_upper_68[iz],
     xe_upper_95[iz]) = qvals

################################################################################
# 6) Plot the results
################################################################################
# 6.1) Plot and save x_e(z) reconstruction
plt.figure(figsize=(12, 7))
plt.fill_between(z_array, xe_lower_95, xe_upper_95, alpha=0.2, color='blue', label='95% CL')
plt.fill_between(z_array, xe_lower_68, xe_upper_68, alpha=0.4, color='blue', label='68% CL')
plt.plot(z_array, xe_median, 'b-', label='Median')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'Redshift, z', fontsize=30)
plt.ylabel(r'Ionization fraction, $X_e$', fontsize=30)
# Increase tick label font size
plt.tick_params(axis='both', which='major', labelsize=24)

# Ensure consistent font in legend
plt.legend(fontsize=23, loc='best')
plt.grid(False)
plt.tight_layout()
plt.savefig('xe_z.png', dpi=300)
# Don't call plt.show() when running on HPC as it will cause the script to wait for user input
# plt.show()  

np.savetxt('xe_z_quantiles.txt',
           np.column_stack([z_array,
                            xe_lower_95, xe_lower_68, xe_median,
                            xe_upper_68, xe_upper_95]),
           header='z  xe_lo95  xe_lo68  xe_med  xe_hi68  xe_hi95')

print("Analysis complete. Results saved to current directory.")
