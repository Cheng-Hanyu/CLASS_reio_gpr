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
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.stats import gaussian_kde

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
print("Parameter names in the chain:")
for i, p in enumerate(param_names):
    print(f"  {i:2d}: {p}")

################################################################################
# 2) Shuffle and downsample the chains
################################################################################
N_samples = 80000  # Reduced from 20000 to make the test run faster - increase as needed
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
# 3) For each z_c value and each sample, set the reionization parameters and run CLASS
################################################################################
# Define range of z_c values
n_zc = 5
z_c_values = np.linspace(20.0, 40.0, n_zc)

# We'll store all tau values for different z_c values
tau_lowz_all = np.zeros((n_zc, N_samples))
tau_highz_all = np.zeros((n_zc, N_samples))
tau_total_all = np.zeros(N_samples)  # tau_total doesn't depend on z_c

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

# Loop over different z_c values
for iz_c, z_c in enumerate(z_c_values):
    print(f"Processing z_c = {z_c:.1f}")
    
    for i, idx in enumerate(indices):
        # Progress indicator
        if i % 50 == 0:
            print(f"  Processing sample {i+1}/{N_samples}")
            
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
            'z_c'           : z_c,  # Using the current z_c value
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
        # 3(d) Get the thermodynamics output and store tau values
        # --------------------------------------------------------------------------
        # Get tau_lowz and tau_highz from thermodynamics
        tau_lowz_all[iz_c, i] = cosmo.tau_lowz() 
        tau_highz_all[iz_c, i] = cosmo.tau_highz() 
        
        # Get tau_total from thermodynamics (only needs to be calculated once per sample)
        if iz_c == 0:  # Only calculate tau_total once per sample
            tau_total_all[i] = cosmo.tau_total()

        # Clean up
        cosmo.struct_cleanup()
        cosmo.empty()

################################################################################
# 4) Calculate the 68% confidence level constraints for each z_c
################################################################################
# Store the confidence intervals
tau_lowz_constraints = np.zeros((n_zc, 5))  # [central, lower68, upper68, lower95, upper95]
tau_highz_constraints = np.zeros((n_zc, 2))  

# Calculate for each z_c
for iz_c in range(n_zc):
    # For tau_lowz - calculate both 68% and 95% CL
    q_lowz = weighted_quantile(tau_lowz_all[iz_c], [0.025, 0.16, 0.5, 0.84, 0.975], weights=weights_subset)
    tau_lowz_constraints[iz_c] = [q_lowz[2], q_lowz[1], q_lowz[3], q_lowz[0], q_lowz[4]]  # [central, lower68, upper68, lower95, upper95]
    
    # For tau_highz - calculate upper limits only (68% and 95% CL)
    q_highz = weighted_quantile(tau_highz_all[iz_c], [0.68, 0.95], weights=weights_subset)
    tau_highz_constraints[iz_c] = [q_highz[0], q_highz[1]]  # [upper68, upper95]

    # Print the constraints
    print(f"\nFor z_c = {z_c_values[iz_c]:.1f}:")
    print(f"  tau_lowz  = {tau_lowz_constraints[iz_c, 0]:.4f} +{tau_lowz_constraints[iz_c, 2] - tau_lowz_constraints[iz_c, 0]:.4f} -{tau_lowz_constraints[iz_c, 0] - tau_lowz_constraints[iz_c, 1]:.4f} (68% CL)")
    print(f"  tau_lowz  = {tau_lowz_constraints[iz_c, 0]:.4f} +{tau_lowz_constraints[iz_c, 4] - tau_lowz_constraints[iz_c, 0]:.4f} -{tau_lowz_constraints[iz_c, 0] - tau_lowz_constraints[iz_c, 3]:.4f} (95% CL)")
    print(f"  tau_highz 68% CL upper limit: < {tau_highz_constraints[iz_c, 0]:.4f}")
    print(f"  tau_highz 95% CL upper limit: < {tau_highz_constraints[iz_c, 1]:.4f}")

# Calculate constraints for tau_total (68% and 95% CL)
q_total = weighted_quantile(tau_total_all, [0.025, 0.16, 0.5, 0.84, 0.975], weights=weights_subset)
tau_total_constraints = [q_total[2], q_total[1], q_total[3], q_total[0], q_total[4]]  # [central, lower68, upper68, lower95, upper95]

print(f"\nFor tau_total:")
print(f"  tau_total = {tau_total_constraints[0]:.4f} +{tau_total_constraints[2] - tau_total_constraints[0]:.4f} -{tau_total_constraints[0] - tau_total_constraints[1]:.4f} (68% CL)")
print(f"  tau_total = {tau_total_constraints[0]:.4f} +{tau_total_constraints[4] - tau_total_constraints[0]:.4f} -{tau_total_constraints[0] - tau_total_constraints[3]:.4f} (95% CL)")

# -----------------------------------------------------------------------------
# Save raw tau arrays to ASCII for post‑processing / verification
# -----------------------------------------------------------------------------
np.savetxt("tau_lowz_all.txt", tau_lowz_all)
np.savetxt("tau_highz_all.txt", tau_highz_all)
np.savetxt("tau_total_all.txt", tau_total_all)

################################################################################
# 5) Plot the posterior distributions with gaussian_kde smoothing and normalized density
################################################################################
# Setup the figure for tau_lowz and tau_highz posterior distributions
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Define a colormap for different z_c values
cmap = plt.cm.viridis
norm = Normalize(vmin=z_c_values.min(), vmax=z_c_values.max())
sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

# Determine the evaluation range for the KDEs
x_lowz = np.linspace(tau_lowz_all.min(), tau_lowz_all.max(), 1000)
x_highz = np.linspace(tau_highz_all.min(), tau_highz_all.max(), 1000)

# Plot normalized PDFs for each z_c value
for iz_c, z_c in enumerate(z_c_values):
    color_val = 0.2 + 0.6 * norm(z_c)  # Maps to darker portion
    color = cmap(color_val)
    
    # For tau_lowz
    kde_lowz = gaussian_kde(tau_lowz_all[iz_c], weights=weights_subset)
    pdf_lowz = kde_lowz(x_lowz)
    normalized_pdf_lowz = pdf_lowz / pdf_lowz.max()  # Normalize to max value = 1
    axes[0].plot(x_lowz, normalized_pdf_lowz, color=color, label=f'$z_c={z_c:.1f}$')

    # For tau_highz with reflection method
    reflected_tau_highz = np.concatenate([tau_highz_all[iz_c], -tau_highz_all[iz_c]])
    weights_extended = np.concatenate([weights_subset, weights_subset])
    kde_highz = gaussian_kde(reflected_tau_highz, weights=weights_extended)
    pdf_highz = kde_highz(x_highz)
    normalized_pdf_highz = pdf_highz / pdf_highz.max()  # Normalize to max value = 1
    axes[1].plot(x_highz, normalized_pdf_highz, color=color)

# Set labels and titles
axes[0].set_xlabel(r'$\tau_{\mathrm{lowz}}$', fontsize=28)
axes[0].set_ylabel('$P/P_{\mathrm{max}}$', fontsize=28)  # Changed to normalized probability
#axes[0].set_title(r'Posterior distribution of $\tau_{\mathrm{lowz}}$', fontsize=16)
axes[0].grid(False)

axes[1].set_xlabel(r'$\tau_{\mathrm{highz}}$', fontsize=28)
axes[1].set_ylabel(r'$P/P_{\mathrm{max}}$', fontsize=28)  # Changed to normalized probability
#axes[1].set_title(r'Posterior distribution of $\tau_{\mathrm{highz}}$', fontsize=16)
axes[1].set_xlim(0, 0.3)
axes[1].grid(False)

# Increase tick label font size for both subplots
axes[0].tick_params(axis='both', which='major', labelsize=23)
axes[1].tick_params(axis='both', which='major', labelsize=23)

# Add a colorbar
#cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
#cbar = plt.colorbar(sm, cax=cbar_ax)
#cbar.set_label('$z_c$', fontsize=14)

# Add legend with consistent font to both subplots
axes[0].legend(fontsize=21, loc='best')
# Make the legend for the second subplot using the same handles and labels as the first
handles, labels = axes[0].get_legend_handles_labels()
axes[1].legend(handles, labels, fontsize=21, loc='best')

plt.tight_layout(rect=[0, 0, 0.9, 1])  # Make room for the colorbar
plt.savefig('tau_p_zc.png', dpi=300)
#plt.show()

################################################################################
# 6) Plot individual figure for tau_highz at z_c = 30 with thick black line
################################################################################
# Find the index for z_c = 30
iz_c_30 = np.argmin(np.abs(z_c_values - 30.0))

fig2, ax2 = plt.subplots(1, 1, figsize=(8, 7))

# For tau_highz at z_c = 30 with reflection method
reflected_tau_highz_30 = np.concatenate([tau_highz_all[iz_c_30], -tau_highz_all[iz_c_30]])
weights_extended_30 = np.concatenate([weights_subset, weights_subset])
kde_highz_30 = gaussian_kde(reflected_tau_highz_30, weights=weights_extended_30)
pdf_highz_30 = kde_highz_30(x_highz)
normalized_pdf_highz_30 = pdf_highz_30 / pdf_highz_30.max()

ax2.plot(x_highz, normalized_pdf_highz_30, color='black', linewidth=3)

ax2.set_xlabel(r'$\tau_{\mathrm{highz}}$', fontsize=28)
ax2.set_ylabel(r'$P/P_{\mathrm{max}}$', fontsize=28)
ax2.set_xlim(0, 0.3)
ax2.grid(False)
ax2.tick_params(axis='both', which='major', labelsize=23)

plt.tight_layout()
plt.savefig('tau_highz_zc30.png', dpi=300)
#plt.show()

################################################################################
# 7) Plot individual figure for tau_total with green color
################################################################################
fig3, ax3 = plt.subplots(1, 1, figsize=(8, 7))

# Determine the evaluation range for tau_total KDE
x_total = np.linspace(tau_total_all.min(), tau_total_all.max(), 1000)

# For tau_total
kde_total = gaussian_kde(tau_total_all, weights=weights_subset)
pdf_total = kde_total(x_total)
normalized_pdf_total = pdf_total / pdf_total.max()

ax3.plot(x_total, normalized_pdf_total, color='green', linewidth=2)

ax3.set_xlabel(r'$\tau_{\mathrm{total}}$', fontsize=28)
ax3.set_ylabel(r'$P/P_{\mathrm{max}}$', fontsize=28)
ax3.set_xlim(0, 0.3)
ax3.grid(False)
ax3.tick_params(axis='both', which='major', labelsize=23)

plt.tight_layout()
plt.savefig('tau_total.png', dpi=300)
#plt.show()

print("Analysis complete. All figures saved.")
print("Figures saved:")
print("1. tau_p_zc.png - Original 1x2 subplot with tau_lowz and tau_highz for all z_c")
print("2. tau_highz_zc30.png - tau_highz for z_c = 30 with thick black line")
print("3. tau_total.png - tau_total distribution in green")
