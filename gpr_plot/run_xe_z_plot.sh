#!/bin/bash
#SBATCH --job-name=xe_z_plot
#SBATCH --output=/users/smq24hc/cosmo/gpr_plot/xe_z_plot_%j.out   # stdout
#SBATCH --error=/users/smq24hc/cosmo/gpr_plot/xe_z_plot_%j.err    # stderr
#SBATCH --nodes=1
#SBATCH --ntasks=1                  # a single Python process is enough here
#SBATCH --cpus-per-task=2
#SBATCH --time=96:00:00
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=hcheng19@sheffield.ac.uk

###############################################################################
# 1. Load software environment
###############################################################################
module load Anaconda3/2022.10
source activate vanilla

module load OpenMPI/4.0.3-GCC-9.3.0
module load OpenBLAS/0.3.9-GCC-9.3.0
module load CFITSIO/3.48-GCCcore-9.3.0

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

# Planck 'clik' (keep if you really need it)
source /users/smq24hc/cosmo/code/planck/clik/bin/clik_profile.sh

###############################################################################
# 2. Make CLASS (classy) visible to Python
###############################################################################
export CLASS_DIR=/users/smq24hc/cosmo/code/classy_gp
export PYTHONPATH="$CLASS_DIR/python:${PYTHONPATH}"
export LD_LIBRARY_PATH="$CLASS_DIR:${LD_LIBRARY_PATH}"

###############################################################################
# 3. Run the script
###############################################################################
WORKDIR=/users/smq24hc/cosmo/gpr_plot
cd "$WORKDIR"

# Redirect the program’s own prints into a separate log if you like
python xe_z_plot.py
