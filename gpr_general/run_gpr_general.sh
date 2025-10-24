#!/bin/bash
#SBATCH --job-name=gpr_general
#SBATCH --cpus-per-task=2
#SBATCH --ntasks=4
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --time=96:00:00
#SBATCH --mail-type=fail
#SBATCH --mail-type=end
#SBATCH --mail-user=hcheng19@sheffield.ac.uk

# Load the modules required by our program
module load Anaconda3/2022.10
source activate vanilla
module load OpenMPI/4.0.3-GCC-9.3.0
module load OpenBLAS/0.3.9-GCC-9.3.0
module load CFITSIO/3.48-GCCcore-9.3.0

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Source the Clik profile
source /users/smq24hc/cosmo/code/planck/clik/bin/clik_profile.sh

# Run Cobaya
srun --export=ALL -n ${SLURM_NTASKS} cobaya-run gpr_general.yaml

