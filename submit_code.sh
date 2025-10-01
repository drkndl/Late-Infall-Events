#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=2
#SBATCH --mem=16000
#SBATCH --time=14-00:00:00
#SBATCH --partition=compute
#SBATCH --output=%x.o%j
#SBATCH --error=%x.e%j
#SBATCH --mail-type=ALL
#SBATCH --mail-user=drishika.nadella@stud.uni-heidelberg.de
#SBATCH -J submit_code

cd $SLURM_SUBMIT_DIR

module purge
module load devel/miniforge
conda activate thesis

python mass_vs_r.py