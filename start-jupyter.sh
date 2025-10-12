#!/bin/bash
#SBATCH --nodes=1
#SBATCH --tasks-per-node=8
#SBATCH --cpus-per-task=2
#SBATCH --mem=256000
#SBATCH --partition=interactive
#SBATCH --time=6:00:00
#SBATCH -J jupyter
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=drishika.nadella@stud.uni-heidelberg.de
#SBATCH --output=%x.o%j

cd $SLURM_SUBMIT_DIR
module load devel/miniforge
conda activate thesis
jupyter server --no-browser --port=8876 --ip 0.0.0.0
