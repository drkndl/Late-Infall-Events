#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=2
#SBATCH --mem=16000
#SBATCH --gres=gpu:a30:2
#SBATCH --time=14-00:00:00
#SBATCH --partition=gpu
#SBATCH --output=%x.o%j
#SBATCH --error=%x.e%j
#SBATCH --mail-type=ALL
#SBATCH --mail-user=drishika.nadella@stud.uni-heidelberg.de
#SBATCH -J cloudlet_lowres_it450

cd $SLURM_SUBMIT_DIR

module purge
module load mpi/openmpi/4.1-gnu-13.3
module load devel/cuda/12.6 

mkdir -p outputs/iras04125_lowres_it450/monitor/{gas,tracer}
mpirun --bind-to core --map-by core -report-bindings --mca pml ucx --mca btl ^openib ./fargo3d ./in/iras04125_lowres_it450.par
