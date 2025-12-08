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
#SBATCH -J dattovtk_450_cloud_disk_it450_retro_rotX45

cd $SLURM_SUBMIT_DIR

mkdir -p outputs/cloud_disk_it450_retro_rotX45_vtk
cp outputs/cloud_disk_it450_retro_rotX45/planet0.dat outputs/cloud_disk_it450_retro_rotX45_vtk/
echo "Directory created" 

module purge
module load devel/miniforge
conda activate thesis

python ../thesis_code/cover_phi_wedge.py
echo "cover_phi_wedge implemented" 

conda deactivate
module purge
module load mpi/openmpi/4.1-gnu-13.3
module load devel/cuda/12.6

mkdir -p outputs/cloud_disk_it450_retro_rotX45_vtk/monitor/{gas,tracer}
mpirun --bind-to core --map-by core -report-bindings --mca pml ucx --mca btl ^openib ./fargo3d -V 450 -o "OutputDir=@outputs/cloud_disk_it450_retro_rotX45_vtk, Nx=102" ./in/cloud_disk_it450_retro_rotX45.par
