#PBS -l nodes=2:ppn=4:gpus=4
#PBS -l walltime=72:00:00
#PBS -q gpu
#PBS -S /bin/bash
#PBS -j oe
#PBS -m bea
#PBS -M drishika.nadella@stud.uni-heidelberg.de

module purge
module load mpi/openmpi/3.1-gnu-8.3

mkdir -p outputs/cloudlet/monitor/{gas,tracer}
mpirun --bind-to core --map-by core -report-bindings ./fargo3d /in/iras04125_c7_highmass_lowRes.par
