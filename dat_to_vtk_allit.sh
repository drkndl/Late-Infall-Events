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
#SBATCH -J dat_to_vtk_allit

cd $SLURM_SUBMIT_DIR

if [ -z "$1" ]; then
  echo "Error: No input directory provided."
  echo "Usage: $0 <input_directory>"
  exit 1
fi

INPUT_DIR="$1"
echo "Input folder: $INPUT_DIR"
PARENT_DIR="$(dirname "$INPUT_DIR")"
OUTPUT_DIR="$PARENT_DIR/vtk_$(basename "$INPUT_DIR")"
PARAM_FILE="$(basename "$INPUT_DIR").par"
echo "Parameter file: $PARAM_FILE"
echo "Output folder: $OUTPUT_DIR"

mkdir -p "$OUTPUT_DIR"
cp "$INPUT_DIR/planet0.dat" "$OUTPUT_DIR/"
echo "Directory created" 

module purge
module load devel/miniforge
conda activate thesis

python ../thesis_code/cover_phi_wedge.py "../fargo3d/$INPUT_DIR"
echo "cover_phi_wedge implemented" 

conda deactivate
module purge
module load mpi/openmpi/4.1-gnu-13.3
module load devel/cuda/12.6

for i in $(seq 0 5 450); do
mpirun --bind-to core --map-by core -report-bindings --mca pml ucx --mca btl ^openib ./fargo3d -V $i -o "OutputDir=@$OUTPUT_DIR, Nx=102, Xmin=-3.1764985, Xmax=3.1764985" "./in/$PARAM_FILE"
done
echo "dat to vtk conversion complete"

rm -rf "$OUTPUT_DIR"/gas*.dat
rm -rf "$OUTPUT_DIR"/summary*.dat
echo "Deleted .dat files in the vtk directory"
