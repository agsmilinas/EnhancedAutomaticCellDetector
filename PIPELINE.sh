#!/bin/bash
#SBATCH --export=ALL,DISABLE_DCGM=1
#SBATCH --account=def-aevans
#SBATCH --cpus-per-task=8
#SBATCH --mem=512G               # memory per node
#SBATCH --time=06:00:00
#SBATCH --gpus-per-node=a100:4


echo "starting neuron detection pipeline (BB Hippocampus 1 micron)..."
# Force unload all modules and then load the required modules in the correct sequence
module --force purge
module load StdEnv/2020
module load python/3.10.2

echo "Modules loaded successfully."

# Create and activate the virtual environment using Python's built-in venv
echo "Creating virtual environment..."
python -m venv $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
echo "Virtual environment activated."

# Upgrade pip and install necessary Python packages
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
echo "Dependencies installed successfully."

#GPU modules
module load cuda
nvidia-smi
cd ../../../BB_NARVAL/NeuronsDetection/scripts/2024/JULY/scripts
ls

export RESULTS_DIR="../results"
export NPY_PATH="../data/hippo4549_left.mnc.npy"
export PNG_UPSAMPLED_PATH="$RESULTS_DIR/upsampled_hippo4549_left.png"
export NPY_UPSAMPLED_PATH="$RESULTS_DIR/upsampled_hippo4549_left.npy"
export ANISOF_NPY_PATH="$RESULTS_DIR/anisof_hippo4549_left.npy"
export PTS_ANISOF_NPY_PATH="$RESULTS_DIR/pts_anisof_hippo4549_left.npy"
export RES_PTS_ANISOF_PNG_PATH="$RESULTS_DIR/RES_hippo4549_left.png"
export CV2_RES_PTS_ANISOF_PNG_PATH="$RESULTS_DIR/CV2_RES_hippo4549_left.png"
export WATERSHED_IMG_PATH="$RESULTS_DIR/watershed_anisof_hippo4549_left.png"
export BINARY_WATERSHED_IMG_PATH="$RESULTS_DIR/binary_watershed_colors_anisof_hippo4549_left.png"
export ANISOF_ITERATIONS="10"

echo "running 0_npy_png_upsample.py"
python 0_npy_png_upsample.py
echo "finished running 0_npy_png_upsample ...."

echo "running 1_parallelized_anisof.py"
python 1_parallelized_anisof.py
echo "finished running 1_parallelized_anisof.py ...."

echo "Running 2_parallelBatchesGPUS.py"
python 2_parallelBatchesGPUS.py
echo "finished running 2_parallelBatchesGPUS.py ...."

module load StdEnv/2023
module load gcc/12.3
module load opencv/4.9.0


echo "running 3_genHighResPNG.py"
python 3_getHighResPNG.py
echo "finished running 3_genHighResPNG.py"

echo "running 4_watershed.py"
python 4_watershed.py

echo "finished running neuron detection pipeline (BB Hippocampus 1 micron)..."
# Optionally, deactivate the virtual environment if more steps follow this.
deactivate
