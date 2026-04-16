#!/bin/bash
#SBATCH --job-name=Spectra_Heal_vit_48L
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00
#SBATCH --output=/storage/bharat/log_files/48L_spectra_output_%j.log
#SBATCH --error=/storage/bharat/log_files/48L_spectra_error_%j.log
#SBATCH --exclude=cn3
#SBATCH --mail-type=ALL
#SBATCH --mail-user=bharat.20221074@students.iiserpune.ac.in

module load anaconda3
export PYTHONPATH="$(pwd)"
source activate /storage/bharat/envs/Heal_vit

PYTHON_PATH=/storage/bharat/envs/Heal_vit/bin/python

echo "Starting Zonal Energy Spectrum evaluation..."
torchrun --nnodes=1 --nproc_per_node=2 Power_Spectra/evaluate_spectra.py

echo "Generating spectra plots..."
$PYTHON_PATH Power_Spectra/plot_spectra.py

echo "Job completed."
