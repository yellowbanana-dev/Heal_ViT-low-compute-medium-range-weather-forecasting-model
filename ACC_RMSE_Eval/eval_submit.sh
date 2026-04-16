#!/bin/bash
#SBATCH --job-name=Eval_Heal_vit_48L
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00
#SBATCH --output=/storage/bharat/log_files/48L_eval_output_%j.log
#SBATCH --error=/storage/bharat/log_files/48L_eval_error_%j.log
#SBATCH --exclude=cn3
#SBATCH --mail-type=ALL
#SBATCH --mail-user=bharat.20221074@students.iiserpune.ac.in

module load anaconda3
export PYTHONPATH="$(pwd)"
source activate /storage/bharat/envs/Heal_vit

PYTHON_PATH=/storage/bharat/envs/Heal_vit/bin/python

echo "Starting evaluation..."
torchrun --nnodes=1 --nproc_per_node=2 ACC_RMSE_Eval/evaluate_variables.py

echo "Evaluation finished! Generating plots..."
$PYTHON_PATH ACC_RMSE_Eval/plot_metrics.py
$PYTHON_PATH ACC_RMSE_Eval/plot_spatial_bias.py

echo "Job completed."
