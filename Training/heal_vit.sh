#!/bin/bash
#SBATCH --job-name=Heal_vit_48L
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --time=7-23:59:59
#SBATCH --mem=200G
#SBATCH --exclude=cn3
#SBATCH --output=/storage/bharat/log_files/48L_test_output_%j.log  # Save logs here
#SBATCH --error=/storage/bharat/log_files/48L_test_error_%j.log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=bharat.20221074@students.iiserpune.ac.in

# Load Anaconda module
module load anaconda3
export PYTHONPATH="$(pwd)"
# Activate the cloned Conda environment in storage
source activate /storage/bharat/envs/Heal_vit

# 4. Explicitly use the environment's python path to be 100% safe
PYTHON_PATH=/storage/bharat/envs/Heal_vit/bin/python
echo "Using Python located at: $PYTHON_PATH"
# Verify exactly 2 GPUs are available
AVAILABLE_GPUS=$($PYTHON_PATH -c "import torch; print(torch.cuda.device_count())")
if [ "$AVAILABLE_GPUS" -ne 2 ]; then
    echo "CRITICAL ERROR: Requested 2 GPUs but PyTorch only sees $AVAILABLE_GPUS GPU(s)!"
    echo "This node ($SLURM_NODENAME) might be misconfigured or faulty."
    
    # Exclude this node and requeue the job so SLURM shifts it to another node
    echo "Excluding node $SLURM_NODENAME and requeuing job $SLURM_JOB_ID..."
    scontrol update JobId=$SLURM_JOB_ID ExcNodeList=$SLURM_NODENAME
    scontrol requeue $SLURM_JOB_ID
    exit 1
fi


# 5. Run with torchrun to properly establish DDP across 2 GPUs
torchrun --nnodes=1 --nproc_per_node=2 Training/main.py

echo "Job ended at $(date)"
