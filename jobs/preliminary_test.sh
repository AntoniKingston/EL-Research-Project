#!/bin/bash
#SBATCH --partition=student
#SBATCH --account=20-stud-2627-z
#SBATCH --gres=gpu:1
#SBATCH --time=0-24:00:00
#SBATCH --job-name=preliminary
#SBATCH --output=logs/preliminary%j.out
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1

source /mnt/evafs/groups/mi2lab/akingston/venvs/el-rp/bin/activate

python -m src.scripts.train_orchestrator --config configs/preliminary_experiment_config.yaml -v  --heartbeat-minutes 5

