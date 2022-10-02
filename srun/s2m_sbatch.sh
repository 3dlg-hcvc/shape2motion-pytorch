#!/bin/bash
#SBATCH --account=rrg-msavva
#SBATCH --gres=gpu:p100:1         # Number of GPUs (per node)
#SBATCH --mem=32000M              # memory (per node)
#SBATCH --time=2-23:59            # time (DD-HH:MM)
#SBATCH --cpus-per-task=6         # Number of CPUs (per task)
#SBATCH --output=logs/%x_%j.out
#SBATCH --mail-user=sam_mao@sfu.ca
#SBATCH --mail-type=ALL
# ---------------------------------------------------------------------
echo "Current working directory: `pwd`"
echo "Starting run at: `date`"
# ---------------------------------------------------------------------
module load python/3.7.9
module load cuda/11.0
module load cudnn/8.0.3
. configure.sh

source $VENV/bin/activate

cd $PROJ_DIR
python pipeline.py network.has_color=$HAS_COLOR network.has_normal=$HAS_NORMAL \
network.num_channels=$NUM_CHANNELS network.random_seed=$SEED paths.result_dir="${OUTPUT_DIR}/${OUTPUT_FOLDER}" \
network.augmentation.jitter=$AUGMENTATION network.augmentation.flip=$AUGMENTATION network.augmentation.rotate=$AUGMENTATION \
network.augmentation.color=$AUGMENTATION

# ---------------------------------------------------------------------
echo "Job finished with exit code $? at: `date`"
# ---------------------------------------------------------------------
