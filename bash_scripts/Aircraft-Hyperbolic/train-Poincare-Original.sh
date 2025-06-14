#!/bin/bash

#SBATCH --output="logs/SimGCD-Aircraft-Hyperbolic-Poincare-Original.log"
#SBATCH --job-name="SimGCD-Aircraft-Hyperbolic-Poincare-Original"
#SBATCH --time=12:00:00
#SBATCH --signal=B:SIGTERM@30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=16G
#SBATCH --nodelist=ailab-l4-04
#SBATCH --exclude=ailab-l4-09

#####################################################################################

# Script arguments
container_path="${HOME}/pytorch-24.08.sif"

# Dynamically set output and error filenames using job ID and iteration
outfile="logs/SimGCD-Aircraft-Hyperbolic-Poincare-Original.out"

exp_id="SimGCD-Aircraft-Hyperbolic-Poincare-Original"

# Print the filenames for debugging
echo "Output file: ${outfile}"
#echo "Error file: ${errfile}"
#echo "Restart num: ${restarts}"
echo "Using container: ${container_path}"

PYTHON='/ceph/home/student.aau.dk/mdalal20/P10-project/Hyperbolic-SimGCD/venv/bin/python'

hostname
nvidia-smi

# export CUDA_VISIBLE_DEVICES=0

# Get unique log file,
#SAVE_DIR=/ceph/home/student.aau.dk/mdalal20/P10-project/hyperbolic-generalized-category-discovery/dev_outputs/

#EXP_NUM=$(ls ${SAVE_DIR} | wc -l)
#EXP_NUM=$((${EXP_NUM}+1))
#echo $EXP_NUM

# TODO: Investigate the double exp_id
srun --output="${outfile}" --error="${outfile}" singularity exec --nv ${container_path} ${PYTHON} train.py \
            --dataset_name 'aircraft' \
            --batch_size 128 \
            --grad_from_block 11 \
            --epochs 200 \
            --epochs_warmup 20 \
            --num_workers 16 \
            --use_ssb_splits \
            --sup_weight 0.35 \
            --weight_decay 5e-5 \
            --exp_id ${exp_id} \
            --transform 'imagenet' \
            --lr 0.1 \
            --eval_funcs 'v2' \
            --warmup_teacher_temp 0.07 \
            --teacher_temp 0.04 \
            --warmup_teacher_temp_epochs 30 \
            --memax_weight 1 \
            --exp_id 'Aircraft-Hyperbolic-Train-Poincare-Original' \
            --wandb_mode 'online' \
            --hyperbolic \
            --poincare \
            --original_poincare_layer \
            --curvature '0.05' \
            --proj_alpha 1.0 \
            --freeze_curvature 'full' \
            --freeze_proj_alpha 'full' \
            --euclidean_clipping 2.3 \
            --angle_loss \
            --max_angle_loss_weight 1.0 \
            --decay_angle_loss_weight \
            --max_grad_norm 1.0 \
            --avg_grad_norm 0.25 \
            --use_dinov2 \
            --checkpoint_path '/ceph/home/student.aau.dk/mdalal20/P10-project/Hyperbolic-SimGCD/dev_outputs/simgcd/log/Aircraft-Hyperbolic-Train-Poincare-Original/checkpoints/model.pt'
#> ${SAVE_DIR}logfile_${EXP_NUM}.out
