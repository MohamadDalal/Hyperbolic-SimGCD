#!/bin/bash

#SBATCH --output="logs/SimGCD-Test-CUB.log"
#SBATCH --job-name="SimGCD-Test-CUB"
#SBATCH --time=12:00:00
#SBATCH --signal=B:SIGTERM@30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=16G
# #SBATCH --nodelist=ailab-l4-07
#SBATCH --exclude=ailab-l4-02

#####################################################################################

# Script arguments
container_path="${HOME}/pytorch-24.08.sif"

# Dynamically set output and error filenames using job ID and iteration
outfile="logs/SimGCD-Test-CUB.out"

exp_id="SimGCD-Test-CUB"

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
srun --output="${outfile}" --error="${outfile}" singularity exec --nv ${container_path} ${PYTHON} test.py \
            --dataset_name 'cub' \
            --batch_size 128 \
            --grad_from_block 11 \
            --num_workers 16 \
            --use_ssb_splits \
            --exp_id ${exp_id} \
            --transform 'imagenet' \
            --eval_funcs 'v2' \
            --use_dinov2 \
            --checkpoint_path '/ceph/home/student.aau.dk/mdalal20/P10-project/Hyperbolic-SimGCD/dev_outputs/simgcd/log/SimGCD-CUB/checkpoints/model_best.pt'
#> ${SAVE_DIR}logfile_${EXP_NUM}.out

#-m methods.contrastive_training.contrastive_training --dataset_name 'cub' --batch_size 128 --grad_from_block 11 --epochs 200 --base_model vit_dino --num_workers 16 --use_ssb_splits 'True' --sup_con_weight 0.35 --weight_decay 5e-5 --contrast_unlabel_only 'False' --exp_id test_exp --transform 'imagenet' --lr 0.1 --eval_funcs 'v1' 'v2'