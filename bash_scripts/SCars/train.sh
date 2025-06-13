#!/bin/bash

#SBATCH --output="logs/SimGCD-SCars.log"
#SBATCH --job-name="SimGCD-SCars"
#SBATCH --time=12:00:00
#SBATCH --signal=B:SIGTERM@30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=16G
# #SBATCH --nodelist=ailab-l4-07
#SBATCH --exclude=ailab-l4-02

#####################################################################################

# This means that the model will potentially run 4 times
max_restarts=6

# Fetch the current restarts value from the job context
scontext=$(scontrol show job ${SLURM_JOB_ID})
restarts=$(echo ${scontext} | grep -o 'Restarts=[0-9]*' | cut -d= -f2)

# If no restarts found, it's the first run, so set restarts to 0
iteration=${restarts:-0}

# Script arguments
container_path="${HOME}/pytorch-24.08.sif"

# Dynamically set output and error filenames using job ID and iteration
outfile="logs/SimGCD-SCars.out"

exp_id="SimGCD-SCars"

# Print the filenames for debugging
echo "Output file: ${outfile}"
#echo "Error file: ${errfile}"
echo "Restart num: ${restarts}"
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

##  Define a term-handler function to be executed           ##
##  when the job gets the SIGTERM (before timeout)          ##

term_handler()
{
    echo "Executing term handler at $(date)"
    if [[ $restarts -lt $max_restarts ]]; then
        # Requeue the job, allowing it to restart with incremented iteration
        scontrol requeue ${SLURM_JOB_ID}
        exit 0
    else
        echo "Maximum restarts reached, exiting."
        exit 1
    fi
}

# Trap SIGTERM to execute the term_handler when the job gets terminated
trap 'term_handler' SIGTERM

# TODO: Investigate the double exp_id
#######################################################################################
if [ $restarts -gt -1 ]; then

    srun --output="${outfile}" --error="${outfile}" singularity exec --nv ${container_path} ${PYTHON} train.py \
            --dataset_name 'scars' \
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
            --wandb_mode 'online' \
            --max_grad_norm 100.0 \
            --avg_grad_norm 100.0 \
            --use_dinov2 \
            --checkpoint_path '/ceph/home/student.aau.dk/mdalal20/P10-project/Hyperbolic-SimGCD/dev_outputs/simgcd/log/SimGCD-SCars/checkpoints/model.pt'
else
    srun --output="${outfile}" --error="${outfile}" singularity exec --nv ${container_path} ${PYTHON} train.py \
            --dataset_name 'scars' \
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
            --wandb_mode 'online' \
            --max_grad_norm 100.0 \
            --avg_grad_norm 100.0 \
            --use_dinov2 \
            #--checkpoint_path '/ceph/home/student.aau.dk/mdalal20/P10-project/Hyperbolic-SimGCD/dev_outputs/simgcd/log/SimGCD-SCars/checkpoints/model.pt'
#> ${SAVE_DIR}logfile_${EXP_NUM}.out
fi

#-m methods.contrastive_training.contrastive_training --dataset_name 'scars' --batch_size 128 --grad_from_block 11 --epochs 200 --base_model vit_dino --num_workers 16 --use_ssb_splits 'True' --sup_con_weight 0.35 --weight_decay 5e-5 --contrast_unlabel_only 'False' --exp_id test_exp --transform 'imagenet' --lr 0.1 --eval_funcs 'v1' 'v2'