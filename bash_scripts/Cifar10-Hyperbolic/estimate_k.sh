#!/bin/bash

#SBATCH --output="logs/GCD-Estimate-Cifar10-Hyperbolic.log"
#SBATCH --job-name="GCD-Estimate-Cifar10-Hyperbolic"
#SBATCH --time=12:00:00
#SBATCH --signal=B:SIGTERM@30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=16G
# #SBATCH --nodelist=ailab-l4-07
# #SBATCH --exclude=nv-ai-03

#####################################################################################

# Script arguments
container_path="${HOME}/pytorch-24.08.sif"

# Dynamically set output and error filenames using job ID and iteration
outfile="logs/GCD-Estimate-Cifar10-Hyperbolic.out"

# Print the filenames for debugging
echo "Output file: ${outfile}"
#echo "Error file: ${errfile}"
#echo "Restart num: ${restarts}"
echo "Using container: ${container_path}"

PYTHON='/ceph/home/student.aau.dk/mdalal20/P10-project/hyperbolic-generalized-category-discovery/venv/bin/python'

hostname

# Get unique log file
#SAVE_DIR=/ceph/home/student.aau.dk/mdalal20/P10-project/hyperbolic-generalized-category-discovery/dev_outputs

#EXP_NUM=$(ls ${SAVE_DIR} | wc -l)
#EXP_NUM=$((${EXP_NUM}+1))
#echo $EXP_NUM

srun --output="${outfile}" --error="${outfile}" singularity exec --nv ${container_path} ${PYTHON} -m methods.estimate_k.estimate_k --max_classes 1000 --dataset_name cifar10 --search_mode other --warmup_model_exp_id 'Hyperbolic_best' --hyperbolic 'True'
        #> ${SAVE_DIR}logfile_${EXP_NUM}.out