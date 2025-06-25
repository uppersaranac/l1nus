#!/bin/bash
#SBATCH -J train_llm_iupac
#SBATCH -t 48:00:00
#SBATCH -N 1
#SBATCH -p batch
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:4
#SBATCH --mail-user=lewis.geer@gmail.com
#SBATCH --mail-type=all
#SBATCH -o /home/lyg/results/logs/%x_%j.out
#SBATCH -e /home/lyg/results/logs/%x_%j.err
# ----------------------------------------------------------

source ./common_single_node.sh

# note that the machine_rank value is escaped so it will be interpreted on the node actually doing the compute
# to turn on DeepSpeed: --config_file mpi_deepspeed_config.json
export LAUNCHER="accelerate launch \
--config_file single_node_config.json \
--num_processes 4 \
--num_machines $SLURM_NNODES \
"

export SCRIPT="../src/llm/training/cli_train.py"

export ARGS="--max_records 0 \
--model_name  \
--output_dir ~/results/$SLURM_JOB_ID \
--num_train_epochs 8 \
--eval_steps 1000 \
--per_device_train_batch_size 2 \
--per_device_eval_batch_size 2\
--eval_num_examples 64 \
--dataset_dir ~/data/pubchem/arrow/pubchem_best_cluster_iupac_naming \
"

export CMD="$LAUNCHER $SCRIPT $ARGS"

echo $CMD

# --- Launch training ------------------------------------------------
# srun --mpi=pmix $LAUNCHER $SCRIPT $ARGS
# ibrun $CMD
SRUN_ARGS=" \
    --wait=60 \
    --kill-on-bad-exit=1 \
    "

srun $SRUN_ARGS bash -lc "$CMD"

#            --deepspeed ds_config.json              # if you enabled Deepspeed
