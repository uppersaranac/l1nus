#!/bin/bash
#SBATCH -J chem_lesson_2           # Job name
#SBATCH -t 36:00:00              # Wall time (1 hour)
#SBATCH -N 16                    # Number of nodes
#SBATCH -p gh                    # GPU partition (modify as needed)
#SBATCH --mail-user=lewis.geer@gmail.com
#SBATCH --mail-type=all
#SBATCH --ntasks-per-node=1
#SBATCH -o /home1/10318/lyg/results/logs/%x_%j.out
#SBATCH -e /home1/10318/lyg/results/logs/%x_%j.err
# ----------------------------------------------------------

source ./common.sh

# note that the machine_rank value is escaped so it will be interpreted on the node actually doing the compute
# to turn on DeepSpeed: --config_file mpi_deepspeed_config.json
export LAUNCHER="accelerate launch \
--config_file mpi_config.json \
--num_processes $SLURM_NNODES \
--num_machines $SLURM_NNODES \
--rdzv_conf $RDZV_CONF \
--machine_rank \$SLURM_PROCID \
--main_process_ip $MASTER_ADDR \
--main_process_port $MASTER_PORT \
"

export SCRIPT="../src/llm/training/cli_train.py"

export ARGS="--limit 0 \
--output_dir ~/results/${SLURM_JOB_NAME}_${SLURM_JOB_ID} \
--per_device_train_batch_size 2 \
--per_device_eval_batch_size 2 \
--gradient_accumulation_steps 8 \
--num_train_epochs 8 \
--eval_steps 5000 \
--eval_num_examples 64 \
--num_example_preds 16 \
--dataset_dir ~/data/pubchem/arrow/chem_lesson_2 \
--model_name Qwen/Qwen3-4B \
--use_position_weighting \
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
