#!/bin/bash
#SBATCH -J train_llm_mpi           # Job name
#SBATCH -t 36:00:00              # Wall time (1 hour)
#SBATCH -N 8                    # Number of nodes
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

export SCRIPT="../src/llm/train_llm.py"

export ARGS="--max_records 0 \
--output_dir ~/results/$SLURM_JOB_ID \
--num_train_epochs 10 \
--eval_steps 200 \
--eval_limit 64 \
--train_file ~/data/pubchem/arrow/cluster_6M_train.arrow \
--eval_file ~/data/pubchem/arrow/cluster_6M_eval.arrow \
--question_set iupac_naming \
--model_name /home1/10318/lyg/results/252674
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
