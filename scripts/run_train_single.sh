#!/bin/bash
#SBATCH -J train_llm           # Job name
#SBATCH -t 36:00:00              # Wall time (1 hour)
#SBATCH -N 1                    # Number of nodes
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
export LAUNCHER="python"

export SCRIPT="../src/llm/train_llm.py"

export ARGS="--max_records 0 \
--output_dir ~/results/$SLURM_JOB_ID \
--num_train_epochs 5 \
--eval_steps 1000 \
--eval_limit 8 \
--question_set molecular_properties \
--train_file ~/data/pubchem/arrow/cluster_6M_train.arrow \
--eval_file ~/data/pubchem/arrow/cluster_6M_eval.arrow \
"

export CMD="$LAUNCHER $SCRIPT $ARGS"

echo $CMD

# --- Launch training ------------------------------------------------
$CMD

#            --deepspeed ds_config.json              # if you enabled Deepspeed
