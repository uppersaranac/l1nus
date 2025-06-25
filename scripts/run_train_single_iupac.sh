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

export SCRIPT="../src/llm/training/cli_train.py"

export ARGS="--limit 0 \
--output_dir ~/results/$SLURM_JOB_ID \
--per_device_train_batch_size 2 \
--per_device_eval_batch_size 2 \
--num_train_epochs 10 \
--eval_steps 10000 \
--eval_num_examples 8 \
--dataset_dir ~/data/pubchem/arrow/pubchem_best_cluster_iupac_naming \
--model_name Qwen/Qwen2.5-1.5B-Instruct \
"

export CMD="$LAUNCHER $SCRIPT $ARGS"

echo $CMD

# --- Launch training ------------------------------------------------
$CMD

#            --deepspeed ds_config.json              # if you enabled Deepspeed
