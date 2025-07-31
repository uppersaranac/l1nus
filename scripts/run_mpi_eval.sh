#!/bin/bash
#SBATCH -J train_llm_mpi           # Job name
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

python ~/source/l1nus/src/llm/evaluate/cli_evaluate.py --model_name ~/results/299198/model7 --dataset_dir ~/data/pubchem/arrow/pubchem_best_cluster_molecular_properties/full --split valid --output_csv 299198_epoch7.csv >& 299198_epoch7.txt

#            --deepspeed ds_config.json              # if you enabled Deepspeed
