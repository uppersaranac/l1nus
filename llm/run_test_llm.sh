#!/bin/bash
#SBATCH -J train_llm           # Job name
#SBATCH -o train_llm.out        # Output file (%j = job ID)
#SBATCH -t 36:00:00              # Wall time (1 hour)
#SBATCH -N 1                    # Number of nodes
#SBATCH -p gh                    # GPU partition (modify as needed)
#SBATCH --mail-user=lewis.geer@gmail.com
#SBATCH --mail-type=all

source ~/source/masskit/.venv/bin/activate
python train_llm.py --max_records 100000 --train_file ~/data/pubchem/arrow/iupac_train.arrow --eval_file ~/data/pubchem/arrow/iupac_valid.arrow

