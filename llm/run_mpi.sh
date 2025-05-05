#!/bin/bash
#SBATCH -J train_llm_mpi           # Job name
#SBATCH -t 36:00:00              # Wall time (1 hour)
#SBATCH -N 2                    # Number of nodes
#SBATCH -p gh                    # GPU partition (modify as needed)
#SBATCH --mail-user=lewis.geer@gmail.com
#SBATCH --mail-type=all
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
# ----------------------------------------------------------

module purge
module load gcc cuda openmpi python3_mpi

# activate your uv venv that was built with python3_mpi
source ~/source/l1nus/llm/.venv/bin/activate

# outside of script run `accelerate config` to set up 

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=ib0

# Accelerate will detect MASTER_ADDR/PORT automatically if they exist;
# we keep them for reproducibility.
export MASTER_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n 1)
export MASTER_PORT=29500

# --- Launch training ------------------------------------------------
srun --mpi=pmix \
     accelerate launch                       \
        --config_file mpi_config.json \
        --num_processes   $SLURM_NTASKS      \
        --num_machines    $SLURM_NNODES      \
        --machine_rank    $SLURM_PROCID      \
        --main_process_ip    $MASTER_ADDR      \
        --main_process_ip    $MASTER_PORT      \
        train_llm.py                          \
            --max_records 0                \
            --output_dir ~/results/$SLURM_JOB_ID     \
            --num_train_epochs 3 \
            --eval_steps 1000 \
            --eval_limit 30 \
            --max_records 0 \
            --train_file ~/data/pubchem/arrow/cluster_6M_train.arrow \
            --eval_file ~/data/pubchem/arrow/cluster_6M_eval.arrow
#            --deepspeed ds_config.json              # if you enabled Deepspeed
