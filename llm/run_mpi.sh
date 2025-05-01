#!/bin/bash
#SBATCH -J llm-accel
#SBATCH -p gh
#SBATCH -N 2
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH -t 24:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
# ----------------------------------------------------------

module purge
module load gcc
module load cuda
# module load openmpi/4.1.5
module load python3_mpi            # MPI-enabled interpreter

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
srun --mpi=pmix_v4 \
     accelerate launch                       \
        --num_processes   $SLURM_NTASKS      \
        --num_machines    $SLURM_NNODES      \
        --machine_rank    $SLURM_PROCID      \
        train_llm.py                          \
            --model_name_or_path facebook/opt-1.3b \
            --dataset_name my_corpus                \
            --output_dir $WORK/llm_ckpt/opt1.3b     \
            --per_device_train_batch_size 4         \
            --gradient_accumulation_steps 8         \
            --learning_rate 2e-5                    \
            --num_train_epochs 3                    \
            --logging_steps 50                      \
            --save_steps 500                        \
            --bf16                                  \
#            --deepspeed ds_config.json              # if you enabled Deepspeed
