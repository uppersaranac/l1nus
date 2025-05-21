#!/bin/bash
#SBATCH -J train_llm_mpi           # Job name
#SBATCH -t 36:00:00              # Wall time (1 hour)
#SBATCH -N 2                    # Number of nodes
#SBATCH -p gh                    # GPU partition (modify as needed)
#SBATCH --mail-user=lewis.geer@gmail.com
#SBATCH --mail-type=all
#SBATCH --ntasks-per-node=1
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
# ----------------------------------------------------------

module purge
module load gcc cuda openmpi python3_mpi

# activate your uv venv that was built with python3_mpi
source ~/source/l1nus/llm/.venv/bin/activate

# outside of script run `accelerate config` to set up 

# export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# export NCCL_DEBUG=INFO
# export NCCL_SOCKET_IFNAME=ib0

# to reduce memory usage
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# see https://github.com/shaneholloman/smollm/blob/542250b39015654d47083245bfb4c03332643bd6/vision/experiments/evaluation/vloom/common/run_cron_evals_multi_task_cluster.slurm
# Accelerate will detect MASTER_ADDR/PORT automatically if they exist;
# we keep them for reproducibility.
export MASTER_ADDR=`scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1`
# From https://i.hsfzxjy.site/2021-03-10-obtain-a-random-unused-tcp-port-with-bash/
function unused_port() {
    N=${1:-1}
    comm -23 \
        <(seq "1025" "65535" | sort) \
        <(ss -Htan |
            awk '{print $4}' |
            cut -d':' -f2 |
            sort -u) |
        shuf |
        head -n "$N"
}

export MASTER_PORT=$(unused_port)

export NUM_PROCESSES=$(( SLURM_NNODES * SLURM_GPUS_ON_NODE ))

export RDZV_CONF="rdzv_backend=c10d,rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT"

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
--num_train_epochs 5 \
--eval_steps 2000 \
--eval_limit 50 \
--train_file ~/data/pubchem/arrow/cluster_6M_train.arrow \
--eval_file ~/data/pubchem/arrow/cluster_6M_eval.arrow \
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
