module purge
module load gcc cuda openmpi python3_mpi

# activate your uv venv that was built with python3_mpi
source ~/source/l1nus/llm/.venv/bin/activate

# to reduce memory usage
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# outside of script run `accelerate config` to set up 

# export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# export NCCL_DEBUG=INFO
# export NCCL_SOCKET_IFNAME=ib0

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
