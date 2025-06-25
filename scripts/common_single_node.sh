
# activate your uv venv that was built with python3_mpi
source ~/source/l1nus/llm/.venv/bin/activate

# to reduce memory usage
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# outside of script run `accelerate config` to set up 

export NUM_PROCESSES=$(( SLURM_NNODES * SLURM_GPUS_ON_NODE ))
