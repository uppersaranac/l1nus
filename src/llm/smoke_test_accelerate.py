# test_accelerate_debug_multinode.py
import logging
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import broadcast
from accelerate.utils import gather

"""
to run from command line:
srun bash -lc "accelerate launch --config_file mpi_config.json --num_processes 2 --num_machines 2 --rdzv_conf rdzv_backend=c10d,rdzv_endpoint=c608-142:12590 --machine_rank \$SLURM_PROCID --main_process_ip c608-142 --main_process_port 12590 test_accelerate.py"
"

# 1) Turn on root‐level DEBUG logs
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s"
)

# 2) Set up Accelerate’s own logger
logger = get_logger(__name__, log_level="DEBUG")

def main():
    # 3) Initialize Accelerator for multi‐node
    accelerator = Accelerator()
    state = accelerator.state

    # 4) Log the overall distributed state
    logger.info(
        f"Initialized Accelerator: "
        f"num_processes={state.num_processes}, "
        f"process_index={state.process_index}, ",
        main_process_only=False
    )

    # 5) Create a dummy tensor encoding machine & rank
    tensor = torch.tensor(
        [state.num_processes, state.process_index],
        device=accelerator.device
    )
    logger.debug(
        f"Before gather (machine {state.num_processes}, rank {state.process_index}): {tensor}",
        main_process_only=False
    )

    # 6) Gather across all ranks on all machines
    gathered = accelerator.gather(tensor)
    logger.debug(
        f"After  gather: {gathered.view(state.num_processes, -1)}",
        main_process_only=False
    )

if __name__ == "__main__":
    main()
